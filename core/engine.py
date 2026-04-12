"""
主交易引擎 —— 统一规则策略与执行层

本版核心变化：
- 通过 strategy_registry 统一选择策略
- 默认走 4h 主周期 + 1d 高级别过滤
- 修复 paper/live 的现金与 NAV 计算
- 回测与实盘共用同一套策略类
"""
from __future__ import annotations

import time
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from alpha.strategy_registry import build_strategy
from config.loader import Config
from core.events import EventBus
from data.client import BinanceClient
from data.features import FeatureEngine
from data.historical import HistoryDownloader
from data.storage import Storage
from execution.executor import OrderExecutor
from execution.position import PositionTracker
from risk.manager import RiskManager
from utils.logger import get_logger

logger = get_logger("engine")


class TradingEngine:
    """
    生命周期：init → warmup → run_cycle / run
    """

    def __init__(self, config: Config, simulate: bool = True, strategy_name: str | None = None):
        self.config = config
        self.simulate = simulate

        self.event_bus = EventBus()

        db_path = config.get_nested("data.database.path", "data/quant.db")
        cache_mb = config.get_nested("data.database.cache_size_mb", 256)
        self.storage = Storage(db_path, cache_size_mb=cache_mb)
        self.client = BinanceClient(config)
        self.history = HistoryDownloader(config, self.client, self.storage)
        self.feature_engine = FeatureEngine(
            windows=config.get_nested("features.lookback_windows", [5, 10, 20, 60, 120, 240, 480])
        )

        self.position_tracker = PositionTracker()
        self.executor = OrderExecutor(
            config,
            self.client,
            self.storage,
            self.position_tracker,
            self.event_bus,
            simulate=simulate,
        )

        self.symbols = config.get_symbols()
        self.strategy = build_strategy(config=config, explicit_name=strategy_name)
        self.strategy_states: dict[str, dict] = {
            symbol: {"bar_index": -1, "cooldown_until_bar": -1} for symbol in self.symbols
        }
        self.strategy_positions: dict[str, Optional[dict]] = {symbol: None for symbol in self.symbols}

        self._initial_capital = float(config.get_nested("system.paper_capital", 10000.0))
        self.cash = self._initial_capital
        self.last_nav = self._initial_capital

        self.risk_mgr = RiskManager(
            config=config,
            position_tracker=self.position_tracker,
            event_bus=self.event_bus,
            initial_capital=self._initial_capital,
        )

        self.cycle_count = 0
        self._running = False

        mode = "paper" if simulate else "live"
        logger.info(
            f"交易引擎初始化 | mode={mode} | strategy={self.strategy.name} | "
            f"intervals={self.strategy.primary_interval}/{self.strategy.higher_interval} | "
            f"symbols={self.symbols}"
        )

    # ==============================================================
    # Warmup
    # ==============================================================
    def warmup(self, skip_history: bool = False, skip_train: bool = True):
        logger.info("=" * 60)
        logger.info("  预热阶段开始")
        logger.info("=" * 60)

        if not self.simulate:
            try:
                balances = self.client.get_balances()
                usdt = balances.get("USDT", {})
                self._initial_capital = float(usdt.get("free", 0)) + float(usdt.get("locked", 0))
                self.cash = self._initial_capital
                self.last_nav = self._initial_capital
                self.risk_mgr.initial_capital = self._initial_capital
                self.risk_mgr.peak_nav = self._initial_capital
                self.risk_mgr.daily_start_nav = self._initial_capital
                self.risk_mgr.weekly_start_nav = self._initial_capital
                logger.info(f"账户 USDT 余额: {self._initial_capital:.2f}")
            except Exception as exc:
                logger.error(f"账户连接失败: {exc}")
                raise

        if not skip_history:
            self._sync_history()

        self.event_bus.start()
        logger.info("预热完成")

    def _sync_history(self):
        logger.info("开始同步历史数据...")
        try:
            self.history.sync_all(max_workers=2)
            tracked_intervals = {self.strategy.primary_interval}
            if self.strategy.higher_interval:
                tracked_intervals.add(self.strategy.higher_interval)

            for symbol in self.symbols:
                for interval in tracked_intervals:
                    report = self.history.validate_data(symbol, interval)
                    logger.info(
                        f"{symbol}/{interval}: {report.get('count', 0)} 条 | "
                        f"缺口={report.get('gaps', 0)} | 状态={report.get('status', 'unknown')}"
                    )
        except Exception as exc:
            logger.error(f"历史数据同步失败: {exc}")

    # ==============================================================
    # Main cycle
    # ==============================================================
    def run_cycle(self):
        self.cycle_count += 1
        logger.info(f"\n{'━' * 60}")
        logger.info(f"交易循环 #{self.cycle_count} | {datetime.utcnow().isoformat()}Z")
        logger.info(f"{'━' * 60}")

        self._update_data()
        prices = self._get_prices()
        if not prices:
            logger.error("价格获取失败，跳过本轮")
            return

        self.position_tracker.update_all_extremes(prices)

        nav = self._calculate_nav(prices)
        cycle_ret = (nav - self.last_nav) / max(self.last_nav, 1e-12)
        self.risk_mgr.record_return(cycle_ret)
        self.last_nav = nav

        port_ok, port_msg = self.risk_mgr.check_portfolio(nav)
        logger.info(f"组合风控: {port_msg}")
        if not port_ok:
            self._log_status(prices, nav)
            return

        risk_actions = self.risk_mgr.check_positions(prices)
        for action in risk_actions:
            self._force_close_from_risk(action, nav, prices)

        nav = self._calculate_nav(prices)
        for symbol in self.symbols:
            self._process_symbol(symbol, prices, nav)

        nav = self._calculate_nav(prices)
        self._log_status(prices, nav)

    def _prepare_symbol_features(self, symbol: str) -> Optional[pd.DataFrame]:
        primary_df = self.storage.get_klines(symbol, self.strategy.primary_interval)
        if len(primary_df) < 200:
            logger.debug(f"{symbol}: {self.strategy.primary_interval} 数据不足 ({len(primary_df)})")
            return None

        for col in ["open", "high", "low", "close", "volume"]:
            if col in primary_df.columns:
                primary_df[col] = pd.to_numeric(primary_df[col], errors="coerce")
        primary_df = primary_df.dropna(subset=["open", "high", "low", "close", "volume"])

        primary_feat = self.feature_engine.compute_all(primary_df)
        if primary_feat.empty:
            return None
        for col in ["open_time", "open", "high", "low", "close", "volume"]:
            if col in primary_df.columns:
                primary_feat[col] = primary_df[col].values[-len(primary_feat):]

        higher_feat = None
        if self.strategy.higher_interval:
            higher_df = self.storage.get_klines(symbol, self.strategy.higher_interval)
            if len(higher_df) >= 80:
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in higher_df.columns:
                        higher_df[col] = pd.to_numeric(higher_df[col], errors="coerce")
                higher_df = higher_df.dropna(subset=["open", "high", "low", "close", "volume"])
                higher_feat = self.feature_engine.compute_all(higher_df)
                if not higher_feat.empty:
                    for col in ["open_time", "open", "high", "low", "close", "volume"]:
                        if col in higher_df.columns:
                            higher_feat[col] = higher_df[col].values[-len(higher_feat):]

        prepared = self.strategy.prepare_features(primary_feat, higher_feat)
        prepared = prepared.dropna(subset=["close"]).reset_index(drop=True)
        return prepared if len(prepared) >= 2 else None

    def _process_symbol(self, symbol: str, prices: dict[str, float], nav: float):
        prepared = self._prepare_symbol_features(symbol)
        if prepared is None:
            return

        row = prepared.iloc[-1]
        prev_row = prepared.iloc[-2]
        current_price = prices.get(symbol, float(row["close"]))

        state = self.strategy_states[symbol]
        state["bar_index"] = len(prepared) - 1

        signal_meta = self.strategy.signal_metadata(row)
        self.storage.save_signal(
            {
                "symbol": symbol,
                "timestamp": int(time.time() * 1000),
                "strategy": self.strategy.name,
                "direction": signal_meta.get("tag", "hold"),
                "strength": signal_meta.get("strength", 0.0),
                "confidence": None,
                "features": None,
                "model_version": self.strategy.name,
            }
        )

        logger.info(
            f"[{symbol}] price={current_price:.2f} | signal={signal_meta.get('tag')} | "
            f"strength={signal_meta.get('strength', 0):.3f}"
        )

        position = self.strategy_positions.get(symbol)

        if position is not None:
            position["highest_since_entry"] = max(position["highest_since_entry"], current_price)
            position["lowest_since_entry"] = min(position["lowest_since_entry"], current_price)
            bar_count = state["bar_index"] - position["entry_bar"]

            should_exit, reason = self.strategy.check_exit(row, prev_row, position, bar_count)
            if should_exit:
                qty = position["qty"]
                ok, risk_reason = self.risk_mgr.pre_trade_check(symbol, "SELL", qty, current_price, nav, prices)
                if ok:
                    result = self.executor.execute(
                        symbol=symbol,
                        side="SELL",
                        qty=qty,
                        price=current_price,
                        strategy=self.strategy.name,
                        algo=self.config.execution.algo,
                    )
                    if result:
                        self._apply_sell_cash(result)
                        pnl = (result["avg_fill_price"] - position["entry_price"]) * qty - result.get("commission", 0)
                        self.risk_mgr.record_trade_result(pnl)
                        self.strategy.on_trade_closed(state, state["bar_index"], reason)
                        self.strategy_positions[symbol] = None
                        self.risk_mgr.record_order()
                        logger.info(
                            f"[{symbol}] 平仓 | reason={reason} | qty={qty:.6f} | "
                            f"fill={result['avg_fill_price']:.2f} | pnl={pnl:+.2f}"
                        )
                else:
                    logger.warning(f"[{symbol}] 平仓被风控拒绝: {risk_reason}")
                return

        if position is None and self.strategy.should_enter(row=row, prev_row=prev_row, state=state):
            qty, stop_loss = self.strategy.calc_position(self.cash, current_price, row)
            if qty <= 0:
                return

            ok, risk_reason = self.risk_mgr.pre_trade_check(symbol, "BUY", qty, current_price, nav, prices)
            if not ok:
                logger.info(f"[{symbol}] 开仓被风控拒绝: {risk_reason}")
                return

            result = self.executor.execute(
                symbol=symbol,
                side="BUY",
                qty=qty,
                price=current_price,
                strategy=self.strategy.name,
                algo=self.config.execution.algo,
            )
            if result:
                self._apply_buy_cash(result)
                self.strategy_positions[symbol] = {
                    "qty": result.get("executed_qty", qty),
                    "entry_price": result.get("avg_fill_price", current_price),
                    "entry_bar": state["bar_index"],
                    "entry_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                    "stop_loss": stop_loss,
                    "highest_since_entry": current_price,
                    "lowest_since_entry": current_price,
                    "atr_at_entry": float(row.get("natr_20", 0) or 0) * current_price,
                }
                self.risk_mgr.record_order()
                logger.info(
                    f"[{symbol}] 开仓 | qty={qty:.6f} | fill={result['avg_fill_price']:.2f} | "
                    f"stop={stop_loss:.2f}"
                )

    # ==============================================================
    # Cash / NAV helpers
    # ==============================================================
    def _apply_buy_cash(self, result: dict):
        fill_price = float(result.get("avg_fill_price", result.get("price", 0.0)) or 0.0)
        qty = float(result.get("executed_qty", result.get("quantity", 0.0)) or 0.0)
        commission = float(result.get("commission", 0.0) or 0.0)
        self.cash -= qty * fill_price + commission

    def _apply_sell_cash(self, result: dict):
        fill_price = float(result.get("avg_fill_price", result.get("price", 0.0)) or 0.0)
        qty = float(result.get("executed_qty", result.get("quantity", 0.0)) or 0.0)
        commission = float(result.get("commission", 0.0) or 0.0)
        self.cash += qty * fill_price - commission

    def _calculate_nav(self, prices: dict[str, float]) -> float:
        exposure = self.position_tracker.get_total_exposure(prices)
        return self.cash + exposure

    def _force_close_from_risk(self, action: dict, nav: float, prices: dict[str, float]):
        symbol = action["symbol"]
        result = self.executor.execute(
            symbol=symbol,
            side="SELL",
            qty=action["qty"],
            price=action["price"],
            strategy="risk_control",
            algo="market",
        )
        if result:
            self._apply_sell_cash(result)
            self.strategy_positions[symbol] = None
            self.risk_mgr.record_trade_result(-1.0)
            self.risk_mgr.record_order()
            logger.warning(f"风控平仓: {symbol} | {action['reason']}")

    # ==============================================================
    # Data helpers
    # ==============================================================
    def _update_data(self):
        intervals = {self.strategy.primary_interval}
        if self.strategy.higher_interval:
            intervals.add(self.strategy.higher_interval)

        for symbol in self.symbols:
            for interval in intervals:
                try:
                    df = self.client.get_klines(symbol, interval, limit=10)
                    if not df.empty:
                        self.storage.upsert_klines(symbol, interval, df)
                except Exception as exc:
                    logger.error(f"更新 {symbol}/{interval} 失败: {exc}")

    def _get_prices(self) -> dict[str, float]:
        prices = {}
        for symbol in self.symbols:
            try:
                prices[symbol] = self.client.get_ticker_price(symbol)
            except Exception:
                df = self.storage.get_klines(symbol, self.strategy.primary_interval, limit=1)
                if not df.empty:
                    prices[symbol] = float(df["close"].iloc[-1])
        return prices

    def _log_status(self, prices: dict[str, float], nav: float):
        risk_snap = self.risk_mgr.get_risk_snapshot(nav, prices)
        port_summary = self.position_tracker.get_portfolio_summary(prices)

        logger.info(
            f"💰 NAV={nav:.2f} | cash={self.cash:.2f} | DD={risk_snap['drawdown_pct']:.2%} | "
            f"日PnL={risk_snap['daily_pnl_pct']:+.2%} | 敞口={risk_snap['exposure_pct']:.1%} | "
            f"持仓={risk_snap['open_positions']}"
        )

        for sym, info in port_summary["positions"].items():
            logger.info(
                f"  📊 {sym}: qty={info['qty']:.6f} | value={info['market_value']:.2f} | "
                f"upnl={info['upnl_pct']:+.2%}"
            )

    # ==============================================================
    # Run loop
    # ==============================================================
    def run(self, interval_seconds: int = 3600):
        self._running = True
        logger.info(f"交易循环启动 | 间隔={interval_seconds}s")

        try:
            while self._running:
                try:
                    self.run_cycle()
                except Exception as exc:
                    logger.error(f"循环异常: {exc}\n{traceback.format_exc()}")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self.shutdown()

    def shutdown(self):
        self._running = False
        self.event_bus.stop()
        logger.info(
            f"引擎关闭 | cycles={self.cycle_count} | cash={self.cash:.2f} | "
            f"stats={self.executor.stats}"
        )

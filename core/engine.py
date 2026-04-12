"""
主交易引擎 —— 串联所有模块的中枢
"""
import time
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config.loader import Config, load_config
from core.events import EventBus, Event, EventType
from data.client import BinanceClient
from data.storage import Storage
from data.historical import HistoryDownloader
from data.features import FeatureEngine
from alpha.ml_model import AlphaModel
from alpha.portfolio import PortfolioOptimizer
from execution.executor import OrderExecutor
from execution.position import PositionTracker
from risk.manager import RiskManager
from utils.logger import get_logger

logger = get_logger("engine")


class TradingEngine:
    """
    量化交易引擎
    生命周期: init → warmup (数据+模型) → run_loop
    """

    def __init__(self, config: Config, simulate: bool = True):
        self.config = config
        self.simulate = simulate

        # 核心总线
        self.event_bus = EventBus()

        # 数据层
        db_path = config.get_nested("data.database.path", "data/quant.db")
        cache_mb = config.get_nested("data.database.cache_size_mb", 256)
        self.storage = Storage(db_path, cache_size_mb=cache_mb)
        self.client = BinanceClient(config)
        self.history = HistoryDownloader(config, self.client, self.storage)
        self.feature_engine = FeatureEngine(
            windows=config.get_nested("features.lookback_windows", [5, 10, 20, 60, 120, 240, 480]))

        # Alpha 层
        self.alpha_model = AlphaModel(config, self.storage)
        self.portfolio_opt = PortfolioOptimizer(config)

        # 执行层
        self.position_tracker = PositionTracker()
        self.executor = OrderExecutor(
            config, self.client, self.storage,
            self.position_tracker, self.event_bus, simulate=simulate)

        # 风控层
        self._initial_capital = 10000.0  # 默认，warmup 时更新
        self.risk_mgr = RiskManager(
            config, self.position_tracker, self.event_bus, self._initial_capital)

        self.symbols = config.get_symbols()
        self.cycle_count = 0
        self._running = False

        mode = "模拟" if simulate else "实盘"
        logger.info(f"交易引擎初始化 | 模式={mode} | 标的={self.symbols}")

    # ==============================================================
    # Phase 1: 预热
    # ==============================================================
    def warmup(self, skip_history: bool = False, skip_train: bool = False):
        """
        预热阶段：
        1. 连接验证
        2. 同步历史数据
        3. 训练/加载模型
        """
        logger.info("=" * 60)
        logger.info("  预热阶段开始")
        logger.info("=" * 60)

        # 1. 连接测试
        if not self.simulate:
            try:
                balances = self.client.get_balances()
                usdt = balances.get("USDT", {})
                self._initial_capital = usdt.get("free", 0) + usdt.get("locked", 0)
                self.risk_mgr.initial_capital = self._initial_capital
                self.risk_mgr.peak_nav = self._initial_capital
                self.risk_mgr.daily_start_nav = self._initial_capital
                logger.info(f"账户余额: {self._initial_capital:.2f} USDT")
            except Exception as e:
                logger.error(f"账户连接失败: {e}")
                if not self.simulate:
                    raise

        # 2. 同步历史数据
        if not skip_history:
            logger.info("开始同步历史数据...")
            try:
                results = self.history.sync_all(max_workers=2)
                # 数据质量报告
                for symbol in self.symbols:
                    for tf in self.config.get_timeframes():
                        report = self.history.validate_data(symbol, tf["interval"])
                        logger.info(f"  {symbol}/{tf['interval']}: "
                                    f"{report.get('count', 0)} 条 | "
                                    f"缺口={report.get('gaps', 0)} | "
                                    f"状态={report.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"历史数据同步失败: {e}")

        # 3. 模型训练/加载
        if not skip_train:
            self._train_or_load_model()

        self.event_bus.start()
        logger.info("预热完成")

    def _train_or_load_model(self):
        """训练或加载已有模型"""
        try:
            self.alpha_model.load()
            logger.info("已加载历史模型")
        except FileNotFoundError:
            logger.info("无历史模型，开始训练...")
            self._train_model()

    def _train_model(self):
        """训练 ML 模型"""
        # 使用最长时间框架的数据训练
        primary_tf = "1h"
        for symbol in self.symbols:
            df = self.storage.get_klines(symbol, primary_tf)
            if len(df) < 1000:
                logger.warning(f"{symbol} 数据不足 ({len(df)})，跳过训练")
                continue

            logger.info(f"为 {symbol} 计算特征... ({len(df)} 条K线)")
            features = self.feature_engine.compute_all(
                df, include_target=True, target_periods=[1, 4, 24])

            if features.empty:
                continue

            features = self.feature_engine.preprocess(features, method="zscore")

            try:
                report = self.alpha_model.train(features, target_col="target_dir_1")
                logger.info(f"模型训练完成: {report['model_id']}")
                logger.info(f"  平均指标: {report['avg_metrics']}")
                logger.info(f"  Top 5 特征: {list(report['top_features'].items())[:5]}")
                break  # 目前只用第一个标的训练
            except Exception as e:
                logger.error(f"训练失败: {e}\n{traceback.format_exc()}")

    # ==============================================================
    # Phase 2: 交易循环
    # ==============================================================
    def run_cycle(self):
        """运行一轮完整的交易循环"""
        self.cycle_count += 1
        logger.info(f"\n{'━' * 60}")
        logger.info(f"  交易循环 #{self.cycle_count} | {datetime.utcnow().isoformat()}Z")
        logger.info(f"{'━' * 60}")

        # 1. 增量数据更新
        self._update_data()

        # 2. 获取最新价格
        prices = self._get_prices()
        if not prices:
            logger.error("无法获取价格，跳过本轮")
            return

        # 3. 更新持仓极值
        self.position_tracker.update_all_extremes(prices)

        # 4. Portfolio 风控检查
        nav = self._calculate_nav(prices)
        port_ok, port_msg = self.risk_mgr.check_portfolio(nav)
        logger.info(f"组合风控: {port_msg}")

        if not port_ok:
            logger.warning("组合风控未通过，跳过信号生成")
            self._log_status(prices, nav)
            return

        # 5. Position 风控（止损止盈）
        risk_actions = self.risk_mgr.check_positions(prices)
        for action in risk_actions:
            logger.warning(f"风控平仓: {action['symbol']} | {action['reason']}")
            self.executor.execute(
                action["symbol"], "SELL", action["qty"],
                action["price"], strategy="risk_control", algo="market")
            self.risk_mgr.record_trade_result(-1)  # 止损视为亏损

        # 6. 生成信号
        for symbol in self.symbols:
            self._process_symbol(symbol, prices, nav)

        # 7. 状态汇报
        self._log_status(prices, nav)

    def _process_symbol(self, symbol: str, prices: dict, nav: float):
        """处理单个标的的信号生成和执行"""
        df = self.storage.get_klines(symbol, "1h")
        if len(df) < 500:
            logger.debug(f"{symbol} 数据不足，跳过")
            return

        # 特征计算
        features = self.feature_engine.compute_all(df)
        if features.empty:
            return
        features = self.feature_engine.preprocess(features, method="zscore")

        # ML 预测
        if self.alpha_model.model is None:
            return

        try:
            predictions = self.alpha_model.predict(features)
            latest = predictions.iloc[-1]
            signal = latest["signal"]
            strength = latest["strength"]
            probability = latest["probability"]
        except Exception as e:
            logger.error(f"{symbol} 预测失败: {e}")
            return

        logger.info(
            f"[{symbol}] 信号={signal} | 强度={strength:.3f} | "
            f"概率={probability:.3f} | 价格={prices.get(symbol, 0):.2f}")

        # 记录信号
        self.storage.save_signal({
            "symbol": symbol,
            "timestamp": int(time.time() * 1000),
            "strategy": "ml_ensemble",
            "direction": signal,
            "strength": strength,
            "confidence": probability,
            "model_version": self.alpha_model.model_id,
        })

        if signal == "HOLD":
            return

        # Pre-trade 风控
        current_price = prices[symbol]
        pos = self.position_tracker.get_position(symbol)

        if signal in ("BUY", "STRONG_BUY") and (not pos or not pos.is_open):
            # 计算下单量
            position_value = nav * 0.20 * strength  # 最大20%仓位 × 信号强度
            qty = position_value / current_price

            ok, reason = self.risk_mgr.pre_trade_check(
                symbol, "BUY", qty, current_price, nav, prices)
            if ok:
                self.executor.execute(
                    symbol, "BUY", qty, current_price,
                    strategy="ml_ensemble", algo=self.config.execution.algo)
                self.risk_mgr.record_order()
            else:
                logger.info(f"[{symbol}] 风控拒绝: {reason}")

        elif signal in ("SELL", "STRONG_SELL") and pos and pos.is_open:
            ok, reason = self.risk_mgr.pre_trade_check(
                symbol, "SELL", pos.quantity, current_price, nav, prices)
            if ok:
                result = self.executor.execute(
                    symbol, "SELL", pos.quantity, current_price,
                    strategy="ml_ensemble", algo="market")
                self.risk_mgr.record_order()
            else:
                logger.info(f"[{symbol}] 风控拒绝: {reason}")

    # ==============================================================
    # 辅助方法
    # ==============================================================
    def _update_data(self):
        """增量更新最新数据"""
        for symbol in self.symbols:
            try:
                df = self.client.get_klines(symbol, "1h", limit=10)
                if not df.empty:
                    self.storage.upsert_klines(symbol, "1h", df)
            except Exception as e:
                logger.error(f"更新 {symbol} 数据失败: {e}")

    def _get_prices(self) -> dict[str, float]:
        prices = {}
        for symbol in self.symbols:
            try:
                prices[symbol] = self.client.get_ticker_price(symbol)
            except Exception:
                df = self.storage.get_klines(symbol, "1h", limit=1)
                if not df.empty:
                    prices[symbol] = float(df["close"].iloc[-1])
        return prices

    def _calculate_nav(self, prices: dict) -> float:
        cash = self._initial_capital
        # 减去买入花费，加上卖出收入（简化版）
        exposure = self.position_tracker.get_total_exposure(prices)
        upnl = self.position_tracker.get_total_unrealized_pnl(prices)
        return cash + upnl

    def _log_status(self, prices: dict, nav: float):
        risk_snap = self.risk_mgr.get_risk_snapshot(nav, prices)
        port_summary = self.position_tracker.get_portfolio_summary(prices)

        logger.info(f"💰 NAV={nav:.2f} | 回撤={risk_snap['drawdown_pct']:.2%} | "
                    f"日PnL={risk_snap['daily_pnl_pct']:+.2%} | "
                    f"敞口={risk_snap['exposure_pct']:.1%} | "
                    f"持仓={risk_snap['open_positions']}")

        for sym, info in port_summary["positions"].items():
            logger.info(f"  📊 {sym}: qty={info['qty']:.6f} "
                        f"价值={info['market_value']:.2f} "
                        f"浮盈={info['upnl_pct']:+.2%}")

    # ==============================================================
    # 运行
    # ==============================================================
    def run(self, interval_seconds: int = 3600):
        """持续运行"""
        self._running = True
        logger.info(f"交易循环启动 | 间隔={interval_seconds}s")

        try:
            while self._running:
                try:
                    self.run_cycle()
                except Exception as e:
                    logger.error(f"循环异常: {e}\n{traceback.format_exc()}")
                logger.info(f"下轮 {interval_seconds}s 后...")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self.shutdown()

    def shutdown(self):
        self._running = False
        self.event_bus.stop()
        logger.info(f"引擎关闭 | 共运行 {self.cycle_count} 轮 | "
                    f"执行统计: {self.executor.stats}")

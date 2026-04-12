"""
回测引擎 v2 — 政体自适应趋势策略（Regime-Adaptive Trend Following）

核心改动（vs v1）：
  1. 使用 FeatureEngine 计算 241 个特征（和实盘完全一致）
  2. 使用 RegimeAdaptiveStrategy 做信号决策（和实盘完全一致）
  3. 支持 regime 统计、月度归因、与 Buy&Hold 对比

用法:
    python backtest_runner.py                          # 默认全量回测
    python backtest_runner.py --capital 5000            # 5000U 本金
    python backtest_runner.py --start 2025-01-01        # 指定起始日期
    python backtest_runner.py --snapshot result.txt     # 指定输出文件
    python backtest_runner.py --no-regime               # 关闭 regime 过滤（对照组）
"""

import argparse
import sqlite3
import time
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.features import FeatureEngine
from alpha.regime_strategy import (
    RegimeAdaptiveStrategy, RegimeStrategyConfig,
    classify_regime, add_regime_column,
    REGIME_BULL_TREND, REGIME_BULL_WEAK, REGIME_RANGE,
    REGIME_BEAR_WEAK, REGIME_BEAR_TREND,
)
from utils.logger import get_logger

logger = get_logger("backtest_v2")

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def ts_str(ms_or_val) -> str:
    """毫秒时间戳 → 可读字符串"""
    try:
        if isinstance(ms_or_val, (int, float, np.integer, np.floating)):
            return datetime.utcfromtimestamp(int(ms_or_val) / 1000).strftime("%Y-%m-%d %H:%M")
        return str(ms_or_val)
    except Exception:
        return str(ms_or_val)


# ─────────────────────────────────────────────────────────────
# 回测引擎
# ─────────────────────────────────────────────────────────────

class BacktestEngineV2:
    """
    政体自适应回测引擎

    关键设计：
      - 使用 FeatureEngine.compute_all() 计算特征（和实盘 _process_symbol 一致）
      - 使用 RegimeAdaptiveStrategy 做 should_enter / check_exit / calc_position
      - 回测循环复用 cmd_validate_strategy 的逻辑，确保策略一致性
    """

    def __init__(self, db_path: str, initial_capital: float = 10000.0,
                 start_date: str = None, end_date: str = None,
                 cfg: RegimeStrategyConfig = None,
                 enable_regime: bool = True):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.enable_regime = enable_regime

        self.cfg = cfg or RegimeStrategyConfig()
        self.strategy = RegimeAdaptiveStrategy(self.cfg)
        self.feature_engine = FeatureEngine()

        # 结果
        self.all_trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.regime_log: list[dict] = []

    # ─────────────────────────────────────────────────────────
    # 数据加载
    # ─────────────────────────────────────────────────────────

    def _load_klines(self, symbol: str, interval: str) -> pd.DataFrame:
        """从 SQLite 加载 K 线数据"""
        conn = sqlite3.connect(self.db_path)

        conditions = ["symbol=?", "interval=?"]
        params: list = [symbol, interval]

        if self.start_date:
            start_ms = int(datetime.strptime(self.start_date, "%Y-%m-%d")
                          .replace(tzinfo=timezone.utc).timestamp() * 1000)
            conditions.append("open_time >= ?")
            params.append(start_ms)
        if self.end_date:
            end_ms = int(datetime.strptime(self.end_date, "%Y-%m-%d")
                        .replace(tzinfo=timezone.utc).timestamp() * 1000)
            conditions.append("open_time <= ?")
            params.append(end_ms)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM klines WHERE {where} ORDER BY open_time ASC"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            logger.warning(f"  ⚠️  {symbol}/{interval} 无数据")
        else:
            start = datetime.utcfromtimestamp(df["open_time"].iloc[0] / 1000)
            end = datetime.utcfromtimestamp(df["open_time"].iloc[-1] / 1000)
            logger.info(f"  ✅ {symbol}/{interval}: {len(df)} 条 | "
                        f"{start.date()} ~ {end.date()}")
        return df

    # ─────────────────────────────────────────────────────────
    # 特征计算（和实盘 _process_symbol 完全一致）
    # ─────────────────────────────────────────────────────────

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        用 FeatureEngine 计算特征 → 添加 regime 列

        这和 main.py 中实盘/模拟交易的 _process_symbol 使用相同的
        FeatureEngine.compute_all() 管线，保证特征一致。
        """
        features = self.feature_engine.compute_all(df)
        if features.empty:
            return pd.DataFrame()

        # 添加原始 OHLCV 列（特征表可能丢失部分）
        for col in ["open", "high", "low", "volume", "open_time"]:
            if col in df.columns and col not in features.columns:
                features[col] = df[col].values[:len(features)]

        # 添加 regime 标签
        if self.enable_regime:
            features = add_regime_column(features)
        else:
            features["regime"] = REGIME_BULL_TREND  # 对照组：始终认为是牛市

        return features

    # ─────────────────────────────────────────────────────────
    # 单标的回测循环
    # ─────────────────────────────────────────────────────────

    def _backtest_symbol(self, symbol: str, features: pd.DataFrame,
                         capital: float) -> tuple[float, list[dict]]:
        """
        对单个标的执行回测

        返回: (final_cash, trades_list)

        *** 核心逻辑与 main.py cmd_validate_strategy 完全一致 ***
        """
        cfg = self.cfg
        strategy = self.strategy
        cash = capital
        position = None
        trades = []

        n = len(features)
        if n < 100:
            logger.warning(f"  {symbol}: 特征数据不足 ({n} 行)，跳过")
            return cash, trades

        for idx in range(1, n):
            row = features.iloc[idx]
            prev_row = features.iloc[idx - 1]

            close = row.get("close", 0)
            high = row.get("high", close)
            ts = ts_str(row.get("open_time", 0))
            regime = row.get("regime", REGIME_RANGE)

            if pd.isna(close) or close <= 0:
                continue

            # 记录 regime 分布（用于统计报告）
            self.regime_log.append({
                "symbol": symbol, "time": ts, "regime": regime,
                "close": close,
            })

            # ════════════════════════════════════════════════
            # 持仓中 → 检查出场
            # ════════════════════════════════════════════════
            if position is not None:
                bar_count = idx - position["entry_bar"]

                # 更新最高价
                position["highest_since_entry"] = max(
                    position["highest_since_entry"], high)

                # 减仓检查
                if strategy.check_partial_exit(row, position):
                    sell_qty = position["original_qty"] * cfg.partial_exit_pct
                    if sell_qty > 0 and sell_qty < position["qty"]:
                        sell_price = close * (1 - cfg.slippage_pct)
                        proceeds = sell_qty * sell_price * (1 - cfg.commission_pct)
                        cash += proceeds
                        position["qty"] -= sell_qty
                        position["partial_done"] = True

                # 出场检查
                should_exit, reason = strategy.check_exit(row, position, bar_count)
                if should_exit:
                    exit_price = close * (1 - cfg.slippage_pct)
                    proceeds = position["qty"] * exit_price * (1 - cfg.commission_pct)
                    cost_basis = position["qty"] * position["entry_price"] * \
                                 (1 + cfg.commission_pct)
                    pnl = proceeds - cost_basis
                    pnl_pct = exit_price / position["entry_price"] - 1
                    cash += proceeds

                    trades.append({
                        "symbol": symbol,
                        "side": "LONG",
                        "entry_price": round(position["entry_price"], 2),
                        "exit_price": round(exit_price, 2),
                        "quantity": position["original_qty"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": pnl_pct,
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "holding_bars": bar_count,
                        "exit_reason": reason,
                        "regime_at_entry": position["regime"],
                        "regime_at_exit": regime,
                    })

                    icon = "🟢" if pnl > 0 else "🔴"
                    logger.info(
                        f"  {icon} {symbol} | {position['entry_time']} → {ts} | "
                        f"入={position['entry_price']:.1f} 出={exit_price:.1f} | "
                        f"PnL={pnl:+.2f} ({pnl_pct:+.1%}) | "
                        f"{bar_count}bars | {reason} | {regime}")

                    position = None

            # ════════════════════════════════════════════════
            # 无持仓 → 检查入场
            # ════════════════════════════════════════════════
            if position is None and strategy.should_enter(row, prev_row):
                natr = row.get("natr_20", 0)
                if pd.isna(natr) or natr <= 0:
                    continue

                entry_price = close * (1 + cfg.slippage_pct)
                qty, stop_loss = strategy.calc_position(
                    cash, entry_price, natr, regime)

                if qty > 0:
                    cost = qty * entry_price * (1 + cfg.commission_pct)
                    if cost <= cash:
                        cash -= cost
                        atr_abs = natr * entry_price
                        position = {
                            "qty": qty,
                            "original_qty": qty,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "highest_since_entry": high,
                            "atr_at_entry": atr_abs,
                            "entry_bar": idx,
                            "entry_time": ts,
                            "regime": regime,
                            "partial_done": False,
                        }
                        logger.info(
                            f"  🔵 {symbol} | {ts} | 开仓 @ {entry_price:.1f} | "
                            f"qty={qty:.6f} | {qty * entry_price:.1f} USDT | "
                            f"止损={stop_loss:.1f} | {regime}")

            # 权益曲线快照（每根 bar 都记录）
            equity = cash
            if position is not None:
                equity += position["qty"] * close
            self.equity_curve.append(equity)

        # 回测结束：强制平仓
        if position is not None:
            last_row = features.iloc[-1]
            last_close = last_row.get("close", 0)
            exit_price = last_close * (1 - cfg.slippage_pct)
            bar_count = n - 1 - position["entry_bar"]
            proceeds = position["qty"] * exit_price * (1 - cfg.commission_pct)
            cost_basis = position["qty"] * position["entry_price"] * \
                         (1 + cfg.commission_pct)
            pnl = proceeds - cost_basis
            pnl_pct = exit_price / position["entry_price"] - 1
            cash += proceeds

            trades.append({
                "symbol": symbol,
                "side": "LONG",
                "entry_price": round(position["entry_price"], 2),
                "exit_price": round(exit_price, 2),
                "quantity": position["original_qty"],
                "pnl": round(pnl, 2),
                "pnl_pct": pnl_pct,
                "entry_time": position["entry_time"],
                "exit_time": ts_str(last_row.get("open_time", 0)),
                "holding_bars": bar_count,
                "exit_reason": "backtest_end",
                "regime_at_entry": position["regime"],
                "regime_at_exit": last_row.get("regime", REGIME_RANGE),
            })
            position = None

        return cash, trades

    # ─────────────────────────────────────────────────────────
    # Buy & Hold 基准
    # ─────────────────────────────────────────────────────────

    def _calc_buy_hold(self, features_dict: dict[str, pd.DataFrame]) -> float:
        """计算等权 Buy & Hold 收益率"""
        returns = []
        for symbol, features in features_dict.items():
            if len(features) < 2:
                continue
            first_close = features["close"].iloc[0]
            last_close = features["close"].iloc[-1]
            if first_close > 0:
                returns.append(last_close / first_close - 1)
        if not returns:
            return 0.0
        return sum(returns) / len(returns)

    # ─────────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        """执行完整回测，返回报告字典"""
        mode_str = "Regime-Adaptive" if self.enable_regime else "No-Regime (对照组)"
        print("=" * 65)
        print(f"  回测引擎 v2 — {mode_str}")
        print("=" * 65)
        print(f"  本金:    {self.initial_capital:.0f} USDT")
        print(f"  标的:    {', '.join(SYMBOLS)}")
        print(f"  日期:    {self.start_date or '全部'} ~ {self.end_date or '全部'}")
        print(f"  Regime:  {'启用' if self.enable_regime else '关闭'}")
        print(f"  手续费:  {self.cfg.commission_pct * 100:.2f}%")
        print(f"  滑点:    {self.cfg.slippage_pct * 100:.3f}%")
        print()

        # ── 加载数据 ──
        print("📂 加载 1h K 线数据...")
        klines = {}
        for symbol in SYMBOLS:
            klines[symbol] = self._load_klines(symbol, "1h")

        # ── 计算特征 ──
        print("\n🔧 计算特征 + Regime 标签...")
        features_dict: dict[str, pd.DataFrame] = {}
        for symbol in SYMBOLS:
            df = klines[symbol]
            if len(df) < 500:
                print(f"  ⚠️  {symbol} 数据不足 ({len(df)} 条)，跳过")
                continue
            features = self._prepare_features(df)
            if not features.empty:
                features_dict[symbol] = features
                regime_counts = features["regime"].value_counts()
                print(f"  {symbol}: {len(features)} 行特征 | "
                      f"regime 分布: {dict(regime_counts)}")

        if not features_dict:
            print("\n❌ 无足够数据进行回测！")
            return {}

        # ── 运行回测 ──
        t0 = time.time()
        print("\n📈 开始回测...")
        print("-" * 65)

        # 每个标的分配等额资金
        per_symbol_capital = self.initial_capital / len(features_dict)
        total_cash = 0.0

        for symbol, features in features_dict.items():
            print(f"\n  ── {symbol} ({len(features)} bars) ──")
            cash, trades = self._backtest_symbol(
                symbol, features, per_symbol_capital)
            total_cash += cash
            self.all_trades.extend(trades)

        elapsed = time.time() - t0

        # ── Buy & Hold 基准 ──
        buy_hold_ret = self._calc_buy_hold(features_dict)

        # ── 生成报告 ──
        report = self._generate_report(total_cash, buy_hold_ret, elapsed)
        return report

    # ─────────────────────────────────────────────────────────
    # 报告生成
    # ─────────────────────────────────────────────────────────

    def _generate_report(self, final_cash: float,
                         buy_hold_ret: float, elapsed: float) -> dict:
        """生成完整回测报告"""
        total_return = (final_cash - self.initial_capital) / self.initial_capital
        trades = self.all_trades

        all_pnl = [t["pnl"] for t in trades]
        wins = [p for p in all_pnl if p > 0]
        losses = [p for p in all_pnl if p < 0]

        win_rate = len(wins) / len(all_pnl) if all_pnl else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else (
            float("inf") if wins else 0)
        expectancy = np.mean(all_pnl) if all_pnl else 0

        # 权益曲线指标
        eq = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_capital])
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.clip(peak, 1e-10, None)
        max_dd = float(dd.max()) if len(dd) > 0 else 0

        # 夏普（简化：假设 1h bar）
        if len(eq) > 2:
            rets = np.diff(eq) / eq[:-1]
            rets = rets[np.isfinite(rets)]
            if len(rets) > 1 and rets.std() > 0:
                sharpe = float(rets.mean() / rets.std() * np.sqrt(8766))
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # 按 regime 分组统计
        regime_stats = {}
        for t in trades:
            r = t.get("regime_at_entry", "UNKNOWN")
            if r not in regime_stats:
                regime_stats[r] = {"trades": 0, "wins": 0, "pnl": 0.0}
            regime_stats[r]["trades"] += 1
            regime_stats[r]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                regime_stats[r]["wins"] += 1

        # 按标的分组
        symbol_stats = {}
        for t in trades:
            s = t["symbol"]
            if s not in symbol_stats:
                symbol_stats[s] = {"trades": 0, "wins": 0, "pnl": 0.0}
            symbol_stats[s]["trades"] += 1
            symbol_stats[s]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                symbol_stats[s]["wins"] += 1

        # 月度 PnL
        monthly_pnl = {}
        for t in trades:
            if t["exit_time"]:
                month = t["exit_time"][:7]
                monthly_pnl[month] = monthly_pnl.get(month, 0) + t["pnl"]

        # 退出原因统计
        exit_reasons = {}
        for t in trades:
            r = t.get("exit_reason", "unknown")
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "pnl": 0.0}
            exit_reasons[r]["count"] += 1
            exit_reasons[r]["pnl"] += t["pnl"]

        report = {
            "summary": {
                "initial_capital": self.initial_capital,
                "final_equity": round(final_cash, 2),
                "total_return_pct": round(total_return * 100, 2),
                "total_pnl": round(final_cash - self.initial_capital, 2),
                "buy_hold_return_pct": round(buy_hold_ret * 100, 2),
                "vs_buy_hold_pct": round((total_return - buy_hold_ret) * 100, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "regime_filter": self.enable_regime,
                "elapsed_seconds": round(elapsed, 1),
            },
            "trades_overview": {
                "total_trades": len(trades),
                "win_rate_pct": round(win_rate * 100, 1),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 3),
                "expectancy": round(expectancy, 2),
                "best_trade": round(max(all_pnl), 2) if all_pnl else 0,
                "worst_trade": round(min(all_pnl), 2) if all_pnl else 0,
                "avg_holding_bars": round(np.mean(
                    [t["holding_bars"] for t in trades]), 1) if trades else 0,
            },
            "regime_stats": {
                r: {
                    "trades": s["trades"],
                    "win_rate_pct": round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0,
                    "total_pnl": round(s["pnl"], 2),
                }
                for r, s in regime_stats.items()
            },
            "symbol_stats": {
                s: {
                    "trades": d["trades"],
                    "win_rate_pct": round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0,
                    "total_pnl": round(d["pnl"], 2),
                }
                for s, d in symbol_stats.items()
            },
            "exit_reasons": {
                r: {"count": d["count"], "pnl": round(d["pnl"], 2)}
                for r, d in exit_reasons.items()
            },
            "monthly_pnl": {
                m: round(p, 2) for m, p in sorted(monthly_pnl.items())
            },
            "strategy_config": {
                "commission_pct": self.cfg.commission_pct,
                "slippage_pct": self.cfg.slippage_pct,
                "risk_per_trade": self.cfg.risk_per_trade,
                "risk_per_trade_weak": self.cfg.risk_per_trade_weak,
                "trailing_atr_mult": self.cfg.trailing_atr_mult,
                "take_profit_atr_mult": self.cfg.take_profit_atr_mult,
                "stop_atr_mult": self.cfg.stop_atr_mult,
                "max_holding_bars": self.cfg.max_holding_bars,
                "rsi_low": self.cfg.rsi_low,
                "rsi_high": self.cfg.rsi_high,
                "adx_min": self.cfg.adx_min,
            },
            "all_trades": trades,
        }

        self._print_report(report)
        return report

    def _print_report(self, report: dict):
        """格式化打印报告到控制台"""
        s = report["summary"]
        t = report["trades_overview"]

        icon = "🟢" if s["total_pnl"] > 0 else "🔴"
        bh_icon = "✅" if s["vs_buy_hold_pct"] > 0 else "❌"

        print()
        print("=" * 65)
        print("                    回 测 报 告 v2")
        print("=" * 65)
        print(f"  {icon} 策略收益:       {s['total_pnl']:>+10.2f} USDT "
              f"({s['total_return_pct']:+.2f}%)")
        print(f"     Buy & Hold:      {s['buy_hold_return_pct']:>+10.2f}%")
        print(f"     vs Buy&Hold:     {s['vs_buy_hold_pct']:>+10.2f}%  {bh_icon}")
        print(f"     最大回撤:        {s['max_drawdown_pct']:>10.2f}%")
        print(f"     夏普比率:        {s['sharpe_ratio']:>10.3f}")
        print(f"     Regime 过滤:     {'启用' if s['regime_filter'] else '关闭'}")
        print("-" * 65)
        print(f"  📊 交易统计:")
        print(f"     总交易数:        {t['total_trades']:>10d}")
        print(f"     胜率:            {t['win_rate_pct']:>9.1f}%")
        print(f"     平均盈利:        {t['avg_win']:>+10.2f}")
        print(f"     平均亏损:        {t['avg_loss']:>+10.2f}")
        print(f"     盈亏比:          {t['profit_factor']:>10.3f}")
        print(f"     期望收益:        {t['expectancy']:>+10.2f}")
        print(f"     平均持仓:        {t['avg_holding_bars']:>10.1f} bars (1h)")

        # Regime 分组
        rs = report.get("regime_stats", {})
        if rs:
            print("-" * 65)
            print(f"  🏛  Regime 分组:")
            for regime, data in sorted(rs.items()):
                r_icon = "🟢" if data["total_pnl"] > 0 else "🔴"
                print(f"     {r_icon} {regime:<15} | "
                      f"{data['trades']:>3d} 笔 | "
                      f"胜率={data['win_rate_pct']:>5.1f}% | "
                      f"PnL={data['total_pnl']:>+10.2f}")

        # 按标的分组
        ss = report.get("symbol_stats", {})
        if ss:
            print("-" * 65)
            print(f"  📈 标的分组:")
            for sym, data in ss.items():
                sym_icon = "🟢" if data["total_pnl"] > 0 else "🔴"
                print(f"     {sym_icon} {sym:<10} | "
                      f"{data['trades']:>3d} 笔 | "
                      f"胜率={data['win_rate_pct']:>5.1f}% | "
                      f"PnL={data['total_pnl']:>+10.2f}")

        # 退出原因
        er = report.get("exit_reasons", {})
        if er:
            print("-" * 65)
            print(f"  🚪 退出原因:")
            for reason, data in sorted(er.items(), key=lambda x: x[1]["count"],
                                       reverse=True):
                print(f"     {reason:<20} | {data['count']:>3d} 次 | "
                      f"PnL={data['pnl']:>+10.2f}")

        # 月度 PnL
        mp = report.get("monthly_pnl", {})
        if mp:
            print("-" * 65)
            print(f"  📅 月度 PnL:")
            for month, pnl in mp.items():
                m_icon = "🟢" if pnl > 0 else "🔴"
                print(f"     {m_icon} {month}: {pnl:+.2f} USDT")

        print("=" * 65)
        print(f"  ⏱  耗时: {s['elapsed_seconds']:.1f}s")


# ─────────────────────────────────────────────────────────────
# 快照输出
# ─────────────────────────────────────────────────────────────

def write_snapshot(report: dict, output_path: str, db_path: str,
                   start_date: str, end_date: str):
    """将回测结果写入快照文件（可直接喂给 Claude 分析）"""
    lines = []
    lines.append("=" * 70)
    lines.append("BACKTEST SNAPSHOT v2 (Regime-Adaptive)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Database: {db_path}")
    lines.append(f"Period: {start_date or 'all'} ~ {end_date or 'all'}")
    lines.append("=" * 70)

    # 概要
    for section_name in ["summary", "trades_overview", "regime_stats",
                         "symbol_stats", "exit_reasons", "strategy_config"]:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"SECTION: {section_name.upper()}")
        lines.append("=" * 70)
        data = report.get(section_name, {})
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    lines.append(f"  [{k}]")
                    for k2, v2 in v.items():
                        lines.append(f"    {k2}: {v2}")
                else:
                    lines.append(f"  {k}: {v}")

    # 月度 PnL
    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: MONTHLY_PNL")
    lines.append("=" * 70)
    for month, pnl in report.get("monthly_pnl", {}).items():
        icon = "+" if pnl > 0 else ""
        lines.append(f"  {month}: {icon}{pnl:.2f} USDT")

    # 交易明细
    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: ALL_TRADES")
    lines.append("=" * 70)
    header = (f"{'#':>4} | {'Symbol':<10} | {'Side':<6} | "
              f"{'Entry':>10} | {'Exit':>10} | {'PnL':>10} | {'PnL%':>8} | "
              f"{'Bars':>5} | {'Reason':<16} | {'Regime':>12} | "
              f"{'Entry Time':<16} | {'Exit Time':<16}")
    lines.append(header)
    lines.append("-" * 150)

    for idx, t in enumerate(report.get("all_trades", []), 1):
        lines.append(
            f"{idx:>4} | {t['symbol']:<10} | {t['side']:<6} | "
            f"{t['entry_price']:>10.2f} | {t['exit_price']:>10.2f} | "
            f"{t['pnl']:>+10.2f} | {t['pnl_pct']:>+7.2%} | "
            f"{t['holding_bars']:>5} | {t['exit_reason']:<16} | "
            f"{t.get('regime_at_entry', '?'):>12} | "
            f"{t['entry_time']:<16} | {t['exit_time']:<16}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SNAPSHOT")
    lines.append("=" * 70)

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n📄 快照已保存: {output_path}")
    print(f"   大小: {len(content)} 字符 / {len(lines)} 行")
    print(f"   💡 将此文件上传给 Claude 即可分析回测结果")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="量化回测引擎 v2 — 政体自适应趋势策略")
    parser.add_argument("--db", type=str, default="data/quant.db",
                        help="数据库路径")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="初始资金 (USDT)")
    parser.add_argument("--start", type=str, default=None,
                        help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="快照输出路径；省略时：开启 regime 为 backtest_v2_regime_snapshot.txt，"
        "--no-regime 为 backtest_v2_no_regime_snapshot.txt",
    )
    parser.add_argument("--no-regime", action="store_true",
                        help="关闭 regime 过滤（对照组，测试 regime 效果）")

    # 可调策略参数
    parser.add_argument("--risk", type=float, default=None,
                        help="单笔风险比例（默认 0.03）")
    parser.add_argument("--trail", type=float, default=None,
                        help="移动止损 ATR 倍数（默认 2.5）")
    parser.add_argument("--tp", type=float, default=None,
                        help="止盈 ATR 倍数（默认 4.0）")
    parser.add_argument("--max-bars", type=int, default=None,
                        help="最大持仓 bars（默认 30）")

    args = parser.parse_args()

    # 检查数据库
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ 数据库不存在: {db_path}")
        print("   请先运行: python main.py --sync-data")
        sys.exit(1)

    # 构建配置
    cfg = RegimeStrategyConfig()
    if args.risk is not None:
        cfg.risk_per_trade = args.risk
        cfg.risk_per_trade_weak = args.risk / 2
    if args.trail is not None:
        cfg.trailing_atr_mult = args.trail
    if args.tp is not None:
        cfg.take_profit_atr_mult = args.tp
    if args.max_bars is not None:
        cfg.max_holding_bars = args.max_bars

    # 运行回测
    engine = BacktestEngineV2(
        db_path=str(db_path),
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
        cfg=cfg,
        enable_regime=not args.no_regime,
    )

    report = engine.run()

    if report:
        snap = args.snapshot or (
            "backtest_v2_no_regime_snapshot.txt"
            if args.no_regime
            else "backtest_v2_regime_snapshot.txt"
        )
        write_snapshot(report, snap, str(db_path),
                       args.start, args.end)


if __name__ == "__main__":
    main()

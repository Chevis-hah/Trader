"""
回测引擎 — 趋势跟踪 + 网格交易 双策略组合回测

用法:
    python backtest_runner.py                      # 默认参数回测
    python backtest_runner.py --capital 5000        # 5000U 本金
    python backtest_runner.py --start 2025-01-01    # 指定起始日期
    python backtest_runner.py --snapshot result.txt  # 指定输出文件

输出:
    1. 控制台实时进度
    2. 快照文件（含完整过程数据，可喂给 Claude 分析）
"""
import argparse
import sqlite3
import time
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 策略
from alpha.trend_following import TrendFollowingStrategy, TrendPosition, TrendSignal
from alpha.grid_strategy import GridTradingStrategy, GridState

# =================================================================
# 回测核心
# =================================================================

@dataclass
class Trade:
    """单笔交易记录"""
    symbol: str
    strategy: str        # "trend" / "grid"
    side: str            # BUY / SELL
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_bars: int = 0
    exit_reason: str = ""


@dataclass
class Snapshot:
    """某一时刻的状态快照（用于过程监控）"""
    timestamp: str = ""
    bar_index: int = 0
    close_btc: float = 0.0
    close_eth: float = 0.0
    equity: float = 0.0
    cash: float = 0.0
    trend_exposure: float = 0.0
    grid_exposure: float = 0.0
    total_exposure_pct: float = 0.0
    drawdown_pct: float = 0.0
    trend_pnl_cum: float = 0.0
    grid_pnl_cum: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_rolling: float = 0.0


class BacktestEngine:
    """
    双策略组合回测引擎

    策略分配：
      - 趋势跟踪：60% 资金上限（4h 级别，BTC+ETH）
      - 网格交易：40% 资金上限（1h 级别，BTC+ETH）
    """

    def __init__(self, db_path: str, initial_capital: float = 10000.0,
                 start_date: str = None, end_date: str = None):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.start_date = start_date
        self.end_date = end_date

        # 策略实例
        self.trend_strategy = TrendFollowingStrategy(
            fast_period=20, slow_period=60,
            atr_period=14, atr_filter_period=60,
            atr_risk_mult=2.0, risk_per_trade=0.02,
            trailing_atr_mult=2.5)

        self.grid_strategy = GridTradingStrategy(
            n_grids=5, grid_atr_mult=0.5,
            qty_per_grid_pct=0.04, max_total_pct=0.40,
            stop_loss_pct=0.05, adx_threshold=25.0)

        # 持仓
        self.trend_positions: dict[str, TrendPosition] = {}  # symbol -> position
        self.grid_states: dict[str, GridState] = {}           # symbol -> state

        # 记录
        self.trades: list[Trade] = []
        self.snapshots: list[Snapshot] = []
        self.equity_curve: list[float] = []
        self.returns: list[float] = []
        self.peak_equity = initial_capital

        # 过程统计
        self.trend_pnl_cum = 0.0
        self.grid_pnl_cum = 0.0
        self.symbols = ["BTCUSDT", "ETHUSDT"]

    # =============================================================
    # 数据加载
    # =============================================================
    def _load_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """从 SQLite 加载 K 线数据"""
        conn = sqlite3.connect(self.db_path)

        conditions = ["symbol=?", "interval=?"]
        params = [symbol, interval]

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
            print(f"  ⚠️  {symbol}/{interval} 无数据")
        else:
            start = datetime.utcfromtimestamp(df["open_time"].iloc[0] / 1000)
            end = datetime.utcfromtimestamp(df["open_time"].iloc[-1] / 1000)
            print(f"  ✅ {symbol}/{interval}: {len(df)} 条 | {start.date()} ~ {end.date()}")

        return df

    # =============================================================
    # 趋势回测
    # =============================================================
    def _backtest_trend(self, dfs_4h: dict[str, pd.DataFrame]):
        """趋势跟踪策略回测"""
        print("\n📈 趋势跟踪策略回测（4h 级别）")
        print("-" * 50)

        for symbol in self.symbols:
            df = dfs_4h.get(symbol)
            if df is None or len(df) < 100:
                print(f"  {symbol}: 数据不足，跳过")
                continue

            # 计算指标
            df = self.trend_strategy.compute_indicators(df)
            position = TrendPosition(symbol=symbol)
            entry_bar = 0

            for i in range(1, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i - 1]

                if pd.isna(row.get("atr", np.nan)):
                    continue

                equity = self._calc_equity(dfs_4h, i)
                signal = self.trend_strategy.generate_signal(
                    row, prev, position if position.direction != "FLAT" else None, equity)

                ts = datetime.utcfromtimestamp(row["open_time"] / 1000).strftime("%Y-%m-%d %H:%M")

                # 开仓
                if signal.direction == "LONG" and position.direction == "FLAT":
                    qty_value = equity * signal.position_size_pct
                    qty = qty_value / row["close"]

                    # 检查现金够不够
                    cost = qty * row["close"]
                    if cost > self.cash * 0.6:  # 趋势策略最多用 60% 资金
                        cost = self.cash * 0.6
                        qty = cost / row["close"]

                    if cost < 10:
                        continue

                    self.cash -= cost
                    position = TrendPosition(
                        symbol=symbol, direction="LONG",
                        entry_price=row["close"], entry_time=row["open_time"],
                        quantity=qty, stop_loss=signal.stop_loss,
                        highest_since_entry=row["close"],
                        trailing_stop=signal.stop_loss,
                        atr_at_entry=row["atr"])
                    entry_bar = i

                # 平仓
                elif signal.direction == "CLOSE_LONG" and position.direction == "LONG":
                    exit_price = row["close"]
                    pnl = position.quantity * (exit_price - position.entry_price)
                    pnl_pct = (exit_price - position.entry_price) / position.entry_price

                    self.cash += position.quantity * exit_price
                    self.trend_pnl_cum += pnl

                    entry_ts = datetime.utcfromtimestamp(
                        position.entry_time / 1000).strftime("%Y-%m-%d %H:%M")

                    trade = Trade(
                        symbol=symbol, strategy="trend", side="LONG",
                        entry_price=position.entry_price, exit_price=exit_price,
                        quantity=position.quantity,
                        entry_time=entry_ts, exit_time=ts,
                        pnl=pnl, pnl_pct=pnl_pct,
                        holding_bars=i - entry_bar,
                        exit_reason="trailing_stop" if exit_price <= position.trailing_stop else "ma_cross")
                    self.trades.append(trade)

                    icon = "🟢" if pnl > 0 else "🔴"
                    print(f"  {icon} {symbol} TREND | {entry_ts} → {ts} | "
                          f"入={position.entry_price:.1f} 出={exit_price:.1f} | "
                          f"PnL={pnl:+.2f} ({pnl_pct:+.1%}) | "
                          f"持仓={i - entry_bar}bars | {trade.exit_reason}")

                    position = TrendPosition(symbol=symbol)

            # 回测结束时如果还有持仓，按最后价格平仓
            if position.direction == "LONG":
                last_price = df["close"].iloc[-1]
                pnl = position.quantity * (last_price - position.entry_price)
                self.cash += position.quantity * last_price
                self.trend_pnl_cum += pnl
                self.trades.append(Trade(
                    symbol=symbol, strategy="trend", side="LONG",
                    entry_price=position.entry_price, exit_price=last_price,
                    quantity=position.quantity, pnl=pnl,
                    pnl_pct=(last_price - position.entry_price) / position.entry_price,
                    exit_reason="backtest_end"))

    # =============================================================
    # 网格回测
    # =============================================================
    def _backtest_grid(self, dfs_1h: dict[str, pd.DataFrame]):
        """网格交易策略回测"""
        print("\n📊 网格交易策略回测（1h 级别）")
        print("-" * 50)

        for symbol in self.symbols:
            df = dfs_1h.get(symbol)
            if df is None or len(df) < 100:
                print(f"  {symbol}: 数据不足，跳过")
                continue

            df = self.grid_strategy.compute_indicators(df)
            state = GridState()
            grid_trade_count = 0
            grid_pnl = 0.0

            for i in range(1, len(df)):
                row = df.iloc[i]
                if pd.isna(row.get("atr", np.nan)):
                    continue

                equity = self.cash  # 简化
                ts = datetime.utcfromtimestamp(row["open_time"] / 1000).strftime("%Y-%m-%d %H:%M")

                # 如果网格不活跃，检查是否该建立新网格
                if not state.active:
                    if self.grid_strategy.should_activate(row):
                        grid_capital = min(self.cash * 0.4, self.cash)  # 网格最多 40%
                        if grid_capital >= 50:
                            state = self.grid_strategy.create_grid(
                                row["close"], row["atr"], grid_capital, row["open_time"])

                # 处理当前 bar
                if state.active:
                    actions = self.grid_strategy.process_bar(row, state, equity)
                    for act in actions:
                        if act["action"] == "GRID_BUY":
                            self.cash -= act["price"] * act["qty"]
                            grid_trade_count += 1
                        elif act["action"] == "GRID_SELL":
                            self.cash += act["price"] * act["qty"]
                            grid_pnl += act["pnl"]
                            self.grid_pnl_cum += act["pnl"]
                            grid_trade_count += 1

                            icon = "🟢" if act["pnl"] > 0 else "🔴"
                            print(f"  {icon} {symbol} GRID  | {ts} | "
                                  f"卖出@{act['price']:.1f} | PnL={act['pnl']:+.2f}")

                        elif act["action"] == "GRID_STOP_LOSS":
                            self.cash += act["price"] * act["qty"]
                            grid_pnl += act["pnl"]
                            self.grid_pnl_cum += act["pnl"]
                            print(f"  🔴 {symbol} GRID STOP | {ts} | PnL={act['pnl']:+.2f}")

                    # 所有网格都被填充了，重置
                    all_filled = all(l.filled for l in state.levels)
                    if all_filled or not state.active:
                        if state.open_qty > 0:
                            # 按收盘价平仓剩余持仓
                            close_pnl = state.open_qty * (row["close"] - state.avg_cost)
                            self.cash += state.open_qty * row["close"]
                            grid_pnl += close_pnl
                            self.grid_pnl_cum += close_pnl
                        state = GridState()

            # 回测结束，平掉网格持仓
            if state.active and state.open_qty > 0:
                last_price = df["close"].iloc[-1]
                close_pnl = state.open_qty * (last_price - state.avg_cost)
                self.cash += state.open_qty * last_price
                self.grid_pnl_cum += close_pnl

            print(f"  {symbol} 网格统计: {grid_trade_count} 笔交易 | 累计PnL={grid_pnl:+.2f}")

    # =============================================================
    # 辅助
    # =============================================================
    def _calc_equity(self, dfs: dict, bar_idx: int) -> float:
        """计算当前权益（简化）"""
        return self.cash + sum(
            pos.quantity * dfs[sym]["close"].iloc[min(bar_idx, len(dfs[sym])-1)]
            for sym, pos in self.trend_positions.items()
            if pos.direction != "FLAT" and sym in dfs)

    # =============================================================
    # 主入口
    # =============================================================
    def run(self) -> dict:
        """执行完整回测"""
        print("=" * 60)
        print("  量化回测引擎 v1.0 — 趋势跟踪 + 网格交易")
        print("=" * 60)
        print(f"  本金: {self.initial_capital:.0f} USDT")
        print(f"  标的: {', '.join(self.symbols)}")
        print(f"  日期: {self.start_date or '全部'} ~ {self.end_date or '全部'}")
        print()

        # 加载数据
        print("📂 加载数据...")
        dfs_4h = {}
        dfs_1h = {}
        for sym in self.symbols:
            dfs_4h[sym] = self._load_data(sym, "4h")
            dfs_1h[sym] = self._load_data(sym, "1h")

        # 检查数据
        has_4h = any(len(df) > 100 for df in dfs_4h.values())
        has_1h = any(len(df) > 100 for df in dfs_1h.values())

        if not has_4h and not has_1h:
            print("\n❌ 无足够数据进行回测！请先运行 python main.py --sync-data")
            return {}

        t0 = time.time()

        # 运行趋势回测
        if has_4h:
            self._backtest_trend(dfs_4h)

        # 运行网格回测
        if has_1h:
            self._backtest_grid(dfs_1h)

        elapsed = time.time() - t0

        # 汇总
        report = self._generate_report(elapsed)
        return report

    # =============================================================
    # 报告生成
    # =============================================================
    def _generate_report(self, elapsed: float) -> dict:
        """生成详细的回测报告"""
        final_equity = self.cash
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # 交易统计
        trend_trades = [t for t in self.trades if t.strategy == "trend"]
        grid_trades = [t for t in self.trades if t.strategy == "grid"]

        all_pnl = [t.pnl for t in self.trades]
        wins = [p for p in all_pnl if p > 0]
        losses = [p for p in all_pnl if p < 0]

        win_rate = len(wins) / len(all_pnl) if all_pnl else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
        expectancy = np.mean(all_pnl) if all_pnl else 0

        # 趋势策略单独统计
        trend_pnls = [t.pnl for t in trend_trades]
        trend_wins = [p for p in trend_pnls if p > 0]
        trend_losses = [p for p in trend_pnls if p < 0]
        trend_win_rate = len(trend_wins) / len(trend_pnls) if trend_pnls else 0
        trend_avg_hold = np.mean([t.holding_bars for t in trend_trades]) if trend_trades else 0

        report = {
            "summary": {
                "initial_capital": self.initial_capital,
                "final_equity": round(final_equity, 2),
                "total_return_pct": round(total_return * 100, 2),
                "total_pnl": round(final_equity - self.initial_capital, 2),
                "trend_pnl": round(self.trend_pnl_cum, 2),
                "grid_pnl": round(self.grid_pnl_cum, 2),
                "elapsed_seconds": round(elapsed, 1),
            },
            "trades_overview": {
                "total_trades": len(self.trades),
                "trend_trades": len(trend_trades),
                "grid_trades": len(grid_trades),
                "win_rate_pct": round(win_rate * 100, 1),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 3),
                "expectancy": round(expectancy, 2),
                "best_trade": round(max(all_pnl), 2) if all_pnl else 0,
                "worst_trade": round(min(all_pnl), 2) if all_pnl else 0,
            },
            "trend_stats": {
                "trades": len(trend_trades),
                "win_rate_pct": round(trend_win_rate * 100, 1),
                "total_pnl": round(self.trend_pnl_cum, 2),
                "avg_holding_bars": round(trend_avg_hold, 1),
                "avg_win": round(np.mean(trend_wins), 2) if trend_wins else 0,
                "avg_loss": round(np.mean(trend_losses), 2) if trend_losses else 0,
            },
            "grid_stats": {
                "total_pnl": round(self.grid_pnl_cum, 2),
            },
            "all_trades": [],
        }

        # 详细交易列表
        for t in self.trades:
            report["all_trades"].append({
                "symbol": t.symbol,
                "strategy": t.strategy,
                "side": t.side,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "qty": round(t.quantity, 6),
                "pnl": round(t.pnl, 2),
                "pnl_pct": f"{t.pnl_pct:+.2%}",
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "bars": t.holding_bars,
                "reason": t.exit_reason,
            })

        # 打印报告
        self._print_report(report)
        return report

    def _print_report(self, report: dict):
        """打印格式化报告"""
        s = report["summary"]
        t = report["trades_overview"]
        ts = report["trend_stats"]

        icon = "🟢" if s["total_pnl"] > 0 else "🔴"

        print()
        print("=" * 65)
        print("                    回 测 报 告")
        print("=" * 65)
        print(f"  {icon} 总收益:        {s['total_pnl']:>+10.2f} USDT ({s['total_return_pct']:+.2f}%)")
        print(f"     初始资金:       {s['initial_capital']:>10.0f} USDT")
        print(f"     最终权益:       {s['final_equity']:>10.2f} USDT")
        print(f"     趋势策略 PnL:   {s['trend_pnl']:>+10.2f} USDT")
        print(f"     网格策略 PnL:   {s['grid_pnl']:>+10.2f} USDT")
        print("-" * 65)
        print(f"  📊 交易统计:")
        print(f"     总交易数:       {t['total_trades']:>10d}")
        print(f"     趋势交易:       {t['trend_trades']:>10d}")
        print(f"     网格交易:       {t['grid_trades']:>10d}")
        print(f"     胜率:           {t['win_rate_pct']:>9.1f}%")
        print(f"     平均盈利:       {t['avg_win']:>+10.2f}")
        print(f"     平均亏损:       {t['avg_loss']:>+10.2f}")
        print(f"     盈亏比:         {t['profit_factor']:>10.3f}")
        print(f"     期望收益:       {t['expectancy']:>+10.2f}")
        print(f"     最大单笔盈利:   {t['best_trade']:>+10.2f}")
        print(f"     最大单笔亏损:   {t['worst_trade']:>+10.2f}")
        print("-" * 65)
        print(f"  📈 趋势策略详情:")
        print(f"     交易次数:       {ts['trades']:>10d}")
        print(f"     胜率:           {ts['win_rate_pct']:>9.1f}%")
        print(f"     平均持仓:       {ts['avg_holding_bars']:>10.1f} bars (4h)")
        print("=" * 65)
        print(f"  ⏱  耗时: {s['elapsed_seconds']:.1f}s")


# =================================================================
# 快照输出
# =================================================================

def write_snapshot(report: dict, output_path: str, db_path: str,
                   start_date: str, end_date: str):
    """将回测结果写入快照文件（可直接喂给 Claude）"""

    lines = []
    lines.append("=" * 70)
    lines.append("BACKTEST SNAPSHOT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Database: {db_path}")
    lines.append(f"Period: {start_date or 'all'} ~ {end_date or 'all'}")
    lines.append("=" * 70)

    # 概要
    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: SUMMARY")
    lines.append("=" * 70)
    for k, v in report.get("summary", {}).items():
        lines.append(f"  {k}: {v}")

    # 交易统计
    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: TRADES_OVERVIEW")
    lines.append("=" * 70)
    for k, v in report.get("trades_overview", {}).items():
        lines.append(f"  {k}: {v}")

    # 趋势策略统计
    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: TREND_STATS")
    lines.append("=" * 70)
    for k, v in report.get("trend_stats", {}).items():
        lines.append(f"  {k}: {v}")

    # 网格策略统计
    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: GRID_STATS")
    lines.append("=" * 70)
    for k, v in report.get("grid_stats", {}).items():
        lines.append(f"  {k}: {v}")

    # 每笔交易详情
    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: ALL_TRADES")
    lines.append("=" * 70)
    lines.append(f"{'#':>4} | {'Symbol':<10} | {'Strategy':<8} | {'Side':<6} | "
                 f"{'Entry':>10} | {'Exit':>10} | {'PnL':>10} | {'PnL%':>8} | "
                 f"{'Bars':>5} | {'Reason':<16} | {'Entry Time':<16} | {'Exit Time':<16}")
    lines.append("-" * 140)

    for idx, t in enumerate(report.get("all_trades", []), 1):
        lines.append(
            f"{idx:>4} | {t['symbol']:<10} | {t['strategy']:<8} | {t['side']:<6} | "
            f"{t['entry']:>10.2f} | {t['exit']:>10.2f} | {t['pnl']:>+10.2f} | "
            f"{t['pnl_pct']:>8} | {t['bars']:>5} | {t['reason']:<16} | "
            f"{t['entry_time']:<16} | {t['exit_time']:<16}")

    # 月度收益分析（从交易中提取）
    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: MONTHLY_PNL")
    lines.append("=" * 70)

    monthly = {}
    for t in report.get("all_trades", []):
        if t["exit_time"]:
            month = t["exit_time"][:7]  # "2025-01"
            monthly[month] = monthly.get(month, 0) + t["pnl"]

    for month in sorted(monthly.keys()):
        pnl = monthly[month]
        icon = "+" if pnl > 0 else ""
        lines.append(f"  {month}: {icon}{pnl:.2f} USDT")

    # 策略配置
    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: STRATEGY_CONFIG")
    lines.append("=" * 70)
    lines.append("  [Trend Following - 4h]")
    lines.append("    fast_ema: 20")
    lines.append("    slow_ema: 60")
    lines.append("    atr_period: 14")
    lines.append("    atr_filter_period: 60")
    lines.append("    trailing_stop_mult: 2.5x ATR")
    lines.append("    risk_per_trade: 2%")
    lines.append("    max_exposure: 60%")
    lines.append("")
    lines.append("  [Grid Trading - 1h]")
    lines.append("    grid_levels: 5 (each side)")
    lines.append("    grid_spacing: 0.5x ATR")
    lines.append("    qty_per_grid: 4% of capital")
    lines.append("    stop_loss: 5%")
    lines.append("    adx_threshold: 25 (only range-bound)")
    lines.append("    max_exposure: 40%")

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


# =================================================================
# CLI
# =================================================================

def main():
    parser = argparse.ArgumentParser(description="量化回测引擎 — 趋势跟踪 + 网格交易")
    parser.add_argument("--db", type=str, default="data/quant.db", help="数据库路径")
    parser.add_argument("--capital", type=float, default=10000.0, help="初始资金 (USDT)")
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--snapshot", type=str, default="backtest_snapshot.txt",
                        help="快照输出路径")
    args = parser.parse_args()

    # 检查数据库
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ 数据库不存在: {db_path}")
        print("   请先运行: python main.py --sync-data")
        sys.exit(1)

    # 运行回测
    engine = BacktestEngine(
        db_path=str(db_path),
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end)

    report = engine.run()

    if report:
        write_snapshot(report, args.snapshot, str(db_path), args.start, args.end)


if __name__ == "__main__":
    main()

"""
统一规则策略回测入口

本版修复：
1. 主时间框架默认切到 4h，并支持 1d 高级别过滤
2. 可切策略：triple_ema / macd_momentum / regime
3. 回测与实盘共用同一策略注册表
4. 修复权益统计与分笔记账逻辑，避免高频噪声误导分析
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from alpha.strategy_registry import available_strategies, build_strategy
from data.features import FeatureEngine
from utils.logger import get_logger

logger = get_logger("backtest")


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def ts_str(ms_or_val) -> str:
    try:
        if isinstance(ms_or_val, (int, float, np.integer, np.floating)):
            return datetime.utcfromtimestamp(int(ms_or_val) / 1000).strftime("%Y-%m-%d %H:%M")
        return str(ms_or_val)
    except Exception:
        return str(ms_or_val)


class BacktestEngine:
    def __init__(
        self,
        db_path: str,
        strategy_name: str = "triple_ema",
        initial_capital: float = 10000.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[list[str]] = None,
        config=None,
    ):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.feature_engine = FeatureEngine()
        self.strategy = build_strategy(config=config, explicit_name=strategy_name)

        self.all_trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.symbol_equity_curves: dict[str, list[float]] = {}
        self.run_meta = {
            "strategy": self.strategy.name,
            "primary_interval": self.strategy.primary_interval,
            "higher_interval": self.strategy.higher_interval,
        }

    def _date_to_ms(self, date_str: str) -> int:
        return int(
            datetime.strptime(date_str, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )

    def _load_klines(self, symbol: str, interval: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        conditions = ["symbol=?", "interval=?"]
        params: list = [symbol, interval]

        if self.start_date:
            conditions.append("open_time >= ?")
            params.append(self._date_to_ms(self.start_date))
        if self.end_date:
            conditions.append("open_time <= ?")
            params.append(self._date_to_ms(self.end_date))

        query = f"""
            SELECT *
            FROM klines
            WHERE {' AND '.join(conditions)}
            ORDER BY open_time ASC
        """
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            logger.warning(f"{symbol}/{interval} 无数据")
            return df

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(
            f"加载 {symbol}/{interval}: {len(df)} 条 | "
            f"{ts_str(df['open_time'].iloc[0])} ~ {ts_str(df['open_time'].iloc[-1])}"
        )
        return df.dropna(subset=["open", "high", "low", "close", "volume"])

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.feature_engine.compute_all(df)
        if features.empty:
            return pd.DataFrame()

        for col in ["open_time", "open", "high", "low", "close", "volume"]:
            if col in df.columns:
                features[col] = df[col].values[-len(features):]

        features = features.replace([np.inf, -np.inf], np.nan).copy()
        return features

    def _prepare_symbol_frames(self, symbol: str) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        primary_df = self._load_klines(symbol, self.strategy.primary_interval)
        if primary_df.empty:
            return primary_df, None, pd.DataFrame()

        higher_df = None
        if getattr(self.strategy, "higher_interval", None):
            higher_df = self._load_klines(symbol, self.strategy.higher_interval)

        primary_feat = self._compute_features(primary_df)
        higher_feat = self._compute_features(higher_df) if higher_df is not None and not higher_df.empty else None
        prepared = self.strategy.prepare_features(primary_feat, higher_feat)
        prepared = prepared.dropna(subset=["close"]).reset_index(drop=True)
        return primary_df, higher_df, prepared

    def _backtest_symbol(self, symbol: str, prepared: pd.DataFrame, capital: float) -> tuple[float, list[dict], list[float]]:
        if prepared.empty or len(prepared) < 120:
            logger.warning(f"{symbol}: 特征不足，跳过")
            return capital, [], [capital]

        cash = capital
        position: Optional[dict] = None
        state: dict = {"bar_index": -1, "cooldown_until_bar": -1}
        trades: list[dict] = []
        equity_curve: list[float] = []

        for idx in range(1, len(prepared)):
            row = prepared.iloc[idx]
            prev_row = prepared.iloc[idx - 1]
            close = float(row["close"])
            high = float(row.get("high", close) or close)
            low = float(row.get("low", close) or close)
            current_ts = ts_str(row.get("open_time", 0))

            state["bar_index"] = idx

            if position is not None:
                position["highest_since_entry"] = max(position["highest_since_entry"], high)
                position["lowest_since_entry"] = min(position["lowest_since_entry"], low)
                position["bar_index"] = idx

                should_exit, reason = self.strategy.check_exit(
                    row=row,
                    prev_row=prev_row,
                    position=position,
                    bar_count=idx - position["entry_bar"],
                )
                if should_exit:
                    exit_price = close * (1 - getattr(self.strategy.cfg, "slippage_pct", 0.0))
                    commission_pct = getattr(self.strategy.cfg, "commission_pct", 0.0)
                    qty = position["qty"]
                    gross_proceeds = qty * exit_price
                    exit_commission = gross_proceeds * commission_pct
                    cash += gross_proceeds - exit_commission

                    total_commission = position["entry_commission"] + exit_commission
                    pnl = (gross_proceeds - exit_commission) - position["entry_cost"]
                    pnl_pct = exit_price / position["entry_price"] - 1

                    trade = {
                        "symbol": symbol,
                        "strategy": self.strategy.name,
                        "side": "LONG",
                        "entry_price": round(position["entry_price"], 4),
                        "exit_price": round(exit_price, 4),
                        "quantity": round(qty, 8),
                        "notional": round(position["entry_price"] * qty, 2),
                        "pnl": round(pnl, 2),
                        "pnl_pct": float(pnl_pct),
                        "entry_time": position["entry_time"],
                        "exit_time": current_ts,
                        "holding_bars": idx - position["entry_bar"],
                        "exit_reason": reason,
                    }
                    for k in ["regime", "higher_regime", "daily_trend_ok"]:
                        if k in row.index:
                            trade[k] = row.get(k)
                    trades.append(trade)
                    self.strategy.on_trade_closed(state, idx, reason)
                    position = None

            if position is None and self.strategy.should_enter(row=row, prev_row=prev_row, state=state):
                entry_price = close * (1 + getattr(self.strategy.cfg, "slippage_pct", 0.0))
                qty, stop_loss = self.strategy.calc_position(cash, entry_price, row)
                if qty > 0:
                    commission_pct = getattr(self.strategy.cfg, "commission_pct", 0.0)
                    gross_cost = qty * entry_price
                    entry_commission = gross_cost * commission_pct
                    total_cost = gross_cost + entry_commission
                    if total_cost <= cash:
                        cash -= total_cost
                        position = {
                            "qty": qty,
                            "entry_price": entry_price,
                            "entry_time": current_ts,
                            "entry_bar": idx,
                            "entry_cost": total_cost,
                            "entry_commission": entry_commission,
                            "stop_loss": stop_loss,
                            "highest_since_entry": high,
                            "lowest_since_entry": low,
                            "atr_at_entry": float(row.get("natr_20", 0) or 0) * entry_price,
                        }

            equity = cash
            if position is not None:
                equity += position["qty"] * close
            equity_curve.append(equity)

        if position is not None:
            last_row = prepared.iloc[-1]
            last_close = float(last_row["close"])
            exit_price = last_close * (1 - getattr(self.strategy.cfg, "slippage_pct", 0.0))
            commission_pct = getattr(self.strategy.cfg, "commission_pct", 0.0)
            qty = position["qty"]
            gross_proceeds = qty * exit_price
            exit_commission = gross_proceeds * commission_pct
            cash += gross_proceeds - exit_commission
            pnl = (gross_proceeds - exit_commission) - position["entry_cost"]

            trades.append(
                {
                    "symbol": symbol,
                    "strategy": self.strategy.name,
                    "side": "LONG",
                    "entry_price": round(position["entry_price"], 4),
                    "exit_price": round(exit_price, 4),
                    "quantity": round(qty, 8),
                    "notional": round(position["entry_price"] * qty, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": float(exit_price / position["entry_price"] - 1),
                    "entry_time": position["entry_time"],
                    "exit_time": ts_str(last_row.get("open_time", 0)),
                    "holding_bars": len(prepared) - 1 - position["entry_bar"],
                    "exit_reason": "backtest_end",
                }
            )
            position = None

        return cash, trades, equity_curve or [capital]

    def _calc_buy_hold(self, features_dict: dict[str, pd.DataFrame]) -> float:
        rets = []
        for _, feat in features_dict.items():
            if feat is None or len(feat) < 2:
                continue
            first_close = float(feat["close"].iloc[0])
            last_close = float(feat["close"].iloc[-1])
            if first_close > 0:
                rets.append(last_close / first_close - 1)
        return float(np.mean(rets)) if rets else 0.0

    def _generate_report(self, final_equity: float, buy_hold_ret: float, elapsed: float) -> dict:
        pnl_list = [t["pnl"] for t in self.all_trades]
        wins = [x for x in pnl_list if x > 0]
        losses = [x for x in pnl_list if x < 0]

        eq = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_capital], dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.clip(peak, 1e-12, None)
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        if len(eq) > 2:
            rets = np.diff(eq) / np.clip(eq[:-1], 1e-12, None)
            rets = rets[np.isfinite(rets)]
            bars_per_year = 365.25 * 6  # 4h
            sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year)) if len(rets) > 1 and rets.std() > 0 else 0.0
        else:
            sharpe = 0.0

        exit_reasons: dict[str, dict] = {}
        symbol_stats: dict[str, dict] = {}
        monthly_pnl: dict[str, float] = {}
        regime_stats: dict[str, dict] = {}

        for trade in self.all_trades:
            reason = trade.get("exit_reason", "unknown")
            exit_reasons.setdefault(reason, {"count": 0, "pnl": 0.0})
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["pnl"] += trade["pnl"]

            sym = trade["symbol"]
            symbol_stats.setdefault(sym, {"trades": 0, "wins": 0, "pnl": 0.0})
            symbol_stats[sym]["trades"] += 1
            symbol_stats[sym]["pnl"] += trade["pnl"]
            if trade["pnl"] > 0:
                symbol_stats[sym]["wins"] += 1

            month = trade["exit_time"][:7]
            monthly_pnl[month] = monthly_pnl.get(month, 0.0) + trade["pnl"]

            regime_key = trade.get("regime", "N/A")
            regime_stats.setdefault(regime_key, {"trades": 0, "wins": 0, "pnl": 0.0})
            regime_stats[regime_key]["trades"] += 1
            regime_stats[regime_key]["pnl"] += trade["pnl"]
            if trade["pnl"] > 0:
                regime_stats[regime_key]["wins"] += 1

        total_return = (final_equity - self.initial_capital) / self.initial_capital
        report = {
            "summary": {
                "strategy": self.strategy.name,
                "primary_interval": self.strategy.primary_interval,
                "higher_interval": self.strategy.higher_interval,
                "initial_capital": round(self.initial_capital, 2),
                "final_equity": round(final_equity, 2),
                "total_return_pct": round(total_return * 100, 2),
                "total_pnl": round(final_equity - self.initial_capital, 2),
                "buy_hold_return_pct": round(buy_hold_ret * 100, 2),
                "vs_buy_hold_pct": round((total_return - buy_hold_ret) * 100, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "elapsed_seconds": round(elapsed, 2),
            },
            "trades_overview": {
                "total_trades": len(self.all_trades),
                "win_rate_pct": round((len(wins) / len(pnl_list) * 100), 1) if pnl_list else 0.0,
                "avg_win": round(float(np.mean(wins)), 2) if wins else 0.0,
                "avg_loss": round(float(np.mean(losses)), 2) if losses else 0.0,
                "profit_factor": round(sum(wins) / abs(sum(losses)), 3) if losses else (999.0 if wins else 0.0),
                "expectancy": round(float(np.mean(pnl_list)), 2) if pnl_list else 0.0,
                "best_trade": round(float(max(pnl_list)), 2) if pnl_list else 0.0,
                "worst_trade": round(float(min(pnl_list)), 2) if pnl_list else 0.0,
                "avg_holding_bars": round(float(np.mean([t["holding_bars"] for t in self.all_trades])), 1) if self.all_trades else 0.0,
            },
            "symbol_stats": {
                s: {
                    "trades": d["trades"],
                    "win_rate_pct": round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0.0,
                    "total_pnl": round(d["pnl"], 2),
                }
                for s, d in symbol_stats.items()
            },
            "regime_stats": {
                r: {
                    "trades": d["trades"],
                    "win_rate_pct": round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0.0,
                    "total_pnl": round(d["pnl"], 2),
                }
                for r, d in regime_stats.items()
            },
            "exit_reasons": {
                r: {"count": d["count"], "pnl": round(d["pnl"], 2)}
                for r, d in exit_reasons.items()
            },
            "monthly_pnl": {m: round(v, 2) for m, v in sorted(monthly_pnl.items())},
            "strategy_config": asdict(self.strategy.cfg),
            "all_trades": self.all_trades,
        }
        return report

    def _print_report(self, report: dict):
        s = report["summary"]
        t = report["trades_overview"]

        print()
        print("=" * 68)
        print(f"回测报告 | strategy={s['strategy']} | {s['primary_interval']} + {s['higher_interval']}")
        print("=" * 68)
        print(f"最终权益:      {s['final_equity']:>10.2f} USDT")
        print(f"策略收益:      {s['total_pnl']:>+10.2f} USDT ({s['total_return_pct']:+.2f}%)")
        print(f"Buy&Hold:      {s['buy_hold_return_pct']:>+10.2f}%")
        print(f"相对超额:      {s['vs_buy_hold_pct']:>+10.2f}%")
        print(f"最大回撤:      {s['max_drawdown_pct']:>10.2f}%")
        print(f"Sharpe:        {s['sharpe_ratio']:>10.3f}")
        print("-" * 68)
        print(f"交易数:        {t['total_trades']:>10d}")
        print(f"胜率:          {t['win_rate_pct']:>9.1f}%")
        print(f"平均盈利:      {t['avg_win']:>+10.2f}")
        print(f"平均亏损:      {t['avg_loss']:>+10.2f}")
        print(f"ProfitFactor:  {t['profit_factor']:>10.3f}")
        print(f"Expectancy:    {t['expectancy']:>+10.2f}")
        print(f"平均持仓:      {t['avg_holding_bars']:>10.1f} bars ({s['primary_interval']})")
        print("-" * 68)
        for sym, d in report["symbol_stats"].items():
            print(f"{sym:<10} | trades={d['trades']:>3d} | win={d['win_rate_pct']:>5.1f}% | pnl={d['total_pnl']:>+10.2f}")
        print("-" * 68)
        for reason, d in sorted(report["exit_reasons"].items(), key=lambda x: x[1]["count"], reverse=True):
            print(f"{reason:<18} | count={d['count']:>3d} | pnl={d['pnl']:>+10.2f}")
        print("=" * 68)

    def run(self) -> dict:
        print("=" * 68)
        print(f"规则策略回测 | strategy={self.strategy.name}")
        print("=" * 68)
        print(f"本金: {self.initial_capital:.2f} USDT")
        print(f"标的: {', '.join(self.symbols)}")
        print(f"日期: {self.start_date or '全部'} ~ {self.end_date or '全部'}")
        print(f"主周期: {self.strategy.primary_interval} | 高级别过滤: {self.strategy.higher_interval}")
        print()

        t0 = time.time()

        prepared_map: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            _, _, prepared = self._prepare_symbol_frames(symbol)
            if not prepared.empty:
                prepared_map[symbol] = prepared

        if not prepared_map:
            print("❌ 无可用特征数据")
            return {}

        per_symbol_capital = self.initial_capital / len(prepared_map)
        final_equity = 0.0
        global_curve: list[float] = []

        for symbol, prepared in prepared_map.items():
            print(f"▶ 回测 {symbol}: {len(prepared)} bars")
            cash, trades, curve = self._backtest_symbol(symbol, prepared, per_symbol_capital)
            final_equity += cash
            self.all_trades.extend(trades)
            self.symbol_equity_curves[symbol] = curve

        max_len = max(len(v) for v in self.symbol_equity_curves.values())
        for i in range(max_len):
            total = 0.0
            for curve in self.symbol_equity_curves.values():
                total += curve[i] if i < len(curve) else curve[-1]
            global_curve.append(total)
        self.equity_curve = global_curve

        buy_hold_ret = self._calc_buy_hold(prepared_map)
        report = self._generate_report(final_equity, buy_hold_ret, time.time() - t0)
        self._print_report(report)
        return report


def write_snapshot(report: dict, output_path: str, db_path: str, start_date: str, end_date: str):
    lines = []
    lines.append("=" * 70)
    lines.append("BACKTEST SNAPSHOT (Unified Strategy)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Database: {db_path}")
    lines.append(f"Period: {start_date or 'all'} ~ {end_date or 'all'}")
    lines.append("=" * 70)

    for section_name in [
        "summary",
        "trades_overview",
        "symbol_stats",
        "regime_stats",
        "exit_reasons",
        "strategy_config",
    ]:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"SECTION: {section_name.upper()}")
        lines.append("=" * 70)
        section = report.get(section_name, {})
        for k, v in section.items():
            if isinstance(v, dict):
                lines.append(f"  [{k}]")
                for kk, vv in v.items():
                    lines.append(f"    {kk}: {vv}")
            else:
                lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: MONTHLY_PNL")
    lines.append("=" * 70)
    for month, pnl in report.get("monthly_pnl", {}).items():
        prefix = "+" if pnl > 0 else ""
        lines.append(f"  {month}: {prefix}{pnl:.2f} USDT")

    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: ALL_TRADES")
    lines.append("=" * 70)
    lines.append(
        f"{'#':>4} | {'Symbol':<10} | {'Strategy':<14} | {'Entry':>10} | {'Exit':>10} | "
        f"{'PnL':>10} | {'PnL%':>8} | {'Bars':>5} | {'Reason':<18} | {'Entry Time':<16} | {'Exit Time':<16}"
    )
    lines.append("-" * 160)
    for idx, trade in enumerate(report.get("all_trades", []), 1):
        lines.append(
            f"{idx:>4} | {trade['symbol']:<10} | {trade['strategy']:<14} | "
            f"{trade['entry_price']:>10.2f} | {trade['exit_price']:>10.2f} | "
            f"{trade['pnl']:>+10.2f} | {trade['pnl_pct']:>+7.2%} | "
            f"{trade['holding_bars']:>5} | {trade['exit_reason']:<18} | "
            f"{trade['entry_time']:<16} | {trade['exit_time']:<16}"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SNAPSHOT")
    lines.append("=" * 70)

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 快照已保存: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一规则策略回测入口")
    parser.add_argument("--db", type=str, default="data/quant.db", help="数据库路径")
    parser.add_argument("--capital", type=float, default=10000.0, help="初始资金")
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--strategy", type=str, default="triple_ema", choices=available_strategies(), help="策略名称")
    parser.add_argument("--snapshot", type=str, default=None, help="输出快照路径")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ 数据库不存在: {db_path}")
        print("   请先运行: python main.py --sync-data")
        sys.exit(1)

    engine = BacktestEngine(
        db_path=str(db_path),
        strategy_name=args.strategy,
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
    )
    report = engine.run()
    if report:
        snapshot_path = args.snapshot or f"backtest_{args.strategy}_snapshot.txt"
        write_snapshot(report, snapshot_path, str(db_path), args.start, args.end)


if __name__ == "__main__":
    main()

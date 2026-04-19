"""
Funding Harvester 回测引擎 — v1.0 (P2A-T03)

按 8 小时 funding bar 推进, 模拟 delta-neutral long-spot + short-perp 的完整生命周期:
  - 每 8h 读取最新 funding_rate + spot_mark_price
  - 根据 FundingHarvester 判断 should_enter / should_exit
  - 成本扣除: 每次开平双边手续费 + 滑点
  - 每 8h 累加 funding_rate * notional 到 PnL
  - 输出 equity curve + 每笔 trade + summary JSON

输入
----
  --db data/quant.db             (含 funding_rates 表 + klines 表 [interval=8h 或 1d])
  --start 2023-01-01             UTC 起始
  --end   now                    UTC 结束
  --symbols BTCUSDT,ETHUSDT,...  默认读 funding_rates 中的所有 symbol
  --output analysis/output/funding_harvester_result.json

输出格式与 cross_sectional_backtest.py 保持一致:
  {
    "config": {...},
    "summary": {
        "total_return_pct": float,
        "annualized_return_pct": float,
        "sharpe_ratio": float,
        "max_drawdown_pct": float,
        "win_rate_pct": float,
        "n_trades": int,
        "verdict": "EXCELLENT" | "GOOD" | "MARGINAL" | "WEAK" | "FAILED"
    },
    "trades": [...],
    "equity_curve": [{"ts": ms, "equity": float}, ...]
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 把项目根目录加入 sys.path, 让本文件既能作脚本也能作模块
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from alpha.funding_harvester import FundingHarvester, FundingHarvesterConfig
from data.storage import Storage
from utils.logger import get_logger

logger = get_logger("funding_backtest")


# ======================================================================
# 核心引擎
# ======================================================================
def run_funding_backtest(
    storage: Storage,
    config: FundingHarvesterConfig,
    symbols: list[str],
    start_ms: int,
    end_ms: int,
    initial_capital: float = 100_000.0,
    funding_history_len: int = 30,
) -> dict:
    """
    主循环: 按 funding_time 时间序列推进。

    Args:
        storage: Storage 实例
        config: FundingHarvesterConfig
        symbols: 要交易的 symbol 列表
        start_ms, end_ms: UTC 毫秒时间戳区间
        initial_capital: 初始资金 (USD)
        funding_history_len: 每个 symbol 保留的历史 funding 数量 (用于 should_enter 持续性判断)

    Returns:
        dict (与 cross_sectional_backtest.py 格式一致)
    """
    strat = FundingHarvester(config)

    # 拉取所有 symbol 的 funding_rates, 合并为一个大 DataFrame
    dfs = []
    for s in symbols:
        df = storage.get_funding_rates(s, start_ms=start_ms, end_ms=end_ms)
        if df.empty:
            logger.warning(f"{s}: 无 funding 数据, 跳过")
            continue
        df["symbol"] = s
        dfs.append(df)
    if not dfs:
        raise RuntimeError("所有 symbol 均无 funding 数据")
    all_funding = pd.concat(dfs, ignore_index=True).sort_values("funding_time")

    # 按 symbol 分组, 预存 numpy array 以加速
    by_sym: dict[str, pd.DataFrame] = {
        s: g.reset_index(drop=True)
        for s, g in all_funding.groupby("symbol")
    }

    # 每个 symbol 的最近若干 funding (deque)
    recent: dict[str, deque] = {s: deque(maxlen=funding_history_len) for s in by_sym}

    # 当前开仓 {symbol: position_dict}
    positions: dict[str, dict] = {}

    # 事件流: 把所有 symbol 的 funding_time 合并成一个时间序列
    events = all_funding[["funding_time", "symbol", "funding_rate", "mark_price"]].values

    capital = float(initial_capital)
    equity_curve: list[dict] = []
    trades: list[dict] = []

    # 为了 "每 8h 结算一次 funding", 我们直接以 event loop 驱动
    for ts, sym, rate, mark in events:
        ts = int(ts)
        rate = float(rate)
        mark = float(mark) if (mark is not None and np.isfinite(mark)) else np.nan

        # 1) 累积最近 funding
        recent[sym].append(rate)

        # 2) 若有持仓, 结算本轮 funding + 检查 exit
        if sym in positions:
            pos = positions[sym]
            if np.isfinite(mark) and mark > 0:
                funding_income = rate * pos["notional_usd"]
                pos["funding_accumulated"] += funding_income
                capital += funding_income

            exit_now, reason = strat.should_exit(rate, pos)
            if exit_now:
                # 平仓: 再次扣除双边成本 (开仓时已扣一次 open 端)
                close_cost = _one_side_cost(config, pos["notional_usd"])
                capital -= close_cost
                pos["realized_cost"] += close_cost

                # 记录 trade
                pnl = pos["funding_accumulated"] - pos["realized_cost"]
                trades.append({
                    "symbol": sym,
                    "open_ts": pos["open_ts"],
                    "close_ts": ts,
                    "duration_h": (ts - pos["open_ts"]) / 3_600_000.0,
                    "notional_usd": pos["notional_usd"],
                    "funding_accumulated": pos["funding_accumulated"],
                    "realized_cost": pos["realized_cost"],
                    "pnl": pnl,
                    "exit_reason": reason,
                    "entry_mark": pos["entry_mark"],
                    "exit_mark": mark if np.isfinite(mark) else None,
                })
                del positions[sym]

        # 3) 否则尝试开仓
        else:
            snap = pd.Series({
                "symbol": sym,
                "funding_rate": rate,
                "recent_funding": list(recent[sym]),
            })
            if strat.should_enter(snap, {"open_positions": len(positions)}):
                if not (np.isfinite(mark) and mark > 0):
                    # mark 缺失, 保守跳过
                    pass
                else:
                    notional = min(config.notional_per_trade_usd, capital)
                    if notional > 0:
                        open_cost = _one_side_cost(config, notional)
                        capital -= open_cost
                        spot_qty, perp_qty = strat.calc_position(
                            capital=capital,
                            spot_price=mark,
                            perp_price=mark,
                        )
                        positions[sym] = {
                            "symbol": sym,
                            "open_ts": ts,
                            "entry_mark": mark,
                            "spot_qty": spot_qty,
                            "perp_qty": perp_qty,
                            "entry_spot_price": mark,
                            "entry_perp_price": mark,
                            "notional_usd": notional,
                            "funding_accumulated": 0.0,
                            "realized_cost": open_cost,
                        }

        # 4) 记录权益曲线 (每个 funding 事件记一次)
        equity_curve.append({
            "ts": ts,
            "equity": float(capital + _sum_open_notional(positions)),
        })

    # 5) 强制在 end_ms 平仓剩余头寸 (按最后一次 mark)
    for sym, pos in list(positions.items()):
        close_cost = _one_side_cost(config, pos["notional_usd"])
        capital -= close_cost
        pos["realized_cost"] += close_cost
        pnl = pos["funding_accumulated"] - pos["realized_cost"]
        trades.append({
            "symbol": sym,
            "open_ts": pos["open_ts"],
            "close_ts": end_ms,
            "duration_h": (end_ms - pos["open_ts"]) / 3_600_000.0,
            "notional_usd": pos["notional_usd"],
            "funding_accumulated": pos["funding_accumulated"],
            "realized_cost": pos["realized_cost"],
            "pnl": pnl,
            "exit_reason": "forced_close_end_of_backtest",
            "entry_mark": pos["entry_mark"],
            "exit_mark": None,
        })
        del positions[sym]

    # 最终 equity
    equity_curve.append({"ts": end_ms, "equity": capital})

    summary = _compute_summary(
        initial_capital=initial_capital,
        final_equity=capital,
        start_ms=start_ms,
        end_ms=end_ms,
        trades=trades,
        equity_curve=equity_curve,
    )

    diagnostics = _compute_diagnostics(trades, initial_capital)

    return {
        "config": asdict(config),
        "symbols": symbols,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "initial_capital": initial_capital,
        "summary": summary,
        "diagnostics": diagnostics,
        "trades": trades,
        "equity_curve": equity_curve,
    }


# ======================================================================
# Diagnostics (P2A-T03 rework 2026-04-19: 根据实盘 Sharpe≈0 诊断需要)
# ======================================================================
def _compute_diagnostics(trades: list[dict], initial_capital: float) -> dict:
    """
    per-symbol breakdown + cost/income 分解 + 最盈/最亏 trade 样本,
    用于定位 Funding Harvester 在真实数据上 Sharpe≈0 的根因:
      - 是 funding 收入本身就少?
      - 还是成本吃光了收入?
      - 还是个别 symbol 拉低整体?
    """
    if not trades:
        return {
            "per_symbol": {},
            "cost_breakdown": {
                "total_funding_income": 0.0,
                "total_trading_cost": 0.0,
                "net_pnl": 0.0,
                "cost_to_income_ratio": float("nan"),
            },
            "top_winners": [],
            "top_losers": [],
            "diagnostic_hint": "no_trades",
        }

    # Per-symbol
    per_sym: dict = {}
    for t in trades:
        s = t["symbol"]
        if s not in per_sym:
            per_sym[s] = {
                "n_trades": 0,
                "total_funding_income": 0.0,
                "total_cost": 0.0,
                "total_pnl": 0.0,
                "wins": 0,
                "avg_duration_h": 0.0,
                "_duration_sum": 0.0,
            }
        per_sym[s]["n_trades"] += 1
        per_sym[s]["total_funding_income"] += t["funding_accumulated"]
        per_sym[s]["total_cost"] += t["realized_cost"]
        per_sym[s]["total_pnl"] += t["pnl"]
        if t["pnl"] > 0:
            per_sym[s]["wins"] += 1
        per_sym[s]["_duration_sum"] += t.get("duration_h", 0.0)

    for s, d in per_sym.items():
        d["avg_duration_h"] = round(d["_duration_sum"] / max(d["n_trades"], 1), 2)
        d["win_rate_pct"] = round(d["wins"] / max(d["n_trades"], 1) * 100, 2)
        d["total_funding_income"] = round(d["total_funding_income"], 4)
        d["total_cost"] = round(d["total_cost"], 4)
        d["total_pnl"] = round(d["total_pnl"], 4)
        del d["_duration_sum"]

    # Cost breakdown
    total_income = sum(t["funding_accumulated"] for t in trades)
    total_cost = sum(t["realized_cost"] for t in trades)
    net = sum(t["pnl"] for t in trades)
    ratio = (total_cost / total_income) if total_income > 1e-9 else float("nan")

    # 诊断提示
    if total_income <= 0:
        hint = "funding_income_negative_or_zero: 累计 funding 收入非正, 策略没有抓到正 funding 机会"
    elif ratio > 1.0:
        hint = f"cost_eats_income: 成本 (${total_cost:.2f}) > 收入 (${total_income:.2f}), 降低换手或提高门槛"
    elif ratio > 0.8:
        hint = f"cost_heavy: 成本占收入 {ratio:.0%}, 仅剩少量利润"
    elif ratio > 0.5:
        hint = f"moderate_cost: 成本占收入 {ratio:.0%}, 可接受但有优化空间"
    else:
        hint = f"healthy: 成本占收入 {ratio:.0%}"

    # Top winners / losers
    sorted_by_pnl = sorted(trades, key=lambda t: t["pnl"], reverse=True)
    top_winners = sorted_by_pnl[:5]
    top_losers = sorted_by_pnl[-5:][::-1]  # 最差的 5 个

    return {
        "per_symbol": per_sym,
        "cost_breakdown": {
            "total_funding_income": round(total_income, 4),
            "total_trading_cost": round(total_cost, 4),
            "net_pnl": round(net, 4),
            "cost_to_income_ratio": round(ratio, 4) if np.isfinite(ratio) else None,
            "net_pnl_pct_of_capital": round(net / initial_capital * 100, 4),
        },
        "top_winners": [_trade_brief(t) for t in top_winners],
        "top_losers": [_trade_brief(t) for t in top_losers],
        "diagnostic_hint": hint,
    }


def _trade_brief(t: dict) -> dict:
    """Trade 的精简摘要, 用于 diagnostics 输出"""
    return {
        "symbol": t["symbol"],
        "open_ts": t["open_ts"],
        "close_ts": t["close_ts"],
        "duration_h": round(t.get("duration_h", 0.0), 2),
        "funding_accumulated": round(t["funding_accumulated"], 4),
        "realized_cost": round(t["realized_cost"], 4),
        "pnl": round(t["pnl"], 4),
        "exit_reason": t["exit_reason"],
    }


# ======================================================================
# Helpers
# ======================================================================
def _one_side_cost(cfg: FundingHarvesterConfig, notional_usd: float) -> float:
    """一次 open OR close 的成本 (USD). open+close = 2 * _one_side_cost."""
    bps = cfg.spot_fee_bps + cfg.perp_fee_bps + 2 * cfg.slippage_bps
    return notional_usd * bps / 10_000.0


def _sum_open_notional(positions: dict) -> float:
    """未平仓头寸的账面价值 (忽略标记价变化, 因 delta-neutral)"""
    return sum(p["notional_usd"] for p in positions.values())


def _compute_summary(
    initial_capital: float,
    final_equity: float,
    start_ms: int,
    end_ms: int,
    trades: list[dict],
    equity_curve: list[dict],
) -> dict:
    total_return_pct = (final_equity / initial_capital - 1.0) * 100
    days = max((end_ms - start_ms) / 86_400_000.0, 1.0)
    years = days / 365.25
    ann = (final_equity / initial_capital) ** (1.0 / max(years, 1e-6)) - 1.0
    annualized_return_pct = ann * 100

    # Sharpe 基于 equity 的 8h 收益率
    if len(equity_curve) >= 3:
        eq = pd.Series([e["equity"] for e in equity_curve])
        rets = eq.pct_change().dropna()
        if rets.std() > 0:
            # 一年 3 * 365 = 1095 个 8h bar
            sharpe = float(rets.mean() / rets.std() * np.sqrt(1095))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    if equity_curve:
        eq_arr = np.asarray([e["equity"] for e in equity_curve], dtype=float)
        running_max = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - running_max) / running_max
        max_dd_pct = float(dd.min() * 100) if len(dd) else 0.0
    else:
        max_dd_pct = 0.0

    # 胜率
    n_trades = len(trades)
    if n_trades > 0:
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / n_trades * 100
    else:
        win_rate = 0.0

    # 判决: 参考路线图 G1 + P2A-T03 Acceptance 标准
    if sharpe > 2.0 and max_dd_pct > -10:
        verdict = "EXCELLENT"
    elif sharpe > 1.5 and max_dd_pct > -15:
        verdict = "GOOD"
    elif sharpe > 0.8:
        verdict = "MARGINAL"
    elif sharpe > 0:
        verdict = "WEAK"
    else:
        verdict = "FAILED"

    return {
        "total_return_pct": round(float(total_return_pct), 4),
        "annualized_return_pct": round(float(annualized_return_pct), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown_pct": round(float(max_dd_pct), 4),
        "win_rate_pct": round(float(win_rate), 4),
        "n_trades": int(n_trades),
        "verdict": verdict,
    }


# ======================================================================
# CLI
# ======================================================================
def _parse_args():
    p = argparse.ArgumentParser(description="Funding Harvester 回测")
    p.add_argument("--db", default="data/quant.db")
    p.add_argument("--start", default="2023-01-01", help="UTC start (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="UTC end (YYYY-MM-DD, 默认 now)")
    p.add_argument("--symbols", default=None,
                   help="逗号分隔 symbol 列表, 默认读 DB 中所有 symbol")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--min-funding-rate", type=float, default=None,
                   help="覆盖配置 min_funding_rate")
    p.add_argument("--min-duration-h", type=int, default=None,
                   help="覆盖配置 min_funding_duration_h")
    p.add_argument("--max-positions", type=int, default=None,
                   help="覆盖配置 max_concurrent_positions")
    p.add_argument("--notional", type=float, default=None,
                   help="覆盖配置 notional_per_trade_usd")
    p.add_argument("--output", default="analysis/output/funding_harvester_result.json")
    return p.parse_args()


def _ts_ms(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").value // 1_000_000)


def main():
    args = _parse_args()
    storage = Storage(args.db)
    start_ms = _ts_ms(args.start)
    end_ms = _ts_ms(args.end) if args.end else int(pd.Timestamp.now("UTC").value // 1_000_000)

    # Symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        # 读所有 funding 表中的 symbol
        with storage._conn() as conn:
            cur = conn.execute(
                "SELECT DISTINCT symbol FROM funding_rates ORDER BY symbol"
            )
            symbols = [r[0] for r in cur.fetchall()]
        if not symbols:
            logger.error("funding_rates 表为空, 请先 python scripts/sync_funding_data.py")
            sys.exit(1)

    # Config (config 默认值 + CLI 覆盖)
    cfg = FundingHarvesterConfig()
    if args.min_funding_rate is not None:
        cfg.min_funding_rate = args.min_funding_rate
    if args.min_duration_h is not None:
        cfg.min_funding_duration_h = args.min_duration_h
    if args.max_positions is not None:
        cfg.max_concurrent_positions = args.max_positions
    if args.notional is not None:
        cfg.notional_per_trade_usd = args.notional
    cfg.validate()

    logger.info(f"回测 {len(symbols)} symbol, {args.start} → {args.end or 'now'}, "
                f"min_rate={cfg.min_funding_rate} duration_h={cfg.min_funding_duration_h}")

    result = run_funding_backtest(
        storage=storage,
        config=cfg,
        symbols=symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        initial_capital=args.capital,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    s = result["summary"]
    d = result.get("diagnostics", {})
    logger.info(
        f"完成: Sharpe={s['sharpe_ratio']} Ret={s['annualized_return_pct']}% "
        f"MDD={s['max_drawdown_pct']}% N={s['n_trades']} → {s['verdict']}"
    )
    if d:
        cb = d.get("cost_breakdown", {})
        logger.info(
            f"诊断: funding_income=${cb.get('total_funding_income', 0):.2f} "
            f"cost=${cb.get('total_trading_cost', 0):.2f} "
            f"net=${cb.get('net_pnl', 0):.2f} "
            f"ratio={cb.get('cost_to_income_ratio')}"
        )
        logger.info(f"诊断提示: {d.get('diagnostic_hint', 'n/a')}")
    logger.info(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()

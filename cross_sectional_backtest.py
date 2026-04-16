"""
Cross-Sectional Momentum Backtest — v1.0 MVP

路线 A 的验证入口。执行:
  python cross_sectional_backtest.py \
      --db data/quant.db \
      --start 2022-01-01 \
      --end 2026-04-01 \
      --top-n 50 \
      --output analysis/output/xs_mom_mvp.json

输出:
  - 权益曲线 (equity curve)
  - Sharpe ratio (年化)
  - Max drawdown
  - Monthly returns
  - Factor exposure 稳定性

判决标准:
  - Sharpe > 1.2  → 方向正确, 继续加因子做多因子聚合
  - Sharpe in (0.8, 1.2) → 可能有 edge 但边缘, 需严格成本分析
  - Sharpe < 0.8  → crypto momentum 衰减严重, 考虑其他方向
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.storage import Storage
from data.universe import UniverseBuilder, UniverseConfig
from alpha.cross_sectional_momentum import (
    CrossSectionalMomentumStrategy,
    CrossSectionalConfig,
)
from utils.logger import get_logger

logger = get_logger("xs_backtest")


def run_backtest(
    db_path: str,
    start_date: str,
    end_date: str,
    top_n: int = 50,
    initial_capital: float = 100_000.0,
    rebalance_days: int = 7,
    output_path: str = None,
) -> dict:
    """
    执行横截面 momentum 回测
    """
    storage = Storage(db_path)

    # Universe builder
    u_cfg = UniverseConfig(top_n=top_n, rebalance_freq_days=rebalance_days)
    ub = UniverseBuilder(storage, u_cfg)

    # Strategy
    strat_cfg = CrossSectionalConfig(
        top_n=top_n,
        rebalance_freq_days=rebalance_days,
    )
    strategy = CrossSectionalMomentumStrategy(strat_cfg)

    # 预加载所有 symbols 的日线到内存 (MVP 简化)
    all_symbols = ub.get_all_symbols()
    logger.info(f"加载 {len(all_symbols)} 个 symbols 日线数据...")

    symbol_data = {}
    for sym in all_symbols:
        try:
            kl = storage.get_klines(sym, "1d", limit=10000)
            if kl.empty or len(kl) < 60:
                continue
            kl["ts"] = pd.to_datetime(kl["open_time"], unit="ms")
            kl = kl.sort_values("ts").reset_index(drop=True)
            symbol_data[sym] = kl
        except Exception as e:
            logger.debug(f"skip {sym}: {e}")

    logger.info(f"有效 symbols: {len(symbol_data)}")

    if len(symbol_data) < 10:
        logger.error("有效 symbols 太少, 回测中止")
        return {"error": "insufficient_symbols", "n_symbols": len(symbol_data)}

    # 时间轴
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    current_ts = start_ts

    # 权益曲线
    equity = initial_capital
    equity_curve = []
    rebalance_log = []
    current_positions = {}   # {symbol: (qty, entry_price, weight)}

    # 主循环: 每 rebalance_days 天触发一次
    while current_ts <= end_ts:
        date_str = current_ts.strftime("%Y-%m-%d")

        # --- 1) 先 MtM 当前持仓 ---
        pnl_this_period = 0.0
        for sym, pos in list(current_positions.items()):
            kl = symbol_data.get(sym)
            if kl is None:
                continue
            mask = kl["ts"] <= current_ts
            if not mask.any():
                continue
            current_price = float(kl[mask].iloc[-1]["close"])
            if current_price <= 0:
                continue
            qty, entry_price, weight = pos
            pnl = qty * (current_price - entry_price)
            pnl_this_period += pnl

        equity += pnl_this_period

        # --- 2) 计算所有 symbol 的因子 ---
        universe = ub.get_universe(date_str)
        universe = [s for s in universe if s in symbol_data]

        factors_by_symbol = {}
        volatilities = {}

        for sym in universe:
            kl = symbol_data[sym]
            mask = kl["ts"] <= current_ts
            sub = kl[mask]
            if len(sub) < strat_cfg.momentum_window + 5:
                continue

            as_of_idx = len(sub) - 1
            factors = strategy.compute_factors(sub, as_of_idx)
            if factors is None:
                continue

            factors_by_symbol[sym] = factors
            # 存波动率 (annualized) 供 inverse vol weighting 用
            volatilities[sym] = factors["volatility_30d"] * np.sqrt(365)

        if len(factors_by_symbol) < 10:
            current_ts += timedelta(days=rebalance_days)
            equity_curve.append({
                "date": date_str,
                "equity": equity,
                "n_positions": len(current_positions),
                "skipped": True,
            })
            continue

        # --- 3) 排序 + 构建新组合 ---
        target_weights = strategy.rank_and_build_portfolio(factors_by_symbol)
        if strat_cfg.use_inverse_vol_weight and target_weights:
            target_weights = strategy.apply_inverse_vol_weighting(
                target_weights, volatilities
            )

        # --- 4) 平旧仓, 开新仓 (简化: 全部 refresh) ---
        # 成本计算 (turnover-based)
        cost_bps = strat_cfg.slippage_bps + strat_cfg.commission_bps
        gross_turnover = 0.0

        # 关闭旧仓
        for sym, pos in list(current_positions.items()):
            qty, entry_price, _ = pos
            kl = symbol_data.get(sym)
            if kl is None:
                continue
            mask = kl["ts"] <= current_ts
            if not mask.any():
                continue
            exit_price = float(kl[mask].iloc[-1]["close"])
            if exit_price > 0:
                gross_turnover += abs(qty) * exit_price

        current_positions = {}

        # 开新仓
        for sym, weight in target_weights.items():
            kl = symbol_data.get(sym)
            if kl is None:
                continue
            mask = kl["ts"] <= current_ts
            if not mask.any():
                continue
            entry_price = float(kl[mask].iloc[-1]["close"])
            if entry_price <= 0:
                continue
            # 仓位 = 权重 * equity / price (允许做空: 负权重)
            notional = weight * equity
            qty = notional / entry_price
            current_positions[sym] = (qty, entry_price, weight)
            gross_turnover += abs(qty) * entry_price

        # 扣成本
        cost = gross_turnover * (cost_bps / 10000.0)
        equity -= cost

        equity_curve.append({
            "date": date_str,
            "equity": round(equity, 2),
            "n_positions": len(current_positions),
            "gross_turnover": round(gross_turnover, 2),
            "cost": round(cost, 2),
            "pnl": round(pnl_this_period, 2),
        })

        rebalance_log.append({
            "date": date_str,
            "n_longs": sum(1 for w in target_weights.values() if w > 0),
            "n_shorts": sum(1 for w in target_weights.values() if w < 0),
            "top_long": sorted(
                [(s, w) for s, w in target_weights.items() if w > 0],
                key=lambda x: -x[1],
            )[:3],
            "top_short": sorted(
                [(s, w) for s, w in target_weights.items() if w < 0],
                key=lambda x: x[1],
            )[:3],
        })

        current_ts += timedelta(days=rebalance_days)

    # --- 绩效统计 ---
    if not equity_curve:
        return {"error": "no_backtest_output"}

    equity_series = pd.Series([p["equity"] for p in equity_curve])
    returns = equity_series.pct_change().dropna()

    total_return = equity_series.iloc[-1] / initial_capital - 1.0
    n_periods = len(returns)
    n_years = n_periods * rebalance_days / 365.0 if n_periods > 0 else 1.0

    annualized_return = (
        (1 + total_return) ** (1.0 / max(n_years, 0.1)) - 1.0
        if total_return > -1 else -1.0
    )

    # Annualized vol
    periods_per_year = 365 / rebalance_days
    annualized_vol = float(returns.std() * np.sqrt(periods_per_year)) if len(returns) > 0 else 0.0
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

    # Max drawdown
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    max_dd = float(dd.min())

    report = {
        "config": {
            "start": start_date,
            "end": end_date,
            "top_n": top_n,
            "rebalance_days": rebalance_days,
            "initial_capital": initial_capital,
            "slippage_bps": strat_cfg.slippage_bps,
            "commission_bps": strat_cfg.commission_bps,
        },
        "summary": {
            "final_equity": round(equity_series.iloc[-1], 2),
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(annualized_return * 100, 2),
            "annualized_vol_pct": round(annualized_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "n_rebalances": len(equity_curve),
            "n_symbols_universe": len(symbol_data),
        },
        "verdict": _make_verdict(sharpe, max_dd),
        "equity_curve": equity_curve[:500],   # 截断避免过大
        "sample_rebalances": rebalance_log[:20],
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"报告已保存: {output_path}")

    return report


def _make_verdict(sharpe: float, max_dd: float) -> dict:
    """给出决策判决"""
    if sharpe >= 1.5:
        verdict = "EXCELLENT"
        next_step = "方向正确且 edge 明显。下一步: 加 funding carry + on-chain 因子做 CTREND 风格聚合。"
    elif sharpe >= 1.2:
        verdict = "GOOD"
        next_step = "方向正确, edge 可验证。下一步: 多因子优化 + purged CPCV 严格验证。"
    elif sharpe >= 0.8:
        verdict = "MARGINAL"
        next_step = "有 edge 但边缘。下一步: 检查成本假设 / 优化 rebalance 频率 / 加流动性过滤。"
    elif sharpe >= 0.3:
        verdict = "WEAK"
        next_step = "momentum 衰减严重。下一步: 考虑加入 funding carry、反转因子, 或转换标的类别。"
    else:
        verdict = "FAILED"
        next_step = "在当前成本结构下 crypto 横截面 momentum 不 work。考虑 LOB 微观结构或其他资产类别。"

    return {
        "grade": verdict,
        "sharpe": sharpe,
        "max_dd_pct": round(max_dd * 100, 2),
        "next_step": next_step,
    }


def main():
    parser = argparse.ArgumentParser(description="路线 A: 横截面 momentum MVP 回测")
    parser.add_argument("--db", default="data/quant.db")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--rebalance-days", type=int, default=7)
    parser.add_argument("--output", default="analysis/output/xs_mom_mvp.json")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    print("=" * 60)
    print("  路线 A: 横截面 momentum MVP")
    print(f"  期间: {args.start} → {args.end}")
    print(f"  Top N: {args.top_n}, Rebalance: {args.rebalance_days} 天")
    print("=" * 60)

    report = run_backtest(
        db_path=args.db,
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        initial_capital=args.capital,
        rebalance_days=args.rebalance_days,
        output_path=args.output,
    )

    if "error" in report:
        print(f"\n❌ 回测失败: {report['error']}")
        sys.exit(1)

    s = report["summary"]
    v = report["verdict"]
    print(f"\n📊 结果:")
    print(f"  最终权益:       {s['final_equity']:,.2f} ({s['total_return_pct']:+.1f}%)")
    print(f"  年化收益:       {s['annualized_return_pct']:+.1f}%")
    print(f"  年化波动:       {s['annualized_vol_pct']:.1f}%")
    print(f"  Sharpe:         {s['sharpe_ratio']:.3f}")
    print(f"  最大回撤:       {s['max_drawdown_pct']:.1f}%")
    print(f"  Rebalance 次数:  {s['n_rebalances']}")
    print(f"\n🎯 判决: {v['grade']}")
    print(f"   {v['next_step']}")
    print(f"\n📄 详细报告: {args.output}")


if __name__ == "__main__":
    main()

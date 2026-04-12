"""
参数敏感性扫描器 — 检测过拟合

对关键参数做网格扫描，看最优点是孤立尖峰还是平坦高原。
孤立尖峰 = 过拟合，平坦高原 = 鲁棒。

用法:
  python analysis/param_sensitivity.py --db data/quant.db --strategy triple_ema --start 2025-01-01
"""
from __future__ import annotations

import argparse
import itertools
import sys
import json
from pathlib import Path
from copy import deepcopy

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_runner import BacktestEngine
from utils.logger import get_logger

logger = get_logger("param_sensitivity")


# ==============================================================
# 参数空间定义
# ==============================================================
PARAM_GRIDS = {
    "triple_ema": {
        "trail_atr_mult": [1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0],
        "min_adx": [12, 15, 18, 22, 26, 30],
        "stop_atr_mult": [1.5, 2.0, 2.5, 3.0],
        "cooldown_bars": [3, 4, 6, 8, 10],
        "risk_per_trade": [0.008, 0.010, 0.012, 0.015, 0.020],
        "max_holding_bars": [20, 30, 45, 60, 80],
    },
    "macd_momentum": {
        "trail_atr_mult": [1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0],
        "min_adx": [10, 13, 16, 20, 25, 30],
        "min_rsi": [40, 44, 48, 52],
        "max_rsi": [68, 72, 76, 80],
        "cooldown_bars": [3, 5, 7, 10],
        "risk_per_trade": [0.008, 0.010, 0.012, 0.015, 0.020],
        "max_holding_bars": [20, 28, 36, 48],
    },
}


def run_single_backtest(db_path: str, strategy_name: str,
                        start_date: str, capital: float,
                        overrides: dict) -> dict:
    """运行一次回测，返回核心指标"""
    try:
        engine = BacktestEngine(
            db_path=db_path,
            strategy_name=strategy_name,
            initial_capital=capital,
            start_date=start_date,
            config=overrides,
        )
        report = engine.run()
        if not report:
            return {"sharpe": -999, "pnl": 0, "trades": 0, "max_dd": 100}

        summary = report.get("summary", {})
        return {
            "sharpe": summary.get("sharpe_ratio", -999),
            "pnl": summary.get("total_pnl", 0),
            "return_pct": summary.get("total_return_pct", 0),
            "trades": report.get("trades_overview", {}).get("total_trades", 0),
            "max_dd": summary.get("max_drawdown_pct", 100),
            "win_rate": report.get("trades_overview", {}).get("win_rate_pct", 0),
            "profit_factor": report.get("trades_overview", {}).get("profit_factor", 0),
        }
    except Exception as e:
        logger.error(f"回测失败 ({overrides}): {e}")
        return {"sharpe": -999, "pnl": 0, "trades": 0, "max_dd": 100}


def scan_single_param(db_path: str, strategy_name: str,
                      start_date: str, capital: float,
                      param_name: str, values: list) -> list[dict]:
    """扫描单个参数"""
    results = []
    for v in values:
        overrides = {param_name: v}
        metrics = run_single_backtest(db_path, strategy_name, start_date, capital, overrides)
        results.append({"param": param_name, "value": v, **metrics})
        icon = "🟢" if metrics["pnl"] > 0 else "🔴"
        print(f"    {icon} {param_name}={v}: "
              f"PnL={metrics['pnl']:+.2f} | "
              f"Sharpe={metrics['sharpe']:.3f} | "
              f"DD={metrics['max_dd']:.2f}%")
    return results


def scan_2d(db_path: str, strategy_name: str,
            start_date: str, capital: float,
            param1: str, values1: list,
            param2: str, values2: list) -> list[dict]:
    """2D 网格扫描"""
    results = []
    total = len(values1) * len(values2)
    done = 0
    for v1, v2 in itertools.product(values1, values2):
        done += 1
        overrides = {param1: v1, param2: v2}
        metrics = run_single_backtest(db_path, strategy_name, start_date, capital, overrides)
        results.append({param1: v1, param2: v2, **metrics})
        if done % 10 == 0 or done == total:
            print(f"    进度: {done}/{total}")
    return results


def assess_robustness(results: list[dict], metric: str = "sharpe") -> dict:
    """
    评估参数鲁棒性:
    - 如果最优值附近的区域也不错 -> 鲁棒
    - 如果最优值是孤立尖峰 -> 过拟合风险
    """
    values = [r[metric] for r in results if r[metric] > -999]
    if not values:
        return {"verdict": "NO_DATA"}

    best = max(values)
    threshold = best * 0.7  # 70% of best
    good_count = sum(1 for v in values if v >= threshold)
    good_ratio = good_count / len(values)

    if good_ratio > 0.4:
        verdict = "ROBUST"
        detail = f"{good_ratio:.0%} 的参数组合达到最优值的 70%+，参数空间平坦"
    elif good_ratio > 0.2:
        verdict = "MODERATE"
        detail = f"{good_ratio:.0%} 的参数组合达到 70%+ 阈值，有一定鲁棒性"
    else:
        verdict = "FRAGILE"
        detail = f"仅 {good_ratio:.0%} 达到 70%+ 阈值，存在过拟合风险"

    return {
        "verdict": verdict,
        "detail": detail,
        "best_value": round(best, 4),
        "good_ratio": round(good_ratio, 4),
        "n_tested": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description="参数敏感性扫描")
    parser.add_argument("--db", type=str, default="data/quant.db")
    parser.add_argument("--strategy", type=str, default="triple_ema")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--mode", type=str, default="1d",
                        choices=["1d", "2d", "full"],
                        help="1d=逐参数扫, 2d=关键2D网格, full=全参数(慢)")
    parser.add_argument("--output-dir", type=str, default="analysis/output")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    grid = PARAM_GRIDS.get(args.strategy, {})
    if not grid:
        print(f"❌ 未知策略: {args.strategy}")
        return

    print("=" * 70)
    print(f"  🔍 参数敏感性扫描 — {args.strategy}")
    print("=" * 70)

    all_results = []
    robustness_report = {}

    if args.mode in ("1d", "full"):
        print("\n[1D 扫描] 逐参数变化，其余保持默认:")
        for param, values in grid.items():
            print(f"\n  📐 {param}: {values}")
            results = scan_single_param(
                args.db, args.strategy, args.start, args.capital,
                param, values
            )
            all_results.extend(results)
            robustness = assess_robustness(results)
            robustness_report[param] = robustness
            icon = {"ROBUST": "🟢", "MODERATE": "🟡", "FRAGILE": "🔴"}[robustness["verdict"]]
            print(f"    {icon} 鲁棒性: {robustness['verdict']} — {robustness['detail']}")

    if args.mode in ("2d", "full"):
        print("\n[2D 扫描] 关键参数对:")
        key_pairs = [
            ("trail_atr_mult", "stop_atr_mult"),
            ("trail_atr_mult", "min_adx"),
            ("risk_per_trade", "max_holding_bars"),
        ]
        for p1, p2 in key_pairs:
            if p1 in grid and p2 in grid:
                print(f"\n  📐 {p1} × {p2}:")
                results = scan_2d(
                    args.db, args.strategy, args.start, args.capital,
                    p1, grid[p1], p2, grid[p2]
                )
                all_results.extend(results)

                # 找最优组合
                best = max(results, key=lambda r: r.get("sharpe", -999))
                print(f"    最优: {p1}={best[p1]}, {p2}={best[p2]} -> "
                      f"Sharpe={best['sharpe']:.3f}, PnL={best['pnl']:+.2f}")

    # ---- 保存结果 ----
    output_path = f"{args.output_dir}/sensitivity_{args.strategy}.json"
    with open(output_path, "w") as f:
        json.dump({
            "strategy": args.strategy,
            "mode": args.mode,
            "robustness": robustness_report,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n📄 完整结果: {output_path}")

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("  📋 鲁棒性总结")
    print("=" * 70)
    fragile_params = []
    for param, report in robustness_report.items():
        icon = {"ROBUST": "🟢", "MODERATE": "🟡", "FRAGILE": "🔴"}.get(report["verdict"], "❓")
        print(f"  {icon} {param}: {report['verdict']}")
        if report["verdict"] == "FRAGILE":
            fragile_params.append(param)

    if fragile_params:
        print(f"\n  ⚠️ 以下参数存在过拟合风险: {', '.join(fragile_params)}")
        print("  建议: 使用更宽松的参数值，或通过 Walk-forward 验证")
    else:
        print("\n  ✅ 参数空间整体较为鲁棒")


if __name__ == "__main__":
    main()

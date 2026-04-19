"""
Funding Harvester 敏感度扫描 — v1.0 (P2A-T04)

扫描 FundingHarvesterConfig 的关键参数, 报告:
  - 每格的 Sharpe / 年化 / MDD / n_trades
  - 最优参数组合
  - 每个参数的 "good_ratio" (≥ 最优 70% 的格子占比)
  - 整体判决: ROBUST / MODERATE / FRAGILE

默认扫描网格 (64 格):
  min_funding_rate:       [5e-5, 1e-4, 1.5e-4, 2e-4]
  min_funding_duration_h: [8, 16, 24, 48]
  max_concurrent_positions: [5, 10, 15, 20]

用法:
  python scripts/run_funding_sensitivity.py --db data/quant.db \
     --start 2023-01-01 --output analysis/output/funding_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

# 确保能找到仓库根
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from alpha.funding_harvester import FundingHarvesterConfig
from data.storage import Storage
from funding_harvester_backtest import run_funding_backtest
from utils.logger import get_logger

logger = get_logger("funding_sensitivity")


DEFAULT_GRID = {
    "min_funding_rate": [5e-5, 1e-4, 1.5e-4, 2e-4],
    "min_funding_duration_h": [8, 16, 24, 48],
    "max_concurrent_positions": [5, 10, 15, 20],
}


def run_grid_scan(
    storage: Storage,
    symbols: list[str],
    start_ms: int,
    end_ms: int,
    grid: dict[str, list] | None = None,
    initial_capital: float = 100_000.0,
) -> list[dict]:
    """
    跑完整笛卡尔积扫描, 返回每格结果 (不含 trades / equity_curve, 仅 summary + params)。
    """
    grid = grid or DEFAULT_GRID
    keys = list(grid.keys())
    combos = list(product(*(grid[k] for k in keys)))
    logger.info(f"开始扫描 {len(combos)} 格参数 × {len(symbols)} symbols")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        cfg = FundingHarvesterConfig(**params)
        try:
            cfg.validate()
        except ValueError as e:
            logger.warning(f"[{i}/{len(combos)}] 跳过非法参数 {params}: {e}")
            continue
        try:
            out = run_funding_backtest(
                storage, cfg, symbols, start_ms, end_ms,
                initial_capital=initial_capital,
            )
            summary = out["summary"]
        except Exception as e:
            logger.warning(f"[{i}/{len(combos)}] 失败 {params}: {e}")
            summary = {
                "sharpe_ratio": float("nan"),
                "annualized_return_pct": float("nan"),
                "max_drawdown_pct": float("nan"),
                "n_trades": 0,
                "verdict": "ERROR",
            }
        results.append({
            "params": params,
            "summary": summary,
        })
        if i % 10 == 0 or i == len(combos):
            logger.info(f"[{i}/{len(combos)}] {params} → Sharpe={summary.get('sharpe_ratio')}")
    return results


def compute_good_ratios(
    results: list[dict], good_threshold: float = 0.7,
) -> dict:
    """
    每个参数的 good_ratio = {val: fraction_of_configs_with_this_val_achieving_>=best_sharpe*0.7}

    返回:
      {
        'best_sharpe': float,
        'best_params': dict,
        'overall_good_ratio': float,
        'per_param_good_ratio': {param: {val: ratio}},
        'verdict': 'ROBUST' | 'MODERATE' | 'FRAGILE',
      }
    """
    rows = [
        {**r["params"], "sharpe": r["summary"].get("sharpe_ratio", float("nan"))}
        for r in results
    ]
    if not rows:
        return {
            "best_sharpe": float("nan"),
            "best_params": {},
            "overall_good_ratio": 0.0,
            "per_param_good_ratio": {},
            "good_threshold": good_threshold,
            "verdict": "FRAGILE",
        }
    df = pd.DataFrame(rows)
    # 丢掉 NaN
    valid = df.dropna(subset=["sharpe"])
    if valid.empty:
        return {
            "best_sharpe": float("nan"),
            "best_params": {},
            "overall_good_ratio": 0.0,
            "per_param_good_ratio": {},
            "verdict": "FRAGILE",
        }
    best_sharpe = float(valid["sharpe"].max())
    cutoff = good_threshold * best_sharpe if best_sharpe > 0 else float("-inf")
    valid["is_good"] = valid["sharpe"] >= cutoff
    overall_good_ratio = float(valid["is_good"].mean())

    param_cols = [c for c in valid.columns if c not in {"sharpe", "is_good"}]
    per_param = {}
    for c in param_cols:
        grp = valid.groupby(c)["is_good"].mean().to_dict()
        per_param[c] = {str(k): float(v) for k, v in grp.items()}

    best_row = valid.loc[valid["sharpe"].idxmax()]
    best_params = {c: _py_val(best_row[c]) for c in param_cols}

    if overall_good_ratio > 0.5:
        verdict = "ROBUST"
    elif overall_good_ratio > 0.3:
        verdict = "MODERATE"
    else:
        verdict = "FRAGILE"

    return {
        "best_sharpe": best_sharpe,
        "best_params": best_params,
        "overall_good_ratio": overall_good_ratio,
        "per_param_good_ratio": per_param,
        "good_threshold": good_threshold,
        "verdict": verdict,
    }


def _py_val(v):
    """把 numpy 标量转 Python 原生类型"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ======================================================================
# CLI
# ======================================================================
def _parse_args():
    p = argparse.ArgumentParser(description="Funding Harvester 敏感度扫描")
    p.add_argument("--db", default="data/quant.db")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--symbols", default=None)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--output", default="analysis/output/funding_sensitivity.json")
    return p.parse_args()


def _ts_ms(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").value // 1_000_000)


def main():
    args = _parse_args()
    storage = Storage(args.db)
    start_ms = _ts_ms(args.start)
    end_ms = _ts_ms(args.end) if args.end else int(pd.Timestamp.now("UTC").value // 1_000_000)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        with storage._conn() as conn:
            cur = conn.execute(
                "SELECT DISTINCT symbol FROM funding_rates ORDER BY symbol"
            )
            symbols = [r[0] for r in cur.fetchall()]

    results = run_grid_scan(storage, symbols, start_ms, end_ms, grid=DEFAULT_GRID,
                            initial_capital=args.capital)
    analysis = compute_good_ratios(results)

    out = {
        "grid": DEFAULT_GRID,
        "n_runs": len(results),
        "results": results,
        "analysis": analysis,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_py_val)

    logger.info(f"扫描完成: {len(results)} 格, 最优 Sharpe={analysis['best_sharpe']:.4f}")
    logger.info(f"判决: {analysis['verdict']} (overall good_ratio={analysis['overall_good_ratio']:.2%})")
    logger.info(f"最优参数: {analysis['best_params']}")
    logger.info(f"结果写入 {out_path}")


if __name__ == "__main__":
    main()

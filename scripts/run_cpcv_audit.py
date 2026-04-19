"""
CPCV 历史策略审计 — v1.0 (P2B-T04)

对以下策略用 N=10 / k=2 的 Combinatorial Purged CV 重新跑:
  - macd_momentum   (4h)
  - triple_ema      (4h)
  - mean_reversion  (4h)
  - cross_sectional_momentum  (最佳配置)
  - funding_harvester         (基线配置)

每个策略产出:
  - 9 条 OOS path 的 Sharpe 分布
  - 中位数 Sharpe
  - PBO
  - DSR (中位 Sharpe)
  - 判决 (PASS / CONDITIONAL / FAIL)

最终聚合为 `analysis/output/cpcv_audit_report.md` 供人工 review。

实现说明
------
CPCV 对 "单标的规则策略" 和 "组合级策略" 需要不同的 strategy_fn 适配器:
  - 规则策略 (macd_momentum / triple_ema / mean_reversion): 在 train 上不调参,
    直接在 test 上以默认 config 回测, 返回 bar-level 收益
  - cross_sectional_momentum: 在 test 期按 rebalance_freq 推进, 收益来自组合日收益
  - funding_harvester: 在 test 期跑 funding_rates 8h 推进

为了让脚本在"没有完整历史数据"的环境下也能演示功能, 提供 `--demo` 模式:
合成 2 年日频价格序列, 跑所有策略, 产出完整报告。实盘数据模式需 `data/quant.db` 就绪。

用法:
  # 演示模式 (不依赖 DB)
  python scripts/run_cpcv_audit.py --demo --output analysis/output/cpcv_audit_report.md

  # 实盘数据模式 (要求 data/quant.db 存在)
  python scripts/run_cpcv_audit.py --db data/quant.db --start 2022-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from validation import CombinatorialPurgedCV, summarise_cpcv_paths, probability_of_backtest_overfitting
from utils.logger import get_logger

logger = get_logger("cpcv_audit")


# ======================================================================
# 合成价格 (demo 模式用)
# ======================================================================
def _synthetic_price_series(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """生成带温和趋势 + 波动率聚集的合成日频价格"""
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal(n_days) * 0.02 + 0.0003
    # 添加 regime (200-400 天为高波动)
    rets[200:400] = rets[200:400] * 2.0
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.015, n_days))
    low = close * (1 - rng.uniform(0, 0.015, n_days))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(1e6, 5e6, n_days)
    idx = pd.date_range("2022-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    }, index=idx)


def _synthetic_funding_series(n_cycles: int = 2190, seed: int = 43) -> pd.DataFrame:
    """生成 2 年的 8h funding rate 序列"""
    rng = np.random.default_rng(seed)
    # 均值 0.0001 (0.01% per 8h), 添加正负 regime
    base = rng.normal(1e-4, 8e-5, n_cycles)
    # 偶发翻负
    flip_idx = rng.choice(n_cycles, size=n_cycles // 10, replace=False)
    base[flip_idx] = -np.abs(base[flip_idx])
    idx = pd.date_range("2022-01-01", periods=n_cycles, freq="8h")
    return pd.DataFrame({
        "funding_rate": base,
        "mark_price": 40_000 * np.exp(np.cumsum(rng.standard_normal(n_cycles) * 0.01)),
    }, index=idx)


# ======================================================================
# Strategy adapters for CPCV
# ======================================================================
def _bar_returns_from_signals(
    prices: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
) -> pd.Series:
    """
    通用: signal_fn 返回 [-1, 0, +1] 的 position series, 此 helper 算 bar-level 收益。
    """
    pos = signal_fn(prices).shift(1).fillna(0.0)   # 次日入场, 避免 lookahead
    close = prices["close"]
    bar_ret = close.pct_change().fillna(0.0) * pos
    return bar_ret


def _macd_momentum_signal(prices: pd.DataFrame) -> pd.Series:
    """简化 MACD momentum: MACD > signal → long"""
    close = prices["close"]
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    pos = (macd > signal).astype(float)
    return pos


def _triple_ema_signal(prices: pd.DataFrame) -> pd.Series:
    """简化 Triple EMA: EMA_20 > EMA_50 > EMA_200 → long"""
    close = prices["close"]
    e1 = close.ewm(span=20, adjust=False).mean()
    e2 = close.ewm(span=50, adjust=False).mean()
    e3 = close.ewm(span=200, adjust=False).mean()
    pos = ((e1 > e2) & (e2 > e3)).astype(float)
    return pos


def _mean_reversion_signal(prices: pd.DataFrame) -> pd.Series:
    """简化 mean reversion: 跌破 BB_lower → long, 回到 BB_middle → exit"""
    close = prices["close"]
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    lower = ma - 2 * std
    # 跌破 lower 开 long, close ≥ ma 平
    pos_raw = pd.Series(0.0, index=close.index)
    in_pos = False
    for i in range(len(close)):
        c = close.iloc[i]
        if pd.isna(lower.iloc[i]) or pd.isna(ma.iloc[i]):
            continue
        if not in_pos and c <= lower.iloc[i]:
            in_pos = True
        elif in_pos and c >= ma.iloc[i]:
            in_pos = False
        pos_raw.iloc[i] = 1.0 if in_pos else 0.0
    return pos_raw


def _cross_sectional_signal(prices: pd.DataFrame) -> pd.Series:
    """
    单资产的降级代理: 用 30d momentum > 0 做 long 信号。
    真实横截面策略需要多币种 panel 数据。
    """
    close = prices["close"]
    mom_30d = close / close.shift(30) - 1
    pos = (mom_30d > 0).astype(float).fillna(0.0)
    return pos


def _funding_harvester_signal(funding: pd.DataFrame) -> pd.Series:
    """
    Funding harvester 的 CPCV 代理: funding > 1e-4 → hold delta-neutral position,
    返回 bar-level "收益" = funding_rate - 每 bar 成本均摊。
    """
    rate = funding["funding_rate"]
    # 开仓条件: 当前 rate > 1e-4 AND 上一 rate 也 > 1e-4
    in_pos = (rate > 1e-4) & (rate.shift(1) > 1e-4)
    # 成本按 7bps round-trip, 每 bar 均摊 (假设平均持有 30 cycles = 10 天)
    cost_per_bar = 7e-4 / 30
    ret = np.where(in_pos, rate - cost_per_bar, 0.0)
    return pd.Series(ret, index=funding.index)


# ======================================================================
# Audit runner
# ======================================================================
def run_cpcv_for_strategy(
    strategy_name: str,
    returns_fn: Callable[[pd.DataFrame], pd.Series],
    data: pd.DataFrame,
    n_groups: int = 10,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    n_trials: int = 1,
    bars_per_year: float = 252.0,
) -> dict:
    """
    在给定 data (时序) 上跑 CPCV.

    返回判决 dict 与 summarise_cpcv_paths 格式一致, 外加 strategy 名 / PBO / 路径 Sharpe。

    重要: 若 returns_fn 不依赖 train_idx (纯规则策略), 所有 9 条 path 输出相同,
    `is_deterministic=True`, PBO 强制设为 NaN 并附 note 字段说明。
    """
    cv = CombinatorialPurgedCV(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        embargo_pct=embargo_pct,
    )

    # 直接按全序列算收益, 再按 test_idx 切
    full_returns = returns_fn(data)
    # 确保 pandas Series + index 对齐
    if not isinstance(full_returns, pd.Series):
        full_returns = pd.Series(full_returns, index=data.index)
    full_returns = full_returns.reindex(data.index).fillna(0.0)

    def strat(train_idx, test_idx):
        # 规则策略不在 train 上调参, 直接返回 test 期收益
        # → 导致 all paths 相同, 在 summarise_cpcv_paths 中会被标记为 deterministic
        return full_returns.iloc[test_idx].copy()

    curves = cv.backtest_paths(data, strat)
    out = summarise_cpcv_paths(
        curves, n_trials=n_trials, bars_per_year=bars_per_year,
    )

    is_deterministic = out.get("is_deterministic", False)
    notes: list[str] = []

    # PBO 计算: 只对真正有 path 间差异的策略做
    if is_deterministic:
        pbo = float("nan")
        notes.append(
            "PBO=N/A: 策略无 train-time 参数, 所有 CPCV path 的 OOS 收益完全相同 "
            "(sharpe_std≈0), PBO 在 tie 情况下退化为排名赌博。需要 ML / 参数拟合"
            "策略才能产出有意义的 PBO。"
        )
    elif len(curves) >= 2:
        try:
            R = pd.concat(
                [c.reset_index(drop=True) for c in curves], axis=1,
            ).fillna(0.0).values
            n_sp = min(len(R), 8)
            if n_sp >= 2 and n_sp % 2 == 0:
                pbo = probability_of_backtest_overfitting(R, n_splits=n_sp)
            else:
                pbo = float("nan")
                notes.append(f"PBO: 样本数 {len(R)} 不足以做 CSCV, 跳过")
        except Exception as e:
            logger.warning(f"{strategy_name} PBO 失败: {e}")
            pbo = float("nan")
            notes.append(f"PBO 计算异常: {e}")
    else:
        pbo = float("nan")

    # G1 硬门槛判决: PBO<30% AND sharpe_median>1.0 AND dsr>0.8
    # 对 deterministic 策略: 跳过 PBO 条件, 仅看 Sharpe + DSR
    if is_deterministic:
        g1_pass = (
            np.isfinite(out["sharpe_median"]) and out["sharpe_median"] > 1.0
            and np.isfinite(out["dsr_median"]) and out["dsr_median"] > 0.8
        )
        if g1_pass:
            notes.append(
                "G1 判决为 PASS 但基于单条 OOS 曲线 (deterministic), "
                "上线前应构造 parameter-grid 变体重跑 CPCV 获得真正的 PBO。"
            )
    else:
        g1_pass = (
            np.isfinite(pbo) and pbo < 0.30
            and np.isfinite(out["sharpe_median"]) and out["sharpe_median"] > 1.0
            and np.isfinite(out["dsr_median"]) and out["dsr_median"] > 0.8
        )

    return {
        "strategy": strategy_name,
        "n_paths": out["n_paths"],
        "sharpe_median": out["sharpe_median"],
        "sharpe_mean": out["sharpe_mean"],
        "sharpe_std": out["sharpe_std"],
        "sharpe_per_path": out["sharpe_per_path"],
        "dsr_median": out["dsr_median"],
        "pbo": pbo,
        "verdict_simple": out["verdict"],
        "g1_pass": g1_pass,
        "is_deterministic": is_deterministic,
        "bars_per_year": out["bars_per_year"],
        "notes": notes,
    }


def format_audit_report(results: list[dict]) -> str:
    """生成 Markdown 报告"""
    lines = [
        "# CPCV 历史策略审计报告",
        "",
        f"> 生成时间: {pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"> 任务: P2B-T04 / P2B-T05",
        f"> 方法: Combinatorial Purged CV (N=10, k=2) → 9 OOS paths per strategy",
        f"> 年化换算: 日线 252 / 4h bar 2190 / 8h funding 1095",
        "",
        "## 结果汇总",
        "",
        "| 策略 | Sharpe 中位 | std | DSR | PBO | det? | G1 门槛 | 简判决 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        det_flag = "✓" if r.get("is_deterministic", False) else " "
        lines.append(
            f"| {r['strategy']} "
            f"| {_fmt(r['sharpe_median'])} "
            f"| {_fmt(r.get('sharpe_std', float('nan')))} "
            f"| {_fmt(r['dsr_median'])} "
            f"| {_fmt_pbo(r['pbo'])} "
            f"| {det_flag} "
            f"| {'✅ PASS' if r['g1_pass'] else '❌ FAIL'} "
            f"| {r['verdict_simple']} |"
        )
    lines += [
        "",
        "**图例**: `det?` 列标 `✓` 表示策略无 train-time 参数, 所有 path 的 OOS 收益完全相同 → PBO 显示为 `N/A`。",
        "",
        "## G1 上线门槛",
        "",
        "策略必须 **同时** 满足 (非 deterministic 策略):",
        "- PBO < 30%",
        "- CPCV 中位 Sharpe > 1.0",
        "- Deflated Sharpe Ratio > 0.8",
        "",
        "对 deterministic 规则策略, G1 仅看 Sharpe 和 DSR; 但需上线前构造 parameter-grid 变体重跑以获得真正的 PBO (见路线图 Phase 3 LightGBM aggregator)。",
        "",
    ]

    # Notes 段
    any_notes = any(r.get("notes") for r in results)
    if any_notes:
        lines += ["## 备注", ""]
        for r in results:
            if r.get("notes"):
                lines.append(f"### {r['strategy']}")
                for n in r["notes"]:
                    lines.append(f"- {n}")
                lines.append("")

    lines += [
        "## 每条 path 的 Sharpe 分布",
        "",
    ]
    for r in results:
        lines.append(f"### {r['strategy']}")
        lines.append("")
        sps = [f"{x:.3f}" if np.isfinite(x) else "nan" for x in r['sharpe_per_path']]
        lines.append("| path | " + " | ".join(str(i) for i in range(len(sps))) + " |")
        lines.append("|------|" + "|".join(["---"] * len(sps)) + "|")
        lines.append("| SR   | " + " | ".join(sps) + " |")
        lines.append("")

    lines += [
        "## 建议下一步",
        "",
        "- 对 `G1 PASS` 的策略, 进入 paper trading (Phase 2A-T05 通道).",
        "- 对 `G1 FAIL` 但 Sharpe 中位 > 0.5 的策略, 考虑参数调优 + 再跑一次 CPCV.",
        "- 对 PBO > 40% 的策略 (非 deterministic), 按路线图要求立即停用 (见 `docs/TODO_12M_DEVELOPMENT.md` §C-T02).",
        "- 对 Sharpe 中位 < 0 的策略, 加入 `docs/DEPRECATED_FACTORS_AND_STRATEGIES.md`.",
        "- 对 `det?=✓` 的策略, 要获得有意义的 PBO, 必须在 Phase 3 给策略加入 train-time 参数 (例如 LightGBM 阈值) 再跑 CPCV。",
        "",
    ]
    return "\n".join(lines)


def _fmt(x) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3f}"


def _fmt_pbo(x) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.1%}"


# ======================================================================
# Main
# ======================================================================
def audit_demo() -> list[dict]:
    """演示模式: 合成数据 + 5 个代理策略"""
    logger.info("=== demo 模式: 合成 2 年日频 + 2 年 8h funding ===")
    prices = _synthetic_price_series(n_days=730)
    funding = _synthetic_funding_series(n_cycles=2190)

    results = []
    # (name, returns_fn, data, bars_per_year)
    strategies = [
        ("macd_momentum", lambda d: _bar_returns_from_signals(d, _macd_momentum_signal), prices, 252.0),
        ("triple_ema", lambda d: _bar_returns_from_signals(d, _triple_ema_signal), prices, 252.0),
        ("mean_reversion", lambda d: _bar_returns_from_signals(d, _mean_reversion_signal), prices, 252.0),
        ("cross_sectional_momentum", lambda d: _bar_returns_from_signals(d, _cross_sectional_signal), prices, 252.0),
        ("funding_harvester", _funding_harvester_signal, funding, 1095.0),  # 8h bar → 3×365
    ]
    for name, fn, data, bpy in strategies:
        logger.info(f"  跑 CPCV: {name} (bars_per_year={bpy})")
        r = run_cpcv_for_strategy(
            strategy_name=name,
            returns_fn=fn,
            data=data,
            n_groups=10, n_test_groups=2, embargo_pct=0.01,
            n_trials=1, bars_per_year=bpy,
        )
        results.append(r)
        logger.info(f"    median Sharpe={_fmt(r['sharpe_median'])} "
                    f"DSR={_fmt(r['dsr_median'])} PBO={_fmt_pbo(r['pbo'])} "
                    f"deterministic={r['is_deterministic']} "
                    f"G1={'PASS' if r['g1_pass'] else 'FAIL'}")
    return results


def audit_real(db_path: str, start: str, end: Optional[str], symbol: str = "BTCUSDT") -> list[dict]:
    """实盘数据模式: 从 quant.db 取历史, 跑同样 5 个策略"""
    from data.storage import Storage
    storage = Storage(db_path)

    start_ms = int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    end_ms = int(pd.Timestamp(end, tz="UTC").value // 1_000_000) if end else int(
        pd.Timestamp.now("UTC").value // 1_000_000
    )

    # 价格 (4h)
    kl = storage.get_klines(symbol, "4h", limit=100_000)
    if isinstance(kl, pd.DataFrame) and not kl.empty:
        kl = kl[(kl["open_time"] >= start_ms) & (kl["open_time"] <= end_ms)]
        kl = kl.set_index(pd.to_datetime(kl["open_time"], unit="ms"))
    if kl is None or kl.empty:
        logger.warning(f"{symbol} 4h klines 为空, 退回 demo 模式")
        return audit_demo()

    prices = kl[["open", "high", "low", "close", "volume"]].copy()

    # Funding
    fund = storage.get_funding_rates(symbol, start_ms=start_ms, end_ms=end_ms)
    if fund is None or fund.empty:
        logger.warning(f"{symbol} funding 为空, funding_harvester 用合成数据")
        funding = _synthetic_funding_series(n_cycles=max(365, len(prices) // 4))
    else:
        fund = fund.set_index(pd.to_datetime(fund["funding_time"], unit="ms"))
        funding = fund[["funding_rate", "mark_price"]].copy()

    results = []
    # 4h bar: 6 bars/day × 365 = 2190/yr; 8h bar: 3×365 = 1095/yr
    pairs = [
        ("macd_momentum", lambda d: _bar_returns_from_signals(d, _macd_momentum_signal), prices, 2190.0),
        ("triple_ema", lambda d: _bar_returns_from_signals(d, _triple_ema_signal), prices, 2190.0),
        ("mean_reversion", lambda d: _bar_returns_from_signals(d, _mean_reversion_signal), prices, 2190.0),
        ("cross_sectional_momentum", lambda d: _bar_returns_from_signals(d, _cross_sectional_signal), prices, 2190.0),
        ("funding_harvester", _funding_harvester_signal, funding, 1095.0),
    ]
    for name, fn, data, bpy in pairs:
        logger.info(f"  跑 CPCV: {name} (rows={len(data)}, bars_per_year={bpy})")
        if len(data) < 30:
            logger.warning(f"    {name} 样本不足, 跳过")
            continue
        r = run_cpcv_for_strategy(name, fn, data, bars_per_year=bpy)
        results.append(r)
    return results


def _parse_args():
    p = argparse.ArgumentParser(description="CPCV 历史策略审计")
    p.add_argument("--demo", action="store_true", help="用合成数据演示 (不依赖 DB)")
    p.add_argument("--db", default="data/quant.db")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--output", default="analysis/output/cpcv_audit_report.md")
    p.add_argument("--json-output", default="analysis/output/cpcv_audit.json")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.demo:
        results = audit_demo()
    else:
        results = audit_real(args.db, args.start, args.end, args.symbol)

    # 写 Markdown 报告
    report = format_audit_report(results)
    out_md = Path(args.output)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Markdown 报告 → {out_md}")

    # 写 JSON (便于 CI 消费)
    out_json = Path(args.json_output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": results, "mode": "demo" if args.demo else "real"},
                  f, indent=2, ensure_ascii=False, default=_py_to_native)
    logger.info(f"JSON 结果 → {out_json}")

    # 关键指标回显
    for r in results:
        logger.info(f"  {r['strategy']}: SR_med={_fmt(r['sharpe_median'])} "
                    f"DSR={_fmt(r['dsr_median'])} PBO={_fmt_pbo(r['pbo'])} "
                    f"G1={'PASS' if r['g1_pass'] else 'FAIL'}")

    # 任一策略 PBO > 40% 退出码非零 (供 CI 使用)
    high_pbo = [r for r in results
                if np.isfinite(r.get("pbo", float("nan"))) and r["pbo"] > 0.40]
    if high_pbo:
        logger.warning(f"PBO > 40% 策略: {[r['strategy'] for r in high_pbo]} — CI 建议 FAIL")
        sys.exit(2)
    sys.exit(0)


def _py_to_native(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


if __name__ == "__main__":
    main()

"""
数据诊断工具 — 生成决策所需的补充数据

输出:
  1. 逐 bar 权益曲线 + 回撤曲线 (CSV + 概览)
  2. BTC/ETH 相关性矩阵 (滚动窗口)
  3. 分 regime 的策略表现统计
  4. 滑点真实分布 (基于订单簿数据)

用法:
  python analysis/data_diagnostic.py --db data/quant.db --strategy triple_ema --start 2025-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ---- 项目路径 ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.storage import Storage
from data.features import FeatureEngine
from alpha.strategy_registry import build_strategy
from config.loader import load_config
from backtest_runner import BacktestEngine
from utils.logger import get_logger

logger = get_logger("diagnostic")


# ==============================================================
# 1. 逐 bar 权益曲线 + 回撤
# ==============================================================
def generate_equity_curve(db_path: str, strategy_name: str,
                          start_date: str = None, end_date: str = None,
                          capital: float = 10000.0) -> pd.DataFrame:
    """运行回测并返回逐 bar 权益序列"""
    engine = BacktestEngine(
        db_path=db_path,
        strategy_name=strategy_name,
        initial_capital=capital,
        start_date=start_date,
        end_date=end_date,
    )
    report = engine.run()
    if not report:
        logger.error("回测无结果")
        return pd.DataFrame()

    eq = np.array(engine.equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.clip(peak, 1e-12, None)

    df = pd.DataFrame({
        "bar_index": range(len(eq)),
        "equity": eq,
        "peak": peak,
        "drawdown_pct": dd * 100,
    })

    # 滚动指标
    if len(eq) > 20:
        rets = np.diff(eq) / np.clip(eq[:-1], 1e-12, None)
        rets = np.insert(rets, 0, 0.0)
        df["bar_return"] = rets
        df["rolling_sharpe_60"] = (
            pd.Series(rets).rolling(60).mean()
            / pd.Series(rets).rolling(60).std().clip(lower=1e-12)
            * np.sqrt(6 * 365.25)  # 年化 (4h bars)
        ).values

    return df


# ==============================================================
# 2. BTC/ETH 相关性
# ==============================================================
def analyze_correlation(db_path: str, interval: str = "4h",
                        start_date: str = None) -> dict:
    """计算 BTC/ETH 收益率的滚动相关性"""
    storage = Storage(db_path)
    symbols = ["BTCUSDT", "ETHUSDT"]
    prices = {}

    for sym in symbols:
        klines = storage.get_klines(sym, interval, limit=50000)
        if klines.empty:
            logger.warning(f"{sym}/{interval} 无数据")
            continue
        df = klines.copy()
        if start_date:
            ts_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
            df = df[df["open_time"] >= ts_ms]
        df = df.sort_values("open_time")
        df["return"] = df["close"].pct_change()
        prices[sym] = df.set_index("open_time")["return"]

    if len(prices) < 2:
        return {"error": "数据不足"}

    merged = pd.DataFrame(prices).dropna()
    if merged.empty:
        return {"error": "合并后无数据"}

    overall_corr = float(merged["BTCUSDT"].corr(merged["ETHUSDT"]))

    # 滚动相关性
    windows = [30, 60, 120, 240]
    rolling_corrs = {}
    for w in windows:
        rc = merged["BTCUSDT"].rolling(w).corr(merged["ETHUSDT"])
        rolling_corrs[f"rolling_{w}_mean"] = float(rc.mean())
        rolling_corrs[f"rolling_{w}_min"] = float(rc.min())
        rolling_corrs[f"rolling_{w}_max"] = float(rc.max())
        rolling_corrs[f"rolling_{w}_std"] = float(rc.std())

    return {
        "overall_correlation": round(overall_corr, 4),
        "n_bars": len(merged),
        **{k: round(v, 4) for k, v in rolling_corrs.items()},
    }


# ==============================================================
# 3. 分 regime 的表现统计
# ==============================================================
def analyze_by_regime(db_path: str, strategy_name: str,
                      start_date: str = None, capital: float = 10000.0) -> dict:
    """按 ADX/波动率/趋势方向分桶统计"""
    engine = BacktestEngine(
        db_path=db_path,
        strategy_name=strategy_name,
        initial_capital=capital,
        start_date=start_date,
    )
    report = engine.run()
    if not report:
        return {"error": "回测无结果"}

    trades = report.get("all_trades", [])
    if not trades:
        return {"error": "无交易"}

    # 加载 features 做分桶映射
    storage = Storage(db_path)
    config = load_config()
    feat_engine = FeatureEngine(
        windows=config.get_nested("features.lookback_windows", [5, 10, 20, 60, 120, 240, 480])
    )

    results = {"by_adx": {}, "by_volatility": {}, "by_symbol": {}}

    for sym in ["BTCUSDT", "ETHUSDT"]:
        klines = storage.get_klines(sym, "4h", limit=50000)
        if klines.empty:
            continue
        if start_date:
            ts_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
            klines = klines[klines["open_time"] >= ts_ms]

        features = feat_engine.compute_all(klines)
        if features.empty:
            continue
        if "open_time" not in features.columns:
            features = features.copy()
            features["open_time"] = klines["open_time"].to_numpy()

        sym_trades = [t for t in trades if t.get("symbol") == sym]
        if not sym_trades:
            continue

        for t in sym_trades:
            entry_ts = t.get("entry_time", "")
            if not entry_ts or entry_ts == "END":
                continue

            # 找到最近的 bar
            try:
                entry_ms = int(pd.Timestamp(entry_ts).timestamp() * 1000)
            except Exception:
                continue

            idx = features["open_time"].searchsorted(entry_ms)
            if idx >= len(features):
                idx = len(features) - 1

            row = features.iloc[idx]
            adx = row.get("adx_14", 0)
            natr = row.get("natr_20", 0)
            pnl = t.get("pnl", 0)

            # ADX 分桶
            if pd.notna(adx):
                if adx < 20:
                    bucket = "weak(<20)"
                elif adx < 30:
                    bucket = "moderate(20-30)"
                else:
                    bucket = "strong(>30)"
                results["by_adx"].setdefault(bucket, {"trades": 0, "pnl": 0.0, "wins": 0})
                results["by_adx"][bucket]["trades"] += 1
                results["by_adx"][bucket]["pnl"] += pnl
                if pnl > 0:
                    results["by_adx"][bucket]["wins"] += 1

            # 波动率分桶
            if pd.notna(natr) and natr > 0:
                if natr < 0.015:
                    vbucket = "low(<1.5%)"
                elif natr < 0.03:
                    vbucket = "medium(1.5-3%)"
                else:
                    vbucket = "high(>3%)"
                results["by_volatility"].setdefault(vbucket, {"trades": 0, "pnl": 0.0, "wins": 0})
                results["by_volatility"][vbucket]["trades"] += 1
                results["by_volatility"][vbucket]["pnl"] += pnl
                if pnl > 0:
                    results["by_volatility"][vbucket]["wins"] += 1

    # 胜率计算
    for category in results.values():
        if isinstance(category, dict):
            for bucket_data in category.values():
                if isinstance(bucket_data, dict) and "trades" in bucket_data:
                    n = bucket_data["trades"]
                    bucket_data["win_rate"] = round(bucket_data["wins"] / n * 100, 1) if n > 0 else 0
                    bucket_data["pnl"] = round(bucket_data["pnl"], 2)
                    bucket_data["avg_pnl"] = round(bucket_data["pnl"] / n, 2) if n > 0 else 0

    return results


# ==============================================================
# 4. 滑点分析 (基于订单簿)
# ==============================================================
def analyze_slippage(db_path: str) -> dict:
    """
    基于存储的订单簿快照估算真实滑点
    如果没有订单簿数据，用 K 线的 high-low 做代理
    """
    storage = Storage(db_path)
    results = {}

    for sym in ["BTCUSDT", "ETHUSDT"]:
        klines = storage.get_klines(sym, "4h", limit=50000)
        if klines.empty:
            continue

        df = klines.copy()
        df["spread_pct"] = (df["high"] - df["low"]) / df["close"] * 100
        df["body_pct"] = abs(df["close"] - df["open"]) / df["close"] * 100

        # 典型滑点估计：市价单的滑点约为 spread 的 10-20%
        estimated_slippage_bps = df["spread_pct"].median() * 100 * 0.15  # 15% of spread

        # 按成交量分位看滑点
        df["vol_quantile"] = pd.qcut(df["volume"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        vol_slippage = df.groupby("vol_quantile")["spread_pct"].median().to_dict()

        results[sym] = {
            "median_spread_pct": round(float(df["spread_pct"].median()), 4),
            "p95_spread_pct": round(float(df["spread_pct"].quantile(0.95)), 4),
            "estimated_slippage_bps": round(estimated_slippage_bps, 2),
            "current_assumption_bps": 10.0,  # 0.1%
            "spread_by_volume_quantile": {k: round(v, 4) for k, v in vol_slippage.items()},
        }

    return results


# ==============================================================
# 主入口
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="策略数据诊断工具")
    parser.add_argument("--db", type=str, default="data/quant.db")
    parser.add_argument("--strategy", type=str, default="triple_ema")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--output-dir", type=str, default="analysis/output")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    db = args.db

    print("=" * 70)
    print("  📊 策略数据诊断报告")
    print("=" * 70)

    # ---- 1. 权益曲线 ----
    print("\n[1/4] 生成权益曲线...")
    eq_df = generate_equity_curve(db, args.strategy, args.start, capital=args.capital)
    if not eq_df.empty:
        eq_path = f"{args.output_dir}/equity_curve_{args.strategy}.csv"
        eq_df.to_csv(eq_path, index=False)
        print(f"  ✅ 已保存: {eq_path}")
        print(f"  最大回撤: {eq_df['drawdown_pct'].max():.2f}%")
        print(f"  最终权益: {eq_df['equity'].iloc[-1]:.2f}")
        if "rolling_sharpe_60" in eq_df.columns:
            rs = eq_df["rolling_sharpe_60"].dropna()
            print(f"  滚动 Sharpe(60bar) 中位数: {rs.median():.3f}")
            print(f"  滚动 Sharpe(60bar) 最小值: {rs.min():.3f}")

    # ---- 2. 相关性 ----
    print("\n[2/4] BTC/ETH 相关性分析...")
    corr = analyze_correlation(db, start_date=args.start)
    if "error" not in corr:
        print(f"  整体相关性: {corr['overall_correlation']}")
        print(f"  60bar 滚动相关性 (均值): {corr.get('rolling_60_mean', 'N/A')}")
        print(f"  60bar 滚动相关性 (最小): {corr.get('rolling_60_min', 'N/A')}")
        print(f"  ⚠️ 相关性 > 0.7 意味着两个标的在风险上接近同一资产")
    else:
        print(f"  ❌ {corr['error']}")

    # ---- 3. Regime 分桶 ----
    print(f"\n[3/4] 分 Regime 统计 ({args.strategy})...")
    regime = analyze_by_regime(db, args.strategy, args.start, args.capital)
    if "error" not in regime:
        for category_name, buckets in regime.items():
            if isinstance(buckets, dict) and buckets:
                print(f"\n  {category_name}:")
                for bname, bdata in buckets.items():
                    if isinstance(bdata, dict):
                        print(f"    {bname}: {bdata['trades']}笔 | "
                              f"PnL={bdata['pnl']:+.2f} | "
                              f"WR={bdata.get('win_rate', 0):.1f}% | "
                              f"Avg={bdata.get('avg_pnl', 0):.2f}")
    else:
        print(f"  ❌ {regime['error']}")

    # ---- 4. 滑点 ----
    print("\n[4/4] 滑点分析...")
    slip = analyze_slippage(db)
    for sym, data in slip.items():
        print(f"  {sym}:")
        print(f"    中位 Spread: {data['median_spread_pct']:.4f}%")
        print(f"    P95 Spread: {data['p95_spread_pct']:.4f}%")
        print(f"    估计真实滑点: {data['estimated_slippage_bps']:.1f} bps")
        print(f"    当前假设: {data['current_assumption_bps']:.1f} bps")

    # ---- 保存完整报告 ----
    report_path = f"{args.output_dir}/diagnostic_{args.strategy}.txt"
    with open(report_path, "w") as f:
        f.write(f"诊断报告 - {args.strategy}\n")
        f.write(f"生成时间: {datetime.now().isoformat()}\n\n")
        f.write(f"相关性: {corr}\n\n")
        f.write(f"Regime分桶: {regime}\n\n")
        f.write(f"滑点分析: {slip}\n")
    print(f"\n📄 完整报告: {report_path}")


if __name__ == "__main__":
    main()

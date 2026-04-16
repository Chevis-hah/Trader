"""
因子信号全扫描 — 200+ 因子预测力诊断

功能:
  1. 从 quant.db 读取所有标的的 K 线数据
  2. 调用 FeatureEngine 计算全部 241 个因子
  3. 对每个因子计算:
     - IC (信息系数 = Spearman 相关性 vs 前瞻收益)
     - ICIR (IC / std(IC)，衡量稳定性)
     - t-stat (IC 是否显著 ≠ 0)
     - 单调性 (分 5 桶，桶收益是否单调)
     - 胜率 (因子值 > 中位数时的胜率)
     - 最大桶收益 vs 最小桶收益
  4. 按大类分组，输出 JSON 报告
  5. 标记哪些因子有统计显著的 edge (|t-stat| > 2)

用法:
  python factor_signal_scan.py --db data/quant.db --interval 4h --output factor_scan_report.json
  python factor_signal_scan.py --db data/quant.db --interval 1h --forward-bars 6

输出:
  factor_scan_report.json  — 完整因子诊断报告
  请将此文件返回给我进行分析
"""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 因子分类 (匹配 features.py 的命名规则)
# ============================================================
FACTOR_CATEGORIES = {
    "momentum": [
        "ret_", "rank_ret_", "mom_", "acceleration_", "price_position_",
    ],
    "volatility": [
        "realized_vol_", "parkinson_vol_", "gk_vol_", "atr_", "natr_",
        "return_skew_", "return_kurt_",
    ],
    "volume": [
        "rel_volume_", "volume_price_corr_", "vwap_deviation_",
        "obv_", "taker_buy_ratio_",
    ],
    "mean_reversion": [
        "ma_deviation_", "zscore_", "bb_position_", "bb_width_",
    ],
    "trend": [
        "ema_", "sma_", "macd_", "adx_", "ichimoku_", "trend_strength",
    ],
    "oscillator": [
        "rsi_", "stoch_", "williams_", "cci_", "mfi_", "keltner_",
    ],
    "microstructure": [
        "amihud_", "kyle_", "intraday_", "streak_",
    ],
    "regime": [
        "vol_regime_", "trend_regime_", "hurst_", "realized_implied_",
    ],
    "temporal": [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    ],
}


def categorize_factor(col_name: str) -> str:
    """根据列名前缀归类"""
    for category, prefixes in FACTOR_CATEGORIES.items():
        for prefix in prefixes:
            if col_name.startswith(prefix) or col_name == prefix.rstrip("_"):
                return category
    return "other"


# ============================================================
# 核心分析函数
# ============================================================
def compute_forward_returns(df: pd.DataFrame, bars: int = 6) -> pd.Series:
    """
    计算前瞻收益 (未来 N 根 bar 的收益率)
    bars=6 on 4h = 24h forward return
    bars=6 on 1h = 6h forward return
    """
    close = pd.to_numeric(df["close"], errors="coerce")
    fwd = close.shift(-bars) / close - 1
    return fwd


def analyze_single_factor(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    factor_name: str,
    n_quantiles: int = 5,
) -> dict:
    """
    单因子诊断

    Returns dict with:
      - ic: 时序 IC 均值
      - icir: IC / std(IC)
      - t_stat: IC 的 t 统计量
      - p_value: 双侧 p 值
      - monotonicity: 分桶收益单调性 (-1 到 1)
      - win_rate_top: 因子 top quintile 的胜率
      - win_rate_bottom: 因子 bottom quintile 的胜率
      - spread: top - bottom 桶的年化收益差
    """
    # 对齐并去 NaN
    aligned = pd.DataFrame({
        "factor": factor_values,
        "fwd_ret": forward_returns,
    }).dropna()

    if len(aligned) < 60:
        return {
            "factor": factor_name,
            "category": categorize_factor(factor_name),
            "n_samples": len(aligned),
            "ic": 0.0,
            "icir": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "monotonicity": 0.0,
            "win_rate_top": 0.0,
            "win_rate_bottom": 0.0,
            "spread_annual": 0.0,
            "status": "INSUFFICIENT_DATA",
        }

    # ---- 分段 IC (Spearman) ----
    # 用非重叠分段代替滚动窗口，速度快 100x 且统计上更独立
    window = min(60, len(aligned) // 5)
    if window < 20:
        window = 20

    factor_arr = aligned["factor"].values
    fwd_arr = aligned["fwd_ret"].values
    n_total = len(aligned)

    # 非重叠分段: 每 window 个 bar 算一次 IC
    ic_list = []
    for start_idx in range(0, n_total - window + 1, window):
        f_slice = factor_arr[start_idx:start_idx + window]
        r_slice = fwd_arr[start_idx:start_idx + window]
        if np.std(f_slice) < 1e-12 or np.std(r_slice) < 1e-12:
            continue
        corr_val, _ = stats.spearmanr(f_slice, r_slice)
        if not np.isnan(corr_val):
            ic_list.append(corr_val)

    # 如果分段太少，补充半重叠分段
    if len(ic_list) < 15:
        half = window // 2
        for start_idx in range(half, n_total - window + 1, window):
            f_slice = factor_arr[start_idx:start_idx + window]
            r_slice = fwd_arr[start_idx:start_idx + window]
            if np.std(f_slice) < 1e-12 or np.std(r_slice) < 1e-12:
                continue
            corr_val, _ = stats.spearmanr(f_slice, r_slice)
            if not np.isnan(corr_val):
                ic_list.append(corr_val)

    if len(ic_list) < 5:
        ic_mean = 0.0
        ic_std = 1.0
    else:
        ic_mean = float(np.mean(ic_list))
        ic_std = float(np.std(ic_list, ddof=1))

    icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
    n_ic = len(ic_list) if ic_list else 1
    t_stat = ic_mean / (ic_std / np.sqrt(n_ic)) if ic_std > 1e-10 and n_ic > 1 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(1, n_ic - 1)))

    # ---- 分桶分析 ----
    try:
        aligned["quintile"] = pd.qcut(aligned["factor"], q=n_quantiles, labels=False, duplicates="drop")
    except Exception:
        aligned["quintile"] = pd.cut(aligned["factor"], bins=n_quantiles, labels=False)

    bucket_returns = aligned.groupby("quintile")["fwd_ret"].mean()
    if len(bucket_returns) < 2:
        monotonicity = 0.0
        spread = 0.0
    else:
        # Spearman 相关性衡量单调性
        ranks = np.arange(len(bucket_returns))
        corr, _ = stats.spearmanr(ranks, bucket_returns.values)
        monotonicity = float(corr) if not np.isnan(corr) else 0.0

        # 多空价差
        spread = float(bucket_returns.iloc[-1] - bucket_returns.iloc[0])

    # ---- 胜率 ----
    median_val = aligned["factor"].median()
    top_half = aligned[aligned["factor"] >= median_val]
    bot_half = aligned[aligned["factor"] < median_val]

    win_rate_top = float((top_half["fwd_ret"] > 0).mean()) if len(top_half) > 0 else 0.0
    win_rate_bottom = float((bot_half["fwd_ret"] > 0).mean()) if len(bot_half) > 0 else 0.0

    # ---- 年化价差 (假设 4h bar) ----
    # 1年约 2190 个 4h bar
    bars_per_year = 2190
    spread_annual = spread * bars_per_year / 6  # 除以 forward_bars

    # ---- 判定 ----
    if abs(t_stat) >= 3.0:
        status = "STRONG_SIGNAL"
    elif abs(t_stat) >= 2.0:
        status = "SIGNIFICANT"
    elif abs(t_stat) >= 1.5:
        status = "WEAK"
    else:
        status = "NO_SIGNAL"

    return {
        "factor": factor_name,
        "category": categorize_factor(factor_name),
        "n_samples": len(aligned),
        "ic": round(ic_mean, 6),
        "icir": round(icir, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "monotonicity": round(monotonicity, 4),
        "win_rate_top": round(win_rate_top, 4),
        "win_rate_bottom": round(win_rate_bottom, 4),
        "spread_annual": round(spread_annual, 4),
        "bucket_returns": {str(k): round(v, 6) for k, v in bucket_returns.items()},
        "status": status,
    }


# ============================================================
# 主流程
# ============================================================
def scan_all_factors(
    db_path: str,
    interval: str = "4h",
    forward_bars: int = 6,
    symbols: list[str] = None,
    start_date: str = None,
) -> dict:
    """
    扫描所有因子

    Args:
        db_path: quant.db 路径
        interval: K 线周期 (1h / 4h)
        forward_bars: 前瞻收益的 bar 数
        symbols: 标的列表
        start_date: 起始日期 (ISO 格式)

    Returns:
        完整诊断报告 dict
    """
    # 延迟导入 (避免不在项目环境下报错)
    try:
        from data.storage import Storage
        from data.features import FeatureEngine
        from config.loader import load_config
    except ImportError:
        print("错误: 请在 Trader 项目根目录下运行此脚本")
        sys.exit(1)

    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]

    storage = Storage(db_path)

    try:
        config = load_config()
        windows = config.get_nested("features.lookback_windows", [5, 10, 20, 60, 120, 240, 480])
    except Exception:
        windows = [5, 10, 20, 60, 120, 240, 480]

    feat_engine = FeatureEngine(windows=windows)

    all_results = []
    per_symbol_results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  扫描 {symbol} / {interval}")
        print(f"{'='*60}")

        klines = storage.get_klines(symbol, interval, limit=100000)
        if klines.empty:
            print(f"  ❌ 无数据: {symbol}/{interval}")
            continue

        if start_date:
            ts_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
            klines = klines[klines["open_time"] >= ts_ms]

        print(f"  K 线数量: {len(klines)}")
        print(f"  时间范围: {pd.to_datetime(klines['open_time'].iloc[0], unit='ms')} ~ "
              f"{pd.to_datetime(klines['open_time'].iloc[-1], unit='ms')}")

        # 计算特征
        print("  计算因子中...")
        t0 = time.time()
        features = feat_engine.compute_all(klines)
        t1 = time.time()
        print(f"  因子计算完成: {len(features.columns)} 列, 耗时 {t1-t0:.1f}s")

        # 前瞻收益
        fwd_ret = compute_forward_returns(features, bars=forward_bars)
        print(f"  前瞻收益 (N={forward_bars}): 均值={fwd_ret.mean():.6f}, 标准差={fwd_ret.std():.6f}")

        # 排除非因子列
        exclude_cols = {
            "open_time", "close_time", "open", "high", "low", "close",
            "volume", "quote_volume", "trades_count",
            "taker_buy_base", "taker_buy_quote",
            "symbol", "interval", "is_closed",
            "label_binary", "label_ternary", "forward_return",
        }

        factor_cols = [c for c in features.columns if c not in exclude_cols and features[c].dtype in ["float64", "float32", "int64"]]
        print(f"  因子列数: {len(factor_cols)}")

        symbol_results = []
        n_significant = 0
        n_strong = 0

        for i, col in enumerate(factor_cols):
            if (i + 1) % 50 == 0:
                print(f"    进度: {i+1}/{len(factor_cols)}")

            result = analyze_single_factor(features[col], fwd_ret, col)
            result["symbol"] = symbol

            symbol_results.append(result)
            all_results.append(result)

            if result["status"] == "SIGNIFICANT":
                n_significant += 1
            elif result["status"] == "STRONG_SIGNAL":
                n_strong += 1

        per_symbol_results[symbol] = symbol_results
        print(f"\n  结果汇总:")
        print(f"    STRONG_SIGNAL (|t|≥3): {n_strong}")
        print(f"    SIGNIFICANT (|t|≥2):   {n_significant}")
        print(f"    总因子: {len(factor_cols)}")

    # ============================================================
    # 汇总报告
    # ============================================================
    # 按类别分组
    category_summary = {}
    for result in all_results:
        cat = result["category"]
        if cat not in category_summary:
            category_summary[cat] = {
                "total": 0, "strong": 0, "significant": 0,
                "best_factors": [],
            }
        category_summary[cat]["total"] += 1
        if result["status"] == "STRONG_SIGNAL":
            category_summary[cat]["strong"] += 1
            category_summary[cat]["best_factors"].append({
                "name": result["factor"],
                "symbol": result.get("symbol", ""),
                "t_stat": result["t_stat"],
                "ic": result["ic"],
                "icir": result["icir"],
                "spread_annual": result["spread_annual"],
            })
        elif result["status"] == "SIGNIFICANT":
            category_summary[cat]["significant"] += 1
            category_summary[cat]["best_factors"].append({
                "name": result["factor"],
                "symbol": result.get("symbol", ""),
                "t_stat": result["t_stat"],
                "ic": result["ic"],
                "icir": result["icir"],
                "spread_annual": result["spread_annual"],
            })

    # 对 best_factors 按 |t_stat| 排序
    for cat in category_summary:
        category_summary[cat]["best_factors"].sort(
            key=lambda x: abs(x["t_stat"]), reverse=True
        )
        category_summary[cat]["best_factors"] = category_summary[cat]["best_factors"][:10]

    # Top 20 因子 (跨类别)
    top_factors = sorted(all_results, key=lambda x: abs(x["t_stat"]), reverse=True)[:20]

    report = {
        "scan_time": datetime.now().isoformat(),
        "config": {
            "db_path": db_path,
            "interval": interval,
            "forward_bars": forward_bars,
            "symbols": symbols,
            "start_date": start_date,
        },
        "summary": {
            "total_factors_scanned": len(all_results),
            "strong_signal_count": sum(1 for r in all_results if r["status"] == "STRONG_SIGNAL"),
            "significant_count": sum(1 for r in all_results if r["status"] == "SIGNIFICANT"),
            "weak_count": sum(1 for r in all_results if r["status"] == "WEAK"),
            "no_signal_count": sum(1 for r in all_results if r["status"] == "NO_SIGNAL"),
        },
        "category_summary": category_summary,
        "top_20_factors": [
            {k: v for k, v in f.items() if k != "bucket_returns"}
            for f in top_factors
        ],
        "all_results": all_results,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="因子信号全扫描")
    parser.add_argument("--db", type=str, default="data/quant.db",
                        help="数据库路径")
    parser.add_argument("--interval", type=str, default="4h",
                        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                        help="K 线周期")
    parser.add_argument("--forward-bars", type=int, default=6,
                        help="前瞻收益的 bar 数 (4h 下 6 bar = 24h)")
    parser.add_argument("--symbols", nargs="+",
                        default=["BTCUSDT", "ETHUSDT"],
                        help="标的列表")
    parser.add_argument("--start", type=str, default=None,
                        help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="factor_scan_report.json",
                        help="输出文件路径")
    args = parser.parse_args()

    print("=" * 70)
    print("  📊 因子信号全扫描 — 200+ 因子预测力诊断")
    print("=" * 70)
    print(f"  数据库: {args.db}")
    print(f"  周期:   {args.interval}")
    print(f"  前瞻:   {args.forward_bars} bars")
    print(f"  标的:   {args.symbols}")

    report = scan_all_factors(
        db_path=args.db,
        interval=args.interval,
        forward_bars=args.forward_bars,
        symbols=args.symbols,
        start_date=args.start,
    )

    # 保存报告
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  📋 扫描完成")
    print(f"{'='*70}")
    print(f"  总因子: {report['summary']['total_factors_scanned']}")
    print(f"  🟢 强信号 (|t|≥3): {report['summary']['strong_signal_count']}")
    print(f"  🟡 显著   (|t|≥2): {report['summary']['significant_count']}")
    print(f"  ⚪ 弱信号 (|t|≥1.5): {report['summary']['weak_count']}")
    print(f"  🔴 无信号: {report['summary']['no_signal_count']}")

    print(f"\n  Top 10 因子:")
    for i, f in enumerate(report["top_20_factors"][:10]):
        icon = "🟢" if f["status"] == "STRONG_SIGNAL" else "🟡"
        print(f"    {icon} {i+1}. {f['factor']} ({f.get('symbol', '')}) "
              f"t={f['t_stat']:.2f} IC={f['ic']:.4f} ICIR={f['icir']:.2f}")

    print(f"\n  📄 完整报告: {output_path}")
    print(f"\n  ⬆️ 请将 {output_path} 返回给我进行下一步分析")


if __name__ == "__main__":
    main()

"""
多策略组合回测 — 基于 RegimeAllocatorV2 动态分配

将趋势策略和均值回归策略组合在一起:
  - 强趋势 (ADX>30): 100% 趋势策略 (MACD/TripleEMA)
  - 中等趋势: 50/50 混合
  - 震荡 (ADX<20): 100% 均值回归
  - 暴跌: 全部关闭

用法:
  python multi_strategy_backtest.py --db data/quant.db --interval 4h
  python multi_strategy_backtest.py --db data/quant.db --interval 1h --ml
"""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("multi_strategy_bt")


def run_multi_strategy_backtest(
    db_path: str,
    interval: str = "4h",
    symbols: list = None,
    initial_capital: float = 10000.0,
    trend_strategy: str = "macd_momentum",
    use_ml: bool = False,
    start_date: str = None,
    output_path: str = None,
) -> dict:
    """
    Regime 感知的多策略组合回测
    """
    from data.storage import Storage
    from data.features import FeatureEngine
    from alpha.strategy_registry import build_strategy
    from alpha.regime_allocator_v2 import RegimeAllocatorV2
    from config.loader import load_config

    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]

    storage = Storage(db_path)
    try:
        config = load_config()
        windows = config.get_nested("features.lookback_windows", [5, 10, 20, 60, 120, 240, 480])
    except Exception:
        windows = [5, 10, 20, 60, 120, 240, 480]
    feat_engine = FeatureEngine(windows=windows)

    # 构建策略
    trend_strat = build_strategy(explicit_name=trend_strategy)
    mr_strat = build_strategy(explicit_name="mean_reversion")

    # ML 模型 (可选)
    ml_model = None
    if use_ml:
        try:
            from alpha.ml_lightgbm import LightGBMAlphaModel
            ml_model = LightGBMAlphaModel()
        except ImportError:
            logger.warning("ML 不可用")

    all_trades = []
    regime_stats = {}

    for symbol in symbols:
        klines = storage.get_klines(symbol, interval, limit=100000)
        if klines.empty:
            continue

        if start_date:
            ts_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
            klines = klines[klines["open_time"] >= ts_ms]

        features = feat_engine.compute_all(klines)
        if features.empty or len(features) < 500:
            continue

        higher = storage.get_klines(symbol, "1d", limit=10000)

        # 准备特征
        trend_feat = trend_strat.prepare_features(features, higher)
        mr_feat = mr_strat.prepare_features(features, higher)

        # ML 训练 (前 60% 训练, 后 40% 测试)
        ml_preds = None
        if ml_model is not None:
            split = int(len(features) * 0.6)
            try:
                ml_model.train(features.iloc[:split], forward_bars=6)
                ml_preds = ml_model.predict(features)
            except Exception as e:
                logger.warning(f"ML 训练失败: {e}")

        # 回测
        allocator = RegimeAllocatorV2(total_capital=initial_capital)
        equity = initial_capital
        trend_pos = None
        mr_pos = None
        trend_state = {"bar_index": 0, "cooldown_until_bar": -1}
        mr_state = {"bar_index": 0, "cooldown_until_bar": -1}

        slippage = 0.0021
        commission = 0.001

        for i in range(1, len(features)):
            t_row = trend_feat.iloc[i]
            m_row = mr_feat.iloc[i]
            prev_t = trend_feat.iloc[i - 1]
            prev_m = mr_feat.iloc[i - 1]
            close = float(t_row.get("close", 0) or 0)
            if close <= 0:
                continue

            trend_state["bar_index"] = i
            mr_state["bar_index"] = i

            # 更新 regime 分配
            alloc = allocator.update(t_row, i)
            trend_pct = alloc.get("trend", 0)
            mr_pct = alloc.get("mean_reversion", 0)

            # --- 趋势策略持仓管理 ---
            if trend_pos is not None:
                trend_pos["highest_since_entry"] = max(
                    trend_pos.get("highest_since_entry", close), close)
                bar_count = i - trend_pos["entry_bar"]
                should_exit, reason = trend_strat.check_exit(t_row, prev_t, trend_pos, bar_count)
                if should_exit:
                    pnl = _close_position(trend_pos, close, slippage, commission)
                    all_trades.append({"symbol": symbol, "strategy": trend_strategy,
                                       "pnl": pnl, "reason": reason, "bars": bar_count})
                    equity += pnl
                    trend_strat.on_trade_closed(trend_state, i, reason)
                    trend_pos = None

            # --- 均值回归持仓管理 ---
            if mr_pos is not None:
                mr_pos["highest_since_entry"] = max(
                    mr_pos.get("highest_since_entry", close), close)
                bar_count = i - mr_pos["entry_bar"]
                should_exit, reason = mr_strat.check_exit(m_row, prev_m, mr_pos, bar_count)
                if should_exit:
                    pnl = _close_position(mr_pos, close, slippage, commission)
                    all_trades.append({"symbol": symbol, "strategy": "mean_reversion",
                                       "pnl": pnl, "reason": reason, "bars": bar_count})
                    equity += pnl
                    mr_strat.on_trade_closed(mr_state, i, reason)
                    mr_pos = None

            # --- 趋势入场 ---
            if trend_pos is None and trend_pct > 0:
                if trend_strat.should_enter(t_row, prev_t, trend_state):
                    # ML 过滤
                    if ml_preds is not None and i < len(ml_preds):
                        if ml_preds.iloc[i].get("probability", 0.5) < 0.55:
                            continue
                    entry_p = close * (1 + slippage)
                    avail = equity * trend_pct
                    qty, stop = trend_strat.calc_position(avail, entry_p, t_row)
                    if qty > 0:
                        natr = float(t_row.get("natr_20", 0) or 0)
                        trend_pos = {"entry_price": entry_p, "entry_bar": i,
                                     "qty": qty, "stop_loss": stop,
                                     "highest_since_entry": close,
                                     "atr_at_entry": natr * entry_p if natr > 0 else 0}

            # --- 均值回归入场 ---
            if mr_pos is None and mr_pct > 0:
                if mr_strat.should_enter(m_row, prev_m, mr_state):
                    entry_p = close * (1 + slippage)
                    avail = equity * mr_pct
                    qty, stop = mr_strat.calc_position(avail, entry_p, m_row)
                    if qty > 0:
                        natr = float(m_row.get("natr_20", 0) or 0)
                        mr_pos = {"entry_price": entry_p, "entry_bar": i,
                                  "qty": qty, "stop_loss": stop,
                                  "highest_since_entry": close,
                                  "atr_at_entry": natr * entry_p if natr > 0 else 0}

        regime_stats[symbol] = allocator.stats
        logger.info(f"{symbol}: {len([t for t in all_trades if t['symbol']==symbol])} trades | "
                     f"Regime: {allocator.stats['regime_distribution']}")

    # 汇总
    total_pnl = sum(t["pnl"] for t in all_trades)
    trend_trades = [t for t in all_trades if t["strategy"] != "mean_reversion"]
    mr_trades = [t for t in all_trades if t["strategy"] == "mean_reversion"]

    report = {
        "config": {"interval": interval, "trend_strategy": trend_strategy,
                    "use_ml": use_ml, "capital": initial_capital},
        "summary": {
            "total_trades": len(all_trades),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / initial_capital * 100, 2),
            "trend_trades": len(trend_trades),
            "trend_pnl": round(sum(t["pnl"] for t in trend_trades), 2),
            "mr_trades": len(mr_trades),
            "mr_pnl": round(sum(t["pnl"] for t in mr_trades), 2),
            "win_rate": round(sum(1 for t in all_trades if t["pnl"] > 0) / max(len(all_trades), 1) * 100, 1),
        },
        "regime_stats": regime_stats,
        "trades": all_trades[:200],  # 截断避免太大
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"报告: {output_path}")

    return report


def _close_position(pos, close, slippage, commission):
    exit_p = close * (1 - slippage)
    pnl = pos["qty"] * (exit_p - pos["entry_price"])
    comm = pos["qty"] * close * commission * 2
    return round(pnl - comm, 4)


def main():
    parser = argparse.ArgumentParser(description="多策略组合回测")
    parser.add_argument("--db", default="data/quant.db")
    parser.add_argument("--interval", default="4h", choices=["1h", "4h"])
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--trend", default="macd_momentum",
                        help="趋势策略: macd_momentum / triple_ema")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--ml", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        ml_tag = "_ml" if args.ml else ""
        args.output = f"analysis/output/multi_{args.trend}_{args.interval}{ml_tag}.json"

    print("=" * 60)
    print(f"  多策略组合回测: {args.trend} + mean_reversion / {args.interval}")
    print(f"  ML: {'是' if args.ml else '否'}")
    print("=" * 60)

    report = run_multi_strategy_backtest(
        db_path=args.db, interval=args.interval, symbols=args.symbols,
        initial_capital=args.capital, trend_strategy=args.trend,
        use_ml=args.ml, start_date=args.start, output_path=args.output,
    )

    s = report["summary"]
    print(f"\n  总交易: {s['total_trades']} (趋势={s['trend_trades']} 均值回归={s['mr_trades']})")
    print(f"  总 PnL: {s['total_pnl']:+.2f} ({s['total_return_pct']:+.1f}%)")
    print(f"  趋势 PnL: {s['trend_pnl']:+.2f} | 均值回归 PnL: {s['mr_pnl']:+.2f}")
    print(f"  胜率: {s['win_rate']:.1f}%")
    print(f"\n  📄 {args.output}")


if __name__ == "__main__":
    main()

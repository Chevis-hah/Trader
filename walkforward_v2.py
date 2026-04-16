"""
Walk-forward + Monte Carlo 验证框架 — v2

功能:
1. Walk-forward 滚动验证 (可配训练/测试窗口)
2. ML 模型在每个 fold 重新训练
3. 真实成本扣除 (21-26 bps)
4. Monte Carlo 仿真 (打乱交易顺序，看权益曲线分布)
5. 输出完整报告 JSON

用法:
  python walkforward_v2.py --db data/quant.db --strategy macd_momentum --interval 4h
  python walkforward_v2.py --db data/quant.db --strategy mean_reversion --interval 1h --ml
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

logger = get_logger("walkforward_v2")


def run_walkforward(
    db_path: str,
    strategy_name: str,
    interval: str = "4h",
    symbols: list = None,
    train_days: int = 360,
    test_days: int = 60,
    initial_capital: float = 10000.0,
    use_ml_filter: bool = False,
    ml_threshold: float = 0.55,
    monte_carlo_runs: int = 0,
    output_path: str = None,
) -> dict:
    """
    Walk-forward 滚动验证

    Returns: 完整报告 dict
    """
    from data.storage import Storage
    from data.features import FeatureEngine
    from alpha.strategy_registry import build_strategy
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

    # ML 模型 (可选)
    ml_model = None
    if use_ml_filter:
        try:
            from alpha.ml_lightgbm import LightGBMAlphaModel
            ml_model = LightGBMAlphaModel(n_splits=3, purge_gap=6, embargo=3)
            logger.info("ML 过滤已启用")
        except ImportError:
            logger.warning("无法导入 ML 模型，跳过 ML 过滤")
            use_ml_filter = False

    # 加载数据
    all_trades = []
    fold_results = []

    for symbol in symbols:
        klines = storage.get_klines(symbol, interval, limit=100000)
        if klines.empty:
            continue

        features = feat_engine.compute_all(klines)
        if features.empty:
            continue

        # 复制必要的时间列 (compute_all 不保留原始列)
        for col in ["open_time", "open", "high", "low", "close", "volume"]:
            if col in klines.columns:
                features[col] = klines[col].values[-len(features):]

        higher_klines = storage.get_klines(symbol, "1d", limit=10000)

        # 时间范围
        timestamps = pd.to_datetime(features["open_time"], unit="ms")
        total_days = (timestamps.iloc[-1] - timestamps.iloc[0]).days

        if total_days < train_days + test_days:
            logger.warning(f"{symbol}: 数据不足 ({total_days} 天 < {train_days + test_days} 天)")
            continue

        logger.info(f"Walk-forward: {symbol}/{interval} | {total_days} 天 | "
                    f"训练={train_days}d 测试={test_days}d")

        # 滚动窗口
        fold_id = 0
        cursor_days = train_days

        while cursor_days + test_days <= total_days:
            train_start = timestamps.iloc[0] + pd.Timedelta(days=cursor_days - train_days)
            train_end = timestamps.iloc[0] + pd.Timedelta(days=cursor_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=test_days)

            train_mask = (timestamps >= train_start) & (timestamps < train_end)
            test_mask = (timestamps >= test_start) & (timestamps < test_end)

            train_feat = features[train_mask].copy()
            test_feat = features[test_mask].copy()

            if len(train_feat) < 100 or len(test_feat) < 10:
                cursor_days += test_days
                continue

            # 构建策略
            strategy = build_strategy(explicit_name=strategy_name)
            prepared = strategy.prepare_features(test_feat,
                                                  higher_klines if higher_klines is not None else None)

            # ML 模型训练 (可选)
            ml_preds = None
            if use_ml_filter and ml_model is not None:
                try:
                    ml_model.train(train_feat, forward_bars=6)
                    ml_preds = ml_model.predict(prepared)
                except Exception as e:
                    logger.warning(f"ML 训练失败 fold {fold_id}: {e}")

            # 运行回测
            trades = _run_single_fold(
                prepared, strategy, initial_capital, symbol,
                ml_preds=ml_preds, ml_threshold=ml_threshold
            )

            # 计算 fold 指标
            fold_pnl = sum(t["pnl"] for t in trades)
            fold_ret = fold_pnl / initial_capital * 100

            fold_result = {
                "fold_id": fold_id,
                "symbol": symbol,
                "test_period": f"{test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')}",
                "pnl": round(fold_pnl, 2),
                "return_pct": round(fold_ret, 2),
                "trades": len(trades),
                "sharpe": _calc_fold_sharpe(trades, initial_capital),
                "max_dd": _calc_fold_max_dd(trades, initial_capital),
            }

            fold_results.append(fold_result)
            all_trades.extend(trades)

            icon = "🟢" if fold_pnl > 0 else "🔴" if fold_pnl < 0 else "⚪"
            logger.info(f"  {icon} Fold {fold_id} [{symbol}] "
                        f"{fold_result['test_period']}: "
                        f"PnL={fold_pnl:+.2f} trades={len(trades)}")

            fold_id += 1
            cursor_days += test_days

    # 汇总
    n_folds = len(fold_results)
    n_positive = sum(1 for f in fold_results if f["pnl"] > 0)
    total_pnl = sum(f["pnl"] for f in fold_results)

    report = {
        "strategy": strategy_name,
        "interval": interval,
        "train_days": train_days,
        "test_days": test_days,
        "use_ml_filter": use_ml_filter,
        "n_folds": n_folds,
        "n_positive_folds": n_positive,
        "fold_win_rate": round(n_positive / n_folds * 100, 1) if n_folds > 0 else 0,
        "total_oos_pnl": round(total_pnl, 2),
        "avg_fold_pnl": round(total_pnl / n_folds, 2) if n_folds > 0 else 0,
        "total_trades": len(all_trades),
        "folds": fold_results,
    }

    # Monte Carlo
    if monte_carlo_runs > 0 and all_trades:
        mc = _monte_carlo(all_trades, initial_capital, monte_carlo_runs)
        report["monte_carlo"] = mc

    # 保存
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"报告已保存: {output_path}")

    return report


def _run_single_fold(features, strategy, capital, symbol,
                     ml_preds=None, ml_threshold=0.55):
    """运行单个 fold 的回测"""
    trades = []
    equity = capital
    position = None
    state = {"bar_index": 0, "cooldown_until_bar": -1}

    slippage = getattr(strategy.cfg, "slippage_pct", 0.0021)
    commission = getattr(strategy.cfg, "commission_pct", 0.001)

    for i in range(1, len(features)):
        row = features.iloc[i]
        prev_row = features.iloc[i - 1]
        state["bar_index"] = i

        close = float(row.get("close", 0) or 0)
        if close <= 0:
            continue

        # 持仓中: 检查出场
        if position is not None:
            bar_count = i - position["entry_bar"]
            position["highest_since_entry"] = max(
                position.get("highest_since_entry", close), close)

            should_exit, reason = strategy.check_exit(row, prev_row, position, bar_count)
            if should_exit:
                exit_price = close * (1 - slippage)
                pnl = position["qty"] * (exit_price - position["entry_price"])
                comm = position["qty"] * close * commission * 2
                pnl -= comm

                trades.append({
                    "symbol": symbol,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "qty": position["qty"],
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl / (position["qty"] * position["entry_price"]), 6),
                    "bars": bar_count,
                    "reason": reason,
                })
                equity += pnl
                strategy.on_trade_closed(state, i, reason)
                position = None
            continue

        # 无持仓: 检查入场
        if strategy.should_enter(row, prev_row, state):
            # ML 过滤
            if ml_preds is not None and i < len(ml_preds):
                prob = ml_preds.iloc[i].get("probability", 0.5)
                if prob < ml_threshold:
                    continue

            entry_price = close * (1 + slippage)
            qty, stop_loss = strategy.calc_position(equity, entry_price, row)
            if qty <= 0:
                continue

            natr = float(row.get("natr_20", 0) or 0)
            position = {
                "entry_price": entry_price,
                "entry_bar": i,
                "qty": qty,
                "stop_loss": stop_loss,
                "highest_since_entry": close,
                "atr_at_entry": natr * entry_price if natr > 0 else 0,
            }

    # 清仓
    if position is not None:
        close = float(features.iloc[-1].get("close", 0))
        if close > 0:
            pnl = position["qty"] * (close - position["entry_price"])
            comm = position["qty"] * close * commission * 2
            trades.append({
                "symbol": symbol,
                "entry_price": position["entry_price"],
                "exit_price": close,
                "qty": position["qty"],
                "pnl": round(pnl - comm, 4),
                "pnl_pct": round((pnl - comm) / (position["qty"] * position["entry_price"]), 6),
                "bars": len(features) - 1 - position["entry_bar"],
                "reason": "fold_end",
            })

    return trades


def _calc_fold_sharpe(trades, capital):
    if not trades:
        return 0.0
    rets = [t["pnl"] / capital for t in trades]
    if len(rets) < 2:
        return 0.0
    mean_r = np.mean(rets)
    std_r = np.std(rets, ddof=1)
    return round(mean_r / std_r * np.sqrt(len(rets)) if std_r > 1e-10 else 0.0, 3)


def _calc_fold_max_dd(trades, capital):
    if not trades:
        return 0.0
    equity = capital
    peak = capital
    max_dd = 0.0
    for t in trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)
    return round(max_dd, 2)


def _monte_carlo(trades, capital, n_runs=1000):
    """Monte Carlo: 打乱交易顺序，观察权益曲线分布"""
    pnls = [t["pnl"] for t in trades]
    n = len(pnls)
    final_equities = []
    max_dds = []

    for _ in range(n_runs):
        shuffled = np.random.permutation(pnls)
        eq = capital
        peak = capital
        worst_dd = 0.0
        for p in shuffled:
            eq += p
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            worst_dd = max(worst_dd, dd)
        final_equities.append(eq)
        max_dds.append(worst_dd)

    final_equities = np.array(final_equities)
    max_dds = np.array(max_dds)

    return {
        "n_runs": n_runs,
        "original_pnl": round(sum(pnls), 2),
        "median_final_equity": round(float(np.median(final_equities)), 2),
        "p5_final_equity": round(float(np.percentile(final_equities, 5)), 2),
        "p95_final_equity": round(float(np.percentile(final_equities, 95)), 2),
        "prob_profitable": round(float((final_equities > capital).mean()), 4),
        "median_max_dd_pct": round(float(np.median(max_dds) * 100), 2),
        "p95_max_dd_pct": round(float(np.percentile(max_dds, 95) * 100), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward + Monte Carlo 验证")
    parser.add_argument("--db", default="data/quant.db")
    parser.add_argument("--strategy", default="macd_momentum",
                        help="策略名: triple_ema / macd_momentum / mean_reversion / regime")
    parser.add_argument("--interval", default="4h", choices=["1h", "4h", "1d"])
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--train-days", type=int, default=360)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--ml", action="store_true", help="启用 ML 信号过滤")
    parser.add_argument("--ml-threshold", type=float, default=0.55)
    parser.add_argument("--monte-carlo", type=int, default=0, help="MC 仿真次数 (0=不做)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        suffix = "_ml" if args.ml else ""
        args.output = f"analysis/output/wf_{args.strategy}_{args.interval}{suffix}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Walk-forward 验证: {args.strategy} / {args.interval}")
    print(f"  ML 过滤: {'是' if args.ml else '否'}")
    print(f"  Monte Carlo: {args.monte_carlo} 次")
    print("=" * 60)

    report = run_walkforward(
        db_path=args.db,
        strategy_name=args.strategy,
        interval=args.interval,
        symbols=args.symbols,
        train_days=args.train_days,
        test_days=args.test_days,
        initial_capital=args.capital,
        use_ml_filter=args.ml,
        ml_threshold=args.ml_threshold,
        monte_carlo_runs=args.monte_carlo,
        output_path=args.output,
    )

    print(f"\n{'='*60}")
    print(f"  📋 结果汇总")
    print(f"{'='*60}")
    print(f"  Folds: {report['n_folds']}")
    print(f"  Fold 胜率: {report['fold_win_rate']}%")
    print(f"  OOS 总 PnL: {report['total_oos_pnl']}")
    print(f"  总交易数: {report['total_trades']}")

    if "monte_carlo" in report:
        mc = report["monte_carlo"]
        print(f"\n  Monte Carlo ({mc['n_runs']} 次):")
        print(f"    盈利概率: {mc['prob_profitable']:.1%}")
        print(f"    P5 权益: {mc['p5_final_equity']}")
        print(f"    P95 最大回撤: {mc['p95_max_dd_pct']:.1f}%")

    print(f"\n  📄 报告: {args.output}")


if __name__ == "__main__":
    main()

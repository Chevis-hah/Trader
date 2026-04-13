"""
集成补丁 — 将 P0-P5 所有模块接入现有回测系统

使用方法:
  1. 将此文件放到项目根目录
  2. 运行: python integration_patch.py --mode [diagnose|walkforward|enhanced|full]

模式说明:
  diagnose:    运行数据诊断 + 参数敏感性 (补充数据需求)
  walkforward: 运行 Walk-Forward 验证 (P0)
  enhanced:    用增强出场 + ML 过滤运行回测 (P3 + P4)
  full:        全量运行所有模块
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.logger import get_logger

logger = get_logger("integration")


def cmd_diagnose(args):
    """运行数据诊断（权益 CSV / 汇总 JSON 写入 analysis/output/）"""
    from analysis.data_diagnostic import (
        generate_equity_curve,
        analyze_correlation,
        analyze_by_regime,
        analyze_slippage,
    )

    out_dir = ROOT / "analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  📊 数据诊断模式")
    print("=" * 70)
    print(f"  输出目录: {out_dir}")

    summary: dict = {"strategies": {}, "start": args.start, "capital": args.capital}

    for strategy in ["triple_ema", "macd_momentum"]:
        print(f"\n{'─'*40}")
        print(f"  策略: {strategy}")
        print(f"{'─'*40}")

        # 权益曲线 → CSV（TODO Phase 1 收集项）
        eq = generate_equity_curve(args.db, strategy, args.start, capital=args.capital)
        if not eq.empty:
            csv_path = out_dir / f"equity_curve_{strategy}.csv"
            eq.to_csv(csv_path, index=False)
            print(f"  📄 权益曲线: {csv_path}")
            print(f"  最终权益: {eq['equity'].iloc[-1]:.2f}")
            print(f"  最大回撤: {eq['drawdown_pct'].max():.2f}%")

        # Regime 分析
        regime = analyze_by_regime(args.db, strategy, args.start, args.capital)
        if "error" not in regime:
            for cat, buckets in regime.items():
                if isinstance(buckets, dict) and buckets:
                    print(f"  {cat}:")
                    for b, d in buckets.items():
                        if isinstance(d, dict):
                            print(f"    {b}: {d['trades']}笔 PnL={d['pnl']:+.2f}")
        summary["strategies"][strategy] = {"regime": regime, "equity_rows": len(eq)}

    # 全局分析
    print(f"\n{'─'*40}")
    print("  全局分析")
    print(f"{'─'*40}")

    corr = analyze_correlation(args.db, start_date=args.start)
    if "error" not in corr:
        print(f"  BTC/ETH 相关性: {corr['overall_correlation']}")
    summary["correlation"] = corr

    slip = analyze_slippage(args.db)
    summary["slippage"] = slip
    for sym, d in slip.items():
        print(f"  {sym} 估计滑点: {d['estimated_slippage_bps']:.1f} bps "
              f"(假设 {d['current_assumption_bps']:.0f} bps)")

    summary_path = out_dir / "phase1_diagnose_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  📄 诊断汇总: {summary_path}")


def cmd_sensitivity(args):
    """运行参数敏感性"""
    from analysis.param_sensitivity import main as run_sensitivity
    sys.argv = [
        "param_sensitivity.py",
        "--db", args.db,
        "--strategy", args.strategy,
        "--start", args.start,
        "--mode", "1d",
    ]
    run_sensitivity()


def cmd_walkforward(args):
    """运行 Walk-Forward 验证"""
    from backtest_walkforward import WalkForwardEngine

    out_dir = ROOT / "analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    for strategy in ["triple_ema", "macd_momentum"]:
        print(f"\n{'='*70}")
        print(f"  Walk-Forward: {strategy}")
        print(f"{'='*70}")

        engine = WalkForwardEngine(
            db_path=args.db,
            strategy_name=strategy,
            initial_capital=args.capital,
            train_days=args.train_days,
            test_days=args.test_days,
        )
        summary = engine.run()

        output = out_dir / f"walkforward_{strategy}.json"
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  📄 {output}")


def cmd_enhanced(args):
    """用增强出场运行回测"""
    from backtest_runner import BacktestEngine, write_snapshot
    from alpha.strategy_registry import build_strategy
    from alpha.enhanced_exit import patch_strategy_exit, EnhancedExitConfig
    from config.loader import load_config

    config = load_config(args.config)

    for strategy_name in ["triple_ema", "macd_momentum"]:
        print(f"\n{'='*70}")
        print(f"  增强回测: {strategy_name}")
        print(f"{'='*70}")

        # 对比: 原版 vs 增强版
        for label, use_enhanced in [("原版", False), ("增强出场", True)]:
            engine = BacktestEngine(
                db_path=args.db,
                strategy_name=strategy_name,
                initial_capital=args.capital,
                start_date=args.start,
            )

            if use_enhanced:
                patch_strategy_exit(engine.strategy, EnhancedExitConfig(
                    partial_exit_enabled=True,
                    adaptive_trail_enabled=True,
                    vol_adaptive_enabled=True,
                ))

            report = engine.run()
            if report:
                s = report["summary"]
                o = report["trades_overview"]
                print(f"\n  [{label}]")
                print(f"    PnL: {s['total_pnl']:+.2f} ({s['total_return_pct']:+.2f}%)")
                print(f"    Sharpe: {s['sharpe_ratio']:.3f}")
                print(f"    MaxDD: {s['max_drawdown_pct']:.2f}%")
                print(f"    交易数: {o['total_trades']}")
                print(f"    胜率: {o['win_rate_pct']:.1f}%")
                print(f"    PF: {o['profit_factor']:.3f}")

                if use_enhanced:
                    snapshot = f"backtest_{strategy_name}_enhanced_snapshot.txt"
                    write_snapshot(report, snapshot, args.db, args.start, None)
                    print(f"    📄 {snapshot}")


def cmd_ml_train(args):
    """训练 ML 过滤模型"""
    from alpha.ml_signal_filter import MLSignalFilter

    print("=" * 70)
    print("  🧠 训练 ML 过滤模型")
    print("=" * 70)

    ml = MLSignalFilter(db_path=args.db)
    result = ml.train(
        start=args.ml_train_start,
        end=args.ml_train_end,
        target_bars=6,
        target_threshold=0.005,
    )

    if result:
        print(f"\n  平均 AUC: {result['avg_auc']:.4f}")
        print(f"  样本数: {result['n_samples']}")
        print(f"  特征数: {result['n_features']}")

        model_path = "models/ml_filter.pkl"
        ml.save(model_path)
        print(f"  📄 模型: {model_path}")

        if result['avg_auc'] < 0.52:
            print("\n  ⚠️ AUC < 0.52, ML 模型预测能力不足，建议不启用过滤")
        elif result['avg_auc'] < 0.55:
            print("\n  🟡 AUC 0.52-0.55, 可作为辅助参考，threshold 设高一些 (0.60+)")
        else:
            print("\n  ✅ AUC >= 0.55, 可以启用 ML 过滤")


def cmd_ml_backtest(args):
    """Phase 4.2: 加载 ML 模型后对 OOS 区间回测（默认 macd_momentum）"""
    from backtest_runner import BacktestEngine, write_snapshot
    from alpha.ml_signal_filter import MLSignalFilter, patch_strategy_with_ml

    model_path = ROOT / "models" / "ml_filter.pkl"
    if not model_path.is_file():
        print(f"❌ 未找到模型: {model_path}")
        print("   请先运行: python integration_patch.py --mode ml_train ...")
        return

    out_dir = ROOT / "analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ml = MLSignalFilter.load(str(model_path))
    strategy = args.ml_strategy

    print("=" * 70)
    print(f"  🤖 ML 过滤回测 | {strategy} | start={args.start}")
    print("=" * 70)

    engine = BacktestEngine(
        db_path=args.db,
        strategy_name=strategy,
        initial_capital=args.capital,
        start_date=args.start,
    )
    patch_strategy_with_ml(engine.strategy, ml, threshold=args.ml_threshold)
    report = engine.run()
    if report:
        snap = out_dir / f"backtest_{strategy}_ml_filtered_snapshot.txt"
        write_snapshot(report, str(snap), args.db, args.start, None)
        print(f"  📄 {snap}")


def cmd_full(args):
    """全量运行"""
    print("=" * 70)
    print("  🚀 全量运行所有模块")
    print("=" * 70)

    print("\n[Step 1/4] 数据诊断...")
    cmd_diagnose(args)

    print("\n[Step 2/4] 参数敏感性 (triple_ema)...")
    args.strategy = "triple_ema"
    cmd_sensitivity(args)

    print("\n[Step 3/4] Walk-Forward 验证...")
    cmd_walkforward(args)

    print("\n[Step 4/4] 增强出场对比...")
    cmd_enhanced(args)

    print("\n" + "=" * 70)
    print("  ✅ 全量运行完成")
    print("  请查看生成的 JSON/CSV/TXT 文件，按 TODO.md 反馈结果")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="集成补丁 — 全模块运行器")
    parser.add_argument("--mode", type=str, default="diagnose",
                        choices=["diagnose", "sensitivity", "walkforward",
                                 "enhanced", "ml_train", "ml_backtest", "full"])
    parser.add_argument("--db", type=str, default="data/quant.db")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="triple_ema")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--ml-train-start", type=str, default="2024-01-01")
    parser.add_argument("--ml-train-end", type=str, default="2024-12-31")
    parser.add_argument("--ml-threshold", type=float, default=0.55,
                        help="ML 过滤回测时的概率阈值")
    parser.add_argument("--ml-strategy", type=str, default="macd_momentum",
                        help="ML 过滤回测使用的策略名")
    args = parser.parse_args()

    mode_map = {
        "diagnose": cmd_diagnose,
        "sensitivity": cmd_sensitivity,
        "walkforward": cmd_walkforward,
        "enhanced": cmd_enhanced,
        "ml_train": cmd_ml_train,
        "ml_backtest": cmd_ml_backtest,
        "full": cmd_full,
    }

    fn = mode_map[args.mode]
    fn(args)


if __name__ == "__main__":
    main()

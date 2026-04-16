"""
结果汇总脚本 — 解析所有 WF 报告并生成对比表

用法:
    python scripts/summarize_results.py
    python scripts/summarize_results.py --dir analysis/output
"""
import argparse
import json
import glob
from pathlib import Path


def load_report(path):
    with open(path, "r") as f:
        return json.load(f)


# 上线标准
GATES = {
    "fold_win_rate": {"min": 50.0, "label": "Fold胜率%"},
    "total_oos_pnl": {"min": 0, "label": "OOS PnL"},
    "total_trades": {"min": 50, "label": "交易数"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="analysis/output")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.dir}/wf_*.json"))
    if not files:
        print(f"未找到报告文件: {args.dir}/wf_*.json")
        return

    print("=" * 100)
    print(f"  📊 Walk-forward 结果汇总 ({len(files)} 个报告)")
    print("=" * 100)

    header = f"{'报告':<40} {'Folds':>6} {'胜率%':>7} {'OOS PnL':>10} {'交易':>6} {'ML':>4} {'MC':>4} {'通过':>4}"
    print(header)
    print("-" * 100)

    best_report = None
    best_wr = -1

    for fpath in files:
        try:
            r = load_report(fpath)
        except Exception as e:
            print(f"  ❌ 加载失败: {fpath}: {e}")
            continue

        name = Path(fpath).stem
        n_folds = r.get("n_folds", 0)
        wr = r.get("fold_win_rate", 0)
        pnl = r.get("total_oos_pnl", 0)
        trades = r.get("total_trades", 0)
        ml = "✓" if r.get("use_ml_filter", False) else ""
        mc = "✓" if "monte_carlo" in r else ""

        # 检查 gate
        passed = wr >= GATES["fold_win_rate"]["min"] and pnl > 0 and trades >= GATES["total_trades"]["min"]
        gate_icon = "✅" if passed else "❌"

        print(f"  {name:<38} {n_folds:>6} {wr:>6.1f}% {pnl:>+10.2f} {trades:>6} {ml:>4} {mc:>4} {gate_icon:>4}")

        if wr > best_wr and trades >= 20:
            best_wr = wr
            best_report = (name, r)

    print("-" * 100)

    # 上线标准
    print(f"\n  📋 上线标准: Fold 胜率 ≥ 50% | OOS PnL > 0 | 交易数 ≥ 50")

    # Monte Carlo 详情
    for fpath in files:
        try:
            r = load_report(fpath)
        except Exception:
            continue
        if "monte_carlo" not in r:
            continue
        mc = r["monte_carlo"]
        name = Path(fpath).stem
        print(f"\n  🎲 Monte Carlo [{name}]:")
        print(f"     盈利概率: {mc['prob_profitable']:.1%}")
        print(f"     P5 权益:  {mc['p5_final_equity']}")
        print(f"     P95 回撤: {mc['p95_max_dd_pct']:.1f}%")

    if best_report:
        print(f"\n  🏆 最佳策略: {best_report[0]} (Fold 胜率 {best_wr:.1f}%)")

    print("")


if __name__ == "__main__":
    main()

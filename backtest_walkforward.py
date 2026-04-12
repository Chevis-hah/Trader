"""
P0: Walk-Forward 验证框架

核心原则：
- 永远在样本外验证策略表现
- 训练窗口 180 天，测试窗口 60 天，滚动 N 次
- 每次窗口独立评估，汇总得到真实的 OOS 表现

用法:
  python backtest_walkforward.py --db data/quant.db --strategy triple_ema
  python backtest_walkforward.py --db data/quant.db --strategy macd_momentum --train-days 180 --test-days 60
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest_runner import BacktestEngine, write_snapshot
from data.storage import Storage
from utils.logger import get_logger

logger = get_logger("walkforward")


@dataclass
class WFWindow:
    """单个 Walk-Forward 窗口"""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # 测试期结果
    pnl: float = 0.0
    return_pct: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    profit_factor: float = 0.0


class WalkForwardEngine:
    """
    Walk-Forward 回测引擎

    流程:
    1. 确定数据的总时间范围
    2. 按 train_days + test_days 滚动切分窗口
    3. 对每个窗口:
       - 训练期: 用于验证参数在该时段的表现 (可选: 参数优化)
       - 测试期: 用验证后的参数做纯 OOS 回测
    4. 汇总所有测试期的表现 = 真实的 OOS 表现
    """

    def __init__(
        self,
        db_path: str,
        strategy_name: str,
        initial_capital: float = 10000.0,
        train_days: int = 180,
        test_days: int = 60,
        step_days: int = None,  # 默认 = test_days (无重叠)
        config: dict = None,
    ):
        self.db_path = db_path
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days or test_days
        self.config = config or {}

    def _get_data_range(self) -> tuple[datetime, datetime]:
        """获取数据库中的时间范围"""
        storage = Storage(self.db_path)
        earliest = None
        latest = None

        for sym in ["BTCUSDT", "ETHUSDT"]:
            klines = storage.get_klines(sym, "4h", limit=100000)
            if klines.empty:
                continue
            t0 = pd.Timestamp(klines["open_time"].min(), unit="ms")
            t1 = pd.Timestamp(klines["open_time"].max(), unit="ms")
            if earliest is None or t0 < earliest:
                earliest = t0
            if latest is None or t1 > latest:
                latest = t1

        return earliest.to_pydatetime(), latest.to_pydatetime()

    def _generate_windows(self) -> list[WFWindow]:
        """生成 Walk-Forward 窗口序列"""
        data_start, data_end = self._get_data_range()
        windows = []
        fold_id = 0

        current = data_start
        while True:
            train_start = current
            train_end = current + timedelta(days=self.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_days)

            if test_end > data_end:
                # 最后一个窗口，test_end 截断到数据末尾
                if test_start < data_end - timedelta(days=10):
                    test_end = data_end
                else:
                    break

            windows.append(WFWindow(
                fold_id=fold_id,
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
            ))
            fold_id += 1
            current += timedelta(days=self.step_days)

        return windows

    def _run_fold(self, window: WFWindow) -> WFWindow:
        """运行单个 fold 的测试期回测"""
        try:
            engine = BacktestEngine(
                db_path=self.db_path,
                strategy_name=self.strategy_name,
                initial_capital=self.initial_capital,
                start_date=window.test_start,
                end_date=window.test_end,
                config=self.config,
            )
            report = engine.run()
            if report:
                summary = report.get("summary", {})
                overview = report.get("trades_overview", {})
                window.pnl = summary.get("total_pnl", 0)
                window.return_pct = summary.get("total_return_pct", 0)
                window.trades = overview.get("total_trades", 0)
                window.win_rate = overview.get("win_rate_pct", 0)
                window.sharpe = summary.get("sharpe_ratio", 0)
                window.max_dd = summary.get("max_drawdown_pct", 0)
                window.profit_factor = overview.get("profit_factor", 0)
        except Exception as e:
            logger.error(f"Fold {window.fold_id} 失败: {e}")

        return window

    def run(self) -> dict:
        """运行完整的 Walk-Forward 验证"""
        windows = self._generate_windows()
        if not windows:
            logger.error("无法生成足够的窗口")
            return {}

        print("=" * 70)
        print(f"  🔄 Walk-Forward 验证 — {self.strategy_name}")
        print(f"  训练期: {self.train_days}天 | 测试期: {self.test_days}天 | "
              f"步进: {self.step_days}天")
        print(f"  总窗口数: {len(windows)}")
        print("=" * 70)

        results = []
        for w in windows:
            print(f"\n  📅 Fold {w.fold_id}: "
                  f"训练 {w.train_start}~{w.train_end} | "
                  f"测试 {w.test_start}~{w.test_end}")

            w = self._run_fold(w)
            results.append(w)

            icon = "🟢" if w.pnl > 0 else "🔴"
            print(f"     {icon} PnL={w.pnl:+.2f} ({w.return_pct:+.2f}%) | "
                  f"{w.trades}笔 | WR={w.win_rate:.1f}% | "
                  f"Sharpe={w.sharpe:.3f} | DD={w.max_dd:.2f}%")

        return self._summarize(results)

    def _summarize(self, windows: list[WFWindow]) -> dict:
        """汇总所有 OOS 窗口的表现"""
        if not windows:
            return {}

        pnls = [w.pnl for w in windows]
        returns = [w.return_pct for w in windows]
        sharpes = [w.sharpe for w in windows]
        max_dds = [w.max_dd for w in windows]
        trades = [w.trades for w in windows]

        n_positive = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 0

        # 年化 OOS Sharpe (基于 fold 粒度)
        if std_pnl > 0:
            folds_per_year = 365.0 / self.test_days
            oos_sharpe = avg_pnl / std_pnl * np.sqrt(folds_per_year)
        else:
            oos_sharpe = 0.0

        summary = {
            "strategy": self.strategy_name,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "n_folds": len(windows),
            "n_positive_folds": n_positive,
            "fold_win_rate": round(n_positive / len(windows) * 100, 1),
            "total_oos_pnl": round(total_pnl, 2),
            "avg_fold_pnl": round(avg_pnl, 2),
            "std_fold_pnl": round(std_pnl, 2),
            "oos_sharpe": round(oos_sharpe, 3),
            "avg_sharpe": round(np.mean(sharpes), 3),
            "worst_fold_pnl": round(min(pnls), 2),
            "best_fold_pnl": round(max(pnls), 2),
            "max_fold_drawdown": round(max(max_dds), 2),
            "avg_trades_per_fold": round(np.mean(trades), 1),
            "total_trades": sum(trades),
            "folds": [
                {
                    "fold_id": w.fold_id,
                    "test_period": f"{w.test_start} ~ {w.test_end}",
                    "pnl": w.pnl,
                    "return_pct": w.return_pct,
                    "trades": w.trades,
                    "sharpe": w.sharpe,
                    "max_dd": w.max_dd,
                }
                for w in windows
            ],
        }

        # ---- 打印总结 ----
        print("\n" + "=" * 70)
        print("  📋 Walk-Forward 总结 (纯 OOS)")
        print("=" * 70)
        print(f"  窗口胜率: {summary['fold_win_rate']:.1f}% "
              f"({n_positive}/{len(windows)})")
        print(f"  总 OOS PnL: {total_pnl:+.2f}")
        print(f"  平均每窗口 PnL: {avg_pnl:+.2f} ± {std_pnl:.2f}")
        print(f"  OOS Sharpe: {oos_sharpe:.3f}")
        print(f"  最差窗口: {min(pnls):+.2f}")
        print(f"  最好窗口: {max(pnls):+.2f}")
        print(f"  最大单窗口回撤: {max(max_dds):.2f}%")

        # 判断
        if summary["fold_win_rate"] >= 60 and oos_sharpe > 0.5:
            print("\n  ✅ 策略在样本外表现一致，可信度较高")
        elif summary["fold_win_rate"] >= 50 and oos_sharpe > 0:
            print("\n  🟡 策略有一定样本外收益，但稳定性有限")
        else:
            print("\n  🔴 策略在样本外表现不佳，当前参数可能过拟合")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward 验证")
    parser.add_argument("--db", type=str, default="data/quant.db")
    parser.add_argument("--strategy", type=str, default="triple_ema",
                        choices=["triple_ema", "macd_momentum"])
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    engine = WalkForwardEngine(
        db_path=args.db,
        strategy_name=args.strategy,
        initial_capital=args.capital,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )

    summary = engine.run()

    output_path = args.output or f"walkforward_{args.strategy}.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n📄 详细结果: {output_path}")


if __name__ == "__main__":
    main()

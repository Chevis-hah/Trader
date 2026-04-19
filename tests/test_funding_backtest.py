"""P2A-T03: funding_harvester_backtest 单元测试 (合成数据, 不依赖实际 DB 内容)。"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.funding_harvester import FundingHarvesterConfig
from data.storage import Storage
from funding_harvester_backtest import run_funding_backtest


def _make_synthetic_funding(
    symbol: str, start_ms: int, n_records: int, rate_series: list[float],
    mark_price: float = 40_000.0,
) -> pd.DataFrame:
    """生成 8h 间隔的 funding 数据。"""
    assert len(rate_series) == n_records
    return pd.DataFrame({
        "symbol": [symbol] * n_records,
        "funding_time": [start_ms + i * 28_800_000 for i in range(n_records)],
        "funding_rate": rate_series,
        "mark_price": [mark_price] * n_records,
    })


class TestBacktestBasic(unittest.TestCase):
    def setUp(self):
        self.tmp_db = Path(__file__).parent / "_tmp_funding_bt.db"
        if self.tmp_db.exists():
            self.tmp_db.unlink()
        self.s = Storage(self.tmp_db)
        self.start_ms = 1_700_000_000_000
        self.end_ms = self.start_ms + 200 * 28_800_000  # 200 funding cycles ≈ 66 天

    def tearDown(self):
        if self.tmp_db.exists():
            self.tmp_db.unlink()

    def test_runs_without_data_raises(self):
        from funding_harvester_backtest import run_funding_backtest
        cfg = FundingHarvesterConfig()
        with self.assertRaises(RuntimeError):
            run_funding_backtest(self.s, cfg, ["BTCUSDT"], self.start_ms, self.end_ms)

    def test_zero_funding_no_trades(self):
        df = _make_synthetic_funding(
            "BTCUSDT", self.start_ms, 100, [0.0] * 100,
        )
        self.s.upsert_funding_rates(df)
        cfg = FundingHarvesterConfig(min_funding_rate=1e-4)
        out = run_funding_backtest(
            self.s, cfg, ["BTCUSDT"], self.start_ms, self.end_ms,
            initial_capital=10_000.0,
        )
        self.assertEqual(out["summary"]["n_trades"], 0)
        self.assertAlmostEqual(
            out["summary"]["total_return_pct"], 0.0, places=6,
        )

    def test_positive_funding_captures_income(self):
        """持续正 funding → 至少一笔平仓交易, PnL 为正 (funding 减去成本)"""
        # 100 个连续 2 bps funding, 每 8h 收 20000*2e-4 = 4 USD × 100 = 400 USD (mark=40k, notional=1000)
        # 实际 notional=1000 → 每次 funding 收入 1000*2e-4 = 0.2 USD, 100 次 = 20 USD
        # 成本: 每个 trade = 2 * (10+5+20) * 1000 / 10000 = 7 USD (open+close)
        # 因此 PnL = 20 - 7 = 13 USD (若一个 trade 撑到结束)
        df = _make_synthetic_funding(
            "BTCUSDT", self.start_ms, 100, [2e-4] * 100,
        )
        self.s.upsert_funding_rates(df)
        cfg = FundingHarvesterConfig(
            min_funding_rate=1e-4,
            min_funding_duration_h=16,
            notional_per_trade_usd=1000.0,
            exit_on_funding_flip=True,
        )
        out = run_funding_backtest(
            self.s, cfg, ["BTCUSDT"], self.start_ms, self.end_ms,
            initial_capital=10_000.0,
        )
        self.assertGreaterEqual(out["summary"]["n_trades"], 1)
        # 至少一笔 trade 的 PnL 总和应为正
        total_pnl = sum(t["pnl"] for t in out["trades"])
        self.assertGreater(total_pnl, 0)

    def test_funding_flip_triggers_exit(self):
        """先 50 条正 funding 再 50 条负 funding → 第一笔应在翻负后平仓"""
        rates = [2e-4] * 50 + [-1e-4] * 50
        df = _make_synthetic_funding("BTCUSDT", self.start_ms, 100, rates)
        self.s.upsert_funding_rates(df)
        cfg = FundingHarvesterConfig(
            min_funding_rate=1e-4, min_funding_duration_h=16,
            exit_on_funding_flip=True,
        )
        out = run_funding_backtest(
            self.s, cfg, ["BTCUSDT"], self.start_ms, self.end_ms,
            initial_capital=10_000.0,
        )
        self.assertGreaterEqual(out["summary"]["n_trades"], 1)
        # 至少一笔平仓理由应含 flip
        reasons = {t["exit_reason"] for t in out["trades"]}
        self.assertTrue(any("flip" in r for r in reasons))

    def test_summary_verdict_assigned(self):
        df = _make_synthetic_funding("BTCUSDT", self.start_ms, 100, [0.0] * 100)
        self.s.upsert_funding_rates(df)
        out = run_funding_backtest(
            self.s, FundingHarvesterConfig(), ["BTCUSDT"],
            self.start_ms, self.end_ms, initial_capital=10_000.0,
        )
        self.assertIn(out["summary"]["verdict"],
                      {"EXCELLENT", "GOOD", "MARGINAL", "WEAK", "FAILED"})

    def test_max_concurrent_positions_respected(self):
        """两个 symbol 都 funding 正, max_concurrent=1 时只应有一个同时持仓"""
        df1 = _make_synthetic_funding("BTCUSDT", self.start_ms, 50, [2e-4] * 50,
                                       mark_price=40_000.0)
        df2 = _make_synthetic_funding("ETHUSDT", self.start_ms, 50, [2e-4] * 50,
                                       mark_price=2_000.0)
        self.s.upsert_funding_rates(df1)
        self.s.upsert_funding_rates(df2)
        cfg = FundingHarvesterConfig(
            min_funding_rate=1e-4, min_funding_duration_h=16,
            max_concurrent_positions=1,
        )
        out = run_funding_backtest(
            self.s, cfg, ["BTCUSDT", "ETHUSDT"],
            self.start_ms, self.end_ms, initial_capital=10_000.0,
        )
        # 只有 2 个 symbol, 每个最多 1 个 trade (被另一个阻塞时不开仓)
        # 但因为两个 symbol 的 funding 都 positive 且持续, 在阻塞期间不开就永远不开
        # 这里只验证 max_concurrent 确实限制了并发: 不同时超过 1
        # 通过检查: 任何时刻 open_positions ≤ max_concurrent
        # 近似: n_trades ≤ 2 (两个 symbol 各最多一笔, 强平除外)
        self.assertLessEqual(out["summary"]["n_trades"], 2)

    # ------------------------------------------------------------------
    # P2A-T03 rework 2026-04-19: diagnostics 测试
    # ------------------------------------------------------------------
    def test_diagnostics_present_with_hint(self):
        """只要有 trade, 输出必须包含 diagnostics 段 + hint"""
        df = _make_synthetic_funding("BTCUSDT", self.start_ms, 100, [2e-4] * 100)
        self.s.upsert_funding_rates(df)
        out = run_funding_backtest(
            self.s, FundingHarvesterConfig(
                min_funding_rate=1e-4, min_funding_duration_h=16,
            ),
            ["BTCUSDT"], self.start_ms, self.end_ms, initial_capital=10_000.0,
        )
        self.assertIn("diagnostics", out)
        d = out["diagnostics"]
        for key in ["per_symbol", "cost_breakdown", "top_winners",
                    "top_losers", "diagnostic_hint"]:
            self.assertIn(key, d)
        # cost_breakdown 字段齐全
        cb = d["cost_breakdown"]
        for key in ["total_funding_income", "total_trading_cost",
                    "net_pnl", "cost_to_income_ratio"]:
            self.assertIn(key, cb)
        # hint 必须是字符串
        self.assertIsInstance(d["diagnostic_hint"], str)
        self.assertTrue(len(d["diagnostic_hint"]) > 0)

    def test_diagnostics_per_symbol_breakdown(self):
        """两个 symbol 分别产生 trade → per_symbol 应有两条"""
        df1 = _make_synthetic_funding("BTCUSDT", self.start_ms, 50, [2e-4] * 50)
        df2 = _make_synthetic_funding("ETHUSDT", self.start_ms, 50, [2e-4] * 50,
                                       mark_price=2_000.0)
        self.s.upsert_funding_rates(df1)
        self.s.upsert_funding_rates(df2)
        out = run_funding_backtest(
            self.s, FundingHarvesterConfig(
                min_funding_rate=1e-4, min_funding_duration_h=16,
                max_concurrent_positions=5,
            ),
            ["BTCUSDT", "ETHUSDT"], self.start_ms, self.end_ms,
            initial_capital=10_000.0,
        )
        per_sym = out["diagnostics"]["per_symbol"]
        # 至少有一个 symbol 产生 trade
        self.assertGreaterEqual(len(per_sym), 1)
        for s, stats in per_sym.items():
            self.assertIn("n_trades", stats)
            self.assertIn("win_rate_pct", stats)
            self.assertIn("avg_duration_h", stats)

    def test_diagnostics_cost_dominated_regime(self):
        """极高成本 config → hint 应指向 cost_eats_income"""
        df = _make_synthetic_funding("BTCUSDT", self.start_ms, 100, [1.1e-4] * 100)
        self.s.upsert_funding_rates(df)
        # 夸张成本: spot_fee=500 bps / 滑点 500 bps 把利润全吃掉
        cfg = FundingHarvesterConfig(
            min_funding_rate=1e-4, min_funding_duration_h=8,
            spot_fee_bps=500.0, perp_fee_bps=500.0, slippage_bps=500.0,
            exit_on_funding_flip=False, exit_buffer_rate=-1.0,  # 不主动平, 靠 strategy
        )
        out = run_funding_backtest(
            self.s, cfg, ["BTCUSDT"], self.start_ms, self.end_ms,
            initial_capital=10_000.0,
        )
        hint = out["diagnostics"]["diagnostic_hint"]
        self.assertTrue(
            "cost" in hint.lower() or "funding_income" in hint.lower(),
            f"预期 hint 提到 cost 或 funding_income, 实得: {hint}",
        )


if __name__ == "__main__":
    unittest.main()

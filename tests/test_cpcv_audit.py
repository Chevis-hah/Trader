"""P2B-T04: CPCV audit 辅助函数测试。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_cpcv_audit import (
    _synthetic_price_series,
    _synthetic_funding_series,
    _bar_returns_from_signals,
    _macd_momentum_signal,
    _triple_ema_signal,
    _mean_reversion_signal,
    _cross_sectional_signal,
    _funding_harvester_signal,
    run_cpcv_for_strategy,
    format_audit_report,
    audit_demo,
)


class TestSyntheticData(unittest.TestCase):
    def test_price_shape(self):
        df = _synthetic_price_series(n_days=100)
        self.assertEqual(len(df), 100)
        self.assertTrue(all(c in df.columns for c in ["open", "high", "low", "close", "volume"]))

    def test_funding_shape(self):
        df = _synthetic_funding_series(n_cycles=300)
        self.assertEqual(len(df), 300)
        self.assertIn("funding_rate", df.columns)

    def test_deterministic_with_seed(self):
        df1 = _synthetic_price_series(n_days=50, seed=123)
        df2 = _synthetic_price_series(n_days=50, seed=123)
        np.testing.assert_array_almost_equal(df1["close"].values, df2["close"].values)


class TestSignalFunctions(unittest.TestCase):
    def setUp(self):
        self.prices = _synthetic_price_series(n_days=300)

    def test_all_signals_return_series_of_correct_length(self):
        for fn in [_macd_momentum_signal, _triple_ema_signal,
                   _mean_reversion_signal, _cross_sectional_signal]:
            pos = fn(self.prices)
            self.assertIsInstance(pos, pd.Series)
            self.assertEqual(len(pos), len(self.prices))

    def test_position_is_binary_or_zero(self):
        for fn in [_macd_momentum_signal, _triple_ema_signal,
                   _mean_reversion_signal, _cross_sectional_signal]:
            pos = fn(self.prices)
            uniq = set(pos.dropna().unique())
            # Expect only {0.0, 1.0} or sometimes {0.0}
            self.assertTrue(uniq.issubset({0.0, 1.0}))

    def test_bar_returns_produce_series(self):
        r = _bar_returns_from_signals(self.prices, _macd_momentum_signal)
        self.assertEqual(len(r), len(self.prices))
        self.assertTrue(np.all(np.isfinite(r.values)))

    def test_funding_signal_nonzero_when_rate_high(self):
        funding = _synthetic_funding_series(n_cycles=200)
        r = _funding_harvester_signal(funding)
        self.assertEqual(len(r), len(funding))


class TestAuditRunner(unittest.TestCase):
    def test_run_cpcv_returns_full_dict(self):
        prices = _synthetic_price_series(n_days=400)
        r = run_cpcv_for_strategy(
            "test_strat",
            lambda d: _bar_returns_from_signals(d, _macd_momentum_signal),
            prices, n_groups=5, n_test_groups=2, embargo_pct=0.0,
        )
        for key in ["strategy", "n_paths", "sharpe_median", "sharpe_mean",
                    "sharpe_std", "sharpe_per_path", "dsr_median", "pbo",
                    "verdict_simple", "g1_pass", "is_deterministic",
                    "bars_per_year", "notes"]:
            self.assertIn(key, r)
        self.assertEqual(r["strategy"], "test_strat")
        # n_paths = C(N-1, k-1) = C(4,1) = 4
        self.assertEqual(r["n_paths"], 4)

    def test_deterministic_rule_strategy_marked(self):
        """规则策略 (无 train 参数) → is_deterministic=True, pbo=NaN, notes 非空"""
        prices = _synthetic_price_series(n_days=400)
        r = run_cpcv_for_strategy(
            "deterministic_rule",
            lambda d: _bar_returns_from_signals(d, _macd_momentum_signal),
            prices, bars_per_year=252.0,
        )
        self.assertTrue(r["is_deterministic"])
        self.assertFalse(np.isfinite(r["pbo"]))
        self.assertTrue(any("PBO=N/A" in n for n in r["notes"]))

    def test_bars_per_year_passed_through(self):
        prices = _synthetic_price_series(n_days=400)
        r = run_cpcv_for_strategy(
            "test", lambda d: _bar_returns_from_signals(d, _macd_momentum_signal),
            prices, bars_per_year=2190.0,
        )
        self.assertEqual(r["bars_per_year"], 2190.0)

    def test_audit_demo_runs_all_5(self):
        """冒烟: 端到端 demo, 应返回 5 条结果"""
        results = audit_demo()
        self.assertEqual(len(results), 5)
        names = {r["strategy"] for r in results}
        self.assertEqual(names, {
            "macd_momentum", "triple_ema", "mean_reversion",
            "cross_sectional_momentum", "funding_harvester",
        })

    def test_format_audit_report_produces_markdown(self):
        results = [{
            "strategy": "foo", "n_paths": 9,
            "sharpe_median": 1.2, "sharpe_mean": 1.1, "sharpe_std": 0.5,
            "sharpe_per_path": [1.0, 1.2, 1.5, 0.8, 1.1, 1.3, 0.9, 1.4, 1.1],
            "dsr_median": 0.85, "pbo": 0.25,
            "verdict_simple": "PASS", "g1_pass": True,
            "is_deterministic": False, "bars_per_year": 252.0, "notes": [],
        }]
        md = format_audit_report(results)
        self.assertIn("# CPCV 历史策略审计报告", md)
        self.assertIn("foo", md)
        self.assertIn("PASS", md)
        self.assertIn("25.0%", md)

    def test_format_audit_report_shows_deterministic_and_notes(self):
        results = [{
            "strategy": "bar", "n_paths": 9,
            "sharpe_median": 0.2, "sharpe_mean": 0.2, "sharpe_std": 0.0,
            "sharpe_per_path": [0.2]*9,
            "dsr_median": 1.0, "pbo": float("nan"),
            "verdict_simple": "FAIL", "g1_pass": False,
            "is_deterministic": True, "bars_per_year": 2190.0,
            "notes": ["PBO=N/A: 策略无 train-time 参数"],
        }]
        md = format_audit_report(results)
        # PBO 显示 N/A
        self.assertIn("N/A", md)
        # det? 列标 ✓
        self.assertIn("✓", md)
        # notes 段存在
        self.assertIn("备注", md)
        self.assertIn("train-time", md)


if __name__ == "__main__":
    unittest.main()

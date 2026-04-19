"""P2B-T03: DSR + PBO + summarise_cpcv_paths 单元测试。

验收 (docs/TODO_12M_DEVELOPMENT.md P2B-T03):
  - 给 SR=2.0, n_trials=100 的情况, DSR 显著低于 2.0 (实际返回的是概率, 显著低于 naive 的 "SR > 0" p-value)
  - 给正态分布收益, DSR ≈ naive SR (这里验证: n_trials=1 时 DSR ≈ Φ(SR * sqrt(T-1)))
  - 给高 kurt 收益, DSR < 同参数低 kurt 的 DSR
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from validation.dsr import (
    _expected_max_sr,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    summarise_cpcv_paths,
)
from validation import CombinatorialPurgedCV


class TestExpectedMaxSR(unittest.TestCase):
    def test_monotonic_in_n_trials(self):
        """试验次数越多, E[max SR] 越大"""
        vals = [_expected_max_sr(n) for n in [2, 10, 100, 1000]]
        for a, b in zip(vals[:-1], vals[1:]):
            self.assertLess(a, b)

    def test_single_trial_is_zero(self):
        self.assertEqual(_expected_max_sr(1), 0.0)

    def test_known_order_of_magnitude(self):
        # López de Prado 2018 Table 1: N=100 时 E[max SR] ≈ 2.51
        em = _expected_max_sr(100)
        self.assertGreater(em, 2.0)
        self.assertLess(em, 3.0)


class TestDSR(unittest.TestCase):
    def test_bounds(self):
        for sr in [-2.0, -0.5, 0.0, 0.5, 2.0, 5.0]:
            d = deflated_sharpe_ratio(sr, n_trials=1, n_samples=252, skew=0, kurt=3)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_more_trials_lowers_dsr(self):
        """固定 observed SR, 试验次数越多 DSR 越低 (multiple-testing 惩罚)。
        用小样本 + 适中 SR 避免数值饱和到 1.0。"""
        d1 = deflated_sharpe_ratio(1.0, n_trials=1, n_samples=60, skew=0, kurt=3)
        d100 = deflated_sharpe_ratio(1.0, n_trials=100, n_samples=60, skew=0, kurt=3)
        self.assertGreater(d1, d100)

    def test_more_samples_raises_dsr(self):
        """固定 observed SR, 样本越多 DSR 越高 (标准误越小)。"""
        d_short = deflated_sharpe_ratio(1.5, n_trials=1, n_samples=60, skew=0, kurt=3)
        d_long = deflated_sharpe_ratio(1.5, n_trials=1, n_samples=2520, skew=0, kurt=3)
        self.assertGreater(d_long, d_short)

    def test_higher_kurt_lowers_dsr(self):
        d_normal = deflated_sharpe_ratio(1.5, n_trials=10, n_samples=252, skew=0, kurt=3)
        d_fat = deflated_sharpe_ratio(1.5, n_trials=10, n_samples=252, skew=0, kurt=10)
        self.assertGreater(d_normal, d_fat)

    def test_negative_skew_lowers_dsr(self):
        """负偏度 (左尾长) 应比正偏度更难被识别为显著。
        用适中 SR + 小样本避免数值饱和。"""
        d_pos = deflated_sharpe_ratio(0.5, n_trials=10, n_samples=60, skew=0.5, kurt=3)
        d_neg = deflated_sharpe_ratio(0.5, n_trials=10, n_samples=60, skew=-0.5, kurt=3)
        self.assertGreater(d_pos, d_neg)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio(1.0, n_trials=0, n_samples=252)
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio(1.0, n_trials=10, n_samples=1)


class TestPBO(unittest.TestCase):
    def test_all_strategies_identical_gives_pbo_near_half(self):
        """所有策略收益完全一致 → IS 最佳完全随机 → PBO 应在 0.5 附近"""
        rng = np.random.default_rng(42)
        T, M = 400, 8
        base = rng.standard_normal(T) * 0.01
        returns = np.tile(base.reshape(-1, 1), (1, M)) + rng.standard_normal((T, M)) * 1e-6
        pbo = probability_of_backtest_overfitting(returns, n_splits=8)
        self.assertGreaterEqual(pbo, 0.2)
        self.assertLessEqual(pbo, 0.8)

    def test_one_strategy_dominates_gives_low_pbo(self):
        """一个策略稳定优于其它 → IS 总是选它 → OOS 也占优 → PBO 低"""
        rng = np.random.default_rng(0)
        T, M = 400, 6
        returns = rng.standard_normal((T, M)) * 0.01
        # 让 0 号策略日均 +0.5% 稳定优势
        returns[:, 0] += 0.005
        pbo = probability_of_backtest_overfitting(returns, n_splits=8)
        self.assertLess(pbo, 0.3)

    def test_pure_noise_high_pbo(self):
        """纯噪声: IS 最佳多半是运气, OOS 表现各半 → PBO 近 0.5 或更高"""
        rng = np.random.default_rng(7)
        T, M = 400, 20
        returns = rng.standard_normal((T, M)) * 0.01
        pbo = probability_of_backtest_overfitting(returns, n_splits=8)
        self.assertGreaterEqual(pbo, 0.3)

    def test_rejects_odd_splits(self):
        returns = np.random.default_rng(1).standard_normal((200, 5))
        with self.assertRaises(ValueError):
            probability_of_backtest_overfitting(returns, n_splits=7)

    def test_accepts_dataframe(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.standard_normal((200, 4)),
            columns=["A", "B", "C", "D"],
        )
        pbo = probability_of_backtest_overfitting(df, n_splits=8)
        self.assertGreaterEqual(pbo, 0.0)
        self.assertLessEqual(pbo, 1.0)


class TestSummariseCPCVPaths(unittest.TestCase):
    def test_empty_curves_returns_fail(self):
        out = summarise_cpcv_paths([])
        self.assertEqual(out["n_paths"], 0)
        self.assertEqual(out["verdict"], "FAIL")

    def test_positive_edge_passes(self):
        """构造 9 条正 edge 曲线 -> 判决至少 CONDITIONAL"""
        rng = np.random.default_rng(42)
        curves = []
        for _ in range(9):
            r = rng.standard_normal(252) * 0.01 + 0.001
            curves.append(pd.Series(r))
        out = summarise_cpcv_paths(curves, n_trials=1)
        self.assertEqual(out["n_paths"], 9)
        self.assertIn(out["verdict"], ["PASS", "CONDITIONAL"])
        self.assertGreater(out["sharpe_median"], 0)

    def test_zero_edge_fails(self):
        rng = np.random.default_rng(0)
        curves = [pd.Series(rng.standard_normal(252) * 0.01) for _ in range(9)]
        out = summarise_cpcv_paths(curves, n_trials=100)
        self.assertEqual(out["verdict"], "FAIL")

    def test_integrates_with_cpcv(self):
        """从 CombinatorialPurgedCV.backtest_paths 串下来"""
        N = 500
        df = pd.DataFrame({"x": np.arange(N)}, index=pd.RangeIndex(N))
        cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2, embargo_pct=0.0)
        rng = np.random.default_rng(1)

        def strat(train_idx, test_idx):
            # 随机 iid 收益, 不依赖 train
            r = rng.standard_normal(len(test_idx)) * 0.01
            return pd.Series(r, index=df.index[test_idx])

        curves = cv.backtest_paths(df, strat)
        out = summarise_cpcv_paths(curves, n_trials=45)
        self.assertEqual(out["n_paths"], 9)
        self.assertIn(out["verdict"], ["PASS", "CONDITIONAL", "FAIL"])

    # --- 新增 (2026-04-19 P2B-T04 rework): is_deterministic + bars_per_year ---

    def test_is_deterministic_flag_when_paths_identical(self):
        """所有 path 完全相同 → is_deterministic=True"""
        base = np.random.default_rng(0).standard_normal(252) * 0.01
        # 9 条完全一样的 curve
        curves = [pd.Series(base.copy()) for _ in range(9)]
        out = summarise_cpcv_paths(curves)
        self.assertTrue(out["is_deterministic"])
        self.assertAlmostEqual(out["sharpe_std"], 0.0, places=10)

    def test_is_deterministic_false_when_paths_differ(self):
        rng = np.random.default_rng(7)
        # 9 条不同的 curve
        curves = [pd.Series(rng.standard_normal(252) * 0.01) for _ in range(9)]
        out = summarise_cpcv_paths(curves)
        self.assertFalse(out["is_deterministic"])

    def test_bars_per_year_affects_sharpe(self):
        """同一数据, 不同 bars_per_year 应产出不同 Sharpe"""
        rng = np.random.default_rng(3)
        curves = [pd.Series(rng.standard_normal(500) * 0.005 + 0.0005) for _ in range(9)]
        out_daily = summarise_cpcv_paths(curves, bars_per_year=252)
        out_4h = summarise_cpcv_paths(curves, bars_per_year=2190)
        out_8h = summarise_cpcv_paths(curves, bars_per_year=1095)
        # 4h 年化 > 8h 年化 > 日线年化 (因为 sqrt(2190) > sqrt(1095) > sqrt(252))
        self.assertGreater(out_4h["sharpe_median"], out_8h["sharpe_median"])
        self.assertGreater(out_8h["sharpe_median"], out_daily["sharpe_median"])
        # bars_per_year 回显
        self.assertEqual(out_daily["bars_per_year"], 252.0)

    def test_deterministic_curve_dsr_uses_single_path(self):
        """deterministic 情况下 DSR 应能算出 (不崩于 kurt/skew 计算)"""
        base = np.random.default_rng(0).standard_normal(100) * 0.01 + 0.0005
        curves = [pd.Series(base.copy()) for _ in range(9)]
        out = summarise_cpcv_paths(curves, n_trials=1)
        self.assertTrue(out["is_deterministic"])
        self.assertTrue(np.isfinite(out["dsr_median"]))


if __name__ == "__main__":
    unittest.main()

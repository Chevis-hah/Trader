"""
测试套件 — 覆盖所有策略 + ML + Walk-forward

python -m pytest tests/ -v
python tests/test_all.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np


class TestMeanReversionStrategy(unittest.TestCase):
    def setUp(self):
        from alpha.mean_reversion_strategy import MeanReversionStrategy, MeanReversionConfig
        self.strategy = MeanReversionStrategy(MeanReversionConfig())

    def _row(self, **kw):
        d = {"close": 50000.0, "adx_14": 18.0, "natr_20": 0.02,
             "zscore_240": -2.0, "rsi_14": 28.0, "ret_5": -0.04,
             "keltner_position": 0.1, "mfi_14": 20.0,
             "bb_width_240": 0.05, "rel_volume_20": 1.0,
             "amihud_60": 0.0001, "bb_position_240": 0.1,
             "ma_dev_120": -0.03}
        d.update(kw)
        return pd.Series(d)

    def test_entry_on_oversold(self):
        row = self._row(zscore_240=-2.0, rsi_14=28.0)
        self.assertTrue(self.strategy.should_enter(row, self._row(), {}))

    def test_no_entry_strong_trend(self):
        row = self._row(adx_14=30.0)
        self.assertFalse(self.strategy.should_enter(row, self._row(), {}))

    def test_no_entry_not_oversold(self):
        row = self._row(zscore_240=0.5, rsi_14=55.0, ret_5=0.01,
                        keltner_position=0.6, mfi_14=50.0)
        self.assertFalse(self.strategy.should_enter(row, self._row(), {}))

    def test_exit_on_mean_reversion(self):
        row = self._row(zscore_240=0.5)
        pos = {"entry_price": 48000, "atr_at_entry": 1000,
               "highest_since_entry": 50000, "stop_loss": 46000}
        ok, reason = self.strategy.check_exit(row, self._row(), pos, 5)
        self.assertTrue(ok)
        self.assertEqual(reason, "mean_reversion_complete")

    def test_position_sizing(self):
        row = self._row()
        qty, stop = self.strategy.calc_position(10000.0, 50000.0, row)
        self.assertGreater(qty, 0)
        self.assertLess(qty * 50000.0, 10000.0 * 0.36)


class TestMACDOverextensionFilter(unittest.TestCase):
    def setUp(self):
        from alpha.macd_momentum_strategy import MACDMomentumStrategy, MACDMomentumStrategyConfig
        self.strategy = MACDMomentumStrategy(MACDMomentumStrategyConfig(
            regime_gate_enabled=True, regime_adx_min=22.0,
            overextension_filter=True, max_zscore_entry=1.5))

    def _row(self, **kw):
        d = {"close": 50000, "ema_50": 48000, "adx_14": 28, "rsi_14": 55,
             "macd_hist": 100, "natr_20": 0.02, "rel_volume_20": 1.2,
             "daily_trend_ok": 1.0, "macd_line": 50, "macd_signal": 40,
             "zscore_240": 0.5, "ma_dev_120": 0.02, "keltner_position": 0.6}
        d.update(kw)
        return pd.Series(d)

    def test_blocks_overextended(self):
        row = self._row(zscore_240=2.0, macd_hist=100)
        prev = self._row(macd_hist=-10)
        self.assertFalse(self.strategy.should_enter(row, prev))

    def test_allows_normal(self):
        row = self._row(zscore_240=0.5, macd_hist=100)
        prev = self._row(macd_hist=-10)
        self.assertTrue(self.strategy.should_enter(row, prev))

    def test_blocks_high_ma_dev(self):
        row = self._row(ma_dev_120=0.08, macd_hist=100)
        prev = self._row(macd_hist=-10)
        self.assertFalse(self.strategy.should_enter(row, prev))


class TestTripleEMAOverextensionFilter(unittest.TestCase):
    def setUp(self):
        from alpha.triple_ema_strategy import TripleEMAStrategy, TripleEMAStrategyConfig
        self.strategy = TripleEMAStrategy(TripleEMAStrategyConfig(
            regime_gate_enabled=True, regime_adx_min=25.0,
            overextension_filter=True))

    def _row(self, **kw):
        d = {"close": 50000, "ema_8": 50500, "ema_21": 49500, "ema_55": 48000,
             "adx_14": 30, "rsi_14": 55, "natr_20": 0.02, "macd_hist": 50,
             "rel_volume_20": 1.2, "daily_trend_ok": 1.0,
             "zscore_240": 0.5, "ma_dev_120": 0.02, "keltner_position": 0.6}
        d.update(kw)
        return pd.Series(d)

    def test_blocks_high_zscore(self):
        row = self._row(zscore_240=2.0, close=49600)
        prev = self._row(close=49400, ema_21=49500, ema_8=49500)
        self.assertFalse(self.strategy.should_enter(row, prev))


class TestStrategyRegistry(unittest.TestCase):
    def test_available(self):
        from alpha.strategy_registry import available_strategies
        names = available_strategies()
        self.assertIn("mean_reversion", names)
        self.assertIn("macd_momentum", names)
        self.assertIn("triple_ema", names)

    def test_build_mean_reversion(self):
        from alpha.strategy_registry import build_strategy
        s = build_strategy(explicit_name="mean_reversion")
        self.assertEqual(s.name, "mean_reversion")

    def test_dict_override(self):
        from alpha.strategy_registry import build_strategy
        s = build_strategy(config={"max_adx": 30.0}, explicit_name="mean_reversion")
        self.assertEqual(s.cfg.max_adx, 30.0)


class TestMLModel(unittest.TestCase):
    def test_import(self):
        from alpha.ml_lightgbm import LightGBMAlphaModel, CORE_FEATURES
        self.assertGreater(len(CORE_FEATURES), 10)

    def test_feature_selection(self):
        from alpha.ml_lightgbm import LightGBMAlphaModel
        model = LightGBMAlphaModel(use_core_only=True)
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "close": np.random.randn(n).cumsum() + 50000,
            "ma_dev_240": np.random.randn(n) * 0.05,
            "zscore_240": np.random.randn(n),
            "rsi_14": np.random.uniform(20, 80, n),
            "mfi_14": np.random.uniform(20, 80, n),
            "ret_5": np.random.randn(n) * 0.02,
            "bb_position_240": np.random.uniform(0, 1, n),
            "close_vs_sma_100": np.random.randn(n) * 0.05,
            "keltner_position": np.random.uniform(0, 1, n),
            "mom_accel_5": np.random.randn(n) * 0.01,
            "amihud_60": np.abs(np.random.randn(n) * 0.0001),
            "taker_buy_ratio_120": np.random.uniform(0.4, 0.6, n),
            "price_vol_corr_120": np.random.randn(n) * 0.3,
            "natr_20": np.abs(np.random.randn(n) * 0.02) + 0.01,
            "natr_60": np.abs(np.random.randn(n) * 0.02) + 0.01,
            "adx_14": np.random.uniform(10, 40, n),
        })
        selected = model._select_features(df)
        self.assertGreaterEqual(len(selected), 10)


class TestEnhancedExit(unittest.TestCase):
    def test_partial_profit(self):
        from alpha.enhanced_exit import EnhancedExitMixin, EnhancedExitConfig
        m = EnhancedExitMixin()
        cfg = EnhancedExitConfig()
        row = pd.Series({"close": 53500.0, "natr_20": 0.02})
        pos = {"entry_price": 50000, "atr_at_entry": 1000,
               "highest_since_entry": 54000, "stop_loss": 48000}
        ok, reason, pct = m.check_enhanced_exit(row, None, pos, 10, cfg)
        self.assertTrue(ok)
        self.assertEqual(reason, "partial_profit_1")

    def test_breakeven(self):
        from alpha.enhanced_exit import EnhancedExitMixin, EnhancedExitConfig
        m = EnhancedExitMixin()
        cfg = EnhancedExitConfig()
        row = pd.Series({"close": 52500.0, "natr_20": 0.02})
        pos = {"entry_price": 50000, "atr_at_entry": 1000,
               "highest_since_entry": 52500, "stop_loss": 48000}
        m.check_enhanced_exit(row, None, pos, 10, cfg)
        self.assertGreaterEqual(pos["stop_loss"], 50000 * 1.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)

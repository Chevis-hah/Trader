"""
基础测试框架 — 策略核心逻辑验证

用法:
    python -m pytest tests/ -v
    python tests/test_strategies.py   # 直接运行也可以
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np


class TestMACDMomentumStrategy(unittest.TestCase):
    """MACD Momentum 策略核心逻辑测试"""

    def setUp(self):
        from alpha.macd_momentum_strategy import MACDMomentumStrategy, MACDMomentumStrategyConfig
        self.cfg = MACDMomentumStrategyConfig(
            regime_gate_enabled=True,
            regime_adx_min=22.0,
        )
        self.strategy = MACDMomentumStrategy(self.cfg)

    def _make_row(self, **kwargs):
        defaults = {
            "close": 50000.0,
            "ema_50": 48000.0,
            "adx_14": 30.0,
            "rsi_14": 55.0,
            "macd_hist": 100.0,
            "natr_20": 0.02,
            "rel_volume_20": 1.2,
            "daily_trend_ok": 1.0,
            "macd_line": 50.0,
            "macd_signal": 40.0,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_regime_gate_blocks_low_adx(self):
        """ADX < regime_adx_min 时不应入场"""
        row = self._make_row(adx_14=18.0, macd_hist=100.0)
        prev = self._make_row(macd_hist=-10.0)
        self.assertFalse(self.strategy.should_enter(row, prev))

    def test_regime_gate_allows_high_adx(self):
        """ADX >= regime_adx_min 时应允许入场"""
        row = self._make_row(adx_14=25.0, macd_hist=100.0)
        prev = self._make_row(macd_hist=-10.0)
        self.assertTrue(self.strategy.should_enter(row, prev))

    def test_macd_cross_required(self):
        """MACD 柱必须从负翻正"""
        row = self._make_row(adx_14=25.0, macd_hist=100.0)
        prev = self._make_row(macd_hist=50.0)  # 前一根也是正的
        self.assertFalse(self.strategy.should_enter(row, prev))

    def test_no_entry_below_ema50(self):
        """价格在 EMA50 下方不应入场"""
        row = self._make_row(close=47000.0, ema_50=48000.0, adx_14=25.0, macd_hist=100.0)
        prev = self._make_row(macd_hist=-10.0)
        self.assertFalse(self.strategy.should_enter(row, prev))

    def test_position_sizing(self):
        """仓位计算：不超过 max_position_pct"""
        row = self._make_row()
        qty, stop = self.strategy.calc_position(10000.0, 50000.0, row)
        position_value = qty * 50000.0
        self.assertLessEqual(position_value, 10000.0 * self.cfg.max_position_pct * 1.01)
        self.assertGreater(stop, 0)

    def test_adaptive_trailing_stop(self):
        """自适应追踪止损：盈利时应更宽松"""
        row = self._make_row(close=55000.0, natr_20=0.02)
        position = {
            "entry_price": 50000.0,
            "atr_at_entry": 1000.0,  # 2% of 50000
            "highest_since_entry": 56000.0,
            "stop_loss": 48000.0,
        }
        should_exit, reason = self.strategy.check_exit(row, self._make_row(), position, 5)
        # 55000 盈利 5 ATR，追踪应该很宽，不应被触出
        self.assertFalse(should_exit, f"不应在大盈利时被追踪止损触出: {reason}")

    def test_cooldown_after_stop(self):
        """止损后冷却期应更长"""
        state = {"bar_index": 100}
        self.strategy.on_trade_closed(state, 100, "adaptive_trailing_stop")
        self.assertEqual(
            state["cooldown_until_bar"],
            100 + self.cfg.cooldown_bars_after_stop
        )

    def test_cooldown_after_signal_exit(self):
        """信号出场后冷却期应正常"""
        state = {"bar_index": 100}
        self.strategy.on_trade_closed(state, 100, "macd_cross")
        self.assertEqual(
            state["cooldown_until_bar"],
            100 + self.cfg.cooldown_bars
        )


class TestTripleEMAStrategy(unittest.TestCase):
    """Triple EMA 策略核心逻辑测试"""

    def setUp(self):
        from alpha.triple_ema_strategy import TripleEMAStrategy, TripleEMAStrategyConfig
        self.cfg = TripleEMAStrategyConfig(
            regime_gate_enabled=True,
            regime_adx_min=25.0,
        )
        self.strategy = TripleEMAStrategy(self.cfg)

    def _make_row(self, **kwargs):
        defaults = {
            "close": 50000.0,
            "ema_8": 50500.0,
            "ema_21": 49500.0,
            "ema_55": 48000.0,
            "adx_14": 30.0,
            "rsi_14": 55.0,
            "natr_20": 0.02,
            "macd_hist": 50.0,
            "rel_volume_20": 1.2,
            "daily_trend_ok": 1.0,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_regime_gate_blocks(self):
        """ADX < 25 时不应入场"""
        row = self._make_row(adx_14=20.0, close=49600.0)
        prev = self._make_row(close=49400.0, ema_21=49500.0, ema_8=49500.0)
        self.assertFalse(self.strategy.should_enter(row, prev))

    def test_ema_alignment_required(self):
        """EMA 必须多头排列"""
        row = self._make_row(ema_8=48000.0, ema_21=49000.0, ema_55=50000.0, adx_14=30.0)
        prev = self._make_row()
        self.assertFalse(self.strategy.should_enter(row, prev))

    def test_slippage_correction(self):
        """校正后的滑点应为 21bps"""
        self.assertEqual(self.cfg.slippage_pct, 0.0021)


class TestEnhancedExit(unittest.TestCase):
    """增强出场逻辑测试"""

    def setUp(self):
        from alpha.enhanced_exit import EnhancedExitMixin, EnhancedExitConfig
        self.mixin = EnhancedExitMixin()
        self.cfg = EnhancedExitConfig()

    def _make_row(self, **kwargs):
        defaults = {"close": 55000.0, "natr_20": 0.02}
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_partial_profit_1(self):
        """盈利 > 3 ATR 应触发第一批止盈"""
        row = self._make_row(close=53500.0)
        position = {
            "entry_price": 50000.0,
            "atr_at_entry": 1000.0,
            "highest_since_entry": 54000.0,
            "stop_loss": 48000.0,
        }
        should_exit, reason, qty_pct = self.mixin.check_enhanced_exit(
            row, None, position, 10, self.cfg
        )
        self.assertTrue(should_exit)
        self.assertEqual(reason, "partial_profit_1")
        self.assertEqual(qty_pct, 0.3)

    def test_breakeven_stop(self):
        """盈利 > 2 ATR 后止损应拉到保本"""
        row = self._make_row(close=52500.0)
        position = {
            "entry_price": 50000.0,
            "atr_at_entry": 1000.0,
            "highest_since_entry": 52500.0,
            "stop_loss": 48000.0,
        }
        # 触发检查 (可能不触发退出，但止损应更新)
        self.mixin.check_enhanced_exit(row, None, position, 10, self.cfg)
        self.assertGreaterEqual(position["stop_loss"], 50000.0 * 1.001)


class TestStrategyRegistry(unittest.TestCase):
    """策略注册表测试"""

    def test_available_strategies(self):
        from alpha.strategy_registry import available_strategies
        names = available_strategies()
        self.assertIn("triple_ema", names)
        self.assertIn("macd_momentum", names)
        self.assertIn("regime", names)
        self.assertIn("grid", names)

    def test_build_triple_ema(self):
        from alpha.strategy_registry import build_strategy
        strategy = build_strategy(explicit_name="triple_ema")
        self.assertEqual(strategy.name, "triple_ema")

    def test_build_macd(self):
        from alpha.strategy_registry import build_strategy
        strategy = build_strategy(explicit_name="macd_momentum")
        self.assertEqual(strategy.name, "macd_momentum")

    def test_dict_override(self):
        from alpha.strategy_registry import build_strategy
        overrides = {"regime_gate_enabled": False, "min_adx": 30.0}
        strategy = build_strategy(config=overrides, explicit_name="macd_momentum")
        self.assertEqual(strategy.cfg.min_adx, 30.0)
        self.assertFalse(strategy.cfg.regime_gate_enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)

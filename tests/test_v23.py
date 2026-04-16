"""
Trader v2.3 冒烟测试

运行:
  python -m pytest tests/test_v23.py -v
  或
  python tests/test_v23.py

验证:
  1. 所有新策略类能实例化
  2. 配置文件能正确加载
  3. 因子计算不崩溃
  4. 已删除的模块确实不再被引用
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# 确保可以 import 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


class TestStrategyImports(unittest.TestCase):
    """验证所有保留的策略都能 import"""

    def test_macd_momentum_imports(self):
        from alpha.macd_momentum_strategy import (
            MACDMomentumStrategy,
            MACDMomentumConfig,
        )
        strat = MACDMomentumStrategy(MACDMomentumConfig())
        self.assertIsNotNone(strat)

    def test_triple_ema_imports(self):
        from alpha.triple_ema_strategy import (
            TripleEMAStrategy,
            TripleEMAConfig,
        )
        strat = TripleEMAStrategy(TripleEMAConfig())
        self.assertIsNotNone(strat)

    def test_mean_reversion_imports(self):
        from alpha.mean_reversion_strategy import (
            MeanReversionStrategy,
            MeanReversionConfig,
        )
        strat = MeanReversionStrategy(MeanReversionConfig())
        self.assertIsNotNone(strat)

    def test_cross_sectional_imports(self):
        from alpha.cross_sectional_momentum import (
            CrossSectionalMomentumStrategy,
            CrossSectionalConfig,
        )
        strat = CrossSectionalMomentumStrategy(CrossSectionalConfig())
        self.assertIsNotNone(strat)


class TestDeletedModulesGone(unittest.TestCase):
    """验证已删除的模块确实不可 import (或已移动到 archive/)"""

    def test_regime_adaptive_removed(self):
        alpha_dir = ROOT / "alpha"
        if not alpha_dir.exists():
            self.skipTest("alpha/ 目录不存在 (尚未迁移)")
        self.assertFalse(
            (alpha_dir / "regime_adaptive_strategy.py").exists(),
            "regime_adaptive_strategy.py 应已被 cleanup_v23.sh 归档",
        )

    def test_regime_allocator_v2_removed(self):
        alpha_dir = ROOT / "alpha"
        if not alpha_dir.exists():
            self.skipTest("alpha/ 目录不存在")
        self.assertFalse(
            (alpha_dir / "regime_allocator_v2.py").exists(),
            "regime_allocator_v2.py 应已被归档",
        )


class TestMACDMomentumLogic(unittest.TestCase):
    """MACD 策略规则回滚验证"""

    def setUp(self):
        from alpha.macd_momentum_strategy import (
            MACDMomentumStrategy,
            MACDMomentumConfig,
        )
        self.strat = MACDMomentumStrategy(MACDMomentumConfig())

    def test_deleted_params_not_in_config(self):
        """验证 v2.2 的无效参数确实不在 config 中"""
        cfg = self.strat.cfg
        self.assertFalse(hasattr(cfg, "min_rsi"))
        self.assertFalse(hasattr(cfg, "max_rsi"))
        self.assertFalse(hasattr(cfg, "max_holding_bars"))
        self.assertFalse(hasattr(cfg, "zscore_threshold"))
        self.assertFalse(hasattr(cfg, "ma_dev_threshold"))
        self.assertFalse(hasattr(cfg, "keltner_threshold"))

    def test_basic_macd_cross_entry(self):
        """MACD 金叉 + ADX 达标应入场"""
        row = pd.Series({
            "macd": 0.5, "macd_signal": 0.3,
            "adx_14": 25.0,
        })
        prev = pd.Series({
            "macd": 0.1, "macd_signal": 0.2,
            "adx_14": 24.0,
        })
        state = {"bar_index": 100, "cooldown_until_bar": 0}
        self.assertTrue(self.strat.should_enter(row, prev, state))

    def test_adx_filter(self):
        """ADX 不达标不入场"""
        row = pd.Series({
            "macd": 0.5, "macd_signal": 0.3,
            "adx_14": 15.0,   # 低于 20
        })
        prev = pd.Series({
            "macd": 0.1, "macd_signal": 0.2,
            "adx_14": 14.0,
        })
        state = {"bar_index": 100, "cooldown_until_bar": 0}
        self.assertFalse(self.strat.should_enter(row, prev, state))


class TestMeanReversionTrendAvoidance(unittest.TestCase):
    """验证 v2.3 新增的趋势回避门"""

    def setUp(self):
        from alpha.mean_reversion_strategy import (
            MeanReversionStrategy,
            MeanReversionConfig,
        )
        self.strat = MeanReversionStrategy(MeanReversionConfig())

    def test_adx_too_high_blocks_entry(self):
        """ADX > 28 应禁止入场，即使其他条件满足"""
        row = pd.Series({
            "close": 100.0, "bb_lower": 105.0,
            "rsi_14": 20.0, "adx_14": 35.0,
        })
        prev = pd.Series({"close": 102.0})
        state = {"bar_index": 100, "cooldown_until_bar": 0}
        self.assertFalse(self.strat.should_enter(row, prev, state))

    def test_monotonic_up_blocks_entry(self):
        """连续 20 bar 单调上涨应禁止入场"""
        state = {"bar_index": 100, "cooldown_until_bar": 0}
        # 模拟 21 个单调上涨的 close
        for i in range(21):
            row = pd.Series({
                "close": 100.0 + i * 0.5,
                "bb_lower": 90.0,
                "rsi_14": 20.0,
                "adx_14": 20.0,
            })
            prev = pd.Series({"close": 99.0 + i * 0.5})
            # 预先喂数据
            self.strat._check_trend_avoidance(row, state)

        # 现在应该被回避
        final_row = pd.Series({
            "close": 110.0, "bb_lower": 115.0,
            "rsi_14": 20.0, "adx_14": 22.0,   # ADX < 28 但单调性应被检测
        })
        prev = pd.Series({"close": 109.5})
        self.assertFalse(self.strat.should_enter(final_row, prev, state))

    def test_normal_range_allows_entry(self):
        """震荡市 + 跌破下轨 + RSI 超卖 → 入场"""
        state = {"bar_index": 100, "cooldown_until_bar": 0}
        # 喂一些震荡的 close
        np.random.seed(42)
        base = 100.0
        for _ in range(25):
            noise = np.random.randn() * 2
            row = pd.Series({
                "close": base + noise, "bb_lower": 99.0,
                "rsi_14": 50.0, "adx_14": 15.0,
            })
            prev = pd.Series({"close": base})
            self.strat._check_trend_avoidance(row, state)

        # 最后一次真实入场信号
        final_row = pd.Series({
            "close": 97.0, "bb_lower": 98.0,
            "rsi_14": 25.0, "adx_14": 15.0,
        })
        prev = pd.Series({"close": 99.0})
        self.assertTrue(self.strat.should_enter(final_row, prev, state))


class TestCrossSectionalFactors(unittest.TestCase):
    """验证横截面因子计算正确"""

    def setUp(self):
        from alpha.cross_sectional_momentum import (
            CrossSectionalMomentumStrategy,
            CrossSectionalConfig,
        )
        self.strat = CrossSectionalMomentumStrategy(CrossSectionalConfig())

    def test_factor_computation(self):
        """正常数据应返回所有 5 个因子"""
        np.random.seed(0)
        n = 60
        prices = 100.0 * np.cumprod(1 + np.random.randn(n) * 0.02)
        volumes = np.abs(np.random.randn(n)) * 1_000_000 + 500_000

        klines = pd.DataFrame({
            "close": prices,
            "volume": volumes,
        })

        factors = self.strat.compute_factors(klines, as_of_idx=n - 1)
        self.assertIsNotNone(factors)
        self.assertIn("momentum_30d", factors)
        self.assertIn("reversal_7d", factors)
        self.assertIn("volatility_30d", factors)
        self.assertIn("liquidity_amihud", factors)
        self.assertIn("size_proxy", factors)

    def test_insufficient_data_returns_none(self):
        """数据不足应返回 None"""
        klines = pd.DataFrame({
            "close": [100.0, 101.0, 99.0],
            "volume": [1e6, 1.1e6, 0.9e6],
        })
        factors = self.strat.compute_factors(klines, as_of_idx=2)
        self.assertIsNone(factors)

    def test_portfolio_construction(self):
        """组合构建: 做多 top quintile, 做空 bottom quintile"""
        # 构造 20 个模拟币种, momentum 各不相同
        factors_by_symbol = {}
        for i in range(20):
            factors_by_symbol[f"COIN{i:02d}"] = {
                "momentum_30d": i * 0.01,    # 从 0 到 0.19
                "reversal_7d": 0.0,
                "volatility_30d": 0.02,
                "liquidity_amihud": 0.001,
                "size_proxy": 5.0,
            }

        weights = self.strat.rank_and_build_portfolio(factors_by_symbol)
        self.assertGreater(len(weights), 0)

        # 应该有做多和做空
        longs = {s: w for s, w in weights.items() if w > 0}
        shorts = {s: w for s, w in weights.items() if w < 0}
        self.assertGreater(len(longs), 0, "应该有做多仓位")
        self.assertGreater(len(shorts), 0, "应该有做空仓位")


class TestConfigFile(unittest.TestCase):
    """验证 settings.yaml 清理正确"""

    def test_config_loads(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("pyyaml 未安装")

        cfg_path = ROOT / "config" / "settings.yaml"
        if not cfg_path.exists():
            self.skipTest(f"{cfg_path} 不存在")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        # 验证 1h 已删除
        intervals = cfg.get("universe", {}).get("intervals", [])
        self.assertNotIn("1h", intervals, "1h 应已删除")

        # 验证 regime 策略已删除
        strategies = cfg.get("strategy", {})
        self.assertNotIn("regime", strategies, "regime 策略应已删除")

        # 验证滑点修正
        slip = cfg.get("execution", {}).get("slippage_bps", {})
        if "BTCUSDT" in slip:
            self.assertGreaterEqual(
                slip["BTCUSDT"], 20,
                "BTC 滑点应 >= 20bps (phase1 实测 20.89)",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

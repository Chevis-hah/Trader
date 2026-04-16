"""
v2.2 修改验证

python scripts/validate_v22.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(name, ok, detail=""):
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    print("=" * 60)
    print("  🔍 v2.2 全量验证")
    print("=" * 60)
    ok = True

    print("\n[1] 策略注册")
    try:
        from alpha.strategy_registry import available_strategies, build_strategy
        names = available_strategies()
        ok &= check("注册表", "mean_reversion" in names and len(names) >= 4, f"{names}")
    except Exception as e:
        ok &= check("注册表", False, str(e))

    print("\n[2] Mean Reversion 策略")
    try:
        from alpha.mean_reversion_strategy import MeanReversionStrategy, MeanReversionConfig
        cfg = MeanReversionConfig()
        ok &= check("zscore 入场门槛", cfg.zscore_entry <= -1.0, f"{cfg.zscore_entry}")
        ok &= check("ADX 上限", cfg.max_adx <= 25, f"{cfg.max_adx}")
        ok &= check("滑点校正", cfg.slippage_pct >= 0.002, f"{cfg.slippage_pct}")
        s = MeanReversionStrategy(cfg)
        ok &= check("实例化", s.name == "mean_reversion")
    except Exception as e:
        ok &= check("Mean Reversion", False, str(e))

    print("\n[3] MACD v2.2 过滤器")
    try:
        from alpha.macd_momentum_strategy import MACDMomentumStrategy, MACDMomentumStrategyConfig
        cfg = MACDMomentumStrategyConfig()
        ok &= check("过度拉伸过滤", cfg.overextension_filter, "enabled")
        ok &= check("zscore 上限", cfg.max_zscore_entry <= 2.0, f"{cfg.max_zscore_entry}")
        ok &= check("ma_dev 上限", cfg.max_ma_dev_entry <= 0.06, f"{cfg.max_ma_dev_entry}")
    except Exception as e:
        ok &= check("MACD v2.2", False, str(e))

    print("\n[4] Triple EMA v2.2 过滤器")
    try:
        from alpha.triple_ema_strategy import TripleEMAStrategy, TripleEMAStrategyConfig
        cfg = TripleEMAStrategyConfig()
        ok &= check("过度拉伸过滤", cfg.overextension_filter, "enabled")
    except Exception as e:
        ok &= check("Triple EMA v2.2", False, str(e))

    print("\n[5] ML LightGBM 模型")
    try:
        from alpha.ml_lightgbm import LightGBMAlphaModel, CORE_FEATURES
        ok &= check("核心因子数", len(CORE_FEATURES) >= 12, f"{len(CORE_FEATURES)}")
        model = LightGBMAlphaModel()
        ok &= check("模型实例化", model.model is None, "ready for training")
    except Exception as e:
        ok &= check("ML 模型", False, str(e))

    print("\n[6] Walk-forward v2")
    try:
        from walkforward_v2 import _monte_carlo
        trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 80}, {"pnl": -30}]
        mc = _monte_carlo(trades, 10000, 100)
        ok &= check("MC 仿真", "prob_profitable" in mc, f"prob={mc['prob_profitable']:.2f}")
    except Exception as e:
        ok &= check("Walk-forward", False, str(e))

    print("\n[7] 测试套件")
    try:
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover("tests", pattern="test_*.py")
        ok &= check("测试发现", suite.countTestCases() > 10, f"{suite.countTestCases()} tests")
    except Exception as e:
        ok &= check("测试", False, str(e))

    print("\n" + "=" * 60)
    print(f"  {'✅ 所有检查通过！' if ok else '❌ 部分检查失败'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

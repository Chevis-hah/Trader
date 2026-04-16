"""
快速验证脚本 — 验证 v2.1 修改是否正确加载

用法:
    cd /path/to/Trader
    python scripts/validate_v21.py

检查项:
1. 策略加载和配置读取
2. 滑点参数是否已校正
3. Regime gate 是否启用
4. 自适应追踪止损参数
5. 因子扫描脚本可导入
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(name: str, condition: bool, detail: str = ""):
    icon = "✅" if condition else "❌"
    msg = f"  {icon} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    print("=" * 60)
    print("  🔍 v2.1 修改验证")
    print("=" * 60)

    all_ok = True

    # ---- 1. 策略加载 ----
    print("\n[1] 策略加载")
    try:
        from alpha.strategy_registry import build_strategy, available_strategies
        strategies = available_strategies()
        all_ok &= check("策略注册表", len(strategies) >= 4, f"可用策略: {strategies}")
    except Exception as e:
        all_ok &= check("策略注册表", False, str(e))

    # ---- 2. MACD Momentum 校正 ----
    print("\n[2] MACD Momentum 策略")
    try:
        from alpha.macd_momentum_strategy import MACDMomentumStrategy, MACDMomentumStrategyConfig
        cfg = MACDMomentumStrategyConfig()

        all_ok &= check("slippage_pct 校正", cfg.slippage_pct >= 0.002,
                         f"当前={cfg.slippage_pct} (需 ≥ 0.002)")
        all_ok &= check("regime_gate 启用", cfg.regime_gate_enabled,
                         f"regime_adx_min={cfg.regime_adx_min}")
        all_ok &= check("自适应追踪参数", hasattr(cfg, "trail_atr_base"),
                         f"base={cfg.trail_atr_base}, bonus={cfg.trail_profit_bonus}")
        all_ok &= check("止损后冷却延长", cfg.cooldown_bars_after_stop > cfg.cooldown_bars,
                         f"normal={cfg.cooldown_bars}, after_stop={cfg.cooldown_bars_after_stop}")

        strategy = MACDMomentumStrategy(cfg)
        all_ok &= check("策略实例化", strategy.name == "macd_momentum")
    except Exception as e:
        all_ok &= check("MACD 策略加载", False, str(e))

    # ---- 3. Triple EMA 校正 ----
    print("\n[3] Triple EMA 策略")
    try:
        from alpha.triple_ema_strategy import TripleEMAStrategy, TripleEMAStrategyConfig
        cfg = TripleEMAStrategyConfig()

        all_ok &= check("slippage_pct 校正", cfg.slippage_pct >= 0.002,
                         f"当前={cfg.slippage_pct}")
        all_ok &= check("regime_gate 启用", cfg.regime_gate_enabled,
                         f"regime_adx_min={cfg.regime_adx_min}")
        all_ok &= check("回踩范围放宽", cfg.pullback_min_atr <= -1.0,
                         f"min={cfg.pullback_min_atr}, max={cfg.pullback_max_atr}")
        all_ok &= check("量比门槛降低", cfg.min_volume_ratio <= 0.85,
                         f"当前={cfg.min_volume_ratio}")
    except Exception as e:
        all_ok &= check("Triple EMA 策略加载", False, str(e))

    # ---- 4. 增强出场 ----
    print("\n[4] 增强出场逻辑")
    try:
        from alpha.enhanced_exit import EnhancedExitMixin, EnhancedExitConfig, patch_strategy_exit
        cfg = EnhancedExitConfig()

        all_ok &= check("分批止盈阈值", cfg.partial_1_atr_mult >= 3.0,
                         f"P1={cfg.partial_1_atr_mult} ATR, P2={cfg.partial_2_atr_mult} ATR")
        all_ok &= check("保本止损", cfg.breakeven_enabled,
                         f"trigger={cfg.breakeven_trigger_atr} ATR")
        all_ok &= check("追踪下限", cfg.trail_min_mult >= 1.8,
                         f"min={cfg.trail_min_mult}")
    except Exception as e:
        all_ok &= check("增强出场加载", False, str(e))

    # ---- 5. 因子扫描脚本 ----
    print("\n[5] 因子扫描脚本")
    try:
        from factor_signal_scan import analyze_single_factor, categorize_factor
        import pandas as pd
        import numpy as np

        # 快速测试
        np.random.seed(42)
        n = 200
        factor = pd.Series(np.random.randn(n))
        fwd = pd.Series(np.random.randn(n) * 0.01)
        result = analyze_single_factor(factor, fwd, "test_factor")

        all_ok &= check("因子分析函数", "ic" in result and "t_stat" in result,
                         f"status={result['status']}")
        all_ok &= check("因子分类", categorize_factor("rsi_14") == "oscillator")
    except Exception as e:
        all_ok &= check("因子扫描脚本", False, str(e))

    # ---- 6. 测试文件 ----
    print("\n[6] 测试框架")
    try:
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover("tests", pattern="test_*.py")
        all_ok &= check("测试发现", suite.countTestCases() > 0,
                         f"发现 {suite.countTestCases()} 个测试")
    except Exception as e:
        all_ok &= check("测试框架", False, str(e))

    # ---- 总结 ----
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ 所有检查通过！")
    else:
        print("  ❌ 部分检查失败，请检查上述错误")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

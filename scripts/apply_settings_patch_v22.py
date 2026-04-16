"""
配置校正 v2.2 — 新增 mean_reversion 段 + 更新过滤器参数

python scripts/apply_settings_patch_v22.py
python scripts/apply_settings_patch_v22.py --dry-run
"""
import sys, yaml, copy
from pathlib import Path


PATCHES = {
    # 全局滑点
    "strategy.common.slippage_pct": 0.0021,

    # MACD 过度拉伸过滤
    "strategy.macd_momentum.overextension_filter": True,
    "strategy.macd_momentum.max_zscore_entry": 1.5,
    "strategy.macd_momentum.max_ma_dev_entry": 0.05,
    "strategy.macd_momentum.max_keltner_entry": 0.85,
    "strategy.macd_momentum.regime_gate_enabled": True,
    "strategy.macd_momentum.regime_adx_min": 22.0,
    "strategy.macd_momentum.slippage_pct": 0.0021,
    "strategy.macd_momentum.trail_atr_base": 2.5,
    "strategy.macd_momentum.trail_profit_bonus": 0.3,
    "strategy.macd_momentum.cooldown_bars_after_stop": 8,

    # Triple EMA 过度拉伸过滤
    "strategy.triple_ema.overextension_filter": True,
    "strategy.triple_ema.max_zscore_entry": 1.5,
    "strategy.triple_ema.max_ma_dev_entry": 0.05,
    "strategy.triple_ema.max_keltner_entry": 0.85,
    "strategy.triple_ema.regime_gate_enabled": True,
    "strategy.triple_ema.regime_adx_min": 25.0,
    "strategy.triple_ema.slippage_pct": 0.0021,
    "strategy.triple_ema.pullback_min_atr": -1.2,
    "strategy.triple_ema.pullback_max_atr": 0.8,
    "strategy.triple_ema.min_volume_ratio": 0.85,

    # Mean Reversion 新增段
    "strategy.mean_reversion.zscore_entry": -1.5,
    "strategy.mean_reversion.rsi_entry": 35.0,
    "strategy.mean_reversion.max_adx": 25.0,
    "strategy.mean_reversion.min_bb_width": 0.02,
    "strategy.mean_reversion.stop_atr_mult": 2.5,
    "strategy.mean_reversion.trail_atr_mult": 2.0,
    "strategy.mean_reversion.risk_per_trade": 0.010,
    "strategy.mean_reversion.max_position_pct": 0.35,
    "strategy.mean_reversion.slippage_pct": 0.0021,
    "strategy.mean_reversion.cooldown_bars": 4,

    # 执行层滑点
    "execution.slippage.base_bps": 5,
    "execution.slippage.impact_coefficient": 0.2,
}


def set_nested(d, key, val):
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/settings.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    path = Path(args.config)
    if not path.exists():
        print(f"❌ {path} 不存在")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    orig = copy.deepcopy(cfg)

    changes = 0
    for key, val in PATCHES.items():
        keys = key.split(".")
        old = cfg
        for k in keys:
            old = old.get(k, "NOT_SET") if isinstance(old, dict) else "NOT_SET"
        if old != val:
            set_nested(cfg, key, val)
            changes += 1
            print(f"  📝 {key}: {old} → {val}")

    print(f"\n  共 {changes} 项变更")
    if args.dry_run or changes == 0:
        return

    with open(path.with_suffix(".yaml.bak"), "w", encoding="utf-8") as f:
        yaml.dump(orig, f, default_flow_style=False, allow_unicode=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"  ✅ 已更新: {path}")


if __name__ == "__main__":
    main()

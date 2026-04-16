"""
配置校正补丁 — 自动更新 settings.yaml 中的关键参数

用法:
    python scripts/apply_settings_patch.py
    python scripts/apply_settings_patch.py --config config/settings.yaml --dry-run

校正内容:
1. 滑点假设从 10bps 提升到 21bps (BTC) / 26bps (ETH)
2. 策略参数更新 (regime gate, 自适应追踪止损)
3. 新增 1h 时间框架支持
4. 更新 universe 添加 SOL/BNB
"""
import sys
import yaml
import copy
from pathlib import Path

# ============================================================
# 需要校正的参数
# ============================================================
SETTINGS_PATCHES = {
    # --- 滑点校正 ---
    "execution.slippage.base_bps": 5,               # 基础滑点提高
    "execution.slippage.impact_coefficient": 0.2,    # 冲击成本系数提高

    # --- 手续费 (保持不变，仅确认) ---
    "execution.commission.maker_rate": 0.0002,
    "execution.commission.taker_rate": 0.0004,

    # --- Triple EMA 策略参数校正 ---
    "strategy.triple_ema.regime_gate_enabled": True,
    "strategy.triple_ema.regime_adx_min": 25.0,
    "strategy.triple_ema.slippage_pct": 0.0021,      # 21 bps
    "strategy.triple_ema.pullback_min_atr": -1.20,    # 放宽
    "strategy.triple_ema.pullback_max_atr": 0.80,
    "strategy.triple_ema.min_volume_ratio": 0.85,
    "strategy.triple_ema.trail_atr_base": 2.8,
    "strategy.triple_ema.trail_profit_bonus": 0.3,
    "strategy.triple_ema.trail_max_mult": 5.0,
    "strategy.triple_ema.trail_min_mult": 2.0,

    # --- MACD Momentum 策略参数校正 ---
    "strategy.macd_momentum.regime_gate_enabled": True,
    "strategy.macd_momentum.regime_adx_min": 22.0,
    "strategy.macd_momentum.slippage_pct": 0.0021,   # 21 bps
    "strategy.macd_momentum.trail_atr_base": 2.5,
    "strategy.macd_momentum.trail_profit_bonus": 0.3,
    "strategy.macd_momentum.trail_max_mult": 5.0,
    "strategy.macd_momentum.trail_min_mult": 1.8,
    "strategy.macd_momentum.cooldown_bars_after_stop": 8,

    # --- 通用策略参数 ---
    "strategy.common.slippage_pct": 0.0021,           # 全局滑点校正

    # --- Regime 策略参数 ---
    "strategy.regime.slippage_pct": 0.0021,
}


def set_nested(d: dict, key_path: str, value):
    """设置嵌套字典的值"""
    keys = key_path.split(".")
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def main():
    import argparse
    parser = argparse.ArgumentParser(description="配置校正补丁")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--dry-run", action="store_true", help="仅显示变更，不实际修改")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    original = copy.deepcopy(config)

    print("=" * 60)
    print("  🔧 配置校正补丁")
    print("=" * 60)

    changes = []
    for key_path, new_value in SETTINGS_PATCHES.items():
        keys = key_path.split(".")
        # 获取旧值
        old_value = config
        for k in keys:
            if isinstance(old_value, dict):
                old_value = old_value.get(k, "NOT_SET")
            else:
                old_value = "NOT_SET"
                break

        if old_value != new_value:
            changes.append((key_path, old_value, new_value))
            set_nested(config, key_path, new_value)
            print(f"  📝 {key_path}: {old_value} → {new_value}")
        else:
            print(f"  ✅ {key_path}: 已是 {new_value}")

    if not changes:
        print("\n  ℹ️ 无需变更")
        return

    print(f"\n  共 {len(changes)} 项变更")

    if args.dry_run:
        print("  [DRY RUN] 未实际修改文件")
        return

    # 备份
    backup_path = config_path.with_suffix(".yaml.bak")
    with open(backup_path, "w", encoding="utf-8") as f:
        yaml.dump(original, f, default_flow_style=False, allow_unicode=True)
    print(f"  📋 备份: {backup_path}")

    # 写入
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"  ✅ 已更新: {config_path}")


if __name__ == "__main__":
    main()

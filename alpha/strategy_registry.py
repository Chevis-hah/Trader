"""
统一策略注册表 — v2.3.1

v2.3 变更:
  - 移除 RegimeAdaptiveStrategy (已归档到 archive/v22_deprecated/)
  - 保留策略: triple_ema / macd_momentum / mean_reversion
  - 新增: cross_sectional_momentum (横截面, 组合级策略, 不走 TradingEngine 循环)

v2.3.1 变更:
  - 正式移除 regime 的导入 (P0-T01 的遗留清理)
  - 加入 cross_sectional_momentum 到 SIMPLE_STRATEGIES (通过骨架实例化)
  - 旧的 MACDMomentumStrategyConfig / TripleEMAStrategyConfig 名字保持向后兼容

注册的策略会被 main.py / backtest_runner / TradingEngine 使用。
"""
from __future__ import annotations
from dataclasses import fields
from typing import Any, Type

from config.loader import Config, load_config
from alpha.triple_ema_strategy import TripleEMAStrategy, TripleEMAStrategyConfig
from alpha.macd_momentum_strategy import MACDMomentumStrategy, MACDMomentumStrategyConfig
from alpha.mean_reversion_strategy import MeanReversionStrategy, MeanReversionConfig

try:
    from alpha.grid_strategy import GridTradingStrategy
    _GRID_AVAILABLE = True
except ImportError:
    _GRID_AVAILABLE = False

try:
    from alpha.cross_sectional_momentum import (
        CrossSectionalMomentumStrategy,
        CrossSectionalConfig,
    )
    _XS_AVAILABLE = True
except ImportError:
    _XS_AVAILABLE = False


# 基于 config 的可配置策略 (TradingEngine 直接消费)
STRATEGY_MAP: dict[str, tuple[type, Type]] = {
    "triple_ema": (TripleEMAStrategy, TripleEMAStrategyConfig),
    "macd_momentum": (MACDMomentumStrategy, MACDMomentumStrategyConfig),
    "mean_reversion": (MeanReversionStrategy, MeanReversionConfig),
}

# 无参数 / 自管理参数的策略
SIMPLE_STRATEGIES: dict[str, type] = {}
if _GRID_AVAILABLE:
    SIMPLE_STRATEGIES["grid"] = GridTradingStrategy
# 横截面策略不走 single-symbol TradingEngine, 仅通过 cross_sectional_backtest 使用
# 这里只注册以让 --list-strategies 可见
if _XS_AVAILABLE:
    SIMPLE_STRATEGIES["cross_sectional_momentum"] = CrossSectionalMomentumStrategy


def available_strategies() -> list[str]:
    return sorted(list(STRATEGY_MAP.keys()) + list(SIMPLE_STRATEGIES.keys()))


def resolve_strategy_name(config=None, explicit_name=None) -> str:
    if explicit_name:
        name = explicit_name.strip().lower()
    elif config is not None and not isinstance(config, dict) and hasattr(config, "get_nested"):
        name = str(config.get_nested("strategy.name", "macd_momentum")).strip().lower()
    else:
        name = "macd_momentum"
    all_names = set(STRATEGY_MAP.keys()) | set(SIMPLE_STRATEGIES.keys())
    if name not in all_names:
        raise ValueError(f"未知策略: {name}. 可选: {', '.join(available_strategies())}")
    return name


def _config_section_to_dict(section):
    if section is None:
        return {}
    if hasattr(section, "_data"):
        return dict(section._data)
    if isinstance(section, dict):
        return dict(section)
    return {}


def _build_cfg(config_cls, raw_dict):
    allowed = {f.name for f in fields(config_cls)}
    return config_cls(**{k: v for k, v in raw_dict.items() if k in allowed})


def _yaml_strategy_params(strategy_name):
    try:
        base_cfg = load_config()
        common = _config_section_to_dict(base_cfg.get_nested("strategy.common", {}))
        spec = _config_section_to_dict(base_cfg.get_nested(f"strategy.{strategy_name}", {}))
        merged = dict(common)
        merged.update(spec)
        return merged
    except Exception:
        return {}


def build_strategy(config=None, explicit_name=None):
    name = resolve_strategy_name(config, explicit_name)
    if name in SIMPLE_STRATEGIES:
        return SIMPLE_STRATEGIES[name]()

    strategy_cls, cfg_cls = STRATEGY_MAP[name]
    raw_cfg = _yaml_strategy_params(name)
    if isinstance(config, dict):
        raw_cfg = {**raw_cfg, **config}
    elif config is not None:
        spec = _config_section_to_dict(config.get_nested(f"strategy.{name}", {}))
        common = _config_section_to_dict(config.get_nested("strategy.common", {}))
        raw_cfg = {**raw_cfg, **common, **spec}

    return strategy_cls(_build_cfg(cfg_cls, raw_cfg))

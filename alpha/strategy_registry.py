"""
统一策略注册表

职责：
- 按配置/CLI 选择策略
- 统一读取策略参数
- 给回测和实盘提供同一份策略实例
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Type

from config.loader import Config
from alpha.regime_strategy import RegimeAdaptiveStrategy, RegimeStrategyConfig
from alpha.triple_ema_strategy import TripleEMAStrategy, TripleEMAStrategyConfig
from alpha.macd_momentum_strategy import MACDMomentumStrategy, MACDMomentumStrategyConfig


STRATEGY_MAP: dict[str, tuple[type, Type]] = {
    "regime": (RegimeAdaptiveStrategy, RegimeStrategyConfig),
    "triple_ema": (TripleEMAStrategy, TripleEMAStrategyConfig),
    "macd_momentum": (MACDMomentumStrategy, MACDMomentumStrategyConfig),
}


def available_strategies() -> list[str]:
    return sorted(STRATEGY_MAP.keys())


def resolve_strategy_name(config: Config | None = None, explicit_name: str | None = None) -> str:
    if explicit_name:
        name = explicit_name.strip().lower()
    elif config is not None:
        name = str(config.get_nested("strategy.name", "triple_ema")).strip().lower()
    else:
        name = "triple_ema"

    if name not in STRATEGY_MAP:
        raise ValueError(
            f"未知策略: {name}. 可选值: {', '.join(available_strategies())}"
        )
    return name


def _config_section_to_dict(section: Any) -> dict[str, Any]:
    if section is None:
        return {}
    if hasattr(section, "_data"):
        return dict(section._data)
    if isinstance(section, dict):
        return dict(section)
    return {}


def _build_cfg(config_cls: Type, raw_dict: dict[str, Any]):
    allowed = {f.name for f in fields(config_cls)}
    kwargs = {k: v for k, v in raw_dict.items() if k in allowed}
    return config_cls(**kwargs)


def build_strategy(config: Config | None = None, explicit_name: str | None = None):
    name = resolve_strategy_name(config, explicit_name)
    strategy_cls, cfg_cls = STRATEGY_MAP[name]

    raw_cfg: dict[str, Any] = {}
    if config is not None:
        raw_cfg = _config_section_to_dict(config.get_nested(f"strategy.{name}", {}))
        common = _config_section_to_dict(config.get_nested("strategy.common", {}))
        merged = dict(common)
        merged.update(raw_cfg)
        raw_cfg = merged

    cfg = _build_cfg(cfg_cls, raw_cfg)
    return strategy_cls(cfg)

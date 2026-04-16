"""
出场逻辑增强 — v2.2 (与 v2.1 相同，已稳定)

分批止盈 + 自适应追踪止损 + 保本止损
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger("enhanced_exit")


@dataclass
class EnhancedExitConfig:
    partial_exit_enabled: bool = True
    partial_1_atr_mult: float = 3.0
    partial_1_pct: float = 0.30
    partial_2_atr_mult: float = 6.0
    partial_2_pct: float = 0.30

    adaptive_trail_enabled: bool = True
    trail_base_mult: float = 2.5
    trail_profit_bonus: float = 0.3
    trail_max_mult: float = 5.0
    trail_min_mult: float = 1.8

    vol_adaptive_enabled: bool = True
    vol_expansion_threshold: float = 1.5
    vol_trail_boost: float = 0.5

    breakeven_enabled: bool = True
    breakeven_trigger_atr: float = 2.0


class EnhancedExitMixin:
    def check_enhanced_exit(self, row, prev_row, position, bar_count, exit_cfg=None):
        cfg = exit_cfg or EnhancedExitConfig()
        close = float(row.get("close", 0) or 0)
        if close <= 0:
            return False, "", None

        entry_price = position.get("entry_price", close)
        atr_at_entry = position.get("atr_at_entry", 0.0)
        highest = position.get("highest_since_entry", close)
        natr = float(row.get("natr_20", 0) or 0)
        current_atr = natr * close if natr > 0 else atr_at_entry
        if atr_at_entry <= 0:
            atr_at_entry = current_atr
        if atr_at_entry <= 0:
            return False, "", None

        profit_atr = (close - entry_price) / atr_at_entry
        init_stop = position.get("stop_loss", 0.0)

        if cfg.breakeven_enabled and profit_atr >= cfg.breakeven_trigger_atr:
            if init_stop < entry_price * 1.001:
                position["stop_loss"] = entry_price * 1.001

        if cfg.partial_exit_enabled:
            if not position.get("partial_1_done") and profit_atr >= cfg.partial_1_atr_mult:
                position["partial_1_done"] = True
                position["stop_loss"] = max(init_stop, entry_price * 1.001)
                return True, "partial_profit_1", cfg.partial_1_pct
            if (not position.get("partial_2_done") and position.get("partial_1_done")
                    and profit_atr >= cfg.partial_2_atr_mult):
                position["partial_2_done"] = True
                position["stop_loss"] = max(position.get("stop_loss", 0), entry_price + atr_at_entry)
                return True, "partial_profit_2", cfg.partial_2_pct

        if cfg.adaptive_trail_enabled and current_atr > 0:
            trail_mult = cfg.trail_base_mult
            if profit_atr > 0:
                trail_mult += profit_atr * cfg.trail_profit_bonus
            if cfg.vol_adaptive_enabled and atr_at_entry > 0:
                if current_atr / atr_at_entry > cfg.vol_expansion_threshold:
                    trail_mult += cfg.vol_trail_boost
            trail_mult = max(cfg.trail_min_mult, min(cfg.trail_max_mult, trail_mult))
            if close <= max(init_stop, highest - trail_mult * current_atr):
                return True, "adaptive_trailing_stop", None

        if close <= init_stop:
            return True, "stop_loss", None
        return False, "", None


def patch_strategy_exit(strategy, exit_cfg=None):
    cfg = exit_cfg or EnhancedExitConfig()
    mixin = EnhancedExitMixin()
    original = strategy.check_exit

    def enhanced(row, prev_row, position, bar_count):
        ok, reason, qty = mixin.check_enhanced_exit(row, prev_row, position, bar_count, cfg)
        if ok:
            return True, reason
        return original(row, prev_row, position, bar_count)

    strategy.check_exit = enhanced
    strategy._enhanced_exit_cfg = cfg
    logger.info(f"已为 {strategy.name} 启用增强出场")
    return strategy

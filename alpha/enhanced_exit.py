"""
P3: 出场逻辑增强 — 分批止盈 + 自适应追踪止损

问题:
- MACD 策略的 trailing_stop 贡献 -74.94 (21次全负)
  -> 追踪太紧，在正常波动中被震出
- 大赢交易 (+40%) 完全靠追踪止损退出
  -> 利润回吐可能很大，需要分批锁定

解决方案:
1. 分批止盈: 盈利 >2 ATR 先平 30%, >4 ATR 再平 30%
2. 自适应追踪: 盈利越多，追踪止损越宽松
3. 波动率感知: 高波动时自动放宽止损

集成方式:
  替换原策略的 check_exit 方法，或作为包装器使用
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
    # 分批止盈
    partial_exit_enabled: bool = True
    partial_1_atr_mult: float = 2.5     # 第一批: 盈利 > 2.5 ATR
    partial_1_pct: float = 0.30          # 平 30%
    partial_2_atr_mult: float = 5.0      # 第二批: 盈利 > 5 ATR
    partial_2_pct: float = 0.30          # 再平 30%

    # 自适应追踪止损
    adaptive_trail_enabled: bool = True
    trail_base_mult: float = 2.5         # 基础追踪倍数
    trail_profit_bonus: float = 0.3      # 每盈利 1 ATR，追踪放宽 0.3
    trail_max_mult: float = 5.0          # 追踪倍数上限
    trail_min_mult: float = 1.5          # 追踪倍数下限

    # 波动率自适应
    vol_adaptive_enabled: bool = True
    vol_expansion_threshold: float = 1.5  # 波动率扩张阈值 (相对入场时)
    vol_trail_boost: float = 0.5          # 波动率扩张时额外放宽


class EnhancedExitMixin:
    """
    增强出场逻辑 — 通过组合模式叠加到任何策略上

    用法:
        class MyStrategy(EnhancedExitMixin, OriginalStrategy):
            def __init__(self):
                super().__init__()
                self.exit_cfg = EnhancedExitConfig()
    """

    def check_enhanced_exit(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        position: dict,
        bar_count: int,
        exit_cfg: EnhancedExitConfig = None,
    ) -> tuple[bool, str, Optional[float]]:
        """
        增强出场检查

        Returns:
            (should_exit, reason, exit_qty_pct)
            exit_qty_pct: None = 全部平仓, 0.3 = 平 30%
        """
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

        # 当前盈利 (ATR 单位)
        profit_atr = (close - entry_price) / atr_at_entry
        init_stop = position.get("stop_loss", 0.0)

        # ==== 1. 分批止盈 ====
        if cfg.partial_exit_enabled:
            partial_1_done = position.get("partial_1_done", False)
            partial_2_done = position.get("partial_2_done", False)

            if not partial_1_done and profit_atr >= cfg.partial_1_atr_mult:
                position["partial_1_done"] = True
                # 同时把止损拉到入场价 (保本止损)
                position["stop_loss"] = max(init_stop, entry_price * 1.001)
                return True, "partial_profit_1", cfg.partial_1_pct

            if not partial_2_done and partial_1_done and profit_atr >= cfg.partial_2_atr_mult:
                position["partial_2_done"] = True
                # 把止损拉到入场价 + 1 ATR
                position["stop_loss"] = max(
                    position.get("stop_loss", 0),
                    entry_price + atr_at_entry
                )
                return True, "partial_profit_2", cfg.partial_2_pct

        # ==== 2. 自适应追踪止损 ====
        if cfg.adaptive_trail_enabled and current_atr > 0:
            trail_mult = cfg.trail_base_mult

            # 盈利越多，追踪越宽松
            if profit_atr > 0:
                trail_mult += profit_atr * cfg.trail_profit_bonus

            # 波动率扩张时放宽
            if cfg.vol_adaptive_enabled and atr_at_entry > 0:
                vol_ratio = current_atr / atr_at_entry
                if vol_ratio > cfg.vol_expansion_threshold:
                    trail_mult += cfg.vol_trail_boost

            # 限制范围
            trail_mult = max(cfg.trail_min_mult, min(cfg.trail_max_mult, trail_mult))

            trailing = max(init_stop, highest - trail_mult * current_atr)

            if close <= trailing:
                return True, "adaptive_trailing_stop", None

        # ==== 3. 原始止损兜底 ====
        if close <= init_stop:
            return True, "stop_loss", None

        return False, "", None


def patch_strategy_exit(strategy, exit_cfg: EnhancedExitConfig = None):
    """
    猴子补丁方式给策略添加增强出场

    用法:
        from alpha.enhanced_exit import patch_strategy_exit, EnhancedExitConfig
        strategy = build_strategy(config=config, explicit_name="triple_ema")
        patch_strategy_exit(strategy, EnhancedExitConfig(partial_exit_enabled=True))
    """
    cfg = exit_cfg or EnhancedExitConfig()
    mixin = EnhancedExitMixin()
    original_check_exit = strategy.check_exit

    def enhanced_check_exit(row, prev_row, position, bar_count):
        # 先检查增强出场
        should_exit, reason, qty_pct = mixin.check_enhanced_exit(
            row, prev_row, position, bar_count, cfg
        )
        if should_exit:
            return True, reason

        # 再检查原始出场
        return original_check_exit(row, prev_row, position, bar_count)

    strategy.check_exit = enhanced_check_exit
    strategy._enhanced_exit_cfg = cfg
    logger.info(f"已为 {strategy.name} 启用增强出场逻辑")
    return strategy

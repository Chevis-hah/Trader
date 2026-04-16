"""
Regime 感知的策略分配器 — v2.2

核心改动:
  强趋势 (ADX > 30): 100% 趋势策略
  中等趋势 (20 < ADX < 30): 50% 趋势 + 50% 均值回归
  震荡 (ADX < 20): 100% 均值回归
  暴跌 (RSI < 25 + 高 NATR): 全部关闭

使用方式 (在回测引擎中):
  allocator = RegimeAllocatorV2(total_capital=10000)
  for each bar:
      alloc = allocator.update(row, bar_index)
      # alloc = {"trend": 0.5, "mean_reversion": 0.5}
      trend_capital = alloc["trend"] * total_capital
      mr_capital = alloc["mean_reversion"] * total_capital
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("regime_allocator_v2")


@dataclass
class RegimeAllocatorConfig:
    # ADX 阈值
    strong_trend_adx: float = 30.0
    moderate_trend_adx: float = 20.0

    # NATR 阈值
    high_vol_natr: float = 0.04         # 4%

    # 分配比例
    strong_trend_allocation: dict = None   # {"trend": 1.0, "mean_reversion": 0.0}
    moderate_trend_allocation: dict = None # {"trend": 0.5, "mean_reversion": 0.5}
    range_allocation: dict = None          # {"trend": 0.0, "mean_reversion": 1.0}
    high_vol_allocation: dict = None       # {"trend": 0.3, "mean_reversion": 0.0}
    crash_allocation: dict = None          # {"trend": 0.0, "mean_reversion": 0.0}

    # 切换冷却
    regime_lookback: int = 6              # 众数窗口
    regime_cooldown_bars: int = 3

    def __post_init__(self):
        if self.strong_trend_allocation is None:
            self.strong_trend_allocation = {"trend": 1.0, "mean_reversion": 0.0}
        if self.moderate_trend_allocation is None:
            self.moderate_trend_allocation = {"trend": 0.5, "mean_reversion": 0.5}
        if self.range_allocation is None:
            self.range_allocation = {"trend": 0.0, "mean_reversion": 1.0}
        if self.high_vol_allocation is None:
            self.high_vol_allocation = {"trend": 0.3, "mean_reversion": 0.0}
        if self.crash_allocation is None:
            self.crash_allocation = {"trend": 0.0, "mean_reversion": 0.0}


def classify_regime_v2(row: pd.Series, cfg: RegimeAllocatorConfig = None) -> str:
    """
    Regime 分类

    Returns:
        STRONG_TREND | MODERATE_TREND | RANGE | HIGH_VOL | CRASH
    """
    cfg = cfg or RegimeAllocatorConfig()

    adx = row.get("adx_14", 0)
    natr = row.get("natr_20", 0)
    rsi = row.get("rsi_14", 50)

    if pd.isna(adx) or pd.isna(natr):
        return "RANGE"

    # 暴跌
    if pd.notna(rsi) and rsi < 25 and natr > cfg.high_vol_natr:
        return "CRASH"

    # 高波动
    if natr > cfg.high_vol_natr:
        return "HIGH_VOL"

    # 趋势强度
    if adx >= cfg.strong_trend_adx:
        return "STRONG_TREND"
    elif adx >= cfg.moderate_trend_adx:
        return "MODERATE_TREND"
    else:
        return "RANGE"


class RegimeAllocatorV2:
    """
    Regime 感知的策略资金分配器

    返回 {"trend": 0.0~1.0, "mean_reversion": 0.0~1.0}
    """

    def __init__(self, total_capital: float = 10000.0,
                 cfg: RegimeAllocatorConfig = None):
        self.total_capital = total_capital
        self.cfg = cfg or RegimeAllocatorConfig()

        self.current_regime = "RANGE"
        self.current_allocation = self.cfg.range_allocation.copy()
        self._history: list[str] = []
        self._last_switch_bar = -999

    def update(self, row: pd.Series, bar_index: int = 0) -> dict:
        """
        更新 regime 并返回分配

        Returns:
            {"trend": 0.0~1.0, "mean_reversion": 0.0~1.0}
        """
        instant = classify_regime_v2(row, self.cfg)
        self._history.append(instant)

        # 众数平滑
        window = self._history[-self.cfg.regime_lookback:]
        mode_regime = Counter(window).most_common(1)[0][0]

        # 冷却期
        if (mode_regime != self.current_regime
                and bar_index - self._last_switch_bar >= self.cfg.regime_cooldown_bars):
            old = self.current_regime
            self.current_regime = mode_regime
            self._last_switch_bar = bar_index
            self._update_allocation()
            logger.debug(f"Regime: {old} → {mode_regime} @ bar {bar_index}")

        return self.current_allocation.copy()

    def _update_allocation(self):
        mapping = {
            "STRONG_TREND": self.cfg.strong_trend_allocation,
            "MODERATE_TREND": self.cfg.moderate_trend_allocation,
            "RANGE": self.cfg.range_allocation,
            "HIGH_VOL": self.cfg.high_vol_allocation,
            "CRASH": self.cfg.crash_allocation,
        }
        self.current_allocation = mapping.get(
            self.current_regime, self.cfg.moderate_trend_allocation
        ).copy()

    def get_capital(self, strategy_type: str) -> float:
        """
        获取某类策略的可用资金

        Args:
            strategy_type: "trend" 或 "mean_reversion"
        """
        pct = self.current_allocation.get(strategy_type, 0.0)
        return self.total_capital * pct

    def get_risk_multiplier(self) -> float:
        """根据 regime 获取风险缩放因子"""
        mult = {
            "STRONG_TREND": 1.0,
            "MODERATE_TREND": 0.8,
            "RANGE": 0.7,
            "HIGH_VOL": 0.4,
            "CRASH": 0.0,
        }
        return mult.get(self.current_regime, 0.5)

    @property
    def stats(self) -> dict:
        dist = Counter(self._history)
        total = len(self._history) or 1
        return {
            "current_regime": self.current_regime,
            "current_allocation": self.current_allocation,
            "regime_distribution": {
                k: f"{v/total*100:.1f}%" for k, v in dist.most_common()
            },
            "total_bars": len(self._history),
        }

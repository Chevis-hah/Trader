"""
P2: 动态 Regime 策略分配器

根据当前市场状态自动选择最优策略组合:
- 强趋势 (ADX>25 + 顺势): Triple EMA 主导 (70%)
- 中等趋势 (ADX 15-25): MACD Momentum 主导 (60%)
- 横盘/弱势 (ADX<15): 减仓或关闭，仅保留最保守的信号

集成方式:
  在 backtest_runner.py 中替换单策略为此分配器
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("regime_allocator")


@dataclass
class RegimeConfig:
    # Regime 判定阈值
    strong_trend_adx: float = 25.0
    moderate_trend_adx: float = 15.0
    high_vol_natr: float = 0.035
    low_vol_natr: float = 0.012

    # 各 regime 下的资金分配
    strong_trend_allocation: dict = field(default_factory=lambda: {
        "triple_ema": 0.65, "macd_momentum": 0.35
    })
    moderate_trend_allocation: dict = field(default_factory=lambda: {
        "triple_ema": 0.35, "macd_momentum": 0.65
    })
    range_allocation: dict = field(default_factory=lambda: {
        "triple_ema": 0.0, "macd_momentum": 0.30
    })
    high_vol_allocation: dict = field(default_factory=lambda: {
        "triple_ema": 0.20, "macd_momentum": 0.20
    })

    # Regime 切换冷却
    regime_cooldown_bars: int = 6
    # 滚动窗口
    regime_lookback: int = 20


def classify_regime(row: pd.Series, cfg: RegimeConfig = None) -> str:
    """
    基于当前 bar 的指标判定 regime

    Returns:
        STRONG_TREND | MODERATE_TREND | RANGE | HIGH_VOL | CRASH
    """
    cfg = cfg or RegimeConfig()

    adx = row.get("adx_14", 0)
    natr = row.get("natr_20", 0)
    rsi = row.get("rsi_14", 50)

    if pd.isna(adx) or pd.isna(natr):
        return "UNKNOWN"

    # 暴跌检测: RSI < 25 + 高波动
    if pd.notna(rsi) and rsi < 25 and natr > cfg.high_vol_natr:
        return "CRASH"

    # 高波动 (不管方向)
    if natr > cfg.high_vol_natr:
        return "HIGH_VOL"

    # 趋势强度
    if adx >= cfg.strong_trend_adx:
        return "STRONG_TREND"
    elif adx >= cfg.moderate_trend_adx:
        return "MODERATE_TREND"
    else:
        return "RANGE"


def get_allocation(regime: str, cfg: RegimeConfig = None) -> dict[str, float]:
    """
    根据 regime 返回策略资金分配比例

    Returns:
        {"triple_ema": 0.65, "macd_momentum": 0.35}
    """
    cfg = cfg or RegimeConfig()

    mapping = {
        "STRONG_TREND": cfg.strong_trend_allocation,
        "MODERATE_TREND": cfg.moderate_trend_allocation,
        "RANGE": cfg.range_allocation,
        "HIGH_VOL": cfg.high_vol_allocation,
        "CRASH": {"triple_ema": 0.0, "macd_momentum": 0.0},  # 完全关闭
        "UNKNOWN": cfg.moderate_trend_allocation,
    }
    return mapping.get(regime, cfg.moderate_trend_allocation)


class RegimeAllocator:
    """
    Regime 感知的策略分配器

    在回测引擎中使用:
        allocator = RegimeAllocator(strategies, total_capital)
        for each bar:
            regime = allocator.update(row)
            for name, strategy in strategies.items():
                capital_for_this = allocator.get_capital(name)
                # 用 capital_for_this 做仓位计算
    """

    def __init__(self, strategy_names: list[str],
                 total_capital: float,
                 cfg: RegimeConfig = None):
        self.strategy_names = strategy_names
        self.total_capital = total_capital
        self.cfg = cfg or RegimeConfig()

        self.current_regime = "UNKNOWN"
        self.current_allocation = {}
        self._regime_history: list[str] = []
        self._last_switch_bar = -999

    def update(self, row: pd.Series, bar_index: int = 0) -> str:
        """
        更新 regime 并调整分配

        使用滚动窗口的众数来避免频繁切换
        """
        instant_regime = classify_regime(row, self.cfg)
        self._regime_history.append(instant_regime)

        # 滚动窗口众数
        window = self._regime_history[-self.cfg.regime_lookback:]
        from collections import Counter
        mode_regime = Counter(window).most_common(1)[0][0]

        # 冷却期内不切换
        if (mode_regime != self.current_regime
                and bar_index - self._last_switch_bar >= self.cfg.regime_cooldown_bars):
            old = self.current_regime
            self.current_regime = mode_regime
            self.current_allocation = get_allocation(mode_regime, self.cfg)
            self._last_switch_bar = bar_index
            logger.debug(f"Regime 切换: {old} -> {mode_regime} @ bar {bar_index}")

        if not self.current_allocation:
            self.current_allocation = get_allocation(self.current_regime, self.cfg)

        return self.current_regime

    def get_capital(self, strategy_name: str) -> float:
        """获取策略可用资金"""
        pct = self.current_allocation.get(strategy_name, 0.0)
        return self.total_capital * pct

    def get_risk_multiplier(self) -> float:
        """
        获取风险乘数: 在高风险 regime 下降低仓位

        STRONG_TREND: 1.0 (正常)
        MODERATE_TREND: 0.8
        RANGE: 0.5
        HIGH_VOL: 0.4
        CRASH: 0.0
        """
        multipliers = {
            "STRONG_TREND": 1.0,
            "MODERATE_TREND": 0.8,
            "RANGE": 0.5,
            "HIGH_VOL": 0.4,
            "CRASH": 0.0,
            "UNKNOWN": 0.6,
        }
        return multipliers.get(self.current_regime, 0.5)

    @property
    def stats(self) -> dict:
        from collections import Counter
        dist = Counter(self._regime_history)
        total = len(self._regime_history)
        return {
            "current_regime": self.current_regime,
            "regime_distribution": {
                k: f"{v/total*100:.1f}%" for k, v in dist.most_common()
            },
            "total_bars": total,
        }

"""
MACD Momentum Strategy — v2.3 (回滚至 v2.0 纯规则)

变更历史:
  v2.0: 纯 MACD 交叉 + ADX 过滤 → 92 trades, +834 PnL, 32.3% fold WR
  v2.1: + trail_atr_mult, cooldown_bars 等调参 → 略改善但不稳
  v2.2: + zscore / ma_dev / keltner 三层过度拉伸过滤 → 12 trades, +57 PnL (过拟合)
  v2.3: 回滚到 v2.0 纯规则 + 删除 min_rsi/max_rsi/max_holding_bars 等无效参数

失效证据:
  - sensitivity_macd_momentum.json:
    * min_rsi good_ratio = 0.0 (4 tested)
    * max_rsi good_ratio = 0.0 (4 tested)
    * max_holding_bars good_ratio = 0.0 (4 tested)
  - wf_macd_momentum_4h.json (v2.2):
    * 12 trades, fold WR 9.1%, 过滤器把 92 笔压到 12 笔
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("macd_momentum")


@dataclass
class MACDMomentumConfig:
    """v2.3 配置 —— 仅保留经 WF 验证有效的参数"""
    # 入场
    min_adx: float = 20.0             # ADX 门限，v2.0 原始值

    # 止损 / 跟踪
    stop_atr_mult: float = 2.0        # 初始止损 ATR 倍数
    trail_atr_mult: float = 2.8       # 跟踪止损 ATR 倍数（敏感度测试最稳定值）

    # 仓位 / 冷却
    risk_per_trade: float = 0.02      # 每笔风险 2%
    cooldown_bars: int = 3            # 平仓后冷却

    # ---- 已删除的参数（作为警示保留注释）----
    # min_rsi / max_rsi          — good_ratio=0, 纯噪声
    # max_holding_bars           — good_ratio=0, 纯噪声
    # zscore_threshold           — v2.2 过度拉伸过滤，回滚删除
    # ma_dev_threshold           — v2.2 过度拉伸过滤，回滚删除
    # keltner_threshold          — v2.2 过度拉伸过滤，回滚删除


class MACDMomentumStrategy:
    """
    MACD 动量突破策略 v2.3

    核心逻辑:
      1. MACD 线上穿 signal 线 (金叉)
      2. ADX >= min_adx (趋势强度过滤)
      3. 止损 = entry - stop_atr_mult * ATR
      4. 跟踪止损 = max(price * (1 - trail_atr_mult * NATR), stop_loss)
    """

    def __init__(self, config: Optional[MACDMomentumConfig] = None):
        self.cfg = config or MACDMomentumConfig()

    def prepare_features(
        self,
        features: pd.DataFrame,
        higher_tf: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        本策略不需要额外特征工程，直接使用 FeatureEngine 的输出。
        保留接口以和 strategy_registry 对齐。
        """
        return features

    def should_enter(
        self, row: pd.Series, prev: pd.Series, state: dict
    ) -> bool:
        """入场判断 (v2.0 纯规则)"""
        # 冷却期检查
        cooldown_until = state.get("cooldown_until_bar", -1)
        if state.get("bar_index", 0) < cooldown_until:
            return False

        # --- MACD 金叉 ---
        macd = row.get("macd")
        macd_signal = row.get("macd_signal")
        prev_macd = prev.get("macd")
        prev_macd_signal = prev.get("macd_signal")

        if any(pd.isna(x) for x in [macd, macd_signal, prev_macd, prev_macd_signal]):
            return False

        cross_above = (prev_macd <= prev_macd_signal) and (macd > macd_signal)
        if not cross_above:
            return False

        # --- ADX 强度过滤 ---
        adx = row.get("adx_14", 0)
        if pd.isna(adx) or adx < self.cfg.min_adx:
            return False

        return True

    def check_exit(
        self,
        row: pd.Series,
        prev: pd.Series,
        position: dict,
        bar_count: int,
    ) -> tuple[bool, str]:
        """
        退出判断

        Returns:
            (should_exit, reason)
        """
        close = float(row.get("close", 0) or 0)
        if close <= 0:
            return False, ""

        # 1) 初始止损
        stop_loss = position.get("stop_loss", 0)
        if stop_loss > 0 and close <= stop_loss:
            return True, "stop_loss"

        # 2) 跟踪止损
        highest = position.get("highest_since_entry", close)
        atr_at_entry = position.get("atr_at_entry", 0)
        if atr_at_entry > 0:
            trail_stop = highest - self.cfg.trail_atr_mult * atr_at_entry
            if close <= trail_stop:
                return True, "trail_stop"

        # 3) MACD 死叉 (反向信号退出)
        macd = row.get("macd")
        macd_signal = row.get("macd_signal")
        prev_macd = prev.get("macd")
        prev_macd_signal = prev.get("macd_signal")

        if not any(pd.isna(x) for x in [macd, macd_signal, prev_macd, prev_macd_signal]):
            cross_below = (prev_macd >= prev_macd_signal) and (macd < macd_signal)
            if cross_below:
                return True, "macd_cross_below"

        return False, ""

    def calc_position(
        self, available_capital: float, entry_price: float, row: pd.Series
    ) -> tuple[float, float]:
        """
        仓位 + 止损计算

        Returns:
            (qty, stop_loss_price)
        """
        natr = float(row.get("natr_20", 0) or 0)
        if natr <= 0:
            return 0.0, 0.0

        atr_price = natr * entry_price
        stop_loss = entry_price - self.cfg.stop_atr_mult * atr_price
        if stop_loss <= 0:
            return 0.0, 0.0

        # 基于固定风险比例计算仓位
        risk_per_unit = entry_price - stop_loss
        if risk_per_unit <= 0:
            return 0.0, 0.0

        risk_budget = available_capital * self.cfg.risk_per_trade
        qty = risk_budget / risk_per_unit

        # 限制单笔最大仓位不超过可用资金 (避免杠杆隐含)
        max_qty = available_capital / entry_price * 0.95
        qty = min(qty, max_qty)

        return qty, stop_loss

    def on_trade_closed(self, state: dict, bar_index: int, reason: str) -> None:
        """平仓后更新冷却"""
        state["cooldown_until_bar"] = bar_index + self.cfg.cooldown_bars

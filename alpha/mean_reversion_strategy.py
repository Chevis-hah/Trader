"""
Mean Reversion Strategy — v2.3 (新增趋势回避门)

变更历史:
  v1.0 (Phase 1): BB + RSI + NATR → wf_mean_reversion_4h: 59 trades, 39.4% fold WR, -425 PnL
  v2.3: + 趋势回避门，避免在强趋势中逆势入场

改进依据:
  - wf_mean_reversion_4h.json 显示, 最大亏损 fold 全部发生在强趋势行情:
    * fold 0 (-229): 2021-10 到 12, BTC 4 万到 6 万的强势上涨
    * fold 5 (-279): 2022-08 到 10, BTC 暴跌期
    * fold 13 (-192): 2023-11 到 2024-01, BTC 底部反弹强势上涨
  - 这三个 fold 的共同特征: ADX 高位 + 20 bar 单调性 > 85%

核心逻辑:
  原有 (v1.0):
    入场: 价格跌破 BB 下轨 + RSI < 30 + NATR 合理
    止损: 固定 ATR 倍数
    退出: 回到 BB 中轨

  新增 (v2.3 趋势回避门):
    禁用: ADX_14 > 28 (强趋势)
    禁用: 最近 20 bar 上涨/下跌比例 > 85% (单边行情)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("mean_reversion")


@dataclass
class MeanReversionConfig:
    """v2.3 配置"""
    # --- 原有参数 ---
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # 止损 / 止盈
    stop_atr_mult: float = 2.0
    target_pct: float = 0.03          # 3% 目标 (回到中轨的保守估计)
    max_holding_bars: int = 24        # 最多持 24 根 4h bar (4 天)

    # 仓位
    risk_per_trade: float = 0.015     # 均值回归用较小仓位
    cooldown_bars: int = 2

    # --- v2.3 新增: 趋势回避门 ---
    max_adx_for_entry: float = 28.0            # ADX 超此值禁用
    monotonic_lookback: int = 20               # 检查最近 N bar
    max_monotonic_ratio: float = 0.85          # 单向占比超此值禁用


class MeanReversionStrategy:
    """
    均值回归策略 v2.3

    适用场景: 震荡行情 (ADX < 28)
    禁用场景: 强趋势 / 单边行情
    """

    def __init__(self, config: Optional[MeanReversionConfig] = None):
        self.cfg = config or MeanReversionConfig()

    def prepare_features(
        self,
        features: pd.DataFrame,
        higher_tf: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """补充 BB 指标 (若 FeatureEngine 未计算)"""
        df = features.copy()

        if "bb_lower" not in df.columns and "close" in df.columns:
            ma = df["close"].rolling(self.cfg.bb_period).mean()
            std = df["close"].rolling(self.cfg.bb_period).std()
            df["bb_middle"] = ma
            df["bb_upper"] = ma + self.cfg.bb_std * std
            df["bb_lower"] = ma - self.cfg.bb_std * std

        return df

    # ----------------------------------------------------------------
    # v2.3 新增: 趋势回避门
    # ----------------------------------------------------------------
    def _check_trend_avoidance(
        self, row: pd.Series, state: dict
    ) -> tuple[bool, str]:
        """
        检查是否应该回避交易 (强趋势行情)

        Returns:
            (should_avoid, reason)
        """
        # 1) ADX 过滤
        adx = row.get("adx_14", 0)
        if pd.notna(adx) and adx > self.cfg.max_adx_for_entry:
            return True, f"adx_too_high({adx:.1f})"

        # 2) 单调性检查
        recent_closes = state.get("_recent_closes_for_trend")
        if recent_closes is None:
            recent_closes = deque(maxlen=self.cfg.monotonic_lookback + 1)
            state["_recent_closes_for_trend"] = recent_closes

        close = row.get("close")
        if pd.notna(close):
            recent_closes.append(float(close))

        if len(recent_closes) >= self.cfg.monotonic_lookback + 1:
            closes_list = list(recent_closes)
            ups = sum(
                1 for i in range(1, len(closes_list))
                if closes_list[i] > closes_list[i - 1]
            )
            downs = sum(
                1 for i in range(1, len(closes_list))
                if closes_list[i] < closes_list[i - 1]
            )
            total = ups + downs
            if total > 0:
                up_ratio = ups / total
                down_ratio = downs / total
                if up_ratio >= self.cfg.max_monotonic_ratio:
                    return True, f"monotonic_up({up_ratio:.0%})"
                if down_ratio >= self.cfg.max_monotonic_ratio:
                    return True, f"monotonic_down({down_ratio:.0%})"

        return False, ""

    # ----------------------------------------------------------------
    # 入场 / 退出
    # ----------------------------------------------------------------
    def should_enter(
        self, row: pd.Series, prev: pd.Series, state: dict
    ) -> bool:
        cooldown_until = state.get("cooldown_until_bar", -1)
        if state.get("bar_index", 0) < cooldown_until:
            return False

        # v2.3 新增: 趋势回避
        should_avoid, reason = self._check_trend_avoidance(row, state)
        if should_avoid:
            # 调试时可以打开
            # logger.debug(f"avoid entry: {reason}")
            return False

        # 原有入场条件
        close = row.get("close")
        bb_lower = row.get("bb_lower")
        rsi = row.get("rsi_14")

        if any(pd.isna(x) for x in [close, bb_lower, rsi]):
            return False

        # 价格跌破 BB 下轨 + RSI 超卖
        price_below_lower = close <= bb_lower
        rsi_oversold = rsi < self.cfg.rsi_oversold

        return price_below_lower and rsi_oversold

    def check_exit(
        self,
        row: pd.Series,
        prev: pd.Series,
        position: dict,
        bar_count: int,
    ) -> tuple[bool, str]:
        close = float(row.get("close", 0) or 0)
        if close <= 0:
            return False, ""

        entry_price = position.get("entry_price", 0)

        # 1) 止损
        stop_loss = position.get("stop_loss", 0)
        if stop_loss > 0 and close <= stop_loss:
            return True, "stop_loss"

        # 2) 目标: 回到 BB 中轨或达到 target_pct
        bb_middle = row.get("bb_middle")
        if pd.notna(bb_middle) and close >= bb_middle:
            return True, "bb_middle_touch"

        if entry_price > 0 and close >= entry_price * (1 + self.cfg.target_pct):
            return True, "target_profit"

        # 3) 超时退出
        if bar_count >= self.cfg.max_holding_bars:
            return True, "max_holding"

        # 4) RSI 超买反转
        rsi = row.get("rsi_14")
        if pd.notna(rsi) and rsi > self.cfg.rsi_overbought:
            return True, "rsi_overbought"

        return False, ""

    def calc_position(
        self, available_capital: float, entry_price: float, row: pd.Series
    ) -> tuple[float, float]:
        natr = float(row.get("natr_20", 0) or 0)
        if natr <= 0:
            return 0.0, 0.0

        atr_price = natr * entry_price
        stop_loss = entry_price - self.cfg.stop_atr_mult * atr_price
        if stop_loss <= 0:
            return 0.0, 0.0

        risk_per_unit = entry_price - stop_loss
        if risk_per_unit <= 0:
            return 0.0, 0.0

        risk_budget = available_capital * self.cfg.risk_per_trade
        qty = risk_budget / risk_per_unit
        max_qty = available_capital / entry_price * 0.95
        qty = min(qty, max_qty)

        return qty, stop_loss

    def on_trade_closed(self, state: dict, bar_index: int, reason: str) -> None:
        state["cooldown_until_bar"] = bar_index + self.cfg.cooldown_bars

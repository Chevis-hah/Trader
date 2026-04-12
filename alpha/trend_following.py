"""
趋势跟踪策略（Trend Following）

核心逻辑：
  - 双均线交叉（EMA20/EMA60 on 4h）判断方向
  - ATR(14) 波动率过滤：只在波动率放大时开仓
  - ATR 仓位法：波动大 → 仓位小，波动小 → 仓位大
  - 移动止损：2x ATR trailing stop

适用场景：
  - 加密市场天然趋势性强、均值回归弱
  - 胜率 35-45%，但盈亏比 > 2:1
  - 在趋势行情中表现优异，震荡行情会小亏
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrendSignal:
    """趋势信号"""
    direction: str = "FLAT"   # LONG / SHORT / FLAT
    strength: float = 0.0      # 信号强度 0~1
    entry_price: float = 0.0
    stop_loss: float = 0.0
    position_size_pct: float = 0.0  # 建议仓位占比


@dataclass
class TrendPosition:
    """趋势持仓"""
    symbol: str = ""
    direction: str = "FLAT"
    entry_price: float = 0.0
    entry_time: int = 0
    quantity: float = 0.0
    stop_loss: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 999999999.0
    trailing_stop: float = 0.0
    atr_at_entry: float = 0.0


class TrendFollowingStrategy:
    """
    趋势跟踪策略

    参数:
      fast_period:    快线 EMA 周期（默认 20）
      slow_period:    慢线 EMA 周期（默认 60）
      atr_period:     ATR 周期（默认 14）
      atr_filter_period: ATR 均线周期，用于判断波动率是否放大（默认 60）
      atr_risk_mult:  每笔交易风险 = atr_risk_mult × ATR（默认 2.0）
      risk_per_trade:  单笔风险占总资金比例（默认 0.02 = 2%）
      trailing_atr_mult: 移动止损倍数（默认 2.5）
    """

    def __init__(self,
                 fast_period: int = 20,
                 slow_period: int = 60,
                 atr_period: int = 14,
                 atr_filter_period: int = 60,
                 atr_risk_mult: float = 2.0,
                 risk_per_trade: float = 0.02,
                 trailing_atr_mult: float = 2.5):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_filter_period = atr_filter_period
        self.atr_risk_mult = atr_risk_mult
        self.risk_per_trade = risk_per_trade
        self.trailing_atr_mult = trailing_atr_mult

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有指标，返回含指标列的 DataFrame"""
        out = df.copy()
        close = out["close"]

        # 均线
        out["ema_fast"] = close.ewm(span=self.fast_period, adjust=False).mean()
        out["ema_slow"] = close.ewm(span=self.slow_period, adjust=False).mean()

        # ATR
        hl = out["high"] - out["low"]
        hc = (out["high"] - close.shift(1)).abs()
        lc = (out["low"] - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        out["atr"] = tr.rolling(self.atr_period).mean()

        # ATR 均线（波动率过滤器）
        out["atr_ma"] = out["atr"].rolling(self.atr_filter_period).mean()

        # 均线差值（归一化）
        out["ma_diff"] = (out["ema_fast"] - out["ema_slow"]) / out["atr"].clip(lower=1e-10)

        # 均线交叉信号
        out["ma_cross"] = 0
        out.loc[out["ema_fast"] > out["ema_slow"], "ma_cross"] = 1
        out.loc[out["ema_fast"] < out["ema_slow"], "ma_cross"] = -1

        # 前一根的交叉状态
        out["ma_cross_prev"] = out["ma_cross"].shift(1)

        # 波动率过滤：当前 ATR > ATR 均线
        out["vol_expanding"] = (out["atr"] > out["atr_ma"]).astype(int)

        return out

    def generate_signal(self, row: pd.Series, prev_row: pd.Series,
                        position: Optional[TrendPosition],
                        capital: float) -> TrendSignal:
        """
        基于当前 bar 生成交易信号

        返回:
          TrendSignal 包含方向、止损位、建议仓位大小
        """
        signal = TrendSignal()

        atr = row.get("atr", 0)
        if atr <= 0 or pd.isna(atr):
            return signal

        close = row["close"]
        ma_cross = row.get("ma_cross", 0)
        ma_cross_prev = row.get("ma_cross_prev", 0)
        vol_expanding = row.get("vol_expanding", 0)

        # ---- 已有持仓：检查是否触发移动止损 ----
        if position and position.direction != "FLAT":
            if position.direction == "LONG":
                # 更新最高价
                position.highest_since_entry = max(position.highest_since_entry, close)
                # 移动止损
                new_trailing = position.highest_since_entry - self.trailing_atr_mult * atr
                position.trailing_stop = max(position.trailing_stop, new_trailing)

                # 触发止损
                if close <= position.trailing_stop:
                    signal.direction = "CLOSE_LONG"
                    signal.entry_price = close
                    return signal

                # 均线死叉 → 平仓
                if ma_cross == -1 and ma_cross_prev == 1:
                    signal.direction = "CLOSE_LONG"
                    signal.entry_price = close
                    return signal

            return signal  # 持仓中，不生成新信号

        # ---- 无持仓：检查是否开仓 ----
        # 条件：均线金叉 + 波动率放大
        if ma_cross == 1 and ma_cross_prev != 1 and vol_expanding:
            stop_loss = close - self.atr_risk_mult * atr
            risk_amount = capital * self.risk_per_trade
            risk_per_unit = close - stop_loss

            if risk_per_unit > 0:
                qty = risk_amount / risk_per_unit
                position_value = qty * close
                position_pct = position_value / capital if capital > 0 else 0

                # 限制最大仓位 50%
                position_pct = min(position_pct, 0.50)

                signal.direction = "LONG"
                signal.entry_price = close
                signal.stop_loss = stop_loss
                signal.position_size_pct = position_pct
                signal.strength = min(abs(row.get("ma_diff", 0)) / 3.0, 1.0)

        return signal

    def update_trailing_stop(self, position: TrendPosition,
                             current_high: float, current_atr: float):
        """外部调用更新移动止损"""
        if position.direction == "LONG":
            position.highest_since_entry = max(position.highest_since_entry, current_high)
            new_stop = position.highest_since_entry - self.trailing_atr_mult * current_atr
            position.trailing_stop = max(position.trailing_stop, new_stop)

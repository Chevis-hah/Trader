"""
Triple EMA Strategy — v2.3 (回滚至 v2.0，候选策略，不单独上线)

变更历史:
  v2.0: 三 EMA 顺排 + ADX 过滤 → 33 folds 仅 11 trades
  v2.2: + 过度拉伸过滤 → 样本仍不足 (11 trades, fold WR 15.2%)
  v2.3: 回滚到 v2.0，标记为 enabled=false，仅作候选

失效证据:
  - wf_triple_ema_4h.json: 11 trades / 33 folds, fold WR 15.2%
  - 虽 OOS PnL +1519, 但完全由 2-3 笔极端赢家驱动
  - sensitivity_triple_ema.json 显示 good_ratio=1.0，但这是因为样本太少参数空间"看起来平"

本策略在 v2.3 中默认 disabled。
保留文件供未来若 universe 扩到 30+ 标的后重新评估。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("triple_ema")


@dataclass
class TripleEMAConfig:
    """v2.3 回滚版配置"""
    # EMA 周期
    ema_short: int = 20
    ema_medium: int = 50
    ema_long: int = 200

    # 入场过滤
    min_adx: float = 20.0

    # 止损
    stop_atr_mult: float = 1.8
    trail_atr_mult: float = 2.8

    # 仓位
    risk_per_trade: float = 0.02
    cooldown_bars: int = 3

    # ---- 已删除 (v2.2 → v2.3) ----
    # zscore_threshold          — 过度拉伸过滤，统计无意义
    # ma_dev_threshold          — 同上
    # keltner_threshold         — 同上
    # max_holding_bars          — good_ratio 低


class TripleEMAStrategy:
    """
    三 EMA 顺排趋势策略 v2.3

    核心逻辑:
      1. EMA20 > EMA50 > EMA200 (多头排列)
      2. 收盘价上穿 EMA20 (入场触发)
      3. ADX >= min_adx
      4. ATR 止损 + 跟踪
    """

    def __init__(self, config: Optional[TripleEMAConfig] = None):
        self.cfg = config or TripleEMAConfig()

    def prepare_features(
        self,
        features: pd.DataFrame,
        higher_tf: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算 EMA 指标 (假设 FeatureEngine 可能没算三条)
        """
        df = features.copy()

        for period in [self.cfg.ema_short, self.cfg.ema_medium, self.cfg.ema_long]:
            col = f"ema_{period}"
            if col not in df.columns and "close" in df.columns:
                df[col] = df["close"].ewm(span=period, adjust=False).mean()

        return df

    def should_enter(
        self, row: pd.Series, prev: pd.Series, state: dict
    ) -> bool:
        """入场判断 (v2.0 纯规则)"""
        cooldown_until = state.get("cooldown_until_bar", -1)
        if state.get("bar_index", 0) < cooldown_until:
            return False

        ema_s = row.get(f"ema_{self.cfg.ema_short}")
        ema_m = row.get(f"ema_{self.cfg.ema_medium}")
        ema_l = row.get(f"ema_{self.cfg.ema_long}")
        close = row.get("close")
        prev_close = prev.get("close")
        prev_ema_s = prev.get(f"ema_{self.cfg.ema_short}")

        if any(pd.isna(x) for x in [ema_s, ema_m, ema_l, close, prev_close, prev_ema_s]):
            return False

        # 多头排列
        if not (ema_s > ema_m > ema_l):
            return False

        # 价格上穿 EMA20
        if not (prev_close <= prev_ema_s and close > ema_s):
            return False

        # ADX 过滤
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
        close = float(row.get("close", 0) or 0)
        if close <= 0:
            return False, ""

        stop_loss = position.get("stop_loss", 0)
        if stop_loss > 0 and close <= stop_loss:
            return True, "stop_loss"

        highest = position.get("highest_since_entry", close)
        atr_at_entry = position.get("atr_at_entry", 0)
        if atr_at_entry > 0:
            trail_stop = highest - self.cfg.trail_atr_mult * atr_at_entry
            if close <= trail_stop:
                return True, "trail_stop"

        # EMA 排列破坏则退出
        ema_s = row.get(f"ema_{self.cfg.ema_short}")
        ema_m = row.get(f"ema_{self.cfg.ema_medium}")
        if not pd.isna(ema_s) and not pd.isna(ema_m) and ema_s < ema_m:
            return True, "ema_cross_down"

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

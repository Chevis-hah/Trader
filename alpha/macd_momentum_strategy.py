"""
低频 4h 趋势跟随策略：MACD Momentum

设计目标：
- 只在大级别顺风时做 MACD 柱状图翻正的动量启动
- 减少震荡区反复开仓
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("macd_momentum_strategy")


@dataclass
class MACDMomentumStrategyConfig:
    primary_interval: str = "4h"
    higher_interval: str = "1d"

    trend_ema: int = 50
    daily_fast: int = 20
    daily_slow: int = 50

    min_adx: float = 16.0
    min_rsi: float = 48.0
    max_rsi: float = 74.0
    min_volume_ratio: float = 0.90

    trail_atr_mult: float = 2.5
    stop_atr_mult: float = 2.0
    max_holding_bars: int = 36
    cooldown_bars: int = 5

    risk_per_trade: float = 0.012
    max_position_pct: float = 0.45
    min_trade_value: float = 12.0

    commission_pct: float = 0.001
    slippage_pct: float = 0.001


class MACDMomentumStrategy:
    name = "macd_momentum"
    display_name = "MACD Momentum"

    def __init__(self, cfg: Optional[MACDMomentumStrategyConfig] = None):
        self.cfg = cfg or MACDMomentumStrategyConfig()

    @property
    def primary_interval(self) -> str:
        return self.cfg.primary_interval

    @property
    def higher_interval(self) -> str:
        return self.cfg.higher_interval

    def prepare_features(
        self,
        features: pd.DataFrame,
        higher_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        feat = features.copy()

        if "ema_50" not in feat.columns:
            close = pd.to_numeric(feat["close"], errors="coerce")
            feat["ema_50"] = close.ewm(span=self.cfg.trend_ema, adjust=False).mean()

        if higher_features is not None and not higher_features.empty:
            hf = higher_features.copy()
            hf["ht_ema_fast"] = pd.to_numeric(hf["close"], errors="coerce").ewm(
                span=self.cfg.daily_fast, adjust=False
            ).mean()
            hf["ht_ema_slow"] = pd.to_numeric(hf["close"], errors="coerce").ewm(
                span=self.cfg.daily_slow, adjust=False
            ).mean()
            hf["daily_trend_ok"] = (
                (pd.to_numeric(hf["close"], errors="coerce") > hf["ht_ema_fast"])
                & (hf["ht_ema_fast"] > hf["ht_ema_slow"])
            ).astype(float)

            keep_cols = ["open_time", "ht_ema_fast", "ht_ema_slow", "daily_trend_ok"]
            feat = pd.merge_asof(
                feat.sort_values("open_time"),
                hf[keep_cols].sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
        else:
            feat["daily_trend_ok"] = 1.0

        return feat

    def should_enter(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        state: Optional[dict] = None,
    ) -> bool:
        state = state or {}

        close = float(row.get("close", 0) or 0)
        ema50 = row.get("ema_50")
        adx = row.get("adx_14")
        rsi = row.get("rsi_14")
        hist = row.get("macd_hist")
        prev_hist = prev_row.get("macd_hist") if prev_row is not None else None
        vol_ratio = row.get("rel_volume_20", 1.0)
        daily_ok = row.get("daily_trend_ok", 1.0)
        natr = row.get("natr_20")

        if close <= 0 or any(pd.isna(v) for v in [ema50, adx, rsi, hist, prev_hist, natr]):
            return False
        if natr <= 0 or daily_ok < 0.5:
            return False

        if close <= ema50:
            return False
        if adx < self.cfg.min_adx:
            return False
        if rsi < self.cfg.min_rsi or rsi > self.cfg.max_rsi:
            return False
        if pd.notna(vol_ratio) and vol_ratio < self.cfg.min_volume_ratio:
            return False

        if not (prev_hist <= 0 < hist):
            return False

        cooldown_until = state.get("cooldown_until_bar", -1)
        current_bar = state.get("bar_index", 0)
        if current_bar <= cooldown_until:
            return False

        return True

    def calc_position(self, equity: float, entry_price: float, row: pd.Series) -> tuple[float, float]:
        natr = float(row.get("natr_20", 0) or 0)
        if entry_price <= 0 or natr <= 0:
            return 0.0, 0.0

        atr = natr * entry_price
        stop_loss = entry_price - self.cfg.stop_atr_mult * atr
        risk_per_unit = entry_price - stop_loss
        if risk_per_unit <= 0:
            return 0.0, 0.0

        risk_amount = equity * self.cfg.risk_per_trade
        qty = risk_amount / risk_per_unit

        max_value = equity * self.cfg.max_position_pct
        position_value = qty * entry_price
        if position_value > max_value:
            qty = max_value / entry_price

        if qty * entry_price < self.cfg.min_trade_value:
            return 0.0, 0.0

        return qty, stop_loss

    def check_exit(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        position: dict,
        bar_count: int,
    ) -> tuple[bool, str]:
        close = float(row.get("close", 0) or 0)
        natr = float(row.get("natr_20", 0) or 0)
        daily_ok = row.get("daily_trend_ok", 1.0)
        ema50 = row.get("ema_50")
        macd_line = row.get("macd_line")
        macd_signal = row.get("macd_signal")
        prev_macd_line = prev_row.get("macd_line") if prev_row is not None else None
        prev_macd_signal = prev_row.get("macd_signal") if prev_row is not None else None

        atr = natr * close if natr > 0 else position.get("atr_at_entry", 0.0)
        highest = position.get("highest_since_entry", close)
        init_stop = position.get("stop_loss", 0.0)

        if atr > 0:
            trailing = max(init_stop, highest - self.cfg.trail_atr_mult * atr)
            if close <= trailing:
                return True, "trailing_stop"

        if (
            pd.notna(macd_line) and pd.notna(macd_signal)
            and pd.notna(prev_macd_line) and pd.notna(prev_macd_signal)
            and macd_line < macd_signal
            and prev_macd_line >= prev_macd_signal
        ):
            return True, "macd_cross"

        if daily_ok < 0.5 and pd.notna(ema50) and close < ema50:
            return True, "daily_trend_break"

        if bar_count >= self.cfg.max_holding_bars and close < position["entry_price"] * 1.01:
            return True, "time_stop"

        return False, ""

    def on_trade_closed(self, state: dict, bar_index: int, reason: str):
        if reason in {"trailing_stop", "macd_cross", "daily_trend_break", "time_stop"}:
            state["cooldown_until_bar"] = bar_index + self.cfg.cooldown_bars

    def signal_metadata(self, row: pd.Series) -> dict:
        return {
            "tag": "momentum_flip",
            "strength": float(min(max(row.get("adx_14", 0) / 35.0, 0.0), 1.0)),
            "context": {
                "macd_hist": round(float(row.get("macd_hist", 0) or 0), 6),
                "rsi": round(float(row.get("rsi_14", 0) or 0), 2),
                "daily_trend_ok": int((row.get("daily_trend_ok", 1.0) or 0) >= 0.5),
            },
        }

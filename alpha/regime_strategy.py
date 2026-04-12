"""
更严格的 Regime 趋势策略

这版不再把 regime 当成“频繁试错框架”，而是把它降级为：
- 强趋势过滤器
- 趋势回踩修复入场器
- 尽量少做，只做高质量 4h/1d 顺势单
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("regime_strategy")


REGIME_BULL_TREND = "BULL_TREND"
REGIME_BULL_WEAK = "BULL_WEAK"
REGIME_RANGE = "RANGE"
REGIME_BEAR_WEAK = "BEAR_WEAK"
REGIME_BEAR_TREND = "BEAR_TREND"


def classify_regime(row: pd.Series) -> str:
    close = row.get("close", 0)
    ema20 = row.get("ema_20", close)
    ema50 = row.get("ema_50", close)
    ema200 = row.get("ema_200", close)
    adx = row.get("adx_14", 0)
    trend_strength = row.get("trend_strength", 0)

    if any(pd.isna(v) for v in [ema20, ema50, ema200, adx]):
        return REGIME_RANGE

    score = 0.0
    score += 1.0 if close > ema50 else -1.0
    score += 0.5 if ema20 > ema50 else -0.5
    score += 0.5 if close > ema200 else -0.5
    score += 0.5 if trend_strength and trend_strength > 0 else -0.5

    if score >= 2.0 and adx >= 20:
        return REGIME_BULL_TREND
    if score >= 1.0 and adx >= 16:
        return REGIME_BULL_WEAK
    if score <= -2.0 and adx >= 20:
        return REGIME_BEAR_TREND
    if score <= -1.0:
        return REGIME_BEAR_WEAK
    return REGIME_RANGE


def add_regime_column(features: pd.DataFrame) -> pd.DataFrame:
    feat = features.copy()
    feat["regime"] = feat.apply(classify_regime, axis=1)
    return feat


@dataclass
class RegimeStrategyConfig:
    primary_interval: str = "4h"
    higher_interval: str = "1d"

    rsi_low: float = 45.0
    rsi_high: float = 68.0
    adx_min: float = 20.0
    adx_min_weak: float = 24.0
    dist_ema_max_atr: float = 0.70
    dist_ema_min_atr: float = -1.00
    vol_ratio_min: float = 0.90

    trailing_atr_mult: float = 2.8
    take_profit_atr_mult: float = 6.0
    max_holding_bars: int = 40
    time_stop_min_profit: float = 0.010

    risk_per_trade: float = 0.012
    risk_per_trade_weak: float = 0.006
    stop_atr_mult: float = 2.0
    max_position_pct: float = 0.40
    min_trade_value: float = 12.0
    cooldown_bars: int = 6
    allow_weak_regime: bool = False

    commission_pct: float = 0.001
    slippage_pct: float = 0.001


class RegimeAdaptiveStrategy:
    name = "regime"
    display_name = "Regime Adaptive"

    def __init__(self, cfg: Optional[RegimeStrategyConfig] = None):
        self.cfg = cfg or RegimeStrategyConfig()

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
        feat = add_regime_column(features)

        if higher_features is not None and not higher_features.empty:
            hf = add_regime_column(higher_features)
            hf["daily_trend_ok"] = hf["regime"].isin([REGIME_BULL_TREND, REGIME_BULL_WEAK]).astype(float)
            keep_cols = ["open_time", "daily_trend_ok", "regime"]
            hf = hf.rename(columns={"regime": "higher_regime"})
            feat = pd.merge_asof(
                feat.sort_values("open_time"),
                hf[["open_time", "daily_trend_ok", "higher_regime"]].sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
        else:
            feat["daily_trend_ok"] = 1.0
            feat["higher_regime"] = REGIME_BULL_TREND

        return feat

    def should_enter(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        state: Optional[dict] = None,
    ) -> bool:
        state = state or {}

        regime = row.get("regime", REGIME_RANGE)
        allowed_regimes = {REGIME_BULL_TREND}
        if self.cfg.allow_weak_regime:
            allowed_regimes.add(REGIME_BULL_WEAK)
        if regime not in allowed_regimes:
            return False

        if row.get("daily_trend_ok", 1.0) < 0.5:
            return False

        close = float(row.get("close", 0) or 0)
        ema20 = row.get("ema_20", close)
        ema50 = row.get("ema_50", close)
        ema200 = row.get("ema_200", close)
        adx = row.get("adx_14", 0)
        rsi = row.get("rsi_14", 50)
        natr = row.get("natr_20", 0)
        vol_ratio = row.get("rel_volume_20", 1.0)

        if close <= 0 or any(pd.isna(v) for v in [ema20, ema50, ema200, adx, rsi, natr]):
            return False
        if natr <= 0:
            return False
        if not (ema20 > ema50 > ema200):
            return False

        if rsi < self.cfg.rsi_low or rsi > self.cfg.rsi_high:
            return False

        adx_threshold = self.cfg.adx_min_weak if regime == REGIME_BULL_WEAK else self.cfg.adx_min
        if adx < adx_threshold:
            return False

        if pd.notna(vol_ratio) and vol_ratio < self.cfg.vol_ratio_min:
            return False

        if prev_row is None:
            return False

        atr = natr * close
        dist_atr = (close - ema20) / atr if atr > 0 else 999.0
        if dist_atr < self.cfg.dist_ema_min_atr or dist_atr > self.cfg.dist_ema_max_atr:
            return False

        prev_close = float(prev_row.get("close", close) or close)
        prev_ema20 = prev_row.get("ema_20", ema20)

        pullback_reclaim = prev_close <= prev_ema20 * 1.002 and close > ema20
        momentum_ok = row.get("macd_hist", 0) > 0
        if not (pullback_reclaim and momentum_ok):
            return False

        cooldown_until = state.get("cooldown_until_bar", -1)
        current_bar = state.get("bar_index", 0)
        if current_bar <= cooldown_until:
            return False

        return True

    def calc_position(self, equity: float, entry_price: float, row: pd.Series) -> tuple[float, float]:
        regime = row.get("regime", REGIME_RANGE)
        natr = float(row.get("natr_20", 0) or 0)
        if entry_price <= 0 or natr <= 0:
            return 0.0, 0.0

        atr = natr * entry_price
        stop_loss = entry_price - self.cfg.stop_atr_mult * atr
        risk_per_unit = entry_price - stop_loss
        if risk_per_unit <= 0:
            return 0.0, 0.0

        risk_pct = self.cfg.risk_per_trade_weak if regime == REGIME_BULL_WEAK else self.cfg.risk_per_trade
        risk_amount = equity * risk_pct
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
        regime = row.get("regime", REGIME_RANGE)
        natr = float(row.get("natr_20", 0) or 0)
        ema20 = row.get("ema_20")
        ema50 = row.get("ema_50")

        atr = natr * close if natr > 0 else position.get("atr_at_entry", 0.0)
        highest = position.get("highest_since_entry", close)
        init_stop = position.get("stop_loss", 0.0)

        if regime == REGIME_BEAR_TREND:
            return True, "regime_bear"

        if row.get("daily_trend_ok", 1.0) < 0.5 and pd.notna(ema20) and close < ema20:
            return True, "daily_trend_break"

        if atr > 0:
            trailing = max(init_stop, highest - self.cfg.trailing_atr_mult * atr)
            if close <= trailing:
                return True, "trailing_stop"

        if pd.notna(ema20) and pd.notna(ema50) and ema20 < ema50:
            return True, "trend_break"

        entry_price = position["entry_price"]
        target = entry_price + self.cfg.take_profit_atr_mult * position.get("atr_at_entry", 0.0)
        if target > entry_price and close >= target:
            return True, "take_profit"

        if bar_count >= self.cfg.max_holding_bars and close <= entry_price * (1 + self.cfg.time_stop_min_profit):
            return True, "time_stop"

        return False, ""

    def on_trade_closed(self, state: dict, bar_index: int, reason: str):
        if reason in {"regime_bear", "daily_trend_break", "trailing_stop", "trend_break", "time_stop"}:
            state["cooldown_until_bar"] = bar_index + self.cfg.cooldown_bars

    def signal_metadata(self, row: pd.Series) -> dict:
        return {
            "tag": str(row.get("regime", REGIME_RANGE)).lower(),
            "strength": float(min(max(row.get("adx_14", 0) / 40.0, 0.0), 1.0)),
            "context": {
                "regime": row.get("regime", REGIME_RANGE),
                "higher_regime": row.get("higher_regime", REGIME_RANGE),
                "rsi": round(float(row.get("rsi_14", 0) or 0), 2),
                "adx": round(float(row.get("adx_14", 0) or 0), 2),
            },
        }

"""
P1: 做空能力扩展

设计:
- 不修改原策略代码，通过包装器模式添加做空逻辑
- 做空信号 = 做多信号的镜像: 死叉/趋势下破 -> 做空
- 做空使用 Binance USDT-M Futures API

使用方式:
  在 backtest_runner.py 中:
    from alpha.bidirectional_wrapper import BidirectionalWrapper
    strategy = BidirectionalWrapper(base_strategy, enable_short=True)
"""
from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("bidirectional")


@dataclass
class BidirectionalConfig:
    """做空参数"""
    enable_short: bool = True
    short_risk_multiplier: float = 0.8   # 做空仓位 = 做多的 80%（波动率更高）
    short_trail_multiplier: float = 1.2  # 做空追踪止损更紧（向上波动更快）
    max_short_holding_bars: int = 24     # 做空持仓不宜太久
    short_cooldown_bars: int = 4


class BidirectionalWrapper:
    """
    双向交易包装器 — 在不修改原策略的前提下添加做空能力

    原理:
    - 做多: 完全委托给原策略
    - 做空: 检测原策略的 "反向条件"
      - Triple EMA: EMA8 < EMA21 < EMA55 (空头排列) + MACD 柱翻负
      - MACD Momentum: MACD 柱从正翻负 + 价格在 EMA50 下方
    """

    def __init__(self, base_strategy, bi_cfg: BidirectionalConfig = None):
        self.base = base_strategy
        self.bi_cfg = bi_cfg or BidirectionalConfig()

        # 继承基础策略属性
        self.name = f"{base_strategy.name}_bidirectional"
        self.display_name = f"{base_strategy.display_name} (Long+Short)"
        self.cfg = base_strategy.cfg

    @property
    def primary_interval(self) -> str:
        return self.base.primary_interval

    @property
    def higher_interval(self) -> str:
        return self.base.higher_interval

    def prepare_features(self, features, higher_features=None):
        return self.base.prepare_features(features, higher_features)

    def should_enter(self, row, prev_row, state=None) -> bool:
        """做多入场 = 原策略逻辑"""
        return self.base.should_enter(row, prev_row, state)

    def should_enter_short(self, row, prev_row, state=None) -> bool:
        """
        做空入场 — 原策略条件的镜像

        根据基础策略类型自动适配:
        - triple_ema: 空头排列 + 回弹到 EMA21 遇阻
        - macd_momentum: MACD 柱翻负 + 价格在 EMA50 下方
        """
        if not self.bi_cfg.enable_short:
            return False

        state = state or {}
        cooldown_until = state.get("short_cooldown_until_bar", -1)
        current_bar = state.get("bar_index", 0)
        if current_bar <= cooldown_until:
            return False

        close = float(row.get("close", 0) or 0)
        if close <= 0:
            return False

        natr = row.get("natr_20")
        adx = row.get("adx_14")
        rsi = row.get("rsi_14")
        daily_ok = row.get("daily_trend_ok", 1.0)

        if any(pd.isna(v) for v in [natr, adx]):
            return False
        if natr <= 0:
            return False

        strategy_type = getattr(self.base, "name", "")

        if "triple_ema" in strategy_type:
            return self._short_triple_ema(row, prev_row, close, adx, rsi, daily_ok)
        elif "macd" in strategy_type:
            return self._short_macd(row, prev_row, close, adx, rsi, daily_ok)
        else:
            return False

    def _short_triple_ema(self, row, prev_row, close, adx, rsi, daily_ok) -> bool:
        """Triple EMA 做空: 空头排列 + 反弹到 EMA21 遇阻"""
        e8 = row.get("ema_8")
        e21 = row.get("ema_21")
        e55 = row.get("ema_55")

        if any(pd.isna(v) for v in [e8, e21, e55]):
            return False

        # 空头排列
        if not (e8 < e21 < e55):
            return False

        # ADX 确认趋势存在
        if adx < self.base.cfg.min_adx:
            return False

        # RSI 不能超卖（避免超跌反弹）
        if pd.notna(rsi) and rsi < 30:
            return False

        # 日线下行或至少不上行
        if daily_ok > 0.5:
            return False

        # 反弹到 EMA21 附近后再次下跌
        natr = float(row.get("natr_20", 0))
        atr = natr * close
        if atr <= 0:
            return False

        dist = (close - e21) / atr
        if dist > 0.5 or dist < -1.5:
            return False

        # 需要确认: 价格从上方回落到 EMA21
        if prev_row is not None:
            prev_close = float(prev_row.get("close", close))
            prev_e21 = prev_row.get("ema_21")
            if pd.notna(prev_e21):
                # 之前在 EMA21 上方或附近，现在跌破
                if prev_close > prev_e21 and close < e21:
                    return True

        # MACD 翻负确认
        macd_hist = row.get("macd_hist", 0)
        if pd.notna(macd_hist) and macd_hist < 0:
            if prev_row is not None:
                prev_hist = prev_row.get("macd_hist", 0)
                if pd.notna(prev_hist) and prev_hist >= 0:
                    return True

        return False

    def _short_macd(self, row, prev_row, close, adx, rsi, daily_ok) -> bool:
        """MACD Momentum 做空: MACD 柱翻负 + 价格在 EMA50 下方"""
        ema50 = row.get("ema_50")
        hist = row.get("macd_hist")

        if pd.isna(ema50) or pd.isna(hist):
            return False

        # 价格在 EMA50 下方
        if close >= ema50:
            return False

        # ADX 确认
        if adx < self.base.cfg.min_adx:
            return False

        # RSI 不能太低（避免超卖反弹）
        if pd.notna(rsi) and rsi < 32:
            return False

        # 日线不看多
        if daily_ok > 0.5:
            return False

        # MACD 柱翻负
        if prev_row is not None:
            prev_hist = prev_row.get("macd_hist")
            if pd.notna(prev_hist) and prev_hist >= 0 and hist < 0:
                return True

        return False

    def calc_short_position(self, equity: float, entry_price: float,
                            row: pd.Series) -> tuple[float, float]:
        """做空仓位计算 — 比做多更保守"""
        qty, _ = self.base.calc_position(equity, entry_price, row)
        if qty <= 0:
            return 0.0, 0.0

        # 做空仓位缩减
        qty *= self.bi_cfg.short_risk_multiplier

        # 止损在上方
        natr = float(row.get("natr_20", 0) or 0)
        if natr <= 0:
            return 0.0, 0.0
        atr = natr * entry_price
        stop_loss = entry_price + self.base.cfg.stop_atr_mult * atr

        return qty, stop_loss

    def check_exit_short(self, row, prev_row, position: dict,
                         bar_count: int) -> tuple[bool, str]:
        """做空出场"""
        close = float(row.get("close", 0) or 0)
        if close <= 0:
            return False, ""

        natr = float(row.get("natr_20", 0) or 0)
        atr = natr * close if natr > 0 else position.get("atr_at_entry", 0.0)

        entry_price = position.get("entry_price", close)
        lowest = position.get("lowest_since_entry", close)
        init_stop = position.get("stop_loss", entry_price * 1.1)

        # 追踪止损（做空时从下方往上追）
        if atr > 0:
            trail_mult = self.base.cfg.trail_atr_mult * self.bi_cfg.short_trail_multiplier
            trailing = min(init_stop, lowest + trail_mult * atr)
            if close >= trailing:
                return True, "short_trailing_stop"

        # MACD 翻正 -> 止损
        strategy_type = getattr(self.base, "name", "")
        if "macd" in strategy_type:
            macd_line = row.get("macd_line")
            macd_signal = row.get("macd_signal")
            prev_ml = prev_row.get("macd_line") if prev_row is not None else None
            prev_ms = prev_row.get("macd_signal") if prev_row is not None else None
            if (pd.notna(macd_line) and pd.notna(macd_signal)
                    and pd.notna(prev_ml) and pd.notna(prev_ms)
                    and macd_line > macd_signal
                    and prev_ml <= prev_ms):
                return True, "short_macd_cross"

        # EMA 翻多 -> 止损
        if "triple_ema" in strategy_type:
            e8 = row.get("ema_8")
            e21 = row.get("ema_21")
            if pd.notna(e8) and pd.notna(e21) and e8 > e21:
                return True, "short_ema_cross"

        # 最大持仓时间
        if bar_count >= self.bi_cfg.max_short_holding_bars:
            if close > entry_price * 0.99:  # 做空亏钱才强平
                return True, "short_time_stop"

        return False, ""

    def on_short_closed(self, state: dict, bar_index: int, reason: str):
        """做空平仓后冷却"""
        state["short_cooldown_until_bar"] = bar_index + self.bi_cfg.short_cooldown_bars

    # ---- 委托给基础策略的方法 ----
    def calc_position(self, *args, **kwargs):
        return self.base.calc_position(*args, **kwargs)

    def check_exit(self, *args, **kwargs):
        return self.base.check_exit(*args, **kwargs)

    def on_trade_closed(self, *args, **kwargs):
        return self.base.on_trade_closed(*args, **kwargs)

    def signal_metadata(self, *args, **kwargs):
        return self.base.signal_metadata(*args, **kwargs)

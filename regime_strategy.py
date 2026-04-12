"""
政体自适应趋势策略 (Regime-Adaptive Trend Following)

设计理念：
  原版 trend_following.py 的核心问题是「熊市照样做多」。
  本策略在其基础上增加政体识别层，用 FeatureEngine 已经计算好的
  ema_20/50/200、adx_14、rsi_14 等特征直接判断市场状态，
  只在牛市政体中做多，熊市/震荡市保持空仓。

与原版的关系：
  - 输入：FeatureEngine.compute_all() 的输出 DataFrame
  - 输出：与 AlphaModel.predict() 兼容的信号 DataFrame
  - 不替换原版，而是作为一个可选策略并行运行

依赖的 FeatureEngine 特征列：
  ema_20, ema_50, ema_200, close_vs_sma_50, close_vs_sma_200,
  adx_14, rsi_14, rel_volume_20, natr_20, trend_strength, above_sma200
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

logger = get_logger("regime_strategy")


# ============================================================
# 政体识别
# ============================================================

REGIME_BULL_TREND = "BULL_TREND"   # 强上升 — 可做多
REGIME_BULL_WEAK  = "BULL_WEAK"    # 弱上升 — 谨慎做多，仓位减半
REGIME_RANGE      = "RANGE"        # 震荡   — 观望
REGIME_BEAR_WEAK  = "BEAR_WEAK"    # 弱下降 — 观望
REGIME_BEAR_TREND = "BEAR_TREND"   # 强下降 — 观望（现货无法做空）


def classify_regime(row: pd.Series) -> str:
    """
    基于 FeatureEngine 已计算的特征判断政体

    评分体系（-2 ~ +2）:
      close vs EMA50  → ±1
      EMA20 vs EMA50  → ±0.5
      close vs EMA200 → ±0.5
    再结合 ADX 判断趋势强度
    """
    close = row.get("close", 0)
    ema20 = row.get("ema_20", close)
    ema50 = row.get("ema_50", close)
    ema200 = row.get("ema_200", close)
    adx = row.get("adx_14", 0)

    if pd.isna(ema200) or pd.isna(adx):
        return REGIME_RANGE

    # 方向评分
    score = 0.0
    score += 1.0  if close > ema50  else -1.0
    score += 0.5  if ema20 > ema50  else -0.5
    score += 0.5  if close > ema200 else -0.5

    # 分类
    if score >= 1.5 and adx >= 20:
        return REGIME_BULL_TREND
    elif score >= 0.5:
        return REGIME_BULL_WEAK
    elif score <= -1.5 and adx >= 20:
        return REGIME_BEAR_TREND
    elif score <= -0.5:
        return REGIME_BEAR_WEAK
    else:
        # 低 ADX 且评分中性
        return REGIME_RANGE


def add_regime_column(features: pd.DataFrame) -> pd.DataFrame:
    """给特征表加上 regime 列"""
    features = features.copy()
    features["regime"] = features.apply(classify_regime, axis=1)
    return features


# ============================================================
# 策略参数
# ============================================================

@dataclass
class RegimeStrategyConfig:
    """策略可调参数，集中管理"""
    # 入场过滤
    rsi_low: float = 35.0           # RSI 下限（低于则过度超卖，不追）
    rsi_high: float = 65.0          # RSI 上限（高于则超买，不追）
    adx_min: float = 18.0           # 最小 ADX（排除无趋势行情）
    adx_min_weak: float = 22.0      # BULL_WEAK 下的最小 ADX
    dist_ema_max_atr: float = 2.0   # 价格距 EMA20 最大距离（ATR 倍数）
    dist_ema_min_atr: float = -1.5  # 价格距 EMA20 最小距离
    vol_ratio_min: float = 0.8      # 最小相对成交量

    # 出场
    trailing_atr_mult: float = 2.5  # 移动止损 ATR 倍数
    take_profit_atr_mult: float = 4.0  # 止盈 ATR 倍数
    partial_exit_atr_mult: float = 1.5  # 减仓触发 ATR 倍数
    partial_exit_pct: float = 0.3   # 减仓比例
    max_holding_bars: int = 30      # 最大持仓 bar 数（4h = 5天）
    time_stop_min_profit: float = 0.005  # 超时平仓的最低盈利阈值

    # 仓位
    risk_per_trade: float = 0.03    # 每笔风险占总权益比
    risk_per_trade_weak: float = 0.015  # BULL_WEAK 下减半
    stop_atr_mult: float = 2.0     # 初始止损 ATR 倍数
    max_position_pct: float = 0.50  # 最大仓位占比
    min_trade_value: float = 12.0   # 最小下单金额

    # 成本
    commission_pct: float = 0.001   # 手续费 0.1%
    slippage_pct: float = 0.001     # 滑点 0.1%


# ============================================================
# 策略核心
# ============================================================

class RegimeAdaptiveStrategy:
    """
    政体自适应趋势策略

    工作流：
      1. FeatureEngine 计算特征
      2. add_regime_column() 添加政体标签
      3. 本策略基于政体 + 特征生成信号
      4. 回测引擎或实盘引擎执行信号
    """

    def __init__(self, cfg: RegimeStrategyConfig = None):
        self.cfg = cfg or RegimeStrategyConfig()

    def should_enter(self, row: pd.Series, prev_row: pd.Series) -> bool:
        """
        判断是否应该做多入场

        前提：row 包含 FeatureEngine 的所有特征列 + regime 列
        """
        regime = row.get("regime", REGIME_RANGE)
        if regime not in (REGIME_BULL_TREND, REGIME_BULL_WEAK):
            return False

        close = row.get("close", 0)
        ema20 = row.get("ema_20", close)
        adx   = row.get("adx_14", 0)
        rsi   = row.get("rsi_14", 50)
        natr  = row.get("natr_20", 0)

        if pd.isna(adx) or pd.isna(rsi) or natr <= 0:
            return False

        # RSI 范围过滤
        if rsi < self.cfg.rsi_low or rsi > self.cfg.rsi_high:
            return False

        # 价格相对 EMA20 的距离（用 ATR 归一化）
        atr = natr * close  # natr = atr / close，所以 atr = natr * close
        if atr <= 0:
            return False
        dist_atr = (close - ema20) / atr
        if dist_atr < self.cfg.dist_ema_min_atr or dist_atr > self.cfg.dist_ema_max_atr:
            return False

        # 成交量确认
        vol_ratio = row.get("rel_volume_20", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        if vol_ratio < self.cfg.vol_ratio_min:
            return False

        # ADX 最低要求
        adx_threshold = self.cfg.adx_min_weak if regime == REGIME_BULL_WEAK else self.cfg.adx_min
        if adx < adx_threshold:
            return False

        # 前一根 K 线收阳 或 当前站上 EMA20
        if prev_row is not None:
            prev_close = prev_row.get("close", 0)
            prev_open  = prev_row.get("close", 0)  # features 表没有 open，用 close 近似
            # 用 ret_5 > 0 替代「收阳」判断
            prev_ret = prev_row.get("ret_5", 0)
            above_ema = close > ema20
            if not (above_ema or (prev_ret is not None and not pd.isna(prev_ret) and prev_ret > 0)):
                return False

        return True

    def calc_position(self, equity: float, entry_price: float,
                      natr: float, regime: str) -> tuple[float, float]:
        """
        计算仓位和止损价

        返回: (quantity, stop_loss_price)
        """
        atr = natr * entry_price  # 还原绝对 ATR
        stop_distance = self.cfg.stop_atr_mult * atr
        stop_loss = entry_price - stop_distance

        if stop_distance <= 0:
            return 0.0, 0.0

        risk_pct = self.cfg.risk_per_trade_weak if regime == REGIME_BULL_WEAK \
            else self.cfg.risk_per_trade
        risk_amount = equity * risk_pct

        qty = risk_amount / stop_distance
        position_value = qty * entry_price

        # 上限
        max_value = equity * self.cfg.max_position_pct
        if position_value > max_value:
            qty = max_value / entry_price

        # 最小交易额
        if qty * entry_price < self.cfg.min_trade_value:
            return 0.0, 0.0

        return qty, stop_loss

    def check_exit(self, row: pd.Series, position: dict,
                   bar_count: int) -> tuple[bool, str]:
        """
        检查出场条件

        position 字典需要包含:
          entry_price, stop_loss, highest_since_entry, atr_at_entry
        """
        close  = row.get("close", 0)
        regime = row.get("regime", REGIME_RANGE)
        natr   = row.get("natr_20", 0)
        atr    = natr * close if natr > 0 else position.get("atr_at_entry", 1)

        entry_price = position["entry_price"]
        highest     = position["highest_since_entry"]
        init_stop   = position["stop_loss"]
        atr_entry   = position["atr_at_entry"]

        # 1. 政体翻熊 → 立刻退出
        if regime == REGIME_BEAR_TREND:
            return True, "regime_bear"

        # 2. 弱熊 + 浮亏 → 退出
        if regime == REGIME_BEAR_WEAK and close < entry_price:
            return True, "regime_weak_loss"

        # 3. 移动止损
        trailing = highest - self.cfg.trailing_atr_mult * atr
        trailing = max(trailing, init_stop)  # 不能低于初始止损
        if close <= trailing:
            return True, "trailing_stop"

        # 4. 止盈
        target = entry_price + self.cfg.take_profit_atr_mult * atr_entry
        if close >= target:
            return True, "take_profit"

        # 5. 超时 + 未盈利
        if bar_count >= self.cfg.max_holding_bars:
            if close <= entry_price * (1 + self.cfg.time_stop_min_profit):
                return True, "time_stop"

        return False, ""

    def check_partial_exit(self, row: pd.Series, position: dict) -> bool:
        """检查是否应该减仓（首次达到 1.5R）"""
        if position.get("partial_done", False):
            return False

        close = row.get("close", 0)
        high  = row.get("close", 0)  # features 表用 close 近似 high
        atr_entry = position["atr_at_entry"]
        target = position["entry_price"] + self.cfg.partial_exit_atr_mult * atr_entry

        return close >= target

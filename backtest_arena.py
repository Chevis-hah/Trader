"""
回测竞技场 — 10 种策略批量回测，一次跑完，横向对比

用法:
    python backtest_arena.py                          # 默认参数
    python backtest_arena.py --capital 10000           # 指定本金
    python backtest_arena.py --start 2025-01-01        # 指定起始
    python backtest_arena.py --snapshot arena_result.txt

输出:
    1. 控制台：10 策略横向对比表
    2. 快照文件：每个策略的详细交易记录（可喂给 Claude 分析）

策略列表:
    S01  TrendMTF        多时间框架趋势（日线方向 + 4h入场）
    S02  BreakoutDC      Donchian 通道突破
    S03  PullbackEMA     趋势回踩均线入场
    S04  RSIMeanRev      RSI 均值回归
    S05  BBSqueeze       布林带缩口突破
    S06  TripleEMA       三均线动量
    S07  MACDMomentum    MACD 动量趋势
    S08  VolBreakout     放量突破
    S09  GridRelaxed     宽松网格
    S10  KeltnerBreak    Keltner 通道突破
"""
import argparse
import sqlite3
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════
COMMISSION_RATE = 0.001     # 0.1% 单边手续费（Binance 现货 taker）
SLIPPAGE_RATE   = 0.0005    # 0.05% 滑点
SYMBOLS         = ["BTCUSDT", "ETHUSDT"]

# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class Trade:
    symbol: str = ""
    strategy: str = ""
    side: str = "LONG"
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_bars: int = 0
    exit_reason: str = ""
    commission: float = 0.0


@dataclass
class Position:
    symbol: str = ""
    direction: str = "FLAT"
    entry_price: float = 0.0
    entry_time: int = 0
    entry_bar: int = 0
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    highest: float = 0.0
    lowest: float = 999999999.0
    trailing_stop: float = 0.0
    atr_at_entry: float = 0.0


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_klines(db_path: str, symbol: str, interval: str,
                start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """从 SQLite 加载 K 线"""
    conn = sqlite3.connect(db_path)
    conds = ["symbol=?", "interval=?"]
    params = [symbol, interval]
    if start_date:
        ms = int(datetime.strptime(start_date, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp() * 1000)
        conds.append("open_time >= ?"); params.append(ms)
    if end_date:
        ms = int(datetime.strptime(end_date, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp() * 1000)
        conds.append("open_time <= ?"); params.append(ms)
    q = f"SELECT * FROM klines WHERE {' AND '.join(conds)} ORDER BY open_time ASC"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df


def resample_to_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    """4h K 线重采样为日线"""
    if df_4h.empty:
        return pd.DataFrame()
    df = df_4h.copy()
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms")
    df["date"] = df["dt"].dt.date
    daily = df.groupby("date").agg(
        open_time=("open_time", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index(drop=True)
    return daily


# ═══════════════════════════════════════════════════════════════
# 通用指标计算
# ═══════════════════════════════════════════════════════════════

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = calc_atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr.replace(0, 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr.replace(0, 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    return dx.rolling(period).mean()


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, 1e-10)
    return mid, upper, lower, width


def calc_donchian(df: pd.DataFrame, period: int = 20):
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    mid = (upper + lower) / 2
    return upper, lower, mid


def calc_keltner(df: pd.DataFrame, ema_period: int = 20, atr_mult: float = 2.0):
    mid = calc_ema(df["close"], ema_period)
    atr = calc_atr(df, ema_period)
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr
    return mid, upper, lower


def apply_costs(entry_price: float, exit_price: float, qty: float, side: str = "LONG"):
    """计算真实 PnL（含手续费+滑点）"""
    # 滑点
    entry_fill = entry_price * (1 + SLIPPAGE_RATE) if side == "LONG" else entry_price * (1 - SLIPPAGE_RATE)
    exit_fill  = exit_price * (1 - SLIPPAGE_RATE) if side == "LONG" else exit_price * (1 + SLIPPAGE_RATE)
    # 手续费
    comm = entry_fill * qty * COMMISSION_RATE + exit_fill * qty * COMMISSION_RATE
    # PnL
    if side == "LONG":
        raw_pnl = qty * (exit_fill - entry_fill)
    else:
        raw_pnl = qty * (entry_fill - exit_fill)
    net_pnl = raw_pnl - comm
    return net_pnl, comm, entry_fill, exit_fill


def ts_str(open_time_ms: int) -> str:
    return datetime.utcfromtimestamp(open_time_ms / 1000).strftime("%Y-%m-%d %H:%M")


# ═══════════════════════════════════════════════════════════════
# 策略基类
# ═══════════════════════════════════════════════════════════════

class BaseStrategy:
    name: str = "base"
    timeframe: str = "4h"
    description: str = ""

    def run(self, dfs: dict[str, pd.DataFrame], capital: float,
            daily_dfs: dict[str, pd.DataFrame] = None) -> list[Trade]:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════
# S01: 多时间框架趋势 — 日线定方向 + 4h 入场
# ═══════════════════════════════════════════════════════════════

class S01_TrendMTF(BaseStrategy):
    """
    核心思想：日线 EMA50 定大方向，4h EMA20/60 金叉入场
    vs 当前策略：增加日线过滤，避免逆势开仓（修复 ETH 连亏问题）
    """
    name = "S01_TrendMTF"
    description = "日线EMA50方向+4h金叉入场+3xATR追踪止损"

    def __init__(self, fast=20, slow=60, atr_period=14, trail_mult=3.0,
                 risk_pct=0.02, daily_ema=50):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.trail_mult = trail_mult
        self.risk_pct = risk_pct
        self.daily_ema = daily_ema

    def run(self, dfs, capital, daily_dfs=None):
        trades = []
        for symbol in SYMBOLS:
            df = dfs.get(symbol)
            if df is None or len(df) < 120:
                continue

            # 日线趋势方向
            d_df = daily_dfs.get(symbol) if daily_dfs else None
            daily_trend = self._get_daily_trend(d_df, df) if d_df is not None and len(d_df) > self.daily_ema else None

            # 4h 指标
            df = df.copy()
            df["ema_fast"] = calc_ema(df["close"], self.fast)
            df["ema_slow"] = calc_ema(df["close"], self.slow)
            df["atr"] = calc_atr(df, self.atr_period)
            df["ma_cross"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)

            cash = capital
            pos = Position()

            for i in range(1, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                atr = row["atr"]
                if pd.isna(atr) or atr <= 0:
                    continue

                close = row["close"]

                # 日线过滤：只在日线趋势方向做
                if daily_trend is not None:
                    dt = pd.to_datetime(row["open_time"], unit="ms").date()
                    d_dir = daily_trend.get(dt, 0)
                    # 找最近的日线方向
                    if d_dir == 0:
                        # 找前面最近的
                        for offset in range(1, 8):
                            prev_dt = dt - pd.Timedelta(days=offset)
                            if hasattr(prev_dt, 'date'):
                                prev_dt = prev_dt
                            d_dir = daily_trend.get(prev_dt, 0)
                            if d_dir != 0:
                                break
                else:
                    d_dir = 1  # 无日线数据，默认可做多

                # === 持仓中 ===
                if pos.direction == "LONG":
                    pos.highest = max(pos.highest, close)
                    new_trail = pos.highest - self.trail_mult * atr
                    pos.trailing_stop = max(pos.trailing_stop, new_trail)

                    exit_reason = ""
                    if close <= pos.trailing_stop:
                        exit_reason = "trailing_stop"
                    elif df.iloc[i]["ma_cross"] == -1 and df.iloc[i-1]["ma_cross"] == 1:
                        exit_reason = "ma_cross"

                    if exit_reason:
                        pnl, comm, _, _ = apply_costs(pos.entry_price, close, pos.quantity)
                        cash += pos.quantity * close * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                        trades.append(Trade(
                            symbol=symbol, strategy=self.name, side="LONG",
                            entry_price=pos.entry_price, exit_price=close,
                            quantity=pos.quantity, pnl=pnl, commission=comm,
                            pnl_pct=(close/pos.entry_price - 1),
                            entry_time=ts_str(pos.entry_time), exit_time=ts_str(row["open_time"]),
                            holding_bars=i - pos.entry_bar, exit_reason=exit_reason))
                        pos = Position()
                    continue

                # === 开仓条件 ===
                if (df.iloc[i]["ma_cross"] == 1 and df.iloc[i-1]["ma_cross"] != 1
                        and d_dir >= 0):  # 日线至少不看空
                    stop = close - 2.0 * atr
                    risk_amount = cash * self.risk_pct
                    risk_per_unit = close - stop
                    if risk_per_unit <= 0:
                        continue
                    qty = risk_amount / risk_per_unit
                    cost = qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if cost > cash * 0.50 or cost < 10:
                        cost = min(cash * 0.50, cost)
                        qty = cost / close / (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if qty * close < 10:
                        continue
                    cash -= qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    pos = Position(symbol=symbol, direction="LONG",
                                   entry_price=close, entry_time=row["open_time"],
                                   entry_bar=i, quantity=qty, stop_loss=stop,
                                   highest=close, trailing_stop=stop, atr_at_entry=atr)

            # 清仓
            if pos.direction == "LONG":
                last = df["close"].iloc[-1]
                pnl, comm, _, _ = apply_costs(pos.entry_price, last, pos.quantity)
                trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                    entry_price=pos.entry_price, exit_price=last, quantity=pos.quantity,
                    pnl=pnl, commission=comm, exit_reason="backtest_end",
                    pnl_pct=(last/pos.entry_price-1),
                    entry_time=ts_str(pos.entry_time), exit_time=ts_str(df["open_time"].iloc[-1]),
                    holding_bars=len(df)-1-pos.entry_bar))
        return trades

    def _get_daily_trend(self, d_df, df_4h):
        """构建日期 → 趋势方向的映射"""
        d = d_df.copy()
        d["ema"] = calc_ema(d["close"], self.daily_ema)
        trend = {}
        for _, row in d.iterrows():
            dt = pd.to_datetime(row["open_time"], unit="ms").date()
            if pd.isna(row["ema"]):
                trend[dt] = 0
            elif row["close"] > row["ema"]:
                trend[dt] = 1
            else:
                trend[dt] = -1
        return trend


# ═══════════════════════════════════════════════════════════════
# S02: Donchian 通道突破
# ═══════════════════════════════════════════════════════════════

class S02_BreakoutDC(BaseStrategy):
    """
    经典海龟交易法变种：价格突破 N 周期高点 + 成交量确认
    """
    name = "S02_BreakoutDC"
    description = "20周期Donchian突破+量能确认+2.5xATR追踪"

    def __init__(self, dc_period=20, vol_mult=1.3, trail_mult=2.5, risk_pct=0.02):
        self.dc_period = dc_period
        self.vol_mult = vol_mult
        self.trail_mult = trail_mult
        self.risk_pct = risk_pct

    def run(self, dfs, capital, daily_dfs=None):
        trades = []
        for symbol in SYMBOLS:
            df = dfs.get(symbol)
            if df is None or len(df) < self.dc_period + 20:
                continue
            df = df.copy()
            df["atr"] = calc_atr(df, 14)
            df["dc_high"], df["dc_low"], _ = calc_donchian(df, self.dc_period)
            df["vol_ma"] = df["volume"].rolling(20).mean()
            # 前一根的 dc_high（避免当根突破看到当根的最高点）
            df["dc_high_prev"] = df["dc_high"].shift(1)

            cash = capital
            pos = Position()

            for i in range(self.dc_period + 1, len(df)):
                row = df.iloc[i]
                atr = row["atr"]
                if pd.isna(atr) or atr <= 0:
                    continue
                close = row["close"]

                if pos.direction == "LONG":
                    pos.highest = max(pos.highest, close)
                    new_trail = pos.highest - self.trail_mult * atr
                    pos.trailing_stop = max(pos.trailing_stop, new_trail)

                    if close <= pos.trailing_stop:
                        pnl, comm, _, _ = apply_costs(pos.entry_price, close, pos.quantity)
                        cash += pos.quantity * close * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                        trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                            entry_price=pos.entry_price, exit_price=close, quantity=pos.quantity,
                            pnl=pnl, commission=comm, pnl_pct=(close/pos.entry_price-1),
                            entry_time=ts_str(pos.entry_time), exit_time=ts_str(row["open_time"]),
                            holding_bars=i-pos.entry_bar, exit_reason="trailing_stop"))
                        pos = Position()
                    continue

                # 突破 + 放量
                dc_prev = row.get("dc_high_prev", 0)
                if (pd.notna(dc_prev) and close > dc_prev
                        and row["volume"] > row["vol_ma"] * self.vol_mult):
                    stop = close - 2.0 * atr
                    risk_amount = cash * self.risk_pct
                    risk_per_unit = close - stop
                    if risk_per_unit <= 0:
                        continue
                    qty = risk_amount / risk_per_unit
                    cost = qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if cost > cash * 0.50:
                        cost = cash * 0.50
                        qty = cost / close / (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if qty * close < 10:
                        continue
                    cash -= qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    pos = Position(symbol=symbol, direction="LONG",
                                   entry_price=close, entry_time=row["open_time"],
                                   entry_bar=i, quantity=qty, stop_loss=stop,
                                   highest=close, trailing_stop=stop, atr_at_entry=atr)

            if pos.direction == "LONG":
                last = df["close"].iloc[-1]
                pnl, comm, _, _ = apply_costs(pos.entry_price, last, pos.quantity)
                trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                    entry_price=pos.entry_price, exit_price=last, quantity=pos.quantity,
                    pnl=pnl, commission=comm, exit_reason="backtest_end",
                    pnl_pct=(last/pos.entry_price-1),
                    entry_time=ts_str(pos.entry_time), exit_time=ts_str(df["open_time"].iloc[-1]),
                    holding_bars=len(df)-1-pos.entry_bar))
        return trades


# ═══════════════════════════════════════════════════════════════
# S03: 趋势回踩均线入场
# ═══════════════════════════════════════════════════════════════

class S03_PullbackEMA(BaseStrategy):
    """
    价格在 EMA50 上方（上升趋势），回踩 EMA21 附近入场
    RSI 不过热（< 65）确认还有上涨空间
    """
    name = "S03_PullbackEMA"
    description = "EMA50上方+回踩EMA21入场+RSI<65确认"

    def __init__(self, trend_ema=50, entry_ema=21, rsi_max=65,
                 pullback_atr=0.5, trail_mult=2.5, risk_pct=0.02):
        self.trend_ema = trend_ema
        self.entry_ema = entry_ema
        self.rsi_max = rsi_max
        self.pullback_atr = pullback_atr
        self.trail_mult = trail_mult
        self.risk_pct = risk_pct

    def run(self, dfs, capital, daily_dfs=None):
        trades = []
        for symbol in SYMBOLS:
            df = dfs.get(symbol)
            if df is None or len(df) < 100:
                continue
            df = df.copy()
            df["ema_trend"] = calc_ema(df["close"], self.trend_ema)
            df["ema_entry"] = calc_ema(df["close"], self.entry_ema)
            df["atr"] = calc_atr(df, 14)
            df["rsi"] = calc_rsi(df["close"], 14)

            cash = capital
            pos = Position()

            for i in range(2, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                atr = row["atr"]
                if pd.isna(atr) or atr <= 0 or pd.isna(row["rsi"]):
                    continue
                close = row["close"]

                if pos.direction == "LONG":
                    pos.highest = max(pos.highest, close)
                    new_trail = pos.highest - self.trail_mult * atr
                    pos.trailing_stop = max(pos.trailing_stop, new_trail)

                    exit_reason = ""
                    if close <= pos.trailing_stop:
                        exit_reason = "trailing_stop"
                    elif close < row["ema_trend"]:
                        exit_reason = "below_ema_trend"

                    if exit_reason:
                        pnl, comm, _, _ = apply_costs(pos.entry_price, close, pos.quantity)
                        cash += pos.quantity * close * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                        trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                            entry_price=pos.entry_price, exit_price=close, quantity=pos.quantity,
                            pnl=pnl, commission=comm, pnl_pct=(close/pos.entry_price-1),
                            entry_time=ts_str(pos.entry_time), exit_time=ts_str(row["open_time"]),
                            holding_bars=i-pos.entry_bar, exit_reason=exit_reason))
                        pos = Position()
                    continue

                # 开仓：趋势向上 + 回踩 + RSI 不过热
                if (close > row["ema_trend"]
                        and abs(close - row["ema_entry"]) < self.pullback_atr * atr
                        and prev["close"] <= prev["ema_entry"]  # 前一根在均线下/触及
                        and close > row["ema_entry"]             # 当前根回到均线上方
                        and row["rsi"] < self.rsi_max):
                    stop = row["ema_entry"] - 1.5 * atr
                    risk_amount = cash * self.risk_pct
                    risk_per_unit = close - stop
                    if risk_per_unit <= 0:
                        continue
                    qty = risk_amount / risk_per_unit
                    cost = qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if cost > cash * 0.50:
                        cost = cash * 0.50
                        qty = cost / close / (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if qty * close < 10:
                        continue
                    cash -= qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    pos = Position(symbol=symbol, direction="LONG",
                                   entry_price=close, entry_time=row["open_time"],
                                   entry_bar=i, quantity=qty, stop_loss=stop,
                                   highest=close, trailing_stop=stop, atr_at_entry=atr)

            if pos.direction == "LONG":
                last = df["close"].iloc[-1]
                pnl, comm, _, _ = apply_costs(pos.entry_price, last, pos.quantity)
                trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                    entry_price=pos.entry_price, exit_price=last, quantity=pos.quantity,
                    pnl=pnl, commission=comm, exit_reason="backtest_end",
                    pnl_pct=(last/pos.entry_price-1),
                    entry_time=ts_str(pos.entry_time), exit_time=ts_str(df["open_time"].iloc[-1]),
                    holding_bars=len(df)-1-pos.entry_bar))
        return trades


# ═══════════════════════════════════════════════════════════════
# S04: RSI 均值回归
# ═══════════════════════════════════════════════════════════════

class S04_RSIMeanRev(BaseStrategy):
    """
    RSI 超卖(< 30) + 布林带下轨 → 买入
    RSI > 60 或 BB 中轨 → 止盈
    只在 ADX < 30 的震荡市中做
    """
    name = "S04_RSIMeanRev"
    description = "RSI<30超卖+BB下轨入场+ADX<30震荡过滤"

    def __init__(self, rsi_buy=30, rsi_exit=60, adx_max=30,
                 trail_mult=2.0, risk_pct=0.015):
        self.rsi_buy = rsi_buy
        self.rsi_exit = rsi_exit
        self.adx_max = adx_max
        self.trail_mult = trail_mult
        self.risk_pct = risk_pct

    def run(self, dfs, capital, daily_dfs=None):
        trades = []
        for symbol in SYMBOLS:
            df = dfs.get(symbol)
            if df is None or len(df) < 100:
                continue
            df = df.copy()
            df["rsi"] = calc_rsi(df["close"], 14)
            df["atr"] = calc_atr(df, 14)
            df["adx"] = calc_adx(df, 14)
            df["bb_mid"], df["bb_up"], df["bb_low"], _ = calc_bollinger(df["close"], 20)

            cash = capital
            pos = Position()

            for i in range(1, len(df)):
                row = df.iloc[i]
                atr = row["atr"]
                if pd.isna(atr) or atr <= 0 or pd.isna(row.get("rsi")):
                    continue
                close = row["close"]

                if pos.direction == "LONG":
                    exit_reason = ""
                    if row["rsi"] > self.rsi_exit:
                        exit_reason = "rsi_exit"
                    elif close >= row.get("bb_mid", close * 10):
                        exit_reason = "bb_midline"
                    elif close <= pos.stop_loss:
                        exit_reason = "stop_loss"

                    if exit_reason:
                        pnl, comm, _, _ = apply_costs(pos.entry_price, close, pos.quantity)
                        cash += pos.quantity * close * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                        trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                            entry_price=pos.entry_price, exit_price=close, quantity=pos.quantity,
                            pnl=pnl, commission=comm, pnl_pct=(close/pos.entry_price-1),
                            entry_time=ts_str(pos.entry_time), exit_time=ts_str(row["open_time"]),
                            holding_bars=i-pos.entry_bar, exit_reason=exit_reason))
                        pos = Position()
                    continue

                # 开仓
                adx = row.get("adx", 50)
                bb_low = row.get("bb_low", 0)
                if (not pd.isna(adx) and adx < self.adx_max
                        and row["rsi"] < self.rsi_buy
                        and close <= bb_low * 1.005):
                    stop = close - self.trail_mult * atr
                    risk_amount = cash * self.risk_pct
                    risk_per_unit = close - stop
                    if risk_per_unit <= 0:
                        continue
                    qty = risk_amount / risk_per_unit
                    cost = qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if cost > cash * 0.40:
                        cost = cash * 0.40
                        qty = cost / close / (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if qty * close < 10:
                        continue
                    cash -= qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    pos = Position(symbol=symbol, direction="LONG",
                                   entry_price=close, entry_time=row["open_time"],
                                   entry_bar=i, quantity=qty, stop_loss=stop,
                                   highest=close, trailing_stop=stop, atr_at_entry=atr)

            if pos.direction == "LONG":
                last = df["close"].iloc[-1]
                pnl, comm, _, _ = apply_costs(pos.entry_price, last, pos.quantity)
                trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                    entry_price=pos.entry_price, exit_price=last, quantity=pos.quantity,
                    pnl=pnl, commission=comm, exit_reason="backtest_end",
                    pnl_pct=(last/pos.entry_price-1),
                    entry_time=ts_str(pos.entry_time), exit_time=ts_str(df["open_time"].iloc[-1]),
                    holding_bars=len(df)-1-pos.entry_bar))
        return trades


# ═══════════════════════════════════════════════════════════════
# S05: 布林带缩口突破
# ═══════════════════════════════════════════════════════════════

class S05_BBSqueeze(BaseStrategy):
    """
    BB 带宽压缩到近期低位 → 蓄势
    价格向上突破上轨 → 做多（缩口后的方向性突破）
    """
    name = "S05_BBSqueeze"
    description = "BB带宽<20%分位数+向上突破+2.5xATR追踪"

    def __init__(self, squeeze_pct=20, lookback=100, trail_mult=2.5, risk_pct=0.02):
        self.squeeze_pct = squeeze_pct
        self.lookback = lookback
        self.trail_mult = trail_mult
        self.risk_pct = risk_pct

    def run(self, dfs, capital, daily_dfs=None):
        trades = []
        for symbol in SYMBOLS:
            df = dfs.get(symbol)
            if df is None or len(df) < self.lookback + 20:
                continue
            df = df.copy()
            df["atr"] = calc_atr(df, 14)
            df["bb_mid"], df["bb_up"], df["bb_low"], df["bb_width"] = calc_bollinger(df["close"])
            # BB 带宽分位数
            df["bb_width_pctile"] = df["bb_width"].rolling(self.lookback).rank(pct=True) * 100

            cash = capital
            pos = Position()

            for i in range(2, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                atr = row["atr"]
                if pd.isna(atr) or atr <= 0:
                    continue
                close = row["close"]

                if pos.direction == "LONG":
                    pos.highest = max(pos.highest, close)
                    new_trail = pos.highest - self.trail_mult * atr
                    pos.trailing_stop = max(pos.trailing_stop, new_trail)

                    exit_reason = ""
                    if close <= pos.trailing_stop:
                        exit_reason = "trailing_stop"
                    elif close < row.get("bb_mid", 0):
                        exit_reason = "below_bb_mid"

                    if exit_reason:
                        pnl, comm, _, _ = apply_costs(pos.entry_price, close, pos.quantity)
                        cash += pos.quantity * close * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                        trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                            entry_price=pos.entry_price, exit_price=close, quantity=pos.quantity,
                            pnl=pnl, commission=comm, pnl_pct=(close/pos.entry_price-1),
                            entry_time=ts_str(pos.entry_time), exit_time=ts_str(row["open_time"]),
                            holding_bars=i-pos.entry_bar, exit_reason=exit_reason))
                        pos = Position()
                    continue

                # 缩口 + 突破上轨
                pctile = row.get("bb_width_pctile", 50)
                prev_pctile = prev.get("bb_width_pctile", 50)
                if (not pd.isna(pctile) and prev_pctile < self.squeeze_pct
                        and close > row.get("bb_up", close * 10)):
                    stop = row.get("bb_mid", close) - 0.5 * atr
                    risk_amount = cash * self.risk_pct
                    risk_per_unit = close - stop
                    if risk_per_unit <= 0:
                        continue
                    qty = risk_amount / risk_per_unit
                    cost = qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if cost > cash * 0.50:
                        cost = cash * 0.50
                        qty = cost / close / (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if qty * close < 10:
                        continue
                    cash -= qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    pos = Position(symbol=symbol, direction="LONG",
                                   entry_price=close, entry_time=row["open_time"],
                                   entry_bar=i, quantity=qty, stop_loss=stop,
                                   highest=close, trailing_stop=stop, atr_at_entry=atr)

            if pos.direction == "LONG":
                last = df["close"].iloc[-1]
                pnl, comm, _, _ = apply_costs(pos.entry_price, last, pos.quantity)
                trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                    entry_price=pos.entry_price, exit_price=last, quantity=pos.quantity,
                    pnl=pnl, commission=comm, exit_reason="backtest_end",
                    pnl_pct=(last/pos.entry_price-1),
                    entry_time=ts_str(pos.entry_time), exit_time=ts_str(df["open_time"].iloc[-1]),
                    holding_bars=len(df)-1-pos.entry_bar))
        return trades


# ═══════════════════════════════════════════════════════════════
# S06: 三均线动量
# ═══════════════════════════════════════════════════════════════

class S06_TripleEMA(BaseStrategy):
    """
    EMA8 > EMA21 > EMA55 = 强势排列
    EMA8 回踩 EMA21 → 入场（趋势中的短暂回调）
    EMA8 < EMA21 → 平仓
    """
    name = "S06_TripleEMA"
    description = "EMA8/21/55多头排列+回踩EMA21入场"

    def __init__(self, ema_fast=8, ema_mid=21, ema_slow=55,
                 trail_mult=2.5, risk_pct=0.02):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.trail_mult = trail_mult
        self.risk_pct = risk_pct

    def run(self, dfs, capital, daily_dfs=None):
        trades = []
        for symbol in SYMBOLS:
            df = dfs.get(symbol)
            if df is None or len(df) < 100:
                continue
            df = df.copy()
            df["e8"] = calc_ema(df["close"], self.ema_fast)
            df["e21"] = calc_ema(df["close"], self.ema_mid)
            df["e55"] = calc_ema(df["close"], self.ema_slow)
            df["atr"] = calc_atr(df, 14)

            cash = capital
            pos = Position()

            for i in range(2, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                atr = row["atr"]
                if pd.isna(atr) or atr <= 0:
                    continue
                close = row["close"]

                if pos.direction == "LONG":
                    pos.highest = max(pos.highest, close)
                    new_trail = pos.highest - self.trail_mult * atr
                    pos.trailing_stop = max(pos.trailing_stop, new_trail)

                    exit_reason = ""
                    if close <= pos.trailing_stop:
                        exit_reason = "trailing_stop"
                    elif row["e8"] < row["e21"]:
                        exit_reason = "ema_cross"

                    if exit_reason:
                        pnl, comm, _, _ = apply_costs(pos.entry_price, close, pos.quantity)
                        cash += pos.quantity * close * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                        trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                            entry_price=pos.entry_price, exit_price=close, quantity=pos.quantity,
                            pnl=pnl, commission=comm, pnl_pct=(close/pos.entry_price-1),
                            entry_time=ts_str(pos.entry_time), exit_time=ts_str(row["open_time"]),
                            holding_bars=i-pos.entry_bar, exit_reason=exit_reason))
                        pos = Position()
                    continue

                # 多头排列 + 回踩
                if (row["e8"] > row["e21"] > row["e55"]
                        and prev["close"] <= prev["e21"]
                        and close > row["e21"]):
                    stop = row["e55"] - 0.5 * atr
                    risk_amount = cash * self.risk_pct
                    risk_per_unit = close - stop
                    if risk_per_unit <= 0:
                        continue
                    qty = risk_amount / risk_per_unit
                    cost = qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if cost > cash * 0.50:
                        cost = cash * 0.50
                        qty = cost / close / (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if qty * close < 10:
                        continue
                    cash -= qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    pos = Position(symbol=symbol, direction="LONG",
                                   entry_price=close, entry_time=row["open_time"],
                                   entry_bar=i, quantity=qty, stop_loss=stop,
                                   highest=close, trailing_stop=stop, atr_at_entry=atr)

            if pos.direction == "LONG":
                last = df["close"].iloc[-1]
                pnl, comm, _, _ = apply_costs(pos.entry_price, last, pos.quantity)
                trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                    entry_price=pos.entry_price, exit_price=last, quantity=pos.quantity,
                    pnl=pnl, commission=comm, exit_reason="backtest_end",
                    pnl_pct=(last/pos.entry_price-1),
                    entry_time=ts_str(pos.entry_time), exit_time=ts_str(df["open_time"].iloc[-1]),
                    holding_bars=len(df)-1-pos.entry_bar))
        return trades


# ═══════════════════════════════════════════════════════════════
# S07: MACD 动量趋势
# ═══════════════════════════════════════════════════════════════

class S07_MACDMomentum(BaseStrategy):
    """
    MACD 柱状图由负转正（动量转向）+ 价格在 EMA50 上方（趋势过滤）
    MACD 死叉或追踪止损平仓
    """
    name = "S07_MACDMomentum"
    description = "MACD柱状图转正+EMA50趋势过滤"

    def __init__(self, trend_ema=50, trail_mult=2.5, risk_pct=0.02):
        self.trend_ema = trend_ema
        self.trail_mult = trail_mult
        self.risk_pct = risk_pct

    def run(self, dfs, capital, daily_dfs=None):
        trades = []
        for symbol in SYMBOLS:
            df = dfs.get(symbol)
            if df is None or len(df) < 100:
                continue
            df = df.copy()
            df["ema_trend"] = calc_ema(df["close"], self.trend_ema)
            df["atr"] = calc_atr(df, 14)
            df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(df["close"])

            cash = capital
            pos = Position()

            for i in range(2, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                atr = row["atr"]
                if pd.isna(atr) or atr <= 0:
                    continue
                close = row["close"]

                if pos.direction == "LONG":
                    pos.highest = max(pos.highest, close)
                    new_trail = pos.highest - self.trail_mult * atr
                    pos.trailing_stop = max(pos.trailing_stop, new_trail)

                    exit_reason = ""
                    if close <= pos.trailing_stop:
                        exit_reason = "trailing_stop"
                    elif row["macd"] < row["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
                        exit_reason = "macd_cross"

                    if exit_reason:
                        pnl, comm, _, _ = apply_costs(pos.entry_price, close, pos.quantity)
                        cash += pos.quantity * close * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                        trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                            entry_price=pos.entry_price, exit_price=close, quantity=pos.quantity,
                            pnl=pnl, commission=comm, pnl_pct=(close/pos.entry_price-1),
                            entry_time=ts_str(pos.entry_time), exit_time=ts_str(row["open_time"]),
                            holding_bars=i-pos.entry_bar, exit_reason=exit_reason))
                        pos = Position()
                    continue

                # MACD 柱状图由负转正 + 趋势过滤
                hist = row.get("macd_hist", 0)
                prev_hist = prev.get("macd_hist", 0)
                if (not pd.isna(hist) and not pd.isna(prev_hist)
                        and prev_hist < 0 and hist > 0
                        and close > row["ema_trend"]):
                    stop = close - 2.0 * atr
                    risk_amount = cash * self.risk_pct
                    risk_per_unit = close - stop
                    if risk_per_unit <= 0:
                        continue
                    qty = risk_amount / risk_per_unit
                    cost = qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if cost > cash * 0.50:
                        cost = cash * 0.50
                        qty = cost / close / (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if qty * close < 10:
                        continue
                    cash -= qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    pos = Position(symbol=symbol, direction="LONG",
                                   entry_price=close, entry_time=row["open_time"],
                                   entry_bar=i, quantity=qty, stop_loss=stop,
                                   highest=close, trailing_stop=stop, atr_at_entry=atr)

            if pos.direction == "LONG":
                last = df["close"].iloc[-1]
                pnl, comm, _, _ = apply_costs(pos.entry_price, last, pos.quantity)
                trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                    entry_price=pos.entry_price, exit_price=last, quantity=pos.quantity,
                    pnl=pnl, commission=comm, exit_reason="backtest_end",
                    pnl_pct=(last/pos.entry_price-1),
                    entry_time=ts_str(pos.entry_time), exit_time=ts_str(df["open_time"].iloc[-1]),
                    holding_bars=len(df)-1-pos.entry_bar))
        return trades


# ═══════════════════════════════════════════════════════════════
# S08: 放量突破
# ═══════════════════════════════════════════════════════════════

class S08_VolBreakout(BaseStrategy):
    """
    价格创 N 周期新高 + 成交量 > 2x 均量 + 阳线实体 > 70% 总幅度
    极度选择性，只做高确信度的爆发行情
    """
    name = "S08_VolBreakout"
    description = "N周期新高+2x放量+强实体阳线"

    def __init__(self, lookback=10, vol_mult=2.0, body_pct=0.60,
                 trail_mult=3.0, risk_pct=0.025):
        self.lookback = lookback
        self.vol_mult = vol_mult
        self.body_pct = body_pct
        self.trail_mult = trail_mult
        self.risk_pct = risk_pct

    def run(self, dfs, capital, daily_dfs=None):
        trades = []
        for symbol in SYMBOLS:
            df = dfs.get(symbol)
            if df is None or len(df) < self.lookback + 30:
                continue
            df = df.copy()
            df["atr"] = calc_atr(df, 14)
            df["vol_ma"] = df["volume"].rolling(20).mean()
            df["high_n"] = df["high"].rolling(self.lookback).max().shift(1)

            cash = capital
            pos = Position()

            for i in range(self.lookback + 1, len(df)):
                row = df.iloc[i]
                atr = row["atr"]
                if pd.isna(atr) or atr <= 0:
                    continue
                close = row["close"]
                open_ = row["open"]
                high = row["high"]
                low = row["low"]

                if pos.direction == "LONG":
                    pos.highest = max(pos.highest, close)
                    new_trail = pos.highest - self.trail_mult * atr
                    pos.trailing_stop = max(pos.trailing_stop, new_trail)

                    if close <= pos.trailing_stop:
                        pnl, comm, _, _ = apply_costs(pos.entry_price, close, pos.quantity)
                        cash += pos.quantity * close * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                        trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                            entry_price=pos.entry_price, exit_price=close, quantity=pos.quantity,
                            pnl=pnl, commission=comm, pnl_pct=(close/pos.entry_price-1),
                            entry_time=ts_str(pos.entry_time), exit_time=ts_str(row["open_time"]),
                            holding_bars=i-pos.entry_bar, exit_reason="trailing_stop"))
                        pos = Position()
                    continue

                # 突破 + 放量 + 强阳线
                bar_range = high - low
                body = abs(close - open_)
                body_ratio = body / bar_range if bar_range > 0 else 0
                is_bullish = close > open_

                high_n = row.get("high_n", 0)
                vol_ma = row.get("vol_ma", 0)
                if (pd.notna(high_n) and close > high_n
                        and row["volume"] > vol_ma * self.vol_mult
                        and is_bullish and body_ratio > self.body_pct):
                    stop = close - 2.0 * atr
                    risk_amount = cash * self.risk_pct
                    risk_per_unit = close - stop
                    if risk_per_unit <= 0:
                        continue
                    qty = risk_amount / risk_per_unit
                    cost = qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if cost > cash * 0.50:
                        cost = cash * 0.50
                        qty = cost / close / (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if qty * close < 10:
                        continue
                    cash -= qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    pos = Position(symbol=symbol, direction="LONG",
                                   entry_price=close, entry_time=row["open_time"],
                                   entry_bar=i, quantity=qty, stop_loss=stop,
                                   highest=close, trailing_stop=stop, atr_at_entry=atr)

            if pos.direction == "LONG":
                last = df["close"].iloc[-1]
                pnl, comm, _, _ = apply_costs(pos.entry_price, last, pos.quantity)
                trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                    entry_price=pos.entry_price, exit_price=last, quantity=pos.quantity,
                    pnl=pnl, commission=comm, exit_reason="backtest_end",
                    pnl_pct=(last/pos.entry_price-1),
                    entry_time=ts_str(pos.entry_time), exit_time=ts_str(df["open_time"].iloc[-1]),
                    holding_bars=len(df)-1-pos.entry_bar))
        return trades


# ═══════════════════════════════════════════════════════════════
# S09: 宽松网格
# ═══════════════════════════════════════════════════════════════

class S09_GridRelaxed(BaseStrategy):
    """
    放宽激活条件（ADX < 35），加大网格间距（1.0x ATR），减少层数（3 层）
    目标：让网格真正跑起来
    """
    name = "S09_GridRelaxed"
    timeframe = "1h"
    description = "ADX<35激活+1xATR间距+3层网格+8%止损"

    def __init__(self, n_grids=3, grid_atr_mult=1.0, qty_pct=0.06,
                 stop_loss_pct=0.08, adx_max=35, grid_life_bars=120):
        self.n_grids = n_grids
        self.grid_atr_mult = grid_atr_mult
        self.qty_pct = qty_pct
        self.stop_loss_pct = stop_loss_pct
        self.adx_max = adx_max
        self.grid_life_bars = grid_life_bars  # 网格最长存活 120 bars(5 天)

    def run(self, dfs, capital, daily_dfs=None):
        trades = []
        for symbol in SYMBOLS:
            df = dfs.get(symbol)
            if df is None or len(df) < 100:
                continue
            df = df.copy()
            df["atr"] = calc_atr(df, 20)
            df["adx"] = calc_adx(df, 14)

            cash = capital
            grid_active = False
            grid_center = 0.0
            grid_spacing = 0.0
            grid_stop = 0.0
            grid_start_bar = 0
            buy_levels = []   # [(price, filled, fill_price, qty)]
            sell_levels = []
            open_qty = 0.0
            avg_cost = 0.0
            grid_pnl = 0.0

            for i in range(1, len(df)):
                row = df.iloc[i]
                atr = row["atr"]
                adx = row.get("adx", 50)
                if pd.isna(atr) or atr <= 0:
                    continue
                close = row["close"]
                low = row["low"]
                high = row["high"]

                # 如果网格超时或止损，关闭
                if grid_active:
                    age = i - grid_start_bar
                    if age > self.grid_life_bars or low <= grid_stop:
                        # 平仓
                        if open_qty > 0:
                            exit_price = grid_stop if low <= grid_stop else close
                            pnl_close = open_qty * (exit_price - avg_cost)
                            pnl_close -= open_qty * exit_price * COMMISSION_RATE
                            cash += open_qty * exit_price * (1 - COMMISSION_RATE)
                            grid_pnl += pnl_close
                            reason = "grid_stop" if low <= grid_stop else "grid_timeout"
                            trades.append(Trade(symbol=symbol, strategy=self.name,
                                side="LONG", entry_price=avg_cost, exit_price=exit_price,
                                quantity=open_qty, pnl=pnl_close,
                                pnl_pct=(exit_price/avg_cost-1) if avg_cost > 0 else 0,
                                entry_time=ts_str(df.iloc[grid_start_bar]["open_time"]),
                                exit_time=ts_str(row["open_time"]),
                                holding_bars=age, exit_reason=reason))
                        grid_active = False
                        open_qty = 0.0
                        avg_cost = 0.0
                        buy_levels = []
                        sell_levels = []
                        continue

                    # 检查买入网格
                    for j, (price, filled, qty) in enumerate(buy_levels):
                        if not filled and low <= price:
                            cost = qty * price * (1 + COMMISSION_RATE)
                            if cost <= cash:
                                cash -= cost
                                # 更新均价
                                total_cost = avg_cost * open_qty + price * qty
                                open_qty += qty
                                avg_cost = total_cost / open_qty if open_qty > 0 else 0
                                buy_levels[j] = (price, True, qty)

                    # 检查卖出网格
                    for j, (price, filled, qty) in enumerate(sell_levels):
                        if not filled and high >= price and open_qty > 0:
                            sell_qty = min(qty, open_qty)
                            pnl_sell = sell_qty * (price - avg_cost)
                            pnl_sell -= sell_qty * price * COMMISSION_RATE
                            cash += sell_qty * price * (1 - COMMISSION_RATE)
                            open_qty -= sell_qty
                            grid_pnl += pnl_sell
                            sell_levels[j] = (price, True, qty)

                    continue

                # 激活新网格
                if not pd.isna(adx) and adx < self.adx_max and cash > 100:
                    grid_active = True
                    grid_center = close
                    grid_spacing = atr * self.grid_atr_mult
                    grid_stop = close * (1 - self.stop_loss_pct)
                    grid_start_bar = i

                    qty_per = (cash * self.qty_pct) / close
                    buy_levels = [(close - k * grid_spacing, False, qty_per) for k in range(1, self.n_grids + 1)]
                    sell_levels = [(close + k * grid_spacing, False, qty_per) for k in range(1, self.n_grids + 1)]
                    open_qty = 0.0
                    avg_cost = 0.0

            # 回测结束清仓
            if grid_active and open_qty > 0:
                last = df["close"].iloc[-1]
                pnl_close = open_qty * (last - avg_cost)
                pnl_close -= open_qty * last * COMMISSION_RATE
                cash += open_qty * last * (1 - COMMISSION_RATE)
                grid_pnl += pnl_close
                trades.append(Trade(symbol=symbol, strategy=self.name,
                    side="LONG", entry_price=avg_cost, exit_price=last,
                    quantity=open_qty, pnl=pnl_close,
                    pnl_pct=(last/avg_cost-1) if avg_cost > 0 else 0,
                    exit_reason="backtest_end",
                    entry_time=ts_str(df.iloc[grid_start_bar]["open_time"]),
                    exit_time=ts_str(df["open_time"].iloc[-1]),
                    holding_bars=len(df)-1-grid_start_bar))

        return trades


# ═══════════════════════════════════════════════════════════════
# S10: Keltner 通道突破
# ═══════════════════════════════════════════════════════════════

class S10_KeltnerBreak(BaseStrategy):
    """
    价格突破 Keltner 上轨（EMA20 + 2xATR）
    RSI > 50 确认多头动量
    用 Keltner 中轨（EMA20）做追踪参考
    """
    name = "S10_KeltnerBreak"
    description = "Keltner上轨突破+RSI>50+EMA20追踪"

    def __init__(self, kc_ema=20, kc_mult=2.0, trail_mult=2.5, risk_pct=0.02):
        self.kc_ema = kc_ema
        self.kc_mult = kc_mult
        self.trail_mult = trail_mult
        self.risk_pct = risk_pct

    def run(self, dfs, capital, daily_dfs=None):
        trades = []
        for symbol in SYMBOLS:
            df = dfs.get(symbol)
            if df is None or len(df) < 100:
                continue
            df = df.copy()
            df["atr"] = calc_atr(df, 14)
            df["rsi"] = calc_rsi(df["close"], 14)
            df["kc_mid"], df["kc_up"], df["kc_low"] = calc_keltner(df, self.kc_ema, self.kc_mult)

            cash = capital
            pos = Position()

            for i in range(2, len(df)):
                row = df.iloc[i]
                prev = df.iloc[i-1]
                atr = row["atr"]
                if pd.isna(atr) or atr <= 0:
                    continue
                close = row["close"]

                if pos.direction == "LONG":
                    pos.highest = max(pos.highest, close)
                    new_trail = pos.highest - self.trail_mult * atr
                    pos.trailing_stop = max(pos.trailing_stop, new_trail)

                    exit_reason = ""
                    if close <= pos.trailing_stop:
                        exit_reason = "trailing_stop"
                    elif close < row.get("kc_mid", 0):
                        exit_reason = "below_kc_mid"

                    if exit_reason:
                        pnl, comm, _, _ = apply_costs(pos.entry_price, close, pos.quantity)
                        cash += pos.quantity * close * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                        trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                            entry_price=pos.entry_price, exit_price=close, quantity=pos.quantity,
                            pnl=pnl, commission=comm, pnl_pct=(close/pos.entry_price-1),
                            entry_time=ts_str(pos.entry_time), exit_time=ts_str(row["open_time"]),
                            holding_bars=i-pos.entry_bar, exit_reason=exit_reason))
                        pos = Position()
                    continue

                # Keltner 突破 + RSI > 50
                kc_up = row.get("kc_up", close * 10)
                rsi = row.get("rsi", 50)
                if (not pd.isna(kc_up) and close > kc_up
                        and prev["close"] <= prev.get("kc_up", close * 10)
                        and not pd.isna(rsi) and rsi > 50):
                    stop = row.get("kc_mid", close) - 0.5 * atr
                    risk_amount = cash * self.risk_pct
                    risk_per_unit = close - stop
                    if risk_per_unit <= 0:
                        continue
                    qty = risk_amount / risk_per_unit
                    cost = qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if cost > cash * 0.50:
                        cost = cash * 0.50
                        qty = cost / close / (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    if qty * close < 10:
                        continue
                    cash -= qty * close * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                    pos = Position(symbol=symbol, direction="LONG",
                                   entry_price=close, entry_time=row["open_time"],
                                   entry_bar=i, quantity=qty, stop_loss=stop,
                                   highest=close, trailing_stop=stop, atr_at_entry=atr)

            if pos.direction == "LONG":
                last = df["close"].iloc[-1]
                pnl, comm, _, _ = apply_costs(pos.entry_price, last, pos.quantity)
                trades.append(Trade(symbol=symbol, strategy=self.name, side="LONG",
                    entry_price=pos.entry_price, exit_price=last, quantity=pos.quantity,
                    pnl=pnl, commission=comm, exit_reason="backtest_end",
                    pnl_pct=(last/pos.entry_price-1),
                    entry_time=ts_str(pos.entry_time), exit_time=ts_str(df["open_time"].iloc[-1]),
                    holding_bars=len(df)-1-pos.entry_bar))
        return trades


# ═══════════════════════════════════════════════════════════════
# 竞技场：批量执行 + 对比
# ═══════════════════════════════════════════════════════════════

def calc_metrics(trades: list[Trade], capital: float) -> dict:
    """计算策略统计指标"""
    if not trades:
        return {"total_pnl": 0, "return_pct": 0, "trades": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "profit_factor": 0, "expectancy": 0,
                "best": 0, "worst": 0, "avg_hold": 0, "total_comm": 0,
                "max_dd_pct": 0, "sharpe_approx": 0}

    pnls = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # 逐笔权益曲线（近似）
    equity = [capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.max())

    # 近似夏普（假设 4h 频率，每笔交易独立）
    if len(pnls) > 1:
        rets = np.array(pnls) / capital
        sharpe = float(np.mean(rets) / max(np.std(rets), 1e-10) * np.sqrt(len(pnls)))
    else:
        sharpe = 0.0

    return {
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / capital * 100, 2),
        "trades": len(trades),
        "win_rate": round(len(wins)/len(pnls)*100, 1) if pnls else 0,
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "profit_factor": round(sum(wins)/abs(sum(losses)), 3) if losses else float("inf"),
        "expectancy": round(np.mean(pnls), 2),
        "best": round(max(pnls), 2),
        "worst": round(min(pnls), 2),
        "avg_hold": round(np.mean([t.holding_bars for t in trades]), 1),
        "total_comm": round(sum(t.commission for t in trades), 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "sharpe_approx": round(sharpe, 3),
    }


def run_arena(db_path: str, capital: float, start_date: str = None,
              end_date: str = None, snapshot_path: str = "arena_snapshot.txt"):
    """运行 10 策略竞技场"""

    print("=" * 70)
    print("  🏟️  回测竞技场 — 10 策略批量对比")
    print("=" * 70)
    print(f"  本金: {capital:.0f} USDT | 标的: {', '.join(SYMBOLS)}")
    print(f"  日期: {start_date or '全部'} ~ {end_date or '全部'}")
    print(f"  手续费: {COMMISSION_RATE*100:.2f}% | 滑点: {SLIPPAGE_RATE*100:.3f}%")
    print()

    # ---- 加载数据 ----
    print("📂 加载数据...")
    dfs_4h = {}
    dfs_1h = {}
    daily_dfs = {}
    for sym in SYMBOLS:
        dfs_4h[sym] = load_klines(db_path, sym, "4h", start_date, end_date)
        dfs_1h[sym] = load_klines(db_path, sym, "1h", start_date, end_date)
        # 日线：尝试直接加载，如果不够则重采样
        d = load_klines(db_path, sym, "1d", start_date, end_date)
        if len(d) < 50:
            d = resample_to_daily(dfs_4h[sym])
        daily_dfs[sym] = d
        print(f"  {sym}: 4h={len(dfs_4h[sym])} | 1h={len(dfs_1h[sym])} | daily={len(d)}")

    # ---- 定义策略 ----
    strategies = [
        S01_TrendMTF(),
        S02_BreakoutDC(),
        S03_PullbackEMA(),
        S04_RSIMeanRev(),
        S05_BBSqueeze(),
        S06_TripleEMA(),
        S07_MACDMomentum(),
        S08_VolBreakout(),
        S09_GridRelaxed(),
        S10_KeltnerBreak(),
    ]

    # ---- 运行 ----
    results = {}
    t0 = time.time()
    for strat in strategies:
        print(f"\n🔄 运行 {strat.name}...", end=" ")
        st = time.time()

        # 选择数据
        if strat.timeframe == "1h":
            trades = strat.run(dfs_1h, capital, daily_dfs)
        else:
            trades = strat.run(dfs_4h, capital, daily_dfs)

        metrics = calc_metrics(trades, capital)
        results[strat.name] = {"trades": trades, "metrics": metrics,
                                "desc": strat.description}

        icon = "🟢" if metrics["total_pnl"] > 0 else "🔴"
        print(f"{icon} {metrics['total_pnl']:+.2f} USDT "
              f"({metrics['return_pct']:+.2f}%) | "
              f"{metrics['trades']} trades | "
              f"WR={metrics['win_rate']:.1f}% | "
              f"PF={metrics['profit_factor']:.2f} | "
              f"{time.time()-st:.1f}s")

    elapsed = time.time() - t0

    # ---- 排行榜 ----
    print()
    print("=" * 110)
    print("  🏆  排 行 榜（按总收益排序）")
    print("=" * 110)
    header = (f"{'排名':>4} | {'策略':<20} | {'总PnL':>10} | {'收益%':>8} | "
              f"{'交易数':>6} | {'胜率':>6} | {'盈亏比':>7} | {'期望':>8} | "
              f"{'最大回撤':>8} | {'夏普':>6} | {'手续费':>8}")
    print(header)
    print("-" * 110)

    ranked = sorted(results.items(), key=lambda x: x[1]["metrics"]["total_pnl"], reverse=True)
    for rank, (name, data) in enumerate(ranked, 1):
        m = data["metrics"]
        icon = "🟢" if m["total_pnl"] > 0 else "🔴"
        print(f" {icon}{rank:>2} | {name:<20} | {m['total_pnl']:>+10.2f} | "
              f"{m['return_pct']:>+7.2f}% | {m['trades']:>6} | "
              f"{m['win_rate']:>5.1f}% | {m['profit_factor']:>7.3f} | "
              f"{m['expectancy']:>+8.2f} | {m['max_dd_pct']:>7.2f}% | "
              f"{m['sharpe_approx']:>6.3f} | {m['total_comm']:>8.2f}")

    print("=" * 110)
    print(f"  ⏱  总耗时: {elapsed:.1f}s")

    # ---- 各策略按标的拆分 ----
    print()
    print("=" * 80)
    print("  📊  各策略按标的拆分")
    print("=" * 80)
    for name, data in ranked:
        by_sym = {}
        for t in data["trades"]:
            by_sym.setdefault(t.symbol, []).append(t)
        parts = []
        for sym in SYMBOLS:
            sym_trades = by_sym.get(sym, [])
            sym_pnl = sum(t.pnl for t in sym_trades)
            parts.append(f"{sym}={sym_pnl:+.0f}")
        print(f"  {name:<20} | {' | '.join(parts)}")

    # ---- 月度交叉对比 ----
    print()
    print("=" * 80)
    print("  📅  月度 PnL 对比（前 5 名策略）")
    print("=" * 80)

    top5 = ranked[:5]
    all_months = set()
    monthly_data = {}
    for name, data in top5:
        monthly_data[name] = {}
        for t in data["trades"]:
            if t.exit_time:
                month = t.exit_time[:7]
                monthly_data[name][month] = monthly_data[name].get(month, 0) + t.pnl
                all_months.add(month)

    months_sorted = sorted(all_months)
    header = f"  {'月份':<10}" + "".join(f" | {n[:12]:>12}" for n, _ in top5)
    print(header)
    print("-" * len(header))
    for month in months_sorted:
        row = f"  {month:<10}"
        for name, _ in top5:
            val = monthly_data.get(name, {}).get(month, 0)
            icon = "+" if val > 0 else ""
            row += f" | {icon}{val:>11.1f}"
        print(row)

    # ---- 写入快照 ----
    write_arena_snapshot(results, ranked, capital, start_date, end_date,
                         db_path, elapsed, snapshot_path)

    return results


# ═══════════════════════════════════════════════════════════════
# 快照输出
# ═══════════════════════════════════════════════════════════════

def write_arena_snapshot(results, ranked, capital, start_date, end_date,
                         db_path, elapsed, output_path):
    lines = []
    lines.append("=" * 70)
    lines.append("ARENA BACKTEST SNAPSHOT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Database: {db_path}")
    lines.append(f"Period: {start_date or 'all'} ~ {end_date or 'all'}")
    lines.append(f"Capital: {capital}")
    lines.append(f"Commission: {COMMISSION_RATE*100:.2f}% | Slippage: {SLIPPAGE_RATE*100:.3f}%")
    lines.append(f"Elapsed: {elapsed:.1f}s")
    lines.append("=" * 70)

    # 排行榜
    lines.append("")
    lines.append("=" * 70)
    lines.append("SECTION: LEADERBOARD")
    lines.append("=" * 70)
    for rank, (name, data) in enumerate(ranked, 1):
        m = data["metrics"]
        lines.append(f"  #{rank} {name}")
        lines.append(f"    desc: {data['desc']}")
        for k, v in m.items():
            lines.append(f"    {k}: {v}")
        lines.append("")

    # 每个策略的详细交易
    for name, data in ranked:
        lines.append("=" * 70)
        lines.append(f"SECTION: TRADES_{name}")
        lines.append("=" * 70)
        lines.append(f"{'#':>4} | {'Symbol':<10} | {'Side':<6} | "
                     f"{'Entry':>10} | {'Exit':>10} | {'PnL':>10} | {'PnL%':>8} | "
                     f"{'Bars':>5} | {'Reason':<16} | {'Entry Time':<16} | {'Exit Time':<16}")
        lines.append("-" * 140)
        for idx, t in enumerate(data["trades"], 1):
            lines.append(
                f"{idx:>4} | {t.symbol:<10} | {t.side:<6} | "
                f"{t.entry_price:>10.2f} | {t.exit_price:>10.2f} | {t.pnl:>+10.2f} | "
                f"{t.pnl_pct:>+7.2%} | {t.holding_bars:>5} | {t.exit_reason:<16} | "
                f"{t.entry_time:<16} | {t.exit_time:<16}")
        lines.append("")

    # 月度 PnL
    lines.append("=" * 70)
    lines.append("SECTION: MONTHLY_COMPARISON")
    lines.append("=" * 70)
    all_months = set()
    monthly_data = {}
    for name, data in ranked:
        monthly_data[name] = {}
        for t in data["trades"]:
            if t.exit_time:
                month = t.exit_time[:7]
                monthly_data[name][month] = monthly_data[name].get(month, 0) + t.pnl
                all_months.add(month)
    months_sorted = sorted(all_months)
    for month in months_sorted:
        parts = []
        for name, _ in ranked:
            val = monthly_data.get(name, {}).get(month, 0)
            parts.append(f"{name}={val:+.1f}")
        lines.append(f"  {month}: {' | '.join(parts)}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF ARENA SNAPSHOT")
    lines.append("=" * 70)

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n📄 竞技场快照已保存: {output_path}")
    print(f"   大小: {len(content)} 字符 / {len(lines)} 行")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="回测竞技场 — 10 策略批量对比")
    parser.add_argument("--db", default="data/quant.db", help="数据库路径")
    parser.add_argument("--capital", type=float, default=10000.0, help="初始资金")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--snapshot", default="arena_snapshot.txt", help="快照输出路径")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"❌ 数据库不存在: {db}")
        print("   请先运行: python main.py --sync-data")
        sys.exit(1)

    run_arena(str(db), args.capital, args.start, args.end, args.snapshot)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 2: Comprehensive Diagnostic & Signal Edge Analysis
=========================================================
Run:  python phase2_diagnostic.py
Output: phase2_report.json  (请把这个文件返回给我)

This script:
1. Discovers DB schema & loads all candle data
2. Computes cross-asset correlation matrix (returns + rolling)
3. Estimates realistic trading costs per symbol
4. Tests RAW signal edge: MACD cross, EMA trend, mean-reversion
5. Analyzes regime distribution per symbol (ADX, ATR-vol, trend strength)
6. Tests long+short combined signal on BTC 4h full history
7. Walk-forward validation on BTC 4h (180/60)
8. Out-of-sample check on ETH/BNB/SOL 4h (no param tuning)
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
DB_PATH = "data/quant.db"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
INITIAL_CAPITAL = 500.0
COMMISSION_PCT = 0.0004   # Binance USDT-M taker
SLIPPAGE_BPS   = 25       # 真实滑点估计

OUTPUT_FILE = "phase2_report.json"

# ==================== HELPERS ====================

def safe_json(obj):
    """Make numpy types JSON-serializable"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def pick_interval_df(intervals: dict) -> pd.DataFrame | None:
    """优先 4h，否则 1d；禁止对 DataFrame 使用 `or`（pandas 会报错）。"""
    d4 = intervals.get("4h")
    if d4 is not None and len(d4) > 0:
        return d4
    d1 = intervals.get("1d")
    if d1 is not None and len(d1) > 0:
        return d1
    return None


# ==================== 1. DB SCHEMA DISCOVERY ====================

def discover_schema(db_path):
    """Discover database schema and available data"""
    conn = sqlite3.connect(db_path)
    info = {"tables": {}}

    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    for tbl in tables['name']:
        cols = pd.read_sql(f"PRAGMA table_info('{tbl}')", conn)
        row_count = pd.read_sql(f"SELECT COUNT(*) as n FROM '{tbl}'", conn)['n'][0]
        info["tables"][tbl] = {
            "columns": cols['name'].tolist(),
            "types": cols['type'].tolist(),
            "row_count": int(row_count),
        }
        # Sample first/last row if has timestamp-like column
        ts_cols = [c for c in cols['name'] if any(k in c.lower() for k in ['time', 'date', 'ts', 'open_time'])]
        if ts_cols and row_count > 0:
            tc = ts_cols[0]
            try:
                first = pd.read_sql(f"SELECT MIN(\"{tc}\") as v FROM '{tbl}'", conn)['v'][0]
                last  = pd.read_sql(f"SELECT MAX(\"{tc}\") as v FROM '{tbl}'", conn)['v'][0]
                info["tables"][tbl]["time_range"] = [str(first), str(last)]
            except:
                pass

    conn.close()
    return info


# ==================== 2. DATA LOADING ====================

def load_candles(db_path, symbol, interval):
    """Try various table naming conventions to load candle data"""
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()

    # Try common naming patterns
    candidates = [
        f"candles_{symbol}_{interval}",
        f"{symbol}_{interval}",
        f"klines_{symbol}_{interval}",
        f"ohlcv_{symbol}_{interval}",
        f"candles_{symbol.lower()}_{interval}",
        f"{symbol.lower()}_{interval}",
        "candles",
        "klines",
        "ohlcv",
    ]

    df = None
    used_table = None
    for tbl in candidates:
        if tbl in tables:
            try:
                query = f'SELECT * FROM "{tbl}"'
                # If it's a shared table, filter by symbol/interval
                if tbl in ["candles", "klines", "ohlcv"]:
                    # Check columns
                    cols = pd.read_sql(f"PRAGMA table_info('{tbl}')", conn)['name'].tolist()
                    sym_col = next((c for c in cols if 'symbol' in c.lower()), None)
                    int_col = next((c for c in cols if 'interval' in c.lower()), None)
                    if sym_col and int_col:
                        query = f'SELECT * FROM "{tbl}" WHERE "{sym_col}"=\'{symbol}\' AND "{int_col}"=\'{interval}\''
                    elif sym_col:
                        query = f'SELECT * FROM "{tbl}" WHERE "{sym_col}"=\'{symbol}\''
                df = pd.read_sql(query, conn)
                if len(df) > 0:
                    used_table = tbl
                    break
            except:
                continue

    conn.close()
    if df is None or len(df) == 0:
        return None, None

    # Normalize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if 'open_time' in cl or (cl == 'timestamp') or (cl == 'time') or (cl == 'date') or (cl == 'ts'):
            col_map[col] = 'timestamp'
        elif cl == 'open' or cl == 'open_price':
            col_map[col] = 'open'
        elif cl == 'high' or cl == 'high_price':
            col_map[col] = 'high'
        elif cl == 'low' or cl == 'low_price':
            col_map[col] = 'low'
        elif cl == 'close' or cl == 'close_price':
            col_map[col] = 'close'
        elif cl == 'volume' or cl == 'vol':
            col_map[col] = 'volume'
    df = df.rename(columns=col_map)

    # Parse timestamp
    if 'timestamp' in df.columns:
        if df['timestamp'].dtype in ['int64', 'float64']:
            # Might be milliseconds
            vals = df['timestamp'].values
            if vals[0] > 1e12:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, used_table


# ==================== 3. TECHNICAL INDICATORS ====================

def add_indicators(df):
    """Add all indicators needed for analysis"""
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    v = df['volume'].values.astype(float)
    n = len(c)

    # --- EMA helper ---
    def ema(arr, period):
        result = np.full(n, np.nan)
        if n < period:
            return result
        result[period-1] = np.mean(arr[:period])
        k = 2.0 / (period + 1)
        for i in range(period, n):
            result[i] = arr[i] * k + result[i-1] * (1-k)
        return result

    # --- EMAs ---
    df['ema_8']  = ema(c, 8)
    df['ema_21'] = ema(c, 21)
    df['ema_55'] = ema(c, 55)
    df['ema_200'] = ema(c, 200)

    # --- MACD ---
    ema12 = ema(c, 12)
    ema26 = ema(c, 26)
    macd_line = ema12 - ema26
    signal_line = ema(np.where(np.isnan(macd_line), 0, macd_line), 9)
    # Fix: only valid where macd_line is valid
    valid_start = max(26, 35)  # 26 for macd + 9 for signal
    signal_line[:valid_start] = np.nan
    macd_line[:26] = np.nan
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_line - signal_line

    # --- ATR ---
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr14 = np.full(n, np.nan)
    if n >= 14:
        atr14[13] = np.mean(tr[:14])
        for i in range(14, n):
            atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    df['atr14'] = atr14
    df['atr_pct'] = atr14 / c * 100  # ATR as % of price

    # --- ADX ---
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = h[i] - h[i-1]
        down_move = l[i-1] - l[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    atr_smooth = np.full(n, np.nan)
    plus_di_smooth = np.full(n, np.nan)
    minus_di_smooth = np.full(n, np.nan)
    period = 14
    if n >= period + 1:
        atr_smooth[period] = np.sum(tr[1:period+1])
        plus_di_smooth[period] = np.sum(plus_dm[1:period+1])
        minus_di_smooth[period] = np.sum(minus_dm[1:period+1])
        for i in range(period+1, n):
            atr_smooth[i] = atr_smooth[i-1] - atr_smooth[i-1]/period + tr[i]
            plus_di_smooth[i] = plus_di_smooth[i-1] - plus_di_smooth[i-1]/period + plus_dm[i]
            minus_di_smooth[i] = minus_di_smooth[i-1] - minus_di_smooth[i-1]/period + minus_dm[i]

    plus_di = np.where(atr_smooth > 0, 100 * plus_di_smooth / atr_smooth, 0)
    minus_di = np.where(atr_smooth > 0, 100 * minus_di_smooth / atr_smooth, 0)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full(n, np.nan)
    adx_start = 2 * period
    if n > adx_start:
        adx[adx_start] = np.mean(dx[period+1:adx_start+1])
        for i in range(adx_start+1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # --- RSI ---
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= 15:
        avg_gain[14] = np.mean(gain[1:15])
        avg_loss[14] = np.mean(loss[1:15])
        for i in range(15, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan
    df['rsi14'] = rsi

    # --- Returns ---
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # --- Volume ratio ---
    vol_ma = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / vol_ma

    return df


# ==================== 4. CORRELATION ANALYSIS ====================

def compute_correlations(data_dict, interval='1d'):
    """Cross-asset return correlations"""
    returns = {}
    for sym, intervals in data_dict.items():
        if interval in intervals and intervals[interval] is not None:
            df = intervals[interval]
            if 'returns' in df.columns:
                returns[sym] = df.set_index('timestamp')['returns'].dropna()

    if len(returns) < 2:
        return {"error": "insufficient_data"}

    ret_df = pd.DataFrame(returns)
    ret_df = ret_df.dropna()

    result = {
        "n_common_bars": len(ret_df),
        "period": f"{ret_df.index.min()} ~ {ret_df.index.max()}",
        "correlation_matrix": {},
        "rolling_90d_corr": {},
    }

    # Full-period correlations
    corr = ret_df.corr()
    for s1 in corr.columns:
        for s2 in corr.columns:
            if s1 < s2:
                key = f"{s1}_vs_{s2}"
                result["correlation_matrix"][key] = round(float(corr.loc[s1, s2]), 4)

    # Rolling 90-bar correlations (stats only, not full series)
    for s1 in ret_df.columns:
        for s2 in ret_df.columns:
            if s1 < s2:
                rolling = ret_df[s1].rolling(90).corr(ret_df[s2]).dropna()
                if len(rolling) > 0:
                    key = f"{s1}_vs_{s2}"
                    result["rolling_90d_corr"][key] = {
                        "mean": round(float(rolling.mean()), 4),
                        "std": round(float(rolling.std()), 4),
                        "min": round(float(rolling.min()), 4),
                        "max": round(float(rolling.max()), 4),
                        "pct_below_0.5": round(float((rolling < 0.5).mean() * 100), 1),
                    }

    return result


# ==================== 5. COST ESTIMATION ====================

def estimate_costs(df, symbol):
    """Estimate realistic per-trade costs"""
    if df is None or 'high' not in df.columns:
        return {}

    # Spread proxy: (high - low) / close as % on each bar
    spread_pct = (df['high'] - df['low']) / df['close'] * 100
    spread_pct = spread_pct.dropna()

    # Volume-weighted analysis
    result = {
        "symbol": symbol,
        "median_bar_range_pct": round(float(spread_pct.median()), 4),
        "p25_bar_range_pct": round(float(spread_pct.quantile(0.25)), 4),
        "p75_bar_range_pct": round(float(spread_pct.quantile(0.75)), 4),
        "estimated_slippage_bps": round(float(spread_pct.median() * 100 / 4), 1),  # ~1/4 of bar range
        "recommended_total_cost_bps": None,  # slippage + commission both sides
    }
    result["recommended_total_cost_bps"] = round(
        result["estimated_slippage_bps"] + COMMISSION_PCT * 10000 * 2, 1  # x2 for round trip
    )

    return result


# ==================== 6. REGIME ANALYSIS ====================

def analyze_regimes(df, symbol):
    """Distribution of market regimes"""
    if df is None or 'adx' not in df.columns:
        return {}

    valid = df.dropna(subset=['adx', 'atr_pct'])

    # ADX buckets
    adx_bins = {'weak(<20)': 0, 'moderate(20-30)': 0, 'strong(>30)': 0}
    adx_vals = valid['adx'].values
    adx_bins['weak(<20)'] = int((adx_vals < 20).sum())
    adx_bins['moderate(20-30)'] = int(((adx_vals >= 20) & (adx_vals < 30)).sum())
    adx_bins['strong(>30)'] = int((adx_vals >= 30).sum())

    total = sum(adx_bins.values())
    adx_pcts = {k: round(v/total*100, 1) if total > 0 else 0 for k, v in adx_bins.items()}

    # Volatility buckets (ATR as % of price)
    vol_bins = {'low(<1.5%)': 0, 'medium(1.5-3%)': 0, 'high(>3%)': 0}
    atr_vals = valid['atr_pct'].values
    vol_bins['low(<1.5%)'] = int((atr_vals < 1.5).sum())
    vol_bins['medium(1.5-3%)'] = int(((atr_vals >= 1.5) & (atr_vals < 3)).sum())
    vol_bins['high(>3%)'] = int((atr_vals >= 3).sum())

    vol_pcts = {k: round(v/total*100, 1) if total > 0 else 0 for k, v in vol_bins.items()}

    # Trend direction (EMA 55 slope)
    if 'ema_55' in df.columns:
        ema_slope = valid['ema_55'].pct_change(10) * 100
        trend_up = int((ema_slope > 0.5).sum())
        trend_down = int((ema_slope < -0.5).sum())
        trend_flat = total - trend_up - trend_down
    else:
        trend_up = trend_down = trend_flat = 0

    return {
        "symbol": symbol,
        "total_bars": total,
        "adx_distribution": adx_bins,
        "adx_pct": adx_pcts,
        "volatility_distribution": vol_bins,
        "volatility_pct": vol_pcts,
        "trend_direction": {
            "uptrend": trend_up,
            "downtrend": trend_down,
            "flat": trend_flat,
        },
        "adx_stats": {
            "mean": round(float(valid['adx'].mean()), 1),
            "median": round(float(valid['adx'].median()), 1),
            "p75": round(float(valid['adx'].quantile(0.75)), 1),
        },
        "atr_pct_stats": {
            "mean": round(float(valid['atr_pct'].mean()), 2),
            "median": round(float(valid['atr_pct'].median()), 2),
            "p75": round(float(valid['atr_pct'].quantile(0.75)), 2),
        }
    }


# ==================== 7. RAW SIGNAL EDGE TEST ====================

def test_signal_edge(df, symbol):
    """
    Test raw directional edge of various signals.
    For each signal, compute forward N-bar returns after signal fires.
    This answers: "Does this signal predict direction AT ALL?"
    """
    if df is None or len(df) < 200:
        return {"error": "insufficient_data"}

    results = {}

    # --- MACD Cross Long: MACD crosses above signal ---
    macd = df['macd'].values
    sig  = df['macd_signal'].values
    cross_long  = np.zeros(len(df), dtype=bool)
    cross_short = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if not np.isnan(macd[i]) and not np.isnan(sig[i]) and not np.isnan(macd[i-1]) and not np.isnan(sig[i-1]):
            if macd[i] > sig[i] and macd[i-1] <= sig[i-1]:
                cross_long[i] = True
            if macd[i] < sig[i] and macd[i-1] >= sig[i-1]:
                cross_short[i] = True

    for horizon in [3, 6, 12, 24]:
        fwd_ret = df['close'].pct_change(horizon).shift(-horizon).values
        # Long signals
        long_rets = fwd_ret[cross_long & ~np.isnan(fwd_ret)]
        # Short signals
        short_rets = -fwd_ret[cross_short & ~np.isnan(fwd_ret)]  # negate for short

        if len(long_rets) > 5:
            results[f"macd_cross_long_{horizon}bar"] = {
                "n_signals": int(len(long_rets)),
                "mean_return_pct": round(float(long_rets.mean() * 100), 3),
                "median_return_pct": round(float(np.median(long_rets) * 100), 3),
                "win_rate": round(float((long_rets > 0).mean() * 100), 1),
                "t_stat": round(float(long_rets.mean() / (long_rets.std() / np.sqrt(len(long_rets)))) if long_rets.std() > 0 else 0, 2),
            }
        if len(short_rets) > 5:
            results[f"macd_cross_short_{horizon}bar"] = {
                "n_signals": int(len(short_rets)),
                "mean_return_pct": round(float(short_rets.mean() * 100), 3),
                "median_return_pct": round(float(np.median(short_rets) * 100), 3),
                "win_rate": round(float((short_rets > 0).mean() * 100), 1),
                "t_stat": round(float(short_rets.mean() / (short_rets.std() / np.sqrt(len(short_rets)))) if short_rets.std() > 0 else 0, 2),
            }

    # --- EMA Trend: Price above EMA55 & EMA21 > EMA55 ---
    trend_long = (df['close'] > df['ema_55']) & (df['ema_21'] > df['ema_55'])
    trend_short = (df['close'] < df['ema_55']) & (df['ema_21'] < df['ema_55'])
    # Combined: MACD cross + trend alignment
    combo_long = cross_long & trend_long.values
    combo_short = cross_short & trend_short.values

    for horizon in [6, 12, 24]:
        fwd_ret = df['close'].pct_change(horizon).shift(-horizon).values
        long_rets = fwd_ret[combo_long & ~np.isnan(fwd_ret)]
        short_rets = -fwd_ret[combo_short & ~np.isnan(fwd_ret)]

        if len(long_rets) > 3:
            results[f"combo_trend_long_{horizon}bar"] = {
                "n_signals": int(len(long_rets)),
                "mean_return_pct": round(float(long_rets.mean() * 100), 3),
                "win_rate": round(float((long_rets > 0).mean() * 100), 1),
                "t_stat": round(float(long_rets.mean() / (long_rets.std() / np.sqrt(len(long_rets)))) if long_rets.std() > 0 else 0, 2),
            }
        if len(short_rets) > 3:
            results[f"combo_trend_short_{horizon}bar"] = {
                "n_signals": int(len(short_rets)),
                "mean_return_pct": round(float(short_rets.mean() * 100), 3),
                "win_rate": round(float((short_rets > 0).mean() * 100), 1),
                "t_stat": round(float(short_rets.mean() / (short_rets.std() / np.sqrt(len(short_rets)))) if short_rets.std() > 0 else 0, 2),
            }

    # --- Regime-filtered: Only strong ADX ---
    adx_gate = df['adx'].values > 25
    regime_long  = combo_long & adx_gate
    regime_short = combo_short & adx_gate

    for horizon in [6, 12, 24]:
        fwd_ret = df['close'].pct_change(horizon).shift(-horizon).values
        long_rets  = fwd_ret[regime_long & ~np.isnan(fwd_ret)]
        short_rets = -fwd_ret[regime_short & ~np.isnan(fwd_ret)]

        if len(long_rets) > 3:
            results[f"regime_filtered_long_{horizon}bar"] = {
                "n_signals": int(len(long_rets)),
                "mean_return_pct": round(float(long_rets.mean() * 100), 3),
                "win_rate": round(float((long_rets > 0).mean() * 100), 1),
                "t_stat": round(float(long_rets.mean() / (long_rets.std() / np.sqrt(len(long_rets)))) if long_rets.std() > 0 else 0, 2),
            }
        if len(short_rets) > 3:
            results[f"regime_filtered_short_{horizon}bar"] = {
                "n_signals": int(len(short_rets)),
                "mean_return_pct": round(float(short_rets.mean() * 100), 3),
                "win_rate": round(float((short_rets > 0).mean() * 100), 1),
                "t_stat": round(float(short_rets.mean() / (short_rets.std() / np.sqrt(len(short_rets)))) if short_rets.std() > 0 else 0, 2),
            }

    return results


# ==================== 8. SIMPLE BACKTEST V2 ====================

def backtest_v2(df, symbol, capital=500.0, risk_pct=0.02, 
                slippage_bps=25, comm_pct=0.0004,
                adx_min=20, trail_atr=2.5, stop_atr=1.8):
    """
    Simple event-driven backtest: long + short with regime gate.
    Returns trade list and equity curve summary.
    """
    if df is None or len(df) < 200:
        return {"error": "insufficient_data"}

    trades = []
    equity = capital
    peak_equity = capital
    max_dd = 0

    # State
    in_position = False
    direction = 0  # 1=long, -1=short
    entry_price = 0
    stop_price = 0
    trail_price = 0
    position_size = 0
    entry_bar = 0
    entry_time = None

    cost_pct = slippage_bps / 10000 + comm_pct  # one-way cost

    for i in range(55, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # Skip if indicators not ready
        if np.isnan(row['macd']) or np.isnan(row['macd_signal']) or np.isnan(row['adx']) or np.isnan(row['atr14']):
            continue

        close = row['close']
        atr = row['atr14']
        adx = row['adx']

        if in_position:
            # --- CHECK EXITS ---
            bars_held = i - entry_bar
            hit_stop = False
            exit_price = close
            exit_reason = ""

            if direction == 1:  # Long
                # Trailing stop update
                new_trail = close - trail_atr * atr
                if new_trail > trail_price:
                    trail_price = new_trail
                # Check stops
                if row['low'] <= stop_price:
                    exit_price = stop_price
                    hit_stop = True
                    exit_reason = "initial_stop"
                elif row['low'] <= trail_price:
                    exit_price = trail_price
                    hit_stop = True
                    exit_reason = "trailing_stop"
                elif bars_held >= 36:
                    exit_reason = "max_hold"
                # MACD cross against
                elif row['macd'] < row['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                    exit_reason = "signal_exit"

            else:  # Short
                new_trail = close + trail_atr * atr
                if new_trail < trail_price:
                    trail_price = new_trail
                if row['high'] >= stop_price:
                    exit_price = stop_price
                    hit_stop = True
                    exit_reason = "initial_stop"
                elif row['high'] >= trail_price:
                    exit_price = trail_price
                    hit_stop = True
                    exit_reason = "trailing_stop"
                elif bars_held >= 36:
                    exit_reason = "max_hold"
                elif row['macd'] > row['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                    exit_reason = "signal_exit"

            if exit_reason:
                # Calculate P&L
                exit_cost = exit_price * cost_pct * position_size
                if direction == 1:
                    gross_pnl = (exit_price - entry_price) * position_size
                else:
                    gross_pnl = (entry_price - exit_price) * position_size
                net_pnl = gross_pnl - exit_cost

                equity += net_pnl
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity * 100
                max_dd = max(max_dd, dd)

                trades.append({
                    "symbol": symbol,
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry_time": str(entry_time),
                    "exit_time": str(row.get('timestamp', i)),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(net_pnl, 2),
                    "pnl_pct": round(net_pnl / capital * 100, 2),
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                })
                in_position = False
                direction = 0
                continue

        else:
            # --- CHECK ENTRIES ---
            # Regime gate
            if adx < adx_min:
                continue

            # Long signal: MACD crosses above signal + price above EMA55
            if (row['macd'] > row['macd_signal'] and prev['macd'] <= prev['macd_signal']
                and close > row['ema_55']
                and row['ema_21'] > row['ema_55']):

                direction = 1
                entry_price = close * (1 + cost_pct)
                stop_dist = stop_atr * atr
                stop_price = entry_price - stop_dist
                trail_price = stop_price

                risk_amount = equity * risk_pct
                position_size = risk_amount / stop_dist if stop_dist > 0 else 0
                # Cap position at available equity
                max_pos = equity * 0.9 / entry_price
                position_size = min(position_size, max_pos)

                if position_size * entry_price < 5:  # Binance minimum
                    continue

                in_position = True
                entry_bar = i
                entry_time = row.get('timestamp', i)

            # Short signal: MACD crosses below signal + price below EMA55
            elif (row['macd'] < row['macd_signal'] and prev['macd'] >= prev['macd_signal']
                  and close < row['ema_55']
                  and row['ema_21'] < row['ema_55']):

                direction = -1
                entry_price = close * (1 - cost_pct)
                stop_dist = stop_atr * atr
                stop_price = entry_price + stop_dist
                trail_price = stop_price

                risk_amount = equity * risk_pct
                position_size = risk_amount / stop_dist if stop_dist > 0 else 0
                max_pos = equity * 0.9 / entry_price
                position_size = min(position_size, max_pos)

                if position_size * entry_price < 5:
                    continue

                in_position = True
                entry_bar = i
                entry_time = row.get('timestamp', i)

    # Summary
    if not trades:
        return {"total_trades": 0, "pnl": 0, "sharpe": 0}

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999

    # Sharpe approximation
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls) / 5.0)  # annualize roughly
    else:
        sharpe = 0

    # Trade direction breakdown
    long_trades  = [t for t in trades if t['direction'] == 'LONG']
    short_trades = [t for t in trades if t['direction'] == 'SHORT']
    long_pnl  = sum(t['pnl'] for t in long_trades)
    short_pnl = sum(t['pnl'] for t in short_trades)

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t['exit_reason']
        if r not in exit_reasons:
            exit_reasons[r] = {"count": 0, "pnl": 0}
        exit_reasons[r]["count"] += 1
        exit_reasons[r]["pnl"] = round(exit_reasons[r]["pnl"] + t['pnl'], 2)

    # Monthly PnL
    monthly = {}
    for t in trades:
        try:
            m = str(t['exit_time'])[:7]
        except:
            m = "unknown"
        monthly[m] = round(monthly.get(m, 0) + t['pnl'], 2)

    return {
        "symbol": symbol,
        "total_trades": len(trades),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "total_pnl": round(total_pnl, 2),
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
        "final_equity": round(equity, 2),
        "return_pct": round((equity - capital) / capital * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "profit_factor": round(float(profit_factor), 3),
        "sharpe": round(float(sharpe), 3),
        "exit_reasons": exit_reasons,
        "monthly_pnl": monthly,
        "first_5_trades": trades[:5],
        "last_5_trades": trades[-5:],
        "config": {
            "capital": capital,
            "risk_pct": risk_pct,
            "slippage_bps": slippage_bps,
            "commission_pct": comm_pct,
            "adx_min": adx_min,
            "trail_atr": trail_atr,
            "stop_atr": stop_atr,
        }
    }


# ==================== 9. WALK-FORWARD V2 ====================

def walk_forward_v2(df, symbol, train_days=365, test_days=90):
    """Walk-forward on a single symbol with fixed params"""
    if df is None or 'timestamp' not in df.columns or len(df) < 500:
        return {"error": "insufficient_data"}

    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    start_date = df['date'].min()
    end_date = df['date'].max()

    folds = []
    current = pd.Timestamp(start_date) + pd.Timedelta(days=train_days)

    while current + pd.Timedelta(days=test_days) <= pd.Timestamp(end_date):
        test_start = current
        test_end = current + pd.Timedelta(days=test_days)

        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
        test_df = df[test_mask].copy().reset_index(drop=True)

        if len(test_df) > 20:
            # Need indicators from before test period
            pre_mask = df['timestamp'] < test_end
            full_df = df[pre_mask].copy().reset_index(drop=True)
            # Re-add indicators
            full_df = add_indicators(full_df)
            # Only backtest on test portion
            test_start_idx = len(full_df) - len(test_df)
            test_slice = full_df.iloc[max(0, test_start_idx-60):].reset_index(drop=True)

            result = backtest_v2(test_slice, symbol, capital=500)
            folds.append({
                "test_period": f"{test_start.date()} ~ {test_end.date()}",
                "pnl": result.get("total_pnl", 0),
                "trades": result.get("total_trades", 0),
                "return_pct": result.get("return_pct", 0),
                "win_rate": result.get("win_rate", 0),
                "long_trades": result.get("long_trades", 0),
                "short_trades": result.get("short_trades", 0),
            })

        current += pd.Timedelta(days=test_days)

    if not folds:
        return {"error": "no_valid_folds"}

    pnls = [f['pnl'] for f in folds]
    positive = [p for p in pnls if p > 0]

    return {
        "symbol": symbol,
        "train_days": train_days,
        "test_days": test_days,
        "n_folds": len(folds),
        "n_positive": len(positive),
        "fold_win_rate": round(len(positive) / len(folds) * 100, 1) if folds else 0,
        "total_oos_pnl": round(sum(pnls), 2),
        "avg_fold_pnl": round(float(np.mean(pnls)), 2),
        "std_fold_pnl": round(float(np.std(pnls)), 2),
        "oos_sharpe": round(float(np.mean(pnls) / np.std(pnls) * np.sqrt(4)) if np.std(pnls) > 0 else 0, 3),  # annualize ~4 folds/year
        "worst_fold": round(float(min(pnls)), 2),
        "best_fold": round(float(max(pnls)), 2),
        "folds": folds,
    }


# ==================== 10. 500 USDT FEASIBILITY ====================

def check_500_feasibility(data_dict):
    """Check if strategies are feasible with 500 USDT capital"""
    result = {}
    for sym, intervals in data_dict.items():
        df = pick_interval_df(intervals)
        if df is None or len(df) == 0:
            continue
        last_price = float(df['close'].iloc[-1])
        min_notional = 5.0  # Binance USDT-M minimum
        min_qty_usd = min_notional
        # With 500 USDT, 2% risk = 10 USDT risk budget
        # If stop is 2 ATR away:
        atr_pct = float(df['atr_pct'].dropna().iloc[-1]) if 'atr_pct' in df.columns else 2.0
        stop_distance_pct = atr_pct * 1.8 / 100  # 1.8 ATR
        position_size_usd = 10.0 / stop_distance_pct if stop_distance_pct > 0 else 0
        leverage_needed = position_size_usd / 500.0

        result[sym] = {
            "last_price": round(last_price, 2),
            "min_notional_usd": min_qty_usd,
            "atr_pct_latest": round(atr_pct, 2),
            "stop_distance_pct": round(stop_distance_pct * 100, 2),
            "ideal_position_usd": round(position_size_usd, 2),
            "leverage_needed": round(leverage_needed, 2),
            "feasible": leverage_needed <= 10,
            "note": "OK" if leverage_needed <= 3 else ("需要杠杆" if leverage_needed <= 10 else "资金不足"),
        }
    return result


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("Phase 2 Diagnostic — Starting")
    print("=" * 60)

    report = {
        "generated": datetime.now().isoformat(),
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "slippage_bps": SLIPPAGE_BPS,
            "commission_pct": COMMISSION_PCT,
            "symbols": SYMBOLS,
        }
    }

    # 1. Schema discovery
    print("\n[1/9] Discovering DB schema...")
    try:
        report["schema"] = discover_schema(DB_PATH)
        print(f"  Found {len(report['schema']['tables'])} tables")
        for tbl, info in report['schema']['tables'].items():
            print(f"    {tbl}: {info['row_count']} rows, cols={info['columns'][:5]}...")
    except Exception as e:
        report["schema"] = {"error": str(e)}
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # 2. Load all data
    print("\n[2/9] Loading candle data...")
    data_dict = {}  # {symbol: {interval: df}}
    for sym in SYMBOLS:
        data_dict[sym] = {}
        for interval in ["4h", "1d"]:
            df, tbl = load_candles(DB_PATH, sym, interval)
            if df is not None and len(df) > 0:
                df = add_indicators(df)
                data_dict[sym][interval] = df
                ts_range = ""
                if 'timestamp' in df.columns:
                    ts_range = f" | {df['timestamp'].min()} ~ {df['timestamp'].max()}"
                print(f"  {sym}/{interval}: {len(df)} bars (table={tbl}){ts_range}")
            else:
                data_dict[sym][interval] = None
                print(f"  {sym}/{interval}: NOT FOUND")

    # 3. Cross-correlation
    print("\n[3/9] Computing cross-asset correlations...")
    try:
        report["correlations_1d"] = compute_correlations(data_dict, '1d')
        report["correlations_4h"] = compute_correlations(data_dict, '4h')
        if "correlation_matrix" in report.get("correlations_1d", {}):
            for k, v in report["correlations_1d"]["correlation_matrix"].items():
                print(f"  {k}: {v}")
    except Exception as e:
        report["correlations"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # 4. Cost estimation
    print("\n[4/9] Estimating trading costs...")
    report["costs"] = {}
    for sym in SYMBOLS:
        df = pick_interval_df(data_dict[sym])
        try:
            report["costs"][sym] = estimate_costs(df, sym)
            if report["costs"][sym]:
                print(f"  {sym}: {report['costs'][sym].get('recommended_total_cost_bps', '?')} bps round trip")
        except Exception as e:
            report["costs"][sym] = {"error": str(e)}

    # 5. Regime analysis
    print("\n[5/9] Analyzing regimes...")
    report["regimes"] = {}
    for sym in SYMBOLS:
        for interval in ["4h", "1d"]:
            df = data_dict[sym].get(interval)
            if df is not None:
                try:
                    key = f"{sym}_{interval}"
                    report["regimes"][key] = analyze_regimes(df, sym)
                    r = report["regimes"][key]
                    print(f"  {key}: ADX mean={r.get('adx_stats',{}).get('mean','?')}, ATR%={r.get('atr_pct_stats',{}).get('mean','?')}")
                except Exception as e:
                    report["regimes"][f"{sym}_{interval}"] = {"error": str(e)}

    # 6. Raw signal edge
    print("\n[6/9] Testing raw signal edge...")
    report["signal_edge"] = {}
    for sym in SYMBOLS:
        for interval in ["4h", "1d"]:
            df = data_dict[sym].get(interval)
            if df is not None and len(df) > 200:
                try:
                    key = f"{sym}_{interval}"
                    report["signal_edge"][key] = test_signal_edge(df, sym)
                    # Print most interesting result
                    for sig_name, sig_data in report["signal_edge"][key].items():
                        if isinstance(sig_data, dict) and 'n_signals' in sig_data:
                            t = sig_data.get('t_stat', 0)
                            if abs(t) > 1.5:
                                print(f"  {key}/{sig_name}: n={sig_data['n_signals']}, ret={sig_data['mean_return_pct']}%, t={t}")
                except Exception as e:
                    report["signal_edge"][f"{sym}_{interval}"] = {"error": str(e)}

    # 7. Backtest V2 (long + short with regime gate)
    print("\n[7/9] Running V2 backtest (long+short, ADX gate)...")
    report["backtest_v2"] = {}
    for sym in SYMBOLS:
        df = data_dict[sym].get('4h')
        if df is not None and len(df) > 500:
            try:
                result = backtest_v2(df, sym, capital=INITIAL_CAPITAL)
                report["backtest_v2"][f"{sym}_4h"] = result
                print(f"  {sym}/4h: trades={result.get('total_trades',0)}, PnL={result.get('total_pnl',0)}, "
                      f"Sharpe={result.get('sharpe',0)}, WR={result.get('win_rate',0)}%")
                print(f"    Long={result.get('long_trades',0)} ({result.get('long_pnl',0)}), "
                      f"Short={result.get('short_trades',0)} ({result.get('short_pnl',0)})")
            except Exception as e:
                report["backtest_v2"][f"{sym}_4h"] = {"error": str(e)}
                traceback.print_exc()

        df = data_dict[sym].get('1d')
        if df is not None and len(df) > 200:
            try:
                result = backtest_v2(df, sym, capital=INITIAL_CAPITAL)
                report["backtest_v2"][f"{sym}_1d"] = result
                print(f"  {sym}/1d: trades={result.get('total_trades',0)}, PnL={result.get('total_pnl',0)}, "
                      f"Sharpe={result.get('sharpe',0)}, WR={result.get('win_rate',0)}%")
            except Exception as e:
                report["backtest_v2"][f"{sym}_1d"] = {"error": str(e)}

    # 8. Walk-forward on BTC 4h (longest history)
    print("\n[8/9] Walk-forward on BTC 4h...")
    try:
        df_btc = data_dict["BTCUSDT"].get("4h")
        if df_btc is not None and len(df_btc) > 1000:
            report["walkforward_btc_4h"] = walk_forward_v2(df_btc, "BTCUSDT", train_days=365, test_days=90)
            wf = report["walkforward_btc_4h"]
            print(f"  Folds={wf.get('n_folds',0)}, WinRate={wf.get('fold_win_rate',0)}%, "
                  f"OOS_PnL={wf.get('total_oos_pnl',0)}, OOS_Sharpe={wf.get('oos_sharpe',0)}")
    except Exception as e:
        report["walkforward_btc_4h"] = {"error": str(e)}
        traceback.print_exc()

    # 9. 500 USDT feasibility
    print("\n[9/9] Checking 500 USDT feasibility...")
    try:
        report["feasibility_500"] = check_500_feasibility(data_dict)
        for sym, info in report["feasibility_500"].items():
            print(f"  {sym}: leverage={info.get('leverage_needed','?')}x, {info.get('note','')}")
    except Exception as e:
        report["feasibility_500"] = {"error": str(e)}

    # Save report
    print(f"\n{'='*60}")
    print(f"Saving report to {OUTPUT_FILE}...")

    # Make JSON-safe
    def make_safe(obj):
        if isinstance(obj, dict):
            return {k: make_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_safe(v) for v in obj]
        else:
            return safe_json(obj)

    report = make_safe(report)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"Done! File: {OUTPUT_FILE}")
    print(f"请把 {OUTPUT_FILE} 返回给我进行分析。")


if __name__ == "__main__":
    main()

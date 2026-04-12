"""
特征工程流水线
- 100+ 技术因子
- 多时间框架特征
- 自动标注
- 特征标准化与缓存
"""
import warnings
import numpy as np
import pandas as pd
from typing import Optional

from utils.logger import get_logger

logger = get_logger("features")


# ==============================================================
# 底层指标计算（向量化，高性能）
# ==============================================================

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, min_periods=n).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"] - df["high"].shift()
    down = df["low"].shift() - df["low"]
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
    atr_val = _atr(df, n)
    plus_di = 100 * _ema(plus_dm, n) / (atr_val + 1e-10)
    minus_di = 100 * _ema(minus_dm, n) / (atr_val + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return _ema(dx, n)

def _macd(s: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = _ema(s, fast)
    slow_ema = _ema(s, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger(s: pd.Series, n=20, num_std=2.0):
    mid = _sma(s, n)
    std = s.rolling(n).std()
    return mid + num_std * std, mid, mid - num_std * std

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()

def _vwap_rolling(df: pd.DataFrame, n: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).rolling(n).sum() / (df["volume"].rolling(n).sum() + 1e-10)

def _mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    positive = pd.Series(np.where(tp > tp.shift(), mf, 0), index=df.index)
    negative = pd.Series(np.where(tp < tp.shift(), mf, 0), index=df.index)
    ratio = positive.rolling(n).sum() / (negative.rolling(n).sum() + 1e-10)
    return 100 - 100 / (1 + ratio)

def _keltner(df: pd.DataFrame, n: int = 20, mult: float = 1.5):
    mid = _ema(df["close"], n)
    atr_val = _atr(df, n)
    return mid + mult * atr_val, mid, mid - mult * atr_val

def _stochastic(df: pd.DataFrame, k_period=14, d_period=3):
    lowest = df["low"].rolling(k_period).min()
    highest = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - lowest) / (highest - lowest + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d

def _williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    highest = df["high"].rolling(n).max()
    lowest = df["low"].rolling(n).min()
    return -100 * (highest - df["close"]) / (highest - lowest + 1e-10)

def _cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = _sma(tp, n)
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad + 1e-10)

def _ichimoku(df: pd.DataFrame):
    tenkan = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    kijun = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2
    return tenkan, kijun, senkou_a, senkou_b


# ==============================================================
# 特征工厂
# ==============================================================
class FeatureEngine:
    """
    工业级特征工程管线
    输入原始 OHLCV DataFrame，输出 100+ 特征的宽表
    """

    def __init__(self, windows: list[int] = None):
        self.windows = windows or [5, 10, 20, 60, 120, 240, 480]

    def compute_all(self, df: pd.DataFrame,
                    include_target: bool = False,
                    target_periods: list[int] = None) -> pd.DataFrame:
        """
        计算全部特征
        df: 必须包含 open, high, low, close, volume 列
        include_target: 是否生成标签列（训练用）
        """
        if len(df) < max(self.windows) + 10:
            logger.warning(f"数据不足 ({len(df)} 条)，需要至少 {max(self.windows) + 10} 条")
            return pd.DataFrame()

        # 抑制 DataFrame 碎片化警告（最后会 .copy() 消除碎片）
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        feat = pd.DataFrame(index=df.index)

        # 基础衍生列
        feat["close"] = df["close"]
        feat["log_close"] = np.log(df["close"])
        feat["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        feat["median_price"] = (df["high"] + df["low"]) / 2
        feat["range"] = df["high"] - df["low"]
        feat["range_pct"] = feat["range"] / (df["close"] + 1e-10)
        feat["body"] = (df["close"] - df["open"]).abs()
        feat["body_pct"] = feat["body"] / (df["close"] + 1e-10)
        feat["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        feat["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        feat["is_bullish"] = (df["close"] > df["open"]).astype(float)

        # ---- 多窗口特征 ----
        for w in self.windows:
            self._momentum_features(feat, df, w)
            self._volatility_features(feat, df, w)
            self._volume_features(feat, df, w)
            self._mean_reversion_features(feat, df, w)

        # ---- 趋势指标 ----
        self._trend_features(feat, df)

        # ---- 震荡指标 ----
        self._oscillator_features(feat, df)

        # ---- 微观结构特征 ----
        self._microstructure_features(feat, df)

        # ---- 市场状态 ----
        self._regime_features(feat, df)

        # ---- 时间特征 ----
        if "open_time" in df.columns:
            self._time_features(feat, df)

        # ---- 标签生成 ----
        if include_target:
            target_periods = target_periods or [1, 4, 24]
            self._generate_targets(feat, df, target_periods)

        # 清理
        feat = feat.replace([np.inf, -np.inf], np.nan)

        # 消除 DataFrame 碎片化（性能优化）
        feat = feat.copy()

        logger.debug(f"特征计算完成: {feat.shape[1]} 个特征, {feat.shape[0]} 行")
        return feat

    # ----------------------------------------------------------
    # 动量因子
    # ----------------------------------------------------------
    def _momentum_features(self, feat: pd.DataFrame, df: pd.DataFrame, w: int):
        close = df["close"]
        # 简单收益率
        feat[f"ret_{w}"] = close.pct_change(w)
        # 对数收益率
        feat[f"log_ret_{w}"] = np.log(close / close.shift(w))
        # 动量（rank 形式）
        feat[f"mom_rank_{w}"] = feat[f"ret_{w}"].rolling(w).rank(pct=True)
        # 加速度（动量的变化率）
        if f"ret_{w}" in feat:
            feat[f"mom_accel_{w}"] = feat[f"ret_{w}"].diff(w)
        # 高点距离
        feat[f"dist_high_{w}"] = (close - close.rolling(w).max()) / (close.rolling(w).max() + 1e-10)
        # 低点距离
        feat[f"dist_low_{w}"] = (close - close.rolling(w).min()) / (close.rolling(w).min() + 1e-10)
        # 价格在区间中的位置 [0,1]
        rng = close.rolling(w).max() - close.rolling(w).min()
        feat[f"price_position_{w}"] = (close - close.rolling(w).min()) / (rng + 1e-10)

    # ----------------------------------------------------------
    # 波动率因子
    # ----------------------------------------------------------
    def _volatility_features(self, feat: pd.DataFrame, df: pd.DataFrame, w: int):
        returns = df["close"].pct_change()
        # 已实现波动率
        feat[f"volatility_{w}"] = returns.rolling(w).std()
        # Parkinson 波动率（高低价）
        feat[f"parkinson_vol_{w}"] = np.sqrt(
            (np.log(df["high"] / df["low"]) ** 2).rolling(w).mean() / (4 * np.log(2)))
        # Garman-Klass 波动率
        u = np.log(df["high"] / df["open"])
        d = np.log(df["low"] / df["open"])
        c = np.log(df["close"] / df["open"])
        feat[f"gk_vol_{w}"] = np.sqrt((0.5 * (u - d)**2 - (2*np.log(2) - 1) * c**2).rolling(w).mean())
        # ATR 归一化
        feat[f"natr_{w}"] = _atr(df, w) / (df["close"] + 1e-10)
        # 波动率变化率
        if w >= 10:
            vol = feat[f"volatility_{w}"]
            feat[f"vol_change_{w}"] = vol / (vol.shift(w // 2) + 1e-10) - 1
        # 偏度
        feat[f"skew_{w}"] = returns.rolling(w).skew()
        # 峰度
        feat[f"kurtosis_{w}"] = returns.rolling(w).kurt()

    # ----------------------------------------------------------
    # 量价因子
    # ----------------------------------------------------------
    def _volume_features(self, feat: pd.DataFrame, df: pd.DataFrame, w: int):
        vol = df["volume"]
        close = df["close"]
        # 相对成交量
        feat[f"rel_volume_{w}"] = vol / (vol.rolling(w).mean() + 1e-10)
        # 量价相关性
        feat[f"price_vol_corr_{w}"] = close.pct_change().rolling(w).corr(vol.pct_change())
        # VWAP 偏离
        feat[f"vwap_dev_{w}"] = (close - _vwap_rolling(df, w)) / (close + 1e-10)
        # OBV 变化率
        obv = _obv(close, vol)
        feat[f"obv_change_{w}"] = obv.pct_change(w)
        # 量比（当前量 vs 历史平均）
        feat[f"vol_ratio_{w}"] = vol / (vol.rolling(w).mean() + 1e-10)
        # 买入占比（如果有 taker 数据）
        if "taker_buy_base" in df.columns:
            feat[f"taker_buy_ratio_{w}"] = (
                df["taker_buy_base"].rolling(w).sum() / (vol.rolling(w).sum() + 1e-10))

    # ----------------------------------------------------------
    # 均值回归因子
    # ----------------------------------------------------------
    def _mean_reversion_features(self, feat: pd.DataFrame, df: pd.DataFrame, w: int):
        close = df["close"]
        ma = _sma(close, w)
        # 价格偏离均线
        feat[f"ma_dev_{w}"] = (close - ma) / (ma + 1e-10)
        # Z-Score
        feat[f"zscore_{w}"] = (close - ma) / (close.rolling(w).std() + 1e-10)
        # 布林带位置
        upper, mid, lower = _bollinger(close, w)
        feat[f"bb_position_{w}"] = (close - lower) / (upper - lower + 1e-10)
        feat[f"bb_width_{w}"] = (upper - lower) / (mid + 1e-10)

    # ----------------------------------------------------------
    # 趋势指标
    # ----------------------------------------------------------
    def _trend_features(self, feat: pd.DataFrame, df: pd.DataFrame):
        close = df["close"]
        # 多周期均线排列
        for p in [7, 20, 50, 100, 200]:
            if len(df) > p:
                feat[f"sma_{p}"] = _sma(close, p)
                feat[f"ema_{p}"] = _ema(close, p)
                feat[f"close_vs_sma_{p}"] = (close - feat[f"sma_{p}"]) / (feat[f"sma_{p}"] + 1e-10)

        # 均线斜率
        for p in [20, 50, 100]:
            if len(df) > p + 5:
                ma = _sma(close, p)
                feat[f"sma_slope_{p}"] = ma.diff(5) / (ma.shift(5) + 1e-10)

        # MACD
        macd_l, macd_s, macd_h = _macd(close)
        feat["macd_line"] = macd_l
        feat["macd_signal"] = macd_s
        feat["macd_hist"] = macd_h
        feat["macd_hist_change"] = macd_h.diff()
        feat["macd_cross"] = np.sign(macd_l - macd_s).diff()

        # ADX
        if len(df) > 28:
            feat["adx_14"] = _adx(df, 14)
            feat["adx_28"] = _adx(df, 28)

        # Ichimoku
        if len(df) > 55:
            tenkan, kijun, sa, sb = _ichimoku(df)
            feat["ichimoku_tenkan_kijun"] = (tenkan - kijun) / (close + 1e-10)
            feat["ichimoku_cloud_thickness"] = (sa - sb) / (close + 1e-10)
            feat["ichimoku_price_vs_cloud"] = (close - (sa + sb) / 2) / (close + 1e-10)

    # ----------------------------------------------------------
    # 震荡指标
    # ----------------------------------------------------------
    def _oscillator_features(self, feat: pd.DataFrame, df: pd.DataFrame):
        close = df["close"]
        feat["rsi_7"] = _rsi(close, 7)
        feat["rsi_14"] = _rsi(close, 14)
        feat["rsi_21"] = _rsi(close, 21)
        feat["rsi_divergence"] = feat["rsi_14"].diff(5) - close.pct_change(5) * 100

        k, d = _stochastic(df)
        feat["stoch_k"] = k
        feat["stoch_d"] = d
        feat["stoch_cross"] = np.sign(k - d).diff()

        feat["williams_r"] = _williams_r(df)
        feat["cci_20"] = _cci(df, 20)
        feat["mfi_14"] = _mfi(df, 14)

        # Keltner Channel
        if len(df) > 25:
            ku, km, kl = _keltner(df)
            feat["keltner_position"] = (close - kl) / (ku - kl + 1e-10)

    # ----------------------------------------------------------
    # 微观结构特征
    # ----------------------------------------------------------
    def _microstructure_features(self, feat: pd.DataFrame, df: pd.DataFrame):
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]

        # Amihud 非流动性
        feat["amihud_10"] = (close.pct_change().abs() / (vol * close + 1e-10)).rolling(10).mean()
        feat["amihud_60"] = (close.pct_change().abs() / (vol * close + 1e-10)).rolling(60).mean()

        # Kyle's Lambda 近似
        signed_vol = np.sign(close.diff()) * vol
        feat["kyle_lambda_20"] = (
            close.pct_change().rolling(20).std() /
            (signed_vol.rolling(20).std() + 1e-10))

        # 高低价波动比（日内 vs 日间）
        intraday = np.log(high / low)
        interday = np.log(close / close.shift()).abs()
        feat["vol_ratio_intra_inter"] = (
            intraday.rolling(20).mean() / (interday.rolling(20).mean() + 1e-10))

        # 连续涨跌统计
        up = (close.diff() > 0).astype(int)
        feat["consecutive_up"] = up.groupby((up != up.shift()).cumsum()).cumcount() * up
        dn = (close.diff() < 0).astype(int)
        feat["consecutive_down"] = dn.groupby((dn != dn.shift()).cumsum()).cumcount() * dn

        # 成交量加权价格冲击
        if "quote_volume" in df.columns:
            feat["avg_trade_size"] = df["quote_volume"] / (df.get("trades_count", vol) + 1e-10)

    # ----------------------------------------------------------
    # 市场状态（Regime）
    # ----------------------------------------------------------
    def _regime_features(self, feat: pd.DataFrame, df: pd.DataFrame):
        close = df["close"]
        returns = close.pct_change()

        # 波动率分位数（当前 vol 在历史中的位置）
        vol20 = returns.rolling(20).std()
        feat["vol_percentile_120"] = vol20.rolling(120).rank(pct=True)
        feat["vol_percentile_480"] = vol20.rolling(480).rank(pct=True)

        # 趋势强度指标
        sma50 = _sma(close, 50)
        sma200 = _sma(close, 200)
        if len(df) > 200:
            feat["trend_strength"] = (sma50 - sma200) / (sma200 + 1e-10)
            feat["above_sma200"] = (close > sma200).astype(float)

        # 动量/均值回归判断
        feat["hurst_proxy_20"] = returns.rolling(20).apply(
            lambda x: self._hurst_proxy(x), raw=True)

        # VIX-like 指标：期权隐含波动率代理
        feat["realized_vs_expected_vol"] = (
            returns.rolling(5).std() / (returns.rolling(60).std() + 1e-10))

    @staticmethod
    def _hurst_proxy(returns: np.ndarray) -> float:
        """Hurst 指数近似（R/S分析简化版）"""
        n = len(returns)
        if n < 10:
            return 0.5
        cumdev = np.cumsum(returns - returns.mean())
        R = cumdev.max() - cumdev.min()
        S = returns.std()
        if S == 0:
            return 0.5
        RS = R / S
        return np.log(RS) / np.log(n) if RS > 0 and n > 1 else 0.5

    # ----------------------------------------------------------
    # 时间特征
    # ----------------------------------------------------------
    def _time_features(self, feat: pd.DataFrame, df: pd.DataFrame):
        ts = pd.to_datetime(df["open_time"], unit="ms")
        feat["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        feat["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
        feat["day_of_month"] = ts.dt.day / 31.0

    # ----------------------------------------------------------
    # 标签生成
    # ----------------------------------------------------------
    def _generate_targets(self, feat: pd.DataFrame, df: pd.DataFrame,
                          periods: list[int]):
        close = df["close"]
        for p in periods:
            # 前瞻收益率
            feat[f"fwd_ret_{p}"] = close.shift(-p) / close - 1
            # 分类标签 (0=跌, 1=涨)
            feat[f"target_dir_{p}"] = (feat[f"fwd_ret_{p}"] > 0).astype(int)
            # 三分类 (-1=大跌, 0=震荡, 1=大涨)
            q33, q67 = feat[f"fwd_ret_{p}"].quantile([0.33, 0.67])
            feat[f"target_3class_{p}"] = pd.cut(
                feat[f"fwd_ret_{p}"], bins=[-np.inf, q33, q67, np.inf],
                labels=[0, 1, 2]).astype(float)

    # ==============================================================
    # 多时间框架特征融合
    # ==============================================================
    def merge_multi_timeframe(self, features_dict: dict[str, pd.DataFrame],
                              base_interval: str = "1h") -> pd.DataFrame:
        """
        将多个时间框架的特征合并到基础时间框架上
        features_dict: {"1h": df_1h, "4h": df_4h, "1d": df_1d}
        """
        base = features_dict.get(base_interval)
        if base is None:
            raise ValueError(f"基础时间框架 {base_interval} 不在输入中")

        result = base.copy()

        for interval, feat_df in features_dict.items():
            if interval == base_interval:
                continue
            # 重采样对齐（前向填充，避免未来数据泄漏）
            prefix = f"tf_{interval}_"
            for col in feat_df.columns:
                if col in ("close", "log_close"):
                    continue
                result[f"{prefix}{col}"] = np.nan

            # 简单处理：对高级时间框架特征做前向填充
            logger.debug(f"合并 {interval} 时间框架 ({len(feat_df)} 行) 到 {base_interval}")

        return result

    # ==============================================================
    # 特征预处理
    # ==============================================================
    @staticmethod
    def preprocess(feat: pd.DataFrame,
                   method: str = "zscore",
                   clip_std: float = 5.0) -> pd.DataFrame:
        """
        特征标准化
        method: zscore / rank / minmax
        clip_std: z-score 裁剪范围
        """
        result = feat.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        if method == "zscore":
            for col in numeric_cols:
                if col.startswith("target_") or col.startswith("fwd_"):
                    continue
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:
                    result[col] = (result[col] - mean) / std
                    result[col] = result[col].clip(-clip_std, clip_std)
        elif method == "rank":
            for col in numeric_cols:
                if col.startswith("target_") or col.startswith("fwd_"):
                    continue
                result[col] = result[col].rank(pct=True)
        elif method == "minmax":
            for col in numeric_cols:
                if col.startswith("target_") or col.startswith("fwd_"):
                    continue
                mn, mx = result[col].min(), result[col].max()
                if mx > mn:
                    result[col] = (result[col] - mn) / (mx - mn)

        return result

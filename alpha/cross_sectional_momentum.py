"""
Cross-Sectional Momentum Strategy — v1.0 (路线 A MVP)

核心思路 (参考 Liu-Tsyvinski-Wu 2022 JoF, Unravel 2025, CTREND JFQA 2025):
  不预测单个币种涨跌, 而是在每个 rebalance 日对 universe 内所有币种按因子排序,
  做多 top quintile, 做空 bottom quintile, 构建市场中性组合。

因子集 (Phase 1 MVP, 5 个):
  1. momentum_30d       - 过去 30 日累计收益率 (主因子)
  2. reversal_7d        - 过去 7 日累计收益率 (反向)
  3. volatility_30d     - 过去 30 日日收益率标准差 (低波优先)
  4. liquidity_amihud   - |return| / volume 的 30 日均值
  5. size_proxy         - log(过去 30 日平均日成交额) 倒序 (小市值优先)

组合构建:
  - Top 20% 做多, Bottom 20% 做空
  - Inverse volatility weighting (crypto 内部波动差异极大)
  - 每 7 天 rebalance (日频 rebalance 会被成本吃光)

usage:
  python cross_sectional_backtest.py \
      --db data/quant.db \
      --start 2022-01-01 \
      --top-n 50 \
      --output analysis/output/xs_mom_mvp.json
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("xs_momentum")


@dataclass
class CrossSectionalConfig:
    # Universe
    top_n: int = 50

    # 因子窗口
    momentum_window: int = 30
    reversal_window: int = 7
    volatility_window: int = 30
    liquidity_window: int = 30

    # 组合构建
    long_pct: float = 0.20        # Top 20%
    short_pct: float = 0.20       # Bottom 20%
    rebalance_freq_days: int = 7

    # 加权
    use_inverse_vol_weight: bool = True
    vol_target_annual: float = 0.20    # 年化 20% 目标波动

    # 成本 (保守估计)
    slippage_bps: float = 20.0
    commission_bps: float = 10.0

    # 因子权重 (简单平均作为 baseline)
    factor_weights: dict = field(default_factory=lambda: {
        "momentum_30d": 1.0,
        "reversal_7d": -0.3,        # 反向符号
        "volatility_30d": -0.5,     # 低波优先
        "liquidity_amihud": -0.3,   # 高流动性优先
        "size_proxy": 0.3,          # 小市值优先
    })


class CrossSectionalMomentumStrategy:
    """
    横截面 momentum 多空策略

    关键约束:
      - 严格 point-in-time 数据使用 (避免 lookahead bias)
      - Inverse vol weighting 避免被高波币种主导
      - Rebalance 7 天一次 (不要日频, 成本吃光)
    """

    def __init__(self, config: Optional[CrossSectionalConfig] = None):
        self.cfg = config or CrossSectionalConfig()

    # ----------------------------------------------------------------
    # 因子计算 (对单个币种, 在给定日期 as-of)
    # ----------------------------------------------------------------
    def compute_factors(
        self,
        klines: pd.DataFrame,
        as_of_idx: int,
    ) -> Optional[dict]:
        """
        在 as_of_idx 位置计算该币种的所有因子

        Args:
            klines: 单币种日线 DataFrame, 列含 close/volume
            as_of_idx: 计算截止位置 (含)

        Returns:
            {factor_name: value} 或 None (数据不足)
        """
        if as_of_idx < max(
            self.cfg.momentum_window,
            self.cfg.volatility_window,
            self.cfg.liquidity_window,
        ):
            return None

        # 只用 as_of_idx 之前的数据
        window_start = as_of_idx - max(
            self.cfg.momentum_window,
            self.cfg.volatility_window,
            self.cfg.liquidity_window,
        )
        sub = klines.iloc[window_start:as_of_idx + 1].copy()
        if sub.empty or len(sub) < self.cfg.momentum_window:
            return None

        close = sub["close"].values
        if close[-1] <= 0 or close[-self.cfg.momentum_window] <= 0:
            return None

        # 1) Momentum 30d
        momentum_30d = close[-1] / close[-self.cfg.momentum_window] - 1.0

        # 2) Reversal 7d
        if len(close) >= self.cfg.reversal_window:
            reversal_7d = close[-1] / close[-self.cfg.reversal_window] - 1.0
        else:
            return None

        # 3) Volatility 30d (日收益率标准差)
        returns = np.diff(close[-self.cfg.volatility_window - 1:]) / close[-self.cfg.volatility_window - 1:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < 10:
            return None
        volatility_30d = float(np.std(returns))

        # 4) Amihud liquidity (成交额调整的波动)
        volume = sub["volume"].values
        volume_window = volume[-self.cfg.liquidity_window:]
        abs_returns = np.abs(np.diff(close[-self.cfg.liquidity_window - 1:]) / close[-self.cfg.liquidity_window - 1:-1])
        abs_returns = abs_returns[np.isfinite(abs_returns)]
        if len(abs_returns) < 10 or volume_window.sum() <= 0:
            return None
        turnover_usd = (close[-self.cfg.liquidity_window:] * volume_window)
        turnover_usd = turnover_usd[turnover_usd > 0]
        if len(turnover_usd) == 0:
            return None
        liquidity_amihud = float(
            np.mean(abs_returns[: len(turnover_usd) - 1] / turnover_usd[:-1])
        ) if len(turnover_usd) > 1 else 0.0

        # 5) Size proxy (成交额倒序作为市值替代)
        avg_turnover = float(np.mean(turnover_usd))
        size_proxy = -np.log(max(avg_turnover, 1e-6))   # 负号 -> 小 turnover 得分高

        return {
            "momentum_30d": float(momentum_30d),
            "reversal_7d": float(reversal_7d),
            "volatility_30d": float(volatility_30d),
            "liquidity_amihud": float(liquidity_amihud),
            "size_proxy": float(size_proxy),
        }

    # ----------------------------------------------------------------
    # 排序 + 组合构建
    # ----------------------------------------------------------------
    def rank_and_build_portfolio(
        self, factors_by_symbol: dict[str, dict]
    ) -> dict[str, float]:
        """
        对所有币种按综合因子得分排序，返回仓位权重

        Returns:
            {symbol: weight} where sum(|weight|) ≈ 2.0 (200% gross exposure)
        """
        if len(factors_by_symbol) < 10:
            return {}

        # 1) 对每个因子做 cross-sectional z-score 标准化
        df = pd.DataFrame.from_dict(factors_by_symbol, orient="index")
        for factor in df.columns:
            mean = df[factor].mean()
            std = df[factor].std()
            if std > 0:
                df[factor] = (df[factor] - mean) / std
            else:
                df[factor] = 0.0

        # 2) 综合得分
        composite = pd.Series(0.0, index=df.index)
        for factor, weight in self.cfg.factor_weights.items():
            if factor in df.columns:
                composite = composite + df[factor] * weight

        # 3) 排序，选 top / bottom quintile
        composite = composite.dropna().sort_values()
        n = len(composite)
        if n < 10:
            return {}

        n_long = max(1, int(n * self.cfg.long_pct))
        n_short = max(1, int(n * self.cfg.short_pct))

        longs = composite.tail(n_long).index.tolist()
        shorts = composite.head(n_short).index.tolist()

        # 4) Inverse volatility weighting (若开启)
        weights = {}
        if self.cfg.use_inverse_vol_weight:
            # 用因子得分反推 volatility (df 已标准化)
            # 这里需要原始 volatility, 留给回测引擎传入
            # baseline: equal weight
            for s in longs:
                weights[s] = 1.0 / n_long
            for s in shorts:
                weights[s] = -1.0 / n_short
        else:
            for s in longs:
                weights[s] = 1.0 / n_long
            for s in shorts:
                weights[s] = -1.0 / n_short

        return weights

    def apply_inverse_vol_weighting(
        self,
        weights: dict[str, float],
        volatilities: dict[str, float],
    ) -> dict[str, float]:
        """
        对已有 weights 应用 inverse volatility scaling

        Args:
            weights: 原始权重
            volatilities: {symbol: annualized_vol}

        Returns:
            scaled weights
        """
        # 分别处理多头和空头
        longs = {s: w for s, w in weights.items() if w > 0}
        shorts = {s: w for s, w in weights.items() if w < 0}

        def rescale(sub: dict[str, float]) -> dict[str, float]:
            if not sub:
                return {}
            inv_vols = {}
            for s in sub:
                vol = volatilities.get(s, 0.0)
                inv_vols[s] = 1.0 / max(vol, 0.01)
            total = sum(inv_vols.values())
            if total <= 0:
                return sub
            sign = 1.0 if list(sub.values())[0] > 0 else -1.0
            return {s: sign * (inv / total) for s, inv in inv_vols.items()}

        result = {}
        result.update(rescale(longs))
        result.update(rescale(shorts))
        return result

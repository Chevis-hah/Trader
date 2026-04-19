"""
Validation utilities for time-series backtests.

遵循 López de Prado 2018《Advances in Financial Machine Learning》第 7 章与
Bailey et al. 2015《Probability of Backtest Overfitting》:
  - PurgedKFold: 带 purge + embargo 的 K-Fold (避免时序泄漏)
  - CombinatorialPurgedCV (CPCV): 组合式 purged CV, 产出多条 OOS 路径
  - DSR / PBO: 多次试验偏差的校正与过拟合概率 (P2B-T03)

参考开源实现: github.com/sam31415/timeseriescv
"""
from __future__ import annotations

from validation.purged_cv import PurgedKFold
from validation.cpcv import CombinatorialPurgedCV
from validation.dsr import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    summarise_cpcv_paths,
)

__all__ = [
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "summarise_cpcv_paths",
]

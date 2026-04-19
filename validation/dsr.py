"""
Deflated Sharpe Ratio (DSR) + Probability of Backtest Overfitting (PBO) — v1.0 (P2B-T03)

Reference:
  - López de Prado, M. (2018) "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality"
  - Bailey, Borwein, López de Prado, Zhu (2015)
    "The Probability of Backtest Overfitting"

Two concepts
------------
1) Deflated Sharpe Ratio (DSR):
   把 "观察到的 Sharpe" 按 "试过多少次 + 样本量 + 偏度 + 峰度" 做折扣,
   扣除 multiple-testing 带来的选择偏差。
   输出为 "观察到的 Sharpe 高于真实零值的概率" ∈ [0, 1]。
   常用阈值 DSR > 0.95 视为统计显著。

2) Probability of Backtest Overfitting (PBO):
   输入 N 条 OOS path 的收益矩阵, 输出 "OOS 最佳策略在 IS 上表现低于中位数的概率"。
   PBO ∈ [0, 1], 越小越好; 路线图 G1 要求 < 30%。

API
---
    dsr = deflated_sharpe_ratio(observed_sr, n_trials, n_samples, skew, kurt)
    # 0.0 - 1.0, 越大越好

    pbo = probability_of_backtest_overfitting(returns_matrix)
    # 0.0 - 1.0, 越小越好
    # returns_matrix: shape (n_samples, n_strategies) 或 (n_paths, n_strategies)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------
# Deflated Sharpe Ratio
# ----------------------------------------------------------------------
def _expected_max_sr(n_trials: int, em_gamma: float = 0.5772156649) -> float:
    """
    期望最大 Sharpe 的估计 (López de Prado 2018 公式 4):

        E[max SR] ≈ (1 - γ) · Φ⁻¹(1 - 1/N) + γ · Φ⁻¹(1 - 1/(N·e))

    γ = Euler-Mascheroni 常量 ≈ 0.5772
    Φ⁻¹: 标准正态累积的逆

    Args:
        n_trials: 试验次数 N
        em_gamma: Euler-Mascheroni 常量, 默认 0.5772156649

    Returns:
        E[max SR] 的无量纲估计 (需要乘上年化 SR 标准差才是绝对量)
    """
    if n_trials < 1:
        raise ValueError(f"n_trials 必须 ≥ 1, 收到 {n_trials}")
    if n_trials == 1:
        return 0.0
    e = np.e
    inv_norm_1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    inv_norm_2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return (1.0 - em_gamma) * inv_norm_1 + em_gamma * inv_norm_2


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_samples: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    benchmark_sr: float = 0.0,
) -> float:
    """
    Deflated Sharpe Ratio — López de Prado 2018 公式 9.

    返回 "观察到的 Sharpe 显著高于 benchmark (扣除多次试验偏差后)" 的概率。

    Args:
        observed_sr: 观察到的 (年化) Sharpe Ratio (这里单位任意, 只需与 benchmark 同单位)
        n_trials: 试验过多少个策略 / 配置组合才挑到这个 SR
        n_samples: 收益样本数 (日收益序列长度 T)
        skew: 收益偏度 (默认 0 = 正态)
        kurt: 收益峰度 (默认 3 = 正态); 注意是 Pearson kurt 而非 excess kurt
        benchmark_sr: 基准 SR (默认 0 = "无 edge")

    Returns:
        prob ∈ [0, 1] — Z-score 转出的右尾概率; 常用阈值 > 0.95 视为显著

    公式:
        SR_0 = benchmark_sr + E[max_SR_null] / sqrt(T)  (此处需要选出 SR 分布的方差项,
               这里简化处理: 我们把 "multiple-testing 噪声" 统一用 expected_max_sr × σ_SR)

        Var[SR_hat] = (1 - skew · SR + ((kurt - 1) / 4) · SR²) / (T - 1)

        Z = (SR_hat - SR_0) / sqrt(Var[SR_hat])

        DSR = Φ(Z)
    """
    if n_samples < 2:
        raise ValueError(f"n_samples 必须 ≥ 2, 收到 {n_samples}")
    if n_trials < 1:
        raise ValueError(f"n_trials 必须 ≥ 1, 收到 {n_trials}")

    sr = float(observed_sr)
    T = int(n_samples)
    N = int(n_trials)

    # Var[SR_hat] — López de Prado 2018 公式 6
    # 把 kurt 理解为 Pearson kurt (正态 = 3)
    var_sr = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)) / (T - 1)
    var_sr = max(var_sr, 1e-12)  # 数值保护
    sigma_sr = np.sqrt(var_sr)

    # 多次试验偏差: 用 E[max_SR] 作为 SR 零分布下的"自然最大值"估计。
    # López de Prado 2018 公式 8 将其作为 deflated benchmark 的一部分:
    #   SR_0 = benchmark_sr + sigma_sr * E[max_SR](n_trials)
    sr_zero = float(benchmark_sr) + sigma_sr * _expected_max_sr(N)

    z = (sr - sr_zero) / sigma_sr
    return float(stats.norm.cdf(z))


# ----------------------------------------------------------------------
# Probability of Backtest Overfitting (PBO)
# ----------------------------------------------------------------------
def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray | pd.DataFrame,
    n_splits: int = 16,
    metric: str = "sharpe",
    random_state: Optional[int] = None,
) -> float:
    """
    CSCV (Combinatorially Symmetric Cross-Validation) 估计 PBO
    — Bailey et al. 2015, §4。

    输入 N 条 path × M 个策略配置的收益矩阵, 输出 PBO ∈ [0, 1]:

        PBO = P( rank_OOS(argmax_IS(SR)) ≤ M/2 )

    高 PBO 意味着 "在 IS 看起来最好的策略在 OOS 排名不如中位数", 即典型过拟合。

    算法:
        1. 把 N 行 (时间) 均匀切成 n_splits 个块
        2. 枚举 C(n_splits, n_splits/2) 种 "一半是 IS, 一半是 OOS" 的组合
           (对称 CV: IS 与 OOS 角色对称)
        3. 对每种组合:
              - 在 IS 上算每个策略的 metric, 找最佳策略 n*
              - 在 OOS 上算每个策略的 metric, 记 n* 的 rank
              - λ = logit(rank_pct), 其中 rank_pct = rank / (M + 1)
        4. PBO = P(λ < 0) = 正态分布估计或经验计数

    Args:
        returns_matrix: shape (n_rows, n_strategies); 行 = 时间点, 列 = 策略
        n_splits: 时间轴切块数, 必须为偶数
        metric: 当前只支持 'sharpe'
        random_state: 保留位 (当前未用)

    Returns:
        PBO ∈ [0, 1]
    """
    if isinstance(returns_matrix, pd.DataFrame):
        R = returns_matrix.values.astype(float)
        m_names = list(returns_matrix.columns)
    else:
        R = np.asarray(returns_matrix, dtype=float)
        m_names = list(range(R.shape[1]))

    if R.ndim != 2:
        raise ValueError(f"returns_matrix 需为 2D, 收到 shape {R.shape}")
    T, M = R.shape
    if M < 2:
        raise ValueError(f"至少需要 2 个策略配置, 收到 {M}")
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits 必须为偶数, 收到 {n_splits}")
    if T < n_splits:
        raise ValueError(f"样本数 {T} < n_splits {n_splits}")

    # 切成 n_splits 个块
    block_bounds = []
    base = T // n_splits
    rem = T - base * n_splits
    start = 0
    for i in range(n_splits):
        sz = base + (1 if i < rem else 0)
        block_bounds.append((start, start + sz))
        start += sz

    # 枚举 C(n_splits, n_splits/2) 种 IS / OOS 分法
    from itertools import combinations

    logits: list[float] = []
    half = n_splits // 2
    for is_blocks in combinations(range(n_splits), half):
        is_set = set(is_blocks)
        is_mask = np.zeros(T, dtype=bool)
        oos_mask = np.zeros(T, dtype=bool)
        for b_idx, (s, e) in enumerate(block_bounds):
            if b_idx in is_set:
                is_mask[s:e] = True
            else:
                oos_mask[s:e] = True

        R_is = R[is_mask, :]
        R_oos = R[oos_mask, :]

        if metric == "sharpe":
            is_metric = _sharpe_columnwise(R_is)
            oos_metric = _sharpe_columnwise(R_oos)
        else:
            raise ValueError(f"不支持的 metric: {metric}")

        # 找 IS 最佳策略
        if np.all(np.isnan(is_metric)):
            continue
        best_is = int(np.nanargmax(is_metric))

        # OOS 上 best_is 的相对排名 (0 = 最差, M-1 = 最好)
        # 处理 NaN: 把 NaN 视为最差
        oos_vals = np.where(np.isnan(oos_metric), -np.inf, oos_metric)
        oos_rank = int(np.sum(oos_vals < oos_vals[best_is]))  # 0 ~ M-1

        # 相对排名 (0, 1)
        rank_pct = (oos_rank + 1.0) / (M + 1.0)

        # logit; 把 rank_pct 映射到 (-inf, inf), 中位 = 0
        rank_pct = float(np.clip(rank_pct, 1e-6, 1.0 - 1e-6))
        lam = np.log(rank_pct / (1.0 - rank_pct))
        logits.append(lam)

    if not logits:
        return float("nan")
    arr = np.asarray(logits)
    # PBO = 经验概率 P(λ < 0) = IS 最佳在 OOS 排名 < 中位数的比例
    return float(np.mean(arr < 0.0))


def _sharpe_columnwise(R: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    对每列收益计算 (非年化) Sharpe = mean / std。样本不足或 sigma 接近 0 返回 NaN。

    使用 np.divide(..., where=valid) 避免 "invalid value encountered in divide" 警告,
    本地 agent 反馈后加固 (2026-04-19)。
    """
    if R.shape[0] < 2:
        return np.full(R.shape[1], np.nan)
    mu = np.nanmean(R, axis=0)
    sigma = np.nanstd(R, axis=0, ddof=1)
    valid = np.isfinite(sigma) & (sigma > eps)
    sharpe = np.full_like(mu, np.nan, dtype=float)
    np.divide(mu, sigma, out=sharpe, where=valid)
    # 把 invalid 位置显式置 NaN (np.divide with out+where 对未写入位置保留原值, 这里是 NaN)
    sharpe = np.where(valid, sharpe, np.nan)
    return sharpe


# ----------------------------------------------------------------------
# 便捷入口: 从 CPCV backtest_paths 的输出直接算
# ----------------------------------------------------------------------
def summarise_cpcv_paths(
    curves: list[pd.Series],
    n_trials: int = 1,
    bars_per_year: float = 252.0,
    deterministic_eps: float = 1e-9,
) -> dict:
    """
    接收 CombinatorialPurgedCV.backtest_paths 返回的曲线列表,
    输出常用判决字段的 dict。

    Args:
        curves: List[pd.Series], 每条为 test 期收益 (per-bar return)
        n_trials: 做 DSR 所需的多次试验次数 (默认 1 = 不打折; 生产中应填真实扫描次数)
        bars_per_year: 年化换算因子 (日线 252 / 4h bar 1460 / 8h bar 1095). 默认 252。
        deterministic_eps: path 间 Sharpe std 小于此阈值时标记为 deterministic, PBO 设 NaN

    Returns:
        {
            'n_paths': int,
            'sharpe_per_path': List[float],
            'sharpe_median': float,
            'sharpe_mean': float,
            'sharpe_std': float,
            'dsr_median': float,           # 中位 Sharpe 对应的 DSR
            'verdict': 'PASS' | 'CONDITIONAL' | 'FAIL',
            'is_deterministic': bool,      # 若 True, 所有 path 输出相同 — CPCV 不提供多样性信息
            'bars_per_year': float,        # 回显, 便于外部验证
        }

    Note on is_deterministic:
      CPCV 的价值在于让策略在不同 train 窗口产生不同 OOS 结果 (ML 模型 / 参数拟合).
      若 strategy_fn 不依赖 train_idx (纯规则策略), 所有 path 的 OOS 收益完全相同,
      sharpe_std ≈ 0, 对这种策略 PBO 退化为"tie 的排名赌博", 不具统计意义。
      本函数会将 is_deterministic=True 的情况下 DSR 仍然计算 (基于单条 OOS 曲线),
      但上游代码应据此把 PBO 视为 N/A。
    """
    if not curves:
        return {
            "n_paths": 0,
            "sharpe_per_path": [],
            "sharpe_median": float("nan"),
            "sharpe_mean": float("nan"),
            "sharpe_std": float("nan"),
            "dsr_median": float("nan"),
            "verdict": "FAIL",
            "is_deterministic": False,
            "bars_per_year": bars_per_year,
        }

    ann_factor = float(np.sqrt(bars_per_year))
    sharpes = []
    for c in curves:
        arr = c.values
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            sharpes.append(float("nan"))
            continue
        mu, sd = np.mean(arr), np.std(arr, ddof=1)
        if sd <= 0:
            sharpes.append(float("nan"))
            continue
        sharpes.append((mu / sd) * ann_factor)

    sharpes_arr = np.asarray(sharpes, dtype=float)
    finite = sharpes_arr[np.isfinite(sharpes_arr)]
    sr_median = float(np.median(finite)) if len(finite) else float("nan")
    sr_mean = float(np.mean(finite)) if len(finite) else float("nan")
    sr_std = float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan")

    # 退化检测: 所有 path 的 Sharpe 完全相同 (方差接近 0)
    # 意味着 strategy_fn 对 train_idx 无依赖 -> PBO 不具统计意义
    is_deterministic = bool(
        len(finite) >= 2
        and np.isfinite(sr_std)
        and sr_std < deterministic_eps
    )

    # DSR: 用最长曲线的样本数
    n_samples = max((len(c) for c in curves), default=2)
    if np.isfinite(sr_median) and n_samples >= 2:
        # 从所有 path 收益取偏度峰度 (或 deterministic 情况下只看 path 0)
        if is_deterministic:
            src = curves[0].values
        else:
            src = np.concatenate([c.values for c in curves])
        src = src[np.isfinite(src)]
        if len(src) >= 4:
            skew = float(stats.skew(src))
            kurt = float(stats.kurtosis(src, fisher=False))  # Pearson
        else:
            skew, kurt = 0.0, 3.0
        dsr_median = deflated_sharpe_ratio(
            observed_sr=sr_median,
            n_trials=n_trials,
            n_samples=n_samples,
            skew=skew,
            kurt=kurt,
        )
    else:
        dsr_median = float("nan")

    # 判决 (路线图 G1 标准)
    if np.isfinite(sr_median) and sr_median > 1.0 and np.isfinite(dsr_median) and dsr_median > 0.8:
        verdict = "PASS"
    elif np.isfinite(sr_median) and sr_median > 0.5:
        verdict = "CONDITIONAL"
    else:
        verdict = "FAIL"

    return {
        "n_paths": len(curves),
        "sharpe_per_path": sharpes,
        "sharpe_median": sr_median,
        "sharpe_mean": sr_mean,
        "sharpe_std": sr_std,
        "dsr_median": dsr_median,
        "verdict": verdict,
        "is_deterministic": is_deterministic,
        "bars_per_year": bars_per_year,
    }

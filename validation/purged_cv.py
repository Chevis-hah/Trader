"""
Purged K-Fold Cross-Validation — v1.0 (P2B-T01)

实现 López de Prado 2018《Advances in Financial Machine Learning》第 7 章的
Purged K-Fold CV, 解决金融时间序列 CV 中两类常见泄漏:

  1. Purging — 测试集的标签若与训练集样本在时间上重叠 (例如 forward return
     横跨多个时间点), 必须从训练集移除这些样本。本类通过 `sample_times`
     (每个样本的标签生效区间) 显式支持; 若未提供则默认样本标签区间为 [t, t]。

  2. Embargoing — 测试集结束后的一小段时间 (embargo_pct * N 个样本) 也要排除,
     以避免"信息从测试集泄漏到训练集"的延迟效应。

与 sklearn KFold 差异:
  - 必须按时间顺序切 (shuffle=False)
  - 每 fold 的 train 是 "除去 test ∪ embargo ∪ purged" 的剩余
  - 当 sample_times 为 None 时, purging 等价于 "按测试集边界的 embargo 对称处理"

API 与 sklearn 兼容: `for train_idx, test_idx in PurgedKFold(...).split(X): ...`

Reference:
  - López de Prado, M. (2018) *Advances in Financial Machine Learning*, Ch.7
  - github.com/sam31415/timeseriescv
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pandas as pd


class PurgedKFold:
    """
    Purged K-Fold CV with embargo.

    Args:
        n_splits: 折数 K (≥ 2)
        embargo_pct: 测试集结束后 embargo 的样本占比 (例如 0.01 = 1%)。
                     全长 N 下, embargo 长度 = floor(embargo_pct * N)。
        sample_times: 可选; Series(index = X.index, values = label_end_time)。
                     表示每个样本的 "label 区间结束时间"; 训练样本若其
                     [t, sample_times[t]] 与测试集的 [min, max] 区间重叠,
                     将被 purge。若为 None, 仅按 "整数索引邻接" 做 embargo。

    使用例:
        >>> cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        >>> for train_idx, test_idx in cv.split(X):
        ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        sample_times: Optional[pd.Series] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits 必须 ≥ 2, 收到 {n_splits}")
        if not (0.0 <= embargo_pct < 1.0):
            raise ValueError(f"embargo_pct 必须在 [0, 1), 收到 {embargo_pct}")
        self.n_splits = int(n_splits)
        self.embargo_pct = float(embargo_pct)
        self.sample_times = sample_times

    # ------------------------------------------------------------------
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    # ------------------------------------------------------------------
    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        产出 (train_idx, test_idx) 对, 共 n_splits 个。

        Args:
            X: DataFrame / ndarray / 有 __len__ 的序列 (按时间顺序)

        Yields:
            (train_indices, test_indices) — 均为 numpy.int64 数组, 指 X 的 iloc 位置
        """
        n = len(X)
        if n < self.n_splits:
            raise ValueError(f"样本数 {n} 少于 n_splits {self.n_splits}")

        indices = np.arange(n)
        embargo_size = int(np.floor(self.embargo_pct * n))

        # 将索引平分成 K 块 (最后一块吸收余数)
        fold_bounds = self._compute_fold_bounds(n, self.n_splits)

        # 若提供 sample_times, 对齐 X.index
        sample_times_arr = self._align_sample_times(X)

        for (test_start, test_end) in fold_bounds:
            test_idx = indices[test_start:test_end]

            # 基础 train = 除 test 外的全部
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:test_end] = False

            # Embargo: 测试集结束后的 embargo_size 样本不进训练集
            if embargo_size > 0:
                emb_start = test_end
                emb_end = min(n, test_end + embargo_size)
                train_mask[emb_start:emb_end] = False

            # Purging: 若 sample_times 提供, 从训练集移除 label 区间跨入测试期的样本
            if sample_times_arr is not None and len(test_idx) > 0:
                test_time_start = sample_times_arr["start"][test_start]
                test_time_end = sample_times_arr["end"][test_end - 1]
                # 训练样本 i 的标签区间 [start_i, end_i], 若与 test 区间重叠则 purge
                starts = sample_times_arr["start"]
                ends = sample_times_arr["end"]
                overlap = (ends >= test_time_start) & (starts <= test_time_end)
                train_mask &= ~overlap
                # 再把 test 本身排除 (overlap 会把 test 自身也标为 True)
                train_mask[test_start:test_end] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_fold_bounds(n: int, k: int) -> list[tuple[int, int]]:
        """
        把 [0, n) 分成 k 个大致等长的连续块。余数分给最后若干块 (更均匀)。
        """
        base = n // k
        rem = n - base * k
        bounds = []
        start = 0
        for i in range(k):
            sz = base + (1 if i < rem else 0)
            bounds.append((start, start + sz))
            start += sz
        return bounds

    # ------------------------------------------------------------------
    def _align_sample_times(self, X) -> Optional[dict]:
        """
        将 self.sample_times 对齐到 X 的整数位置, 返回 {'start': np.ndarray, 'end': np.ndarray}。
        未提供则返回 None。
        """
        if self.sample_times is None:
            return None
        st = self.sample_times
        if isinstance(X, (pd.DataFrame, pd.Series)):
            if not st.index.equals(X.index):
                try:
                    st = st.reindex(X.index)
                except Exception as e:
                    raise ValueError(
                        f"sample_times 与 X 的 index 无法对齐: {e}"
                    ) from e
            start_arr = np.asarray(st.index.values)
            end_arr = np.asarray(st.values)
        else:
            start_arr = np.arange(len(X))
            end_arr = np.asarray(st)
        if len(end_arr) != len(X):
            raise ValueError(
                f"sample_times 长度 {len(end_arr)} 与 X 长度 {len(X)} 不符"
            )
        return {"start": start_arr, "end": end_arr}

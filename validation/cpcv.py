"""
Combinatorial Purged Cross-Validation (CPCV) — v1.0 (P2B-T02)

López de Prado 2018 第 12 章 + Arian-Norouzi-Seco 2024 的关键构造:
把时间轴切为 N 个 group, 每次挑 k 个 group 作为 test, 产生 C(N, k) 个 splits。
每个 group 在全部 splits 中作为 test 出现 `C(N-1, k-1)` 次, 这恰好也是我们能
构造出的 OOS 路径数 (path 数 = C(N-1, k-1) = k * C(N,k) / N)。

典型: N=10, k=2 → 45 splits, 9 paths。

路径 (path) 的含义
--------------------
单次 walk-forward 只产生 1 条 OOS 曲线, 这条曲线很容易被 "运气" 支配。
CPCV 通过枚举组合, 对每个 group 给出多个独立的 OOS 估计, 拼接成 `n_paths`
条完整 OOS 曲线, 让我们看到 Sharpe 分布 (不是单点), 计算 PBO / DSR 等。

Path 构造算法:
  - 对每个 group g, 收集所有 "g 被用作 test" 的 split 索引, 共 n_paths 个。
  - 按 split 枚举顺序给它们编号 0..n_paths-1 (同一 group 内)。
  - 第 j 条 path: 对每个 group g, 取它在 "被 test" 列表中的第 j 个 split 的结果。

这保证:
  - 每条 path 覆盖整个时间轴 (N 个 group 都出现恰好一次);
  - 每个 split 在 n_paths 条 path 中至少被用一次, 至多 k 次 (因为它覆盖 k 个 group);
  - 不同 path 之间的 OOS 估计独立 (使用不同 split)。

接口
----
    cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2, embargo_pct=0.01)
    splits = list(cv.split(X))
    # splits[i] = (train_idx, test_idx, test_group_tuple)
    paths = cv.get_paths()  # list[list[int]], paths[j] = [split_idx for group 0..N-1]
    # 或, 一步到位:
    eq_curves = cv.backtest_paths(X, strategy_fn)

Reference:
  - López de Prado, M. (2018), *Advances in Financial ML*, Ch.12
  - Arian, Norouzi, Seco (2024), "The Combinatorial Purged CV Method"
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


class CombinatorialPurgedCV:
    """
    Combinatorial Purged K-Fold CV.

    Args:
        n_groups: 时间轴切成的 group 数 (类似 K-Fold 的 K)
        n_test_groups: 每次选多少个 group 作为 test (k)
        embargo_pct: 测试集两侧的 embargo 占比
        sample_times: 可选, 标签结束时间 Series (同 PurgedKFold)
    """

    def __init__(
        self,
        n_groups: int = 10,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        sample_times: Optional[pd.Series] = None,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"n_groups 必须 ≥ 2, 收到 {n_groups}")
        if n_test_groups < 1 or n_test_groups >= n_groups:
            raise ValueError(
                f"n_test_groups 必须在 [1, n_groups), 收到 {n_test_groups}/{n_groups}"
            )
        if not (0.0 <= embargo_pct < 1.0):
            raise ValueError(f"embargo_pct 必须在 [0, 1), 收到 {embargo_pct}")
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.embargo_pct = float(embargo_pct)
        self.sample_times = sample_times

        # 懒计算缓存
        self._last_n: Optional[int] = None
        self._last_group_bounds: Optional[List[Tuple[int, int]]] = None
        self._last_splits: Optional[List[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]]] = None
        self._last_paths: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    # 组合学基本量
    # ------------------------------------------------------------------
    @property
    def n_splits(self) -> int:
        """C(N, k) = split 数"""
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """C(N-1, k-1) = k * C(N,k) / N = 可构造的 OOS path 数"""
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    # ------------------------------------------------------------------
    # Group 划分
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_group_bounds(n: int, n_groups: int) -> List[Tuple[int, int]]:
        """把 [0, n) 切成 n_groups 个连续块 (余数分给前若干块)。"""
        if n < n_groups:
            raise ValueError(f"样本数 {n} 少于 n_groups {n_groups}")
        base = n // n_groups
        rem = n - base * n_groups
        bounds = []
        start = 0
        for i in range(n_groups):
            sz = base + (1 if i < rem else 0)
            bounds.append((start, start + sz))
            start += sz
        return bounds

    # ------------------------------------------------------------------
    # Split 枚举
    # ------------------------------------------------------------------
    def split(
        self, X, y=None, groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]]:
        """
        枚举所有 C(N, k) 个 (train, test, test_group_tuple) 组合。

        Yields:
            (train_idx, test_idx, test_groups)
              - train_idx, test_idx: numpy.int64 数组, iloc 位置
              - test_groups: 用作 test 的 group 下标 tuple, 长度 k, 升序
        """
        splits = self._build_splits(X)
        for item in splits:
            yield item

    # ------------------------------------------------------------------
    def _build_splits(
        self, X,
    ) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]]:
        n = len(X)
        if self._last_splits is not None and self._last_n == n:
            return self._last_splits

        group_bounds = self._compute_group_bounds(n, self.n_groups)
        embargo_size = int(np.floor(self.embargo_pct * n))
        sample_times_arr = self._align_sample_times(X, n)

        indices = np.arange(n)
        splits: List[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]] = []

        for test_groups in combinations(range(self.n_groups), self.n_test_groups):
            # 测试集 = 选中 groups 的并集
            test_mask = np.zeros(n, dtype=bool)
            for g in test_groups:
                s, e = group_bounds[g]
                test_mask[s:e] = True

            # 训练集 = 全部 \ 测试集 \ embargo \ purged
            train_mask = ~test_mask

            # embargo: 每个选中 group 的右边界后 embargo_size 样本排除
            if embargo_size > 0:
                for g in test_groups:
                    _, e = group_bounds[g]
                    emb_end = min(n, e + embargo_size)
                    train_mask[e:emb_end] = False

            # purging: 若提供 sample_times, 移除 label 跨入 test 的训练样本
            if sample_times_arr is not None:
                # 对每个选中 group 分别检查重叠
                for g in test_groups:
                    s, e = group_bounds[g]
                    test_time_start = sample_times_arr["start"][s]
                    test_time_end = sample_times_arr["end"][e - 1]
                    overlap = (
                        (sample_times_arr["end"] >= test_time_start)
                        & (sample_times_arr["start"] <= test_time_end)
                    )
                    train_mask &= ~overlap
                train_mask &= ~test_mask  # 再保证测试集不入训练

            splits.append((
                indices[train_mask].copy(),
                indices[test_mask].copy(),
                tuple(test_groups),
            ))

        self._last_n = n
        self._last_group_bounds = group_bounds
        self._last_splits = splits
        return splits

    # ------------------------------------------------------------------
    # Path 构造
    # ------------------------------------------------------------------
    def get_paths(self, X=None) -> List[List[int]]:
        """
        产出 n_paths 条 OOS path, 每条是 [split_idx for each group 0..N-1]。

        算法: 对每个 group g, 收集 "g 在 test 里" 的 split 索引共 n_paths 个;
              按枚举顺序给它们编号 0..n_paths-1 (同一 group 内独立计数);
              path j 的 group g 指 "第 j 次遇到含 g 的 split"。

        Note: 不依赖 X, 只依赖 (n_groups, n_test_groups)。X 仅在需要真实 n 校验时用。
        """
        if self._last_paths is not None and (X is None or self._last_n == len(X)):
            return self._last_paths

        n_paths = self.n_paths
        # For each group, list of split indices in which it is a test group.
        # Because we enumerate in itertools.combinations order, this is deterministic.
        per_group_splits: List[List[int]] = [[] for _ in range(self.n_groups)]
        for split_idx, test_groups in enumerate(
            combinations(range(self.n_groups), self.n_test_groups)
        ):
            for g in test_groups:
                per_group_splits[g].append(split_idx)

        # Sanity: each group has exactly n_paths entries
        for g, lst in enumerate(per_group_splits):
            assert len(lst) == n_paths, (
                f"group {g} has {len(lst)} occurrences, expected {n_paths}"
            )

        # Path j, group g -> per_group_splits[g][j]
        paths: List[List[int]] = []
        for j in range(n_paths):
            paths.append([per_group_splits[g][j] for g in range(self.n_groups)])

        self._last_paths = paths
        return paths

    # ------------------------------------------------------------------
    # 一键跑完
    # ------------------------------------------------------------------
    def backtest_paths(
        self,
        X,
        strategy_fn: Callable[[np.ndarray, np.ndarray], pd.Series],
    ) -> List[pd.Series]:
        """
        对每条 OOS path 跑一遍策略, 返回 n_paths 条权益曲线 / 收益序列。

        Args:
            X: 输入数据 (DataFrame / ndarray), 只决定长度与 index
            strategy_fn: (train_idx, test_idx) -> pd.Series, 返回 test 期的收益 / 权益
                         返回 Series 的 index 应为 X.iloc[test_idx] 的 index (或整数下标)

        Returns:
            list[pd.Series], 长度 = n_paths; 每条 Series 按时间顺序拼接了该 path 在
            所有 N 个 group 上的 OOS 结果, index 覆盖全 X。
        """
        splits = self._build_splits(X)
        paths = self.get_paths(X)

        # 先跑所有 splits, 按 group 保存 test 期的每个 group 的片段
        # split_results[split_idx] -> pd.Series on test_idx
        split_results: List[Optional[pd.Series]] = [None] * len(splits)
        for s_idx, (train_idx, test_idx, _) in enumerate(splits):
            out = strategy_fn(train_idx, test_idx)
            if not isinstance(out, pd.Series):
                raise TypeError(
                    f"strategy_fn 返回类型 {type(out).__name__}, 需要 pd.Series"
                )
            split_results[s_idx] = out

        # 把 strategy output 切到 group 粒度:
        # 对 path j, group g: 取 split_results[ paths[j][g] ] 中属于 group g 区间的那段
        group_bounds = self._last_group_bounds
        assert group_bounds is not None
        # 用 iloc 位置作为锚点
        curves: List[pd.Series] = []
        index = X.index if hasattr(X, "index") else pd.RangeIndex(len(X))

        for j, path in enumerate(paths):
            pieces: List[pd.Series] = []
            for g in range(self.n_groups):
                s_idx = path[g]
                gstart, gend = group_bounds[g]
                # Slice strategy output at group-g positions
                piece_index = index[gstart:gend]
                sr = split_results[s_idx]
                # strategy_fn 可能返回 Series index = integer iloc 或 X.index
                # 统一按 iloc 对齐
                try:
                    piece = sr.loc[piece_index]
                except KeyError:
                    # 退化: 尝试 integer-based
                    piece = sr.iloc[
                        [sr.index.get_loc(idx) for idx in piece_index if idx in sr.index]
                    ]
                pieces.append(piece)
            curves.append(pd.concat(pieces))

        return curves

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _align_sample_times(self, X, n: int) -> Optional[dict]:
        if self.sample_times is None:
            return None
        st = self.sample_times
        if isinstance(X, (pd.DataFrame, pd.Series)):
            if not st.index.equals(X.index):
                st = st.reindex(X.index)
            start_arr = np.asarray(st.index.values)
            end_arr = np.asarray(st.values)
        else:
            start_arr = np.arange(n)
            end_arr = np.asarray(st)
        if len(end_arr) != n:
            raise ValueError(
                f"sample_times 长度 {len(end_arr)} 与 X 长度 {n} 不符"
            )
        return {"start": start_arr, "end": end_arr}

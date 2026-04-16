"""
Universe 构建器 — 路线 A (横截面因子模型) 基础设施

核心功能:
  1. 根据市值排序构建 Top N 币种 universe
  2. 滚动更新 (point-in-time, 避免 survivorship bias)
  3. 流动性过滤 (最小成交量阈值)
  4. Turnover smoothing (减少频繁换入换出)

数据要求:
  - OHLCV 数据表 (来自 data/storage.py)
  - 可选: 市值数据表 (若无，退化为 volume-based 排序)

使用示例:
  >>> builder = UniverseBuilder(storage, top_n=50)
  >>> universe_at_t = builder.get_universe('2023-06-01')
  >>> # → ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', ...] 共 50 个
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("universe")


@dataclass
class UniverseConfig:
    """Universe 构建参数"""
    top_n: int = 50                           # 选取 Top N 币种
    min_volume_30d_usd: float = 5_000_000     # 30 日日均成交额下限 (USD)
    min_history_days: int = 180               # 至少有 N 日历史数据
    rebalance_freq_days: int = 7              # Universe 每 N 天重新排序
    turnover_smoothing: int = 2               # 候选币种需连续 N 个 rebalance 入围才纳入


class UniverseBuilder:
    """
    Point-in-time universe 构建器

    关键原则:
      - 严禁用 "今天的 Top 50" 回测历史 (survivorship bias)
      - 必须用 "当时的 Top 50" 才有统计效度
    """

    def __init__(self, storage, config: Optional[UniverseConfig] = None):
        """
        Args:
            storage: data.storage.Storage 实例，提供 get_klines() 接口
            config: UniverseConfig
        """
        self.storage = storage
        self.cfg = config or UniverseConfig()
        self._cache: dict[str, list[str]] = {}        # date_str → universe list
        self._candidate_history: dict[str, int] = {}  # symbol → 连续入围次数

    # ----------------------------------------------------------------
    # 对外主接口
    # ----------------------------------------------------------------
    def get_universe(self, as_of_date: str) -> list[str]:
        """
        返回在指定日期应该交易的币种列表

        Args:
            as_of_date: 'YYYY-MM-DD' 格式

        Returns:
            按市值或成交量从大到小排序的 symbol 列表
        """
        if as_of_date in self._cache:
            return self._cache[as_of_date]

        universe = self._build_universe(as_of_date)
        self._cache[as_of_date] = universe
        return universe

    def get_all_symbols(self) -> list[str]:
        """
        返回 storage 中所有可用 symbols (辅助方法)
        """
        if hasattr(self.storage, "get_symbols"):
            return self.storage.get_symbols()

        # Fallback: 硬编码 Top 60 by mcap (2025-2026)
        # 实际使用时应该从 storage 动态读取
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT",
            "LINKUSDT", "MATICUSDT", "LTCUSDT", "BCHUSDT", "UNIUSDT",
            "ATOMUSDT", "ETCUSDT", "XLMUSDT", "FILUSDT", "NEARUSDT",
            "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT",
            "TIAUSDT", "SEIUSDT", "AAVEUSDT", "MKRUSDT", "GRTUSDT",
            "SANDUSDT", "MANAUSDT", "AXSUSDT", "FLOWUSDT", "ALGOUSDT",
            "HBARUSDT", "VETUSDT", "EOSUSDT", "XTZUSDT", "EGLDUSDT",
            "ICPUSDT", "FTMUSDT", "ROSEUSDT", "THETAUSDT", "KAVAUSDT",
            "CRVUSDT", "SNXUSDT", "COMPUSDT", "CHZUSDT", "RUNEUSDT",
            "IMXUSDT", "FETUSDT", "RNDRUSDT", "GALAUSDT", "LDOUSDT",
            "PYTHUSDT", "JUPUSDT", "WIFUSDT", "PEPEUSDT", "ENAUSDT",
        ]

    # ----------------------------------------------------------------
    # 内部逻辑
    # ----------------------------------------------------------------
    def _build_universe(self, as_of_date: str) -> list[str]:
        """
        构建 point-in-time universe

        排序依据: 过去 30 日平均日成交额 (USD)
          - 这是市值的 proxy (更易从 OHLCV 数据估算)
          - 避免对外部市值数据源的依赖
        """
        all_symbols = self.get_all_symbols()
        as_of_ts = pd.Timestamp(as_of_date)

        ranked: list[tuple[str, float]] = []

        for symbol in all_symbols:
            try:
                klines = self.storage.get_klines(symbol, "1d", limit=50)
                if klines.empty or len(klines) < self.cfg.min_history_days:
                    continue

                # 只用 as_of_date 之前的数据 (避免 lookahead)
                klines["ts"] = pd.to_datetime(klines["open_time"], unit="ms")
                klines = klines[klines["ts"] <= as_of_ts]

                if len(klines) < 30:
                    continue

                last_30 = klines.tail(30)
                # 日成交额 = close * volume
                if "close" not in last_30.columns or "volume" not in last_30.columns:
                    continue

                daily_turnover_usd = (last_30["close"] * last_30["volume"]).mean()
                if daily_turnover_usd < self.cfg.min_volume_30d_usd:
                    continue

                ranked.append((symbol, daily_turnover_usd))

            except Exception as e:
                logger.debug(f"skip {symbol}: {e}")
                continue

        # 按成交额降序
        ranked.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [s for s, _ in ranked[: self.cfg.top_n * 2]]

        # Turnover smoothing: 连续 N 次入围才纳入
        universe = self._apply_smoothing(top_candidates)

        return universe[: self.cfg.top_n]

    def _apply_smoothing(self, candidates: list[str]) -> list[str]:
        """避免因临时流动性尖峰 (如 memecoin) 污染 universe"""
        smoothing = self.cfg.turnover_smoothing
        result = []

        for symbol in candidates:
            self._candidate_history[symbol] = (
                self._candidate_history.get(symbol, 0) + 1
            )

        # 清理不再入围的币种记录
        candidate_set = set(candidates)
        for symbol in list(self._candidate_history.keys()):
            if symbol not in candidate_set:
                # 衰减而非立刻归零，避免抖动
                self._candidate_history[symbol] -= 1
                if self._candidate_history[symbol] <= 0:
                    del self._candidate_history[symbol]

        # 连续入围次数 >= smoothing 的币种进入 universe
        for symbol in candidates:
            if self._candidate_history.get(symbol, 0) >= smoothing:
                result.append(symbol)

        # 如果 smoothing 过滤掉太多，回退到原始列表
        if len(result) < self.cfg.top_n * 0.5:
            return candidates

        return result

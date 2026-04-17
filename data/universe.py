"""
Universe 构建器 — 路线 A (横截面因子模型) 基础设施

核心功能:
  1. 根据市值排序构建 Top N 币种 universe
  2. 滚动更新 (point-in-time, 避免 survivorship bias)
  3. 流动性过滤 (最小成交量阈值)
  4. Turnover smoothing (减少频繁换入换出)

数据要求:
  - OHLCV 数据表 (来自 data/storage.py)
  - 可选: 市值数据表 (若无, 退化为 volume-based 排序)

v2.3.1 修复:
  - _build_universe 不再用 `limit=50` 截断历史数据, 改为 start_time 过滤到 as_of_date
    (原实现与 min_history_days=180 直接冲突, 导致任何 as_of_date 都拿不到足够窗口)
  - get_all_symbols 优先从 storage 读已入库的 symbol 列表, 不再走硬编码 fallback
  - 缓存 key 增加 config 指纹, 避免同一 builder 切参数时命中过期缓存

使用示例:
  >>> builder = UniverseBuilder(storage, UniverseConfig(top_n=50))
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
    # 查询窗口: 覆盖 min_history_days 但不过长, 避免把整个历史都拉回来
    lookback_buffer_days: int = 30            # 在 min_history_days 基础上再多留的缓冲天数


class UniverseBuilder:
    """
    Point-in-time universe 构建器

    关键原则:
      - 严禁用 "今天的 Top 50" 回测历史 (survivorship bias)
      - 必须用 "当时的 Top 50" 才有统计效度
    """

    # 当 storage 里还没数据时使用的硬编码候选池 (仅用于首轮 sync-universe-data 启动)
    _BOOTSTRAP_SYMBOLS = [
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

    def __init__(self, storage, config: Optional[UniverseConfig] = None):
        """
        Args:
            storage: data.storage.Storage 实例
            config: UniverseConfig
        """
        self.storage = storage
        self.cfg = config or UniverseConfig()
        self._cache: dict[tuple, list[str]] = {}
        self._candidate_history: dict[str, int] = {}

    # ----------------------------------------------------------------
    # 对外主接口
    # ----------------------------------------------------------------
    def get_universe(self, as_of_date: str) -> list[str]:
        """
        返回在指定日期应该交易的币种列表

        Args:
            as_of_date: 'YYYY-MM-DD' 格式

        Returns:
            按成交额从大到小排序的 symbol 列表
        """
        key = (as_of_date, self.cfg.top_n, self.cfg.min_volume_30d_usd,
               self.cfg.min_history_days)
        if key in self._cache:
            return self._cache[key]

        universe = self._build_universe(as_of_date)
        self._cache[key] = universe
        return universe

    def get_all_symbols(self) -> list[str]:
        """
        返回候选 symbol 池

        优先级:
          1. storage.list_kline_symbols('1d')  — 已入库的真实 symbols
          2. bootstrap 硬编码池 (首轮 sync 前)
        """
        try:
            if hasattr(self.storage, "list_kline_symbols"):
                symbols = self.storage.list_kline_symbols(interval="1d")
                if symbols:
                    return symbols
        except Exception as e:
            logger.debug(f"list_kline_symbols failed, fallback to bootstrap: {e}")

        return list(self._BOOTSTRAP_SYMBOLS)

    def clear_cache(self) -> None:
        """手动清空 universe / smoothing 缓存 (测试用)"""
        self._cache.clear()
        self._candidate_history.clear()

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
        as_of_ts = pd.Timestamp(as_of_date, tz="UTC").tz_localize(None) \
            if pd.Timestamp(as_of_date).tzinfo is None \
            else pd.Timestamp(as_of_date).tz_convert("UTC").tz_localize(None)

        # 只拉 as_of_date 之前 min_history_days+buffer 天的数据
        lookback_days = self.cfg.min_history_days + self.cfg.lookback_buffer_days
        end_ms = int(as_of_ts.timestamp() * 1000)
        start_ms = end_ms - lookback_days * 86_400_000

        ranked: list[tuple[str, float]] = []

        for symbol in all_symbols:
            try:
                klines = self.storage.get_klines(
                    symbol, "1d",
                    start_time=start_ms,
                    end_time=end_ms,
                )
                if klines.empty:
                    continue
                if len(klines) < self.cfg.min_history_days:
                    continue

                # 数据库是升序的, 直接取最后 30 行
                last_30 = klines.tail(30)
                if "close" not in last_30.columns or "volume" not in last_30.columns:
                    continue

                # 优先使用 quote_volume (USD 计价), 退化到 close*volume
                if "quote_volume" in last_30.columns and last_30["quote_volume"].sum() > 0:
                    daily_turnover_usd = float(last_30["quote_volume"].mean())
                else:
                    daily_turnover_usd = float(
                        (last_30["close"] * last_30["volume"]).mean()
                    )

                if daily_turnover_usd < self.cfg.min_volume_30d_usd:
                    continue

                ranked.append((symbol, daily_turnover_usd))

            except Exception as e:
                logger.debug(f"skip {symbol}: {e}")
                continue

        ranked.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [s for s, _ in ranked[: self.cfg.top_n * 2]]

        universe = self._apply_smoothing(top_candidates)
        return universe[: self.cfg.top_n]

    def _apply_smoothing(self, candidates: list[str]) -> list[str]:
        """避免因临时流动性尖峰 (如 memecoin) 污染 universe"""
        smoothing = self.cfg.turnover_smoothing

        # 入围 +1
        for symbol in candidates:
            self._candidate_history[symbol] = (
                self._candidate_history.get(symbol, 0) + 1
            )

        # 非入围 -1 (衰减而非归零, 避免抖动)
        candidate_set = set(candidates)
        for symbol in list(self._candidate_history.keys()):
            if symbol not in candidate_set:
                self._candidate_history[symbol] -= 1
                if self._candidate_history[symbol] <= 0:
                    del self._candidate_history[symbol]

        result = [s for s in candidates
                  if self._candidate_history.get(s, 0) >= smoothing]

        # 如果 smoothing 过滤掉太多 (冷启动), 回退到原始列表
        if len(result) < self.cfg.top_n * 0.5:
            return candidates
        return result

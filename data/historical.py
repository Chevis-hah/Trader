"""
全量历史数据下载器
- 分页拉取，支持断点续传
- 多时间框架并行下载
- 数据质量检查（缺口检测、异常值）
- 增量更新
"""
import time
from datetime import datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from config.loader import Config
from data.client import BinanceClient
from data.storage import Storage, INTERVAL_MS
from utils.logger import get_logger

logger = get_logger("history")


class HistoryDownloader:
    """
    全量历史数据管理器
    - 首次运行：拉取配置的全部历史数据（可达数十万条）
    - 后续运行：增量更新到最新
    - 自动检测数据缺口并补全
    """

    def __init__(self, config: Config, client: BinanceClient, storage: Storage):
        self.config = config
        self.client = client
        self.storage = storage
        self._hist_cfg = config.data.history
        self._symbols = config.get_symbols()
        self._timeframes = config.get_timeframes()

    # ==============================================================
    # 主入口
    # ==============================================================
    def sync_all(self, max_workers: int = 3):
        """
        同步所有交易对和时间框架的数据
        首次运行会拉取全部历史，后续增量更新
        """
        tasks = []
        for symbol in self._symbols:
            for tf in self._timeframes:
                tasks.append((symbol, tf["interval"], tf.get("retention_days", 730)))

        logger.info(f"开始数据同步 | {len(tasks)} 个任务 | 标的={self._symbols}")

        # 多线程并行（避免单线程太慢）
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._sync_one, sym, itv, days): (sym, itv)
                for sym, itv, days in tasks
            }
            for future in as_completed(futures):
                sym, itv = futures[future]
                try:
                    count = future.result()
                    results[(sym, itv)] = count
                    logger.info(f"✅ {sym}/{itv} 同步完成，共 {count} 条")
                except Exception as e:
                    logger.error(f"❌ {sym}/{itv} 同步失败: {e}")
                    results[(sym, itv)] = -1

        # 汇总
        total = sum(v for v in results.values() if v > 0)
        failed = sum(1 for v in results.values() if v < 0)
        logger.info(f"数据同步完成 | 总写入={total} 条 | 失败={failed} 个")
        return results

    def _download_range(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        until_ms: int,
        interval_ms: int,
        cap_open_time_lt: Optional[int] = None,
        log_prefix: str = "",
    ) -> int:
        """
        从 start_ms 分页拉取到游标 >= until_ms。
        cap_open_time_lt：仅写入 open_time 小于该值的 K 线（向前补全时避免与已有首根重叠）。
        """
        total_written = 0
        batch_size = 1000
        current_start = start_ms
        consecutive_empty = 0
        span = max(until_ms - start_ms, 1)

        while current_start < until_ms:
            try:
                df = self.client.get_klines(
                    symbol=symbol, interval=interval,
                    limit=batch_size, start_time=current_start)

                if df.empty:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        break
                    current_start += interval_ms * batch_size
                    continue

                consecutive_empty = 0

                if cap_open_time_lt is not None:
                    df = df[df["open_time"] < cap_open_time_lt]
                    if df.empty:
                        break

                written = self.storage.upsert_klines(symbol, interval, df)
                total_written += written

                last_time = int(df["open_time"].iloc[-1])
                current_start = last_time + interval_ms

                if len(df) < batch_size:
                    break

                if total_written % 10000 < batch_size:
                    pct = min(100.0, (current_start - start_ms) / span * 100)
                    logger.info(
                        f"  {symbol}/{interval}{log_prefix} 进度 {pct:.1f}% | "
                        f"已下载 {total_written} 条 | "
                        f"当前到 {datetime.utcfromtimestamp(last_time/1000).strftime('%Y-%m-%d %H:%M')}")

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"下载 {symbol}/{interval} 出错 @ {current_start}: {e}")
                time.sleep(1)
                current_start += interval_ms * batch_size

        return total_written

    def _sync_one(self, symbol: str, interval: str, retention_days: int) -> int:
        """同步单个交易对的单个时间框架"""
        interval_ms = INTERVAL_MS.get(interval, 3600000)
        now_ms = int(time.time() * 1000)
        earliest_ms = now_ms - retention_days * 86400 * 1000

        latest_in_db = self.storage.get_latest_kline_time(symbol, interval)
        earliest_in_db = self.storage.get_earliest_kline_time(symbol, interval)
        total_written = 0

        # 扩大保留窗口后：若库里最早一根仍晚于目标起点，则向前补全
        if earliest_in_db is not None and earliest_in_db > earliest_ms:
            est = retention_days * 86400000 // interval_ms
            logger.info(
                f"向前补全 {symbol}/{interval} | 已有首根 "
                f"{datetime.utcfromtimestamp(earliest_in_db/1000).strftime('%Y-%m-%d %H:%M')} | "
                f"补至 {datetime.utcfromtimestamp(earliest_ms/1000).strftime('%Y-%m-%d')} | 预估约 {est} 条")
            total_written += self._download_range(
                symbol, interval, earliest_ms, earliest_in_db, interval_ms,
                cap_open_time_lt=earliest_in_db, log_prefix=" [向前]")

        if latest_in_db and latest_in_db > earliest_ms:
            start_ms = latest_in_db + interval_ms
            existing = self.storage.get_kline_count(symbol, interval)
            logger.info(
                f"增量更新 {symbol}/{interval} | 已有 {existing} 条 | "
                f"从 {datetime.utcfromtimestamp(start_ms/1000).strftime('%Y-%m-%d %H:%M')} 开始")
        else:
            start_ms = earliest_ms
            logger.info(
                f"全量下载 {symbol}/{interval} | "
                f"从 {datetime.utcfromtimestamp(start_ms/1000).strftime('%Y-%m-%d')} 开始 | "
                f"预估 {retention_days * 86400000 // interval_ms} 条")

        total_written += self._download_range(
            symbol, interval, start_ms, now_ms, interval_ms, log_prefix="")

        return total_written

    # ==============================================================
    # 数据质量
    # ==============================================================
    def check_gaps(self, symbol: str, interval: str) -> list[tuple[int, int]]:
        """
        检测数据缺口
        返回: [(gap_start, gap_end), ...]
        """
        df = self.storage.get_klines(symbol, interval)
        if len(df) < 2:
            return []

        interval_ms = INTERVAL_MS.get(interval, 3600000)
        times = df["open_time"].values
        diffs = np.diff(times)
        expected = interval_ms

        gaps = []
        # 允许 5% 的误差（交易所偶尔有微小延迟）
        gap_mask = diffs > expected * 1.5
        gap_indices = np.where(gap_mask)[0]

        for idx in gap_indices:
            gap_start = int(times[idx])
            gap_end = int(times[idx + 1])
            gap_bars = (gap_end - gap_start) // interval_ms - 1
            gaps.append((gap_start, gap_end))
            logger.warning(
                f"数据缺口 {symbol}/{interval}: "
                f"{datetime.utcfromtimestamp(gap_start/1000)} ~ "
                f"{datetime.utcfromtimestamp(gap_end/1000)} "
                f"({gap_bars} bars)")

        return gaps

    def fill_gaps(self, symbol: str, interval: str):
        """补全检测到的数据缺口"""
        gaps = self.check_gaps(symbol, interval)
        interval_ms = INTERVAL_MS.get(interval, 3600000)

        for gap_start, gap_end in gaps:
            logger.info(f"补全缺口 {symbol}/{interval}: {gap_start} ~ {gap_end}")
            current = gap_start + interval_ms
            while current < gap_end:
                try:
                    df = self.client.get_klines(
                        symbol, interval, limit=1000, start_time=current)
                    if not df.empty:
                        self.storage.upsert_klines(symbol, interval, df)
                        current = int(df["open_time"].iloc[-1]) + interval_ms
                    else:
                        break
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"补全失败: {e}")
                    break

    def validate_data(self, symbol: str, interval: str) -> dict:
        """数据质量校验"""
        df = self.storage.get_klines(symbol, interval)
        if df.empty:
            return {"status": "empty", "count": 0}

        report = {
            "symbol": symbol,
            "interval": interval,
            "count": len(df),
            "start": datetime.utcfromtimestamp(df["open_time"].iloc[0] / 1000).isoformat(),
            "end": datetime.utcfromtimestamp(df["open_time"].iloc[-1] / 1000).isoformat(),
            "days_covered": (df["open_time"].iloc[-1] - df["open_time"].iloc[0]) / 86400000,
        }

        # 检查 OHLCV 异常
        report["null_count"] = int(df[["open", "high", "low", "close", "volume"]].isnull().sum().sum())
        report["zero_volume_count"] = int((df["volume"] == 0).sum())
        report["high_lt_low"] = int((df["high"] < df["low"]).sum())
        report["negative_price"] = int((df["close"] <= 0).sum())

        # 检查缺口
        gaps = self.check_gaps(symbol, interval)
        report["gaps"] = len(gaps)
        report["status"] = "ok" if report["gaps"] == 0 and report["null_count"] == 0 else "issues"

        return report

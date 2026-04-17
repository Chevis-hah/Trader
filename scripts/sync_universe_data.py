#!/usr/bin/env python3
"""
P1-T01: 同步 Top 60 币种日线数据到 data/quant.db

使用 Binance 公开 REST API (/api/v3/klines), 无需 API key。

特性:
  - 增量同步 (已有数据从最后 bar 之后继续拉)
  - Rate limit 友好 (默认 每分钟 < 1000 请求)
  - 自动重试 (指数退避) + 429/418 兜底
  - 数据完整性校验 (gap 检测, 0 成交量连续日检测)
  - 断点续传 (中途 Ctrl-C 后重跑只会拉缺的)

用法:
  python scripts/sync_universe_data.py                    # 默认 60 币种 5 年
  python scripts/sync_universe_data.py --top-n 30         # 只拉 30 个
  python scripts/sync_universe_data.py --start 2023-01-01 # 自定义起始日
  python scripts/sync_universe_data.py --interval 4h      # 拉 4h 不是 1d
  python scripts/sync_universe_data.py --validate-only    # 只跑数据完整性校验, 不拉数据

验收参考 (P1-T01):
  - >= 45 symbol 有 >= 4 年日线
  - DB 行数 >= 65_700 (约 45 * 1460)
  - BTCUSDT 1d rows > 1500
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# 允许从项目根目录直接跑
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.storage import Storage, INTERVAL_MS  # noqa: E402
from data.universe import UniverseBuilder  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("sync_universe")

BINANCE_SPOT_API = "https://api.binance.com/api/v3/klines"
BATCH_LIMIT = 1000  # Binance 单次请求上限
DEFAULT_START = "2020-01-01"

# 为了不一定要去跑 UniverseBuilder 冷启动, 直接用其 bootstrap 池
DEFAULT_SYMBOL_POOL = list(UniverseBuilder._BOOTSTRAP_SYMBOLS)


# ============================================================
# HTTP 层
# ============================================================
def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _fetch_klines_batch(symbol: str, interval: str,
                         start_ms: int, end_ms: int,
                         max_retries: int = 5) -> list[list]:
    """
    单次 fetch, 最多 BATCH_LIMIT 条

    Returns: raw Binance klines list (每条 12 字段)
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": BATCH_LIMIT,
    }
    backoff = 1.0
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(BINANCE_SPOT_API, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (418, 429):
                # IP 被限或 rate limit, 遵守 Retry-After
                retry_after = int(resp.headers.get("Retry-After", "5"))
                logger.warning(
                    f"{symbol} {interval}: HTTP {resp.status_code}, "
                    f"Retry-After={retry_after}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_after + random.uniform(0, 1))
                continue

            if 500 <= resp.status_code < 600:
                logger.warning(
                    f"{symbol} {interval}: HTTP {resp.status_code}, "
                    f"retry in {backoff:.1f}s"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            # 4xx (400, 404 未上市符号等) 不重试
            logger.debug(
                f"{symbol} {interval}: HTTP {resp.status_code} body={resp.text[:200]}"
            )
            return []

        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            logger.warning(f"{symbol}: network error {e}, retry in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    raise RuntimeError(
        f"fetch {symbol} failed after {max_retries} retries, last={last_err}"
    )


def _klines_to_df(raw: list[list]) -> pd.DataFrame:
    """把 Binance 原始 klines 数组转成 storage.upsert_klines 期望的 DataFrame"""
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades_count",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df.drop(columns=["ignore"])
    df = df.astype({
        "open_time": "int64", "close_time": "int64", "trades_count": "int64",
        "open": "float64", "high": "float64", "low": "float64",
        "close": "float64", "volume": "float64", "quote_volume": "float64",
        "taker_buy_base": "float64", "taker_buy_quote": "float64",
    })
    return df


# ============================================================
# Sync 层
# ============================================================
def sync_symbol(storage: Storage, symbol: str, interval: str,
                start_ms: int, end_ms: int,
                request_sleep: float = 0.15) -> dict:
    """
    增量同步单个 symbol 的 klines

    Returns: {"symbol": ..., "rows_added": int, "status": "ok"|"no_data"|"error",
              "start_ms": ..., "end_ms": ...}
    """
    interval_ms = INTERVAL_MS.get(interval)
    if interval_ms is None:
        raise ValueError(f"未知 interval: {interval}")

    # 增量: 从 DB 中最后一条 bar 的下一个 bar 开始
    latest = storage.get_latest_kline_time(symbol, interval)
    cursor = latest + interval_ms if latest else start_ms
    if cursor >= end_ms:
        return {"symbol": symbol, "rows_added": 0, "status": "ok",
                "start_ms": cursor, "end_ms": end_ms, "note": "already up to date"}

    total_rows = 0
    windows_fetched = 0

    while cursor < end_ms:
        # 单次 BATCH_LIMIT 条 * interval_ms = 窗口长度
        window_end = min(cursor + BATCH_LIMIT * interval_ms, end_ms)
        raw = _fetch_klines_batch(symbol, interval, cursor, window_end)
        windows_fetched += 1

        if not raw:
            # 可能是未上市/下架, 也可能是这个窗口真的空
            if total_rows == 0 and windows_fetched == 1:
                return {"symbol": symbol, "rows_added": 0, "status": "no_data",
                        "start_ms": cursor, "end_ms": end_ms}
            # 往后推一个窗口
            cursor = window_end
            continue

        df = _klines_to_df(raw)
        if df.empty:
            cursor = window_end
            continue

        rows = storage.upsert_klines(symbol, interval, df)
        total_rows += rows

        last_open_time = int(df["open_time"].iloc[-1])
        # 下一批从 last_open_time + interval_ms 开始
        cursor = last_open_time + interval_ms

        # 友好 sleep, 避免压垮 Binance
        time.sleep(request_sleep)

    return {"symbol": symbol, "rows_added": total_rows, "status": "ok",
            "start_ms": start_ms, "end_ms": end_ms}


# ============================================================
# 数据校验
# ============================================================
def validate_coverage(storage: Storage, symbols: list[str], interval: str,
                      min_years: float = 4.0,
                      min_rows_hint: int = 1400) -> dict:
    """
    校验 DB 中每个 symbol 的覆盖度

    Returns: {"pass": int, "fail": int, "per_symbol": [...]}
    """
    interval_ms = INTERVAL_MS[interval]
    min_rows = int(min_years * 365 * 86_400_000 / interval_ms * 0.95)  # 允许 5% 缺口
    min_rows = max(min_rows, min_rows_hint)

    results = []
    passed = 0
    total_rows = 0

    for symbol in symbols:
        count = storage.get_kline_count(symbol, interval)
        total_rows += count

        gap_report = storage.check_kline_gaps(symbol, interval)
        ok = count >= min_rows and gap_report["zero_vol_streak_max"] < 30
        results.append({
            "symbol": symbol,
            "rows": count,
            "gaps": len(gap_report["gaps"]),
            "zero_vol_streak_max": gap_report["zero_vol_streak_max"],
            "pass": ok,
        })
        if ok:
            passed += 1

    return {
        "pass": passed,
        "fail": len(symbols) - passed,
        "min_rows_required": min_rows,
        "total_db_rows": total_rows,
        "per_symbol": results,
    }


def print_coverage_report(report: dict, top_n: int = 15) -> None:
    print("\n" + "=" * 72)
    print(f" 数据覆盖度报告 (min_rows_required={report['min_rows_required']})")
    print("=" * 72)
    print(f" 通过: {report['pass']}   失败: {report['fail']}   "
          f"总行数: {report['total_db_rows']:,}")
    print("-" * 72)
    print(f" {'symbol':<14} {'rows':>7} {'gaps':>5} {'0vol_streak':>12} {'pass':>6}")
    print("-" * 72)
    # 打印前 N 和未通过的
    fails = [r for r in report["per_symbol"] if not r["pass"]]
    head = report["per_symbol"][:top_n]
    shown_syms = {r["symbol"] for r in head}
    for r in head:
        mark = "✓" if r["pass"] else "✗"
        print(f" {r['symbol']:<14} {r['rows']:>7} {r['gaps']:>5} "
              f"{r['zero_vol_streak_max']:>12} {mark:>6}")
    extra_fails = [r for r in fails if r["symbol"] not in shown_syms]
    if extra_fails:
        print(f" ... 另有 {len(extra_fails)} 个未通过:")
        for r in extra_fails[:20]:
            print(f" {r['symbol']:<14} {r['rows']:>7} {r['gaps']:>5} "
                  f"{r['zero_vol_streak_max']:>12} {'✗':>6}")
    print("=" * 72 + "\n")


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Sync Top N crypto klines to quant.db")
    parser.add_argument("--db", type=str, default="data/quant.db")
    parser.add_argument("--top-n", type=int, default=60,
                        help="要拉取的币种数量 (从 bootstrap pool 头开始)")
    parser.add_argument("--start", type=str, default=DEFAULT_START,
                        help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="结束日期 YYYY-MM-DD, 默认今天")
    parser.add_argument("--interval", type=str, default="1d",
                        choices=list(INTERVAL_MS.keys()))
    parser.add_argument("--symbols", type=str, default=None,
                        help="逗号分隔 symbol 列表, 覆盖默认池")
    parser.add_argument("--request-sleep", type=float, default=0.15,
                        help="每次请求之间的 sleep 秒数")
    parser.add_argument("--validate-only", action="store_true",
                        help="跳过下载, 只校验现有数据")
    parser.add_argument("--min-years", type=float, default=4.0,
                        help="校验时要求的最小年数")
    args = parser.parse_args()

    # 决定 symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = DEFAULT_SYMBOL_POOL[:args.top_n]

    storage = Storage(args.db)

    if args.validate_only:
        report = validate_coverage(storage, symbols, args.interval,
                                    min_years=args.min_years)
        print_coverage_report(report)
        sys.exit(0 if report["fail"] == 0 else 1)

    start_ms = _date_to_ms(args.start)
    end_ms = _date_to_ms(args.end) if args.end else _now_ms()

    logger.info(
        f"同步 {len(symbols)} 个 symbol, interval={args.interval}, "
        f"范围={args.start} ~ {args.end or 'now'}"
    )

    t0 = time.time()
    summary = {"ok": 0, "no_data": 0, "error": 0, "total_rows": 0}

    for i, symbol in enumerate(symbols, 1):
        try:
            res = sync_symbol(storage, symbol, args.interval,
                              start_ms, end_ms,
                              request_sleep=args.request_sleep)
            logger.info(
                f"[{i}/{len(symbols)}] {symbol}: "
                f"+{res['rows_added']} rows ({res['status']})"
            )
            summary[res["status"]] = summary.get(res["status"], 0) + 1
            summary["total_rows"] += res["rows_added"]
        except KeyboardInterrupt:
            logger.warning("用户中断, 已入库数据保留, 下次重跑会从断点续传")
            break
        except Exception as e:
            logger.error(f"[{i}/{len(symbols)}] {symbol}: ERROR {e}")
            summary["error"] += 1

    elapsed = time.time() - t0
    logger.info("-" * 60)
    logger.info(f"完成: ok={summary['ok']} no_data={summary.get('no_data', 0)} "
                f"error={summary['error']} total_rows={summary['total_rows']:,} "
                f"耗时={elapsed:.1f}s")

    # 跑一次校验
    report = validate_coverage(storage, symbols, args.interval,
                                min_years=args.min_years)
    print_coverage_report(report)

    # vacuum 一次, 释放空间
    storage.vacuum()

    sys.exit(0 if report["fail"] < len(symbols) * 0.3 else 1)


if __name__ == "__main__":
    main()

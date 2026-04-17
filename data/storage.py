"""
数据存储层
- SQLite with WAL mode（单机部署）
- 可扩展为 PostgreSQL + TimescaleDB
- 按时间分区思想管理数据
- 批量写入优化
"""
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("storage")

# Binance interval -> 毫秒
INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000, "1w": 604_800_000,
}


class Storage:
    """生产级数据存储"""

    def __init__(self, db_path: str | Path, cache_size_mb: int = 256,
                 busy_timeout_ms: int = 5000):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_size_mb = cache_size_mb
        self._busy_timeout_ms = busy_timeout_ms
        self._init_db()
        logger.info(f"数据库初始化完成: {self.db_path}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=self._busy_timeout_ms / 1000)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA cache_size=-{self._cache_size_mb * 1024}")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                -- ============================================================
                -- K线数据（核心表）
                -- ============================================================
                CREATE TABLE IF NOT EXISTS klines (
                    symbol          TEXT    NOT NULL,
                    interval        TEXT    NOT NULL,
                    open_time       INTEGER NOT NULL,
                    open            REAL    NOT NULL,
                    high            REAL    NOT NULL,
                    low             REAL    NOT NULL,
                    close           REAL    NOT NULL,
                    volume          REAL    NOT NULL,
                    close_time      INTEGER NOT NULL,
                    quote_volume    REAL,
                    trades_count    INTEGER,
                    taker_buy_base  REAL,
                    taker_buy_quote REAL,
                    PRIMARY KEY (symbol, interval, open_time)
                ) WITHOUT ROWID;

                CREATE INDEX IF NOT EXISTS idx_klines_time
                    ON klines(symbol, interval, open_time DESC);

                -- ============================================================
                -- 订单簿快照
                -- ============================================================
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT    NOT NULL,
                    timestamp       INTEGER NOT NULL,
                    bids            TEXT    NOT NULL,
                    asks            TEXT    NOT NULL,
                    mid_price       REAL,
                    spread          REAL,
                    bid_depth       REAL,
                    ask_depth       REAL,
                    imbalance       REAL
                );

                CREATE INDEX IF NOT EXISTS idx_ob_time
                    ON orderbook_snapshots(symbol, timestamp DESC);

                -- ============================================================
                -- 订单生命周期
                -- ============================================================
                CREATE TABLE IF NOT EXISTS orders (
                    order_id        TEXT    PRIMARY KEY,
                    client_order_id TEXT,
                    symbol          TEXT    NOT NULL,
                    side            TEXT    NOT NULL,
                    order_type      TEXT    NOT NULL,
                    time_in_force   TEXT,
                    quantity        REAL    NOT NULL,
                    price           REAL,
                    stop_price      REAL,
                    executed_qty    REAL    DEFAULT 0,
                    avg_fill_price  REAL,
                    commission      REAL    DEFAULT 0,
                    commission_asset TEXT,
                    status          TEXT    NOT NULL,
                    strategy        TEXT,
                    algo            TEXT,
                    parent_order_id TEXT,
                    created_at      INTEGER NOT NULL,
                    updated_at      INTEGER,
                    filled_at       INTEGER,
                    slippage_bps    REAL,
                    latency_ms      REAL
                );

                CREATE INDEX IF NOT EXISTS idx_orders_symbol_time
                    ON orders(symbol, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_orders_strategy
                    ON orders(strategy, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_orders_status
                    ON orders(status);

                -- ============================================================
                -- 持仓历史
                -- ============================================================
                CREATE TABLE IF NOT EXISTS positions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT    NOT NULL,
                    side            TEXT    NOT NULL DEFAULT 'LONG',
                    quantity        REAL    NOT NULL,
                    entry_price     REAL    NOT NULL,
                    entry_time      INTEGER NOT NULL,
                    exit_price      REAL,
                    exit_time       INTEGER,
                    realized_pnl    REAL,
                    realized_pnl_pct REAL,
                    commission_total REAL   DEFAULT 0,
                    strategy        TEXT,
                    max_favorable   REAL,
                    max_adverse     REAL,
                    holding_bars    INTEGER,
                    status          TEXT    DEFAULT 'OPEN'
                );

                CREATE INDEX IF NOT EXISTS idx_pos_status
                    ON positions(status, symbol);

                -- ============================================================
                -- 因子值快照
                -- ============================================================
                CREATE TABLE IF NOT EXISTS factor_values (
                    symbol          TEXT    NOT NULL,
                    timestamp       INTEGER NOT NULL,
                    factor_name     TEXT    NOT NULL,
                    value           REAL,
                    PRIMARY KEY (symbol, timestamp, factor_name)
                ) WITHOUT ROWID;

                -- ============================================================
                -- 信号记录
                -- ============================================================
                CREATE TABLE IF NOT EXISTS signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT    NOT NULL,
                    timestamp       INTEGER NOT NULL,
                    strategy        TEXT    NOT NULL,
                    direction       TEXT    NOT NULL,
                    strength        REAL,
                    confidence      REAL,
                    features        TEXT,
                    model_version   TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_signals_time
                    ON signals(symbol, timestamp DESC);

                -- ============================================================
                -- 每日绩效
                -- ============================================================
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date            TEXT    PRIMARY KEY,
                    starting_nav    REAL,
                    ending_nav      REAL,
                    daily_return    REAL,
                    cumulative_return REAL,
                    drawdown        REAL,
                    sharpe_rolling  REAL,
                    total_trades    INTEGER,
                    total_volume    REAL,
                    total_commission REAL,
                    win_count       INTEGER,
                    loss_count      INTEGER,
                    gross_pnl       REAL,
                    net_pnl         REAL
                );

                -- ============================================================
                -- 模型元数据
                -- ============================================================
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id        TEXT    PRIMARY KEY,
                    model_type      TEXT,
                    train_start     INTEGER,
                    train_end       INTEGER,
                    val_start       INTEGER,
                    val_end         INTEGER,
                    features_used   TEXT,
                    hyperparams     TEXT,
                    train_ic        REAL,
                    val_ic          REAL,
                    val_sharpe      REAL,
                    created_at      INTEGER,
                    file_path       TEXT
                );

                -- ============================================================
                -- 系统状态
                -- ============================================================
                CREATE TABLE IF NOT EXISTS system_state (
                    key             TEXT    PRIMARY KEY,
                    value           TEXT,
                    updated_at      INTEGER
                );
            """)

    # ==============================================================
    # K线 CRUD - 批量高效写入
    # ==============================================================
    def upsert_klines(self, symbol: str, interval: str, df: pd.DataFrame):
        """批量 upsert K线（使用事务，万级数据秒级写入）"""
        if df.empty:
            return 0

        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades_count",
                "taker_buy_base", "taker_buy_quote"]

        rows = []
        for _, r in df.iterrows():
            rows.append((
                symbol, interval, int(r["open_time"]),
                float(r["open"]), float(r["high"]), float(r["low"]),
                float(r["close"]), float(r["volume"]), int(r["close_time"]),
                float(r.get("quote_volume", 0)),
                int(r.get("trades_count", r.get("trades", 0))),
                float(r.get("taker_buy_base", 0)),
                float(r.get("taker_buy_quote", 0)),
            ))

        with self._conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO klines
                   (symbol, interval, open_time, open, high, low, close,
                    volume, close_time, quote_volume, trades_count,
                    taker_buy_base, taker_buy_quote)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows)

        logger.debug(f"写入 {symbol}/{interval} K线 {len(rows)} 条")
        return len(rows)

    def get_klines(self, symbol: str, interval: str,
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None,
                   limit: Optional[int] = None) -> pd.DataFrame:
        """读取K线 - 支持时间范围和条数限制"""
        conditions = ["symbol=?", "interval=?"]
        params: list = [symbol, interval]

        if start_time is not None:
            conditions.append("open_time >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("open_time <= ?")
            params.append(end_time)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM klines WHERE {where} ORDER BY open_time ASC"
        if limit:
            query += f" LIMIT {limit}"

        with self._conn() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_latest_kline_time(self, symbol: str, interval: str) -> Optional[int]:
        """获取最新一条K线的 open_time"""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT MAX(open_time) FROM klines WHERE symbol=? AND interval=?",
                (symbol, interval))
            row = cur.fetchone()
        return row[0] if row and row[0] else None

    def get_earliest_kline_time(self, symbol: str, interval: str) -> Optional[int]:
        """获取最早一条K线的 open_time"""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT MIN(open_time) FROM klines WHERE symbol=? AND interval=?",
                (symbol, interval))
            row = cur.fetchone()
        return row[0] if row and row[0] else None

    def get_kline_count(self, symbol: str, interval: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM klines WHERE symbol=? AND interval=?",
                (symbol, interval))
            return cur.fetchone()[0]

    # ==============================================================
    # 通用写入
    # ==============================================================
    def save_order(self, order: dict):
        cols = ["order_id", "client_order_id", "symbol", "side", "order_type",
                "time_in_force", "quantity", "price", "stop_price",
                "executed_qty", "avg_fill_price", "commission",
                "commission_asset", "status", "strategy", "algo",
                "parent_order_id", "created_at", "updated_at", "filled_at",
                "slippage_bps", "latency_ms"]
        vals = tuple(order.get(c) for c in cols)
        placeholders = ",".join(["?"] * len(cols))
        col_names = ",".join(cols)
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO orders ({col_names}) VALUES ({placeholders})",
                vals)

    def save_signal(self, signal: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO signals
                   (symbol, timestamp, strategy, direction, strength, confidence,
                    features, model_version)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (signal["symbol"], signal["timestamp"], signal["strategy"],
                 signal["direction"], signal.get("strength"),
                 signal.get("confidence"), signal.get("features"),
                 signal.get("model_version")))

    def save_position(self, pos: dict):
        with self._conn() as conn:
            if "id" in pos and pos["id"]:
                conn.execute(
                    """UPDATE positions SET exit_price=?, exit_time=?,
                       realized_pnl=?, realized_pnl_pct=?, commission_total=?,
                       max_favorable=?, max_adverse=?, holding_bars=?, status=?
                       WHERE id=?""",
                    (pos.get("exit_price"), pos.get("exit_time"),
                     pos.get("realized_pnl"), pos.get("realized_pnl_pct"),
                     pos.get("commission_total"), pos.get("max_favorable"),
                     pos.get("max_adverse"), pos.get("holding_bars"),
                     pos.get("status", "CLOSED"), pos["id"]))
            else:
                conn.execute(
                    """INSERT INTO positions
                       (symbol, side, quantity, entry_price, entry_time, strategy, status)
                       VALUES (?,?,?,?,?,?,?)""",
                    (pos["symbol"], pos.get("side", "LONG"), pos["quantity"],
                     pos["entry_price"], pos["entry_time"],
                     pos.get("strategy"), "OPEN"))
                return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_open_positions(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM positions WHERE status='OPEN'", conn)

    def save_state(self, key: str, value: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?,?,?)",
                (key, value, int(time.time() * 1000)))

    def get_state(self, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def get_closed_positions(self, limit: int = 200) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM positions WHERE status='CLOSED' ORDER BY exit_time DESC LIMIT ?",
                conn, params=(limit,))

    def vacuum(self):
        """整理数据库碎片"""
        with self._conn() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ==============================================================
    # Universe / 数据质量辅助 (v2.3.1 新增)
    # ==============================================================
    def list_kline_symbols(self, interval: Optional[str] = None) -> list[str]:
        """
        返回 klines 表中已入库的 symbol 列表 (去重, 字典序)

        Args:
            interval: 若指定, 只返回在该 interval 上有数据的 symbol
        """
        query = "SELECT DISTINCT symbol FROM klines"
        params: tuple = ()
        if interval is not None:
            query += " WHERE interval = ?"
            params = (interval,)
        query += " ORDER BY symbol ASC"
        with self._conn() as conn:
            cur = conn.execute(query, params)
            return [row[0] for row in cur.fetchall()]

    def get_kline_coverage(self, interval: str) -> pd.DataFrame:
        """
        返回每个 symbol 在指定 interval 上的数据覆盖情况

        Returns:
            DataFrame with columns: symbol, rows, first_time, last_time
        """
        query = """
            SELECT symbol,
                   COUNT(*) AS rows,
                   MIN(open_time) AS first_time,
                   MAX(open_time) AS last_time
            FROM klines
            WHERE interval = ?
            GROUP BY symbol
            ORDER BY rows DESC
        """
        with self._conn() as conn:
            df = pd.read_sql_query(query, conn, params=(interval,))
        return df

    def check_kline_gaps(self, symbol: str, interval: str,
                        expected_interval_ms: Optional[int] = None
                        ) -> dict:
        """
        检测 K 线数据中的时间跳缺

        Args:
            symbol: 交易对
            interval: 时间框架 (如 '1d')
            expected_interval_ms: 期望的 bar 间隔毫秒数, 默认从 INTERVAL_MS 查

        Returns:
            {'total': N, 'gaps': [(from_ts, to_ts, missing_bars), ...],
             'zero_vol_streak_max': int}
        """
        if expected_interval_ms is None:
            expected_interval_ms = INTERVAL_MS.get(interval)
            if expected_interval_ms is None:
                raise ValueError(f"未知 interval: {interval}")

        df = self.get_klines(symbol, interval)
        if df.empty:
            return {"total": 0, "gaps": [], "zero_vol_streak_max": 0}

        total = len(df)
        times = df["open_time"].tolist()
        gaps: list[tuple[int, int, int]] = []
        for i in range(1, len(times)):
            delta = times[i] - times[i - 1]
            if delta > expected_interval_ms * 1.5:
                missing = int(delta / expected_interval_ms) - 1
                gaps.append((times[i - 1], times[i], missing))

        # 连续 0 成交量
        zero_streak = 0
        zero_streak_max = 0
        for v in df["volume"].tolist():
            if v == 0:
                zero_streak += 1
                zero_streak_max = max(zero_streak_max, zero_streak)
            else:
                zero_streak = 0

        return {
            "total": total,
            "gaps": gaps,
            "zero_vol_streak_max": zero_streak_max,
        }

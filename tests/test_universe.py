"""
tests/test_universe.py — P1-T02

验证 UniverseBuilder 的关键正确性:
  1. Point-in-time: 2021-06-01 的 universe 不包含 2022 才上市的币种
  2. 缓存一致性: 同一日期多次调用返回一致结果
  3. Turnover smoothing 过滤掉瞬时流动性尖峰 (pump-and-dump memecoin)
  4. min_volume_30d_usd 流动性过滤生效
  5. min_history_days 最小历史数据过滤生效

运行:
  python -m pytest tests/test_universe.py -v
  python tests/test_universe.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.storage import Storage
from data.universe import UniverseBuilder, UniverseConfig


# ============================================================
# Helpers
# ============================================================
def _make_daily_klines(
    start_date: str,
    n_days: int,
    base_price: float = 100.0,
    base_volume: float = 1_000.0,
    price_drift: float = 0.0,
    noise: float = 0.0,
    spike_day: int | None = None,
    spike_volume_mult: float = 100.0,
) -> pd.DataFrame:
    """
    构造合成日线数据

    Args:
        start_date: 起始日 YYYY-MM-DD (UTC)
        n_days: 天数
        base_price: 基准 close
        base_volume: 基准日成交量 (token 单位)
        price_drift: 每日价格漂移率
        noise: 每日随机波动率
        spike_day: 第 N 天注入成交量尖峰 (用于模拟 memecoin pump)
        spike_volume_mult: 尖峰倍数
    """
    start_ts = pd.Timestamp(start_date, tz="UTC").timestamp() * 1000
    rows = []
    price = base_price
    rng = np.random.default_rng(42)
    for i in range(n_days):
        open_time = int(start_ts + i * 86_400_000)
        close_time = open_time + 86_399_999
        if noise > 0:
            price = price * (1 + price_drift + rng.normal(0, noise))
        else:
            price = price * (1 + price_drift)
        price = max(price, 0.01)

        vol = base_volume
        if spike_day is not None and i == spike_day:
            vol = base_volume * spike_volume_mult

        quote_vol = price * vol
        rows.append({
            "open_time": open_time,
            "open": price * 0.999,
            "high": price * 1.001,
            "low": price * 0.998,
            "close": price,
            "volume": vol,
            "close_time": close_time,
            "quote_volume": quote_vol,
            "trades_count": 100,
            "taker_buy_base": vol * 0.5,
            "taker_buy_quote": quote_vol * 0.5,
        })
    return pd.DataFrame(rows)


def _seed_storage(storage: Storage) -> None:
    """
    造一套带时间戳语义的 universe 测试数据:

      - ALPHAUSDT: 2020-01-01 起, 5 年, 大成交额 (固定入围)
      - BRAVOUSDT: 2020-01-01 起, 5 年, 中等成交额 (固定入围)
      - CHARLIEUSDT: 2020-01-01 起, 5 年, 小成交额 (低于 min_volume)
      - LATECOIN: 2022-06-01 起 (模拟晚上市, 2021 查 universe 时不应出现)
      - OLDCOIN: 2020-01-01 ~ 2020-06-01 存在, 之后下架 (只有 150 天数据)
      - MEMEPUMP: 2020-01-01 起, 大部分天超低成交额, 只在某一天尖峰
    """
    # ALPHA: turnover_usd ≈ 100 * 5_000_000 = 5e8 USD/day
    storage.upsert_klines(
        "ALPHAUSDT", "1d",
        _make_daily_klines("2020-01-01", 365 * 5,
                            base_price=100, base_volume=5_000_000)
    )
    # BRAVO: turnover_usd ≈ 50 * 2_000_000 = 1e8 USD/day
    storage.upsert_klines(
        "BRAVOUSDT", "1d",
        _make_daily_klines("2020-01-01", 365 * 5,
                            base_price=50, base_volume=2_000_000)
    )
    # CHARLIE: turnover_usd ≈ 10 * 50_000 = 5e5 USD/day  (低于 min_volume)
    storage.upsert_klines(
        "CHARLIEUSDT", "1d",
        _make_daily_klines("2020-01-01", 365 * 5,
                            base_price=10, base_volume=50_000)
    )
    # LATECOIN: 从 2022-06-01 才开始
    storage.upsert_klines(
        "LATECOIN", "1d",
        _make_daily_klines("2022-06-01", 365 * 2,
                            base_price=1, base_volume=10_000_000)
    )
    # OLDCOIN: 只有 150 天数据 (少于 min_history_days=180)
    storage.upsert_klines(
        "OLDCOIN", "1d",
        _make_daily_klines("2020-01-01", 150,
                            base_price=20, base_volume=3_000_000)
    )
    # MEMEPUMP: 平时只有 100 USD/day, 但某一天突然 100 倍成交
    storage.upsert_klines(
        "MEMEPUMP", "1d",
        _make_daily_klines("2020-01-01", 365 * 5,
                            base_price=0.01, base_volume=10_000,
                            spike_day=60, spike_volume_mult=500)
    )


class TestUniversePointInTime(unittest.TestCase):
    """核心: point-in-time 正确性 — 2021 年 universe 不能有 2022 年才上市的币"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.tmp.name) / "test_universe.db"
        cls.storage = Storage(str(cls.db_path))
        _seed_storage(cls.storage)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _make_builder(self, **cfg_overrides) -> UniverseBuilder:
        base = dict(
            top_n=10,
            min_volume_30d_usd=5_000_000,
            min_history_days=180,
            turnover_smoothing=1,   # 测试大多场景不需要 smoothing
        )
        base.update(cfg_overrides)
        return UniverseBuilder(self.storage, UniverseConfig(**base))

    # -------------------------------
    # Test 1: Point-in-time
    # -------------------------------
    def test_point_in_time_excludes_future_listings(self):
        """2021-06-01 的 universe 不应包含 LATECOIN (上市于 2022-06-01)"""
        builder = self._make_builder()
        u = builder.get_universe("2021-06-01")
        self.assertNotIn(
            "LATECOIN", u,
            "LATECOIN 在 2021-06-01 还未上市, 不应出现在 universe"
        )
        self.assertIn("ALPHAUSDT", u)

    def test_point_in_time_includes_active_listings(self):
        """2023-06-01 的 universe 应该包含 LATECOIN (已上市 1 年)"""
        builder = self._make_builder()
        u = builder.get_universe("2023-06-01")
        self.assertIn("LATECOIN", u,
                      "LATECOIN 在 2023-06-01 已上市 1 年, 应出现")

    def test_different_dates_yield_different_universes(self):
        """2022-01-01 和 2024-01-01 的 universe 必须有差别 (point-in-time)"""
        b1 = self._make_builder()
        u_2022 = b1.get_universe("2022-01-01")
        b2 = self._make_builder()
        u_2024 = b2.get_universe("2024-01-01")
        # LATECOIN 2022 初还没上市, 2024 已上市
        self.assertNotIn("LATECOIN", u_2022)
        self.assertIn("LATECOIN", u_2024)

    # -------------------------------
    # Test 2: 缓存一致性
    # -------------------------------
    def test_cache_returns_identical_result(self):
        """同一 builder 同一日期多次调用返回一致 list"""
        builder = self._make_builder()
        u1 = builder.get_universe("2023-01-01")
        u2 = builder.get_universe("2023-01-01")
        self.assertEqual(u1, u2)

    # -------------------------------
    # Test 3: 最小历史过滤
    # -------------------------------
    def test_insufficient_history_excluded(self):
        """OLDCOIN 只有 150 天, min_history_days=180 应过滤掉"""
        builder = self._make_builder(min_history_days=180)
        u = builder.get_universe("2023-01-01")
        self.assertNotIn("OLDCOIN", u)

    # -------------------------------
    # Test 4: 流动性阈值
    # -------------------------------
    def test_min_volume_filter(self):
        """CHARLIE 每日 turnover ≈ 500k USD, 低于 5M USD 阈值, 应被过滤"""
        builder = self._make_builder(min_volume_30d_usd=5_000_000)
        u = builder.get_universe("2023-01-01")
        self.assertNotIn("CHARLIEUSDT", u)

    def test_low_threshold_lets_low_liquidity_pass(self):
        """把阈值降到 100k 以下, CHARLIE 应重新入围"""
        builder = self._make_builder(min_volume_30d_usd=100_000)
        u = builder.get_universe("2023-01-01")
        self.assertIn("CHARLIEUSDT", u)

    # -------------------------------
    # Test 5: Turnover smoothing 过滤 memecoin
    # -------------------------------
    def test_turnover_smoothing_blocks_memecoin_spike(self):
        """
        MEMEPUMP 只有一天有成交量尖峰, 其它天都极低.
        即便尖峰当天被算为高成交额, 把查询日期定在尖峰之后,
        MEMEPUMP 的 30 日均成交额会回落到低位, 应被 min_volume 阈值过滤掉.
        """
        builder = self._make_builder(min_volume_30d_usd=5_000_000)
        # 在尖峰 (第 60 天 ~2020-03-01) 之后很久查询
        u = builder.get_universe("2023-01-01")
        self.assertNotIn("MEMEPUMP", u,
                          "MEMEPUMP 长期低流动, 不应被尖峰拉入 universe")

    # -------------------------------
    # Test 6: top_n 限制
    # -------------------------------
    def test_top_n_caps_result(self):
        """top_n=2 时只返回 2 个 symbol"""
        builder = self._make_builder(top_n=2)
        u = builder.get_universe("2023-01-01")
        self.assertLessEqual(len(u), 2)

    def test_ordering_by_turnover(self):
        """ALPHA 的 turnover 应 > BRAVO, 排名靠前"""
        builder = self._make_builder(top_n=10)
        u = builder.get_universe("2023-01-01")
        self.assertIn("ALPHAUSDT", u)
        self.assertIn("BRAVOUSDT", u)
        self.assertLess(u.index("ALPHAUSDT"), u.index("BRAVOUSDT"))


class TestUniverseEdgeCases(unittest.TestCase):
    """边界情况"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "edge.db"
        self.storage = Storage(str(self.db_path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_storage_returns_empty(self):
        """空 DB 应返回空 universe 而非报错"""
        builder = UniverseBuilder(self.storage, UniverseConfig(top_n=10))
        u = builder.get_universe("2023-01-01")
        self.assertEqual(u, [])

    def test_get_all_symbols_falls_back_to_bootstrap(self):
        """storage 没数据时 get_all_symbols 退化为 bootstrap pool"""
        builder = UniverseBuilder(self.storage)
        syms = builder.get_all_symbols()
        self.assertIn("BTCUSDT", syms)
        self.assertIn("ETHUSDT", syms)

    def test_get_all_symbols_uses_storage_when_available(self):
        """storage 有数据时 get_all_symbols 返回真实入库列表"""
        self.storage.upsert_klines(
            "TESTUSDT", "1d",
            _make_daily_klines("2022-01-01", 200,
                                base_price=5, base_volume=1_000_000)
        )
        builder = UniverseBuilder(self.storage)
        syms = builder.get_all_symbols()
        self.assertIn("TESTUSDT", syms)
        # 不应含 bootstrap 池里的 DOGEUSDT
        self.assertNotIn("DOGEUSDT", syms)


if __name__ == "__main__":
    unittest.main(verbosity=2)

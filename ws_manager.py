"""
Binance WebSocket 实时数据管理器
- 实时 K 线推送（仅写入已收盘 K 线，避免脏数据）
- 20 档订单簿深度流
- 自动重连 + 指数退避
- 心跳 / pong 处理
- HTTP/SOCKS5 代理支持
- 独立线程运行，不阻塞主引擎
"""
import json
import time
import asyncio
import threading
from typing import Optional, Callable
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

try:
    import websockets
    import websockets.exceptions
except ImportError:
    raise ImportError("请安装 websockets: pip install websockets>=12.0")

from config.loader import Config
from data.storage import Storage
from core.events import EventBus, Event, EventType
from utils.logger import get_logger

logger = get_logger("ws_manager")


class BinanceWSManager:
    """
    Binance Combined Streams WebSocket 管理器

    数据流：
      <symbol>@kline_<interval>    → 实时 K 线（仅收盘写库）
      <symbol>@depth20@100ms       → 20 档订单簿快照

    生命周期：
      start() → 独立线程内跑 asyncio event loop → stop() 优雅关闭
    """

    def __init__(self, config: Config, storage: Storage,
                 event_bus: Optional[EventBus] = None):
        self.config = config
        self.storage = storage
        self.event_bus = event_bus

        exc = config.exchange
        self._testnet = exc.testnet
        self._ws_base_url = exc.ws_url_test if exc.testnet else exc.ws_url_live

        # 代理配置
        proxy_url = exc._data.get("proxy", "") if hasattr(exc, "_data") else ""
        self._proxy_url = proxy_url

        # 交易对和时间框架
        self._symbols = [s.lower() for s in config.get_symbols()]
        self._symbols_upper = config.get_symbols()

        # WebSocket 配置
        ws_cfg = config.get_nested("exchange.websocket", {})
        if isinstance(ws_cfg, dict):
            self._ping_interval = ws_cfg.get("ping_interval_seconds", 20)
            self._ping_timeout = ws_cfg.get("ping_timeout_seconds", 10)
            self._reconnect_delay_init = ws_cfg.get("reconnect_delay_init", 1.0)
            self._reconnect_delay_max = ws_cfg.get("reconnect_delay_max", 60.0)
            self._kline_intervals = ws_cfg.get("kline_intervals", ["1m", "1h"])
            self._orderbook_enabled = ws_cfg.get("orderbook_enabled", True)
        else:
            self._ping_interval = 20
            self._ping_timeout = 10
            self._reconnect_delay_init = 1.0
            self._reconnect_delay_max = 60.0
            self._kline_intervals = ["1m", "1h"]
            self._orderbook_enabled = True

        # 运行状态
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._connected = False
        self._ws = None
        self._reconnect_count = 0

        # 统计
        self._msg_count = 0
        self._kline_count = 0
        self._orderbook_count = 0
        self._last_msg_time = 0.0
        self._start_time = 0.0

        # 回调（可选，供外部挂载额外处理）
        self._kline_callbacks: list[Callable] = []
        self._orderbook_callbacks: list[Callable] = []

        proxy_info = f" | 代理={proxy_url}" if proxy_url else ""
        logger.info(
            f"WS 管理器初始化 | testnet={self._testnet} | "
            f"标的={self._symbols_upper} | K线={self._kline_intervals} | "
            f"订单簿={'启用' if self._orderbook_enabled else '关闭'}{proxy_info}")

    # ==============================================================
    # 公共 API
    # ==============================================================
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "connected": self._connected,
            "running": self._running,
            "uptime_seconds": round(uptime, 1),
            "total_messages": self._msg_count,
            "klines_received": self._kline_count,
            "orderbook_received": self._orderbook_count,
            "reconnect_count": self._reconnect_count,
            "last_msg_ago": round(time.time() - self._last_msg_time, 1) if self._last_msg_time else None,
        }

    def on_kline(self, callback: Callable):
        """注册 K 线回调: callback(symbol, interval, kline_data)"""
        self._kline_callbacks.append(callback)

    def on_orderbook(self, callback: Callable):
        """注册订单簿回调: callback(symbol, orderbook_data)"""
        self._orderbook_callbacks.append(callback)

    def start(self):
        """启动 WebSocket（独立线程）"""
        if self._running:
            logger.warning("WS 管理器已在运行")
            return

        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._run_loop, name="ws-manager", daemon=True)
        self._thread.start()
        logger.info("WS 管理器已启动（独立线程）")

    def stop(self):
        """优雅关闭 WebSocket"""
        if not self._running:
            return

        self._running = False
        logger.info("正在关闭 WS 管理器...")

        # 关闭 asyncio loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        # 等待线程结束
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

        self._connected = False
        logger.info(f"WS 管理器已关闭 | 统计: {self.stats}")

    # ==============================================================
    # 内部：asyncio 事件循环
    # ==============================================================
    def _run_loop(self):
        """在独立线程中运行 asyncio 事件循环"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_with_retry())
        except Exception as e:
            logger.error(f"WS 事件循环异常退出: {e}")
        finally:
            self._loop.close()
            self._connected = False
            logger.info("WS 事件循环已退出")

    async def _connect_with_retry(self):
        """带自动重连的连接主循环"""
        delay = self._reconnect_delay_init

        while self._running:
            try:
                await self._connect_and_listen()
                # 正常断开（如 stop()），不重连
                if not self._running:
                    break
                # 异常断开，重置延迟
                delay = self._reconnect_delay_init

            except websockets.exceptions.ConnectionClosed as e:
                self._connected = False
                self._reconnect_count += 1
                logger.warning(
                    f"WS 连接关闭 (code={e.code}): {e.reason} | "
                    f"第 {self._reconnect_count} 次重连，{delay:.1f}s 后...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._reconnect_delay_max)

            except Exception as e:
                self._connected = False
                self._reconnect_count += 1
                logger.error(
                    f"WS 连接异常: {e} | "
                    f"第 {self._reconnect_count} 次重连，{delay:.1f}s 后...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._reconnect_delay_max)

    def _build_ws_connect_kwargs(self) -> dict:
        """构建 websockets.connect 的关键字参数，含代理"""
        kwargs = {
            "ping_interval": self._ping_interval,
            "ping_timeout": self._ping_timeout,
            "close_timeout": 5,
            "max_size": 10 * 1024 * 1024,  # 10MB
        }

        # 代理支持：websockets >= 12.0 通过 additional_headers 不直接支持代理
        # 需要通过 HTTPS_PROXY / HTTP_PROXY 环境变量或 python-socks
        # 这里使用 websockets 自带的 open_timeout 并依赖系统代理环境变量
        if self._proxy_url:
            import os
            # 设置当前线程的环境变量让 websockets 走代理
            os.environ["HTTP_PROXY"] = self._proxy_url
            os.environ["HTTPS_PROXY"] = self._proxy_url
            os.environ["http_proxy"] = self._proxy_url
            os.environ["https_proxy"] = self._proxy_url
            logger.debug(f"WS 代理已设置: {self._proxy_url}")

        return kwargs

    async def _connect_and_listen(self):
        """建立连接并持续监听"""
        url = self._build_stream_url()
        connect_kwargs = self._build_ws_connect_kwargs()
        logger.info(f"连接 WebSocket: {url[:100]}...")

        async with websockets.connect(url, **connect_kwargs) as ws:
            self._ws = ws
            self._connected = True
            logger.info(f"WS 已连接 | 订阅 {len(self._symbols)} 个标的")

            async for raw_msg in ws:
                if not self._running:
                    break
                self._msg_count += 1
                self._last_msg_time = time.time()

                try:
                    msg = json.loads(raw_msg)
                    self._dispatch(msg)
                except json.JSONDecodeError:
                    logger.warning(f"WS 消息解析失败: {raw_msg[:100]}")
                except Exception as e:
                    logger.error(f"WS 消息处理异常: {e}")

    # ==============================================================
    # 流 URL 构建
    # ==============================================================
    def _build_stream_url(self) -> str:
        """
        构建 Binance Combined Streams URL
        格式: wss://stream.binance.com:9443/stream?streams=s1@kline_1h/s1@depth20@100ms/...
        """
        streams = []
        for sym in self._symbols:
            for interval in self._kline_intervals:
                streams.append(f"{sym}@kline_{interval}")
            if self._orderbook_enabled:
                streams.append(f"{sym}@depth20@100ms")

        stream_path = "/".join(streams)
        return f"{self._ws_base_url}/stream?streams={stream_path}"

    # ==============================================================
    # 消息分发
    # ==============================================================
    def _dispatch(self, msg: dict):
        """根据 stream 类型分发到对应处理器"""
        # Combined stream 格式: {"stream": "btcusdt@kline_1h", "data": {...}}
        stream = msg.get("stream", "")
        data = msg.get("data", msg)  # 兼容单 stream 模式

        if "@kline_" in stream:
            self._handle_kline(data)
        elif "@depth" in stream:
            symbol_lower = stream.split("@")[0]
            self._handle_orderbook(symbol_lower, data)

    # ==============================================================
    # K 线处理
    # ==============================================================
    def _handle_kline(self, data: dict):
        """
        处理 K 线推送
        关键：只有 kline.x == True（K 线已收盘）时才写入数据库
        未收盘的 K 线仅触发事件通知，不写库
        """
        k = data.get("k", {})
        if not k:
            return

        symbol = k.get("s", "").upper()  # e.g. "BTCUSDT"
        interval = k.get("i", "")        # e.g. "1h"
        is_closed = k.get("x", False)    # K 线是否已收盘

        kline_data = {
            "symbol": symbol,
            "interval": interval,
            "open_time": k["t"],
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "close_time": k["T"],
            "quote_volume": float(k["q"]),
            "trades_count": k["n"],
            "taker_buy_base": float(k["V"]),
            "taker_buy_quote": float(k["Q"]),
            "is_closed": is_closed,
        }

        # 只有已收盘 K 线写库
        if is_closed:
            self._kline_count += 1
            df = pd.DataFrame([kline_data])
            try:
                self.storage.upsert_klines(symbol, interval, df)
                logger.debug(
                    f"K线写入 {symbol}/{interval} | "
                    f"O={kline_data['open']:.2f} H={kline_data['high']:.2f} "
                    f"L={kline_data['low']:.2f} C={kline_data['close']:.2f} "
                    f"V={kline_data['volume']:.2f}")
            except Exception as e:
                logger.error(f"K线写入失败 {symbol}/{interval}: {e}")

        # 触发事件（包括未收盘的也通知，实时价格更新用）
        if self.event_bus:
            self.event_bus.publish(Event(
                type=EventType.KLINE_UPDATE,
                data=kline_data,
                source="ws_manager"))

        # 外部回调
        for cb in self._kline_callbacks:
            try:
                cb(symbol, interval, kline_data)
            except Exception as e:
                logger.error(f"K线回调异常: {e}")

    # ==============================================================
    # 订单簿处理
    # ==============================================================
    def _handle_orderbook(self, symbol_lower: str, data: dict):
        """
        处理 20 档深度推送
        计算 mid_price / spread / imbalance 并写入 storage
        """
        symbol = symbol_lower.upper()
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        if not bids or not asks:
            return

        self._orderbook_count += 1

        # 计算衍生指标
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        # 深度加总
        bid_depth = sum(float(b[1]) for b in bids)
        ask_depth = sum(float(a[1]) for a in asks)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)

        ob_snapshot = {
            "symbol": symbol,
            "timestamp": int(time.time() * 1000),
            "bids": json.dumps(bids),
            "asks": json.dumps(asks),
            "mid_price": mid_price,
            "spread": spread,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "imbalance": imbalance,
        }

        # 写入数据库
        try:
            self.storage.save_orderbook_snapshot(ob_snapshot)
        except Exception as e:
            logger.error(f"订单簿写入失败 {symbol}: {e}")

        # 触发事件
        if self.event_bus:
            self.event_bus.publish(Event(
                type=EventType.ORDERBOOK_UPDATE,
                data={
                    "symbol": symbol,
                    "mid_price": mid_price,
                    "spread": spread,
                    "imbalance": imbalance,
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                },
                source="ws_manager"))

        # 外部回调
        for cb in self._orderbook_callbacks:
            try:
                cb(symbol, ob_snapshot)
            except Exception as e:
                logger.error(f"订单簿回调异常: {e}")

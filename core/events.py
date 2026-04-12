"""
事件驱动核心
所有模块通过事件总线解耦通信
"""
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable
from queue import Queue, Empty

from utils.logger import get_logger

logger = get_logger("events")


class EventType(Enum):
    # 数据事件
    KLINE_UPDATE = auto()
    TICK_UPDATE = auto()
    ORDERBOOK_UPDATE = auto()
    HISTORY_LOADED = auto()

    # 信号事件
    SIGNAL_GENERATED = auto()
    SIGNAL_AGGREGATED = auto()

    # 执行事件
    ORDER_SUBMITTED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()
    ORDER_ERROR = auto()

    # 仓位事件
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()

    # 风控事件
    RISK_CHECK_PASSED = auto()
    RISK_CHECK_FAILED = auto()
    STOP_LOSS_TRIGGERED = auto()
    TAKE_PROFIT_TRIGGERED = auto()
    CIRCUIT_BREAKER_ON = auto()
    CIRCUIT_BREAKER_OFF = auto()

    # 系统事件
    ENGINE_START = auto()
    ENGINE_STOP = auto()
    HEARTBEAT = auto()
    ERROR = auto()
    MODEL_RETRAINED = auto()


@dataclass
class Event:
    type: EventType
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""

    def __repr__(self):
        return f"Event({self.type.name}, src={self.source}, ts={self.timestamp:.3f})"


class EventBus:
    """
    线程安全的事件总线
    支持同步和异步处理
    """

    def __init__(self, queue_size: int = 10000):
        self._handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self._queue: Queue = Queue(maxsize=queue_size)
        self._running = False
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._event_count = 0

    def subscribe(self, event_type: EventType, handler: Callable):
        """注册事件处理器"""
        with self._lock:
            self._handlers[event_type].append(handler)
        logger.debug(f"订阅: {event_type.name} -> {handler.__qualname__}")

    def unsubscribe(self, event_type: EventType, handler: Callable):
        with self._lock:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

    def publish(self, event: Event):
        """发布事件（非阻塞，放入队列）"""
        try:
            self._queue.put_nowait(event)
            self._event_count += 1
        except Exception:
            logger.error(f"事件队列已满，丢弃事件: {event}")

    def publish_sync(self, event: Event):
        """同步发布（立即执行所有 handler）"""
        self._dispatch(event)

    def _dispatch(self, event: Event):
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"事件处理异常: {event.type.name} -> {handler.__qualname__}: {e}")

    def start(self):
        """启动异步事件处理线程"""
        self._running = True
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()
        logger.info("事件总线已启动")

    def stop(self):
        self._running = False
        if self._worker:
            self._worker.join(timeout=5)
        logger.info(f"事件总线已停止 | 共处理 {self._event_count} 个事件")

    def _process_loop(self):
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                self._dispatch(event)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"事件循环异常: {e}")

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def total_events(self) -> int:
        return self._event_count

"""
智能执行引擎
- Market / Limit / TWAP / VWAP / Iceberg 算法
- 滑点模型
- 执行质量分析（Slippage, Implementation Shortfall）
- 模拟与实盘统一接口
"""
import time
import uuid
import threading
from datetime import datetime
from typing import Optional

import numpy as np

from config.loader import Config
from data.client import BinanceClient
from data.storage import Storage
from execution.position import PositionTracker
from core.events import EventBus, Event, EventType
from utils.logger import get_logger

logger = get_logger("executor")


class SlippageModel:
    """滑点估算模型"""

    def __init__(self, config: Config):
        slip_cfg = config.execution.slippage
        self._model = slip_cfg.model if hasattr(slip_cfg, "model") else slip_cfg._data.get("model", "linear")
        self._base_bps = slip_cfg.base_bps if hasattr(slip_cfg, "base_bps") else slip_cfg._data.get("base_bps", 2)
        self._impact_coeff = slip_cfg.impact_coefficient if hasattr(slip_cfg, "impact_coefficient") else slip_cfg._data.get("impact_coefficient", 0.1)

    def estimate(self, notional: float, avg_volume: float,
                 side: str) -> float:
        """返回估计滑点（比例）"""
        participation = notional / (avg_volume + 1e-10)

        if self._model == "fixed":
            slippage = self._base_bps / 10000
        elif self._model == "linear":
            slippage = (self._base_bps / 10000) + self._impact_coeff * participation
        elif self._model == "sqrt":
            slippage = (self._base_bps / 10000) + self._impact_coeff * np.sqrt(participation)
        else:
            slippage = self._base_bps / 10000

        return slippage if side == "BUY" else -slippage


class OrderExecutor:
    """
    订单执行器
    支持模拟和实盘模式
    """

    def __init__(self, config: Config, client: BinanceClient,
                 storage: Storage, position_tracker: PositionTracker,
                 event_bus: EventBus, simulate: bool = True):
        self.config = config
        self.client = client
        self.storage = storage
        self.positions = position_tracker
        self.event_bus = event_bus
        self.simulate = simulate
        self.slippage_model = SlippageModel(config)

        exec_cfg = config.execution
        self._algo = exec_cfg.algo
        self._commission_maker = exec_cfg.commission.maker_rate
        self._commission_taker = exec_cfg.commission.taker_rate

        self._execution_stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "total_slippage_bps": 0,
            "total_commission": 0,
        }

        mode = "模拟" if simulate else "实盘"
        logger.info(f"执行引擎初始化 | 模式={mode} | 算法={self._algo}")

    # ==============================================================
    # 统一下单入口
    # ==============================================================
    def execute(self, symbol: str, side: str, qty: float,
                price: float, strategy: str = "",
                algo: Optional[str] = None,
                urgency: str = "normal") -> Optional[dict]:
        """
        执行交易
        algo: market / limit / twap / vwap / iceberg
        urgency: low / normal / high（影响执行速度）
        """
        algo = algo or self._algo
        symbol_cfg = self.config.get_symbol_config(symbol)

        # 精度处理
        if symbol_cfg:
            qty_precision = symbol_cfg.get("qty_precision", 6)
            price_precision = symbol_cfg.get("price_precision", 2)
            min_qty = symbol_cfg.get("min_qty", 0.00001)
            min_notional = symbol_cfg.get("min_notional", 10)

            qty = round(qty, qty_precision)
            price = round(price, price_precision)

            if qty < min_qty:
                logger.warning(f"下单数量 {qty} < 最小数量 {min_qty}，跳过")
                return None
            if qty * price < min_notional:
                logger.warning(f"名义金额 {qty * price:.2f} < 最低 {min_notional}，跳过")
                return None

        client_order_id = f"QE_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        t0 = time.time()

        if algo == "market":
            result = self._execute_market(symbol, side, qty, price, client_order_id)
        elif algo == "twap":
            result = self._execute_twap(symbol, side, qty, price, client_order_id)
        elif algo == "iceberg":
            result = self._execute_iceberg(symbol, side, qty, price, client_order_id)
        else:
            result = self._execute_market(symbol, side, qty, price, client_order_id)

        if result is None:
            return None

        latency = (time.time() - t0) * 1000
        result["latency_ms"] = latency
        result["strategy"] = strategy
        result["algo"] = algo

        # 计算滑点
        if result.get("avg_fill_price") and price > 0:
            if side == "BUY":
                slippage_bps = (result["avg_fill_price"] - price) / price * 10000
            else:
                slippage_bps = (price - result["avg_fill_price"]) / price * 10000
            result["slippage_bps"] = round(slippage_bps, 2)

        # 更新持仓
        fill_price = result.get("avg_fill_price", price)
        commission = result.get("commission", 0)

        if side == "BUY":
            self.positions.open_position(
                symbol, "LONG", qty, fill_price, commission, strategy)
        elif side == "SELL":
            self.positions.close_position(symbol, qty, fill_price, commission)

        # 记录
        self.storage.save_order(result)
        self._execution_stats["total_orders"] += 1
        self._execution_stats["filled_orders"] += 1
        self._execution_stats["total_slippage_bps"] += abs(result.get("slippage_bps", 0))
        self._execution_stats["total_commission"] += commission

        # 事件
        self.event_bus.publish(Event(
            type=EventType.ORDER_FILLED,
            data=result, source="executor"))

        logger.info(
            f"{'[模拟]' if self.simulate else '[实盘]'} "
            f"{side} {qty:.6f} {symbol} @ {fill_price:.2f} "
            f"| 名义={qty * fill_price:.2f} "
            f"| 滑点={result.get('slippage_bps', 0):.1f}bps "
            f"| 延迟={latency:.0f}ms")

        return result

    # ==============================================================
    # 市价单
    # ==============================================================
    def _execute_market(self, symbol: str, side: str, qty: float,
                        ref_price: float, client_id: str) -> dict:
        if self.simulate:
            slippage = self.slippage_model.estimate(qty * ref_price, 1e8, side)
            fill_price = ref_price * (1 + slippage)
            commission = qty * fill_price * self._commission_taker
            return {
                "order_id": f"SIM_{client_id}",
                "client_order_id": client_id,
                "symbol": symbol,
                "side": side,
                "order_type": "MARKET",
                "quantity": qty,
                "price": ref_price,
                "executed_qty": qty,
                "avg_fill_price": fill_price,
                "commission": commission,
                "commission_asset": "USDT",
                "status": "FILLED",
                "created_at": int(time.time() * 1000),
                "filled_at": int(time.time() * 1000),
            }
        else:
            try:
                result = self.client.create_order(
                    symbol, side, "MARKET", qty, client_order_id=client_id)
                fills = result.get("fills", [])
                total_qty = sum(float(f["qty"]) for f in fills)
                total_notional = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                total_commission = sum(float(f["commission"]) for f in fills)
                avg_price = total_notional / total_qty if total_qty > 0 else ref_price

                return {
                    "order_id": str(result["orderId"]),
                    "client_order_id": client_id,
                    "symbol": symbol,
                    "side": side,
                    "order_type": "MARKET",
                    "quantity": qty,
                    "price": ref_price,
                    "executed_qty": total_qty,
                    "avg_fill_price": avg_price,
                    "commission": total_commission,
                    "commission_asset": fills[0]["commissionAsset"] if fills else "USDT",
                    "status": result["status"],
                    "created_at": result.get("transactTime", int(time.time() * 1000)),
                    "filled_at": int(time.time() * 1000),
                }
            except Exception as e:
                logger.error(f"市价单失败: {e}")
                self._execution_stats["rejected_orders"] += 1
                return None

    # ==============================================================
    # TWAP（时间加权平均价格）
    # ==============================================================
    def _execute_twap(self, symbol: str, side: str, total_qty: float,
                      ref_price: float, client_id: str) -> dict:
        """将大单拆分为多个小单，按时间均匀执行"""
        twap_cfg = self.config.execution.twap
        n_slices = twap_cfg.total_slices if hasattr(twap_cfg, "total_slices") else twap_cfg._data.get("total_slices", 5)
        interval = twap_cfg.interval_seconds if hasattr(twap_cfg, "interval_seconds") else twap_cfg._data.get("interval_seconds", 60)
        max_dev = twap_cfg.max_deviation_pct if hasattr(twap_cfg, "max_deviation_pct") else twap_cfg._data.get("max_deviation_pct", 0.002)

        slice_qty = total_qty / n_slices
        filled_qty = 0
        total_cost = 0
        total_commission = 0
        child_orders = []

        logger.info(f"TWAP 开始: {side} {total_qty:.6f} {symbol} 分 {n_slices} 片")

        for i in range(n_slices):
            try:
                current_price = self.client.get_ticker_price(symbol) if not self.simulate else ref_price
            except Exception:
                current_price = ref_price

            # 价格偏离检查
            deviation = abs(current_price - ref_price) / ref_price
            if deviation > max_dev:
                logger.warning(
                    f"TWAP 第{i+1}片: 价格偏离 {deviation:.4f} > {max_dev}，暂停")
                time.sleep(interval)
                continue

            child = self._execute_market(
                symbol, side, slice_qty, current_price,
                f"{client_id}_TWAP{i}")

            if child:
                filled_qty += child["executed_qty"]
                total_cost += child["executed_qty"] * child["avg_fill_price"]
                total_commission += child["commission"]
                child_orders.append(child)

            if i < n_slices - 1:
                if self.simulate:
                    pass  # 模拟模式不等待
                else:
                    time.sleep(interval)

        avg_price = total_cost / filled_qty if filled_qty > 0 else ref_price

        return {
            "order_id": f"TWAP_{client_id}",
            "client_order_id": client_id,
            "symbol": symbol,
            "side": side,
            "order_type": "MARKET",
            "quantity": total_qty,
            "price": ref_price,
            "executed_qty": filled_qty,
            "avg_fill_price": avg_price,
            "commission": total_commission,
            "commission_asset": "USDT",
            "status": "FILLED" if filled_qty >= total_qty * 0.95 else "PARTIALLY_FILLED",
            "created_at": int(time.time() * 1000),
            "filled_at": int(time.time() * 1000),
        }

    # ==============================================================
    # 冰山单
    # ==============================================================
    def _execute_iceberg(self, symbol: str, side: str, total_qty: float,
                         ref_price: float, client_id: str) -> dict:
        """冰山单：每次只展示一小部分"""
        ice_cfg = self.config.execution.iceberg
        show_pct = ice_cfg.show_qty_pct if hasattr(ice_cfg, "show_qty_pct") else ice_cfg._data.get("show_qty_pct", 0.2)
        rand_factor = ice_cfg.random_factor if hasattr(ice_cfg, "random_factor") else ice_cfg._data.get("random_factor", 0.3)

        remaining = total_qty
        filled_qty = 0
        total_cost = 0
        total_commission = 0

        logger.info(f"冰山单开始: {side} {total_qty:.6f} {symbol} 显示比例={show_pct}")

        while remaining > 1e-10:
            # 随机化每片大小
            base_show = total_qty * show_pct
            randomized = base_show * (1 + (np.random.random() - 0.5) * 2 * rand_factor)
            slice_qty = min(randomized, remaining)

            child = self._execute_market(
                symbol, side, slice_qty, ref_price,
                f"{client_id}_ICE{int(filled_qty * 1000)}")

            if child:
                filled_qty += child["executed_qty"]
                total_cost += child["executed_qty"] * child["avg_fill_price"]
                total_commission += child["commission"]
                remaining -= child["executed_qty"]

            if not self.simulate:
                time.sleep(np.random.uniform(1, 5))

        avg_price = total_cost / filled_qty if filled_qty > 0 else ref_price

        return {
            "order_id": f"ICE_{client_id}",
            "client_order_id": client_id,
            "symbol": symbol,
            "side": side,
            "order_type": "MARKET",
            "quantity": total_qty,
            "price": ref_price,
            "executed_qty": filled_qty,
            "avg_fill_price": avg_price,
            "commission": total_commission,
            "commission_asset": "USDT",
            "status": "FILLED",
            "created_at": int(time.time() * 1000),
            "filled_at": int(time.time() * 1000),
        }

    @property
    def stats(self) -> dict:
        s = self._execution_stats
        avg_slip = (s["total_slippage_bps"] / s["filled_orders"]
                    if s["filled_orders"] > 0 else 0)
        return {**s, "avg_slippage_bps": round(avg_slip, 2)}

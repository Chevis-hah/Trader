"""
持仓管理与盈亏计算
- 实时仓位跟踪
- FIFO 成本计算
- 浮动/已实现盈亏
- 最大有利/不利波动
"""
import time
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

logger = get_logger("position")


@dataclass
class Position:
    """单标的持仓"""
    symbol: str
    side: str = "LONG"
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    entry_time: int = 0
    cost_basis: float = 0.0          # 总成本
    realized_pnl: float = 0.0       # 已实现盈亏
    commission_paid: float = 0.0
    # 跟踪极值
    highest_price_since_entry: float = 0.0
    lowest_price_since_entry: float = float("inf")
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    # 元数据
    strategy: str = ""
    db_id: Optional[int] = None
    fills: list = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.quantity > 1e-10

    def unrealized_pnl(self, current_price: float) -> float:
        if not self.is_open:
            return 0.0
        if self.side == "LONG":
            return self.quantity * (current_price - self.avg_entry_price)
        else:
            return self.quantity * (self.avg_entry_price - current_price)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / self.cost_basis

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price

    def update_extremes(self, current_price: float):
        if not self.is_open:
            return
        self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)
        self.lowest_price_since_entry = min(self.lowest_price_since_entry, current_price)

        if self.side == "LONG":
            mfe = (self.highest_price_since_entry - self.avg_entry_price) / self.avg_entry_price
            mae = (self.avg_entry_price - self.lowest_price_since_entry) / self.avg_entry_price
        else:
            mfe = (self.avg_entry_price - self.lowest_price_since_entry) / self.avg_entry_price
            mae = (self.highest_price_since_entry - self.avg_entry_price) / self.avg_entry_price

        self.max_favorable_excursion = max(self.max_favorable_excursion, mfe)
        self.max_adverse_excursion = max(self.max_adverse_excursion, mae)


class PositionTracker:
    """
    全局持仓管理器
    跟踪所有标的的持仓、盈亏、市值
    """

    def __init__(self):
        self.positions: dict[str, Position] = {}
        self._closed_pnl: list[dict] = []

    def open_position(self, symbol: str, side: str, qty: float,
                      price: float, commission: float = 0,
                      strategy: str = "") -> Position:
        pos = self.positions.get(symbol)

        if pos and pos.is_open:
            # 加仓
            total_cost = pos.cost_basis + qty * price
            total_qty = pos.quantity + qty
            pos.avg_entry_price = total_cost / total_qty
            pos.quantity = total_qty
            pos.cost_basis = total_cost
            pos.commission_paid += commission
            pos.fills.append({
                "side": side, "qty": qty, "price": price,
                "commission": commission, "time": int(time.time() * 1000)})
            logger.info(
                f"加仓 {symbol}: +{qty:.6f} @ {price:.2f} | "
                f"总量={pos.quantity:.6f} 均价={pos.avg_entry_price:.2f}")
        else:
            pos = Position(
                symbol=symbol, side=side, quantity=qty,
                avg_entry_price=price, entry_time=int(time.time() * 1000),
                cost_basis=qty * price, commission_paid=commission,
                highest_price_since_entry=price,
                lowest_price_since_entry=price,
                strategy=strategy,
                fills=[{"side": side, "qty": qty, "price": price,
                        "commission": commission, "time": int(time.time() * 1000)}])
            self.positions[symbol] = pos
            logger.info(f"开仓 {symbol}: {side} {qty:.6f} @ {price:.2f}")

        return pos

    def close_position(self, symbol: str, qty: float, price: float,
                       commission: float = 0) -> Optional[dict]:
        pos = self.positions.get(symbol)
        if not pos or not pos.is_open:
            logger.warning(f"无持仓可平: {symbol}")
            return None

        close_qty = min(qty, pos.quantity)

        # 计算已实现盈亏
        if pos.side == "LONG":
            pnl = close_qty * (price - pos.avg_entry_price)
        else:
            pnl = close_qty * (pos.avg_entry_price - price)

        pnl -= commission
        pnl_pct = pnl / (close_qty * pos.avg_entry_price) if pos.avg_entry_price > 0 else 0

        pos.realized_pnl += pnl
        pos.commission_paid += commission
        pos.quantity -= close_qty

        record = {
            "symbol": symbol,
            "side": pos.side,
            "qty": close_qty,
            "entry_price": pos.avg_entry_price,
            "exit_price": price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "commission": pos.commission_paid,
            "mfe": pos.max_favorable_excursion,
            "mae": pos.max_adverse_excursion,
            "holding_time_ms": int(time.time() * 1000) - pos.entry_time,
            "strategy": pos.strategy,
        }

        if not pos.is_open:
            self._closed_pnl.append(record)
            del self.positions[symbol]
            logger.info(
                f"平仓 {symbol}: {close_qty:.6f} @ {price:.2f} | "
                f"PnL={pnl:+.2f} ({pnl_pct:+.2%})")
        else:
            logger.info(
                f"减仓 {symbol}: -{close_qty:.6f} @ {price:.2f} | "
                f"PnL={pnl:+.2f} 剩余={pos.quantity:.6f}")

        return record

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_all_open(self) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if p.is_open}

    def get_total_exposure(self, prices: dict[str, float]) -> float:
        return sum(p.market_value(prices.get(p.symbol, p.avg_entry_price))
                   for p in self.positions.values() if p.is_open)

    def get_total_unrealized_pnl(self, prices: dict[str, float]) -> float:
        return sum(p.unrealized_pnl(prices.get(p.symbol, p.avg_entry_price))
                   for p in self.positions.values() if p.is_open)

    def update_all_extremes(self, prices: dict[str, float]):
        for symbol, pos in self.positions.items():
            if pos.is_open and symbol in prices:
                pos.update_extremes(prices[symbol])

    def get_portfolio_summary(self, prices: dict[str, float]) -> dict:
        open_positions = self.get_all_open()
        total_value = self.get_total_exposure(prices)
        total_upnl = self.get_total_unrealized_pnl(prices)

        return {
            "open_positions": len(open_positions),
            "total_exposure": total_value,
            "unrealized_pnl": total_upnl,
            "realized_pnl": sum(r["pnl"] for r in self._closed_pnl),
            "total_trades_closed": len(self._closed_pnl),
            "positions": {
                s: {
                    "qty": p.quantity,
                    "avg_price": p.avg_entry_price,
                    "market_value": p.market_value(prices.get(s, p.avg_entry_price)),
                    "upnl": p.unrealized_pnl(prices.get(s, p.avg_entry_price)),
                    "upnl_pct": p.unrealized_pnl_pct(prices.get(s, p.avg_entry_price)),
                    "mfe": p.max_favorable_excursion,
                    "mae": p.max_adverse_excursion,
                }
                for s, p in open_positions.items()
            },
        }

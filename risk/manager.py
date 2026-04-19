"""
风控引擎
- Pre-trade 检查（下单前）
- Position-level 监控（持仓级）
- Portfolio-level 监控（组合级）
- 熔断机制
- VaR 计算
"""
import time
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from config.loader import Config
from execution.position import PositionTracker, Position
from core.events import EventBus, Event, EventType
from utils.logger import get_logger

logger = get_logger("risk")


class RiskManager:
    """
    生产级风控引擎
    三层防护: Pre-trade → Position → Portfolio
    """

    def __init__(self, config: Config, position_tracker: PositionTracker,
                 event_bus: EventBus, initial_capital: float):
        self.config = config
        self.positions = position_tracker
        self.event_bus = event_bus
        self.initial_capital = initial_capital

        risk_cfg = config.risk
        self._pre = risk_cfg.pre_trade
        self._pos = risk_cfg.position
        self._port = risk_cfg.portfolio
        self._cb = risk_cfg.circuit_breaker

        # 状态
        self.peak_nav = initial_capital
        self.daily_start_nav = initial_capital
        self.daily_start_date = datetime.now(timezone.utc).date()
        self.weekly_start_nav = initial_capital
        self.weekly_start_date = datetime.now(timezone.utc).date()

        # 熔断
        self.circuit_breaker_active = False
        self.circuit_breaker_until: Optional[datetime] = None
        self.consecutive_losses = 0

        # 下单追踪
        self._order_times: deque = deque()
        self._minute_orders: deque = deque()

        # 历史收益率（用于 VaR）
        self._returns_history: deque = deque(maxlen=5000)

        logger.info(f"风控引擎初始化 | 初始资金={initial_capital:.2f}")

    # ==============================================================
    # Pre-trade 检查
    # ==============================================================
    def pre_trade_check(self, symbol: str, side: str, qty: float,
                        price: float, current_nav: float,
                        prices: dict[str, float]) -> tuple[bool, str]:
        """
        下单前风控检查
        返回 (是否放行, 原因)
        """
        # 0. 熔断检查
        if self.circuit_breaker_active:
            if self.circuit_breaker_until and datetime.now(timezone.utc) < self.circuit_breaker_until:
                return False, f"熔断中，冷却至 {self.circuit_breaker_until.isoformat()}"
            else:
                self._deactivate_circuit_breaker()

        notional = qty * price

        # 1. 单笔限额
        max_notional = self._get("pre_trade", "max_single_order_notional", 5000)
        if notional > max_notional:
            return False, f"单笔名义 {notional:.2f} 超限 {max_notional}"

        max_pct = self._get("pre_trade", "max_single_order_pct", 0.10)
        if notional / current_nav > max_pct:
            return False, f"单笔占比 {notional/current_nav:.2%} 超限 {max_pct:.0%}"

        # 2. 价格偏离检查
        if side == "BUY":
            try:
                market_price = prices.get(symbol, price)
                dev = abs(price - market_price) / market_price
                limit = self._get("pre_trade", "price_deviation_limit_pct", 0.02)
                if dev > limit:
                    return False, f"价格偏离 {dev:.4f} > {limit}"
            except Exception:
                pass

        # 3. 下单频率
        now = time.time()
        cutoff_minute = now - 60
        cutoff_hour = now - 3600
        while self._minute_orders and self._minute_orders[0] < cutoff_minute:
            self._minute_orders.popleft()
        while self._order_times and self._order_times[0] < cutoff_hour:
            self._order_times.popleft()

        max_per_min = self._get("pre_trade", "max_orders_per_minute", 10)
        if len(self._minute_orders) >= max_per_min:
            return False, f"每分钟下单 {len(self._minute_orders)} 超限 {max_per_min}"

        max_per_hour = self._get("pre_trade", "max_orders_per_hour", 60)
        if len(self._order_times) >= max_per_hour:
            return False, f"每小时下单 {len(self._order_times)} 超限 {max_per_hour}"

        # 4. 仓位检查（仅买入）
        if side == "BUY":
            # 单标的仓位
            pos = self.positions.get_position(symbol)
            existing_value = pos.market_value(price) if pos and pos.is_open else 0
            new_total = existing_value + notional
            max_pos = self._get("position", "max_position_pct", 0.25)
            if new_total / current_nav > max_pos:
                return False, f"单标的仓位 {new_total/current_nav:.2%} 超限 {max_pos:.0%}"

            # 总敞口
            total_exposure = self.positions.get_total_exposure(prices)
            max_exposure = self._get("position", "max_total_exposure_pct", 0.80)
            if (total_exposure + notional) / current_nav > max_exposure:
                return False, f"总敞口 {(total_exposure+notional)/current_nav:.2%} 超限 {max_exposure:.0%}"

            # 集中度
            max_conc = self._get("position", "max_concentration", 0.40)
            if total_exposure > 0 and new_total / (total_exposure + notional) > max_conc:
                return False, f"集中度过高"

        return True, "通过"

    def record_order(self):
        """记录一次下单时间"""
        now = time.time()
        self._order_times.append(now)
        self._minute_orders.append(now)

    # ==============================================================
    # Position-level 监控
    # ==============================================================
    def check_positions(self, prices: dict[str, float]) -> list[dict]:
        """
        检查所有持仓的止损/止盈/超时
        返回需要平仓的信号列表
        """
        actions = []
        stop_loss = self._get("position", "stop_loss_pct", 0.03)
        trailing_stop = self._get("position", "trailing_stop_pct", 0.05)
        take_profit = self._get("position", "take_profit_pct", 0.08)
        max_hold = self._get("position", "max_holding_hours", 168) * 3600 * 1000

        for symbol, pos in self.positions.get_all_open().items():
            current_price = prices.get(symbol)
            if current_price is None:
                continue

            pos.update_extremes(current_price)
            pnl_pct = pos.unrealized_pnl_pct(current_price)

            # 固定止损
            if pnl_pct < -stop_loss:
                actions.append({
                    "symbol": symbol, "action": "CLOSE",
                    "reason": f"止损 {pnl_pct:.2%} < -{stop_loss:.0%}",
                    "qty": pos.quantity, "price": current_price})
                continue

            # 移动止损：从最高盈利回撤超过 trailing_stop
            if pos.max_favorable_excursion > 0.01:  # 只在曾经盈利后启用
                drawback = pos.max_favorable_excursion - pnl_pct
                if drawback > trailing_stop and pnl_pct > 0:
                    actions.append({
                        "symbol": symbol, "action": "CLOSE",
                        "reason": f"移动止损 回撤={drawback:.2%} MFE={pos.max_favorable_excursion:.2%}",
                        "qty": pos.quantity, "price": current_price})
                    continue

            # 止盈
            if pnl_pct > take_profit:
                actions.append({
                    "symbol": symbol, "action": "CLOSE",
                    "reason": f"止盈 {pnl_pct:.2%} > +{take_profit:.0%}",
                    "qty": pos.quantity, "price": current_price})
                continue

            # 超时平仓
            holding_ms = int(time.time() * 1000) - pos.entry_time
            if holding_ms > max_hold:
                actions.append({
                    "symbol": symbol, "action": "CLOSE",
                    "reason": f"持仓超时 {holding_ms / 3600000:.1f}h",
                    "qty": pos.quantity, "price": current_price})

        return actions

    # ==============================================================
    # Portfolio-level 监控
    # ==============================================================
    def check_portfolio(self, current_nav: float) -> tuple[bool, str]:
        """
        组合层面风控检查
        返回 (是否正常, 描述)
        """
        self._check_period_reset(current_nav)
        self.peak_nav = max(self.peak_nav, current_nav)

        # 日内亏损
        daily_pnl = (current_nav - self.daily_start_nav) / self.daily_start_nav
        max_daily = self._get("portfolio", "max_daily_loss_pct", 0.03)
        if daily_pnl < -max_daily:
            self._activate_circuit_breaker(f"日内亏损 {daily_pnl:.2%}")
            return False, f"日内亏损 {daily_pnl:.2%} 触发熔断"

        # 周亏损
        weekly_pnl = (current_nav - self.weekly_start_nav) / self.weekly_start_nav
        max_weekly = self._get("portfolio", "max_weekly_loss_pct", 0.06)
        if weekly_pnl < -max_weekly:
            self._activate_circuit_breaker(f"周亏损 {weekly_pnl:.2%}")
            return False, f"周亏损 {weekly_pnl:.2%} 触发熔断"

        # 最大回撤
        drawdown = (self.peak_nav - current_nav) / self.peak_nav
        max_dd = self._get("portfolio", "max_drawdown_pct", 0.12)
        if drawdown > max_dd:
            self._activate_circuit_breaker(f"最大回撤 {drawdown:.2%}")
            return False, f"最大回撤 {drawdown:.2%} 触发熔断"

        # VaR 检查
        if len(self._returns_history) > 100:
            returns = np.array(self._returns_history)
            var_conf = self._get("portfolio", "var_confidence", 0.95)
            var_limit = self._get("portfolio", "var_limit_pct", 0.05)
            var = -np.percentile(returns, (1 - var_conf) * 100)
            if var > var_limit:
                logger.warning(f"VaR({var_conf:.0%}) = {var:.4f} > 限额 {var_limit:.2%}")

        return True, f"日PnL={daily_pnl:+.2%} DD={drawdown:.2%}"

    def record_return(self, ret: float):
        """记录周期收益率（用于 VaR）"""
        self._returns_history.append(ret)

    def record_trade_result(self, pnl: float):
        """记录交易结果（用于连续亏损统计）"""
        if pnl < 0:
            self.consecutive_losses += 1
            max_consec = self._get("circuit_breaker", "consecutive_losses", 5)
            if self.consecutive_losses >= max_consec:
                self._activate_circuit_breaker(f"连续亏损 {self.consecutive_losses} 笔")
        else:
            self.consecutive_losses = 0

    # ==============================================================
    # 熔断
    # ==============================================================
    def _activate_circuit_breaker(self, reason: str):
        cooldown = self._get("circuit_breaker", "cooldown_hours", 4)
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now(timezone.utc) + timedelta(hours=cooldown)
        logger.critical(f"🚨 熔断触发: {reason} | 冷却至 {self.circuit_breaker_until}")
        self.event_bus.publish(Event(
            type=EventType.CIRCUIT_BREAKER_ON,
            data={"reason": reason, "until": self.circuit_breaker_until.isoformat()},
            source="risk"))

    def _deactivate_circuit_breaker(self):
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        self.consecutive_losses = 0
        logger.info("熔断解除")
        self.event_bus.publish(Event(
            type=EventType.CIRCUIT_BREAKER_OFF, source="risk"))

    # ==============================================================
    # 辅助
    # ==============================================================
    def _check_period_reset(self, nav: float):
        today = datetime.now(timezone.utc).date()
        if today != self.daily_start_date:
            self.daily_start_nav = nav
            self.daily_start_date = today
        # 周一重置
        if today.weekday() == 0 and today != self.weekly_start_date:
            self.weekly_start_nav = nav
            self.weekly_start_date = today

    def _get(self, section: str, key: str, default=None):
        try:
            sec = getattr(self.config.risk, section)
            val = getattr(sec, key)
            return val
        except AttributeError:
            try:
                return self.config.risk._data[section][key]
            except (KeyError, TypeError):
                return default

    def get_risk_snapshot(self, current_nav: float,
                         prices: dict[str, float]) -> dict:
        """当前风控状态快照"""
        self.peak_nav = max(self.peak_nav, current_nav)
        drawdown = (self.peak_nav - current_nav) / self.peak_nav
        daily_pnl = (current_nav - self.daily_start_nav) / self.daily_start_nav
        exposure = self.positions.get_total_exposure(prices)

        snapshot = {
            "nav": current_nav,
            "peak_nav": self.peak_nav,
            "drawdown_pct": drawdown,
            "daily_pnl_pct": daily_pnl,
            "total_exposure": exposure,
            "exposure_pct": exposure / current_nav if current_nav > 0 else 0,
            "unrealized_pnl": self.positions.get_total_unrealized_pnl(prices),
            "circuit_breaker": self.circuit_breaker_active,
            "consecutive_losses": self.consecutive_losses,
            "open_positions": len(self.positions.get_all_open()),
        }

        if len(self._returns_history) > 50:
            returns = np.array(self._returns_history)
            snapshot["var_95"] = float(-np.percentile(returns, 5))
            snapshot["var_99"] = float(-np.percentile(returns, 1))

        return snapshot
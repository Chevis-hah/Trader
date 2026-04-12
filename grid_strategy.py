"""
网格交易策略（Grid Trading）

核心逻辑：
  - 在震荡行情中自动低买高卖
  - 网格间距基于 ATR 动态调整（不是固定比例）
  - 只在 ADX < 25 的震荡行情中激活（避免趋势行情被套）
  - 每个网格持仓独立管理
  - 总止损：整个网格亏 5% 全部平仓

适用场景：
  - 月化 2-5%，回撤小
  - 与趋势策略互补：趋势策略在震荡行情中小亏，网格策略在震荡行情中赚钱
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GridLevel:
    """单个网格层级"""
    price: float = 0.0
    side: str = ""       # BUY / SELL
    filled: bool = False
    fill_price: float = 0.0
    fill_time: int = 0
    quantity: float = 0.0


@dataclass
class GridState:
    """网格状态"""
    active: bool = False
    center_price: float = 0.0
    grid_spacing: float = 0.0
    levels: list = field(default_factory=list)
    total_invested: float = 0.0
    total_returned: float = 0.0
    realized_pnl: float = 0.0
    open_qty: float = 0.0
    avg_cost: float = 0.0
    grid_stop_loss: float = 0.0
    creation_time: int = 0


class GridTradingStrategy:
    """
    网格交易策略

    参数:
      n_grids:         网格层数（上下各 n 层，默认 5）
      grid_atr_mult:   网格间距 = ATR × grid_atr_mult（默认 0.5）
      qty_per_grid_pct: 每格投入占总资金比例（默认 0.04 = 4%）
      max_total_pct:   网格总投入上限（默认 0.40 = 40%）
      stop_loss_pct:   总止损比例（默认 0.05 = 5%）
      adx_threshold:   ADX 阈值，低于此值才激活网格（默认 25）
      atr_period:      ATR 周期（默认 20）
      adx_period:      ADX 周期（默认 14）
    """

    def __init__(self,
                 n_grids: int = 5,
                 grid_atr_mult: float = 0.5,
                 qty_per_grid_pct: float = 0.04,
                 max_total_pct: float = 0.40,
                 stop_loss_pct: float = 0.05,
                 adx_threshold: float = 25.0,
                 atr_period: int = 20,
                 adx_period: int = 14):
        self.n_grids = n_grids
        self.grid_atr_mult = grid_atr_mult
        self.qty_per_grid_pct = qty_per_grid_pct
        self.max_total_pct = max_total_pct
        self.stop_loss_pct = stop_loss_pct
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.adx_period = adx_period

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 ATR 和 ADX"""
        out = df.copy()
        close = out["close"]

        # ATR
        hl = out["high"] - out["low"]
        hc = (out["high"] - close.shift(1)).abs()
        lc = (out["low"] - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        out["atr"] = tr.rolling(self.atr_period).mean()

        # ADX 简化计算
        up = out["high"] - out["high"].shift(1)
        down = out["low"].shift(1) - out["low"]
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)

        atr_val = out["atr"].values
        plus_di = np.zeros(len(df))
        minus_di = np.zeros(len(df))

        for i in range(self.adx_period, len(df)):
            atr_sum = atr_val[i]
            if atr_sum > 0:
                plus_di[i] = 100 * pd.Series(plus_dm).iloc[i-self.adx_period:i].mean() / atr_sum
                minus_di[i] = 100 * pd.Series(minus_dm).iloc[i-self.adx_period:i].mean() / atr_sum

        dx = np.where(
            (plus_di + minus_di) > 0,
            100 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
            0)
        out["adx"] = pd.Series(dx, index=df.index).rolling(self.adx_period).mean()

        # 布林带（用于判断震荡区间）
        out["bb_mid"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        out["bb_upper"] = out["bb_mid"] + 2 * bb_std
        out["bb_lower"] = out["bb_mid"] - 2 * bb_std
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]

        return out

    def should_activate(self, row: pd.Series) -> bool:
        """判断是否应该激活网格（震荡行情）"""
        adx = row.get("adx", 30)
        return not pd.isna(adx) and adx < self.adx_threshold

    def create_grid(self, center_price: float, atr: float,
                    capital: float, timestamp: int) -> GridState:
        """以当前价格为中心创建网格"""
        spacing = atr * self.grid_atr_mult
        if spacing <= 0:
            return GridState()

        qty_per_grid = (capital * self.qty_per_grid_pct) / center_price
        levels = []

        # 下方买入网格
        for i in range(1, self.n_grids + 1):
            price = center_price - i * spacing
            levels.append(GridLevel(
                price=price, side="BUY", quantity=qty_per_grid))

        # 上方卖出网格（匹配买入网格）
        for i in range(1, self.n_grids + 1):
            price = center_price + i * spacing
            levels.append(GridLevel(
                price=price, side="SELL", quantity=qty_per_grid))

        # 总止损价
        grid_stop = center_price * (1 - self.stop_loss_pct)

        return GridState(
            active=True,
            center_price=center_price,
            grid_spacing=spacing,
            levels=levels,
            grid_stop_loss=grid_stop,
            creation_time=timestamp)

    def process_bar(self, row: pd.Series, state: GridState,
                    capital: float) -> list[dict]:
        """
        处理一根 K 线，检查是否有网格被触发

        返回: [{"action": "BUY"/"SELL", "price": ..., "qty": ..., "pnl": ...}, ...]
        """
        if not state.active:
            return []

        actions = []
        low = row["low"]
        high = row["high"]
        close = row["close"]
        timestamp = row.get("open_time", 0)

        # 总止损检查
        if low <= state.grid_stop_loss:
            # 全部平仓
            if state.open_qty > 0:
                pnl = state.open_qty * (state.grid_stop_loss - state.avg_cost)
                actions.append({
                    "action": "GRID_STOP_LOSS",
                    "price": state.grid_stop_loss,
                    "qty": state.open_qty,
                    "pnl": pnl,
                    "time": timestamp})
                state.realized_pnl += pnl
                state.open_qty = 0
                state.avg_cost = 0
            state.active = False
            return actions

        for level in state.levels:
            if level.filled:
                continue

            # 买入网格：价格下穿
            if level.side == "BUY" and low <= level.price:
                level.filled = True
                level.fill_price = level.price
                level.fill_time = timestamp

                # 更新持仓
                total_cost = state.avg_cost * state.open_qty + level.price * level.quantity
                state.open_qty += level.quantity
                state.avg_cost = total_cost / state.open_qty if state.open_qty > 0 else 0
                state.total_invested += level.price * level.quantity

                actions.append({
                    "action": "GRID_BUY",
                    "price": level.price,
                    "qty": level.quantity,
                    "pnl": 0,
                    "time": timestamp})

            # 卖出网格：价格上穿（前提是有持仓）
            elif level.side == "SELL" and high >= level.price and state.open_qty > 0:
                sell_qty = min(level.quantity, state.open_qty)
                if sell_qty <= 0:
                    continue

                level.filled = True
                level.fill_price = level.price
                level.fill_time = timestamp

                pnl = sell_qty * (level.price - state.avg_cost)
                state.open_qty -= sell_qty
                state.realized_pnl += pnl
                state.total_returned += level.price * sell_qty

                actions.append({
                    "action": "GRID_SELL",
                    "price": level.price,
                    "qty": sell_qty,
                    "pnl": pnl,
                    "time": timestamp})

        return actions

    def get_unrealized_pnl(self, state: GridState, current_price: float) -> float:
        """计算网格未实现盈亏"""
        if state.open_qty <= 0:
            return 0.0
        return state.open_qty * (current_price - state.avg_cost)

"""
组合优化
- 等权 / 均值方差 / 风险平价 / 最大夏普
- 考虑换手率约束
- 目标波动率缩放
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional

from config.loader import Config
from utils.logger import get_logger

logger = get_logger("portfolio")


class PortfolioOptimizer:
    """组合权重优化器"""

    def __init__(self, config: Config):
        self.config = config
        port_cfg = config.portfolio
        self._method = port_cfg.optimization.method
        self._max_weight = port_cfg.constraints.max_weight_per_asset
        self._min_weight = port_cfg.constraints.min_weight_per_asset
        self._target_vol = port_cfg.constraints.target_volatility

    def optimize(self, expected_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 current_weights: Optional[np.ndarray] = None,
                 method: Optional[str] = None) -> np.ndarray:
        """
        计算最优权重
        expected_returns: 各标的预期收益向量
        cov_matrix: 协方差矩阵
        current_weights: 当前权重（用于换手率约束）
        """
        method = method or self._method
        n = len(expected_returns)

        if method == "equal_weight":
            weights = np.ones(n) / n

        elif method == "risk_parity":
            weights = self._risk_parity(cov_matrix)

        elif method == "mean_variance":
            weights = self._mean_variance(expected_returns, cov_matrix)

        elif method == "max_sharpe":
            weights = self._max_sharpe(expected_returns, cov_matrix)

        else:
            weights = np.ones(n) / n

        # 约束裁剪
        weights = np.clip(weights, self._min_weight, self._max_weight)
        weights = weights / weights.sum()  # 重新归一化

        # 目标波动率缩放
        if self._target_vol > 0:
            port_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(8766)  # 年化
            if port_vol > 0:
                scale = self._target_vol / port_vol
                weights = weights * min(scale, 1.0)  # 只缩小，不放大

        return weights

    def _risk_parity(self, cov: np.ndarray) -> np.ndarray:
        """风险平价：使每个标的对组合风险的贡献相等"""
        n = cov.shape[0]
        x0 = np.ones(n) / n

        def risk_budget_obj(w):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol == 0:
                return 0
            marginal = cov @ w
            risk_contrib = w * marginal / port_vol
            target = port_vol / n
            return np.sum((risk_contrib - target) ** 2)

        bounds = [(0.01, self._max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

        result = minimize(risk_budget_obj, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        logger.warning("风险平价优化未收敛，使用等权")
        return x0

    def _mean_variance(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """最小方差（考虑收益约束）"""
        n = len(mu)
        x0 = np.ones(n) / n

        def objective(w):
            return w @ cov @ w  # 最小化方差

        bounds = [(0.0, self._max_weight)] * n
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: w @ mu},  # 收益 > 0
        ]

        result = minimize(objective, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else x0

    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """最大夏普比率"""
        n = len(mu)
        x0 = np.ones(n) / n

        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            return -ret / (vol + 1e-10)

        bounds = [(0.0, self._max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

        result = minimize(neg_sharpe, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else x0

    def calculate_rebalance_orders(
        self, current_weights: np.ndarray,
        target_weights: np.ndarray,
        symbols: list[str],
        portfolio_value: float,
        prices: dict[str, float]
    ) -> list[dict]:
        """
        计算再平衡需要的订单
        返回: [{"symbol": ..., "side": ..., "qty": ..., "notional": ...}, ...]
        """
        orders = []
        for i, symbol in enumerate(symbols):
            delta = target_weights[i] - current_weights[i]
            notional = abs(delta) * portfolio_value
            price = prices.get(symbol, 0)
            if price == 0 or notional < 10:  # 最小名义金额
                continue

            qty = notional / price
            orders.append({
                "symbol": symbol,
                "side": "BUY" if delta > 0 else "SELL",
                "qty": qty,
                "notional": notional,
                "weight_change": delta,
            })

        return orders

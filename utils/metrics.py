"""
专业绩效指标计算
涵盖所有基金业绩评估标准指标
"""
import numpy as np
import pandas as pd
from typing import Optional


def annualization_factor(frequency: str = "1h") -> float:
    """获取年化因子"""
    factors = {
        "1m": 525960, "5m": 105192, "15m": 35064,
        "1h": 8766, "4h": 2191.5, "1d": 365.25,
    }
    return factors.get(frequency, 8766)


def total_return(equity_curve: np.ndarray) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return (equity_curve[-1] / equity_curve[0]) - 1


def cagr(equity_curve: np.ndarray, periods_per_year: float = 8766) -> float:
    """年化复合增长率"""
    n = len(equity_curve)
    if n < 2:
        return 0.0
    total = equity_curve[-1] / equity_curve[0]
    years = n / periods_per_year
    return total ** (1 / years) - 1 if years > 0 else 0.0


def max_drawdown(equity_curve: np.ndarray) -> tuple[float, int, int]:
    """最大回撤及其起止位置"""
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / peak
    max_dd = dd.max()
    end_idx = int(dd.argmax())
    start_idx = int(equity_curve[:end_idx + 1].argmax()) if end_idx > 0 else 0
    return float(max_dd), start_idx, end_idx


def drawdown_duration(equity_curve: np.ndarray) -> int:
    """最长回撤持续期（bar 数）"""
    if len(equity_curve) < 2:
        return 0
    peak = np.maximum.accumulate(equity_curve)
    in_drawdown = equity_curve < peak
    max_dur = 0
    current = 0
    for v in in_drawdown:
        if v:
            current += 1
            max_dur = max(max_dur, current)
        else:
            current = 0
    return max_dur


def sharpe_ratio(returns: np.ndarray, rf_rate: float = 0.0,
                 periods_per_year: float = 8766) -> float:
    """年化夏普比率"""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf_rate / periods_per_year
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, rf_rate: float = 0.0,
                  periods_per_year: float = 8766) -> float:
    """Sortino 比率（只计算下行波动）"""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf_rate / periods_per_year
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def calmar_ratio(equity_curve: np.ndarray,
                 periods_per_year: float = 8766) -> float:
    """Calmar 比率 = CAGR / MaxDrawdown"""
    if len(equity_curve) < 2:
        return 0.0
    mdd, _, _ = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    return cagr(equity_curve, periods_per_year) / mdd


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Omega 比率"""
    if len(returns) == 0:
        return 0.0
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(pnl_series: np.ndarray) -> float:
    if len(pnl_series) == 0:
        return 0.0
    wins = (pnl_series > 0).sum()
    total = (pnl_series != 0).sum()
    return float(wins / total) if total > 0 else 0.0


def profit_factor(pnl_series: np.ndarray) -> float:
    """盈亏比"""
    if len(pnl_series) == 0:
        return 0.0
    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss = abs(pnl_series[pnl_series < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def expectancy(pnl_series: np.ndarray) -> float:
    """期望收益 = 平均盈利 * 胜率 - 平均亏损 * 败率"""
    if len(pnl_series) == 0:
        return 0.0
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    if len(wins) == 0 and len(losses) == 0:
        return 0.0
    total = len(pnl_series[pnl_series != 0])
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    wr = len(wins) / total if total > 0 else 0
    return float(avg_win * wr - avg_loss * (1 - wr))


def value_at_risk(returns: np.ndarray, confidence: float = 0.95,
                  method: str = "historical") -> float:
    """VaR 计算"""
    if len(returns) == 0:
        return 0.0
    if method == "historical":
        return float(-np.percentile(returns, (1 - confidence) * 100))
    elif method == "parametric":
        from scipy.stats import norm
        std = returns.std()
        if std == 0 or np.isnan(std):
            return 0.0
        z = norm.ppf(1 - confidence)
        return float(-(returns.mean() + z * std))
    return 0.0


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """条件 VaR（Expected Shortfall）"""
    if len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


def information_coefficient(predictions: np.ndarray,
                            actual_returns: np.ndarray) -> float:
    """信息系数 IC（Spearman rank correlation）"""
    if len(predictions) < 3 or len(actual_returns) < 3:
        return 0.0
    from scipy.stats import spearmanr
    corr, _ = spearmanr(predictions, actual_returns)
    return float(corr) if not np.isnan(corr) else 0.0


def ic_ir(ic_series: np.ndarray) -> float:
    """IC 信息比率 = IC均值 / IC标准差"""
    if len(ic_series) < 2 or ic_series.std() == 0:
        return 0.0
    return float(ic_series.mean() / ic_series.std())


def turnover(weights_before: np.ndarray, weights_after: np.ndarray) -> float:
    """换手率"""
    return float(np.abs(weights_after - weights_before).sum() / 2)


def generate_report(equity_curve: np.ndarray, trades_pnl: np.ndarray,
                    frequency: str = "1h") -> dict:
    """生成完整绩效报告"""
    ann = annualization_factor(frequency)

    # 防空数组
    if len(equity_curve) < 2:
        returns = np.array([0.0])
    else:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        # 清理 NaN/Inf
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            returns = np.array([0.0])

    mdd, mdd_start, mdd_end = max_drawdown(equity_curve)

    # 安全计算统计量
    def _safe_stat(func, *args, default=0.0):
        try:
            val = func(*args)
            return val if np.isfinite(val) else default
        except Exception:
            return default

    report = {
        "total_return": _safe_stat(total_return, equity_curve),
        "cagr": _safe_stat(cagr, equity_curve, ann),
        "max_drawdown": mdd,
        "max_dd_start": mdd_start,
        "max_dd_end": mdd_end,
        "max_dd_duration_bars": drawdown_duration(equity_curve),
        "sharpe_ratio": _safe_stat(sharpe_ratio, returns, 0.0, ann),
        "sortino_ratio": _safe_stat(sortino_ratio, returns, 0.0, ann),
        "calmar_ratio": _safe_stat(calmar_ratio, equity_curve, ann),
        "omega_ratio": _safe_stat(omega_ratio, returns),
        "volatility_annual": _safe_stat(lambda r: float(r.std() * np.sqrt(ann)), returns),
        "skewness": _safe_stat(lambda r: float(pd.Series(r).skew()), returns),
        "kurtosis": _safe_stat(lambda r: float(pd.Series(r).kurtosis()), returns),
        "var_95": _safe_stat(value_at_risk, returns, 0.95),
        "cvar_95": _safe_stat(conditional_var, returns, 0.95),
        "total_trades": int((trades_pnl != 0).sum()) if len(trades_pnl) > 0 else 0,
        "win_rate": win_rate(trades_pnl),
        "profit_factor": profit_factor(trades_pnl),
        "expectancy": expectancy(trades_pnl),
        "avg_win": float(trades_pnl[trades_pnl > 0].mean()) if len(trades_pnl) > 0 and (trades_pnl > 0).any() else 0,
        "avg_loss": float(trades_pnl[trades_pnl < 0].mean()) if len(trades_pnl) > 0 and (trades_pnl < 0).any() else 0,
        "best_trade": float(trades_pnl.max()) if len(trades_pnl) > 0 else 0,
        "worst_trade": float(trades_pnl.min()) if len(trades_pnl) > 0 else 0,
    }
    return report


def format_report(report: dict) -> str:
    """格式化绩效报告为可读文本"""
    lines = [
        "=" * 65,
        "                    绩 效 报 告",
        "=" * 65,
        f"  总收益率:          {report['total_return']:>+10.2%}",
        f"  年化收益 (CAGR):   {report['cagr']:>+10.2%}",
        f"  年化波动率:        {report['volatility_annual']:>10.2%}",
        f"  最大回撤:          {report['max_drawdown']:>10.2%}",
        f"  最长回撤期:        {report['max_dd_duration_bars']:>10d} bars",
        "-" * 65,
        f"  夏普比率:          {report['sharpe_ratio']:>10.3f}",
        f"  Sortino 比率:      {report['sortino_ratio']:>10.3f}",
        f"  Calmar 比率:       {report['calmar_ratio']:>10.3f}",
        f"  Omega 比率:        {report['omega_ratio']:>10.3f}",
        f"  VaR (95%):         {report['var_95']:>10.4f}",
        f"  CVaR (95%):        {report['cvar_95']:>10.4f}",
        "-" * 65,
        f"  总交易数:          {report['total_trades']:>10d}",
        f"  胜率:              {report['win_rate']:>10.1%}",
        f"  盈亏比:            {report['profit_factor']:>10.2f}",
        f"  期望收益:          {report['expectancy']:>+10.4f}",
        f"  平均盈利:          {report['avg_win']:>+10.4f}",
        f"  平均亏损:          {report['avg_loss']:>+10.4f}",
        f"  最大单笔盈利:      {report['best_trade']:>+10.4f}",
        f"  最大单笔亏损:      {report['worst_trade']:>+10.4f}",
        "=" * 65,
    ]
    return "\n".join(lines)

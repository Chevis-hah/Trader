# analysis/ - 诊断与分析模块

## 职责

数据质量诊断、策略性能分析、参数敏感性扫描。

> v2.3 仓库级变更与 WF 脚本调整见根目录 `SYNC_UPDATE_LOG.md`。

## 模块组成

### data_diagnostic.py - 数据诊断

四个诊断工具:

1. **`generate_equity_curve()`**: 从回测结果生成逐 bar 权益曲线 + 回撤 + 滚动 Sharpe
2. **`analyze_correlation()`**: BTC/ETH 滚动收益率相关性 (30/60/120/240 bar 窗口)
3. **`analyze_by_regime()`**: 按 ADX/NATR/标的 分桶统计交易表现
4. **`analyze_slippage()`**: 从 K 线高低价差估算实际滑点

### param_sensitivity.py - 参数敏感性分析

策略参数稳健性扫描，防止过拟合。

**扫描模式**:
- 1D: 单参数扫描
- 2D: 双参数交叉扫描
- full: 全参数网格

**扫描参数**: trail_atr_mult, min_adx, stop_atr_mult, cooldown_bars, risk_per_trade, max_holding_bars

**稳健性评估** (`assess_robustness()`):
| 评级 | 条件 | 含义 |
|------|------|------|
| ROBUST | >40% 网格点在最优 70% 内 | 参数安全 |
| MODERATE | 中间 | 需谨慎 |
| FRAGILE | 过度集中最优区域 | 过拟合风险高 |

## 输出

分析结果输出到 `analysis/output/` 目录:
- `phase1_diagnose_summary.json`: 数据诊断摘要
- `sensitivity_*.json`: 参数敏感性结果
- `walkforward_*.json`: Walk-forward 验证结果

## 依赖关系

- **输入**: `data/storage.py` (klines 数据), 回测结果
- **输出**: JSON 报告文件
- **被使用**: 回测验证流程 (非主交易循环)

## 关键约束

- 敏感性分析计算密集 (全参数网格)，注意运行时间
- Walk-forward 输出由 `backtest_walkforward.py` 生成，本模块主要读取分析
- output/ 目录文件为生成产物，不纳入版本控制

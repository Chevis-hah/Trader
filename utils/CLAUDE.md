# utils/ - 工具模块

## 职责

结构化日志、性能指标计算与报告生成。

> v2.3 无本目录变更；同步记录见 `SYNC_UPDATE_LOG.md`。

## 模块组成

### logger.py - get_logger

结构化日志工具，单例模式。

**输出双通道**:
- Console: 彩色、人类可读格式
- File: JSON 格式，自动按日轮转，存放在 `logs/` 目录

**用法**:
```python
from utils.logger import get_logger
logger = get_logger("module_name")
logger.info("message", extra={"key": "value"})
```

**单例**: 同名 logger 只创建一次实例

### metrics.py - 性能指标

全面的策略性能评估指标集。

**收益指标**: total_return, CAGR

**风险指标**: max_drawdown, VaR, CVaR

**风险调整指标**: sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio

**交易指标**: win_rate, profit_factor, expectancy, turnover

**ML 指标**: IC (信息系数), IC-IR (信息系数信息比)

**报告生成**:
- `generate_report(equity_curve, trade_pnls)`: 从权益曲线 + 交易 PnL 生成完整指标字典
- `format_report(report_dict)`: 格式化打印指标表格

## 依赖关系

- **被依赖**: 所有模块使用 logger; metrics 被 backtest_runner, analysis 使用
- **无业务依赖**: 纯工具模块，不依赖其他业务模块

## 关键约束

- logger 的 JSON 文件格式用于后续分析，保持结构一致
- Sharpe/Sortino 等指标计算假设日频，注意与实际频率匹配
- IC/IC-IR 仅在 ML 模式下有实际用途

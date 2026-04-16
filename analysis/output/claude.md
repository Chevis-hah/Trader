# analysis/output 目录说明

本目录存放 **Trader** 量化策略的回测、走步验证（walk-forward）、敏感度与阶段诊断等**机器可读结果**。给 Claude 或其他助手阅读时，可优先看本文件与合并后的 `analysis_output_consolidated.txt`（全文拼接，便于一次性检索）。

**v2.3 提示**: `scripts/run_all_phases.sh` 已收敛为以 **4h** 为主；1h 证伪类结果可能由 `cleanup_v23.sh` 归档。仓库批次说明见根目录 `SYNC_UPDATE_LOG.md`。

## 合并文件

| 文件 | 说明 |
|------|------|
| `analysis_output_consolidated.txt` | 将本目录下**除** `claude.md` 与**自身**以外的**所有源文件**按文件名排序拼接而成；每段前有 `FILE:` / `SIZE_BYTES:` 分隔头。编码为 UTF-8。 |

## 文本快照（人类可读摘要）

| 文件 | 说明 |
|------|------|
| `backtest_triple_ema_enhanced_snapshot.txt` | **Triple EMA** 增强版统一策略回测快照：摘要、交易概览、品种统计、Regime 等分段文本。 |
| `backtest_macd_momentum_enhanced_snapshot.txt` | **MACD 动量** 增强版回测快照，结构同上。 |
| `backtest_macd_momentum_ml_filtered_snapshot.txt` | **MACD 动量 + ML 过滤** 的回测快照，结构同上。 |

## 权益曲线（CSV）

| 文件 | 说明 |
|------|------|
| `equity_curve_triple_ema.csv` | Triple EMA 策略逐 bar 权益序列；含 `bar_index`, `equity`, `peak`, `drawdown_pct`, `bar_return`, `rolling_sharpe_60` 等列。 |
| `equity_curve_macd_momentum.csv` | MACD 动量策略权益曲线，列结构同上。 |

## 阶段诊断（JSON）

| 文件 | 说明 |
|------|------|
| `phase1_diagnose_summary.json` | **Phase1** 诊断汇总：按策略（如 `triple_ema`, `macd_momentum`）汇总 Regime 维度（ADX、波动率、品种等）下的交易数、盈亏、胜率等；含权益行数等元信息。 |

## 参数敏感度 / 稳健性（JSON）

| 文件 | 说明 |
|------|------|
| `sensitivity_triple_ema.json` | Triple EMA 参数扰动结果：`robustness` 下各参数（如 `trail_atr_mult`, `min_adx`, `stop_atr_mult` 等）的 `verdict`、`good_ratio`、`best_value`、`detail` 等。 |
| `sensitivity_macd_momentum.json` | MACD 动量策略的敏感度与稳健性结构，字段风格与上类似。 |

## Walk-forward（JSON，日线滚动）

| 文件 | 说明 |
|------|------|
| `walkforward_triple_ema.json` | Triple EMA 走步结果：`train_days` / `test_days`、`n_folds`、OOS 总盈亏、Sharpe、各 fold 的 `test_period` / `pnl` / `trades` 等。 |
| `walkforward_triple_ema_120_45.json` | 同上策略，**训练窗 120 天、测试窗 45 天**（或项目内约定参数）变体。 |
| `walkforward_macd_momentum.json` | MACD 动量走步汇总，字段与 Triple EMA 系列类似。 |
| `walkforward_macd_momentum_120_45.json` | MACD 动量，120/45 窗口变体。 |

## Walk-forward（JSON，`wf_*`，多周期 / 多策略）

前缀 `wf_` 一般为 **按周期或策略名拆文件** 的走步结果；结构常含 `strategy`、`interval`（如 `1h` / `4h`）、`train_days`、`test_days`、按 fold 的 `symbol` / `pnl` / `trades` 等。

| 文件 | 说明 |
|------|------|
| `wf_triple_ema_4h.json` | Triple EMA，**4h** 周期走步。 |
| `wf_macd_momentum_1h.json` | MACD 动量，**1h**。 |
| `wf_macd_momentum_4h.json` | MACD 动量，**4h**。 |
| `wf_mean_reversion_1h.json` | 均值回归类策略，**1h**。 |
| `wf_mean_reversion_4h.json` | 均值回归类策略，**4h**。 |
| `wf_regime_4h.json` | Regime 相关策略，**4h**；可能含 `use_ml_filter` 等字段。 |

## 给助手的用法建议

- 需要**全文搜索**或一次性投喂上下文：使用 `analysis_output_consolidated.txt`。
- 需要**结构化解析**（脚本 / 图表）：直接读对应 `.json` 或 `.csv` 原文件，不要用合并 txt 做程序输入。
- 快速了解某次回测叙述：读 `backtest_*_snapshot.txt`。

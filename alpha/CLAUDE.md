# alpha/ - 策略与信号模块

## 职责

策略定义、信号生成、ML 模型、组合优化、策略增强补丁。

## 核心发现（历史 v2.2 扫描，仍作参考）

**BTC/ETH 上部分因子呈均值回归型**（`factor_scan_report.json` 等）: ma_dev_240、zscore_240、rsi_14 等。  
v2.3 **不再**把多重「过度拉伸过滤」默认堆在 MACD/TripleEMA 上；改为用 WF 样本量与 OOS 表现约束复杂度，详见 `CHANGELOG_v23.md` 与 `docs/DEPRECATED_FACTORS_AND_STRATEGIES.md`。

## 策略注册表 (`strategy_registry.py`)

```python
from alpha.strategy_registry import build_strategy
strategy = build_strategy(config, "triple_ema")
```

**新增策略必须**: 实现完整接口并在 `STRATEGY_MAP`（或 `SIMPLE_STRATEGIES`）注册，并在 `config/settings.yaml` 增加参数节。

**当前主线（v2.3）**:

- `triple_ema` — TripleEMA Pullback，**v2.3** 去掉 v2.2 的过度拉伸过滤链。
- `macd_momentum` — MACD Momentum，**v2.3** 回滚为接近 v2.0 的纯规则 + ADX；配置类为 `MACDMomentumConfig`。
- `mean_reversion` — **v2.3** 在震荡逻辑上增加**趋势回避门**（高 ADX、连续单边走势等）。
- `regime` — 代码可能仍存在于注册表供兼容；**默认配置已移除**（见 `test_v23` 与 `config/settings.yaml`）。
- `grid` — `SIMPLE_STRATEGIES`（若 `grid_strategy` 可用）。

### 路线 A（实验）

| 文件 | 说明 |
|------|------|
| `cross_sectional_momentum.py` | 横截面因子 + 分位多空组合权重 |

## 规则策略对照（v2.3）

| 策略 | 文件 | 说明 |
|------|------|------|
| TripleEMA | `triple_ema_strategy.py` | EMA 排列 + ADX/RSI/回踩等规则为主，无 v2.2 overextension 堆叠 |
| MACDMomentum | `macd_momentum_strategy.py` | MACD 金叉 + ADX 等；已移除 v2.2 无效参数（min_rsi/max_rsi/max_holding_bars 等） |
| MeanReversion | `mean_reversion_strategy.py` | 震荡均值回复 + **趋势回避** |

## ML 与组合（与 v2.2 大体连续）

| 文件 | 说明 |
|------|------|
| `ml_lightgbm.py` | LightGBM，Purged+Embargo CV |
| `ml_model.py` | sklearn 集成 |
| `ml_signal_filter.py` | ML 过滤补丁 |
| `portfolio.py` | 组合优化器 |
| `regime_allocator.py` | Regime 分配（若仍使用） |

`regime_allocator_v2` 等若已执行 `cleanup_v23.sh` 则归档至 `archive/`，以仓库实际状态为准。

## 增强补丁（可选）

| 补丁 | 文件 | 说明 |
|------|------|------|
| P1 做空 | `bidirectional_wrapper.py` | Wrapper |
| P3 出场 | `enhanced_exit.py` | v2.3 迁移包可归档 |
| P4 ML 过滤 | `ml_signal_filter.py` | Monkey-patch |

## 依赖与约束

- **输入**: `data/features.py`, `data/client.py`
- **输出**: `execution/executor.py`, `risk/manager.py`
- **配置**: `config/settings.yaml` → `strategy` 节
- Position sizing 基于 ATR + `risk_per_trade%`，不超过 `max_position%`

仓库级同步记录见根目录 **`SYNC_UPDATE_LOG.md`**。

# CHANGELOG — v2.3（规则回滚 + 配置清理 + 路线 A 脚手架）

> 日期: 2026-04-16  
> 来源: `web_dev_files/Trader_v23_cleanup/` 经 `sync_to_repo.py` 合入

---

## 背景

v2.2 在 MACD / TripleEMA 上叠加的过度拉伸类过滤在 walk-forward 上样本骤减、表现退化；v2.3 将两条趋势策略回滚为更稳健的纯规则集，并强化均值回归侧的趋势回避。与此同时删除已证伪的 1h 与 regime 相关默认配置，并加入横截面路线 A 的最小可运行代码与文档。

---

## 策略与因子

### `alpha/macd_momentum_strategy.py` → v2.3

- 移除 v2.2 的 zscore / ma_dev / keltner 等「追高过滤」主路径依赖，回到经 WF 支持的 MACD 交叉 + ADX 逻辑。
- 配置类更名为 `MACDMomentumConfig`；删除敏感性扫描中表现为无效的参数项。

### `alpha/triple_ema_strategy.py` → v2.3

- 移除过度拉伸过滤，与 v2.3「先求稳健样本量再谈复杂门控」的原则一致。

### `alpha/mean_reversion_strategy.py` → v2.3

- 增加趋势回避门（高 ADX、连续单边走势等），降低在强趋势段错误接刀的概率。

### 新增 `alpha/cross_sectional_momentum.py`

- 横截面因子（动量、反转、波动、Amihud、规模代理等）与分位多空组合构建，供路线 A 实验。

---

## 数据与回测

### 新增 `data/universe.py`

- 为横截面策略提供可复现的 Top N universe 构造。

### 新增 `cross_sectional_backtest.py`

- 路线 A 的 MVP 回测入口（多标的、横截面权重）。

---

## 配置

### `config/settings.yaml`

- `universe.intervals` 不再包含 `1h`（与 v2.3 测试及迁移文档一致）。
- `strategy` 节移除 `regime` 块（配置驱动侧与 v2.3 清理对齐）。
- 执行滑点：`BTCUSDT` / `ETHUSDT` 等与 phase1 实测一致的 bps 档位（见仓库内 yaml）。

---

## 脚本与运维

### `scripts/run_all_phases.sh`

- 流水线仅保留 4h 规则/ML 相关步骤；删除对 1h ML+Monte Carlo 段的调用。
- 结尾提示改为：若 4h 均值回归 WF 不达标则启动 `bash scripts/run_mvp_path_a.sh`。

### 新增 `scripts/cleanup_v23.sh`

- 将 regime、enhanced_exit、扩表配置、证伪 wf json 等移至 `archive/v22_deprecated/`；执行前请自行备份。本仓库在备份后已执行一次，归档文件以 `archive/v22_deprecated/` 下实际文件为准。

### 新增 `scripts/run_mvp_path_a.sh`

- 一键拉起路线 A（横截面 MVP）相关命令的入口脚本。

---

## 文档

- `docs/README_v23_MIGRATION.md` — 覆盖安装顺序、与 `cleanup_v23.sh` 的配合方式。
- `docs/DEPRECATED_FACTORS_AND_STRATEGIES.md` — 记录弃用因子与策略证据。
- `docs/RESEARCH_2025_SOTA_QUANT_ARCHITECTURES.md` — 外部架构调研摘要。

---

## 测试

### `tests/test_v23.py`

- Import 与实例化：`MACDMomentum`、`TripleEMA`、`MeanReversion`、`CrossSectionalMomentum`。
- MACD：已删参数不存在于 config；ADX 门控与金叉逻辑冒烟。
- 均值回归：趋势回避与正常震荡入场。
- 横截面：因子计算、数据不足、组合多空权重。
- 配置：YAML 可加载；无 `1h`、无 `regime` 策略块；BTC 滑点阈值断言。

```bash
python -m pytest tests/test_v23.py -v
```

---

## 与 v2.2 文档的关系

- 历史因子驱动与 v2.2 工具链说明仍保留在 `CHANGELOG_v22.md`；v2.3 在其基础上做**减法**（证伪路径下线）与**增量**（路线 A）。根目录 `CLAUDE.md` 以 v2.3 为当前叙事主线。

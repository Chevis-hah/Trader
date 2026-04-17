# CHANGELOG — v2.3（规则回滚 + 配置清理 + 路线 A 脚手架）

> 日期: 2026-04-16  
> 来源: `web_dev_files/Trader_v23_cleanup/` 经 `sync_to_repo.py` 合入

---

## v2.3.1 — 2026-04-17 (P0-T01 清理 + P1-T01/T02 交付)

### 修复

- **`alpha/strategy_registry.py`**: 正式移除对 `RegimeAdaptiveStrategy` 和 `RegimeStrategyConfig` 的导入与注册（P0-T01 遗留）。默认策略从 `triple_ema` 改为 `macd_momentum`（后者 enabled=true；triple_ema 样本不足，已在 settings.yaml 中标记 enabled=false）。
- **`data/universe.py`**: 修复严重 bug —— 原实现 `storage.get_klines(symbol, "1d", limit=50)` 与 `UniverseConfig.min_history_days=180` 直接冲突，导致任何 as_of_date 都无法取到足够窗口；改为按 `start_time`/`end_time` 窗口查询，窗口宽度 = `min_history_days + lookback_buffer_days`。排序时优先使用 `quote_volume`（USD 计价）而非 `close * volume`（更准确）。缓存 key 增加参数指纹，避免切换 `top_n`/`min_volume_30d_usd` 时命中过期缓存。
- **`data/universe.py`**: `get_all_symbols()` 优先从 `storage.list_kline_symbols('1d')` 读取已入库的真实 symbol 列表；仅在 DB 为空时才回退到硬编码 bootstrap 池。

### 新增

- **`data/storage.py`** 扩展三个工具方法：
  - `list_kline_symbols(interval=None)`: 返回已入库的 symbol 列表
  - `get_kline_coverage(interval)`: 每 symbol 的行数/起止时间
  - `check_kline_gaps(symbol, interval)`: 检测时间跳缺和 0 成交量连续段
- **`scripts/sync_universe_data.py`** (P1-T01): 从 Binance 公开 REST API 同步 Top N 币种多 interval 历史 K 线。特性：增量同步（断点续传）、自动重试 + 429 回退、请求节流、数据完整性校验。默认拉 60 币种 2020-01-01 起的日线。
- **`tests/test_universe.py`** (P1-T02): 13 个测试覆盖 point-in-time 正确性、缓存一致性、流动性/历史过滤、turnover smoothing 过滤 memecoin 尖峰、边界情况。**全部通过**。
- **`requirements.txt`**: 按 P0-T02 补齐 `scipy`, `websockets`, `pytest-cov`, `matplotlib` 声明，并在注释中标注 `timeseriescv`（Phase 2B CPCV 的外部选项）。
- **`.env.example`**: 覆盖 Binance (spot/testnet) + CryptoQuant + CoinGlass 三类 API key 占位符，以及 `LOG_LEVEL` / `DB_PATH` 系统变量。
- **`.gitignore`**: 按 P0-T02 要求加入 `.env`、`data/`、`logs/`、`__pycache__/`、`archive/`（后者作为注释保留，便于追溯）、`backtest_*_snapshot.txt`、虚拟环境目录等。

### 本地验收命令

```bash
# 1. 安装依赖 (或在已有 venv 中 pip install -r requirements.txt --break-system-packages)
pip install -r requirements.txt --break-system-packages

# 2. 跑 universe 测试
python -m pytest tests/test_universe.py -v       # 预期 13 passed
python tests/test_v23.py                         # 原 16 个 smoke test 不变

# 3. 拉 5 年日线 (需联网 ~10-30min)
python scripts/sync_universe_data.py --top-n 60 --start 2020-01-01

# 4. 校验覆盖度
python scripts/sync_universe_data.py --validate-only --min-years 4
```

### P1-T01 验收指标

- [x] 代码已交付 `scripts/sync_universe_data.py`
- [ ] （本地执行后）至少 45 个 symbol 有 ≥ 4 年完整日线
- [ ] DB 行数 ≥ 65,700
- [ ] `BTCUSDT` 1d rows > 1500

### 合库记录（2026-04-17 晚）

- 自 `web_dev/Trader_v231_patch/` 经 `sync_to_repo.py` 合入；与上文 v2.3.1 条目为同一批网页端补丁内容。
- **路径纠偏**：首次以外层目录为源时曾将文件落在 `Trader/Trader/`；已合并回仓库根相对路径。`sync_to_repo.py` 现已支持「顶层仅含一个 `Trader/` 子目录」时自动下沉为该内层目录再同步，避免重复嵌套。

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

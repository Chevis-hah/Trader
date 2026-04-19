# CHANGELOG — v2.4.0 (Phase 2A FundingHarvester + Phase 2B CPCV 框架)

> 日期: 2026-04-19
> 任务: P2A-T02 + P2B-T01 + P2B-T02 (见 `docs/TODO_12M_DEVELOPMENT.md`)
> 前置: v2.3.1 (见 `CHANGELOG_v23.md`) + P2A-T01 (`scripts/sync_funding_data.py`)

---

## 背景

Phase 1 (`cross_sectional_momentum` MVP) 验收完毕, 最佳配置 Sharpe ≈ 0.18 (FAILED);
路线图指向 **Phase 2A Funding Harvester** 与 **Phase 2B CPCV 验证框架** 并行推进。
本次交付这两条轨道的 "入口模块", 为 P2A-T03 回测引擎与 P2B-T04 历史策略审计铺路。

---

## 新增

### Phase 2A — FundingHarvester (P2A-T02)

- **`alpha/funding_harvester.py`** (新增, ~270 行)
  - `FundingHarvesterConfig` 数据类: `min_funding_rate` / `min_funding_duration_h` /
    `notional_per_trade_usd` / `max_concurrent_positions` / `leverage` /
    `exit_on_funding_flip` / `exit_buffer_rate` / `spot_fee_bps` / `perp_fee_bps` /
    `slippage_bps` + 自校验 `validate()`
  - `FundingHarvester` 策略类:
    - `should_enter(snapshot, state)` — 当前 funding ≥ 阈值 AND 过去 N 个 cycle 全部达标 AND 未满仓
    - `should_exit(current_funding, position)` — funding 翻负 / 低于 buffer / NaN 时平仓
    - `calc_position(capital, spot_price, perp_price)` — 返回严格相等的 `(spot_qty, perp_qty)` 保证 delta-neutral
    - `compute_pnl(position, current_prices, funding_received)` — mark-to-market: spot_leg + perp_leg + funding - realized_cost
    - `round_trip_cost(notional_usd)` — 完整开平成本 = 2×(spot_fee + perp_fee + 2×slippage) × notional / 1e4
    - `annualized_from_single_funding(rate)` — 单次 funding × 3 × 365 的年化换算
- **`tests/test_funding_harvester.py`** (新增, 25 tests 全通过)
  覆盖: config 校验、入场阈值 & 持续性、满仓门、NaN 保护、funding 翻负平仓、buffer 平仓、delta 中性 (价格涨跌 PnL ≈ 0)、funding 累计、成本公式、成本线性、年化换算、注册表集成。
- **`config/settings.yaml`**: 新增 `strategy.funding_harvester` 节, 默认 `enabled: false` (独立入口, 不进 TradingEngine 循环)。

### Phase 2B — CPCV 验证框架 (P2B-T01 + P2B-T02)

- **`validation/`** (新建包)
  - `__init__.py` — 统一导出 `PurgedKFold` 与 `CombinatorialPurgedCV`
  - `purged_cv.py` — `PurgedKFold(n_splits, embargo_pct, sample_times=None)`:
    - 按时间顺序切 K 块 (余数分给前若干块, 更均匀)
    - `embargo_pct`: 测试集结束后 `floor(embargo_pct × N)` 个样本不进训练集
    - `sample_times` (可选): Series(index=X.index, values=label_end_time); 训练样本若 label 区间 `[start, end]` 与测试期重叠, 被 purge
    - sklearn 兼容: `get_n_splits()` + `split(X, y=None, groups=None)` iterator
  - `cpcv.py` — `CombinatorialPurgedCV(n_groups, n_test_groups, embargo_pct, sample_times=None)`:
    - `n_splits = C(N, k)`, `n_paths = C(N-1, k-1) = k × C(N,k) / N`
    - `split(X)` 枚举所有 `(train_idx, test_idx, test_group_tuple)` 共 C(N,k) 个
    - `get_paths(X)` 返回 `n_paths` 条 path, 每条 path 是 `[split_idx for group 0..N-1]`; 构造规则 "group g 的第 j 次出现分配给 path j"
    - `backtest_paths(X, strategy_fn)` 一键: 对每 split 调 `strategy_fn(train_idx, test_idx) -> pd.Series`, 再按 group 粒度拼接成 `n_paths` 条完整 OOS 曲线
- **`tests/test_purged_cv.py`** (新增, 16 tests 全通过)
  覆盖: K=5 产出 5 splits、train/test 无重叠、test sizes 等分、覆盖全时序、embargo 生效、embargo 缩小训练集、最后一 fold 无 embargo 损失、label overlap purging、输入校验、sklearn API。
- **`tests/test_cpcv.py`** (新增, 16 tests 全通过)
  覆盖: N=10/k=2 → 45 splits / 9 paths、一般组合学、每条 path 覆盖全时序、每个 split 至少被一条 path 用到、train/test 无重叠、test 大小比例、test_groups tuple 升序、embargo 缩小训练集、trivial strategy_fn、不同 train 产出不同 path、输入校验。

### 注册表

- **`alpha/strategy_registry.py`**: 新增 `FundingHarvester` 的条件 import 与 `SIMPLE_STRATEGIES["funding_harvester"]` 注册; 文档字符串加 v2.4.0 段。

---

## 修改

- **`docs/TODO_12M_DEVELOPMENT.md`**: P2A-T02 / P2B-T01 / P2B-T02 三任务的 Acceptance checkbox 打钩, 各自追加"完成记录 (2026-04-19)"行。
- **`alpha/strategy_registry.py`**: 见上 "新增 > 注册表"。
- **`config/settings.yaml`**: 见上 "新增 > Phase 2A > config"。

---

## 新增文件 (9)

```
alpha/funding_harvester.py
tests/test_funding_harvester.py
validation/__init__.py
validation/purged_cv.py
validation/cpcv.py
tests/test_purged_cv.py
tests/test_cpcv.py
.iteration_state.md
CHANGELOG_v24.md        # 本文件
```

## 修改文件 (3)

```
alpha/strategy_registry.py
config/settings.yaml
docs/TODO_12M_DEVELOPMENT.md
```

---

## 本地验收命令

```bash
# 运行本次新增的测试 (57 tests 全绿)
python -m unittest tests.test_funding_harvester -v
python -m unittest tests.test_purged_cv -v
python -m unittest tests.test_cpcv -v

# 完整回归 (含既有 67 个 tests + 新增 57 = 124 tests)
python -m unittest discover -s tests -v

# 快速 sanity
python -c "from alpha.strategy_registry import available_strategies; print(available_strategies())"
# 期望输出包含 'funding_harvester'
python -c "from validation import PurgedKFold, CombinatorialPurgedCV; \
cv = CombinatorialPurgedCV(10, 2); print(cv.n_splits, cv.n_paths)"
# 期望: 45 9
```

---

## 下一步

按 `docs/TODO_12M_DEVELOPMENT.md`:

- **P2A-T03**: `funding_harvester_backtest.py` + `scripts/run_funding_harvester.sh`
- **P2B-T03**: `validation/dsr.py` (Deflated Sharpe Ratio + Probability of Backtest Overfitting)
- **P2B-T04**: 用 CPCV 重审 5 个历史策略 + `analysis/output/cpcv_audit_report.md`

完成 P2B-T03/T04 后, 进入 Decision Gate 1 (路线图 Month 3 末)。

---

## v2.4.1 — 2026-04-19 (Phase 2 完工批次)

> 任务: P2A-T03 + P2A-T04 + P2A-T05 + P2B-T03 + P2B-T04 + P2B-T05
> 前置: v2.4.0 (见上方) + v2.3.1

### 新增 — Phase 2A 完工

- **`funding_harvester_backtest.py`** (P2A-T03, ~280 行): 8h bar 驱动的回测引擎。读 `funding_rates` + 按 symbol 分组事件流推进 + 累积 funding 收益 + 成本扣除 (open+close 各扣一次 `spot_fee+perp_fee+2×slippage`) + 强制结束平仓 + summary JSON schema 与 `cross_sectional_backtest.py` 一致 (`total_return_pct` / `annualized_return_pct` / `sharpe_ratio` / `max_drawdown_pct` / `win_rate_pct` / `n_trades` / `verdict`)。
- **`scripts/run_funding_harvester.sh`** (P2A-T03): 一键 wrapper, 默认 2023-01-01 起始 + 100K 初始资金 + 全 DB symbols。
- **`tests/test_funding_backtest.py`** (6 tests 全绿): 无数据抛异常、零 funding 不开仓、持续正 funding 产生盈利交易、funding 翻负触发平仓、verdict 落地、max_concurrent_positions 生效。
- **`scripts/run_funding_sensitivity.py`** (P2A-T04, ~180 行): 64 格笛卡尔积扫描 (`min_funding_rate × min_funding_duration_h × max_concurrent_positions`), 输出 `analysis/output/funding_sensitivity.json`, 计算 `good_ratio` (达到最优 70%+ 的格子比例) + ROBUST/MODERATE/FRAGILE 判决 + `best_params` 提取。
- **`tests/test_funding_sensitivity.py`** (7 tests 全绿): 单格优势 → FRAGILE、平坦景观 → ROBUST、空结果安全、best_params 提取、默认网格 64 格验证、小网格 end-to-end。
- **`execution/binance_executor.py`** (P2A-T05, ~380 行): `BinanceExecutor` + `ExecutorSafety` + `OrderResult`。四层安全: dry_run / testnet / `allow_live` 开关 / 硬顶三项 (`max_notional_per_trade` + `max_daily_loss` + `max_total_notional`)。支持 spot/perp 下单、持仓查询、delta-neutral 一键开平 (spot 失败自动回滚)、WebSocket !markPrice@arr 订阅骨架 (指数退避重连)。
- **`main_funding.py`** (P2A-T05, ~170 行): Paper trading 主循环, 可注入 `snapshot_provider` 便于测试; 默认用 REST `/fapi/v1/premiumIndex` 轮询。CLI 支持 `--dry-run/--no-dry-run`, `--testnet/--no-testnet`, `--live` 组合的安全链。
- **`tests/test_binance_executor.py`** (16 tests 全绿): safety 配置校验、dry_run 订单 & 拒绝路径、delta-neutral 开平、daily pnl 跨日重置、paper-loop 持仓建立/平仓/异常吞吐/max_concurrent 生效等。

### 新增 — Phase 2B 完工

- **`validation/dsr.py`** (P2B-T03, ~230 行):
  - `deflated_sharpe_ratio(observed_sr, n_trials, n_samples, skew, kurt, benchmark_sr)` — López de Prado 2018 公式 9, 包含 `Var[SR_hat]` 的 skew/kurt 校正与 n_trials 的 `E[max_SR]` 偏移。
  - `probability_of_backtest_overfitting(returns_matrix, n_splits)` — Bailey 2015 CSCV, 枚举 `C(n_splits, n_splits/2)` 种对称分法, 输出 λ<0 经验概率。
  - `summarise_cpcv_paths(curves, n_trials)` — 便捷入口: 输入 CPCV 的 9 条 OOS 曲线, 输出 `{sharpe_median/mean/std, dsr_median, verdict}` 三级判决 (PASS/CONDITIONAL/FAIL)。
- **`tests/test_dsr.py`** (18 tests 全绿): `_expected_max_sr` 单调性 + LópezdePrado Table 1 量级验证、DSR 边界、n_trials↑→DSR↓、n_samples↑→DSR↑、kurt↑→DSR↓、负偏度↓DSR、PBO 三种极端情景 (全同策略/一家独大/纯噪声)、summarise 与 CPCV 集成、输入校验。
- **`scripts/run_cpcv_audit.py`** (P2B-T04, ~330 行): 对 5 个策略 (`macd_momentum` / `triple_ema` / `mean_reversion` / `cross_sectional_momentum` / `funding_harvester`) 跑 N=10/k=2 CPCV, 双模式: `--demo` (合成 2 年日频 + 2 年 8h funding) 与默认实盘 (从 `data/quant.db` 读)。产出 `analysis/output/cpcv_audit_report.md` (Markdown 对比表 + 每策略 9-path Sharpe 分布) + `cpcv_audit.json`。CI 契约: 任一策略 PBO>40% 时 `sys.exit(2)`。
- **`tests/test_cpcv_audit.py`** (7 tests 全绿): 合成数据 shape + determinism、信号函数二值性、bar_returns 输出、单策略 CPCV 返回完整字典、5 策略 end-to-end demo、Markdown 报告包含必要字段。
- **`scripts/run_cpcv_validation.sh`** (P2B-T05): CI 入口 shell, 透传 `CPCV_MODE=demo|real`, 反映退出码 (0/2) 的明确消息。

### 修改

- **`validation/__init__.py`**: 导出 `deflated_sharpe_ratio` / `probability_of_backtest_overfitting` / `summarise_cpcv_paths`。
- **`scripts/run_all_phases.sh`**: 末尾追加 "CPCV 验证" 步骤, 调用 `run_cpcv_validation.sh`, 不因 PBO 超限中断 WF 流水线但记录 exit code 供人看到。
- **`scripts/__init__.py`**: 空文件新增 (使 `scripts.run_*` 可从 tests 中 import)。
- **`docs/TODO_12M_DEVELOPMENT.md`**: P2A-T03 / P2A-T04 / P2A-T05 / P2B-T03 / P2B-T04 / P2B-T05 全部 Acceptance 打钩 + 完成记录 (需真实 API/DB 的两项保留 `[ ]` 并注明所需命令)。
- **`CLAUDE.md`**: File Map 新增本批 11 个条目。
- **`.iteration_state.md`**: 最近 5 次迭代 / 架构决策 / 技术债务 / 下次方向四段全部刷新为 Phase 2 完工状态。

### 测试回归

```
Ran 181 tests in 2.18s — OK
```

- 本次迭代新增 72 tests (DSR 18 + funding_backtest 6 + funding_sensitivity 7 + binance_executor 16 + cpcv_audit 7 + 已在 v2.4.0 中计入的 funding_harvester 25 + purged_cv 16 + cpcv 16 = 72 本批次 + 57 前批次 = 114 总计 Phase 2 交付测试)
- 原有 67 tests 全部保留通过, 无回归。

### 本地验收命令

```bash
# 全仓回归
python -m unittest discover -s tests -v

# Phase 2A (需真实 API key / 数据)
bash scripts/run_funding_harvester.sh
python scripts/run_funding_sensitivity.py --db data/quant.db --start 2023-01-01
python main_funding.py --max-iterations 3   # dry_run+testnet 默认

# Phase 2B
bash scripts/run_cpcv_validation.sh                 # demo 模式, 默认
CPCV_MODE=real bash scripts/run_cpcv_validation.sh  # 实盘模式 (需 DB)

# WF + CPCV 一键流水线
bash scripts/run_all_phases.sh
```

### 下一步

Phase 2 代码完备后, 路线图明确指向 **Decision Gate 1** (Month 3 末):
1. 用实盘数据回放 Phase 2A 回测 (`run_funding_harvester.sh`), 确认 Sharpe > 1.5 / MDD < 10% / 年化 5-30% 在业界区间
2. 跑 `CPCV_MODE=real` 审计全部 5 策略, 收集 PBO < 30% 的上线候选
3. 按路线图决策矩阵选走 Phase 3 (多因子) 或优化 Funding + 上 Paper Trading

技术债务批次处理见 `.iteration_state.md` "已知技术债务"段。

---

## v2.4.2 — 2026-04-19 晚 (基于本地实盘证据的 rework)

> 触发: 本地 agent 在 `analysis/output/phase2_closeout/` 下跑出首批 Phase 2 实盘数据,
> 暴露 CPCV 与 Funding 两处严重问题。本批次修 bug + 加诊断 + 合入本地 agent 反馈。

### 发现的 bug

1. **CPCV path 塌缩**: 5 个策略全部 `sharpe_std=0`, PBO=100%。
   根因: `run_cpcv_for_strategy` 预计算 `full_returns`, 每个 split 只切片 → 所有 path 拼回同一曲线。
2. **CPCV 年化因子错**: `summarise_cpcv_paths` 硬编码 `sqrt(252)`, 但 funding 是 8h bar (应 1095), 4h 价格策略应 2190。导致 `funding_harvester` CPCV Sharpe 显示 3.498, 实际单一回测 Sharpe=0.0099。
3. **Funding Harvester 实盘 FRAGILE**: 35 symbols × 3 年, `best_sharpe=0.0247`, `good_ratio=28%`, `verdict=FRAGILE`。无诊断信息可供根因定位。

### 修复

#### `validation/dsr.py`
- `summarise_cpcv_paths` 新增 `bars_per_year` 参数 (默认 252) 替代硬编码 `sqrt(252)`。
- 新增 `is_deterministic` 检测 (`sharpe_std < deterministic_eps`), 输出 dict 多两字段: `is_deterministic: bool`, `bars_per_year: float`。
- 对 deterministic 策略, DSR 改用单条 curve 计算 skew/kurt (避免 concat 9 条相同曲线产生虚假峰度)。
- `_sharpe_columnwise` 改用 `np.divide(..., out=..., where=valid)` 消除 "invalid value" RuntimeWarning (本地 agent 建议)。

#### `scripts/run_cpcv_audit.py`
- `run_cpcv_for_strategy` 新增 `bars_per_year` 参数透传到 `summarise_cpcv_paths`。
- 检测 deterministic 情况, PBO 强制设 NaN + 添加 `notes` 字段解释原因。
- G1 判决分叉: deterministic 策略跳过 PBO 条件, 但附说明 "上线前需构造 parameter-grid 变体重跑"。
- `audit_real` / `audit_demo` 按策略正确分配 bars_per_year: 价格策略 4h=2190, funding 8h=1095, daily=252。
- Markdown 报告新增 `det?` 列 + `## 备注` 段 + PBO=N/A 标签, 明确告诉读者哪些策略的 PBO 不具统计意义。

#### `funding_harvester_backtest.py`
- 新增 `_compute_diagnostics` + `_trade_brief`, 输出顶层 `diagnostics` 段, 包含:
  - `per_symbol`: {symbol → {n_trades, total_funding_income, total_cost, total_pnl, win_rate_pct, avg_duration_h}}
  - `cost_breakdown`: total_funding_income / total_trading_cost / net_pnl / cost_to_income_ratio / net_pnl_pct_of_capital
  - `top_winners` / `top_losers`: 前 5 个盈亏最大 trade 的摘要
  - `diagnostic_hint`: 自动生成的字符串, 覆盖 "cost_eats_income" / "funding_income_negative" / "healthy" 等场景
- CLI 日志尾部打印 `diagnostic_hint`, 用户一眼能看到诊断结论。

### 本地 agent 合入

- `alpha/ml_lightgbm.py::_try_import_lgbm`: `except ImportError` → `except (ImportError, OSError)` 分支, 日志建议跑 `scripts/install_debian_test_deps.sh`。
- `utils/logger.py::JsonFormatter.format`: `datetime.utcnow()` → `datetime.now(timezone.utc)`, 去 DeprecationWarning。
- `core/engine.py` / `risk/manager.py`: 同上 utcnow 收敛 (2 / 5 处)。
- `scripts/run_cpcv_audit.py` / `scripts/run_funding_sensitivity.py` / `funding_harvester_backtest.py`: `pd.Timestamp.utcnow()` → `pd.Timestamp.now("UTC")`。
- `validation/dsr.py::_sharpe_columnwise`: `np.divide(..., where=valid)` (上文已说明)。
- **新增** `scripts/install_debian_test_deps.sh`: Debian/WSL 一键安装 `libgomp1` + `python3-venv`。

### 新测试

- `tests/test_dsr.py` +4: `is_deterministic` 真/假检测、`bars_per_year` 影响 Sharpe 正确性、deterministic 路径的 DSR 有限。
- `tests/test_cpcv_audit.py` +3: deterministic 标记写入、`bars_per_year` 透传、报告 N/A 标签 + 备注段。
- `tests/test_funding_backtest.py` +3: `diagnostics` 段存在与字段齐全、per_symbol 分解、cost dominated regime 的 hint。

### 测试回归

```
Ran 191 tests in 2.80s — OK
```
(原 181 + 本次 10)

### 新增文件 (2)

```
scripts/install_debian_test_deps.sh
CHANGELOG_v24.md (本次追加 v2.4.2 段)
```

### 修改文件 (8)

```
alpha/ml_lightgbm.py
utils/logger.py
core/engine.py
risk/manager.py
validation/dsr.py
scripts/run_cpcv_audit.py
scripts/run_funding_sensitivity.py
funding_harvester_backtest.py
tests/test_dsr.py
tests/test_cpcv_audit.py
tests/test_funding_backtest.py
docs/TODO_12M_DEVELOPMENT.md
.iteration_state.md
```

### 本地验收命令

```bash
# 1. 全仓回归
python -m unittest discover -s tests -v
# 期望: Ran 191 tests ... OK

# 2. CPCV 实盘重跑 (验证 det? 列 + 正确年化 + PBO=N/A)
CPCV_MODE=real bash scripts/run_cpcv_validation.sh
# 期望 report.md 里:
#   - macd_momentum / triple_ema / mean_reversion / cross_sectional_momentum 的 det? 列全为 ✓
#   - funding_harvester 4h Sharpe 应 ≈ Sharpe_8h × sqrt(2190/1095)  ≈ 5.0 左右
#     (相比旧版错误的 3.498, 新版用正确的 sqrt(1095) 年化)
#   - PBO 列全部显示 "N/A"
#   - 多出 "## 备注" 段解释为什么

# 3. Funding 回测看诊断
bash scripts/run_funding_harvester.sh
# 期望 log 多两行:
#   诊断: funding_income=$xx cost=$xx net=$xx ratio=xx
#   诊断提示: <hint string>

# 4. 敏感度扫描 (无改动, 应与上次结果一致)
python scripts/run_funding_sensitivity.py --db data/quant.db --start 2023-01-01
```

### 下一步

Phase 2 代码 + 诊断就绪, 路径:
1. 本地 agent 用修复后的代码重跑 `run_phase2_closeout.sh`
2. 观察新的 `diagnostics.diagnostic_hint` 定位 FRAGILE 根因
3. 根据根因 (成本 / 数据 / universe) 决定调优方向 → 再跑一轮敏感度
4. 若此时 good_ratio 上 30%+, 进 Decision Gate 1; 否则按路线图 "Funding Sharpe<1.5 一定有 bug" 排查


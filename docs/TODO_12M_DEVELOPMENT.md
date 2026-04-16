# Trader 项目开发 TODO — 12 个月阶段化计划

> **如何使用本文档（给 LLM）**: 本文档是 Trader 量化交易项目的完整开发路线图，按阶段和任务组织。你需要按顺序执行任务，每个任务都有明确的产出物、验收标准和依赖关系。在每个阶段结束有 **Decision Gate**，必须先读取并确认通过标准才能进入下一阶段。
>

---

## 📋 目录

- [文档约定](#文档约定)
- [项目背景](#项目背景)
- [参考文档](#参考文档)
- [全局规则](#全局规则)
- [Phase 0: 环境准备 + 清理](#phase-0-环境准备--清理)
- [Phase 1: 横截面 Momentum MVP (Month 1)](#phase-1-横截面-momentum-mvp-month-1)
- [Phase 2A: Funding Harvester (Month 2-3，并行)](#phase-2a-funding-harvester-month-2-3并行)
- [Phase 2B: CPCV 验证框架 (Month 2-3，并行)](#phase-2b-cpcv-验证框架-month-2-3并行)
- [🚪 Decision Gate 1 (Month 3 末)](#-decision-gate-1-month-3-末)
- [Phase 3: 多因子扩展 (Month 4-5)](#phase-3-多因子扩展-month-4-5)
- [Phase 4: On-chain 因子 (Month 6-9)](#phase-4-on-chain-因子-month-6-9)
- [🚪 Decision Gate 2 (Month 9 末)](#-decision-gate-2-month-9-末)
- [Phase 5: LOB 做市 (Month 10-12，可选)](#phase-5-lob-做市-month-10-12可选)
- [持续性任务](#持续性任务)

---

## 文档约定

### 任务 ID 规则

`P{阶段}-T{任务号}` 例如 `P1-T01`, `P2A-T03`

### 任务字段含义

| 字段 | 说明 |
|---|---|
| **Effort** | 预估工时（LLM 执行时长 + 人工审核时长） |
| **Depends on** | 前置任务 ID，全部完成才能开始本任务 |
| **Deliverable** | 产出物（文件/功能/报告） |
| **Files** | 需要创建或修改的文件路径 |
| **Steps** | 具体执行步骤 |
| **Acceptance** | 验收标准，checkbox 全部打钩才算完成 |
| **Reference** | 相关论文、文档、代码库 |

### 执行约定

1. **一次只做一个任务**: 完成一个任务停下来等用户确认，不要一口气连做多个
2. **遇到决策点必须停下**: 看到 🚪 标记必须先报告当前状态
3. **验收不通过不能推进**: 如果 Acceptance 中有打不了钩的项，必须先修复
4. **写测试**: 每个涉及策略逻辑的任务都必须配套单元测试
5. **Purged CPCV 是硬门槛**: Phase 2B 完成后，所有回测必须走 CPCV，PBO > 30% 的策略不能上线

---

## 项目背景

### 当前状态 (v2.3)

- **代码库**: 已通过 `Trader_v23_cleanup.zip` 清理了 v2.2 死代码
- **删除的策略**: `regime`、`enhanced_exit`、1h 时间框架的所有配置
- **保留的策略**:
  - `macd_momentum` (4h, v2.0 回滚)
  - `triple_ema` (4h, enabled=false，样本不足)
  - `mean_reversion` (4h, 加趋势回避门)
- **新增**: `cross_sectional_momentum` 策略骨架 + `universe.py` + `cross_sectional_backtest.py`
- **成本假设已修正**: BTC 滑点 21 bps, ETH 26 bps
- **测试**: `tests/test_v23.py` 16 个测试全部通过

### 已知问题

- v2.0-v2.2 所有策略 walk-forward OOS 均不达标（最高 `mean_reversion 4h` fold 胜率 39.4%）
- BTC-ETH 相关性 0.827，组合不分散
- 仅用 OHLCV + 技术指标，没有 funding rate / on-chain / LOB 数据
- Walk-forward 仅给 1 条 OOS path，无法判断统计显著性

### 目标

到 Month 12 建立一个多策略组合：
```
50%  Funding Harvester (Phase 2A)
30%  Cross-Sectional Multi-Factor (Phase 3)
10%  Mean Reversion (已有)
10%  预留 on-chain / event (Phase 4)
```

总 Sharpe 目标 > 1.5，最大回撤 < 20%，PBO < 30%。

---

## 参考文档

阅读顺序：
1. `docs/DEPRECATED_FACTORS_AND_STRATEGIES.md` — 了解哪些策略已被证伪
2. `docs/RESEARCH_2025_SOTA_QUANT_ARCHITECTURES.md` — 2025-26 主流架构
3. `docs/ITERATION_ROADMAP_12M.md` — 本 TODO 的依据文档
4. `docs/README_v23_MIGRATION.md` — 迁移说明

### 关键论文引用

| 用途 | 论文 | 关键信息 |
|---|---|---|
| Purged CPCV | López de Prado 2018 *Advances in Financial ML* 第 7 章 | 金融 CV 正确做法 |
| PBO | Bailey et al 2015 *Probability of Backtest Overfitting* | 过拟合概率计算 |
| DSR | López de Prado 2018 *Deflated Sharpe Ratio* | 调整多次试验后的真实 Sharpe |
| Funding Arbitrage | 1Token 2026 Jan *Crypto Quant Strategy Index* | 业界基准 |
| BIS Carry | Schmeling-Schrimpf-Todorov 2025 BIS WP 1087 | 基差套利的风险分析 |
| CTREND Factor | Han et al JFQA Nov 2025 | ML 因子聚合方法 |
| Cross-Sectional Crypto | Liu-Tsyvinski-Wu 2022 JoF | 横截面基础 |

### 开源代码参考

- `sam31415/timeseriescv`: Purged/Combinatorial CV 实现
- `AI4Finance-Foundation/FinRL`: 金融 RL 框架（备用，不建议直接用）
- `thuml/Time-Series-Library`: 时序模型 benchmark

---

## 全局规则

### G1: 任何新策略的 "可上线" 硬门槛

必须**全部**满足才能进入 paper trading：

- [ ] 经过 Purged CPCV 验证（不是 walk-forward），至少 10 条 OOS paths
- [ ] CPCV 10 paths 中位数 Sharpe > 1.0
- [ ] PBO (Probability of Backtest Overfitting) < 30%
- [ ] Deflated Sharpe Ratio > 0.8
- [ ] 最大回撤 < 25%
- [ ] 每 fold 至少 10 笔交易
- [ ] 真实成本建模（BTC 21bps, ETH 26bps，或更保守）
- [ ] 通过所有单元测试

### G2: 代码规范

- 所有策略继承统一接口（`should_enter`, `check_exit`, `calc_position`, `on_trade_closed`）
- 配置使用 `@dataclass`，不用 dict
- 所有时间戳用 UTC
- 所有价格/数量用 float（不用 Decimal 避免性能问题，但要注意精度）
- 日志用 `utils.logger.get_logger()`
- 新增策略必须在 `alpha/strategy_registry.py` 注册

### G3: 测试要求

- 每个策略类必须有：
  - 入场逻辑测试
  - 退出逻辑测试
  - 仓位计算测试
  - 边界情况测试（数据缺失、极端价格等）
- 每个因子必须有：
  - 计算正确性测试
  - 数据不足时返回 None 测试
- 运行方式: `python -m pytest tests/ -v`

### G4: 数据规范

- 数据源优先级: 免费 API > 付费 API > 手动
- 所有数据必须 point-in-time（避免 survivorship bias / lookahead）
- 增量同步优先于全量重拉
- 数据存储在 `data/quant.db` (SQLite) 或 `data/parquet/` (大数据集)

### G5: 分支和提交

- 每个 Phase 一个分支: `phase-1-xs-momentum`, `phase-2a-funding`, etc.
- 每个 Task 一个 commit，commit message 格式: `[P1-T03] 具体描述`
- Decision Gate 之前 merge 到 main

---

## Phase 0: 环境准备 + 清理

### P0-T01: 解压 v23 并执行清理

**Effort**: 0.5 day  
**Depends on**: 无  
**Deliverable**: 干净的 v2.3 项目结构

**Steps**:
1. 备份当前项目: `tar -czf ../Trader_backup_$(date +%Y%m%d).tar.gz .`
2. 解压 `Trader_v23_cleanup.zip` 覆盖原项目
3. 执行 `bash scripts/cleanup_v23.sh`
4. 按 `docs/README_v23_MIGRATION.md` Step 4 手动修改以下文件:
   - `alpha/strategy_registry.py`: 移除对 `regime_adaptive_strategy` 和 `regime_allocator_v2` 的引用
   - 任何引用已删除文件的 `import` 都需清理
5. 跑 `grep -r "regime_adaptive_strategy\|regime_allocator_v2\|enhanced_exit" . --include="*.py"` 确认无残留
6. 运行测试: `python tests/test_v23.py`

**Acceptance**:
- [ ] `archive/v22_deprecated/` 下至少 5 个归档文件
- [ ] `python tests/test_v23.py` 通过全部 16 个测试
- [ ] `grep` 无残留 import
- [ ] `python -c "from alpha.strategy_registry import build_strategy; print('OK')"` 无错误

---

### P0-T02: 配置新的环境依赖

**Effort**: 0.5 day  
**Depends on**: P0-T01  
**Deliverable**: 所有后续阶段需要的依赖已安装

**Steps**:
1. 更新 `requirements.txt`:
```
# 核心
numpy>=1.24.0
pandas>=1.5.0
scipy>=1.10.0
PyYAML>=6.0
requests>=2.28.0
websockets>=12.0

# ML
scikit-learn>=1.2.0
lightgbm>=4.0.0

# 金融专用 (Phase 2B CPCV 需要)
# 尝试安装 sam31415/timeseriescv，如果失败就自己实现
# pip install git+https://github.com/sam31415/timeseriescv.git

# 测试
pytest>=7.0.0
pytest-cov>=4.0.0

# 可视化（可选）
matplotlib>=3.7.0
```

2. `pip install -r requirements.txt --break-system-packages`
3. 创建 `.env.example`:
```
# Binance API (Phase 2A 需要)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# CryptoQuant API (Phase 4 需要)
CRYPTOQUANT_API_KEY=

# CoinGlass API (Phase 2A 需要)
COINGLASS_API_KEY=
```

4. 添加 `.env` 到 `.gitignore`

**Acceptance**:
- [ ] `python -c "import lightgbm; import numpy; import pandas; import websockets; print('OK')"` 成功
- [ ] `.env.example` 存在且所有 Phase 都能找到需要的 API key 占位符
- [ ] `.gitignore` 包含 `.env`, `data/`, `logs/`, `__pycache__/`, `archive/`

---

## Phase 1: 横截面 Momentum MVP (Month 1)

**目标**: 验证"crypto 横截面 momentum 因子是否仍有 edge"。

### P1-T01: 拉取 Top 60 币种 5 年日线数据

**Effort**: 1 day  
**Depends on**: P0-T02  
**Deliverable**: `data/quant.db` 中 60 个 symbol 的完整日线

**Steps**:
1. 创建 `scripts/sync_universe_data.py`:
   - 使用 Binance 公开 API (`/api/v3/klines`), 不需要 API key
   - Symbol list 用 `data/universe.py` 中 `UniverseBuilder.get_all_symbols()` 返回的 60 个
   - 时间范围: 2020-01-01 到 今天
   - 时间框架: 1d
   - 保存到 `storage.py` 的 klines 表

2. 加入增量逻辑（已有数据跳过）
3. 处理 rate limit（每分钟 1200 请求）
4. 加入数据完整性校验：
   - 检查是否有跳天
   - 检查是否有 0 volume 连续天（可能是下架）

**Files**:
- `scripts/sync_universe_data.py` (新增)
- `data/storage.py` (仅调用，不改)

**Acceptance**:
- [ ] 至少 45 个 symbol 有 ≥ 4 年完整日线数据
- [ ] 数据库行数 ≥ 45 × 365 × 4 = 65,700 行
- [ ] `python -c "from data.storage import Storage; s = Storage('data/quant.db'); kl = s.get_klines('BTCUSDT', '1d', 2000); print(f'BTC rows: {len(kl)}')"` 输出 > 1500

**Reference**:
- Binance API docs: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data

---

### P1-T02: 验证 universe 构建器 point-in-time 正确性

**Effort**: 1 day  
**Depends on**: P1-T01  
**Deliverable**: `UniverseBuilder` 通过所有边界测试

**Steps**:
1. 写测试 `tests/test_universe.py`:
   - Test: 2021-06-01 时 universe 不包含 2022 年才上市的币种
   - Test: 同一日期多次调用返回一致结果（缓存正确）
   - Test: `turnover_smoothing` 过滤掉临时流动性尖峰（构造一个只在 1 天内成交激增的假币种）
   - Test: 流动性过滤 `min_volume_30d_usd` 生效

2. 如果 universe.py 有 bug 就修复

**Acceptance**:
- [ ] `tests/test_universe.py` 至少 5 个测试全部通过
- [ ] 手动打印 2022-01-01 和 2024-01-01 的 universe，确认不同（证明 point-in-time 生效）

---

### P1-T03: 运行横截面 Momentum MVP 回测

**Effort**: 0.5 day  
**Depends on**: P1-T02  
**Deliverable**: `analysis/output/xs_mom_*.json` 多个配置的回测结果

**Steps**:
1. `bash scripts/run_mvp_path_a.sh`
2. 如果运行中失败，根据错误修复 `cross_sectional_backtest.py`
3. 全部跑完后汇总结果到 `analysis/output/xs_mvp_summary.md`

**Acceptance**:
- [ ] 至少 6 个 `xs_mom_*.json` 文件生成 (不同 top-n × rebalance-days 组合)
- [ ] 每个 json 包含 `summary.sharpe_ratio` 字段
- [ ] `xs_mvp_summary.md` 有对比表格

**执行完后必须报告给用户**:
- 最佳配置是什么?
- Sharpe 范围是多少?
- 得到 EXCELLENT / GOOD / MARGINAL / WEAK / FAILED 哪个判决?

---

### P1-T04: 加入 inverse volatility weighting 的完整实现

**Effort**: 0.5 day  
**Depends on**: P1-T03  
**Deliverable**: 完整的仓位权重缩放逻辑

**Steps**:
1. 当前 `cross_sectional_momentum.py` 的 `apply_inverse_vol_weighting` 已经有骨架
2. 完善它：
   - 接收原始权重 dict 和 volatilities dict
   - 对多头和空头分别做 inverse vol scaling
   - 最终保证 `sum(|weights|) = 2.0` (200% gross exposure)
3. 在回测主循环中调用这个方法（已有调用，但可能有 bug）
4. 写单元测试

**Acceptance**:
- [ ] 给定多头和空头各 5 个币，最高波动币的权重应 < 最低波动币权重的 60%
- [ ] `sum(|weights|)` 在 [1.95, 2.05] 范围内
- [ ] `sum(positive_weights) ≈ 1.0` 和 `sum(negative_weights) ≈ -1.0`

---

### P1-T05: 加入 turnover 限制机制

**Effort**: 1 day  
**Depends on**: P1-T04  
**Deliverable**: 避免频繁换仓导致成本失控

**Steps**:
1. 在 `CrossSectionalConfig` 新增:
   - `max_turnover_per_rebalance: float = 0.5` (每次 rebalance 最多换仓 50%)
2. 在回测主循环中:
   - 计算新旧组合的 L1 差距
   - 如果超过阈值，只对最 "确信" 的 symbols 换仓
3. 实现思路: 按 |new_weight - old_weight| 降序排序，只换前 N 个直到满足 turnover 限制

**Acceptance**:
- [ ] 新增 config 字段生效
- [ ] 设置 `max_turnover_per_rebalance=0.3` 后，成本应降低至少 30%
- [ ] Sharpe 不因此大幅下降（允许下降 < 20%）

---

### P1-T06: MVP 结果分析与下一步决策

**Effort**: 0.5 day  
**Depends on**: P1-T05  
**Deliverable**: `analysis/output/phase1_final_report.md`

**Steps**:
1. 用最佳配置重新跑一次（启用 P1-T04/T05 改进）
2. 生成分析报告，包含：
   - 最终 Sharpe / 年化收益 / MDD / 胜率
   - 按年分解的收益率
   - 因子重要性（各因子独立贡献）
   - 最赢和最亏的 5 个 rebalance 期间
3. 按 `docs/ITERATION_ROADMAP_12M.md` §0 决策树给出"下一步建议"

**Acceptance**:
- [ ] 报告包含上述所有部分
- [ ] 明确给出 Sharpe 数字和判决等级

🚪 **Checkpoint**: 完成后停下来报告给用户，根据 Sharpe 等级决定是否进入 Phase 3（或直接跳到 Phase 2A）。

---

## Phase 2A: Funding Harvester (Month 2-3，并行)

**重要**: 这个 Phase **无论 Phase 1 结果如何都要做**。2025 业界数据显示 funding arbitrage 是最稳定的 crypto alpha。

### P2A-T01: 拉取 Binance Futures 历史 funding rate 数据

**Effort**: 1 day  
**Depends on**: P0-T02  
**Deliverable**: 3 年 funding rate 历史数据入库

**Steps**:
1. 创建 `scripts/sync_funding_data.py`:
   - Binance Futures API: `/fapi/v1/fundingRate`
   - 拉取 BTC/ETH 及 Phase 1 universe 中前 30 个币种
   - 时间范围: 2023-01-01 到 今天 (8 小时一次，每天 3 条记录)
   - 每年约 1095 条 × 30 币种 × 3 年 ≈ 100K 行

2. 创建新表 `funding_rates`:
```sql
CREATE TABLE funding_rates (
    symbol TEXT,
    funding_time INTEGER,  -- unix ms
    funding_rate REAL,
    mark_price REAL,
    PRIMARY KEY (symbol, funding_time)
);
CREATE INDEX idx_funding_symbol_time ON funding_rates(symbol, funding_time);
```

3. 在 `data/storage.py` 加入 `get_funding_rates(symbol, start_ms, end_ms)` 接口

**Files**:
- `scripts/sync_funding_data.py` (新增)
- `data/storage.py` (扩展)

**Acceptance**:
- [ ] funding_rates 表包含 ≥ 30 个 symbol 的数据
- [ ] BTC 和 ETH 至少有 3 年数据（≥ 3000 条记录）
- [ ] `storage.get_funding_rates('BTCUSDT', ...)` 返回 DataFrame

**Reference**: https://binance-docs.github.io/apidocs/futures/en/#funding-rate-history

---

### P2A-T02: Funding Harvester 策略类

**Effort**: 1.5 days  
**Depends on**: P2A-T01  
**Deliverable**: `alpha/funding_harvester.py` 可运行的策略类

**Steps**:
1. 创建 `alpha/funding_harvester.py`，结构:
```python
@dataclass
class FundingHarvesterConfig:
    # 入场阈值
    min_funding_rate: float = 0.0001  # 0.01% 单次, 年化约 10.95%
    min_funding_duration_h: int = 16   # 至少持续 2 个 funding cycle
    
    # 仓位
    notional_per_trade_usd: float = 1000.0
    max_concurrent_positions: int = 10
    leverage: float = 1.0  # 现货 = 1x, 不加杠杆
    
    # 退出
    exit_on_funding_flip: bool = True       # funding 翻负立即平仓
    exit_buffer_rate: float = -0.00005      # 或 funding < -0.005% 时平
    
    # 成本
    spot_fee_bps: float = 10.0              # 现货吃单
    perp_fee_bps: float = 5.0               # 永续挂单
    slippage_bps: float = 10.0              # 每边

class FundingHarvester:
    """
    Delta-neutral funding rate harvesting:
      Long spot + Short perp
    当 funding > 0 时收取 funding，方向中性
    """
    
    def should_enter(self, funding_snapshot: pd.Series, state: dict) -> bool:
        """当前时点是否应该开仓"""
        ...
    
    def should_exit(self, current_funding: float, position: dict) -> tuple[bool, str]:
        """是否应该平仓"""
        ...
    
    def calc_position(self, capital: float) -> tuple[float, float]:
        """(spot_qty, perp_qty) - 保证 delta = 0"""
        ...
    
    def compute_pnl(self, position: dict, current_prices: dict, 
                    funding_received: float) -> float:
        """计算 PnL: 价格变动对冲掉 + funding 收取 - 成本"""
        ...
```

2. 关键逻辑:
   - Spot 做多，Perp 做空，size 严格相等（delta neutral）
   - 每 8 小时检查 funding rate
   - funding 翻负立即平仓
   - 不开单边仓位

3. 写单元测试 `tests/test_funding_harvester.py`

**Acceptance**:
- [ ] 类可实例化，所有方法有基础实现
- [ ] 至少 5 个单元测试通过，包括:
  - 价格涨跌时 delta 保持中性
  - funding 正时开仓，负时平仓
  - 成本正确计算（spot + perp 双边 fee + slippage）
- [ ] 注册到 `strategy_registry.py`

**Reference**: 
- 1Token 2026 Jan Crypto Quant Strategy Index
- `docs/ITERATION_ROADMAP_12M.md` §2.2 策略 B1

---

### P2A-T03: Funding Harvester 回测引擎

**Effort**: 2 days  
**Depends on**: P2A-T02  
**Deliverable**: `funding_harvester_backtest.py` + 回测结果 JSON

**Steps**:
1. 创建 `funding_harvester_backtest.py`:
   - 读取 funding_rates 和 klines
   - 按 8 小时 bar 推进
   - 调用 `FundingHarvester` 逻辑
   - 产出权益曲线 + 每笔交易记录

2. 关键点：
   - 必须同时模拟 spot 和 perp 的价格演化
   - funding 收取时点精确（每 8 小时）
   - 强平保护: 即使 perp 爆仓，spot 仍然有价值

3. 输出格式与 `cross_sectional_backtest.py` 一致:
   - `summary.sharpe_ratio`, `max_drawdown_pct`, `annualized_return_pct`
   - `verdict` 字段

**Files**:
- `funding_harvester_backtest.py` (新增)
- `scripts/run_funding_harvester.sh` (新增，一键跑)

**Acceptance**:
- [ ] 跑完 3 年历史后输出 JSON
- [ ] 年化收益在 5-30% 区间（符合业界数据）
- [ ] MDD < 10%（funding strategy 应该非常稳）
- [ ] Sharpe > 1.5（业界基准应 > 2.0）

**Reference**: `docs/ITERATION_ROADMAP_12M.md` §2 Phase 2B

---

### P2A-T04: Funding 信号的敏感度扫描

**Effort**: 1 day  
**Depends on**: P2A-T03  
**Deliverable**: 最优参数组合 + 敏感度报告

**Steps**:
1. 扫描 `FundingHarvesterConfig` 的关键参数:
   - `min_funding_rate`: [0.00005, 0.0001, 0.00015, 0.0002]
   - `min_funding_duration_h`: [8, 16, 24, 48]
   - `max_concurrent_positions`: [5, 10, 15, 20]

2. 使用 grid search, 总共 4 × 4 × 4 = 64 次回测

3. 生成 `analysis/output/funding_sensitivity.json`

4. **判决 good_ratio**: 达到最优 70%+ 的参数组合占比
   - good_ratio > 0.5 → ROBUST
   - good_ratio 0.3-0.5 → MODERATE  
   - good_ratio < 0.3 → FRAGILE（需要重新思考）

**Acceptance**:
- [ ] 敏感度 JSON 包含 64 个回测结果
- [ ] 报告最优参数组合
- [ ] 每个参数的 good_ratio 计算并输出

---

### P2A-T05: Paper Trading 接入准备

**Effort**: 2 days  
**Depends on**: P2A-T04  
**Deliverable**: 可以连接 Binance testnet 的执行模块

**Steps**:
1. 创建 `execution/binance_executor.py`:
   - Binance Testnet 连接（spot 和 futures 两个端点）
   - 下单接口: `place_spot_order()`, `place_perp_order()`
   - 持仓查询: `get_spot_balance()`, `get_perp_positions()`
   - 实时 funding rate 订阅: WebSocket `!markPrice@arr`

2. 创建 `main_funding.py` 主循环:
```python
while True:
    snapshot = executor.get_current_snapshot()  # prices + funding
    
    # 处理现有持仓
    for pos in current_positions:
        should_exit, reason = harvester.should_exit(pos, snapshot)
        if should_exit:
            executor.close_position(pos)
    
    # 寻找新机会
    for symbol in universe:
        if harvester.should_enter(snapshot[symbol], state):
            executor.open_delta_neutral_position(symbol, size)
    
    sleep(60)  # 每分钟检查一次
```

3. **重要安全措施**:
   - 默认只连 testnet, 需要 `--live` flag 才连主网
   - 最大单笔金额硬编码限制（`$100` for testing）
   - 每天最大亏损限制（`-$50`）
   - 所有订单都走 `dry_run=True` 先打印再执行

**Acceptance**:
- [ ] 连接 Binance testnet 成功，能下单和查询
- [ ] 至少有 1 个 dry_run 的完整 open-close 周期
- [ ] 不会因为 WebSocket 断线崩溃（自动重连）
- [ ] 安全限制生效（超过金额/亏损时拒绝下单）

**警告**: 不要在未经测试的情况下上主网。

---

## Phase 2B: CPCV 验证框架 (Month 2-3，并行)

**重要**: 这个阶段完成后，**所有历史策略都要重新审一遍**。很可能会发现一些被错误放弃的，和一些被错误认为 work 的。

### P2B-T01: 实现 Purged K-Fold CV

**Effort**: 1.5 days  
**Depends on**: P0-T02  
**Deliverable**: `validation/purged_cv.py`

**Steps**:
1. 创建 `validation/` 目录
2. 实现 `PurgedKFold` 类:
```python
class PurgedKFold:
    """
    Lopez de Prado 2018 Ch7 的 Purged K-Fold CV
    
    Args:
        n_splits: K 折数
        embargo_pct: 训练/测试之间的 embargo 窗口比例 (默认 1%)
    """
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(self, X: pd.DataFrame) -> Iterator[tuple[np.array, np.array]]:
        """yields (train_indices, test_indices)"""
        ...
```

3. **关键**: 
   - Purging: 从训练集移除与测试集有时间重叠的样本
   - Embargoing: 测试集结束后的 `embargo_pct * len(X)` 个样本不能进入训练集
   - 必须按时间顺序切，不是随机切

4. 参考开源实现: `sam31415/timeseriescv` 的 `CombPurgedKFoldCV` 类

5. 写单元测试:
   - 5 fold 时产生 5 个 split
   - 验证训练和测试集无时间重叠
   - 验证 embargo 生效

**Files**:
- `validation/__init__.py`
- `validation/purged_cv.py`
- `tests/test_purged_cv.py`

**Acceptance**:
- [ ] `PurgedKFold(n_splits=5).split(df)` 产生 5 个 (train, test) 对
- [ ] 所有测试通过
- [ ] 训练集大小约 80%, 测试集约 20%

**Reference**: 
- López de Prado 2018 Ch7
- https://github.com/sam31415/timeseriescv

---

### P2B-T02: 实现 Combinatorial Purged CV (CPCV)

**Effort**: 2 days  
**Depends on**: P2B-T01  
**Deliverable**: `validation/cpcv.py` 可生成多条 OOS 路径

**Steps**:
1. 实现 `CombinatorialPurgedCV`:
```python
class CombinatorialPurgedCV:
    """
    N 个 group, 每次选 k 个作为 test set
    产生 C(N, k) 个 splits，组合成 N * k / N = k * C(N,k)/N 条 paths
    
    典型: N=10, k=2 -> 45 splits -> 9 paths
    """
    def __init__(self, n_groups: int = 10, n_test_groups: int = 2, 
                 embargo_pct: float = 0.01):
        ...
    
    def split(self, X) -> List[tuple]:
        """返回所有 (train_idx, test_idx) 组合"""
        ...
    
    def backtest_paths(self, X, strategy_fn) -> List[pd.Series]:
        """
        对每条 OOS path 跑一遍策略，返回多条权益曲线
        核心：每个 group 可以出现在多条 path 上
        """
        ...
```

2. 难点：如何把 splits 组合成 paths
   - 参考 Wikipedia: https://en.wikipedia.org/wiki/Purged_cross-validation
   - 每个 group 在不同 split 中扮演 test 的次数 = k * C(N-1, k-1) / C(N, k) = k/N
   - 每条 path 由每个 group 的某一次 test 结果组合而成

3. 写单元测试验证 paths 数量和覆盖

**Files**:
- `validation/cpcv.py`
- `tests/test_cpcv.py`

**Acceptance**:
- [ ] N=10, k=2 的配置下产生 45 个 splits 和 9 条 paths
- [ ] 每个时间点在每条 path 上只出现一次
- [ ] 所有 path 覆盖整个时间序列

**Reference**: 
- Arian-Norouzi-Seco 2024 paper
- Towards AI blog: https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method

---

### P2B-T03: 实现 Deflated Sharpe Ratio (DSR)

**Effort**: 0.5 day  
**Depends on**: P2B-T02  
**Deliverable**: `validation/dsr.py`

**Steps**:
1. 实现公式 (López de Prado 2018 *Deflated Sharpe Ratio*):
```python
def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,          # 试过多少次才找到这个 Sharpe
    n_samples: int,         # 样本量
    skew: float,            # 收益偏度
    kurt: float,            # 收益峰度
) -> float:
    """
    DSR: 扣除多次试验 bias 和非正态性后的 Sharpe
    
    Returns:
        调整后的 Sharpe ratio
    """
    ...

def probability_of_backtest_overfitting(
    sharpe_ratios: np.array,  # N 条 path 的 Sharpe
    n_trials: int,
) -> float:
    """
    PBO: 选出最佳策略在 OOS 上表现低于中位数的概率
    
    Returns:
        PBO ∈ [0, 1], 越小越好
    """
    ...
```

2. 写单元测试

**Acceptance**:
- [ ] 给 SR=2.0, n_trials=100 的情况，DSR 应显著低于 2.0
- [ ] 给正态分布收益，DSR ≈ naive SR
- [ ] 给高 kurtosis 收益，DSR < naive SR

---

### P2B-T04: 用 CPCV 重新审查所有历史策略

**Effort**: 2 days  
**Depends on**: P2B-T03  
**Deliverable**: `analysis/output/cpcv_audit_report.md`

**Steps**:
1. 对以下策略用 N=10, k=2 的 CPCV 重新跑：
   - `macd_momentum` 4h (v2.3 回滚版)
   - `triple_ema` 4h
   - `mean_reversion` 4h (v2.3 带趋势回避)
   - `cross_sectional_momentum` 最佳配置（来自 P1）
   - `funding_harvester` (来自 P2A)

2. 每个策略产出:
   - 9 条 OOS path 的 Sharpe 分布
   - 中位数 Sharpe
   - PBO
   - DSR
   - 判决 (PASS / CONDITIONAL / FAIL)

3. 生成对比报告

**Acceptance**:
- [ ] 5 个策略全部跑完 CPCV
- [ ] 每个策略有中位数 Sharpe + PBO + DSR
- [ ] 报告中明确哪些策略通过 G1 上线门槛

🚪 **Checkpoint**: 完成后报告给用户。这会发现一些意外的结果。

---

### P2B-T05: 把 CPCV 集成到主回测流水线

**Effort**: 1 day  
**Depends on**: P2B-T04  
**Deliverable**: 新的 CI-style 验证脚本

**Steps**:
1. 创建 `scripts/run_cpcv_validation.sh`:
   - 对所有 enabled 策略跑 CPCV
   - 生成统一报告
   - 如果任何策略 PBO > 40%，标记为 FAIL

2. 更新 `docs/DEPRECATED_FACTORS_AND_STRATEGIES.md`，加入新的"基于 CPCV 已证伪"的策略

3. 修改 `scripts/run_all_phases.sh`，加入 CPCV 作为最后一步

**Acceptance**:
- [ ] 新脚本可一键跑完所有 CPCV 验证
- [ ] 输出符合 G1 标准的 pass/fail 判决

---

## 🚪 Decision Gate 1 (Month 3 末)

### 必须回答的问题

1. **Phase 1 MVP 的 CPCV 中位数 Sharpe 是多少？**
2. **Phase 2A Funding Harvester 的 Sharpe 是多少？**
3. **哪些策略的 PBO < 30% 达到 G1 上线标准？**
4. **BTC/ETH 相关性是否仍然 0.8+？（是否需要加 on-chain 因子？）**

### 决策矩阵

| Scenario | 动作 |
|---|---|
| Funding Sharpe > 2.0 AND Momentum Sharpe > 1.2 | 进入 Phase 3 (多因子) |
| Funding Sharpe > 2.0 AND Momentum Sharpe < 1.0 | 跳过 Phase 3, 优化 Funding + 上 Paper Trading |
| Funding Sharpe < 1.5 | 停下来调试 Funding, 这是 crypto 最稳的策略, 如果它不 work 一定有 bug |
| 所有策略 PBO > 40% | 回到数据层面, 可能有 lookahead bias |

**停下等用户决策**

---

## Phase 3: 多因子扩展 (Month 4-5)

**前提**: Phase 1 MVP Sharpe ≥ 1.2 或 Funding Harvester 已稳定。

### P3-T01: 订阅 CryptoQuant + CoinGlass API

**Effort**: 0.5 day  
**Depends on**: Decision Gate 1 通过  
**Deliverable**: API key 存入 `.env`, 连通性验证

**Steps**:
1. 注册 CryptoQuant (基础版 $99/月)
2. 注册 CoinGlass API ($49/月)
3. 写连通性测试:
```python
# scripts/test_api_connectivity.py
from cryptoquant import Client as CQ
cq = CQ(api_key=os.getenv('CRYPTOQUANT_API_KEY'))
resp = cq.get('/v1/btc/exchange-flows/inflow')
assert resp['status'] == 'success'
```

**Acceptance**:
- [ ] CryptoQuant API 可返回 BTC exchange netflow 数据
- [ ] CoinGlass API 可返回 funding rate / OI 数据

---

### P3-T02: 实现 Funding Rate Z-score 因子

**Effort**: 1 day  
**Depends on**: P3-T01 + P2A-T01  
**Deliverable**: 新因子加入横截面策略

**Steps**:
1. 在 `alpha/cross_sectional_momentum.py` 的 `compute_factors` 方法中添加:
```python
def _compute_funding_zscore(self, symbol: str, as_of_ts: int) -> float:
    """
    过去 30 天 funding rate 的 z-score
    负 z-score (funding 偏负) → 预期反弹 → 做多
    """
    ...
```

2. 添加到 `factor_weights`:
```python
factor_weights = {
    ...,
    "funding_z": -1.2,  # 负 → 做多 (反向指标)
}
```

3. 跑 CPCV 验证加入这个因子后 Sharpe 变化

**Acceptance**:
- [ ] 因子计算正确（有单元测试）
- [ ] 加入后 CPCV 中位 Sharpe 提升 ≥ 0.2
- [ ] PBO 不显著上升（变化 < 5pp）

---

### P3-T03: 实现 Basis Carry 因子

**Effort**: 1 day  
**Depends on**: P3-T02  
**Deliverable**: 季度期货基差因子

**Steps**:
1. 拉取 Binance 季度期货数据 (`BTCUSDT_250328`, `ETHUSDT_250328` 等):
```python
# scripts/sync_quarterly_futures.py
```

2. 实现因子:
```python
def _compute_basis_z(self, symbol: str, as_of_ts: int) -> float:
    """
    (quarterly_futures_price - spot_price) / spot_price
    年化后的 basis carry
    正基差 (contango) → 可以 short future + long spot 套利
    """
    ...
```

3. 加入 factor_weights

**Acceptance**:
- [ ] 季度期货数据表创建并入库
- [ ] 因子计算正确
- [ ] CPCV 中位 Sharpe 提升或持平

**Reference**: BIS WP 1087 (Schmeling-Schrimpf-Todorov 2025)

---

### P3-T04: 实现 Exchange Netflow 因子（CryptoQuant）

**Effort**: 1 day  
**Depends on**: P3-T01  
**Deliverable**: 交易所净流入因子

**Steps**:
1. 拉取 CryptoQuant 的 exchange netflow 数据 (BTC/ETH 优先)
2. 计算因子:
```python
def _compute_exchange_netflow_z(self, symbol: str, as_of_ts: int) -> float:
    """
    24h exchange netflow 的 7d z-score
    正 netflow (流入交易所) → 抛压 → 做空
    """
    ...
```

3. 在因子权重中加入 `-0.5`（反向指标）

**注意**: CryptoQuant 对山寨币覆盖有限。因子构造时如果某 symbol 没数据，赋 `NaN`，在排序时忽略。

**Acceptance**:
- [ ] BTC 和 ETH 有 1 年以上 netflow 数据
- [ ] 因子可计算
- [ ] 加入后 CPCV Sharpe 不下降

---

### P3-T05: 多因子 LightGBM 聚合 (CTREND 风格)

**Effort**: 2 days  
**Depends on**: P3-T02 ~ P3-T04  
**Deliverable**: `alpha/ml_alpha_aggregator.py`

**Steps**:
1. 创建 `alpha/ml_alpha_aggregator.py`:
```python
class LightGBMAlphaAggregator:
    """
    参考 CTREND (Han et al JFQA 2025):
    用 LightGBM 把所有因子聚合成一个综合 alpha 信号
    
    预测目标: 5d forward return
    输入: 全部因子的当前值
    """
    
    def train(self, factors: pd.DataFrame, returns: pd.DataFrame,
              forward_days: int = 5):
        ...
    
    def predict(self, factors: pd.DataFrame) -> pd.Series:
        """返回每个 symbol 的预测 forward return"""
        ...
```

2. 关键实现点:
   - 使用 `purged_cv` 做内部 CV 调参
   - 特征: 所有因子的当前值 + 一阶差分
   - 标签: 5d forward return
   - 防止数据泄漏: 训练数据和预测数据之间至少 embargo 5 天

3. 在 `cross_sectional_backtest.py` 中加一个 mode:
   - `--mode=equal_weight`: 现有的等权综合
   - `--mode=ml_aggregated`: 用 LightGBM 预测综合

4. CPCV 对比两种方法

**Acceptance**:
- [ ] ML 模式可运行
- [ ] CPCV 显示 ML 模式 Sharpe > equal_weight 模式至少 0.2
- [ ] 特征重要性合理（funding_z 应该是 top 3）

**Reference**: Han et al *A Trend Factor for the Cross Section of Cryptocurrency Returns*, JFQA 60(7), Nov 2025.

---

### P3-T06: 集成到多策略组合器

**Effort**: 1 day  
**Depends on**: P3-T05 + P2A-T05  
**Deliverable**: `portfolio/multi_strategy_allocator.py`

**Steps**:
1. 创建 portfolio 模块:
```python
# portfolio/multi_strategy_allocator.py
class MultiStrategyAllocator:
    """
    组合多个策略:
      - funding_harvester: 50% 资金
      - xs_momentum (ML aggregated): 30%
      - mean_reversion: 10%
      - 现金缓冲: 10%
    
    每日 rebalance 各策略的资金分配
    """
    ALLOCATION = {
        'funding_harvester': 0.50,
        'xs_momentum_ml': 0.30,
        'mean_reversion': 0.10,
        'cash_buffer': 0.10,
    }
    
    def allocate(self, total_capital: float) -> dict[str, float]:
        ...
    
    def compute_correlation_matrix(self, lookback_days: int = 60) -> pd.DataFrame:
        """计算策略间相关性, 如果 > 0.7 报警"""
        ...
```

2. 跑综合回测，对比单策略 vs 组合

**Acceptance**:
- [ ] 组合回测可运行
- [ ] 组合 Sharpe > 任一单策略 Sharpe
- [ ] 策略间相关性矩阵显示 < 0.6 (verify 分散化)
- [ ] 组合最大回撤 < 各策略最大回撤的加权平均

---

## Phase 4: On-chain 因子 (Month 6-9)

**前提**: Phase 3 多因子组合 Sharpe > 1.5，且 paper trading 至少 1 个月正收益。

### P4-T01: 深度拉取 CryptoQuant 全指标数据

**Effort**: 2 days  
**Depends on**: Paper trading 已运行 30+ 天  
**Deliverable**: 完整的 on-chain 数据湖

**Steps**:
1. 扩展 `scripts/sync_onchain_data.py` 拉取以下指标:
   - SOPR (Spent Output Profit Ratio)
   - MVRV Z-Score
   - NUPL (Net Unrealized P&L)
   - Whale transaction count
   - Miner outflow
   - Stablecoin supply ratio
   - Exchange whale ratio (CryptoQuant 独有)

2. 所有数据存到 `onchain_metrics` 表，按日频

**Acceptance**:
- [ ] 至少 7 个 on-chain 指标入库
- [ ] BTC 覆盖 ≥ 3 年，ETH 覆盖 ≥ 2 年
- [ ] 数据完整性验证通过（无明显跳天）

---

### P4-T02: 单因子 IC 分析

**Effort**: 1 day  
**Depends on**: P4-T01  
**Deliverable**: `analysis/output/onchain_factor_ic.md`

**Steps**:
1. 对每个 on-chain 指标:
   - 计算与 BTC 7d forward return 的 Information Coefficient
   - Rolling 180 天的 IC 稳定性
   - 分位数收益差（top - bottom quintile）

2. 筛选标准:
   - |IC| > 0.05（静态）
   - Rolling IC 标准差 < 0.10（稳定）

**Acceptance**:
- [ ] 每个因子有 IC 数据
- [ ] 识别出 3-5 个有效 on-chain 因子
- [ ] 明确列出 IC 不达标的因子（不进入下一步）

⚠️ **关键警告**: on-chain 因子的 IC 在 2023 年后普遍衰减。如果 2023+ 数据上 IC < 0.03，**不要用 2020 年数据复现的结果自欺欺人**。结构已变，historical edge 可能不存在了。

---

### P4-T03: 加入 on-chain 因子到 ML aggregator

**Effort**: 1 day  
**Depends on**: P4-T02  
**Deliverable**: 扩展的 LightGBM 模型

**Steps**:
1. 把 P4-T02 筛选出的 on-chain 因子加入 LightGBM 训练
2. CPCV 对比：
   - 仅价格/衍生品因子
   - + on-chain 因子
3. 报告特征重要性变化

**Acceptance**:
- [ ] CPCV 中位 Sharpe 提升 ≥ 0.15（否则不值得接入付费 API）
- [ ] on-chain 因子在特征重要性 top 10 中至少占 2 个

**决策**: 如果提升不足 0.15，考虑取消 CryptoQuant 订阅，省钱。

---

### P4-T04: 市场 regime 分类器

**Effort**: 2 days  
**Depends on**: P4-T03  
**Deliverable**: `alpha/regime_classifier.py`

**Steps**:
1. 用 on-chain + 市场数据构造 regime 分类:
   - Bull (NUPL > 0.5, 高 RSI)
   - Bear (NUPL < 0, 持续下跌)
   - Accumulation (低 NUPL, exchange netflow < 0)
   - Distribution (高 NUPL, exchange netflow > 0)

2. 每个 regime 下评估各策略的表现
3. 动态调整 `MultiStrategyAllocator` 的权重:
   - Bull: 加码 momentum 到 40%
   - Bear: 减少 momentum 到 15%, 加码 funding 到 60%
   - Accumulation: mean reversion 加码到 20%
   - Distribution: 全部减仓 30%

**Acceptance**:
- [ ] Regime 分类器可分类每一天
- [ ] 动态配置比固定配置 Sharpe 提升 ≥ 0.2
- [ ] 切换不会太频繁（一年 < 20 次）

---

## 🚪 Decision Gate 2 (Month 9 末)

### 评估标准

1. **综合策略组合的 CPCV 中位 Sharpe**
2. **Paper trading 6+ 月的实际结果**
3. **是否有明显的 on-chain edge?**
4. **本项目已经花了多少时间 vs 产出了多少 IRR?**

### 决策矩阵

| Scenario | 动作 |
|---|---|
| 综合 Sharpe > 1.8 AND Paper 实际 > 1.5 | 进入 Phase 5 (LOB) 或放大资金 |
| 综合 Sharpe 1.2-1.8 | 停在当前状态，深耕当前策略 |
| 综合 Sharpe < 1.2 | 考虑暂停, 或大幅简化回到 Funding only |

**停下等用户决策**

---

## Phase 5: LOB 做市 (Month 10-12，可选)

**前提**: Phase 4 成功 + 有低延迟基础设施预算 + 资金 > $50K。

### P5-T01: 评估基础设施需求

**Effort**: 1 day  
**Depends on**: Decision Gate 2 通过  
**Deliverable**: 基础设施成本估算

**需要评估**:
- VPS 位置（必须在 Binance 东京/AWS Tokyo）
- 预算（低延迟 VPS $200-$500/月）
- 网络延迟（目标 <10ms to Binance)
- 是否需要 C++/Rust 重写

**如果评估下来 ROI < 预期，直接跳过 Phase 5。**

---

### P5-T02: 订阅 Binance Futures L2 LOB 数据

**Effort**: 2 days  
**Depends on**: P5-T01  
**Deliverable**: 实时 L2 订阅 + 历史数据

**Steps**:
1. Binance Futures WebSocket: `@depth20@100ms`
2. 存储到 Parquet (每天一个文件)
3. 计算基础 LOB 特征:
   - Order Flow Imbalance (OFI)
   - Micro price
   - Spread
   - Depth imbalance

**Acceptance**:
- [ ] 可实时接收 L2 数据
- [ ] 1 天数据 < 1GB 存储
- [ ] 特征计算正确

**Reference**: Briola-Aste 2026 paper

---

### P5-T03: 训练 LOB 预测模型 (XGBoost)

**Effort**: 3 days  
**Depends on**: P5-T02  
**Deliverable**: 可部署的 XGBoost 模型

**Steps**:
1. 按 Briola 2026 和 Wang 2025 的方法:
   - 标签: 10 秒后 mid-price 变动方向 (up/flat/down)
   - 特征: 40 个 LOB 特征 + 前 10 秒时序
2. XGBoost 训练（不用深度网络，因为 Wang 2025 证明 XGBoost 更好）
3. 关键: **跨币种 feature 稳定性验证**（Briola 2026 核心结论）

**Acceptance**:
- [ ] 模型准确率 > 55%（binary up/down）
- [ ] 推理延迟 < 1ms
- [ ] 跨 BTC/ETH 的 feature importance 相关性 > 0.7

**Reference**:
- Wang 2025 (arxiv 2506.05764)
- Briola 2026 (arxiv 2602.00776)

---

### P5-T04: 简单做市策略 MVP

**Effort**: 3 days  
**Depends on**: P5-T03  
**Deliverable**: Paper trading 的做市 bot

**Steps**:
1. 简单逻辑:
   - 挂买单 @ best_bid - 1 tick
   - 挂卖单 @ best_ask + 1 tick  
   - 根据 XGBoost 预测调整挂单方向 bias
   - Inventory risk 限制 (最大持仓 $1000)
2. 每次成交后重新下单
3. 实时监控 P&L

**警告**: Briola 2026 Fig 16 展示 2025-10-10 flash crash 中做市 repeatedly adversely selected，损失巨大。必须有 **circuit breaker**:
- 3 分钟内亏损 > $100 停机
- spread 超过 5 bps 停机

**Acceptance**:
- [ ] 在 Binance Testnet 跑 1 周
- [ ] 未触发 circuit breaker
- [ ] 日 PnL 正向（即使只是几美元）

---

## 持续性任务

### C-T01: 月度策略健康检查

**频率**: 每月 1 次  
**Deliverable**: `analysis/monthly_health/YYYY_MM.md`

**检查项**:
- [ ] 所有实盘策略的 live Sharpe vs backtest Sharpe 差距
- [ ] 因子 IC 是否衰减（与 1 个月前对比）
- [ ] 相关性矩阵是否改变
- [ ] 是否有新的业界 paper 需要 incorporate

---

### C-T02: 季度 CPCV 重新验证

**频率**: 每季度 1 次  
**Deliverable**: 更新的 `analysis/output/cpcv_audit_report_QX.md`

**Steps**:
1. 用最新数据重新跑所有策略的 CPCV
2. 如果任何策略 PBO 上升到 > 40%，**立即停用**
3. 更新 `docs/DEPRECATED_FACTORS_AND_STRATEGIES.md`

---

### C-T03: 资金管理规则

**永远遵守**:
- 单笔交易不超过总资金 5%
- 单策略不超过总资金 60%
- 当 rolling 30d Sharpe < 0.5 时减仓 50%
- 当总回撤 > 20% 时停机审查

---

## 📌 立即开始的第一步

从 `P0-T01: 解压 v23 并执行清理` 开始。执行后停下来汇报给用户，由用户确认后再做下一个。

```bash
# 开始的命令
cd /path/to/Trader
# 读取本文档
cat TODO.md | head -200
# 执行 P0-T01
bash scripts/cleanup_v23.sh
```

---

**文档版本**: v1.0 (2026-04-15)  
**维护**: 每个 Phase 结束后更新 Decision Gate 的实际结果  
**总计划时长**: 12 个月  
**总预算**: $3,000 - $6,000（数据订阅 + VPS，不含本金）  
**最终目标**: 多策略组合 Sharpe > 1.5，MDD < 20%，PBO < 30%

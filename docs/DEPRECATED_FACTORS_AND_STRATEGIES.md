# Trader — 失效策略与因子沉淀（v2.2 清理文档）

> **文档目的**：将 v2.0 → v2.1 → v2.2 三轮迭代中经 walk-forward 验证**不 work** 的策略、因子、参数记录在此，避免未来重复踩坑；并给出代码层面的删除/简化清单。
>
> **数据来源**：`analysis/output/wf_*.json`、`sensitivity_*.json`、`phase1_diagnose_summary.json`（2026-04 最后一次全量 WF）。

---

## 1. 结论一览（上线门槛：fold 胜率 ≥ 50% 且 OOS PnL > 0 且 trades ≥ 50）

| 策略 / 配置 | OOS PnL | trades | fold 胜率 | 结论 | 处置 |
|---|---:|---:|---:|---|---|
| `regime` 4h (RegimeAdaptive) | 0.0 | **0** | 0.0% | 过滤器叠加导致**完全不入场**，策略等于不存在 | **删除** |
| `mean_reversion` 1h | −3,015.4 | 159 | 25.0% | 高频 + 42bps 双边成本 = 结构性亏损 | **删除 1h 配置** |
| `macd_momentum` 1h | −790.0 | 54 | 7.7% | 同上，1h 在当前成本下不可行 | **删除 1h 配置** |
| `triple_ema` 4h (v2.2) | +1,519.2 | **11** | 15.2% | PnL 由 2-3 笔极端赢家驱动，统计不显著 | **降级为候选** |
| `macd_momentum` 4h (v2.2) | +57.6 | 12 | 9.1% | 三层过滤器把 92 笔压到 12 笔，edge 被扼杀 | **回滚到 v2.0 过滤器** |
| `mean_reversion` 4h | −425.2 | 59 | **39.4%** | 唯一有系统性 edge 迹象，但在强趋势行情中翻车 | **保留 + 加趋势回避门** |

**没有任何策略达到上线门槛。** 最接近的是 `mean_reversion` 4h。

---

## 2. 已证伪的策略（逐项说明）

### 2.1 `RegimeAdaptiveStrategy` — 完全失效

**症状**：33 个 fold，全部 0 笔交易。

**根因**：v2.2 把 regime gate（ADX ≥ 22）叠加到一个已经需要 regime 判断才入场的策略上，形成**两次 AND**，入场条件几乎不可能同时满足。

**判决**：这个策略在 v2.2 框架下没有任何观察值，**无法被验证也无法被调参**。删除。

**对应文件**：
- `alpha/regime_adaptive_strategy.py`
- `alpha/regime_allocator_v2.py`（依赖上者）
- `config/settings.yaml` 中 `strategy.regime` 相关段

---

### 2.2 1h 时间框架的所有策略 — 成本不可战胜

**数据**：
- `mean_reversion` 1h：159 trades，PnL −3015，每笔平均亏损 19 USDT
- `macd_momentum` 1h：54 trades，PnL −790

**根因**（phase1 诊断已经给出明确证据）：
- BTC 真实双边成本 ≈ **42 bps**（20.89 slippage + 10 commission）
- ETH 真实双边成本 ≈ **51 bps**
- 1h 级别预测目标收益一般 ≤ 0.5%，成本占比 80%+
- 4h 级别预测目标收益一般 ≥ 1.5%，成本占比 20-30%

**判决**：在现货 VIP0 费率结构下，1h 级别任何基于 OHLCV 的策略都无法打败成本。放弃 1h。如果要做高频，必须走 LOB 微观结构路线（见 SOTA 调研文档）。

**对应删除**：
- `config/settings.yaml` 中 `universe.intervals` 删除 `"1h"`
- `scripts/run_all_phases.sh` 中所有 `--interval 1h` 行
- `analysis/output/wf_*_1h.json` 归档到 `archive/`

---

### 2.3 `MACDMomentumStrategy` v2.2 的过滤器叠加 — 过度工程

**演变史**：
- v2.0：纯 MACD 交叉 + `ADX ≥ 20` → 92 trades，+834 PnL，32.3% fold WR
- v2.1：+ `trail_atr_mult`、`cooldown_bars` 等 → 略有改善但不稳
- v2.2：+ zscore 过滤 + ma_dev 过滤 + keltner 过滤 → **12 trades, +57 PnL**

**根因**：三个过滤器**都在同一个"过度拉伸"维度**上做判断，彼此高度相关。相当于要求信号通过三个几乎一样的关卡，留下的只是训练集上碰巧都过关的少数样本，典型过拟合信号。

**敏感度佐证**（`sensitivity_macd_momentum.json`）：
- `trail_atr_mult` good_ratio = 0.125（FRAGILE）
- `min_adx` good_ratio = 0.167（FRAGILE）
- `risk_per_trade` good_ratio = 0.20（FRAGILE）
- **对比** `sensitivity_triple_ema.json` 的所有参数 good_ratio = 1.0 —— 不是 triple_ema 更 robust，是它交易笔数太少以至于参数空间"看起来平"

**判决**：回滚到 v2.0 纯规则，**删除**以下过滤器：
```python
# 删除的判断（macd_momentum_strategy.py）
zscore_240 < 1.5
ma_dev_240 < 0.05
keltner_pct < 0.85
```

保留：
```python
# 保留的判断
macd_cross_above_signal
adx_14 >= 20
```

---

### 2.4 `TripleEMAStrategy` 4h v2.2 — 样本不足，无统计意义

**数据**：33 folds 共 11 笔交易，其中 5 笔盈利。

**分析**：虽然 PnL +1519 看上去亮眼，但拆解：
- ETH 2025-05 单笔 +196
- ETH 2025-07 单笔 +105
- 其余 9 笔净 PnL +1218（平均 135/笔，但分布极偏）

**判决**：**不具备上线资格**。一个 5 年 WF 只产生 11 笔交易的策略，哪怕 PnL 正也是运气。保留代码作为候选，但不单独运行，不作为资金分配目标。

---

## 3. 已证伪的因子/参数

### 3.1 完全无贡献的参数（good_ratio = 0）

`sensitivity_macd_momentum.json` 中：

| 参数 | good_ratio | n_tested | 判决 |
|---|---:|---:|---|
| `min_rsi` | 0.0 | 4 | 删除，无任何 value 达到 70% 阈值 |
| `max_rsi` | 0.0 | 4 | 删除 |
| `max_holding_bars` | 0.0 | 4 | 删除 |

**从策略签名中移除这三个参数。** 它们在训练集上看起来有效，OOS 全部失效，是典型的拟合噪声。

### 3.2 因子 edge 量化（phase1 + factor_scan 综合）

| 因子 | IC | 说明 | 去留 |
|---|---:|---|---|
| `ma_dev_240` | −0.43 | 均值回归类因子，IC 高但 top/bottom 分位胜率差仅 2.7pp | 保留但须配合成本阈值 |
| `zscore_240` | ≈ 0 | 在 v2.2 作为 MACD 过滤器表现糟糕 | **从过滤器用途中移除** |
| `keltner_pct` | 微弱 | 同上 | **从过滤器用途中移除** |
| `rsi_14` 作为入场条件 | 无 | MACD 策略实测 good_ratio=0 | 仅作为诊断指标，不入信号 |
| `adx_14` | 有效但非单调 | MACD 在 ADX>30 胜率 58%, ADX 20-30 降到 28% | **保留但仅用作 regime 判断，不用作连续门限** |
| `natr_20` | 诊断性 | 波动率过滤效果存疑 | 保留用于仓位 sizing，不用作入场过滤 |

### 3.3 相关性结构（phase1_diagnose_summary）

```
BTC-ETH 整体相关性  : 0.827
滚动 30d 最小值    : 0.261
滚动 240d 最小值    : 0.640
```

**结论**：BTC/ETH 组合**不是分散**，是同一个 beta 的两份仓位。增加 SOL/BNB 未经证明能打破这个结构（而且 SOL/BNB 和 BTC 相关性也在 0.7+）。**扩标的在没有底层 edge 之前是无意义的**。

对应删除：`config/symbols_extended_v22.yaml`

---

## 4. 代码层面清理清单

### 4.1 立即删除（Phase 1 清理）

```bash
# 死策略
rm alpha/regime_adaptive_strategy.py
rm alpha/regime_allocator_v2.py

# 未经验证的扩展配置
rm config/symbols_extended_v22.yaml

# v2.2 过度工程产物
rm alpha/enhanced_exit.py                    # 分批止盈在 11 笔样本上无意义

# 临时/冗余文件
rm main_patch.py
rm integration_patch.py
rm modified_files.json

# 归档（不删除，移到 archive/）
mkdir -p archive/v22_deprecated
mv analysis/output/wf_*_1h.json archive/v22_deprecated/
mv analysis/output/wf_regime_*.json archive/v22_deprecated/
```

### 4.2 代码简化（v2.2 → v2.0 + mean_reversion）

**`alpha/macd_momentum_strategy.py`** — 删除以下段落：

```python
# ❌ 删除（v2.2 过度拉伸过滤器）
if row.get("zscore_240", 0) > 1.5:
    return False
if row.get("ma_dev_240", 0) > 0.05:
    return False
if row.get("keltner_pct", 0) > 0.85:
    return False

# ❌ 删除无效参数
self.min_rsi = ...
self.max_rsi = ...
self.max_holding_bars = ...
```

保留：
```python
# ✅ 保留（v2.0 核心规则）
macd_cross = (row["macd"] > row["macd_signal"]) and \
             (prev["macd"] <= prev["macd_signal"])
adx_ok = row["adx_14"] >= 20
return macd_cross and adx_ok
```

**`alpha/triple_ema_strategy.py`** — 同样删除 v2.2 的过度拉伸过滤段落。

**`alpha/mean_reversion_strategy.py`** — **新增**趋势回避门（这是唯一的改造方向）：

```python
def should_enter(self, row, prev, state):
    # ... 原有入场条件 ...
    
    # 新增：趋势回避门
    if row.get("adx_14", 0) > 28:
        return False
    
    # 新增：最近 20 bar 单调性检查
    recent_closes = state.get("recent_closes", [])
    if len(recent_closes) >= 20:
        ups = sum(1 for i in range(1, 20) 
                  if recent_closes[-i] > recent_closes[-i-1])
        if ups >= 17 or ups <= 3:   # 90% 单调则禁用
            return False
    
    return True
```

### 4.3 配置清理（`config/settings.yaml`）

```yaml
# ❌ 删除
universe:
  intervals:
    - "1h"      # 删除
    - "4h"
    - "1d"

strategy:
  regime:       # 整段删除
    enabled: ...
    params: ...

# ✅ 保留并简化
universe:
  intervals:
    - "4h"
    - "1d"
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
  # SOL/BNB 暂时不加，等有横截面策略后再扩

strategy:
  macd_momentum:
    enabled: true
    min_adx: 20
    # 删除 min_rsi, max_rsi, max_holding_bars
    # 删除 v2.2 的三层过滤器参数
  
  triple_ema:
    enabled: false   # 样本不足，不单独上线
  
  mean_reversion:
    enabled: true
    trend_avoidance:    # 新增
      max_adx: 28
      max_monotonic_ratio: 0.85
```

### 4.4 成本假设修正

v2.2 的所有回测假设 `slippage = 10 bps`，但 phase1 实测：
- BTC = 20.89 bps
- ETH = 25.81 bps

**立即更新 `config/settings.yaml`**：
```yaml
execution:
  slippage_bps:
    BTCUSDT: 21    # 从 10 改为 21
    ETHUSDT: 26    # 从 10 改为 26
  commission_bps: 10  # 保持
```

这会让所有历史回测 PnL 下修 **40-60%**，但这是正确的。

---

## 5. 清理后的最小可执行集

经过本次清理，Trader 项目应收缩到以下最小集：

```
Trader/
├── alpha/
│   ├── macd_momentum_strategy.py       # v2.0 纯规则（回滚）
│   ├── triple_ema_strategy.py          # 仅作为候选，不运行
│   ├── mean_reversion_strategy.py      # v1.0 + 趋势回避
│   └── strategy_registry.py
├── data/
│   ├── storage.py
│   └── features.py
├── walkforward_v2.py
├── multi_strategy_backtest.py
├── config/settings.yaml                # 已清理
├── scripts/
│   ├── run_all_phases.sh               # 仅保留 4h 配置
│   └── summarize_results.py
└── archive/v22_deprecated/             # 死代码归档
```

**删除/简化后，代码量预计减少 35-40%，但功能等价**（因为删掉的都是被证伪的东西）。

---

## 6. 下一步决策点

本次清理**不会**让任何策略达到上线门槛。要突破 `mean_reversion` 4h 的 39.4% fold 胜率瓶颈，可选：

### 6.1 路径 A — 继续优化现有规则（保守）
在 `mean_reversion` 基础上加趋势回避 + 降低 1h 的影响 → 预计 fold 胜率提升到 45-50%，但仍是单标的时间序列策略，天花板有限。

### 6.2 路径 B — 横截面迁移（推荐）
放弃 BTC/ETH 单标的思路，转向 Top 30 coins 横截面 momentum + funding rate carry。参见 `RESEARCH_2025_SOTA_QUANT_ARCHITECTURES.md` 第 5 节。这是 2025 年机构派主流，公开报告 Sharpe 1.5-2.5。

### 6.3 路径 C — ML 因子聚合
保留现有数据管道，但把 20+ 技术指标一起喂给 LightGBM/CatBoost 预测 5 日 forward return。参见 CTREND 论文（JFQA 2025，3000+ 币种验证）。

**建议顺序**：先执行本文档的清理，再做路径 B 的 MVP（1-2 周即可验证），如果路径 B 确认有 edge，再考虑路径 C 精调。

---

**文档版本**：v1.0 (2026-04-15)  
**下次更新触发条件**：任何新策略经 walk-forward 被证伪时追加

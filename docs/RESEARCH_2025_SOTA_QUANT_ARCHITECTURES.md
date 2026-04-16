# 2025-2026 量化交易 SOTA 架构调研

> **文档目的**：基于 2025 年初到 2026 年初（截至本文撰写时点）的学术论文、工业报告和开源项目，梳理 crypto 量化领域**真正 work 的主流架构**，对比 Trader 项目现状，给出迁移路径。
>
> **核心判断**：传统的单标的 + 技术指标 + 规则策略已经接近 edge 枯竭。2025 年的机构派和学术派都在往**横截面因子模型**、**LOB 微观结构**和**ML 因子聚合**三个方向收敛。

---

## 1. 本项目当前定位 vs 2025 主流

### 1.1 现状定位

Trader v2.2 本质是：**单标的（BTC/ETH）× 时间序列技术指标 × 规则策略 × walk-forward 验证**

这在学术界属于 **2015-2020 的范式**。2020 年后的主流论文基本不再单独发表"我写了个 MACD + ADX 策略并测试了它"这类工作——这个设计空间已被挖穿。

### 1.2 Gap 表

| 维度 | Trader v2.2 现状 | 2025-26 主流做法 | Gap 等级 |
|---|---|---|---|
| 数据源 | 仅 OHLCV，2 标的 | OHLCV + LOB + funding + OI + basis + on-chain + 新闻 | 🔴 大 |
| Universe | BTC/ETH（相关性 0.83） | Top 30-100 币种横截面 | 🔴 大 |
| 信号维度 | 单标的时间序列 | 横截面排名 + 时间序列双重 | 🔴 大 |
| 特征工程 | ~10 个技术指标 | 50-200+ 因子进 ML ensemble | 🟡 中 |
| 模型 | 规则 + 可选 LightGBM 过滤 | CatBoost/XGBoost/Transformer 主信号生成 | 🟡 中 |
| 回测验证 | 单模型 walk-forward | Purged K-fold + combinatorial CV + MC variance | 🟡 中 |
| 市场中性 | 无，纯方向性 | L/S 市场中性是默认设定 | 🔴 大 |
| 成本建模 | 固定 42 bps | Tick-level 滑点 + maker/taker 分档 | 🟡 中 |
| 退出规则 | 硬规则（止盈止损） | 学习型退出 / RL policy | 🟢 小 |

---

## 2. 三条主流技术路线总览

```
┌──────────────────────────────────────────────────────────────┐
│                   2025-26 主流架构                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │  路线 A      │   │  路线 B      │   │  路线 C      │         │
│  │  横截面因子   │   │  LOB 微观   │   │  LLM+RL     │         │
│  │             │   │  结构       │   │             │         │
│  ├─────────────┤   ├─────────────┤   ├─────────────┤         │
│  │ 时间尺度     │   │ 时间尺度     │   │ 时间尺度     │         │
│  │ 日/周 rebal │   │ 毫秒-秒      │   │ 日/小时      │         │
│  │             │   │             │   │             │         │
│  │ 典型 Sharpe  │   │ 典型 Sharpe  │   │ 典型 Sharpe  │         │
│  │ 1.5 - 2.5   │   │ 2.0 - 4.0   │   │ 0.3 - 1.0   │         │
│  │ (多 30 币种)│   │ (做市/套利) │   │ (尚不成熟)  │         │
│  │             │   │             │   │             │         │
│  │ 入门难度     │   │ 入门难度     │   │ 入门难度     │         │
│  │ ⭐⭐        │   │ ⭐⭐⭐⭐⭐  │   │ ⭐⭐⭐⭐     │         │
│  │             │   │             │   │             │         │
│  │ ← 推荐路线   │   │             │   │             │         │
│  └─────────────┘   └─────────────┘   └─────────────┘         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 路线 A：横截面因子模型（Cross-sectional Factor Models）

**这是机构派当前首选，也是本项目最应该迁移的方向。**

### 3.1 核心思路

不预测单个币种涨跌，而是在每个 rebalance 时点对 universe 内所有币种**按某个因子排序**，做多 top quintile、做空 bottom quintile，构建市场中性组合。

优势：
- **消除 beta 风险**：BTC 大跌大涨不直接影响策略 PnL
- **数据需求低**：日频 OHLCV 就够
- **学术支撑强**：20+ 篇近年论文验证在 crypto 上有效
- **公开可复现**：Unravel Finance、CF Benchmarks 等都发过可复现的 notebook

### 3.2 有效因子清单（2025 文献综合）

| 因子 | 构造方式 | 文献证据 |
|---|---|---|
| **Size (市值)** | log(market_cap) 倒序，小市值反而更高收益 | Liu-Tsyvinski-Wu 2022 (JoF), Isac et al 2026 |
| **30d Momentum** | 过去 30 日累计收益率 | Zaremba et al 2021, Unravel 2025 |
| **Short-term Reversal** | 过去 7 日收益率倒序 | Shen et al 2023 |
| **Volatility** | 过去 30 日日收益率标准差（低波优于高波） | Liu-Tsyvinski-Wu 2022 |
| **Liquidity (Amihud)** | \|return\| / volume 的移动均值 | Wei 2018, 多篇后续 |
| **Funding Rate Carry** | 负费率做多，正费率做空 | BitMEX 2025 Q3 Report |
| **Downside Beta** | 对 BTC 下行期的 beta（高者反而收益差） | CF Benchmarks 2024 |
| **On-chain Active Addr** | 7d 活跃地址 z-score | Mann 2025 SSRN review |
| **MAX** | 过去 30 日最大单日收益（彩票效应） | Li et al 2021 |

**关键论文**：**CTREND** (JFQA Nov 2025) 用 ML 把上面这些因子打包成一个综合 trend factor，在 3,000+ 币种上做 cross-section，确认对已知因子有增量，且**成本之后仍然 work**。

### 3.3 典型实现（Unravel Finance 2025 blog 公开框架）

```python
def cross_sectional_momentum():
    # 每日操作
    universe = top_n_by_marketcap(n=50, min_volume_30d=5_000_000)
    
    # 因子计算
    momentum_30d = {c: price[c][-1]/price[c][-30] - 1 for c in universe}
    
    # 排序
    sorted_coins = sorted(universe, key=lambda c: momentum_30d[c])
    n = len(sorted_coins)
    top_20pct = sorted_coins[int(n*0.8):]
    bot_20pct = sorted_coins[:int(n*0.2)]
    
    # 构建市场中性组合 (200% gross exposure)
    positions = {}
    for c in top_20pct:
        positions[c] = +1.0 / len(top_20pct)
    for c in bot_20pct:
        positions[c] = -1.0 / len(bot_20pct)
    
    # Inverse volatility weighting（关键！crypto 内部波动差异极大）
    positions = inverse_vol_weight(positions, lookback=30)
    
    return positions
```

### 3.4 对 crypto 特有的注意点

1. **Momentum 形成窗口更短**：equities 用 6-12 月，crypto 只用 ~30 天，且 decay 很快
2. **Momentum crashes 更剧烈**：经常被 short-squeeze 短期反转
3. **大市值 dominance**：主要 alpha 来自 top-40，小币 liquidity 风险高
4. **Survivorship bias 严重**：必须用时点 universe（point-in-time），不能用"今天的 Top 50"回测历史

### 3.5 预期效果

公开报告（Unravel 2025、Liu-Tsyvinski 2022、CTREND 2025）的 Sharpe 区间：
- **纯 momentum** L/S：1.3 - 1.8
- **Momentum + Size + Volatility 多因子** L/S：1.8 - 2.5
- **+ 横截面 ML**：2.0 - 3.0（CTREND 报告）

**这些是扣除 tick-level 成本后的数字**，显著优于 Trader v2.2 目前的任何策略。

### 3.6 MVP 实现路径（1-2 周）

```
Week 1:
  Day 1-2: 拉取 Binance 现货 Top 50 coins 5 年日线数据
  Day 3-4: 实现 5 个因子（30d mom、7d reversal、vol、amihud liq、mcap）
  Day 5-7: 单因子 quintile 测试，建立 baseline

Week 2:
  Day 8-10: 多因子 equal-weighted 组合
  Day 11-12: Inverse volatility weighting
  Day 13-14: Purged walk-forward + Monte Carlo 稳健性
```

---

## 4. 路线 B：LOB 微观结构深度学习

**适合：做市、高频套利。不适合：bar-level 策略。**

### 4.1 核心进展

**TLOB** (arxiv 2502.15757, WWW 2025)：Dual attention transformer 在 LOB 数据上做方向预测，跨 5 个主要交易所稳定跑赢 DeepLOB baseline。

**HLOB** (LSE 2025)：用 information persistence 度量 LOB 信息衰减速度，给出不同品种的最优预测 horizon。

**Briola-Aste** (arxiv 2602.00776, Jan 2026) "Explainable Patterns in Cryptocurrency Microstructure"：
- 数据：Binance Futures 5 币种（BTC/LTC/ETC/ENJ/ROSE）LOB 1-second 数据，2022-01 到 2025-10
- 关键发现：**同一套 LOB 特征在市值跨 100 倍的币种上 SHAP importance 高度稳定**
- 意味着：一个训好的模型可以跨币种部署，大大降低开发成本

**Wang** (arxiv 2506.05764, Jun 2025) "Better Inputs Matter More Than Stacking Another Hidden Layer"：
- 对比 DeepLOB (3 CNN + LSTM) vs 单层 CNN+LSTM vs XGBoost
- 结论：**XGBoost + 好的 LOB 特征超过深度网络**，且训练/推理快 10 倍
- 输入（T=10 窗口、prediction horizon、LOB depth）的影响远大于模型复杂度

### 4.2 典型特征（都已被多篇论文验证有效）

- **Order Flow Imbalance (OFI)**：bid vol − ask vol 的加权版
- **Micro-price**：(ask × bid_size + bid × ask_size) / (bid_size + ask_size)
- **Spread**：ask_1 - bid_1，以 tick 为单位
- **Depth imbalance across levels**：前 5 档 vol 的分布不对称度
- **Trade imbalance**：近 N 秒成交的 buy/sell 比例
- **Queue dynamics**：订单到达率 lambda

### 4.3 为什么本项目不建议走这条路

1. **数据成本高**：LOB tick 数据一天 GB 级，5 年 TB 级
2. **基础设施要求**：colocation、low-latency 连接、FPGA/C++ 实现
3. **ROI 错配**：bar-level 策略投入 LOB 数据 = overkill，日频策略用不上 tick 信号
4. **Flash crash 风险**：Briola 论文 Fig 16 展示 2025-10-10 flash crash 中做市策略连续被 adversely selected

**结论**：除非你打算做 market making 或 cross-exchange arbitrage，否则 LOB 路线暂时跳过。

---

## 5. 路线 C：LLM 增强强化学习（LLM-augmented RL）

**最热门但最不成熟。暂时不建议作为主线。**

### 5.1 代表工作

- **FinRL-DeepSeek** (Benhenda 2025, arxiv 2502.07393)：DeepSeek 从新闻提取风险信号，注入 PPO agent
- **FLAG-Trader** (Xiong et al 2025, arxiv 2502.11433)：LLM agent + gradient-based RL
- **HedgeAgents** (WWW 2025)：多智能体金融交易，各自扮演不同角色
- **FinRL Contests 2023-2025** 综述（arxiv 2504.02281）

### 5.2 实测表现（问题所在）

FinRL Contest 2025 crypto 任务公开的**最佳 ensemble 结果**：
- Sharpe = **0.28**
- Max drawdown = −0.73%
- Win/loss ratio = 1.62

**只比 BTC buy-and-hold 略好**，和路线 A 的 1.5-2.5 Sharpe 差一个数量级。

### 5.3 为什么还不行

1. **Sample efficiency 灾难**：RL 需要海量 episode，crypto 历史数据 10 年不够
2. **Non-stationarity**：市场状态分布在变，训练好的 policy 快速过时
3. **Reward hacking**：agent 学会在验证集上刷分的投机行为
4. **LLM 信号 noise 大**：新闻 sentiment 到价格的传导有滞后且间歇性

### 5.4 值得关注的子方向

- **离线 RL（offline RL）** + conservative Q-learning：减少 online exploration 的成本
- **LLM 作为 feature extractor** 而非直接决策者：只用 LLM 提取新闻 embedding，下游用传统 ML

但这些目前都在"论文演示"阶段，不具备生产价值。

---

## 6. 时间序列基础模型（横切工具）

这些不是独立策略路线，但可以作为路线 A、C 中的**预测模块**。

### 6.1 SOTA 模型（按 2025-26 时间顺序）

| 模型 | 发表 | 核心创新 | 在金融数据上的表现 |
|---|---|---|---|
| **PatchTST** | ICLR 2023 | 时间序列分 patch + channel independence | 至今仍是强 baseline |
| **TimesNet** | ICLR 2023 | 2D 变换捕捉多周期 | 通用 |
| **DLinear** | AAAI 2023 | 极简线性模型，挑战 Transformer | 竟然 ≥ 多数 Transformer |
| **TimeMoE / Chronos** | 2024 | 零样本预测，大模型范式 | 跨领域迁移但金融上未突破 |
| **TimeKAN** | ICLR 2025 | KAN 做频域分解 | 新 |
| **xPatch** | AAAI 2025 | 双流分解 | 新 |
| **MambaSL** | ICLR 2026 | Mamba 单层时序分类 | 最新 |

### 6.2 清华 Time-Series-Library (Oct 2025) 的"Accuracy Law"

项目维护者的关键观点：**在标准 benchmark 上，当前 Transformer 变体之间的差别微小**，所谓 SOTA 更多是刷点。**真正的进步来自输入质量**（patching + channel independence），而不是架构堆砌。

这和 Briola 的 LOB 结论完全一致：**better inputs > deeper networks**。

### 6.3 在 crypto 上的应用建议

不要直接拿 PatchTST 预测 BTC 价格——大概率不会比 LightGBM + 手工特征好。
**正确用法**：用 PatchTST/TimesNet 做**特征提取器**，把学到的 representation 作为输入之一喂给下游 ML 模型。

---

## 7. 成本建模与回测方法学（横切）

### 7.1 Purged Walk-Forward (López de Prado)

你当前的 WF 实现有一个隐患：训练集和测试集**在时间上相邻**，如果一笔交易在训练集结束时开仓、在测试集开始时平仓，会发生 label leakage。

**修复方法**：在训练集和测试集之间加 `embargo` 窗口（通常 1-5 个 bar）。参考 *Advances in Financial Machine Learning* 第 7 章。

### 7.2 Combinatorial Purged CV

López de Prado 推荐的 CPCV（Combinatorial Purged Cross-Validation）：生成所有可能的 train/test 组合，报告 Sharpe 的分布而不是单点。这是 walk-forward 的严格升级版。

### 7.3 成本模型

```python
# v2.2 现在：过于简化
cost_bps = 42  # 固定

# 主流做法：分档成本
def real_cost(order_size, book_depth, fee_tier, is_maker):
    fee = 0 if is_maker else fee_tier  # maker 可能是负费率
    spread_cost = 0.5 * spread_bps
    impact = sqrt_impact(order_size, book_depth)  # 平方根冲击
    return fee + spread_cost + impact
```

**实测差异**：日频 L/S 策略用真实成本模型回测，结果往往比固定 bps 好 15-30%（因为 maker/limit 占大头）。

### 7.4 Deflated Sharpe Ratio

López de Prado 2018：考虑"试了多少次才找到这个策略"后的 Sharpe。一个 2.0 Sharpe 经过 100 次试验后的 DSR 可能只有 0.8。**任何多次迭代的策略报告时都应该报 DSR，不是 naive Sharpe**。

---

## 8. 对 Trader 项目的迁移建议

### 8.1 按 ROI 排序的行动项

**短期（1-2 周，ROI 最高）**

1. **清理 v2.2 死代码**（见 `DEPRECATED_FACTORS_AND_STRATEGIES.md`）
2. **搭横截面 momentum MVP**：
   - 拉 Top 50 coins 日线
   - 5 因子（mom、reversal、vol、liquidity、mcap）
   - L/S equal-weighted + inverse vol
   - Purged WF 验证
   - 目标：Sharpe > 1.2 就算证明方向正确

**中期（1-2 月）**

3. **接入资金费率数据**：Binance 永续 funding rate 是 crypto 特有的 carry 因子
4. **CTREND 风格的 ML 因子聚合**：把 20-30 个技术/on-chain 因子打包进 LightGBM，预测 5 日 forward return
5. **Purged walk-forward + CPCV** 升级

**长期（3+ 月）**

6. **on-chain 数据管道**：CryptoQuant API 或 Glassnode，加入活跃地址、持有者结构等
7. **如果横截面 momentum 证实有 edge**：扩展到 L/S + carry + value 多因子组合
8. **如果想做主动择时**：转向 LOB 微观结构（但需要基础设施投入）

### 8.2 什么是**不应该做**的

基于本次调研，以下方向投入低 ROI：

- ❌ 继续在单标的（BTC、ETH）上调参技术指标
- ❌ 尝试更多过滤器组合（已证过度工程）
- ❌ 直接上 LLM + RL（学术上都还不 work）
- ❌ 训练大 Transformer 预测价格（PatchTST 直接拿来用不会比 LightGBM 好）
- ❌ 扩展到 SOL/BNB 增加样本（不解决相关性 0.83 的问题）

### 8.3 决策树

```
Q: 你手上的预算和目标是什么？

├─ "我只想最快验证有没有 edge 的方向"
│   └─ 路线 A (横截面 momentum MVP) + CTREND 因子聚合
│      预算：2-4 周，Python + Binance API 即可
│
├─ "我想做一个能长期跑的产品化策略"
│   └─ 路线 A 做底，加 funding carry + on-chain 因子 + 
│      Purged CPCV 验证 + 严格成本建模
│      预算：3-6 月
│
├─ "我有 HPC + 低延迟基础设施"
│   └─ 路线 B (LOB 做市 / cross-exchange arb)
│      预算：6-12 月，需要团队
│
└─ "我要做研究性项目，追学术前沿"
    └─ 路线 C (LLM + RL) 但准备好长期不赚钱
       预算：12+ 月
```

---

## 9. 参考文献清单

### 9.1 横截面因子（路线 A）

- Liu, Tsyvinski, Wu. *Common Risk Factors in Cryptocurrency*. Journal of Finance, 2022.
- Han et al. *A Trend Factor for the Cross Section of Cryptocurrency Returns (CTREND)*. JFQA 60(7), Nov 2025.
- Isac, Erdösi, Han. *A Factor Model for Digital Assets*. MARBLE 2025 / Springer 2026.
- Zaremba et al. *Up or down? Short-term reversal, momentum, and liquidity effects in cryptocurrency markets*. IRFA 2021.
- Mann, W. *Quantitative Alpha in Crypto Markets: A Systematic Review*. SSRN 5225612, April 2025.
- Cakici et al. *Machine learning and the cross-section of cryptocurrency returns*. IRFA 94, 2024.

### 9.2 微观结构（路线 B）

- Berti, Kasneci. *TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data*. arxiv 2502.15757, WWW 2025.
- Briola, Bartolucci, Aste. *HLOB – Information persistence and structure in limit order books*. LSE Research Online 2025.
- Briola et al. *Explainable Patterns in Cryptocurrency Microstructure*. arxiv 2602.00776, Jan 2026.
- Wang, H. *Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books: Better Inputs Matter More Than Stacking Another Hidden Layer*. arxiv 2506.05764, Jun 2025.
- Zhang, Zohren, Roberts. *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*. 2020.

### 9.3 LLM + RL（路线 C）

- Wang et al. *FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agents*. arxiv 2504.02281, 2025.
- Benhenda. *FinRL-DeepSeek: LLM-Infused Risk-Sensitive Reinforcement Learning for Trading Agents*. arxiv 2502.07393, 2025.
- Xiong et al. *FLAG-Trader: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading*. arxiv 2502.11433, ACL 2025.

### 9.4 时间序列基础模型

- Nie et al. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)*. ICLR 2023.
- Wu et al. *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*. ICLR 2023.
- Moreno-Pino, Zohren. *DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions*. Quantitative Finance 24(8), 2024.

### 9.5 方法学

- López de Prado. *Advances in Financial Machine Learning*. Wiley 2018（第 7 章 Purged CV，第 11-14 章回测风险）
- López de Prado. *The Deflated Sharpe Ratio*. Journal of Portfolio Management, 2018.
- Wang, Zohren (ed). *Deep Learning in Quantitative Trading*. Cambridge Elements 2025（2025 年最新综合教材）

### 9.6 工业报告

- BitMEX Research. *2025 Q3 Derivatives Report: Funding Rates Structure*. 2025-10.
- CoinGlass. *2025 Cryptocurrency Derivatives Market Report*. 2025-12.
- CF Benchmarks. *Institutional-grade Factor Model for Digital Assets*. 2024.
- Unravel Finance. *Cross-Sectional Alpha Factors in Crypto: 2+ Sharpe Ratio Without Overfitting*. 2025-08.

### 9.7 开源代码

- **FinRL**: https://github.com/AI4Finance-Foundation/FinRL
- **FinRL-Meta**: https://github.com/AI4Finance-Foundation/FinRL-Meta
- **Time-Series-Library** (清华): https://github.com/thuml/Time-Series-Library
- **PatchTST**: https://github.com/yuqinie98/PatchTST
- **BasicTS** (时序 benchmark): https://github.com/GestaltCogTeam/BasicTS
- **TFB** (PVLDB 2024 Best Paper Nomination): https://github.com/decisionintelligence/TFB

---

**文档版本**：v1.0 (2026-04-15)  
**核心结论 TL;DR**：
1. 你现在做的单标的 TS + 技术指标方向已接近 edge 枯竭
2. 2025 机构派 / 学术派主流是横截面因子模型（L/S 市场中性）
3. 建议 2-4 周做一个横截面 momentum MVP 验证方向
4. 如果 MVP 验证有 edge，再投入到多因子 ML 聚合（CTREND 风格）
5. LOB 微观结构路线需要基础设施投入，LLM+RL 还不成熟

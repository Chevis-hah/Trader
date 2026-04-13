# Phase 2 — 量化策略重构路线图

> Generated: 2026-04-12
> 状态: 🔴 尚未达到上线标准，需要完成以下所有步骤

---

## 当前步骤：运行诊断

### ✅ 你需要做的

1. **将 `phase2_diagnostic.py` 放到项目根目录下**（和 `data/quant.db` 同级）
2. **运行诊断脚本**：
   ```bash
   python phase2_diagnostic.py
   ```
3. **返回以下文件给我**：
   - `phase2_report.json` — 诊断脚本的输出（最重要）
   - 你的策略进场逻辑核心代码（`strategy.py` 或类似文件中的 `generate_signal` / `check_entry` 方法）
   - 如果方便，也把你的回测引擎主循环代码发给我（`backtest.py` 中的 `run()` 方法）

### 📋 我需要从 `phase2_report.json` 中获取的关键信息

| 信息 | 用途 |
|------|------|
| DB schema（表结构） | 理解你的数据存储方式，后续代码要能直接跑 |
| 4 个标的的交叉相关性 | 决定多标的组合和仓位分配 |
| 真实交易成本估计 | 修正回测假设 |
| 各 regime 分布 | 设计 regime gate 的阈值 |
| 原始信号 edge 统计 | **最关键** — 判断 MACD 信号本身有没有 edge |
| Long+Short 回测结果 | 验证做空是否有正向贡献 |
| BTC 4h walk-forward | 在最长历史上验证策略稳定性 |
| 500 USDT 可行性 | 确认小资金的执行约束 |

---

## 总体路线图

### Phase 2A — 信号 edge 验证 ← 当前位置
- [x] 数据可用性审计
- [ ] 运行 `phase2_diagnostic.py`
- [ ] 确认信号是否有统计显著的 edge（t-stat > 2）
- [ ] 确认做空信号的贡献
- [ ] 确认 500 USDT 可行性

### Phase 2B — 策略重构
> *仅在 2A 确认有 edge 后才进行*
- [ ] 基于诊断结果重写策略入场/出场逻辑
- [ ] Regime gate 硬编码（ADX + 波动率阈值）
- [ ] 简化出场（去掉不赚钱的出场方式）
- [ ] 多标的回测（BTC/ETH/BNB/SOL）
- [ ] 相关性感知仓位管理

### Phase 2C — 稳健性验证
- [ ] 全量历史 walk-forward（BTC 4h, 5.5 年）
- [ ] 参数敏感性（仅测 2-3 个核心参数）
- [ ] OOS 验证：在 ETH/BNB/SOL 上不调参直接跑
- [ ] Monte Carlo 仿真（打乱交易顺序，看权益曲线分布）

### Phase 2D — 上线准备
- [ ] Paper trading 模块开发
- [ ] 60 天模拟盘运行
- [ ] 风控模块（日亏损限制、连亏暂停）
- [ ] 异常处理（API 断连、数据延迟）
- [ ] 500 USDT 小资金实盘启动

### Phase 2E — 扩展与优化
- [ ] 观察 30 天实盘后评估是否扩展到 5000 USDT
- [ ] 根据实盘数据校准滑点模型
- [ ] 考虑增加 1h 时间框架
- [ ] 考虑增加更多低相关标的

---

## 关键决策点

### 🚦 Gate 1: 信号 edge 是否存在？
如果 `phase2_report.json` 中 MACD 信号的 t-stat < 1.5 且所有方向、所有标的都没有显著 edge：
→ **停止当前策略方向**，需要重新设计信号逻辑

如果 t-stat > 2 在部分条件下成立：
→ **继续但缩窄范围**，只在有 edge 的 regime 和方向上交易

### 🚦 Gate 2: 500 USDT 够不够？
如果需要 >5x 杠杆才能达到最小仓位：
→ 建议先在 paper trading 中验证，等资金到 5000 再实盘
→ 或切换到允许更小仓位的交易所

### 🚦 Gate 3: 做空能贡献正收益吗？
如果做空信号在回测中持续亏损：
→ 去掉做空，但需要接受交易频率低的现实
→ 或切换到 1h 时间框架增加交易频率

---

## 上线盈利的最低标准

| 指标 | 阈值 | 当前 |
|------|------|------|
| 总交易笔数（回测） | ≥150 | ~24（严重不足） |
| Walk-forward OOS Sharpe | ≥0.6 | 0.395（不达标） |
| WF fold win rate | ≥40% | 32.3%（不达标） |
| Profit factor | ≥1.3 | 1.18（不达标） |
| 参数 FRAGILE 比例 | <30% | ~86%（严重不达标） |
| 真实成本后仍盈利 | ✅ | ❓（待验证） |
| Paper trading 30 天 | 正收益 | 未做 |

---

## 你现在的 action item

```bash
# 1. 复制文件到项目目录
cp phase2_diagnostic.py /path/to/your/project/

# 2. 运行
cd /path/to/your/project/
python phase2_diagnostic.py

# 3. 返回结果
# - phase2_report.json
# - 你的 strategy.py 核心代码（进场逻辑部分）
# - 你的 backtest.py 核心代码（如果方便）
```

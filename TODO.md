# 🎯 执行 TODO — 按顺序操作

## 前置: 文件部署

```bash
# 1. 将输出文件复制到 Trader 仓库
cp -r analysis/ ~/Trader/analysis/
cp -r alpha/bidirectional_wrapper.py ~/Trader/alpha/
cp -r alpha/regime_allocator.py ~/Trader/alpha/
cp -r alpha/enhanced_exit.py ~/Trader/alpha/
cp -r alpha/ml_signal_filter.py ~/Trader/alpha/
cp backtest_walkforward.py ~/Trader/
cp integration_patch.py ~/Trader/

# 2. 确保 analysis/ 目录下有 __init__.py
touch ~/Trader/analysis/__init__.py
```

---

## Phase 1: 补充数据 (30分钟)

> 目标: 获取做决策需要的基础数据

### 1.1 同步更多历史数据

如果数据库中数据不足 2 年，先补数据:

```bash
cd ~/Trader
# 确认数据量
python main.py --validate

# 如果需要补充 (会自动断点续传)
python main.py --sync-data
```

### 1.2 运行数据诊断

```bash
python integration_patch.py --mode diagnose --start 2025-01-01
```

**收集输出:**
- [ ] BTC/ETH 相关性数值
- [ ] 各 Regime 分桶下的 PnL 分布
- [ ] 滑点估算 vs 当前假设的差距
- [ ] 权益曲线 CSV (analysis/output/ 下)

---

## Phase 2: P0 — 验证框架 (1小时)

> 目标: 确认当前策略不是过拟合

### 2.1 Walk-Forward 验证

```bash
# 两个策略都要跑
python integration_patch.py --mode walkforward \
  --train-days 180 --test-days 60

# 如果数据足够长，也可以尝试不同窗口
python backtest_walkforward.py --strategy triple_ema \
  --train-days 120 --test-days 45
python backtest_walkforward.py --strategy macd_momentum \
  --train-days 120 --test-days 45
```

**关键判断标准:**
- [ ] 窗口胜率 >= 60%? (至少超过半数窗口盈利)
- [ ] OOS Sharpe > 0.5?
- [ ] 最差窗口亏损 < 总资金的 5%?

### 2.2 参数敏感性扫描

```bash
python integration_patch.py --mode sensitivity \
  --strategy triple_ema --start 2025-01-01

python analysis/param_sensitivity.py \
  --strategy macd_momentum --start 2025-01-01 --mode 1d
```

**关键判断标准:**
- [ ] 哪些参数被标记为 FRAGILE?
- [ ] 最优点周围的参数组合表现如何?

---

## Phase 3: P3 — 增强出场 (30分钟)

> 目标: 解决追踪止损过紧 + 缺少分批止盈的问题

```bash
python integration_patch.py --mode enhanced --start 2025-01-01
```

**对比原版 vs 增强版:**
- [ ] PnL 变化
- [ ] Sharpe 变化
- [ ] 最大回撤变化
- [ ] 交易数量变化 (如果大幅减少说明分批止盈在工作)

---

## Phase 4: P4 — ML 过滤 (1小时)

> 目标: 用 ML 模型过滤低质量信号

### 4.1 训练 ML 模型

```bash
# 用 2024 年数据训练 (如果有)
python integration_patch.py --mode ml_train \
  --ml-train-start 2024-01-01 --ml-train-end 2024-12-31

# 如果没有 2024 数据，用 2025 前半年
python integration_patch.py --mode ml_train \
  --ml-train-start 2025-01-01 --ml-train-end 2025-06-30
```

**关键判断:**
- [ ] AUC 是否 >= 0.55? (低于 0.52 不建议启用)
- [ ] Top 10 特征是否合理?

### 4.2 ML 过滤回测

如果 AUC >= 0.55:

```python
# 在 Python 中手动跑 (或添加到 integration_patch)
from backtest_runner import BacktestEngine
from alpha.ml_signal_filter import MLSignalFilter, patch_strategy_with_ml

engine = BacktestEngine(db_path="data/quant.db", strategy_name="macd_momentum",
                        start_date="2025-07-01")  # 用 ML 没见过的数据
ml = MLSignalFilter.load("models/ml_filter.pkl")
patch_strategy_with_ml(engine.strategy, ml, threshold=0.55)
report = engine.run()
```

---

## Phase 5: P1 + P2 (后续迭代)

> 这两个需要修改 backtest_runner.py 的核心循环，建议在 Phase 2-4 验证完毕后再做

### 5.1 P1: 做空 — 需要 Binance Futures API 支持

- [ ] 先在 `backtest_runner.py` 中添加 SHORT 仓位逻辑
- [ ] 在 `BidirectionalWrapper` 中包装现有策略
- [ ] 回测对比 Long-only vs Long+Short

### 5.2 P2: Regime 动态分配

- [ ] 在 `backtest_runner.py` 中集成 `RegimeAllocator`
- [ ] 按 regime 动态调整两个策略的资金比例
- [ ] 回测对比 固定分配 vs 动态分配

### 5.3 P5: 新增标的

```bash
# 1. 更新 config/settings.yaml (参考 config/symbols_extended.yaml)
# 2. 同步新标的数据
python main.py --sync-data
# 3. 回测
python main.py --backtest --strategy triple_ema
```

---

## ⚡ 速查: 命令汇总

| 步骤 | 命令 | 耗时 |
|------|------|------|
| 诊断 | `python integration_patch.py --mode diagnose` | 2分钟 |
| WF验证 | `python integration_patch.py --mode walkforward` | 5-10分钟 |
| 敏感性 | `python integration_patch.py --mode sensitivity` | 10-30分钟 |
| 增强出场 | `python integration_patch.py --mode enhanced` | 3分钟 |
| ML训练 | `python integration_patch.py --mode ml_train` | 3-5分钟 |
| 全量 | `python integration_patch.py --mode full` | 30-60分钟 |

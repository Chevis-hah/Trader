# Trader v2.2 → v2.3 迁移包 — 使用说明

## 本包内容

```
Trader_v23_cleanup/
├── docs/
│   ├── DEPRECATED_FACTORS_AND_STRATEGIES.md    # 失效策略沉淀
│   ├── RESEARCH_2025_SOTA_QUANT_ARCHITECTURES.md  # 2025 SOTA 调研
│   └── README_v23_MIGRATION.md                  # 本文件
│
├── scripts/
│   ├── cleanup_v23.sh                          # 死代码归档脚本
│   ├── run_all_phases.sh                       # 清理过的 WF 流水线（去掉 1h）
│   └── run_mvp_path_a.sh                       # 路线 A 横截面 MVP 启动脚本
│
├── alpha/
│   ├── macd_momentum_strategy.py               # 回滚到 v2.0
│   ├── triple_ema_strategy.py                  # 回滚到 v2.0
│   ├── mean_reversion_strategy.py              # 加趋势回避门
│   └── cross_sectional_momentum.py             # 路线 A 新策略
│
├── config/
│   └── settings.yaml                           # 清理 + 修正滑点
│
├── data/
│   └── universe.py                             # 路线 A 需要的 universe 构建
│
├── cross_sectional_backtest.py                 # 路线 A MVP 回测
│
└── tests/
    └── test_v23.py                             # 基础冒烟测试
```

## 一键覆盖使用方法

```bash
cd /path/to/Trader

# 1. 先归档原项目（以防万一）
tar -czf ../Trader_backup_$(date +%Y%m%d).tar.gz .

# 2. 解压本包覆盖
unzip -o /path/to/Trader_v23_cleanup.zip

# 3. 执行清理脚本（归档死代码到 archive/）
bash scripts/cleanup_v23.sh

# 4. 运行冒烟测试
python -m pytest tests/test_v23.py -v
# 或: python tests/test_v23.py

# 5. 重新跑 walk-forward 验证（只有 4h）
bash scripts/run_all_phases.sh

# 6. 如果 mean_reversion 4h 的 fold 胜率达到 45%+，可以启动路线 A MVP
bash scripts/run_mvp_path_a.sh
```

## 关键变更摘要

### 已删除 / 归档（由 cleanup_v23.sh 处理）
- `alpha/regime_adaptive_strategy.py` — 0 trades 死策略
- `alpha/regime_allocator_v2.py` — 依赖上者
- `alpha/enhanced_exit.py` — 11 笔样本上无意义
- `config/symbols_extended_v22.yaml` — 扩标的未经验证
- `wf_*_1h.json` — 1h 全部证伪
- `wf_regime_*.json` — regime 全 0

### 已修改（本包直接覆盖）
- `alpha/macd_momentum_strategy.py` — 删除三层过滤器 + 无效参数
- `alpha/triple_ema_strategy.py` — 删除过度拉伸过滤
- `alpha/mean_reversion_strategy.py` — 新增趋势回避门
- `config/settings.yaml` — 删 1h、删 regime、修正滑点（BTC 21bps、ETH 26bps）
- `scripts/run_all_phases.sh` — 只保留 4h 配置

### 已新增（路线 A 脚手架）
- `alpha/cross_sectional_momentum.py` — 横截面 momentum 策略
- `data/universe.py` — Top N 币种滚动 universe 构建
- `cross_sectional_backtest.py` — MVP 回测入口
- `scripts/run_mvp_path_a.sh` — 一键跑 MVP

## 预期结果

### Step 5 跑完 WF 后（仍是规则策略）：
- `mean_reversion` 4h fold 胜率从 39.4% → 45-50%（加趋势回避后）
- `macd_momentum` 4h 回滚后应该恢复到 v2.0 的 ~32% fold 胜率
- `triple_ema` 4h 保留但不作为主策略

### Step 6 跑完路线 A MVP 后：
- 预期 Sharpe：1.2-1.8（业界公开报告区间 1.5-2.5，保守估计）
- 若 Sharpe > 1.2：确认方向正确，下一步做 CTREND 风格的 ML 聚合
- 若 Sharpe < 1.0：说明 crypto momentum 因子衰减严重，考虑加入 funding carry / on-chain 因子

## 不兼容提醒

本包**不包含**以下原项目文件，保持原样：
- `data/storage.py`
- `data/features.py`
- `walkforward_v2.py`
- `multi_strategy_backtest.py`
- `factor_signal_scan.py`
- `main.py`
- `core/*`
- `utils/*`

如果上述文件引用了已删除的 `regime_adaptive_strategy` 或 `regime_allocator_v2`，解压后可能报 ImportError。需要手动删除这些 import。可以用：

```bash
grep -r "regime_adaptive_strategy\|regime_allocator_v2\|enhanced_exit" . \
    --include="*.py" --include="*.yaml"
```

找出所有残留引用并手动清理。

## 问题排查

### 回测 PnL 变差了？
正常的。v2.2 的假设滑点是 10 bps，实测应该是 21-26 bps（phase1 诊断数据）。本次修正后，历史回测 PnL 下修 40-60% 是**正确的**结果，之前的数字虚高。

### `mean_reversion` 4h 交易笔数减少了？
预期中的。新增的趋势回避门（ADX > 28 或最近 20 bar 单调性 > 85%）会过滤掉在强趋势中逆势入场的情况。从 59 笔减少到 40-50 笔属于正常，目的是提升 fold 胜率，不是交易频次。

### 路线 A MVP 需要哪些额外数据？
Top 50 coins 的 5 年日线 OHLCV。可以用 Binance 公开 API 拉取，不需要 VIP 或付费数据。

## 下一步规划建议

根据 `DEPRECATED_FACTORS_AND_STRATEGIES.md` 第 6 节的决策树：

1. **清理后立刻跑 WF**：确认 `mean_reversion` 4h 是否有希望
2. **启动路线 A MVP**：这是验证"crypto 是否还有可挖 edge"最快的路径（1-2 周）
3. **根据 MVP 结果决策**：
   - Sharpe > 1.2 → 继续加因子（funding carry、on-chain），做 CTREND 风格聚合
   - Sharpe < 1.0 → 考虑转换标的类别（大宗商品期货 / 外汇）或转向做市

详细参考：`docs/RESEARCH_2025_SOTA_QUANT_ARCHITECTURES.md` 第 8 节。

---

**包版本**：Trader_v23_cleanup.zip  
**生成日期**：2026-04-15  
**对应文档**：DEPRECATED_FACTORS_AND_STRATEGIES.md v1.0 + RESEARCH_2025_SOTA_QUANT_ARCHITECTURES.md v1.0

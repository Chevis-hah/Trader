# CHANGELOG — v2.2 (因子驱动重构)

> 日期: 2026-04-14
> 基于 factor_scan_report.json + factor_scan_1h.json 的数据驱动决策

---

## 核心发现 → 架构变更

因子扫描揭示: **BTC/ETH 在 4h/1h 级别的主导模式是均值回归, 不是动量**
- ma_dev_240: IC=-0.43, ICIR=-1.99 (极强反转信号)
- 当前趋势策略在价格过度拉伸时入场 → 这是亏损的根本原因

---

## Phase 1: 均值回归策略 + 过度拉伸过滤

### 新增 `alpha/mean_reversion_strategy.py`
- 入场: zscore_240 < -1.5 + RSI < 35 + ADX < 25 (至少 2/4 信号)
- 出场: zscore > 0.3 (回归均值) 或 RSI > 55
- amihud_60 流动性加权仓位
- 与趋势策略互补: 震荡市赚均值回归, 趋势市赚动量

### 修改 `alpha/macd_momentum_strategy.py` → v2.2
- **新增 overextension_filter**: zscore_240 > 1.5 / ma_dev_120 > 5% / keltner > 85% 时不追高
- 保留 regime_gate + 自适应追踪

### 修改 `alpha/triple_ema_strategy.py` → v2.2
- 同上 overextension_filter

### 修改 `alpha/strategy_registry.py`
- 注册 mean_reversion 策略

## Phase 2: ML 组合模型

### 新增 `alpha/ml_lightgbm.py`
- LightGBM 替代 sklearn GBM (速度 10x+)
- 自动降级到 sklearn (如未安装 lightgbm)
- 核心 12 因子 + 扩展 16 因子
- Purged + Embargo CV
- 支持 predict_proba 接入信号过滤

## Phase 3: Walk-forward + Monte Carlo

### 新增 `walkforward_v2.py`
- 支持所有策略的 WF 验证
- --ml 选项: 每个 fold 重新训练 ML 模型
- --monte-carlo N: 打乱交易顺序 N 次
- 输出完整报告 JSON

## 执行工具

### `scripts/run_all_phases.sh`
- 一键跑完 Phase 1-3 全部 11 个验证
- 6 个纯规则 WF + 3 个 ML WF + 2 个 MC 仿真

### `scripts/summarize_results.py`
- 解析所有 WF 报告, 生成对比表
- 自动检查上线 gate (fold 胜率 ≥50%, OOS PnL > 0, 交易 ≥50)

### `scripts/apply_settings_patch_v22.py`
- 自动更新 settings.yaml

### `scripts/validate_v22.py`
- 全量检查所有修改

### `scripts/cleanup.sh`
- 删除冗余文件

## 测试

### `tests/test_all.py`
- 17+ 测试: MeanReversion / MACD 过滤器 / TripleEMA 过滤器 / ML / Registry / EnhancedExit

---

## 需要删除的文件

```
engine.py              (根目录, 与 core/engine.py 重复)
settings.yaml          (根目录, 与 config/settings.yaml 重复)
main_patch.py          (临时补丁)
integration_patch.py   (临时补丁)
modified_files.json    (构建产物)
```

---

## 执行步骤

```bash
cd /path/to/Trader

# 1. 解压覆盖
unzip -o Trader_v22_patch.zip

# 2. 验证
python scripts/validate_v22.py

# 3. 更新配置
python scripts/apply_settings_patch_v22.py

# 4. 跑测试
python -m pytest tests/test_all.py -v

# 5. 清理冗余
bash scripts/cleanup.sh

# 6. 一键跑全部验证 (需要数分钟~十几分钟)
bash scripts/run_all_phases.sh

# 7. 查看结果汇总
python scripts/summarize_results.py

# 8. 将 analysis/output/wf_*.json 返回给我
```

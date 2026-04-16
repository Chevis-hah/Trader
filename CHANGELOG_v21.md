# CHANGELOG — v2.1 修复版

> 日期: 2026-04-13
> 状态: 策略信号修复 + 架构清理 + 因子扫描工具

---

## 策略层修复

### MACD Momentum (`alpha/macd_momentum_strategy.py`)
- **修复 trailing_stop** — 从固定倍数改为自适应：盈利越多追踪越宽，解决 "21次止损全负" 问题
  - 新增 `trail_atr_base` / `trail_profit_bonus` / `trail_max_mult` / `trail_min_mult`
  - 新增波动率扩张检测：ATR 扩大 1.5x 时自动放宽追踪
- **新增 Regime Gate** — `regime_adx_min=22.0`，仅 ADX ≥ 22 时入场
  - 依据诊断数据：ADX > 30 胜率 57.9%，ADX < 20 胜率虽 45.5% 但 PF 低
- **校正滑点** — `slippage_pct` 从 0.001 (10bps) 提升到 0.0021 (21bps)
  - BTC 真实滑点 20.9bps，ETH 25.8bps
- **差异化冷却** — 止损后冷却 8 bar，信号出场后冷却 5 bar

### Triple EMA (`alpha/triple_ema_strategy.py`)
- **新增 Regime Gate** — `regime_adx_min=25.0`，仅 ADX ≥ 25 时入场
  - 依据：ADX > 30 时 1 笔 +1020，ADX < 30 时 6 笔 -172
- **放宽回踩条件** — 增加交易频率（原 4 笔太少）
  - `pullback_min_atr`: -0.90 → -1.20
  - `pullback_max_atr`: 0.60 → 0.80
  - `min_volume_ratio`: 0.90 → 0.85
  - 新增 MACD 柱加速确认（不要求回踩到 EMA21）
- **自适应追踪止损** — 替代原固定追踪
- **校正滑点** — 同 MACD

### 增强出场 (`alpha/enhanced_exit.py`)
- 分批止盈阈值调高：P1 从 2.5 → 3.0 ATR（避免过早锁利）
- 追踪下限从 1.5 → 1.8（避免正常波动触出）
- 新增保本止损：盈利 > 2 ATR 后自动拉到保本

### 策略注册表 (`alpha/strategy_registry.py`)
- 新增 Grid 策略注册
- 支持 dict 覆盖模式增强

---

## 新增工具

### 因子信号扫描 (`factor_signal_scan.py`)
- 扫描全部 200+ 因子的预测力
- 每个因子计算 IC / ICIR / t-stat / 单调性 / 胜率 / 桶收益
- 按大类分组汇总（动量/波动率/量价/均值回归/趋势/震荡/微观/regime）
- 标记 STRONG_SIGNAL / SIGNIFICANT / WEAK / NO_SIGNAL
- 输出 JSON 报告供下一步分析

### 配置校正补丁 (`scripts/apply_settings_patch.py`)
- 自动更新 settings.yaml 中的滑点和策略参数
- 支持 dry-run 预览

### 验证脚本 (`scripts/validate_v21.py`)
- 检查所有修改是否正确加载
- 验证参数校正、regime gate、自适应追踪止损

### 清理脚本 (`scripts/cleanup.sh`)
- 删除冗余文件（重复的 engine.py / settings.yaml / patch 文件）

### 测试框架 (`tests/test_strategies.py`)
- MACD Momentum 核心逻辑测试（regime gate / MACD cross / 仓位 / 追踪止损）
- Triple EMA 核心逻辑测试
- 增强出场测试（分批止盈 / 保本止损）
- 策略注册表测试

---

## 需要删除的文件

| 文件 | 原因 |
|------|------|
| `engine.py` (根目录) | 与 `core/engine.py` 重复 |
| `settings.yaml` (根目录) | 与 `config/settings.yaml` 重复 |
| `main_patch.py` | 临时补丁，已合并 |
| `integration_patch.py` | 临时补丁，已合并 |
| `modified_files.json` | 构建产物 |

---

## 下一步

1. 运行 `python scripts/validate_v21.py` 验证修改
2. 运行 `python scripts/apply_settings_patch.py` 更新配置
3. 运行 `python factor_signal_scan.py --db data/quant.db` 扫描因子
4. 将 `factor_scan_report.json` 返回给我分析
5. 基于因子扫描结果决定是否需要重新设计信号逻辑

# Trader - Crypto Quant Engine v2.3

## Project Overview

Binance USDT-M 永续合约量化交易系统，支持 paper trading / live trading / backtesting。  
v2.3 在 v2.2 因子驱动实验的基础上做了**收敛**：趋势策略（MACD、TripleEMA）去掉 walk-forward 上过拟合的过度拉伸过滤，回到更稳健的规则集；均值回归侧增加**趋势回避门**；默认配置**移除 1h 与 regime**，并加入**横截面路线 A**（`cross_sectional_momentum` + `universe` + `cross_sectional_backtest`）作为下一阶段实验脚手架。

**仍成立的经验结论**（来自 v2.2 扫描，见 `CHANGELOG_v22.md`）:

- BTC/ETH 在若干尺度上呈现强均值回归型因子（如 ma_dev_240、zscore_240）。
- 复杂入场过滤若未经 OOS 严格验证，容易把交易次数压到过少而导致假象。

**v2.3 相对 v2.2 的工程动作**:

- 规则策略「做减法」+ 配置清理 + 冒烟测试 `tests/test_v23.py`。
- 同步批次与文件级清单见 **`SYNC_UPDATE_LOG.md`**。

## Architecture

```
main.py                    # CLI 入口: --sync-data / --validate / --backtest / 默认实盘
├── core/engine.py         # TradingEngine（REST + 规则策略）
├── config/                # 配置加载 (YAML + 环境变量解析)
├── data/                  # 数据层 + universe 构建（路线 A）
├── alpha/                 # 策略注册表 + 规则策略(v2.3) + ML + 横截面实验
├── execution/             # 执行: OrderExecutor + PositionTracker
├── risk/                  # 风控: RiskManager 三层防御
├── analysis/              # 诊断: 数据质量 + 参数敏感性
├── scripts/               # 运维: cleanup_v23 / WF 一键脚本 / 验证
├── tests/                 # 单元测试: test_v23 + 既有用例
└── utils/                 # 日志 + 指标计算
```

## Data Flow（主路径）

```
main.py → TradingEngine.run_cycle():
  1. BinanceClient REST 增量获取 klines → Storage(SQLite)
  2. Strategy.prepare_features() → FeatureEngine 特征
  3. Strategy.should_enter() / check_exit() → 信号
     - 趋势策略 (v2.3): MACD / TripleEMA 以 WF 支持的规则为主，无 v2.2 那套过度拉伸堆叠
     - 均值回归: 震荡信号 + 趋势回避门
  4. RiskManager → 风控
  5. OrderExecutor → 成交 / 模拟
  6. PositionTracker + Storage 持久化
```

## Key Conventions

- **活跃代码**: 子目录版本（`data/client.py`, `core/engine.py`）是当前使用版本。
- **策略注册**: `should_enter` / `check_exit` / `calc_position` / `prepare_features` / `on_trade_closed`，经 `strategy_registry` 映射名称到类。
- **配置驱动**: `config/settings.yaml` 的 `strategy` 节；v2.3 默认不再包含 `1h` universe 与 `regime` 策略块（与 `test_v23` 一致）。
- **同步追溯**: 网页端合并记录写在 **`SYNC_UPDATE_LOG.md`**。

## Commands

```bash
python main.py --sync-data
python main.py --validate
python main.py --backtest --strategy mean_reversion

# v2.3 冒烟测试（推荐每次同步后跑）
python -m pytest tests/test_v23.py -v

# 仅 4h 的 walk-forward 流水线（耗时视数据而定）
bash scripts/run_all_phases.sh

# 路线 A（横截面 MVP）
bash scripts/run_mvp_path_a.sh

# 可选：归档死代码到 archive/v22_deprecated/（执行前请备份）
bash scripts/cleanup_v23.sh

python scripts/summarize_results.py
```

## Testing & Validation

- **v2.3**: `python -m pytest tests/test_v23.py -v`
- 回测: `main.py --backtest` 或 `backtest_runner.py`
- Walk-forward: `walkforward_v2.py`（脚本层已偏向 4h，见 `scripts/run_all_phases.sh`）

## Dependencies

- Python 3.8+
- 见 `requirements.txt`

## File Map（v2.3 增量与要点）

| 文件/目录 | 说明 |
|-----------|------|
| `SYNC_UPDATE_LOG.md` | 各次 `sync_to_repo.py` 合入清单 |
| `CHANGELOG_v23.md` | v2.3 版本级变更说明 |
| `docs/README_v23_MIGRATION.md` | 迁移包使用说明 |
| `alpha/cross_sectional_momentum.py` | 横截面策略（路线 A） |
| `data/universe.py` | Universe 构建 |
| `cross_sectional_backtest.py` | 横截面 MVP 回测入口 |
| `scripts/cleanup_v23.sh` | v2.3 归档脚本（幂等 mv，非交互） |
| `scripts/run_mvp_path_a.sh` | 路线 A 启动 |
| `tests/test_v23.py` | v2.3 冒烟测试 |

## Important Notes

- 新策略或改名须同步 `alpha/strategy_registry.py` 与 `config/settings.yaml`。
- 修改风控与执行参数前评估实盘影响。
- **`cleanup_v23.sh`** 会移动/归档文件，仅在已备份且接受路径变更时执行。

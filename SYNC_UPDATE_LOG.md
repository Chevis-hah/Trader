# Trader 同步更新记录

本文件记录通过 `sync_to_repo.py` 从 `web_dev_files/` 等目录合并回仓库的批次与要点，便于追溯网页端改动的落地情况。

---

## 2026-04-16 — `Trader_v23_cleanup`（v2.3 清理与路线 A 脚手架）

**源目录**: `/DATA/workshop/personal/cu/web_dev_files/Trader_v23_cleanup/`  
**命令**: `python sync_to_repo.py /DATA/workshop/personal/cu/web_dev_files/Trader_v23_cleanup/ --no-confirm`（在 `/DATA/workshop/personal/cu/Trader` 执行）

**新增文件**

- `alpha/cross_sectional_momentum.py` — 横截面动量/因子与组合权重（路线 A）。
- `cross_sectional_backtest.py` — 路线 A MVP 回测入口。
- `data/universe.py` — Top N 滚动 universe 构建。
- `docs/DEPRECATED_FACTORS_AND_STRATEGIES.md` — 失效因子与策略沉淀。
- `docs/README_v23_MIGRATION.md` — v2.2→v2.3 迁移包说明与步骤。
- `docs/RESEARCH_2025_SOTA_QUANT_ARCHITECTURES.md` — 2025 SOTA 架构调研笔记。
- `scripts/cleanup_v23.sh` — 将死代码/证伪结果归档到 `archive/v22_deprecated/`（执行前请自行备份仓库）。
- `scripts/run_mvp_path_a.sh` — 路线 A 横截面 MVP 一键脚本。
- `tests/test_v23.py` — v2.3 冒烟测试（策略实例化、配置、横截面因子、已删模块检测）。

**修改文件**

- `alpha/macd_momentum_strategy.py` — v2.3：回滚至接近 v2.0 的纯规则，移除 v2.2 中过拟合的过度拉伸类过滤与无效参数（`min_rsi`/`max_rsi`/`max_holding_bars` 等）。
- `alpha/mean_reversion_strategy.py` — v2.3：趋势回避门（如 ADX 上限、单调上涨检测等，与迁移文档一致）。
- `alpha/triple_ema_strategy.py` — v2.3：去掉过度拉伸过滤，与 WF 结论对齐。
- `config/settings.yaml` — 删除 `1h` universe 条目、调整执行/策略相关配置（与 `test_v23` 中 YAML 断言一致）；滑点按阶段实测修正（见 CHANGELOG_v23）。
- `scripts/run_all_phases.sh` — 仅保留 4h 相关 walk-forward；去掉 Phase 3 的 1h ML+MC 段落；汇总提示改为指向 `run_mvp_path_a.sh` 的路线 A 说明。

**跳过（源与目标一致）**

- `tests/__init__.py`

**验证**

- `python -m pytest tests/test_v23.py -v`（本机 16 passed）。

**文档**

- 根目录与各子目录 `CLAUDE.md` 已按 v2.3 要点修订；本条目为权威变更列表。
- 版本级说明见 `CHANGELOG_v23.md`。

**后续执行（备份后）**

- 已在仓库根执行 `bash scripts/cleanup_v23.sh`，归档 6 个文件至 `archive/v22_deprecated/`（含 `alpha/enhanced_exit.py`、`alpha/regime_allocator_v2.py`、`config/symbols_extended_v22.yaml`、部分 `analysis/output/wf_*_1h.json` / `wf_regime_4h.json` 等，以本机当时存在者为准）。此后 `python -m pytest tests/test_v23.py -v` 仍为 16 passed。
- `git push` 与远端合并需在本机凭据与网络可用时单独完成。

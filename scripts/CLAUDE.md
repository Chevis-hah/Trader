# scripts/ - 运维脚本

## 职责

开发环境搭建、配置校正、清理冗余文件、版本验证、一键 walk-forward 与路线 A 启动。

## 脚本（v2.3）

| 脚本 | 用途 |
|------|------|
| `setup_venv_wsl.sh` | WSL2 下创建 venv 并安装依赖 |
| `apply_settings_patch.py` | v2.1 配置校正（历史） |
| `apply_settings_patch_v22.py` | v2.2 配置校正（历史） |
| `cleanup.sh` | 删除顶层重复 engine/settings 等 |
| **`cleanup_v23.sh`** | **v2.3** 幂等归档：将现存死代码与证伪 wf 等移至 `archive/v22_deprecated/`（执行前请备份） |
| `validate_v21.py` / `validate_v22.py` | 历史版本自检 |
| **`run_all_phases.sh`** | **v2.3**：仅 **4h** 相关 WF；去掉 1h ML+MC 段；结尾指向路线 A |
| **`run_mvp_path_a.sh`** | **v2.3 新增** — 横截面 MVP 一键入口 |
| `summarize_results.py` | 汇总 `wf_*.json` 等 |

## 用法

### 环境与清理

```bash
bash scripts/setup_venv_wsl.sh
source .venv/bin/activate
bash scripts/cleanup.sh --dry-run
```

### v2.3 验证（推荐）

```bash
python -m pytest tests/test_v23.py -v
```

### Walk-forward（仅 4h 流水线）

```bash
bash scripts/run_all_phases.sh
python scripts/summarize_results.py
```

### 路线 A

```bash
bash scripts/run_mvp_path_a.sh
```

### 归档死代码（先备份再执行）

```bash
bash scripts/cleanup_v23.sh   # 非交互，直接 mv 到 archive/v22_deprecated/
```

批次级变更列表见根目录 **`SYNC_UPDATE_LOG.md`**。

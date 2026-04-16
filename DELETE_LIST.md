# 需要删除的文件

## 必须删除

```
engine.py              # 与 core/engine.py 重复
settings.yaml          # 与 config/settings.yaml 重复
main_patch.py          # 临时补丁
integration_patch.py   # 临时补丁
modified_files.json    # 构建产物
```

## 建议删除

```
phase2_report.json     # 93KB, 已分析完, 可归档到 analysis/output/
```

## Phase 4 后删除 (先迁移策略)

```
backtest_arena.py      # 72KB, 含 S01-S09 策略
                       # 迁移到 alpha/ 后再删
```

## v2.1 遗留文件 (已被 v2.2 覆盖)

```
scripts/validate_v21.py        # 被 validate_v22.py 替代
scripts/apply_settings_patch.py # 被 apply_settings_patch_v22.py 替代
CHANGELOG_v21.md               # 保留或归档到 docs/
```

## 执行命令

```bash
bash scripts/cleanup.sh
```

#!/bin/bash
# 清理冗余文件
# bash scripts/cleanup.sh --dry-run

DRY=false; [[ "$1" == "--dry-run" ]] && DRY=true && echo "🔍 DRY RUN"

FILES=(engine.py settings.yaml main_patch.py integration_patch.py modified_files.json)

for f in "${FILES[@]}"; do
    if [[ -f "$f" ]]; then
        $DRY && echo "  📝 将删除: $f" || (rm -f "$f" && echo "  ✅ 已删: $f")
    fi
done

echo ""
echo "建议手动确认后删除:"
echo "  - phase2_report.json (93KB, 可移到 analysis/output/)"
echo "  - backtest_arena.py (72KB, 先迁移策略再删)"

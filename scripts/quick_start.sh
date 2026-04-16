#!/bin/bash
# ============================================================
# quick_start.sh — 一键部署 v2.2 并执行全部验证
#
# 用法:
#   cd /path/to/Trader
#   bash scripts/quick_start.sh
#
# 这个脚本会:
#   1. 安装/更新依赖 (lightgbm)
#   2. 验证代码修改
#   3. 更新配置
#   4. 跑单元测试
#   5. 清理冗余文件
#   6. 执行 Phase 1-3 全部 WF 验证
#   7. 汇总结果
# ============================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Trader v2.2 — 一键部署 & 验证                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ---- Step 1: 依赖 ----
echo ""
echo "━━━ Step 1/7: 安装依赖 ━━━"
pip install lightgbm --break-system-packages -q 2>/dev/null || \
pip install lightgbm -q 2>/dev/null || \
echo "⚠️ lightgbm 安装失败, 将使用 sklearn 降级模式"

pip install pytest --break-system-packages -q 2>/dev/null || \
pip install pytest -q 2>/dev/null || true

# ---- Step 2: 验证代码 ----
echo ""
echo "━━━ Step 2/7: 验证代码修改 ━━━"
python scripts/validate_v22.py
echo ""

# ---- Step 3: 更新配置 ----
echo "━━━ Step 3/7: 更新配置 ━━━"
python scripts/apply_settings_patch_v22.py || echo "⚠️ 配置更新跳过 (可能已是最新)"
echo ""

# ---- Step 4: 单元测试 ----
echo "━━━ Step 4/7: 单元测试 ━━━"
python -m pytest tests/test_all.py -v --tb=short 2>/dev/null || \
python tests/test_all.py 2>&1 || echo "⚠️ 部分测试失败"
echo ""

# ---- Step 5: 清理 ----
echo "━━━ Step 5/7: 清理冗余文件 ━━━"
bash scripts/cleanup.sh 2>/dev/null || true
echo ""

# ---- Step 6: Walk-forward 验证 ----
echo "━━━ Step 6/7: Walk-forward 全量验证 ━━━"
echo "⏱️ 预计 10-20 分钟..."
echo ""
bash scripts/run_all_phases.sh
echo ""

# ---- Step 7: 汇总 ----
echo "━━━ Step 7/7: 结果汇总 ━━━"
python scripts/summarize_results.py
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  全部完成！                                                  ║"
echo "║                                                            ║"
echo "║  下一步: 将 analysis/output/wf_*.json 全部返回给我            ║"
echo "║  我会据此确定最终上线策略和仓位分配                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

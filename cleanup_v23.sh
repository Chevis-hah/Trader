#!/usr/bin/env bash
# ============================================================
# Trader v2.2 → v2.3 清理脚本
#
# 基于 DEPRECATED_FACTORS_AND_STRATEGIES.md 的结论，
# 删除经 walk-forward 证伪的策略、配置、因子。
#
# 用法：
#   cd /path/to/Trader
#   bash scripts/cleanup_v23.sh
#
# 本脚本是幂等的：多次运行结果一致。
# 所有删除前先归档到 archive/v22_deprecated/
# ============================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Trader v2.2 → v2.3 死代码清理                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ----------------------------------------------------------------
# Step 1: 归档目录
# ----------------------------------------------------------------
echo ""
echo "━━━ Step 1/5: 创建归档目录 ━━━"
mkdir -p archive/v22_deprecated/alpha
mkdir -p archive/v22_deprecated/config
mkdir -p archive/v22_deprecated/analysis_output
echo "  ✓ archive/v22_deprecated/"

# ----------------------------------------------------------------
# Step 2: 归档并删除死策略
# ----------------------------------------------------------------
echo ""
echo "━━━ Step 2/5: 归档失效策略 ━━━"

for f in \
    alpha/regime_adaptive_strategy.py \
    alpha/regime_allocator_v2.py \
    alpha/enhanced_exit.py \
    config/symbols_extended_v22.yaml \
    main_patch.py \
    integration_patch.py \
    modified_files.json; do
    if [ -f "$f" ]; then
        dst="archive/v22_deprecated/${f}"
        mkdir -p "$(dirname "$dst")"
        mv "$f" "$dst"
        echo "  → archived: $f"
    fi
done

# ----------------------------------------------------------------
# Step 3: 归档 1h 时间框架的 WF 结果
# ----------------------------------------------------------------
echo ""
echo "━━━ Step 3/5: 归档 1h / regime 的 WF 结果 ━━━"

for f in analysis/output/wf_*_1h.json analysis/output/wf_regime_*.json; do
    if [ -f "$f" ]; then
        mv "$f" "archive/v22_deprecated/analysis_output/"
        echo "  → archived: $f"
    fi
done

# ----------------------------------------------------------------
# Step 4: 打印需要手动 diff 的文件
# ----------------------------------------------------------------
echo ""
echo "━━━ Step 4/5: 以下文件需要手动 diff 修改 ━━━"
cat <<'EOF'

  [1] alpha/macd_momentum_strategy.py
      删除：zscore / ma_dev / keltner 三层过滤器
      删除：min_rsi, max_rsi, max_holding_bars 参数
      保留：MACD cross + adx_14 >= 20

  [2] alpha/triple_ema_strategy.py
      删除：v2.2 的过度拉伸过滤段落
      在 settings.yaml 中把 triple_ema.enabled 设为 false

  [3] alpha/mean_reversion_strategy.py
      新增：趋势回避门
        - if row.get("adx_14", 0) > 28: return False
        - 最近 20 bar 单调性检查（上涨或下跌比例 > 85% 则禁用）

  [4] config/settings.yaml
      删除：universe.intervals 中的 "1h"
      删除：strategy.regime 整段
      修改：execution.slippage_bps (BTC: 10→21, ETH: 10→26)
      新增：strategy.mean_reversion.trend_avoidance

  [5] scripts/run_all_phases.sh
      删除：所有 --interval 1h 的行

EOF

# ----------------------------------------------------------------
# Step 5: 统计清理效果
# ----------------------------------------------------------------
echo "━━━ Step 5/5: 清理效果统计 ━━━"
archived_count=$(find archive/v22_deprecated -type f 2>/dev/null | wc -l)
echo "  ✓ 已归档文件数: $archived_count"

if [ -d "alpha" ]; then
    remaining=$(find alpha -name "*.py" | wc -l)
    echo "  ✓ 剩余 alpha 策略文件: $remaining"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  清理完成。                                                   ║"
echo "║                                                              ║"
echo "║  下一步:                                                     ║"
echo "║    1. 按 Step 4 手动修改 5 个文件                             ║"
echo "║    2. 运行 python -m pytest tests/ 确认无破坏                ║"
echo "║    3. 跑 bash scripts/run_all_phases.sh 重新验证              ║"
echo "║    4. 如果 mean_reversion 4h fold 胜率达到 45%+, 开始路线 A  ║"
echo "║       MVP（横截面 momentum）                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

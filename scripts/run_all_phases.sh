#!/usr/bin/env bash
# ============================================================
# Walk-Forward 全量验证流水线 v2.3
#
# v2.2 → v2.3 变更:
#   - 删除所有 --interval 1h 配置 (全部证伪)
#   - 删除 regime 策略的 WF 跑批 (0 trades)
#   - 只保留 4h 时间框架的 3 个规则策略
#
# 用法:
#   bash scripts/run_all_phases.sh
#
# 输出:
#   analysis/output/wf_*.json
# ============================================================

set -e

mkdir -p analysis/output logs

DB="data/quant.db"
SYMBOLS="BTCUSDT ETHUSDT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Trader v2.3 Walk-Forward 验证流水线                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ----------------------------------------------------------------
# Round 1: MACD Momentum 4h (v2.3 回滚版)
# ----------------------------------------------------------------
echo ""
echo "━━━ [1/3] macd_momentum 4h ━━━"
python walkforward_v2.py \
    --db "$DB" \
    --strategy macd_momentum \
    --interval 4h \
    --symbols $SYMBOLS \
    --train-days 360 \
    --test-days 60 \
    --output analysis/output/wf_macd_momentum_4h.json \
    2>&1 | tee logs/wf_macd_4h.log

# ----------------------------------------------------------------
# Round 2: Triple EMA 4h (候选, 样本不足但保留基准)
# ----------------------------------------------------------------
echo ""
echo "━━━ [2/3] triple_ema 4h (候选基准) ━━━"
python walkforward_v2.py \
    --db "$DB" \
    --strategy triple_ema \
    --interval 4h \
    --symbols $SYMBOLS \
    --train-days 360 \
    --test-days 60 \
    --output analysis/output/wf_triple_ema_4h.json \
    2>&1 | tee logs/wf_triple_ema_4h.log

# ----------------------------------------------------------------
# Round 3: Mean Reversion 4h (v2.3 加趋势回避门)
# ----------------------------------------------------------------
echo ""
echo "━━━ [3/3] mean_reversion 4h (重点验证) ━━━"
python walkforward_v2.py \
    --db "$DB" \
    --strategy mean_reversion \
    --interval 4h \
    --symbols $SYMBOLS \
    --train-days 360 \
    --test-days 60 \
    --output analysis/output/wf_mean_reversion_4h.json \
    2>&1 | tee logs/wf_mean_reversion_4h.log

# ----------------------------------------------------------------
# 已删除的 rounds (仅供参考, 不再执行):
#   ❌ macd_momentum 1h        - wf_macd_momentum_1h PnL -790
#   ❌ mean_reversion 1h       - wf_mean_reversion_1h PnL -3015
#   ❌ regime 4h               - 0 trades
#   ❌ ML filtered 变体        - v2.3 MVP 阶段先验证纯规则
#   ❌ Monte Carlo 变体        - 等主策略稳定后再加
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------
echo ""
echo "━━━ 结果汇总 ━━━"
python scripts/summarize_results.py || echo "⚠️  summarize_results.py 未找到, 请手动查看 analysis/output/"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  完成                                                        ║"
echo "║                                                              ║"
echo "║  判决标准:                                                   ║"
echo "║    mean_reversion 4h fold 胜率 ≥ 45% & OOS PnL > 0          ║"
echo "║      → v2.3 规则策略方向成功, 可考虑纸面交易                 ║"
echo "║    fold 胜率 < 45%                                           ║"
echo "║      → 规则策略天花板到了, 启动路线 A:                       ║"
echo "║          bash scripts/run_mvp_path_a.sh                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"

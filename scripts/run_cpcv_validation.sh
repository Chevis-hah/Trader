#!/usr/bin/env bash
# Phase 2B-T05: CPCV validation CI 入口
# 跑一次完整的 CPCV 审计, 退出码反映 G1 门槛:
#   exit 0 — 全部策略 PBO < 40%
#   exit 2 — 至少一个策略 PBO ≥ 40% (CI 应红)
# 其他错误正常退出码。

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${CPCV_MODE:-demo}"                       # demo (默认) 或 real
DB="${DB_PATH:-data/quant.db}"
START="${START:-2022-01-01}"
OUTPUT="${OUTPUT:-analysis/output/cpcv_audit_report.md}"
JSON_OUTPUT="${JSON_OUTPUT:-analysis/output/cpcv_audit.json}"

echo "============================================================"
echo "  CPCV Validation (P2B-T05)"
echo "============================================================"
echo "  mode     : $MODE"
echo "  db       : $DB   (real 模式生效)"
echo "  start    : $START"
echo "  output   : $OUTPUT"
echo "  json     : $JSON_OUTPUT"
echo

if [[ "$MODE" == "demo" ]]; then
  python scripts/run_cpcv_audit.py --demo \
    --output "$OUTPUT" \
    --json-output "$JSON_OUTPUT" \
    "$@" || EC=$?
else
  python scripts/run_cpcv_audit.py \
    --db "$DB" --start "$START" \
    --output "$OUTPUT" \
    --json-output "$JSON_OUTPUT" \
    "$@" || EC=$?
fi

EC="${EC:-0}"

echo
if [[ $EC -eq 0 ]]; then
  echo "✅ CPCV validation PASS (no strategy with PBO > 40%)"
elif [[ $EC -eq 2 ]]; then
  echo "❌ CPCV validation FAIL — at least one strategy has PBO > 40%"
  echo "   See $OUTPUT for details."
else
  echo "⚠️  CPCV validation encountered error (exit $EC)"
fi

exit $EC

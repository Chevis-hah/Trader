#!/usr/bin/env bash
# Run Funding Harvester backtest with default Phase 2A parameters
# 用法: bash scripts/run_funding_harvester.sh [extra args passed to .py]

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="analysis/output"
mkdir -p "$OUTPUT_DIR"

DB="${DB_PATH:-data/quant.db}"
START="${START:-2023-01-01}"
END="${END:-}"
CAPITAL="${CAPITAL:-100000}"

echo "==> Funding Harvester 回测"
echo "    DB:      $DB"
echo "    window:  $START .. ${END:-now}"
echo "    capital: $CAPITAL USDT"

END_ARG=""
if [[ -n "$END" ]]; then
  END_ARG="--end $END"
fi

# 基线跑
python funding_harvester_backtest.py \
  --db "$DB" \
  --start "$START" \
  $END_ARG \
  --capital "$CAPITAL" \
  --output "$OUTPUT_DIR/funding_harvester_baseline.json" \
  "$@"

echo
echo "==> 基线结果:"
python -c "import json; r=json.load(open('$OUTPUT_DIR/funding_harvester_baseline.json')); \
  s=r['summary']; print(f\"  Sharpe={s['sharpe_ratio']}  Ret={s['annualized_return_pct']}%  \" \
                       f\"MDD={s['max_drawdown_pct']}%  N={s['n_trades']}  {s['verdict']}\")"

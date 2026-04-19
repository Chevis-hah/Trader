#!/usr/bin/env bash
# Phase 2 收尾流水线 - Decision Gate 1 数据采集
# 本地/实盘数据环境专属: 见开发模型反馈 §4
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

# 让子脚本里的 `python` 与单元测试使用同一 venv
# 从 Windows 侧调用 bash 时 PATH 常含带括号的路径, 会破坏本脚本; 故显式收敛
VENV_BIN="$(pwd)/.venv/bin"
export PATH="${VENV_BIN}:/usr/local/bin:/usr/bin:/bin"
PY="${VENV_BIN}/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: 未找到可执行 venv: $PY" >&2
  exit 1
fi

mkdir -p analysis/output/phase2_closeout logs

echo "=== Step 1/5: 环境 smoke ==="
$PY -m unittest discover -s tests 2>&1 | tail -5

echo
echo "=== Step 2/5: 数据覆盖度 ==="
$PY -c "
from data.storage import Storage
s = Storage('data/quant.db')
with s._conn() as c:
  row = c.execute('SELECT COUNT(DISTINCT symbol), COUNT(*) FROM funding_rates').fetchone()
  print(f'Funding: {row[0]} symbols, {row[1]} rows')
  row = c.execute('SELECT COUNT(DISTINCT symbol), COUNT(*) FROM klines WHERE interval=\"4h\"').fetchone()
  print(f'Klines 4h: {row[0]} symbols, {row[1]} rows')
" 2>&1 | tee analysis/output/phase2_closeout/01_data_coverage.txt

echo
echo "=== Step 3/5: Funding Harvester 基线回测 ==="
bash scripts/run_funding_harvester.sh 2>&1 | tee analysis/output/phase2_closeout/02_funding_baseline.log

echo
echo "=== Step 4/5: 敏感度扫描 (可能耗时 10-60 分钟) ==="
$PY scripts/run_funding_sensitivity.py \
  --db data/quant.db --start 2023-01-01 \
  --output analysis/output/phase2_closeout/03_sensitivity.json \
  2>&1 | tee analysis/output/phase2_closeout/03_sensitivity.log

echo
echo "=== Step 5/5: CPCV 实盘审计 ==="
CPCV_MODE=real OUTPUT=analysis/output/phase2_closeout/04_cpcv_audit.md \
  JSON_OUTPUT=analysis/output/phase2_closeout/04_cpcv_audit.json \
  bash scripts/run_cpcv_validation.sh 2>&1 | tee analysis/output/phase2_closeout/04_cpcv.log

echo
echo "=== 完成 ==="
echo "证据集: analysis/output/phase2_closeout/"
ls -la analysis/output/phase2_closeout/

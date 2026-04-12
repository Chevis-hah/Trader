#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
rm -rf .venv
if command -v virtualenv >/dev/null 2>&1; then
  virtualenv .venv
elif python3 -m venv .venv; then
  :
else
  echo "请先安装其一: sudo apt install python3.12-venv   或   pip install --user virtualenv"
  exit 1
fi
# shellcheck source=/dev/null
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
echo "完成。激活: source .venv/bin/activate"

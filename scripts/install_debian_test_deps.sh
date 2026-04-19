#!/usr/bin/env bash
# ============================================================
# Debian/Ubuntu/WSL 测试环境依赖安装脚本
#
# 本地 agent 反馈 (2026-04-19): 在 WSL (Debian-based) 上,
# `pip install lightgbm` 会成功但 import 时若系统缺 `libgomp.so.1`
# 会抛 OSError 导致测试崩溃 (已在 `alpha/ml_lightgbm.py::_try_import_lgbm`
# 捕获 OSError 降级到 sklearn, 但实际使用 LightGBM 训练仍需此库)。
#
# 用法:
#   sudo bash scripts/install_debian_test_deps.sh
# ============================================================

set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "本脚本仅支持 Debian/Ubuntu 系统 (需要 apt-get)。"
  echo "其他系统请手动安装: libgomp / python3-venv"
  exit 1
fi

echo "==> apt-get update"
sudo apt-get update -y

echo "==> 安装 libgomp1 (LightGBM 运行时依赖)"
sudo apt-get install -y libgomp1

# 可选: 用于创建隔离的 venv
if ! command -v python3 >/dev/null 2>&1; then
  echo "==> 安装 python3"
  sudo apt-get install -y python3
fi

# 检测是否需要 python3-venv (用于 .venv 创建)
if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "==> 安装 python3-venv"
  sudo apt-get install -y python3-venv
fi

echo
echo "==> 完成。建议验证:"
echo "   python3 -c 'import lightgbm; print(lightgbm.__version__)'"
echo "   若仍报 OSError, 尝试: export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"

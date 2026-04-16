#!/usr/bin/env bash
# ============================================================
# 路线 A: 横截面 Momentum MVP 一键验证
#
# 依据:
#   - Liu-Tsyvinski-Wu 2022 (JoF)
#   - CTREND (JFQA Nov 2025)
#   - Unravel Finance 2025 开源方案
#
# 预期耗时: 10-30 分钟 (取决于 universe 大小)
# 预期 Sharpe:
#   1.5+  EXCELLENT  → 下一步加 CTREND 风格 ML 聚合
#   1.2+  GOOD       → 方向对, 继续细化
#   0.8+  MARGINAL   → 边缘, 检查成本
#   < 0.8 FAILED     → crypto momentum 衰减, 考虑其他方向
# ============================================================

set -e

mkdir -p analysis/output logs

DB="data/quant.db"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       路线 A MVP: 横截面 Momentum 多空策略                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ----------------------------------------------------------------
# 第一步: 检查 universe 中的 symbols 是否已同步
# ----------------------------------------------------------------
echo ""
echo "━━━ Step 1/3: 检查数据完整性 ━━━"
python <<'EOF'
from data.universe import UniverseBuilder
from data.storage import Storage

storage = Storage("data/quant.db")
ub = UniverseBuilder(storage)
all_syms = ub.get_all_symbols()

available = 0
for sym in all_syms:
    try:
        kl = storage.get_klines(sym, "1d", limit=10)
        if not kl.empty:
            available += 1
    except Exception:
        pass

print(f"Universe 期望: {len(all_syms)} symbols")
print(f"数据库实际:   {available} symbols")

if available < 20:
    print("\n⚠️  警告: 可用 symbols 少于 20, 横截面策略效果会很差")
    print("   建议先跑: python main.py --sync-data --universe-top 60")
EOF

# ----------------------------------------------------------------
# 第二步: 主回测 (全 universe, 7 天 rebalance)
# ----------------------------------------------------------------
echo ""
echo "━━━ Step 2/3: 主回测 (2022-01-01 至今, Top 50, 7d rebalance) ━━━"
python cross_sectional_backtest.py \
    --db "$DB" \
    --start 2022-01-01 \
    --top-n 50 \
    --rebalance-days 7 \
    --capital 100000 \
    --output analysis/output/xs_mom_top50_7d.json \
    2>&1 | tee logs/xs_mom_top50_7d.log

# ----------------------------------------------------------------
# 第三步: 敏感度测试 (不同参数)
# ----------------------------------------------------------------
echo ""
echo "━━━ Step 3/3: 敏感度测试 ━━━"

# 3a: 不同 universe 大小
for n in 30 50 80; do
    echo ""
    echo "--- Top ${n}, 7d rebalance ---"
    python cross_sectional_backtest.py \
        --db "$DB" \
        --start 2022-01-01 \
        --top-n $n \
        --rebalance-days 7 \
        --output "analysis/output/xs_mom_top${n}_7d.json" \
        2>&1 | tail -15
done

# 3b: 不同 rebalance 频率
for days in 3 7 14 30; do
    echo ""
    echo "--- Top 50, ${days}d rebalance ---"
    python cross_sectional_backtest.py \
        --db "$DB" \
        --start 2022-01-01 \
        --top-n 50 \
        --rebalance-days $days \
        --output "analysis/output/xs_mom_top50_${days}d.json" \
        2>&1 | tail -15
done

# ----------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------
echo ""
echo "━━━ 结果对比 ━━━"
python <<'EOF'
import json
from pathlib import Path

files = sorted(Path("analysis/output").glob("xs_mom_*.json"))
if not files:
    print("未找到结果文件")
    exit(0)

print(f"\n{'配置':<30} {'Sharpe':>8} {'年化%':>8} {'MaxDD%':>8} {'判决':<12}")
print("-" * 75)
for f in files:
    try:
        with open(f) as fp:
            data = json.load(fp)
        if "summary" not in data:
            continue
        s = data["summary"]
        v = data.get("verdict", {})
        name = f.stem.replace("xs_mom_", "")
        print(f"{name:<30} {s['sharpe_ratio']:>8.3f} "
              f"{s['annualized_return_pct']:>8.1f} "
              f"{s['max_drawdown_pct']:>8.1f} "
              f"{v.get('grade', '?'):<12}")
    except Exception as e:
        print(f"{f.stem}: ERROR {e}")
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  完成                                                        ║"
echo "║                                                              ║"
echo "║  下一步决策:                                                 ║"
echo "║                                                              ║"
echo "║    任一配置 Sharpe > 1.2:                                    ║"
echo "║      → crypto 横截面 momentum 仍有 edge                      ║"
echo "║      → 下一迭代: 加 funding rate carry 因子                  ║"
echo "║      → 再下一迭代: CTREND 风格的 ML 因子聚合                 ║"
echo "║                                                              ║"
echo "║    所有配置 Sharpe < 0.8:                                    ║"
echo "║      → crypto momentum 因子衰减严重                          ║"
echo "║      → 考虑大宗商品/外汇期货市场, 或转 LOB 做市方向          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

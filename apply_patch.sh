#!/bin/bash
# ============================================================
# apply_patch.sh — 自动将 Claude 输出的文件同步到 Trader 仓库
#
# 用法（在 Trader 仓库的父目录下执行）：
#   bash apply_patch.sh <claude_output_dir>
#
# 示例：
#   bash apply_patch.sh ~/Downloads/claude_output
#
# 工作流：
#   1. 扫描 claude_output 目录中的所有文件
#   2. 按目录结构复制到 Trader/ 仓库对应位置
#   3. 显示 diff 摘要
#   4. 自动 git add + commit（需确认）
#   5. 重新生成仓库快照
# ============================================================

set -euo pipefail

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

REPO_DIR="Trader"

# ---- 参数检查 ----
if [ $# -lt 1 ]; then
    echo -e "${RED}用法: bash apply_patch.sh <claude_output_dir>${NC}"
    echo "  示例: bash apply_patch.sh ~/Downloads/claude_output"
    exit 1
fi

OUTPUT_DIR="$1"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}错误: 目录不存在: $OUTPUT_DIR${NC}"
    exit 1
fi

if [ ! -d "$REPO_DIR" ]; then
    echo -e "${RED}错误: 未找到 $REPO_DIR 目录（请在仓库父目录下运行）${NC}"
    exit 1
fi

# ---- 扫描并复制文件 ----
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Trader 仓库自动更新工具${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}源目录:${NC}  $OUTPUT_DIR"
echo -e "${YELLOW}目标仓库:${NC} $REPO_DIR"
echo ""

CHANGED=0
NEW=0

# 遍历 output 目录中的所有文件
while IFS= read -r -d '' file; do
    # 计算相对路径
    rel_path="${file#$OUTPUT_DIR/}"
    target="$REPO_DIR/$rel_path"
    target_dir=$(dirname "$target")

    # 确保目标目录存在
    mkdir -p "$target_dir"

    if [ -f "$target" ]; then
        # 文件已存在 → 检查是否有变化
        if ! diff -q "$file" "$target" > /dev/null 2>&1; then
            echo -e "  ${YELLOW}[修改]${NC} $rel_path"
            # 显示简要 diff
            diff --color=auto -u "$target" "$file" | head -20 || true
            echo "  ..."
            cp "$file" "$target"
            CHANGED=$((CHANGED + 1))
        else
            echo -e "  ${GREEN}[无变化]${NC} $rel_path"
        fi
    else
        echo -e "  ${GREEN}[新增]${NC} $rel_path"
        cp "$file" "$target"
        NEW=$((NEW + 1))
    fi
done < <(find "$OUTPUT_DIR" -type f -not -name '.*' -print0)

echo ""
echo -e "${GREEN}完成: ${CHANGED} 个文件修改, ${NEW} 个新增文件${NC}"

if [ $((CHANGED + NEW)) -eq 0 ]; then
    echo "没有需要更新的文件。"
    exit 0
fi

# ---- Git 操作 ----
echo ""
read -p "是否自动 git add + commit? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    cd "$REPO_DIR"
    git add -A
    
    # 生成 commit message
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
    COMMIT_MSG="feat(data): WebSocket实时数据流 + REST降级 [$TIMESTAMP]

- 新增 data/ws_manager.py: Binance WS 管理器（K线+订单簿）
- 更新 data/client.py: 新增 ws_base_url / get_ws_url()
- 更新 data/storage.py: 新增 save_orderbook_snapshot / cleanup_orderbook
- 更新 core/engine.py: 集成WS，历史→实时无缝衔接，REST降级
- 更新 config/settings.yaml: 新增 websocket 配置段
- 更新 requirements.txt: 启用 websockets / aiohttp"

    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}已提交: git commit 完成${NC}"
    
    read -p "是否 git push? [y/N] " push_confirm
    if [[ "$push_confirm" =~ ^[Yy]$ ]]; then
        git push
        echo -e "${GREEN}已推送到远程${NC}"
    fi
    
    cd ..
fi

# ---- 重新生成快照 ----
echo ""
read -p "是否重新生成仓库快照? [y/N] " snap_confirm
if [[ "$snap_confirm" =~ ^[Yy]$ ]]; then
    if [ -f "repo_to_file.py" ]; then
        python repo_to_file.py
        echo -e "${GREEN}快照已重新生成${NC}"
    else
        echo -e "${YELLOW}未找到 repo_to_file.py，请手动生成快照${NC}"
    fi
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  全部完成！可以开始下一轮迭代了${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"

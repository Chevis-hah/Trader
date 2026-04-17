#!/usr/bin/env python3
"""
sync_to_repo - 将目录下的新增/修改文件同步到 Trader 仓库

场景: 在网页端 Claude 修改代码后，下载的文件保持原有目录结构，
      用此脚本将修改同步回仓库。

用法:
  python sync_to_repo.py /path/to/downloads                     # 自动查找 Trader 仓库
  python sync_to_repo.py /path/to/downloads --repo /path/Trader  # 指定目标仓库
  python sync_to_repo.py /path/to/downloads --dry-run            # 仅预览，不实际修改
  python sync_to_repo.py /path/to/downloads --no-confirm         # 跳过确认，直接覆盖
  python sync_to_repo.py /path/to/downloads --backup             # 覆盖前备份原文件
"""

import argparse
import difflib
import filecmp
import shutil
import sys
from pathlib import Path

# ── 颜色输出 ──────────────────────────────────────────────────────────
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def color(text: str, *codes: str) -> str:
    return f"{''.join(codes)}{text}{RESET}"


def diff_files(src: Path, dst: Path) -> list[str]:
    """比较两个文件的差异，返回 unified diff 行"""
    try:
        src_lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        dst_lines = dst.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError:
        return []

    return list(difflib.unified_diff(
        dst_lines, src_lines,
        fromfile=f"a/{dst}",
        tofile=f"b/{src}",
    ))


def print_diff(diff_lines: list[str]):
    """带颜色输出 diff"""
    for line in diff_lines:
        if line.startswith("---") or line.startswith("+++"):
            print(color(line, BOLD))
        elif line.startswith("@@"):
            print(color(line, CYAN))
        elif line.startswith("+"):
            print(color(line, GREEN))
        elif line.startswith("-"):
            print(color(line, RED))
        else:
            print(line, end="")


def collect_files(src_dir: Path) -> list[Path]:
    """收集源目录下所有文件"""
    files = []
    for p in src_dir.rglob("*"):
        if p.is_file():
            files.append(p)
    files.sort()
    return files


def find_trader_repo() -> Path | None:
    """自动查找本地 Trader 仓库（需存在 main.py）"""
    script_dir = Path(__file__).resolve().parent
    for base in (script_dir, script_dir.parent, Path.cwd().resolve()):
        cand = base / "Trader"
        if cand.is_dir() and (cand / "main.py").is_file():
            return cand.resolve()
    return None


def sync(
    src_dir: Path,
    repo_dir: Path,
    dry_run: bool = False,
    no_confirm: bool = False,
    backup: bool = False,
):
    """执行同步"""

    src_files = collect_files(src_dir)
    if not src_files:
        print("源目录为空，没有文件需要同步。")
        return

    new_files = []       # 仓库中不存在的文件
    modified_files = []  # 仓库中已存在但内容不同的文件
    identical_files = [] # 内容完全相同的文件

    for src_file in src_files:
        rel_path = src_file.relative_to(src_dir)
        dst_file = repo_dir / rel_path

        if not dst_file.exists():
            new_files.append((src_file, rel_path, dst_file))
        elif not filecmp.cmp(src_file, dst_file, shallow=False):
            modified_files.append((src_file, rel_path, dst_file))
        else:
            identical_files.append((src_file, rel_path, dst_file))

    # ── 打印摘要 ──
    print(color(f"\n{'=' * 60}", BOLD))
    print(color(f"  源目录:   {src_dir}", CYAN))
    print(color(f"  目标仓库: {repo_dir}", CYAN))
    print(color(f"{'=' * 60}\n", BOLD))

    if identical_files:
        print(color(f"  跳过 {len(identical_files)} 个相同文件:", YELLOW))
        for _, rel, _ in identical_files:
            print(f"    {rel}")
        print()

    if not new_files and not modified_files:
        print("没有新增或修改的文件，无需同步。")
        return

    if new_files:
        print(color(f"  新增 {len(new_files)} 个文件:", GREEN))
        for _, rel, _ in new_files:
            print(color(f"    + {rel}", GREEN))
        print()

    if modified_files:
        print(color(f"  修改 {len(modified_files)} 个文件:", YELLOW))
        for _, rel, _ in modified_files:
            print(color(f"    ~ {rel}", YELLOW))
        print()

    # ── 展示 diff ──
    if modified_files:
        print(color("── 修改详情 ──", BOLD))
        for src_file, rel_path, dst_file in modified_files:
            diff_lines = diff_files(src_file, dst_file)
            if diff_lines:
                print()
                print_diff(diff_lines)
        print()

    # ── dry-run 则到此结束 ──
    if dry_run:
        print(color("[Dry Run] 以上为预览，未做任何修改。", YELLOW))
        return

    # ── 确认 ──
    total = len(new_files) + len(modified_files)
    if not no_confirm:
        answer = input(color(f"\n确认同步 {total} 个文件到仓库? [y/N] ", BOLD))
        if answer.strip().lower() not in ("y", "yes"):
            print("已取消。")
            return

    # ── 执行同步 ──
    done = 0
    for src_file, rel_path, dst_file in new_files + modified_files:
        # 备份
        if backup and dst_file.exists():
            bak = dst_file.with_suffix(dst_file.suffix + ".bak")
            shutil.copy2(dst_file, bak)

        # 创建父目录
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        # 复制
        shutil.copy2(src_file, dst_file)
        done += 1

        is_new = any(s[1] == rel_path for s in new_files)
        label = color("+", GREEN) if is_new else color("~", YELLOW)
        print(f"  {label} {rel_path}")

    print(color(f"\n完成! 已同步 {done} 个文件。", GREEN, BOLD))

    if backup:
        print(color("备份文件以 .bak 后缀保存在仓库中。", CYAN))


def main():
    parser = argparse.ArgumentParser(
        description="将目录下的新增/修改文件同步到 Trader 仓库"
    )
    parser.add_argument(
        "src_dir",
        help="源目录 (包含待同步文件的目录)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="目标仓库路径 (默认: 自动查找 Trader/ 目录)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览差异，不实际修改文件",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="跳过确认，直接覆盖",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="覆盖前将原文件备份为 .bak",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()

    # 网页端下载常见为「外层文件夹 + 唯一 Trader/ 子树」；若直接以外层为源会把文件同步到仓库的 Trader/Trader/ 下
    inner = src_dir / "Trader"
    if inner.is_dir():
        top = [p for p in src_dir.iterdir() if not p.name.startswith(".")]
        if len(top) == 1 and top[0] == inner:
            print(color(f"检测到单层补丁包装，源目录自动切换为: {inner}", CYAN))
            src_dir = inner

    # 解析仓库路径: 指定 > 自动查找
    if args.repo:
        repo_dir = Path(args.repo).resolve()
    else:
        repo_dir = find_trader_repo()
        if repo_dir is None:
            print("Error: 未找到 Trader 仓库（在脚本目录、上一级、当前目录下查找 Trader/main.py）。", file=sys.stderr)
            print("       请使用 --repo 显式指定仓库路径。", file=sys.stderr)
            sys.exit(1)
        print(color(f"自动找到 Trader 仓库: {repo_dir}", CYAN))

    if not src_dir.is_dir():
        print(f"Error: 源目录 {src_dir} 不存在", file=sys.stderr)
        sys.exit(1)
    if not repo_dir.is_dir():
        print(f"Error: 仓库目录 {repo_dir} 不存在", file=sys.stderr)
        sys.exit(1)

    # 防止误操作: 源和目标不能相同
    if src_dir == repo_dir:
        print("Error: 源目录和目标仓库不能相同", file=sys.stderr)
        sys.exit(1)

    sync(
        src_dir=src_dir,
        repo_dir=repo_dir,
        dry_run=args.dry_run,
        no_confirm=args.no_confirm,
        backup=args.backup,
    )


if __name__ == "__main__":
    main()

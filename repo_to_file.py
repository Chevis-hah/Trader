#!/usr/bin/env python3
"""
repo_to_file - 将 Trader 代码仓库整合成一个 txt 文件，方便上传给网页端 Claude

用法:
    python repo_to_file.py
        默认扫描「本脚本目录 / 上一级 / 当前工作目录」下的 Trader/（需存在 main.py），
        直接使用本地文件，不克隆、不拉取。

    python repo_to_file.py /path/to/Trader
        显式指定本地仓库路径（只读磁盘）。

    python repo_to_file.py https://github.com/Chevis-hah/Trader
        仅当参数是 http(s)/git@ URL 时才会 git clone（浅克隆）。

    python repo_to_file.py -o output.txt             # 指定输出文件名
    python repo_to_file.py --include "*.md,*.txt"    # 额外包含的 glob 模式
    python repo_to_file.py --exclude "*.csv,*.xlsx"  # 额外排除的 glob 模式
    python repo_to_file.py --no-gitignore            # 不读取 .gitignore
    python repo_to_file.py --max-size 50             # 单文件最大 KB 数 (默认 100)

每次本地改完后重新跑一遍，上传新的 txt 即可。
"""

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
# 无参数且未找到本地 Trader 时，提示里展示的远端示例（不会自动克隆）
REMOTE_REPO_EXAMPLE = "https://github.com/Chevis-hah/Trader"

# ── 默认包含的文件扩展名 ──────────────────────────────────────────────
INCLUDE_EXTENSIONS = {
    # Python
    ".py", ".pyi", ".pyx", ".pxd",
    # JavaScript / TypeScript
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # Web
    ".html", ".css", ".scss", ".less", ".svg",
    # Config / Data
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".env.local",
    # Shell
    ".sh", ".bash", ".zsh", ".fish",
    # Markup / Docs
    ".md", ".rst", ".txt", ".adoc",
    # Build
    ".cmake", ".mk", "Makefile", "Dockerfile",
    # Database
    ".sql",
    # Go
    ".go", ".mod", ".sum",
    # Rust
    ".rs", ".toml",
    # Java / JVM
    ".java", ".kt", ".scala", ".gradle", ".properties",
    # C/C++
    ".c", ".cpp", ".h", ".hpp", ".cc",
    # Other
    ".proto", ".graphql", ".prisma", ".tf", ".hcl",
}

# ── 默认排除的目录名 ──────────────────────────────────────────────────
EXCLUDE_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".nox",
    "venv", ".venv", "env", ".env",
    "dist", "build", "out", "target", ".next", ".nuxt",
    ".cargo", ".rustup",
    "vendor", "third_party",
    ".idea", ".vscode", ".vs",
    "static",
    "logs",       # 运行日志目录
    "models",     # ML 模型文件目录
    "data",       # 数据目录（含 .gitkeep 但无代码）
    "research",   # 研究笔记目录
}

# ── 默认排除的文件名模式 ──────────────────────────────────────────────
EXCLUDE_PATTERNS = {
    "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll", "*.exe",
    "*.zip", "*.tar", "*.gz", "*.rar", "*.7z",
    "*.woff", "*.woff2", "*.ttf", "*.eot",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.ico", "*.webp",
    "*.mp3", "*.mp4", "*.wav", "*.avi", "*.mov",
    "*.pdf", "*.doc", "*.docx", "*.ppt", "*.pptx",
    "*.xlsx", "*.xls",
    "*.db", "*.sqlite", "*.sqlite3",
    "*.pkl", "*.joblib", "*.h5", "*.hdf5",
    "*.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    ".DS_Store", "Thumbs.db",
    "*_snapshot.txt",  # 避免把旧的快照文件也打进去
}

# ── 二进制文件检测 ────────────────────────────────────────────────────
NULL_BYTE = b"\x00"

# 超过此大小的文件只显示前后部分（字节）
MAX_FILE_SIZE = 100_000  # 100KB


def is_binary(filepath: Path) -> bool:
    """检测文件是否为二进制文件 (读取前 8KB)"""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
        return NULL_BYTE in chunk
    except OSError:
        return True


def parse_gitignore(repo_root: Path) -> list[str]:
    """解析 .gitignore 文件，返回 glob 模式列表"""
    gitignore_path = repo_root / ".gitignore"
    patterns = []
    if gitignore_path.exists():
        with open(gitignore_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
    return patterns


def gitignore_match(rel_path: str, patterns: list[str]) -> bool:
    """简单检查路径是否匹配 gitignore 模式 (仅处理基本模式)"""
    for pattern in patterns:
        if pattern.startswith("!"):
            continue
        # 目录尾部模式 (如 data/*)
        base_pattern = pattern.rstrip("/")
        if base_pattern.endswith("/*"):
            dir_name = base_pattern[:-2]
            if rel_path.startswith(dir_name + "/") or rel_path.startswith(dir_name + os.sep):
                return True
        # 通配符匹配
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        # 对不含 / 的模式，匹配任意层级
        if "/" not in pattern and fnmatch.fnmatch(os.path.basename(rel_path), pattern):
            return True
    return False


def should_include_file(
    filepath: Path,
    rel_path: str,
    repo_root: Path,
    gitignore_patterns: list[str],
    use_gitignore: bool,
    extra_includes: list[str],
    extra_excludes: list[str],
    max_size_kb: int,
) -> bool:
    """判断文件是否应该包含在输出中"""

    # 1. 排除目录检查 (检查路径中每个部分)
    parts = Path(rel_path).parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return False

    # 2. 排除文件名模式
    filename = filepath.name
    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            if any(fnmatch.fnmatch(filename, p) for p in extra_includes):
                break
            return False

    # 3. 额外排除
    for pattern in extra_excludes:
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(rel_path, pattern):
            return False

    # 4. gitignore
    if use_gitignore and gitignore_patterns:
        if gitignore_match(rel_path, gitignore_patterns):
            if not any(fnmatch.fnmatch(filename, p) for p in extra_includes):
                return False

    # 5. 文件大小
    try:
        size_kb = filepath.stat().st_size / 1024
        if size_kb > max_size_kb:
            return False
    except OSError:
        return False

    # 6. 二进制检测
    if is_binary(filepath):
        return False

    # 7. 扩展名检查 (或额外包含)
    ext = filepath.suffix.lower()
    name = filepath.name
    if ext in INCLUDE_EXTENSIONS or name in INCLUDE_EXTENSIONS:
        return True

    # 无扩展名但可能是代码文件 (Makefile, Dockerfile, etc.)
    if name in {"Makefile", "Dockerfile", "Vagrantfile", "Rakefile", "Gemfile"}:
        return True

    # 额外包含模式
    for pattern in extra_includes:
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(rel_path, pattern):
            return True

    # .gitignore 等点文件有时也有价值
    if name.startswith(".") and ext in {".yml", ".yaml", ".json", ".toml", ".env"}:
        return True

    return False


def collect_files(
    repo_root: Path,
    use_gitignore: bool,
    extra_includes: list[str],
    extra_excludes: list[str],
    max_size_kb: int,
) -> list[tuple[str, Path]]:
    """收集所有应包含的文件，返回 [(相对路径, 绝对路径), ...]"""
    gitignore_patterns = parse_gitignore(repo_root) if use_gitignore else []
    collected = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # 就地修改 dirnames 实现剪枝，避免进入排除目录
        dirnames[:] = [
            d for d in sorted(dirnames)
            if d not in EXCLUDE_DIRS
            and not d.startswith(".")  # 排除所有隐藏目录
        ]

        for filename in sorted(filenames):
            filepath = Path(dirpath) / filename
            rel_path = filepath.relative_to(repo_root).as_posix()

            if should_include_file(
                filepath, rel_path, repo_root,
                gitignore_patterns, use_gitignore,
                extra_includes, extra_excludes,
                max_size_kb,
            ):
                collected.append((rel_path, filepath))

    collected.sort(key=lambda x: x[0])
    return collected


def get_file_tree(root: Path) -> str:
    """生成目录树字符串"""
    tree_lines = []
    root = Path(root)

    def _walk(directory, prefix=""):
        entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        # 过滤
        entries = [e for e in entries if not (
            e.is_dir() and (e.name in EXCLUDE_DIRS or e.name.startswith("."))
        )]

        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            if entry.is_dir():
                tree_lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if is_last else "│   "
                _walk(entry, prefix + extension)
            else:
                # 跳过排除模式的文件
                if any(fnmatch.fnmatch(entry.name, p) for p in EXCLUDE_PATTERNS):
                    continue
                size = entry.stat().st_size
                size_str = f"{size:,}B" if size < 1024 else f"{size/1024:.1f}KB"
                tree_lines.append(f"{prefix}{connector}{entry.name}  ({size_str})")

    _walk(root)
    return "\n".join(tree_lines)


def read_file_safe(filepath: Path) -> str:
    """安全读取文件内容，大文件截断"""
    size = os.path.getsize(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            if size > MAX_FILE_SIZE:
                content = f.read(MAX_FILE_SIZE)
                return content + f"\n\n... [文件过大，已截断。总大小: {size/1024:.1f}KB] ..."
            return f.read()
    except UnicodeDecodeError:
        return f"[二进制文件，跳过。大小: {size/1024:.1f}KB]"


def find_trader_repo() -> Path | None:
    """自动查找本地 Trader 仓库（需存在 main.py）"""
    script_dir = Path(__file__).resolve().parent
    for base in (script_dir, script_dir.parent, Path.cwd().resolve()):
        cand = base / "Trader"
        if cand.is_dir() and (cand / "main.py").is_file():
            return cand.resolve()
    return None


def clone_repo(repo_url: str, target_dir: str):
    """克隆仓库"""
    print(f"📥 正在克隆 {repo_url} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, target_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ 克隆失败: {result.stderr}")
        sys.exit(1)
    print("✅ 克隆完成")


def get_git_info(repo_dir: str) -> tuple[str, str]:
    """获取当前 commit 信息"""
    try:
        commit = subprocess.run(
            ["git", "-C", repo_dir, "log", "-1", "--format=%H %s (%ci)"],
            capture_output=True, text=True
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", repo_dir, "branch", "--show-current"],
            capture_output=True, text=True
        ).stdout.strip()
        return branch, commit
    except Exception:
        return "unknown", "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="将 Trader 代码仓库整合成一个 txt 文件，方便上传给网页端 Claude"
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        default=None,
        help="仓库路径或 URL (默认: 自动查找 Trader/ 目录)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出文件路径 (默认: Trader_snapshot.txt)",
    )
    parser.add_argument(
        "--include",
        default="",
        help="额外包含的 glob 模式，逗号分隔 (如 '*.md,*.txt')",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="额外排除的 glob 模式，逗号分隔 (如 '*.csv,*.xlsx')",
    )
    parser.add_argument(
        "--no-gitignore",
        action="store_true",
        help="不读取 .gitignore 规则",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=100,
        help="单文件最大 KB 数 (默认: 100)",
    )
    args = parser.parse_args()

    extra_includes = [p.strip() for p in args.include.split(",") if p.strip()]
    extra_excludes = [p.strip() for p in args.exclude.split(",") if p.strip()]

    # ── 解析仓库来源 ──
    repo_name = "Trader"
    repo_dir = None
    header_source = None
    need_clone = False
    tmp_dir = None

    if args.repo_path:
        arg = args.repo_path.strip()
        is_url = arg.startswith(("http://", "https://", "git@"))
        if is_url:
            header_source = arg
            need_clone = True
            repo_name = arg.rstrip("/").split("/")[-1].replace(".git", "")
        else:
            p = Path(arg).expanduser().resolve()
            if not p.is_dir():
                print(f"❌ 路径不存在或不是目录: {arg}")
                sys.exit(1)
            repo_dir = str(p)
            header_source = str(p)
            repo_name = p.name
    else:
        # 无参数：自动查找本地 Trader
        found = find_trader_repo()
        if found is None:
            print("❌ 未找到本地 Trader（在脚本目录、上一级、当前目录下查找 Trader/main.py）。")
            print(f"   指定路径: python repo_to_file.py /path/to/Trader")
            print(f"   或克隆远端: python repo_to_file.py {REMOTE_REPO_EXAMPLE}")
            sys.exit(1)
        repo_dir = str(found)
        header_source = str(found)
        print(f"📂 自动找到 Trader 仓库: {found}")

    if need_clone:
        tmp_dir = tempfile.mkdtemp()
        repo_dir = os.path.join(tmp_dir, "repo")
        clone_repo(header_source, repo_dir)

    try:
        branch, commit = get_git_info(repo_dir)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 收集文件
        print("📂 正在扫描文件...")
        files = collect_files(
            Path(repo_dir),
            use_gitignore=not args.no_gitignore,
            extra_includes=extra_includes,
            extra_excludes=extra_excludes,
            max_size_kb=args.max_size,
        )

        if not files:
            print("No files collected. Check your filters.", file=sys.stderr)
            sys.exit(1)

        tree = get_file_tree(Path(repo_dir))

        # 构建输出
        lines = []
        lines.append(f"{'=' * 70}")
        lines.append(f"REPOSITORY SNAPSHOT: {repo_name}")
        lines.append(f"Source: {header_source}")
        lines.append(f"Branch: {branch}")
        lines.append(f"Commit: {commit}")
        lines.append(f"Generated: {now}")
        lines.append(f"Files: {len(files)}")
        lines.append(f"{'=' * 70}")
        lines.append("")

        lines.append(f"{'=' * 70}")
        lines.append("FILE TREE")
        lines.append(f"{'=' * 70}")
        lines.append(tree)
        lines.append("")

        for rel_path, filepath in files:
            lines.append(f"{'=' * 70}")
            lines.append(f"FILE: {rel_path}")
            lines.append(f"{'=' * 70}")
            content = read_file_safe(filepath)
            lines.append(content)
            lines.append("")

        lines.append(f"{'=' * 70}")
        lines.append("END OF SNAPSHOT")
        lines.append(f"{'=' * 70}")

        merged = "\n".join(lines)

        # 写入文件
        output_name = args.output or f"{repo_name}_snapshot.txt"
        with open(output_name, "w", encoding="utf-8") as f:
            f.write(merged)

        size_kb = os.path.getsize(output_name) / 1024
        print(f"")
        print(f"✅ 完成！合并了 {len(files)} 个文件")
        print(f"📄 输出: {output_name} ({size_kb:.1f} KB)")
        print(f"")
        print(f"👉 把 {output_name} 上传给 Claude 即可开始工作")
        print(f"👉 本地修改后重新运行此脚本，上传新快照继续迭代")

    finally:
        if need_clone and tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

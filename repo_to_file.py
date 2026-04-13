"""
从仓库生成单文件快照，方便上传给 Claude 交互。

用法:
    python repo_to_file.py
        默认扫描「本脚本目录 / 上一级 / 当前工作目录」下的 Trader/（需存在 main.py），
        直接使用本地文件，不克隆、不拉取。

    python repo_to_file.py C:\\path\\to\\Trader
        显式指定本地仓库路径（同上，只读磁盘）。

    python repo_to_file.py https://github.com/Chevis-hah/Trader
        仅当参数是 http(s)/git@ URL 时才会 git clone（浅克隆）。

每次本地改完后重新跑一遍，上传新的 txt 即可。
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# ── 配置 ──────────────────────────────────────────────
# 无参数且未找到本地 Trader 时，提示里展示的远端示例（不会自动克隆）
REMOTE_REPO_EXAMPLE = "https://github.com/Chevis-hah/Trader"

SKIP_DIRS = {
    '.git', '__pycache__', '.venv', 'venv', 'env',
    'node_modules', '.idea', '.vscode', '.mypy_cache',
    '.pytest_cache', '.tox', 'dist', 'build', '*.egg-info',
}

SKIP_FILES = {'.DS_Store', 'Thumbs.db'}

BINARY_EXTENSIONS = {
    '.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.o', '.a',
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.webp', '.bmp',
    '.db', '.sqlite', '.sqlite3', '.pkl', '.joblib', '.h5', '.hdf5',
    '.zip', '.tar', '.gz', '.bz2', '.rar', '.7z', '.xz',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv',
    '.woff', '.woff2', '.ttf', '.eot',
}

# 超过此大小的文件只显示前后部分（字节）
MAX_FILE_SIZE = 100_000  # 100KB


def should_skip_dir(dirname):
    if dirname in SKIP_DIRS:
        return True
    if dirname.endswith('.egg-info'):
        return True
    return False


def should_skip_file(filename):
    if filename in SKIP_FILES:
        return True
    _, ext = os.path.splitext(filename)
    if ext.lower() in BINARY_EXTENSIONS:
        return True
    return False


def get_file_tree(root):
    """生成目录树字符串"""
    tree_lines = []
    root = Path(root)

    def _walk(directory, prefix=""):
        entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        # 过滤
        entries = [e for e in entries if not (
            e.is_dir() and should_skip_dir(e.name)
        ) and not (
            e.is_file() and should_skip_file(e.name)
        )]

        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            if entry.is_dir():
                tree_lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if is_last else "│   "
                _walk(entry, prefix + extension)
            else:
                size = entry.stat().st_size
                size_str = f"{size:,}B" if size < 1024 else f"{size/1024:.1f}KB"
                tree_lines.append(f"{prefix}{connector}{entry.name}  ({size_str})")

    _walk(root)
    return "\n".join(tree_lines)


def read_file_safe(filepath):
    """安全读取文件内容"""
    size = os.path.getsize(filepath)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if size > MAX_FILE_SIZE:
                content = f.read(MAX_FILE_SIZE)
                return content + f"\n\n... [文件过大，已截断。总大小: {size/1024:.1f}KB] ..."
            return f.read()
    except UnicodeDecodeError:
        return f"[二进制文件，跳过。大小: {size/1024:.1f}KB]"


def collect_files(root):
    """收集所有文件，返回 (相对路径, 内容) 列表"""
    root = Path(root)
    results = []

    for dirpath, dirnames, filenames in os.walk(root):
        # 原地修改 dirnames 来跳过目录
        dirnames[:] = [d for d in sorted(dirnames) if not should_skip_dir(d)]

        for filename in sorted(filenames):
            if should_skip_file(filename):
                continue
            filepath = Path(dirpath) / filename
            rel_path = filepath.relative_to(root)
            content = read_file_safe(filepath)
            results.append((str(rel_path), content))

    return results


def resolve_repo_source():
    """
    解析仓库来源。
    返回 (source_for_snapshot_header, repo_dir, cleanup_tmp, repo_name, is_url_clone)。
    """
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        is_url = arg.startswith(("http://", "https://", "git@"))
        if is_url:
            return arg, None, True, arg.rstrip("/").split("/")[-1].replace(".git", ""), True
        p = Path(arg).expanduser().resolve()
        if not p.is_dir():
            print(f"❌ 路径不存在或不是目录: {arg}")
            sys.exit(1)
        name = p.name
        return str(p), str(p), False, name, False

    # 无参数：只找本地 Trader（不访问网络）
    script_dir = Path(__file__).resolve().parent
    for base in (script_dir, script_dir.parent, Path.cwd().resolve()):
        cand = base / "Trader"
        if cand.is_dir() and (cand / "main.py").is_file():
            rp = str(cand.resolve())
            return rp, rp, False, cand.name, False

    print("❌ 未找到本地 Trader（在脚本目录、上一级、当前目录下查找 Trader/main.py）。")
    print(f"   指定路径: python repo_to_file.py C:\\\\...\\\\Trader")
    print(f"   或克隆远端: python repo_to_file.py {REMOTE_REPO_EXAMPLE}")
    sys.exit(1)


def clone_repo(repo_url, target_dir):
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


def get_git_info(repo_dir):
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
    header_source, repo_dir, cleanup, repo_name, need_clone = resolve_repo_source()
    tmp_dir = None

    if need_clone:
        tmp_dir = tempfile.mkdtemp()
        repo_dir = os.path.join(tmp_dir, "repo")
        clone_repo(header_source, repo_dir)

    try:
        branch, commit = get_git_info(repo_dir)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 收集文件
        print("📂 正在扫描文件...")
        files = collect_files(repo_dir)
        tree = get_file_tree(repo_dir)

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

        for rel_path, content in files:
            lines.append(f"{'=' * 70}")
            lines.append(f"FILE: {rel_path}")
            lines.append(f"{'=' * 70}")
            lines.append(content)
            lines.append("")

        lines.append(f"{'=' * 70}")
        lines.append("END OF SNAPSHOT")
        lines.append(f"{'=' * 70}")

        merged = "\n".join(lines)

        # 写入文件
        output_name = f"{repo_name}_snapshot.txt"
        with open(output_name, 'w', encoding='utf-8') as f:
            f.write(merged)

        size_kb = os.path.getsize(output_name) / 1024
        print(f"")
        print(f"✅ 完成！合并了 {len(files)} 个文件")
        print(f"📄 输出: {output_name} ({size_kb:.1f} KB)")
        print(f"")
        print(f"👉 把 {output_name} 上传给 Claude 即可开始工作")
        print(f"👉 本地修改后重新运行此脚本，上传新快照继续迭代")

    finally:
        if cleanup and tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

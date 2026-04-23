#!/usr/bin/env python
"""macOS 磁盘空间审计 — 找占用大头 + 标记"通常可清理"的位置。

只读, 不删任何东西。输出按 (尺寸, 可清理度) 排序。

用法: python scripts/disk_audit.py
"""

from __future__ import annotations

import concurrent.futures
import subprocess
from dataclasses import dataclass
from pathlib import Path

HOME = Path.home()


@dataclass(frozen=True, slots=True)
class Probe:
    path: Path
    label: str
    cleanable: str  # "safe" | "review" | "keep"
    note: str = ""


# 已知大占用 / 可清理位置 (macOS)
PROBES: tuple[Probe, ...] = (
    # ===== 完全可删 (重启即重建) =====
    Probe(HOME / "Library/Caches", "用户应用缓存", "safe", "重启应用会重建"),
    Probe(HOME / ".cache", "Unix 应用缓存", "safe", ""),
    Probe(
        HOME / "Library/Developer/Xcode/DerivedData",
        "Xcode 派生数据",
        "safe",
        "下次 build 重建, 通常 5-50GB",
    ),
    Probe(HOME / "Library/Developer/CoreSimulator/Caches", "iOS 模拟器缓存", "safe", ""),
    Probe(HOME / "Library/Logs", "用户应用日志", "safe", ""),
    Probe(HOME / ".Trash", "垃圾桶", "safe", "清空即可"),
    Probe(Path("/private/var/log"), "系统日志", "safe", "需 sudo 才能清"),
    # ===== 看情况清 (有用但通常可重建) =====
    Probe(
        HOME / "Library/Developer/Xcode/Archives",
        "Xcode Archives",
        "review",
        "已发布 IPA 不再需要可删",
    ),
    Probe(
        HOME / "Library/Developer/CoreSimulator/Devices",
        "iOS 模拟器设备",
        "review",
        "里面是各 iOS 版本快照, 不用的可删",
    ),
    Probe(
        HOME / "Library/Application Support/MobileSync/Backup",
        "iPhone/iPad 备份",
        "review",
        "iCloud 也备份了的话可删",
    ),
    Probe(
        HOME / "Library/Containers/com.docker.docker",
        "Docker Desktop 容器卷",
        "review",
        "image/volume 大头, `docker system prune -a` 清",
    ),
    Probe(HOME / "Library/Group Containers/group.com.docker", "Docker 共享容器", "review", ""),
    Probe(HOME / "Library/Application Support/Slack", "Slack 缓存数据", "review", ""),
    Probe(HOME / "Library/Application Support/Code/Cache", "VS Code 缓存", "safe", ""),
    Probe(HOME / "Library/Application Support/Cursor/Cache", "Cursor 缓存", "safe", ""),
    Probe(HOME / "Library/Caches/Homebrew", "Homebrew 下载缓存", "safe", "`brew cleanup` 清"),
    Probe(Path("/Library/Caches"), "系统级应用缓存", "safe", "需 sudo"),
    Probe(HOME / "Library/Developer/CoreSimulator/Volumes", "模拟器 Volumes", "review", ""),
    # ===== 项目类大头 (要看哪些活跃) =====
    Probe(HOME / "node_modules", "全局 node_modules", "review", ""),
    Probe(HOME / ".npm", "npm 缓存", "safe", "`npm cache clean --force`"),
    Probe(HOME / ".pnpm-store", "pnpm 全局存储", "review", "活跃项目要的"),
    Probe(HOME / ".yarn", "yarn 缓存", "safe", ""),
    Probe(HOME / ".gradle/caches", "Gradle 缓存", "safe", "需 sudo"),
    Probe(HOME / ".m2", "Maven 本地仓库", "review", ""),
    Probe(HOME / "go/pkg", "Go 模块缓存", "safe", "`go clean -modcache`"),
    Probe(HOME / ".cargo", "Cargo 缓存", "review", ""),
    Probe(HOME / ".rustup", "Rust 工具链", "review", ""),
    Probe(HOME / ".pyenv", "pyenv 多版本", "review", ""),
    Probe(HOME / ".conda", "Conda 安装", "review", ""),
    Probe(HOME / "miniconda3", "Miniconda", "review", ""),
    Probe(HOME / "Library/Android/sdk", "Android SDK", "review", ""),
    # ===== 媒体 / 文档 (看你有没有) =====
    Probe(HOME / "Pictures/Photos Library.photoslibrary", "照片图库", "keep", ""),
    Probe(HOME / "Movies", "影片", "keep", ""),
    Probe(HOME / "Downloads", "下载", "review", "看一眼老文件可删"),
    Probe(HOME / "Desktop", "桌面", "review", ""),
    Probe(HOME / "Documents", "文档", "keep", ""),
    # ===== 系统 (看一眼别动) =====
    Probe(Path("/private/var/vm"), "系统休眠/swap 镜像", "keep", "macOS 自动管理"),
    Probe(Path("/Applications"), "应用程序", "keep", ""),
)


def du_bytes(path: Path) -> int | None:
    """`du -sk` 拿目录大小 (kilobytes), 不存在/不可读返 None。"""
    if not path.exists():
        return None
    try:
        out = subprocess.run(
            ["du", "-sk", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if out.returncode != 0:
            return None
        kb = int(out.stdout.split()[0])
        return kb * 1024
    except Exception:
        return None


def humanize(n: int) -> str:
    size: float = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024:
            return f"{size:6.1f}{unit}"
        size /= 1024
    return f"{size:6.1f}P"


def disk_usage() -> None:
    """分区使用 (df -h)。"""
    out = subprocess.run(["df", "-h"], capture_output=True, text=True, check=False)
    print("=== 分区占用 ===")
    for line in out.stdout.splitlines():
        if "/System/Volumes/Data" in line or "Filesystem" in line or "/dev/disk" in line:
            print(" ", line)
    print()


def dir_top(root: Path, n: int = 15, include_hidden: bool = True) -> list[tuple[Path, int]]:
    """root 下直接子目录的 top N (du -s, 不递归)。"""
    if not root.exists():
        return []
    results = []
    try:
        items = [
            p
            for p in root.iterdir()
            if p.is_dir() and (include_hidden or not p.name.startswith("."))
        ]
    except PermissionError:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        sizes = list(pool.map(du_bytes, items))
    for p, sz in zip(items, sizes, strict=False):
        if sz:
            results.append((p, sz))
    results.sort(key=lambda t: t[1], reverse=True)
    return results[:n]


def main() -> None:
    print("\n📊 macOS 磁盘审计  (只扫已知热点目录, 不动任何文件)\n")

    disk_usage()

    print("=== 家目录顶层 TOP-15 (含隐藏目录) ===")
    home_items = dir_top(HOME, n=15, include_hidden=True)
    for p, sz in home_items:
        print(f"  {humanize(sz)}  {p}")
    home_total = sum(sz for _, sz in home_items)
    print(f"  ── 顶层合计 (TOP-15): {humanize(home_total)}\n")

    # ~/Library 是大头, 深扫一层
    print("=== ~/Library 内部 TOP-15 ===")
    lib_items = dir_top(HOME / "Library", n=15)
    for p, sz in lib_items:
        print(f"  {humanize(sz)}  {p}")
    print()

    # ~/Library/Application Support 又是大头, 再深一层
    print("=== ~/Library/Application Support TOP-15 ===")
    appsup_items = dir_top(HOME / "Library/Application Support", n=15)
    for p, sz in appsup_items:
        print(f"  {humanize(sz)}  {p}")
    print()

    # ~/Library/Containers (沙盒应用数据) 也常常大
    print("=== ~/Library/Containers TOP-10 ===")
    containers = dir_top(HOME / "Library/Containers", n=10)
    for p, sz in containers:
        print(f"  {humanize(sz)}  {p}")
    print()

    # 系统级位置
    print("=== 系统盘位置 ===")
    for sys_root in (
        Path("/usr/local"),
        Path("/opt/homebrew"),
        Path("/Library"),
        Path("/Users/Shared"),
        Path("/private/var"),
    ):
        if not sys_root.exists():
            continue
        sz = du_bytes(sys_root)
        if sz:
            print(f"  {humanize(sz)}  {sys_root}")
    print()

    # 已知热点并行 du
    print("=== 已知热点 (按是否可清理分组) ===\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(lambda p: du_bytes(p.path), PROBES))

    by_cat: dict[str, list[tuple[Probe, int]]] = {"safe": [], "review": [], "keep": []}
    for probe, sz in zip(PROBES, sizes, strict=False):
        if sz is None or sz == 0:
            continue
        by_cat[probe.cleanable].append((probe, sz))

    cat_label = {
        "safe": "🟢 SAFE — 可放心清(自动重建)",
        "review": "🟡 REVIEW — 看情况清",
        "keep": "⚪ KEEP — 实际数据, 别动",
    }
    for cat in ("safe", "review", "keep"):
        items = by_cat[cat]
        if not items:
            continue
        items.sort(key=lambda t: t[1], reverse=True)
        total = sum(sz for _, sz in items)
        print(f"--- {cat_label[cat]}  (合计 {humanize(total)}) ---")
        for probe, sz in items:
            note = f"  — {probe.note}" if probe.note else ""
            print(f"  {humanize(sz)}  {probe.label:<22}  {probe.path}{note}")
        print()

    # 总结建议
    safe_total = sum(sz for _, sz in by_cat["safe"])
    review_total = sum(sz for _, sz in by_cat["review"])
    print(f"💡 安全可清: {humanize(safe_total)}    需审视: {humanize(review_total)}")
    print()
    print("常用清理:")
    print("  brew cleanup -s              # Homebrew 老版本/缓存")
    print("  rm -rf ~/Library/Developer/Xcode/DerivedData/*")
    print("  docker system prune -a       # 如有 Docker")
    print("  npm cache clean --force")
    print("  go clean -modcache")


if __name__ == "__main__":
    main()

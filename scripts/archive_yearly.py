#!/usr/bin/env python3
"""年度数据归档脚本

将完整自然年的日线数据导出为独立的 SQLite 文件，用于 GitHub Releases 发布。

使用:
    # 归档 2024 年数据
    python scripts/archive_yearly.py 2024

    # 归档并上传到 GitHub Release
    python scripts/archive_yearly.py 2024 --upload
"""

import argparse
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "stock_daily.db"
OUTPUT_DIR = PROJECT_ROOT / "data" / "archives"


def archive_year(year: int, source_db: Path, output_dir: Path) -> Path:
    """导出指定年份的数据"""

    # 验证是完整自然年
    today = date.today()
    if year >= today.year:
        print(f"错误: {year} 年尚未结束，只能归档完整自然年")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"stock_daily_{year}.db"

    # 删除旧文件
    if output_file.exists():
        output_file.unlink()

    print(f"正在归档 {year} 年数据...")
    print(f"源数据库: {source_db}")
    print(f"目标文件: {output_file}")

    # 连接源数据库
    src_conn = sqlite3.connect(source_db)
    src_cursor = src_conn.cursor()

    # 创建目标数据库
    dst_conn = sqlite3.connect(output_file)
    dst_cursor = dst_conn.cursor()

    # 创建表结构
    dst_cursor.execute("""
        CREATE TABLE stock_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT,
            trade_date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            pre_close REAL,
            change REAL,
            pct_change REAL,
            adj_factor REAL DEFAULT 1.0,
            UNIQUE(code, trade_date)
        )
    """)

    # 创建索引
    dst_cursor.execute("CREATE INDEX idx_code_date ON stock_daily(code, trade_date)")
    dst_cursor.execute("CREATE INDEX idx_date ON stock_daily(trade_date)")

    # 复制数据
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    src_cursor.execute("""
        SELECT code, trade_date, open, high, low, close, volume, amount,
               pre_close, change, pct_change, adj_factor
        FROM stock_daily
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY code, trade_date
    """, (start_date, end_date))

    rows = src_cursor.fetchall()

    if not rows:
        print(f"警告: {year} 年没有数据")
        dst_conn.close()
        src_conn.close()
        output_file.unlink()
        sys.exit(1)

    dst_cursor.executemany("""
        INSERT INTO stock_daily
        (code, trade_date, open, high, low, close, volume, amount,
         pre_close, change, pct_change, adj_factor)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    dst_conn.commit()

    # 统计
    dst_cursor.execute("SELECT COUNT(DISTINCT code) FROM stock_daily")
    stock_count = dst_cursor.fetchone()[0]

    dst_cursor.execute("SELECT COUNT(DISTINCT trade_date) FROM stock_daily")
    day_count = dst_cursor.fetchone()[0]

    # 压缩数据库
    dst_cursor.execute("VACUUM")

    dst_conn.close()
    src_conn.close()

    # 文件大小
    size_mb = output_file.stat().st_size / 1024 / 1024

    print(f"\n归档完成:")
    print(f"  股票数: {stock_count}")
    print(f"  交易日: {day_count}")
    print(f"  总行数: {len(rows):,}")
    print(f"  文件大小: {size_mb:.1f} MB")

    return output_file


def upload_to_github(file_path: Path, year: int):
    """上传到 GitHub Release"""

    tag = f"data-{year}"
    title = f"{year}年日线数据"

    print(f"\n上传到 GitHub Release: {tag}")

    # 检查 gh 命令
    try:
        subprocess.run(["gh", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("错误: 未安装 gh 命令行工具")
        print("安装: https://cli.github.com/")
        sys.exit(1)

    # 创建 Release 并上传
    cmd = [
        "gh", "release", "create", tag,
        str(file_path),
        "--title", title,
        "--notes", f"{year}年A股日线数据 (完整自然年)"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"上传成功: {result.stdout.strip()}")
    else:
        # Release 可能已存在，尝试上传到现有 Release
        cmd = ["gh", "release", "upload", tag, str(file_path), "--clobber"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("上传成功 (更新已有 Release)")
        else:
            print(f"上传失败: {result.stderr}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="年度数据归档")
    parser.add_argument("year", type=int, help="要归档的年份 (完整自然年)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="源数据库路径")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--upload", action="store_true", help="上传到 GitHub Release")

    args = parser.parse_args()

    if not args.db.exists():
        print(f"错误: 数据库不存在: {args.db}")
        sys.exit(1)

    output_file = archive_year(args.year, args.db, args.output)

    if args.upload:
        upload_to_github(output_file, args.year)


if __name__ == "__main__":
    main()

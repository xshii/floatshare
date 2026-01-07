#!/usr/bin/env python
"""数据获取脚本"""

import sys
from pathlib import Path
from datetime import date, timedelta
import argparse

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.storage.database import DatabaseStorage
from src.monitor.logger import get_logger

logger = get_logger("fetch_data")


def fetch_stock_list(loader: DataLoader, storage: DatabaseStorage) -> int:
    """获取股票列表"""
    logger.info("正在获取股票列表...")
    df = loader.get_stock_list()

    if df.empty:
        logger.warning("获取股票列表为空")
        return 0

    count = storage.save_stock_list(df)
    logger.info(f"保存了 {count} 只股票信息")
    return count


def fetch_daily_data(
    loader: DataLoader,
    storage: DatabaseStorage,
    codes: list,
    start_date: date,
    end_date: date,
) -> int:
    """获取日线数据"""
    total = 0

    for i, code in enumerate(codes):
        logger.info(f"[{i+1}/{len(codes)}] 获取 {code} 数据...")

        try:
            # 检查已有数据
            latest_date = storage.get_latest_date(code)
            if latest_date and latest_date >= end_date:
                logger.info(f"  {code} 数据已是最新")
                continue

            actual_start = latest_date + timedelta(days=1) if latest_date else start_date

            df = loader.get_daily(code, actual_start, end_date)
            if df.empty:
                continue

            count = storage.save_daily(df)
            total += count
            logger.info(f"  保存了 {count} 条记录")

        except Exception as e:
            logger.error(f"  获取 {code} 失败: {e}")

    return total


def main():
    parser = argparse.ArgumentParser(description="数据获取工具")
    parser.add_argument(
        "--source", type=str, default="akshare",
        choices=["tushare", "akshare"],
        help="数据源"
    )
    parser.add_argument(
        "--start", type=str, default="2020-01-01",
        help="开始日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="结束日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--codes", type=str, nargs="*",
        help="股票代码列表"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="获取所有股票数据"
    )

    args = parser.parse_args()

    # 初始化
    loader = DataLoader(source=args.source)
    storage = DatabaseStorage()
    storage.init_tables()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end) if args.end else date.today()

    # 获取股票列表
    fetch_stock_list(loader, storage)

    # 获取日线数据
    if args.codes:
        codes = args.codes
    elif args.all:
        stock_list = storage.load_stock_list()
        codes = stock_list["code"].tolist()[:100]  # 限制数量
    else:
        # 默认获取几只示例股票
        codes = ["000001.SZ", "600000.SH", "000002.SZ"]

    logger.info(f"准备获取 {len(codes)} 只股票的数据")
    total = fetch_daily_data(loader, storage, codes, start_date, end_date)

    logger.info(f"数据获取完成，共保存 {total} 条记录")


if __name__ == "__main__":
    main()

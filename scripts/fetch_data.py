#!/usr/bin/env python
"""数据增量拉取脚本 — 默认 AKShare。"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from floatshare.application import create_default_loader
from floatshare.infrastructure.storage import DatabaseStorage
from floatshare.interfaces.data_source import DataSourceError
from floatshare.observability import logger


def fetch_stock_list(loader, storage: DatabaseStorage) -> int:
    logger.info("拉取股票列表")
    df = loader.get_stock_list()
    if df.empty:
        logger.warning("股票列表为空")
        return 0
    return storage.save_stock_list(df)


def fetch_daily_data(loader, storage, codes, start: date, end: date) -> int:
    total = 0
    for i, code in enumerate(codes, 1):
        logger.info(f"[{i}/{len(codes)}] {code}")
        try:
            latest = storage.get_latest_date(code)
            if latest and latest >= end:
                continue
            actual_start = latest + timedelta(days=1) if latest else start
            df = loader.get_daily(code, actual_start, end)
            if df.empty:
                continue
            total += storage.save_daily(df)
        except DataSourceError as e:
            logger.error(f"  {code} 失败: {e}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="数据获取工具")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--codes", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    loader = create_default_loader()
    storage = DatabaseStorage()
    storage.init_tables()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    fetch_stock_list(loader, storage)

    if args.codes:
        codes = args.codes
    elif args.all:
        codes = storage.load_stock_list()["code"].tolist()[:100]
    else:
        codes = ["000001.SZ", "600000.SH", "000002.SZ"]

    logger.info(f"准备拉取 {len(codes)} 只")
    total = fetch_daily_data(loader, storage, codes, start, end)
    logger.info(f"完成，共 {total} 条")


if __name__ == "__main__":
    main()

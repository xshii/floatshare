#!/usr/bin/env python
"""运行回测脚本"""

import sys
from pathlib import Path
from datetime import date
import argparse

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.backtest.engine import BacktestEngine
from src.strategy.registry import StrategyRegistry
from src.monitor.logger import get_logger

# 导入策略以完成注册
import strategies  # noqa

logger = get_logger("backtest")


def main():
    parser = argparse.ArgumentParser(description="回测工具")
    parser.add_argument(
        "--strategy", type=str, required=True,
        help="策略名称"
    )
    parser.add_argument(
        "--codes", type=str, nargs="+", required=True,
        help="股票代码列表"
    )
    parser.add_argument(
        "--start", type=str, default="2023-01-01",
        help="开始日期"
    )
    parser.add_argument(
        "--end", type=str, default="2024-12-01",
        help="结束日期"
    )
    parser.add_argument(
        "--capital", type=float, default=1000000,
        help="初始资金"
    )
    parser.add_argument(
        "--source", type=str, default="akshare",
        help="数据源"
    )
    parser.add_argument(
        "--params", type=str, nargs="*",
        help="策略参数 (key=value格式)"
    )

    args = parser.parse_args()

    # 解析策略参数
    strategy_params = {}
    if args.params:
        for param in args.params:
            key, value = param.split("=")
            # 尝试转换为数字
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            strategy_params[key] = value

    # 列出可用策略
    available = StrategyRegistry.list_strategies()
    logger.info(f"可用策略: {available}")

    # 创建策略
    try:
        strategy = StrategyRegistry.create(args.strategy, strategy_params)
        logger.info(f"使用策略: {strategy.name}")
    except ValueError as e:
        logger.error(f"策略错误: {e}")
        return

    # 加载数据
    logger.info("正在加载数据...")
    loader = DataLoader(source=args.source)

    import pandas as pd
    all_data = []

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    for code in args.codes:
        logger.info(f"  加载 {code}...")
        df = loader.get_daily(code, start_date, end_date)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        logger.error("没有获取到数据")
        return

    data = pd.concat(all_data, ignore_index=True)
    logger.info(f"共加载 {len(data)} 条数据")

    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=args.capital,
        commission=0.0003,
        slippage=0.001,
    )

    # 运行回测
    logger.info("开始回测...")
    report = engine.run(
        strategy=strategy,
        data=data,
        start_date=start_date,
        end_date=end_date,
    )

    # 输出结果
    report.print_summary()


if __name__ == "__main__":
    main()

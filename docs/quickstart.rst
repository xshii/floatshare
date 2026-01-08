快速开始
========

安装
----

从源码安装
~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/xshii/floatshare.git
   cd floatshare
   pip install -r requirements.txt

依赖说明
~~~~~~~~

核心依赖：

- pandas >= 2.0.0
- numpy >= 1.24.0
- sqlalchemy >= 2.0.0
- pydantic >= 2.0.0

可选依赖：

- aiohttp: 异步数据获取
- structlog: 结构化日志
- apscheduler: 任务调度
- opentelemetry: 分布式追踪

配置
----

环境变量
~~~~~~~~

.. code-block:: bash

   # 数据库配置
   export FLOATSHARE_DATABASE__PATH=data/floatshare.db

   # 数据源配置
   export FLOATSHARE_DATA_SOURCE__PRIMARY=akshare

   # 日志配置
   export FLOATSHARE_LOG_LEVEL=INFO

配置文件
~~~~~~~~

创建 ``.env`` 文件：

.. code-block:: ini

   # 数据库
   FLOATSHARE_DATABASE__PATH=data/floatshare.db

   # 交易参数
   FLOATSHARE_TRADING__COMMISSION_RATE=0.0003
   FLOATSHARE_TRADING__STAMP_TAX=0.001

基本使用
--------

数据同步
~~~~~~~~

.. code-block:: python

   from src.data.loader import DataLoader
   from src.data.storage.database import DatabaseStorage

   # 初始化
   loader = DataLoader(source="akshare")
   storage = DatabaseStorage()
   storage.init_tables()

   # 获取股票列表
   stocks = loader.get_stock_list()

   # 同步数据
   for code in stocks["code"][:10]:
       df = loader.get_daily(code, "2024-01-01", "2024-12-31")
       if not df.empty:
           storage.save_daily(df)

策略回测
~~~~~~~~

.. code-block:: python

   from src.strategy.base import Strategy
   from src.backtest.engine import BacktestEngine

   # 定义策略
   class SimpleMA(Strategy):
       def on_bar(self, bar):
           # 简单均线策略
           if self.ma5 > self.ma20:
               self.buy(bar.close, 100)
           elif self.ma5 < self.ma20:
               self.sell(bar.close, 100)

   # 回测
   engine = BacktestEngine(
       initial_capital=100000,
       commission_rate=0.0003,
   )
   result = engine.run(SimpleMA(), "000001.SZ", "2024-01-01", "2024-12-31")
   print(result.summary())

CLI 工具
~~~~~~~~

.. code-block:: bash

   # 数据同步
   python -m src.cli.app data sync --source akshare --codes 000001.SZ

   # 查看数据状态
   python -m src.cli.app data status

   # 运行回测
   python -m src.cli.app backtest run --strategy simple_ma

进阶使用
--------

异步数据获取
~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from src.utils.async_loader import AsyncDataLoader

   async def main():
       loader = AsyncDataLoader()
       data = await loader.fetch_batch(
           codes=["000001.SZ", "000002.SZ"],
           start_date="2024-01-01",
           end_date="2024-12-31",
       )
       print(data)

   asyncio.run(main())

任务调度
~~~~~~~~

.. code-block:: python

   from src.utils.scheduler import TaskScheduler

   def sync_job():
       print("同步数据...")

   scheduler = TaskScheduler()
   scheduler.add_daily_job("daily_sync", sync_job, hour=18, minute=0)
   scheduler.start()

结构化日志
~~~~~~~~~~

.. code-block:: python

   from src.utils.logging import configure_structlog, get_structlog

   # 配置
   configure_structlog(level="INFO", json_format=True)

   # 使用
   logger = get_structlog("myapp")
   logger.info("sync_started", source="akshare", codes_count=100)

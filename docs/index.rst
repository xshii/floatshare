FloatShare 文档
================

FloatShare 是一个 A 股量化交易框架，提供数据获取、策略回测、交易执行等功能。

.. toctree::
   :maxdepth: 2
   :caption: 目录

   quickstart
   api/index
   changelog

快速开始
--------

安装
~~~~

.. code-block:: bash

   pip install -r requirements.txt

基本使用
~~~~~~~~

.. code-block:: python

   from src.data.loader import DataLoader
   from src.data.storage.database import DatabaseStorage

   # 加载数据
   loader = DataLoader(source="akshare")
   df = loader.get_daily("000001.SZ", "2024-01-01", "2024-12-31")

   # 存储数据
   storage = DatabaseStorage()
   storage.init_tables()
   storage.save_daily(df)

功能特性
--------

数据模块
~~~~~~~~

- 多数据源支持 (AKShare, BaoStock)
- 数据校验和清洗
- 数据库持久化
- LRU 缓存

策略模块
~~~~~~~~

- 策略基类
- 回测引擎
- 性能指标计算

工具模块
~~~~~~~~

- 依赖注入容器
- 事件总线
- 任务调度
- 结构化日志
- 分布式追踪

索引
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

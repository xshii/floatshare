更新日志
========

v1.1.0 (开发中)
---------------

新增
~~~~

- 异步数据加载器 (``AsyncDataLoader``)
- 重试机制 (``tenacity`` 集成)
- 结构化日志 (``structlog`` 集成)
- 任务调度器 (``APScheduler`` 集成)
- 高效序列化 (``msgpack`` 支持)
- 分布式追踪 (``OpenTelemetry`` 集成)
- Sphinx 文档

改进
~~~~

- 依赖注入容器支持可重入锁
- 数据库 schema 添加 adj_factor 列
- CI 拆分为并行任务

v1.0.0
------

初始版本

- 数据获取 (AKShare, BaoStock)
- 数据校验
- 数据管道
- LRU 缓存
- 依赖注入
- 事件总线
- CLI 工具
- 插件系统
- Prometheus 指标

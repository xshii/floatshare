"""用户策略集合。

本目录下的所有 .py 模块会被 `floatshare.strategy.discover()` 通过
`pkgutil.iter_modules` 自动加载，触发其 `@register` 装饰器。
新增策略只需在此目录放一个文件，无需修改本 `__init__.py`。
"""

"""FloatShare CLI 模块

提供统一的命令行接口：
- 数据同步命令
- 回测命令
- 系统管理命令
"""

from src.cli.app import app, main

__all__ = ["app", "main"]

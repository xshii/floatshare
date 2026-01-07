#!/usr/bin/env python
"""初始化数据库"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.storage.database import DatabaseStorage


def main():
    """初始化数据库表"""
    print("正在初始化数据库...")

    storage = DatabaseStorage()
    storage.init_tables()

    print("数据库初始化完成!")
    print(f"数据库路径: {storage.db_path}")


if __name__ == "__main__":
    main()

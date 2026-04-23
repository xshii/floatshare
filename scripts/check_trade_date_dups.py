#!/usr/bin/env python3
"""daily-sync 后置检测 — 扫 (code, trade_date) 重复行.

Usage:
    python scripts/check_trade_date_dups.py [db_path]

Exit code:
    0 = 干净
    1 = 发现重复 (详情打 stderr)

集成到 scripts/daily-sync.sh 末尾示例:
    python scripts/check_trade_date_dups.py data/floatshare.db || {
        echo "DB 重复行告警, 通知运维"; exit 1;
    }
"""

from __future__ import annotations

import sys
from pathlib import Path

from floatshare.application.db_integrity import (
    DbIntegrityError,
    check_trade_date_duplicates,
)


def main() -> int:
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/floatshare.db"
    if not Path(db_path).exists():
        print(f"✗ DB 不存在: {db_path}", file=sys.stderr)
        return 1

    dups = check_trade_date_duplicates(db_path)
    if not dups:
        print(f"✓ {db_path} 无 (code, trade_date) 重复行")
        return 0

    try:
        raise DbIntegrityError(dups)
    except DbIntegrityError as e:
        print(f"✗ {e}", file=sys.stderr)
        print("Top 20 重复:", file=sys.stderr)
        for d in dups[:20]:
            print(f"  [{d.table}] {d.code} {d.trade_date} × {d.count}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

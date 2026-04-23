"""universe.py — _top_mv / select_universe 单测.

核心回归: 北交所 (*.BJ) 必须从 universe 排除
  - tushare moneyflow 不覆盖 BJ → 7/39 特征全 NaN
  - 涨跌幅规则 ±30% 与 A 股 ±10% 不同, 抓涨停语义不适用
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from floatshare.ml.data.universe import _top_mv


def _make_minimal_db(db_path: Path) -> None:
    """造一个带 daily_basic + stock_lifecycle 的小 DB, 含 BJ / ST / 普通股混合."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE daily_basic (
            code TEXT, trade_date TEXT,
            circ_mv REAL, turnover_rate REAL,
            PRIMARY KEY (code, trade_date)
        );
        CREATE TABLE stock_lifecycle (
            code TEXT PRIMARY KEY,
            name TEXT,
            list_date TEXT,
            delist_date TEXT
        );
        """
    )
    # 插入 3 只普通沪深 + 2 只 BJ (高市值, 应被排除) + 1 只 ST
    codes_data = [
        # (code, name, list_date, circ_mv)
        ("600000.SH", "浦发银行", "2010-01-01", 9e10),
        ("000001.SZ", "平安银行", "2010-01-01", 8e10),
        ("300750.SZ", "宁德时代", "2018-06-01", 7e10),
        ("920001.BJ", "北交所示例A", "2015-01-01", 9.5e10),  # 高市值应被排除
        ("920002.BJ", "北交所示例B", "2015-01-01", 8.5e10),
        ("600999.SH", "*ST股", "2010-01-01", 6e10),
    ]
    conn.executemany(
        "INSERT INTO stock_lifecycle (code, name, list_date, delist_date) VALUES (?,?,?,NULL)",
        [(c, n, d) for c, n, d, _ in codes_data],
    )
    # 将 name 中带 ST 前缀的改成 ST 开头
    conn.execute("UPDATE stock_lifecycle SET name='*ST风险' WHERE code='600999.SH'")
    conn.executemany(
        "INSERT INTO daily_basic (code, trade_date, circ_mv, turnover_rate) VALUES (?,?,?,?)",
        [(c, "2026-04-20", mv, 1.5) for c, _, _, mv in codes_data],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def mini_db(tmp_path: Path) -> str:
    p = tmp_path / "mini.db"
    _make_minimal_db(p)
    return str(p)


def test_top_mv_excludes_bj(mini_db: str) -> None:
    """即使 BJ 股市值高, 也不应出现在 top_mv universe."""
    engine = create_engine(f"sqlite:///{mini_db}")
    with engine.connect() as conn:
        codes = _top_mv(conn, n=10, as_of_date="2026-04-21")
    bj = [c for c in codes if c.endswith(".BJ")]
    assert bj == [], f"BJ 股未被过滤: {bj}"


def test_top_mv_excludes_st(mini_db: str) -> None:
    """ST / *ST 股不应进 universe."""
    engine = create_engine(f"sqlite:///{mini_db}")
    with engine.connect() as conn:
        codes = _top_mv(conn, n=10, as_of_date="2026-04-21")
    assert "600999.SH" not in codes


def test_top_mv_returns_mainboard_stocks(mini_db: str) -> None:
    """普通沪深主板股应在 universe."""
    engine = create_engine(f"sqlite:///{mini_db}")
    with engine.connect() as conn:
        codes = _top_mv(conn, n=10, as_of_date="2026-04-21")
    assert "600000.SH" in codes
    assert "000001.SZ" in codes
    assert "300750.SZ" in codes

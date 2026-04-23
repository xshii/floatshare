"""ST / 除权 / 停牌 的 per-stock mask 生成测试.

用临时 sqlite + 合成 panel, 验证 compute_trading_mask 能抓到三种情况.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from floatshare.ml.audit_mask import compute_trading_mask


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """最小 DB 仅含 stock_lifecycle + dividend."""
    db = tmp_path / "mask.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""
            CREATE TABLE stock_lifecycle (
                code TEXT PRIMARY KEY, name TEXT, list_date TEXT,
                delist_date TEXT, list_status TEXT, market TEXT,
                industry TEXT, updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE dividend (
                code TEXT, end_date TEXT, div_proc TEXT, ann_date TEXT,
                ex_date TEXT, cash_div REAL,
                PRIMARY KEY (code, end_date, div_proc)
            )
        """)
        conn.executemany(
            "INSERT INTO stock_lifecycle(code, name) VALUES (?, ?)",
            [
                ("000001.SZ", "平安银行"),
                ("000430.SZ", "ST张家界"),  # ST 股
                ("600004.SH", "*ST国华"),  # *ST 股
                ("600036.SH", "招商银行"),
                ("600519.SH", "贵州茅台"),
            ],
        )
        conn.execute(
            "INSERT INTO dividend(code, end_date, div_proc, ann_date, ex_date, cash_div) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("600519.SH", "2025-12-31", "实施", "2026-04-15", "2026-04-20", 30.88),
        )
        conn.commit()
    return db


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """5 只股 × 4 天. 000001.SZ 2026-04-20 当日停牌 (panel 里删掉该行)."""
    codes = ["000001.SZ", "000430.SZ", "600004.SH", "600036.SH", "600519.SH"]
    dates = pd.to_datetime(["2026-04-17", "2026-04-18", "2026-04-19", "2026-04-20"])
    rows = [{"code": c, "trade_date": d, "close": 10.0} for c in codes for d in dates]
    df = pd.DataFrame(rows)
    # 模拟 000001.SZ 在 2026-04-20 停牌 (该行不存在)
    return df[~((df["code"] == "000001.SZ") & (df["trade_date"] == pd.Timestamp("2026-04-20")))]


class TestTradingMask:
    def test_st_stock_masked(self, tmp_db: Path, synthetic_panel: pd.DataFrame) -> None:
        report = compute_trading_mask(synthetic_panel, "2026-04-20", str(tmp_db))
        assert "000430.SZ" in report.masked
        assert "ST" in report.masked["000430.SZ"]
        assert "600004.SH" in report.masked
        assert "ST" in report.masked["600004.SH"]

    def test_ex_dividend_day_masked(self, tmp_db: Path, synthetic_panel: pd.DataFrame) -> None:
        """贵州茅台 (600519.SH) 2026-04-20 除权 → 应被 mask."""
        report = compute_trading_mask(synthetic_panel, "2026-04-20", str(tmp_db))
        assert "600519.SH" in report.masked
        assert "ex_dividend" in report.masked["600519.SH"]

    def test_suspended_stock_masked(self, tmp_db: Path, synthetic_panel: pd.DataFrame) -> None:
        """000001.SZ 2026-04-20 停牌 (panel 缺该行) → 应被 mask."""
        report = compute_trading_mask(synthetic_panel, "2026-04-20", str(tmp_db))
        assert "000001.SZ" in report.masked
        assert "suspended" in report.masked["000001.SZ"]

    def test_normal_stock_not_masked(self, tmp_db: Path, synthetic_panel: pd.DataFrame) -> None:
        """招商银行 (600036.SH) 不是 ST / 不除权 / 不停牌 → 不应 mask."""
        report = compute_trading_mask(synthetic_panel, "2026-04-20", str(tmp_db))
        assert "600036.SH" not in report.masked

    def test_ex_dividend_before_day_not_masked(
        self,
        tmp_db: Path,
        synthetic_panel: pd.DataFrame,
    ) -> None:
        """贵州茅台的除权日是 2026-04-20, 04-19 不应 mask."""
        report = compute_trading_mask(synthetic_panel, "2026-04-19", str(tmp_db))
        assert "600519.SH" not in report.masked

    def test_summary_format(self, tmp_db: Path, synthetic_panel: pd.DataFrame) -> None:
        report = compute_trading_mask(synthetic_panel, "2026-04-20", str(tmp_db))
        s = report.summary()
        assert "ST=2" in s
        assert "ex_dividend=1" in s
        assert "suspended=1" in s

    def test_iso8601_ex_date_also_matched(self, tmp_path: Path) -> None:
        """DB 里 ex_date 可能存 'YYYY-MM-DDTHH:MM:SS' 格式, 也要匹配."""
        db = tmp_path / "iso.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE stock_lifecycle (code TEXT, name TEXT, list_date TEXT, "
                "delist_date TEXT, list_status TEXT, market TEXT, industry TEXT, updated_at TEXT)",
            )
            conn.execute(
                "CREATE TABLE dividend (code TEXT, end_date TEXT, div_proc TEXT, ann_date TEXT, "
                "ex_date TEXT, cash_div REAL)",
            )
            conn.execute(
                "INSERT INTO dividend VALUES ('600519.SH', '2025-12-31', '实施', "
                "'2026-04-15', '2026-04-20T00:00:00', 30.88)",
            )
            conn.commit()
        panel = pd.DataFrame(
            [
                {"code": "600519.SH", "trade_date": pd.Timestamp("2026-04-20"), "close": 1500.0},
            ]
        )
        report = compute_trading_mask(panel, "2026-04-20", str(db))
        assert "600519.SH" in report.masked

    def test_multiple_reasons_combined(self, tmp_path: Path) -> None:
        """同一股既 ST 又除权 → 多个 reasons."""
        db = tmp_path / "combo.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE stock_lifecycle (code TEXT, name TEXT, list_date TEXT, "
                "delist_date TEXT, list_status TEXT, market TEXT, industry TEXT, updated_at TEXT)",
            )
            conn.execute(
                "CREATE TABLE dividend (code TEXT, end_date TEXT, div_proc TEXT, ann_date TEXT, "
                "ex_date TEXT, cash_div REAL)",
            )
            conn.execute("INSERT INTO stock_lifecycle(code, name) VALUES ('600519.SH', 'ST茅台')")
            conn.execute(
                "INSERT INTO dividend VALUES ('600519.SH', '2025-12-31', '实施', "
                "'2026-04-15', '2026-04-20', 30.88)",
            )
            conn.commit()
        panel = pd.DataFrame(
            [
                {"code": "600519.SH", "trade_date": pd.Timestamp("2026-04-20"), "close": 1500.0},
            ]
        )
        report = compute_trading_mask(panel, "2026-04-20", str(db))
        assert set(report.masked["600519.SH"]) == {"ST", "ex_dividend"}

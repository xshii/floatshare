"""news 因果性测试 — 验证 CctvNewsConfig.max_mention_date 生效.

核心场景: panel 截到 T-5, 但 DB 里有 T-1 的 news mention. 若 news feature 查 DB 不
加时点过滤, 截短版 feats 会读到 T-1 的 mention → 泄漏未来. 加过滤后两版应一致.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from floatshare.ml.audit import _build_truncated_sources
from floatshare.ml.features import CctvNewsConfig, _CctvNewsSource, compute_features


@pytest.fixture
def db_with_mentions(tmp_path: Path) -> Path:
    """建 DB: industry + cctv_news_mentions. 600036.SH 属 801780 银行行业.

    mentions 表含:
        2026-04-17  801780  1
        2026-04-18  801780  1
        2026-04-19  801780  1
    """
    db = tmp_path / "news_causal.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""
            CREATE TABLE industry (
                code TEXT PRIMARY KEY, l1_code TEXT,
                l2_code TEXT, l3_code TEXT,
                l1_name TEXT, l2_name TEXT, l3_name TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE cctv_news_mentions (
                trade_date TEXT, l1_code TEXT, mentioned INTEGER,
                match_score REAL, matched_keywords TEXT,
                news_source TEXT, ingested_at TEXT,
                PRIMARY KEY (trade_date, l1_code)
            )
        """)
        conn.execute(
            "INSERT INTO industry(code, l1_code) VALUES (?, ?)",
            ("600036.SH", "801780.SI"),
        )
        for d in ("2026-04-17", "2026-04-18", "2026-04-19"):
            conn.execute(
                "INSERT INTO cctv_news_mentions "
                "(trade_date, l1_code, mentioned, ingested_at) VALUES (?, ?, 1, ?)",
                (d, "801780.SI", datetime.now().isoformat()),
            )
        conn.commit()
    return db


def _make_stock_panel(dates: list[str]) -> pd.DataFrame:
    """构造 1 只股的 minimal panel (只要含 news 需要的列)."""
    rows = [
        {
            "code": "600036.SH",
            "trade_date": pd.Timestamp(d),
            "close": 40.0,
            "open": 39.8,
            "high": 40.2,
            "low": 39.5,
            "volume": 1e6,
            "amount": 4e7,
            "pe_ttm": 6.0,
            "pb": 1.0,
            "turnover_rate": 1.0,
            "total_mv": 1e11,
            "circ_mv": 7e10,
            "net_mf_amount": 0.0,
            "buy_sm_amount": 1e5,
            "buy_md_amount": 1e5,
            "buy_lg_amount": 1e5,
            "buy_elg_amount": 1e5,
            "sell_sm_amount": 1e5,
            "sell_md_amount": 1e5,
            "sell_lg_amount": 1e5,
            "sell_elg_amount": 1e5,
        }
        for d in dates
    ]
    return pd.DataFrame(rows)


class TestNewsCausalityTimeFilter:
    def test_cctv_source_without_filter_reads_all(self, db_with_mentions: Path) -> None:
        """baseline: 不加 max_mention_date, source 读全量 DB."""
        src = _CctvNewsSource(CctvNewsConfig(db_path=str(db_with_mentions)))
        # panel 只到 2026-04-17
        panel = _make_stock_panel(["2026-04-17"])
        g = panel.set_index("trade_date")
        out = src.compute(g, {})
        # 应读到 04-17 的 mention=1
        assert out["news_mentioned_t"].iloc[0] == 1.0

    def test_cctv_source_with_max_date_truncates(self, db_with_mentions: Path) -> None:
        """加 max_mention_date=2026-04-17, 应只读到 04-17 及之前."""
        src = _CctvNewsSource(
            CctvNewsConfig(
                db_path=str(db_with_mentions),
                max_mention_date="2026-04-17",
            )
        )
        # panel 到 04-19, 但 max_mention_date 限制到 04-17
        panel = _make_stock_panel(["2026-04-17", "2026-04-18", "2026-04-19"])
        g = panel.set_index("trade_date")
        out = src.compute(g, {})
        # 04-17 可见, 04-18 / 04-19 虽然 DB 有但被过滤 → 回落到 0
        assert out["news_mentioned_t"].iloc[0] == 1.0
        assert out["news_mentioned_t"].iloc[1] == 0.0
        assert out["news_mentioned_t"].iloc[2] == 0.0

    def test_build_truncated_sources_replaces_only_cctv(self) -> None:
        """_build_truncated_sources 只改 _CctvNewsSource, 其它 source 原样返回."""
        trunc = _build_truncated_sources("2026-04-17")
        from floatshare.ml.features import FEATURE_SOURCES

        # 长度一致
        assert len(trunc) == len(FEATURE_SOURCES)
        # CCTV 被替换成新实例 (带 max_mention_date)
        orig_cctv = next(s for s in FEATURE_SOURCES if isinstance(s, _CctvNewsSource))
        new_cctv = next(s for s in trunc if isinstance(s, _CctvNewsSource))
        assert new_cctv is not orig_cctv
        assert new_cctv._cfg.max_mention_date == "2026-04-17"
        # 非 CCTV source 未被改 (引用相同)
        for s_orig, s_new in zip(FEATURE_SOURCES, trunc, strict=True):
            if not isinstance(s_orig, _CctvNewsSource):
                assert s_orig is s_new

    def test_compute_features_sources_override(self, db_with_mentions: Path) -> None:
        """compute_features 传 sources 参数 → news feature 用新 config."""
        panel = _make_stock_panel(
            [
                "2026-04-15",
                "2026-04-16",
                "2026-04-17",
                "2026-04-18",
                "2026-04-19",
            ]
        )
        # 用 max=2026-04-17 的 sources 跑
        from floatshare.ml.features import FEATURE_SOURCES

        trunc_sources: tuple = tuple(
            _CctvNewsSource(
                CctvNewsConfig(
                    db_path=str(db_with_mentions),
                    max_mention_date="2026-04-17",
                )
            )
            if isinstance(s, _CctvNewsSource)
            else s
            for s in FEATURE_SOURCES
        )
        feats = compute_features(panel, sources=trunc_sources)
        feats = feats.sort_values("trade_date").reset_index(drop=True)
        # shift_days=0, news[D] = mention@D. 04-17 mention=1, 04-18/19 被过滤 → 0
        # 但 panel 只有 5 天, 04-15/04-16 DB 里没 mention → 0
        assert (
            feats.loc[feats["trade_date"] == pd.Timestamp("2026-04-17"), "news_mentioned_t"].iloc[0]
            == 1.0
        )
        assert (
            feats.loc[feats["trade_date"] == pd.Timestamp("2026-04-18"), "news_mentioned_t"].iloc[0]
            == 0.0
        )
        assert (
            feats.loc[feats["trade_date"] == pd.Timestamp("2026-04-19"), "news_mentioned_t"].iloc[0]
            == 0.0
        )

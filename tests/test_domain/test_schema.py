"""OHLCV schema normalize_ohlcv 单元测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from floatshare.domain.schema import (
    OHLCV_OPTIONAL,
    OHLCV_REQUIRED,
    normalize_ohlcv,
)


def _full_df() -> pd.DataFrame:
    """构造完整的 OHLCV DataFrame。"""
    return pd.DataFrame(
        {
            "code": ["X"],
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.2],
            "volume": [1000.0],
            "amount": [10200.0],
            "pre_close": [9.8],
            "pct_change": [4.08],
            "turnover": [0.5],
            "adj_factor": [1.0],
        }
    )


class TestNormalizeOHLCV:
    def test_empty_df_passthrough(self) -> None:
        assert normalize_ohlcv(pd.DataFrame()).empty

    def test_full_df_kept_intact(self) -> None:
        df = _full_df()
        out = normalize_ohlcv(df)
        # 列顺序可能变了，但内容相同
        for col in df.columns:
            assert col in out.columns
        # 行数不变
        assert len(out) == len(df)

    def test_missing_required_raises(self) -> None:
        df = pd.DataFrame({"open": [1.0]})  # 缺一堆必需列
        with pytest.raises(ValueError, match="缺少必需列"):
            normalize_ohlcv(df)

    def test_optional_filled_with_NA(self) -> None:
        """tushare 风格：有 pre_close，没 turnover；akshare 反之。"""
        minimal = pd.DataFrame(
            {
                "code": ["X"],
                "trade_date": pd.to_datetime(["2024-01-01"]),
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [10.2],
                "volume": [1000.0],
            }
        )
        out = normalize_ohlcv(minimal)
        # 所有可选列都补上了
        for col in OHLCV_OPTIONAL:
            assert col in out.columns, f"missing optional col {col}"
            assert pd.isna(out[col].iloc[0]), f"{col} should be NA, got {out[col].iloc[0]}"

    def test_column_order_consistent(self) -> None:
        """无论传入列顺序如何，输出顺序一致 — 降级链 schema 稳定。"""
        df = _full_df()
        # 打乱列顺序
        shuffled = df[df.columns[::-1]]
        out = normalize_ohlcv(shuffled)
        expected_prefix = list(OHLCV_REQUIRED) + list(OHLCV_OPTIONAL)
        assert list(out.columns)[: len(expected_prefix)] == expected_prefix

    def test_extra_columns_preserved_after_standard(self) -> None:
        """业务额外列不丢失，按字母序排在标准列之后。"""
        df = _full_df()
        df["custom_metric"] = 42
        df["another_one"] = "x"
        out = normalize_ohlcv(df)
        # 标准列在前
        n_standard = len(OHLCV_REQUIRED) + len(OHLCV_OPTIONAL)
        # 额外列按字母序在后
        extras = list(out.columns)[n_standard:]
        assert extras == sorted(["another_one", "custom_metric"])

    def test_simulates_tushare_to_akshare_failover(self) -> None:
        """模拟降级场景：Tushare 返回带 pre_close，AKShare 不带。

        normalize 后两者 schema 一致，下游用 df["pre_close"] 不会 KeyError。
        """
        # Tushare 形态
        tushare_df = pd.DataFrame(
            {
                "code": ["X"],
                "trade_date": pd.to_datetime(["2024-01-01"]),
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [10.2],
                "volume": [1000.0],
                "amount": [10200.0],
                "pre_close": [9.8],
            }
        )
        # AKShare 形态（无 pre_close，但有 change）
        akshare_df = pd.DataFrame(
            {
                "code": ["X"],
                "trade_date": pd.to_datetime(["2024-01-01"]),
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [10.2],
                "volume": [1000.0],
                "amount": [10200.0],
                "change": [0.4],
            }
        )

        ts_norm = normalize_ohlcv(tushare_df)
        ak_norm = normalize_ohlcv(akshare_df)

        # 两者都有标准列集合
        for col in [*OHLCV_REQUIRED, *OHLCV_OPTIONAL]:
            assert col in ts_norm.columns
            assert col in ak_norm.columns

        # ak 缺的 pre_close 是 NA，不报错
        assert pd.isna(ak_norm["pre_close"].iloc[0])
        # ak 多的 change 保留
        assert "change" in ak_norm.columns

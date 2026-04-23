"""tushare stk_factor 第三方对拍测试 — mock tushare API 验证逻辑."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from floatshare.ml.audit_tushare import run_tushare_spot_check


@pytest.fixture
def fake_feats() -> pd.DataFrame:
    """我们 feats[T=2026-04-20] 上 3 只股的 (rsi12, kdj_j, macd_hist_norm).

    反归一化后期望值:
        rsi12:          (-0.2 + 0.5) * 100 = 30     (对 tushare.rsi_12)
        kdj_j:          (0.1 + 0.5) * 100  = 60     (对 tushare.kdj_j)
        macd_hist_norm: 0.02 * close[T-1]           (对 tushare.macd)
    """
    return pd.DataFrame(
        [
            {
                "code": "000001.SZ",
                "trade_date": pd.Timestamp("2026-04-20"),
                "close": 11.0,
                "rsi12": -0.2,
                "kdj_j": 0.1,
                "macd_hist_norm": 0.02,
            },
            {
                "code": "600519.SH",
                "trade_date": pd.Timestamp("2026-04-20"),
                "close": 1500.0,
                "rsi12": 0.2,
                "kdj_j": -0.1,
                "macd_hist_norm": -0.01,
            },
            {
                "code": "600036.SH",
                "trade_date": pd.Timestamp("2026-04-20"),
                "close": 40.0,
                "rsi12": 0.0,
                "kdj_j": 0.0,
                "macd_hist_norm": 0.0,
            },
        ]
    )


class TestTushareSpotCheck:
    def test_all_matched_returns_empty(self, fake_feats: pd.DataFrame) -> None:
        """tushare 返回的值跟我们期望完全一致 → 0 mismatch."""
        # tushare 在 T-1=2026-04-19 (周日 → 回退到 2026-04-17 周五, 这里用 close 关联)
        # close 对应 tushare.close (T-1), 影响 macd 反归一化
        tushare_df = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "close": 11.0,
                    "rsi_12": 30.0,
                    "kdj_j": 60.0,
                    "macd": 0.02 * 11.0,
                },
                {
                    "ts_code": "600519.SH",
                    "close": 1500.0,
                    "rsi_12": 70.0,
                    "kdj_j": 40.0,
                    "macd": -0.01 * 1500.0,
                },
                {"ts_code": "600036.SH", "close": 40.0, "rsi_12": 50.0, "kdj_j": 50.0, "macd": 0.0},
            ]
        )
        source = MagicMock()
        source.pro.stk_factor.return_value = tushare_df

        mismatches = run_tushare_spot_check(fake_feats, "2026-04-20", source)
        assert mismatches == []

    def test_rsi_mismatch_detected(self, fake_feats: pd.DataFrame) -> None:
        """tushare rsi_12 跟我们差 > atol → 报 mismatch."""
        tushare_df = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "close": 11.0,
                    "rsi_12": 50.0,
                    "kdj_j": 60.0,
                    "macd": 0.02 * 11.0,
                },  # rsi 错 20 (超 0.5 atol)
            ]
        )
        source = MagicMock()
        source.pro.stk_factor.return_value = tushare_df
        mismatches = run_tushare_spot_check(fake_feats, "2026-04-20", source)
        rsi_alerts = [m for m in mismatches if m.feature == "rsi12"]
        assert len(rsi_alerts) == 1
        assert rsi_alerts[0].code == "000001.SZ"
        assert abs(rsi_alerts[0].our_value - 30.0) < 1e-6
        assert rsi_alerts[0].tushare_value == 50.0

    def test_kdj_mismatch_detected(self, fake_feats: pd.DataFrame) -> None:
        tushare_df = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "close": 11.0,
                    "rsi_12": 30.0,
                    "kdj_j": 100.0,
                    "macd": 0.02 * 11.0,
                },  # kdj 错 40
            ]
        )
        source = MagicMock()
        source.pro.stk_factor.return_value = tushare_df
        mismatches = run_tushare_spot_check(fake_feats, "2026-04-20", source)
        kdj_alerts = [m for m in mismatches if m.feature == "kdj_j"]
        assert len(kdj_alerts) == 1
        assert kdj_alerts[0].tushare_value == 100.0

    def test_macd_relative_tolerance(self, fake_feats: pd.DataFrame) -> None:
        """macd 用相对误差. 我们期望 0.02 * 11 = 0.22; tushare 给 0.225 (相对 2.3% > 2% atol)."""
        tushare_df = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "close": 11.0,
                    "rsi_12": 30.0,
                    "kdj_j": 60.0,
                    "macd": 0.225,
                },
            ]
        )
        source = MagicMock()
        source.pro.stk_factor.return_value = tushare_df
        mismatches = run_tushare_spot_check(
            fake_feats,
            "2026-04-20",
            source,
            atol_macd_rel=0.02,
        )
        macd_alerts = [m for m in mismatches if m.feature == "macd_hist_norm"]
        assert len(macd_alerts) == 1

    def test_within_atol_no_mismatch(self, fake_feats: pd.DataFrame) -> None:
        """差值 < atol → 不报."""
        tushare_df = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "close": 11.0,
                    "rsi_12": 30.3,
                    "kdj_j": 60.5,
                    "macd": 0.02 * 11.0 * 1.01,
                },  # 1% 误差 in macd
            ]
        )
        source = MagicMock()
        source.pro.stk_factor.return_value = tushare_df
        mismatches = run_tushare_spot_check(
            fake_feats,
            "2026-04-20",
            source,
            atol_rsi=0.5,
            atol_kdj=1.0,
            atol_macd_rel=0.02,
        )
        assert mismatches == []

    def test_empty_tushare_response_returns_empty(self, fake_feats: pd.DataFrame) -> None:
        """tushare 返回空 → 跳过对拍, 不 raise."""
        source = MagicMock()
        source.pro.stk_factor.return_value = pd.DataFrame()
        mismatches = run_tushare_spot_check(fake_feats, "2026-04-20", source)
        assert mismatches == []

    def test_sample_codes_limits_checks(self, fake_feats: pd.DataFrame) -> None:
        """sample_codes=1 → 只检查 1 只股."""
        tushare_df = pd.DataFrame(
            [
                {"ts_code": code, "close": 11.0, "rsi_12": 999.0, "kdj_j": 60.0, "macd": 0.22}
                for code in ("000001.SZ", "600519.SH", "600036.SH")
            ]
        )
        source = MagicMock()
        source.pro.stk_factor.return_value = tushare_df
        mismatches = run_tushare_spot_check(
            fake_feats,
            "2026-04-20",
            source,
            sample_codes=1,
        )
        # 每只股会各自报 rsi12 错, 但只抽了 1 只 → mismatches 至多 1 个 feature × 1 code
        rsi_alerts = [m for m in mismatches if m.feature == "rsi12"]
        assert len(rsi_alerts) == 1

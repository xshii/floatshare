"""DataLoader 回写回调测试 — 验证远程拉取成功后触发缓存回写。"""

from __future__ import annotations

import pandas as pd

from floatshare.application.data_loader import DataLoader
from tests.conftest import FakeDataSource


class TestWriteback:
    def test_daily_writeback_called(self) -> None:
        written: list[tuple[str, pd.DataFrame]] = []
        src = FakeDataSource()
        loader = DataLoader(
            daily=[src],
            on_daily_fetched=lambda code, df: written.append((code, df)),
        )
        loader.get_daily("000001.SZ")
        assert len(written) == 1
        assert written[0][0] == "000001.SZ"

    def test_stock_list_writeback_called(self) -> None:
        written: list[pd.DataFrame] = []
        src = FakeDataSource()
        loader = DataLoader(
            stock_list=[src],
            on_stock_list_fetched=written.append,
        )
        loader.get_stock_list()
        assert len(written) == 1

    def test_writeback_failure_non_fatal(self) -> None:
        """回写失败不应影响数据返回。"""

        def bad_callback(code: str, df: pd.DataFrame) -> None:
            raise RuntimeError("写入爆炸")

        src = FakeDataSource()
        loader = DataLoader(daily=[src], on_daily_fetched=bad_callback)
        df = loader.get_daily("000001.SZ")
        assert not df.empty

    def test_no_writeback_when_no_callback(self) -> None:
        src = FakeDataSource()
        loader = DataLoader(daily=[src])
        df = loader.get_daily("000001.SZ")
        assert not df.empty

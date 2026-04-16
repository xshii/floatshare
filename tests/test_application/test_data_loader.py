"""DataLoader 链式降级契约测试。"""

from __future__ import annotations

import pytest

from floatshare.application.data_loader import AllSourcesFailed, DataLoader
from tests.conftest import FakeDataSource


class TestDataLoaderChain:
    def test_single_source_success(self, fake_source):
        loader = DataLoader(daily=[fake_source])
        df = loader.get_daily("000001.SZ")
        assert not df.empty
        assert fake_source.calls["get_daily"] == 1

    def test_falls_back_to_second(self):
        bad = FakeDataSource(fail_modes={"get_daily"})
        good = FakeDataSource()
        loader = DataLoader(daily=[bad, good])
        df = loader.get_daily("000001.SZ")
        assert not df.empty
        assert bad.calls["get_daily"] == 1
        assert good.calls["get_daily"] == 1

    def test_all_fail_raises(self):
        bad1 = FakeDataSource(fail_modes={"get_daily"})
        bad2 = FakeDataSource(fail_modes={"get_daily"})
        loader = DataLoader(daily=[bad1, bad2])
        with pytest.raises(AllSourcesFailed):
            loader.get_daily("000001.SZ")

    def test_each_capability_independent_chain(self):
        d_bad = FakeDataSource(fail_modes={"get_daily"})
        d_good = FakeDataSource()
        i_only = FakeDataSource()
        loader = DataLoader(daily=[d_bad, d_good], index=[i_only])

        loader.get_daily("000001.SZ")
        loader.get_index_daily("399300.SZ")

        assert d_bad.calls["get_daily"] == 1
        assert d_good.calls["get_daily"] == 1
        assert i_only.calls["get_index_daily"] == 1
        assert "get_daily" not in i_only.calls

    def test_empty_chain_raises(self):
        loader = DataLoader(daily=[])
        with pytest.raises(AllSourcesFailed):
            loader.get_daily("X")

    def test_default_factory_returns_loader(self):
        from floatshare.application import create_default_loader

        loader = create_default_loader()
        assert isinstance(loader, DataLoader)
        assert len(loader.daily) >= 1

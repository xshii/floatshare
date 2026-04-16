"""告警/日志层测试。"""

from __future__ import annotations

import os

from floatshare.observability import notify
from floatshare.observability.alert import _client


class TestNotify:
    def test_no_url_returns_false(self, monkeypatch):
        monkeypatch.delenv("FLOATSHARE_NOTIFY_URLS", raising=False)
        _client.cache_clear()
        assert notify("hello", "world") is False

    def test_lru_cache_resettable(self, monkeypatch):
        monkeypatch.delenv("FLOATSHARE_NOTIFY_URLS", raising=False)
        _client.cache_clear()
        first = _client()
        second = _client()
        # 单例语义
        assert first is second
        _client.cache_clear()
        # 清掉之后是新实例
        assert _client() is not first

    def test_env_picked_up_after_reset(self, monkeypatch):
        # 用一个永远不会真发的 url：apprise 注册时不会校验远端
        monkeypatch.setenv("FLOATSHARE_NOTIFY_URLS", "json://example.invalid")
        _client.cache_clear()
        client = _client()
        assert len(client) == 1
        # 测试结束后 fixture 会清缓存
        os.environ.pop("FLOATSHARE_NOTIFY_URLS", None)

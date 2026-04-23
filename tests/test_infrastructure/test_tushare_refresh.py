"""Tushare token 自动刷新逻辑的单元测试（纯 mock，不依赖网络）。"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from floatshare.infrastructure.data_sources import tushare as tushare_mod
from floatshare.infrastructure.data_sources.tushare import (
    TushareSource,
    _extract_token,
    _is_token_error,
    _load_cached_token,
    _save_cached_token,
)
from floatshare.interfaces.data_source import DataSourceError

# --- _is_token_error ---------------------------------------------------------


class TestIsTokenError:
    @pytest.mark.parametrize(
        "msg",
        [
            "token error",
            "认证失败",
            "权限不足",
            "auth failure",
            "invalid credential",
            "抱歉，您没有访问该接口的权限",
        ],
    )
    def test_recognizes_token_errors(self, msg: str) -> None:
        assert _is_token_error(Exception(msg))

    @pytest.mark.parametrize("msg", ["timeout", "network error", "500 server error"])
    def test_ignores_non_token_errors(self, msg: str) -> None:
        assert not _is_token_error(Exception(msg))


# --- _extract_token (YAGNI 化后只接受纯文本) ---------------------------------


class TestExtractToken:
    def test_plain_text_token(self) -> None:
        token = "abc123" * 5  # 30 字符
        assert _extract_token(token) == token

    def test_token_with_underscore(self) -> None:
        token = "tk_" + "a" * 20
        assert _extract_token(token) == token

    def test_strips_whitespace(self) -> None:
        token = "a" * 30
        assert _extract_token(f"  {token}\n") == token

    def test_rejects_short_string(self) -> None:
        assert _extract_token("short") is None
        assert _extract_token("a" * 19) is None  # 边界

    def test_rejects_non_alnum(self) -> None:
        assert _extract_token("not a token!" + "x" * 20) is None


# --- @_auto_refresh 装饰器 ---------------------------------------------------


class TestAutoRefresh:
    """测试 @_auto_refresh 装饰器在 token 失效时的重试逻辑。"""

    def _make_source(self) -> TushareSource:
        src = TushareSource(token="old_token", refresh_key="test_key")
        src._pro = MagicMock()
        return src

    def test_success_without_refresh(self) -> None:
        src = self._make_source()
        src._pro.stock_basic.return_value = MagicMock()
        src._pro.stock_basic.return_value.rename.return_value = "ok"
        result = src.get_stock_list()
        assert result == "ok"

    def test_refresh_on_token_error(self) -> None:
        src = self._make_source()
        # 第一次调用抛 token 错误，刷新后第二次成功
        mock_result = MagicMock()
        mock_result.rename.return_value = "ok"
        src._pro.stock_basic.side_effect = [Exception("token error"), mock_result]

        with (
            patch.object(src, "_refresh_token", return_value="new_token"),
            patch.object(src, "_reset_pro"),
        ):
            result = src.get_stock_list()

        assert result == "ok"
        assert src.token == "new_token"

    def test_gives_up_after_max_retries(self) -> None:
        src = self._make_source()
        src._pro.stock_basic.side_effect = Exception("token error")

        with (
            patch.object(src, "_refresh_token", return_value="new_token"),
            patch.object(src, "_reset_pro"),
            pytest.raises(DataSourceError, match="刷新 2 次后仍失败"),
        ):
            src.get_stock_list()

    def test_non_token_error_not_retried(self) -> None:
        src = self._make_source()
        src._pro.stock_basic.side_effect = ValueError("bad data")

        with pytest.raises(ValueError, match="bad data"):
            src.get_stock_list()

    def test_refresh_failure_raises_immediately(self) -> None:
        src = self._make_source()
        src._pro.stock_basic.side_effect = Exception("认证失败")

        with (
            patch.object(src, "_refresh_token", side_effect=DataSourceError("刷新失败")),
            pytest.raises(DataSourceError, match="失效且刷新失败"),
        ):
            src.get_stock_list()


# --- Token 持久化 cache ----------------------------------------------------


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 token cache 路径重定向到 tmp_path，避免污染 ~/.floatshare/。"""
    cache_path = tmp_path / "tushare_token"
    monkeypatch.setattr(tushare_mod, "_TOKEN_CACHE_PATH", cache_path)
    return cache_path


class TestTokenCache:
    def test_load_returns_none_when_missing(self, isolated_cache: Path) -> None:
        assert not isolated_cache.exists()
        assert _load_cached_token() is None

    def test_save_then_load_roundtrip(self, isolated_cache: Path) -> None:
        _save_cached_token("abc123" * 5)
        assert _load_cached_token() == "abc123" * 5
        # 文件确实创建了
        assert isolated_cache.exists()

    def test_save_strips_whitespace_on_load(self, isolated_cache: Path) -> None:
        isolated_cache.parent.mkdir(parents=True, exist_ok=True)
        isolated_cache.write_text("  abc123abc123abc123  \n")
        assert _load_cached_token() == "abc123abc123abc123"

    def test_load_returns_none_for_empty_file(self, isolated_cache: Path) -> None:
        isolated_cache.parent.mkdir(parents=True, exist_ok=True)
        isolated_cache.write_text("")
        assert _load_cached_token() is None

    def test_save_sets_file_permissions(self, isolated_cache: Path) -> None:
        _save_cached_token("xyz" * 10)
        # POSIX 权限: owner read+write only (0o600)
        if os.name == "posix":
            mode = isolated_cache.stat().st_mode & 0o777
            assert mode == 0o600

    def test_init_priority_constructor_arg_wins(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """显式 token 参数 > cache > env"""
        _save_cached_token("CACHED_TOKEN_xxxxxxxxxxxxx")
        monkeypatch.setenv("TUSHARE_TOKEN", "ENV_TOKEN")
        src = TushareSource(token="EXPLICIT_TOKEN")
        assert src.token == "EXPLICIT_TOKEN"

    def test_init_priority_cache_over_env(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _save_cached_token("CACHED_TOKEN_xxxxxxxxxxxxx")
        monkeypatch.setenv("TUSHARE_TOKEN", "ENV_TOKEN")
        src = TushareSource()
        assert src.token == "CACHED_TOKEN_xxxxxxxxxxxxx"

    def test_init_falls_back_to_env_when_no_cache(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # cache 不存在
        assert not isolated_cache.exists()
        monkeypatch.setenv("TUSHARE_TOKEN", "ENV_TOKEN")
        src = TushareSource()
        assert src.token == "ENV_TOKEN"

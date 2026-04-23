"""Feature flags — 注册、env 解析、约束校验 单元测试。"""

from __future__ import annotations

import pytest

from floatshare.observability import features as ff


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    ff.reset_registry()


# ============================================================================
# 基础: register / is_enabled
# ============================================================================


class TestRegisterAndRead:
    def test_default_off(self):
        ff.register("foo", "test")
        assert ff.is_enabled("foo") is False

    def test_default_on(self):
        ff.register("foo", "test", default=True)
        assert ff.is_enabled("foo") is True

    def test_unknown_returns_false(self):
        assert ff.is_enabled("never_registered") is False

    def test_register_lowercases(self):
        ff.register("FooBar", "test")
        assert ff.is_enabled("foobar") is True or ff.is_enabled("foobar") is False
        assert "foobar" in {f.name for f in ff.all_flags()}


class TestEnvParsing:
    def test_single_env_truthy(self, monkeypatch: pytest.MonkeyPatch):
        ff.register("foo", "test")
        for v in ("1", "true", "TRUE", "yes", "on", "Enabled"):
            monkeypatch.setenv("FLOATSHARE_FEATURE_FOO", v)
            assert ff.is_enabled("foo") is True, f"failed for {v!r}"

    def test_single_env_falsy(self, monkeypatch: pytest.MonkeyPatch):
        ff.register("foo", "test", default=True)
        monkeypatch.setenv("FLOATSHARE_FEATURE_FOO", "0")
        assert ff.is_enabled("foo") is False  # 单 env 优先于 default

    def test_multi_env(self, monkeypatch: pytest.MonkeyPatch):
        ff.register("a", "")
        ff.register("b", "")
        ff.register("c", "")
        monkeypatch.setenv("FLOATSHARE_FEATURES", "a, b ,d")
        assert ff.is_enabled("a") is True
        assert ff.is_enabled("b") is True
        assert ff.is_enabled("c") is False
        assert ff.is_enabled("d") is True  # 即使没注册, 在 multi 里也算开

    def test_single_overrides_multi(self, monkeypatch: pytest.MonkeyPatch):
        ff.register("foo", "")
        monkeypatch.setenv("FLOATSHARE_FEATURES", "foo")
        monkeypatch.setenv("FLOATSHARE_FEATURE_FOO", "0")
        assert ff.is_enabled("foo") is False


# ============================================================================
# 约束: requires / conflicts
# ============================================================================


class TestRegistryValidation:
    def test_clean_registry_passes(self):
        ff.register("foo", "")
        assert ff.validate_registry() == []

    def test_unknown_requires_caught(self):
        ff.register("foo", "", requires=("ghost",))
        errors = ff.validate_registry()
        assert any("ghost" in e for e in errors)

    def test_unknown_conflicts_caught(self):
        ff.register("foo", "", conflicts=("ghost",))
        errors = ff.validate_registry()
        assert any("ghost" in e for e in errors)

    def test_cycle_detection(self):
        ff.register("a", "", requires=("b",))
        ff.register("b", "", requires=("c",))
        ff.register("c", "", requires=("a",))
        errors = ff.validate_registry()
        assert any("循环" in e for e in errors)


class TestEnabledValidation:
    def test_satisfied_dependency(self, monkeypatch: pytest.MonkeyPatch):
        ff.register("base", "", default=True)
        ff.register("ext", "", default=True, requires=("base",))
        assert ff.validate_enabled() == []

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch):
        ff.register("base", "")  # off
        ff.register("ext", "", default=True, requires=("base",))
        errors = ff.validate_enabled()
        assert any("base" in e and "ext" in e for e in errors)

    def test_conflict_violation(self, monkeypatch: pytest.MonkeyPatch):
        ff.register("a", "", default=True, conflicts=("b",))
        ff.register("b", "", default=True)
        errors = ff.validate_enabled()
        assert any("互斥" in e for e in errors)


class TestSummary:
    def test_enabled_summary_includes_default(self):
        ff.register("a", "", default=True)
        ff.register("b", "", default=False)
        s = ff.enabled_summary()
        assert s == {"a": True, "b": False}

    def test_all_flags_sorted(self):
        ff.register("z", "z desc", category="z_cat")
        ff.register("a", "a desc", category="a_cat")
        names = [f.name for f in ff.all_flags()]
        # 按 (category, name) 排序 — a_cat 在 z_cat 前面
        assert names == ["a", "z"]

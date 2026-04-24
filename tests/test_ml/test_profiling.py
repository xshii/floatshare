"""SectionTimer / maybe_section 单测."""

from __future__ import annotations

import time

from floatshare.ml.profiling import SectionTimer, maybe_section


def test_section_timer_accumulates() -> None:
    t = SectionTimer(device=None)
    for _ in range(3):
        with t.section("a"):
            time.sleep(0.01)
        with t.section("b"):
            time.sleep(0.005)

    r = t.report()
    assert set(r.keys()) == {"a", "b"}
    assert r["a"]["count"] == 3
    assert r["b"]["count"] == 3
    assert r["a"]["sum_s"] > r["b"]["sum_s"]  # a 每次 2× b
    assert r["a"]["pct"] + r["b"]["pct"] == 100.0


def test_section_timer_format_report_not_empty() -> None:
    t = SectionTimer(device=None)
    with t.section("x"):
        pass
    report = t.format_report()
    assert "x" in report
    assert "count" in report  # header present


def test_section_timer_empty_report() -> None:
    t = SectionTimer(device=None)
    assert t.report() == {}
    assert "empty" in t.format_report().lower()


def test_maybe_section_none_is_zero_overhead() -> None:
    """timer=None 时 maybe_section 返回 nullcontext, 不应做任何记录."""
    with maybe_section(None, "anything"):
        pass  # 不应 raise 不应记录


def test_maybe_section_records_when_timer_given() -> None:
    t = SectionTimer(device=None)
    with maybe_section(t, "recorded"):
        time.sleep(0.002)
    r = t.report()
    assert "recorded" in r
    assert r["recorded"]["count"] == 1


def test_section_timer_nested_sections() -> None:
    """嵌套 section: 各自独立记录, inner 不影响 outer."""
    t = SectionTimer(device=None)
    with t.section("outer"):
        time.sleep(0.005)
        with t.section("inner"):
            time.sleep(0.003)
    r = t.report()
    assert r["outer"]["sum_s"] >= r["inner"]["sum_s"]

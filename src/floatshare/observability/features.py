"""Feature flags — 全局功能开关 + 依赖/冲突约束。

设计原则:
- 注册式: 加新 feature 时 `register(...)` 一次, 单一真相来源
- 影响透明: 每个 flag 必填 impact (启用后改变了什么)
- 依赖约束: requires/conflicts 字段 + 启动时校验
- 多种打开方式 (按优先级):
    1. 单 flag env:    FLOATSHARE_FEATURE_AUTO_ANALYZE=1
    2. 多 flag env:    FLOATSHARE_FEATURES=auto_analyze,verbose_sync
    3. 注册时 default
    4. False
- 多 flag 可同时生效, 互不冲突 (除非显式声明 conflicts)

依赖约束实现:
  用 stdlib `graphlib.TopologicalSorter` — 不引入外部 feature flag 平台。
  - register 时不立即检查 (允许文件加载顺序无关)
  - 启动时调 validate_registry() / validate_enabled() 检查

策略级 flag 不走这里 — 直接用 backtrader `params = (("use_xxx", False), ...)`,
WFO 自动可搜索 (trial.suggest_categorical)。

用法:
    from floatshare.observability import features

    features.register(
        "auto_analyze",
        description="sync 完成后跑 SQLite ANALYZE",
        impact="DB 增 1-2 分钟; 后续 COUNT(*) 等查询略快",
        category="sync",
        default=False,
    )

    if features.is_enabled("auto_analyze"):
        run_analyze()

环境配置:
    export FLOATSHARE_FEATURE_AUTO_ANALYZE=1                      # 单独
    export FLOATSHARE_FEATURES="auto_analyze,verbose_sync"        # 多个

CLI 调试:
    python -c "from floatshare.observability import features; features.print_flags()"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter


@dataclass(frozen=True, slots=True)
class FeatureFlag:
    """单个 feature 的元信息。"""

    name: str
    description: str  # 一句话说明 flag 干啥
    impact: str = ""  # 启用后的副作用 (用户必读)
    category: str = "general"  # sync / strategy / web / ...
    default: bool = False
    requires: tuple[str, ...] = field(default_factory=tuple)  # 依赖的其它 flag
    conflicts: tuple[str, ...] = field(default_factory=tuple)  # 互斥的 flag


_REGISTRY: dict[str, FeatureFlag] = {}
_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})


def register(
    name: str,
    description: str,
    *,
    impact: str = "",
    category: str = "general",
    default: bool = False,
    requires: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
) -> FeatureFlag:
    """登记一个 feature flag, 返回 FeatureFlag (调用方可忽略)。

    重复登记同名 flag 会覆盖。
    """
    flag = FeatureFlag(
        name=name.lower().strip(),
        description=description,
        impact=impact,
        category=category,
        default=default,
        requires=tuple(r.lower().strip() for r in requires),
        conflicts=tuple(c.lower().strip() for c in conflicts),
    )
    _REGISTRY[flag.name] = flag
    return flag


def is_enabled(name: str) -> bool:
    """检查 feature 当前是否启用 (env > default 优先级合并)。"""
    name = name.lower().strip()
    single = os.getenv(f"FLOATSHARE_FEATURE_{name.upper()}")
    if single is not None:
        return single.strip().lower() in _TRUTHY
    multi = os.getenv("FLOATSHARE_FEATURES", "").strip()
    if multi:
        enabled = {s.strip().lower() for s in multi.split(",") if s.strip()}
        if name in enabled:
            return True
    flag = _REGISTRY.get(name)
    return flag.default if flag else False


def all_flags() -> list[FeatureFlag]:
    """已登记的全部 flag (按 category/name 排序)。"""
    return sorted(_REGISTRY.values(), key=lambda f: (f.category, f.name))


def enabled_summary() -> dict[str, bool]:
    """每个已登记 flag 的当前生效状态。"""
    return {f.name: is_enabled(f.name) for f in _REGISTRY.values()}


# ============================================================================
# 约束校验
# ============================================================================


def validate_registry() -> list[str]:
    """检查注册表本身: 循环依赖 / 引用了未登记的 flag。

    应用启动时调一次，发现错误立即报。
    返回错误列表 (空 = OK)。
    """
    errors: list[str] = []
    graph: dict[str, set[str]] = {n: set(f.requires) for n, f in _REGISTRY.items()}

    # 1. 引用未登记的 flag
    for name, deps in graph.items():
        errors.extend(
            f"❌ {name} requires '{dep}' but '{dep}' 未登记" for dep in deps if dep not in _REGISTRY
        )
        errors.extend(
            f"❌ {name} conflicts with '{c}' 但 '{c}' 未登记"
            for c in _REGISTRY[name].conflicts
            if c not in _REGISTRY
        )

    # 2. 循环依赖 (stdlib graphlib 检测)
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as exc:
        errors.append(f"❌ 循环依赖: {exc.args[1] if len(exc.args) > 1 else exc}")

    return errors


def validate_enabled() -> list[str]:
    """检查当前启用的 flag 集合是否满足约束 (requires/conflicts)。

    应用启动后调，决定是 fail-fast 还是 warn。返回错误列表。
    """
    enabled = {n for n, on in enabled_summary().items() if on}
    errors: list[str] = []
    for name in enabled:
        flag = _REGISTRY[name]
        errors.extend(
            f"❌ '{name}' 启用了, 但依赖的 '{req}' 未启用"
            for req in flag.requires
            if req not in enabled
        )
        errors.extend(
            f"❌ '{name}' 和 '{c}' 不能同时启用 (互斥)" for c in flag.conflicts if c in enabled
        )
    return errors


# ============================================================================
# 调试 / 展示
# ============================================================================


def print_flags() -> None:
    """日志打印全部已登记 flag 的状态 + 影响 (CLI 调试用)。

    输出走 loguru — 既彩色显示在 stderr, 也落到 logs/floatshare_*.log,
    方便事后审计 (例如 sync 失败时看当时启用了哪些 flag)。
    """
    from loguru import logger

    by_cat: dict[str, list[FeatureFlag]] = {}
    for f in all_flags():
        by_cat.setdefault(f.category, []).append(f)

    for cat, flags in by_cat.items():
        logger.info(f"=== feature-flags [{cat}] ===")
        for f in flags:
            on = is_enabled(f.name)
            mark = "✓ ON " if on else "✗ off"
            logger.info(f"  {mark}  {f.name:<28} {f.description}")
            if f.impact:
                logger.info(f"           影响: {f.impact}")
            if f.requires:
                logger.info(f"           依赖: {', '.join(f.requires)}")
            if f.conflicts:
                logger.info(f"           冲突: {', '.join(f.conflicts)}")

    errs = validate_enabled()
    if errs:
        logger.warning("当前启用集合违反约束:")
        for e in errs:
            logger.warning(f"  {e}")


def reset_registry() -> None:
    """清空注册表 (仅测试用)。"""
    _REGISTRY.clear()

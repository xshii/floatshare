"""所有 feature flags 在这里集中登记 — cli / web / 任何上层都 import 这一处。

为什么集中:
- cli 和 web 是兄弟层 (互不能 import), 必须共享通过更下层
- 单一真相来源, 所有 flag 一处可见
- import 多次安全 (register 是幂等覆盖)

策略级 flag 不放这里 — 直接在 strategy class 的 `params` 里定义,
WFO 自动可调 (trial.suggest_categorical)。
"""

from __future__ import annotations

from floatshare.observability import features

# ============================================================================
# sync — daily-sync.sh 等用
# ============================================================================

features.register(
    "auto_analyze",
    description="sync 完成后跑 SQLite ANALYZE",
    impact="DB 多 1-2 分钟; 后续 COUNT(*)/JOIN 查询略快 (优化器统计更新)",
    category="sync",
    default=False,
)

features.register(
    "verbose_section",
    description="每个 sync section 完成单独推一条 Bark",
    impact="每天多 ~10 条推送; 适合调试期监控分阶段进度",
    category="sync",
    conflicts=("quiet_mode",),
)

features.register(
    "quiet_mode",
    description="完全关闭 sync 期间的 Bark 推送",
    impact="只最终成功/失败时推 1 条; 中间过程静默",
    category="sync",
    conflicts=("verbose_section",),
)

features.register(
    "skip_chip_dist",
    description="sync 时跳过 chip_dist (筹码分布, 数据量大)",
    impact="单只股票省 ~50 行/天, 全 A 股省 ~3000 万行/年; chip 信号不可用",
    category="sync",
    default=True,  # 默认就跳过 — 4 亿行级别太重
)


# ============================================================================
# web — Dash 看板用
# ============================================================================

features.register(
    "show_beta_panel",
    description="账户 tab 显示 beta 实验卡片",
    impact="多一个'β' 区块, 内部仍是 placeholder, 占空间",
    category="web",
)

features.register(
    "auto_refresh_counts",
    description="表行数自动 60s 刷新 (默认手动 ⟳)",
    impact="每分钟跑一次 ~25s 的 COUNT(*) 重算, 占 CPU; web 不会卡因为有缓存",
    category="web",
    requires=("auto_analyze",),  # 自动刷新依赖 auto_analyze 让查询变快才合理
)


# ============================================================================
# strategy — 框架级共享 (单策略私有 flag 走 params)
# ============================================================================

features.register(
    "tune_with_pbo",
    description="WFO 调参时计算 PBO (过拟合概率)",
    impact="每次调参多 ~5x 算力 (CCV 切分); 输出 PBO 概率帮判断是否真信号",
    category="strategy",
)

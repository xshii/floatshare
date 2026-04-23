# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` 已同步重写，与当前代码一致。本文件补充 README 不讲的内部约束（分层契约细节、组合根、feature flag 跨进程语义、ruff/mypy 特例豁免等）。

## Commands

安装假设 `.venv/` 已建好，`pip install -e ".[all]"` 后以下 entry points 可用：

- `floatshare-backtest` — 跑策略回测（见 `src/floatshare/cli/run_backtest.py`）
- `floatshare-sync` — 增量同步数据到 SQLite（`src/floatshare/cli/run_sync.py`）
- `floatshare-healthcheck` — DB / 数据完整性体检
- `floatshare-web` — Dash 看板（默认 `0.0.0.0:8050`）

### 质量门（静态扫描全覆盖）

```bash
ruff check src tests strategies
ruff format --check src tests strategies
mypy src/floatshare strategies
pyright                                    # 与 Pylance 同核
lint-imports                               # 分层架构契约 (import-linter)
pytest                                     # 全量测试
pytest tests/test_application/test_backtest.py::test_name  # 单用例
pytest -k "sync and not slow"              # 按名字过滤
```

`lint-imports` 读 `pyproject.toml` 里的 `[tool.importlinter]` 契约，**任何违反分层的 import 都会 fail**，是本仓库最重要的结构检查。

### 每日 sync（macOS launchd）

`scripts/install-daily-sync.sh` 装 `scripts/com.floatshare.daily-sync.plist`。入口脚本 `scripts/daily-sync.sh` 分两步：`floatshare-sync --all-stocks --end $TODAY` + `floatshare-sync --include top_list top_inst --start $TODAY --end $TODAY`。日志进 `logs/daily-sync-YYYY-MM-DD.log`。

### Git hooks

`.githooks/pre-commit` 拦截 `.env` 文件和疑似 token（tushare / openai / github / 硬编码密码）。**启用方式**：
```bash
git config core.hooksPath .githooks
```

## Architecture

### 分层（DDD / 六边形）

`pyproject.toml` 的 import-linter 契约强制以下单向流向（下层不可 import 上层，同层互斥靠契约限制）：

```
cli | web                                   ← 入口 / 表示层
    ↓
application                                 ← 用例编排（组合根，只有它可 import infrastructure）
    ↓
infrastructure                              ← 适配器实现（data_sources/, storage/, broker/）
    ↓
factors | registry | analytics              ← 领域服务 / 工具（禁止反向依赖 application/infrastructure）
    ↓
interfaces | observability                  ← 端口（Protocol/ABC）+ 日志/告警/feature-flags
    ↓
domain                                      ← 纯领域核心（enums, schema, records, trading），零 floatshare 依赖
```

`domain/` 的纯度由 "Domain forbidden modules" 契约显式保证——任何 domain 反向引用都会被 lint-imports 报错。

### Composition root

- 包入口 `src/floatshare/__init__.py` re-export 顶层 API（`register`, `run_backtest`, `create_default_loader`, `logger`, `notify`）。同一文件最顶部 `_load_dotenv()`，所以 `TUSHARE_TOKEN` / `FLOATSHARE_NOTIFY_URLS` 等不必 `export`。因此该文件 ruff 豁免 `E402`。
- `application.create_default_loader()` 是**唯一**允许 `import infrastructure` 的位置（cli 作为入口也算组合根）。其它上层要数据源，走 `interfaces.data_source.*` Protocol + 依赖注入。

### 数据源降级链

`DataLoader._try_chain` 按顺序试每个源，`DataSourceError` 静默降级，全链路失败才抛 `AllSourcesFailed`。默认链（见 `application/data_loader.py`）：

```
Cache → SQLite(local_db) → Tushare → AKShare → EastMoney
```

成功时可选的 `on_daily_fetched` / `on_stock_list_fetched` 回调用于 writeback 到本地 DB。测试里 `tests/conftest.py::FakeDataSource` 可按 `fail_modes` 模拟任意源失败。

### 策略注册 + 自动发现

`src/floatshare/registry.py` 是模块级单例（`_REGISTRY: dict[str, type]`），装饰器 `@register("name")` 登记。项目根 `strategies/` 目录下的 `ma_cross.py` / `dual_thrust.py` / `sector_relay.py` 继承 `bt.Strategy`，写 `@register(...)`。

`discover("strategies")` 用 `pkgutil.iter_modules` 递归 import 触发副作用——cli 和 web 都是这样把策略装载起来。注意：策略在项目根的 `strategies/`，**不是** `src/floatshare/strategies/`；`pyproject.toml::tool.pytest.ini_options.pythonpath = ["src", "."]` 让两边都能 import。

策略参数用 backtrader `params = (...)`；可选 `search_space(cls, trial)` 类方法声明 Optuna 搜索空间，供 WFO 框架 (`application/optimization/`) 调参。

### 新闻联播 NLP (CCTV news → SW L1 mentions)

`data/news/industry_keywords.json` 是**单一真相**的 SW L1 关键词词典（30 行业，删"综合"），每月维护一次；`data/news/industry_baseline.json` 是 90 天 rolling IDF 基线，`build_news_baseline.py` 生成。`infrastructure/nlp/cctv_local.py::extract_industry_mentions` 算 `raw_score = hits / n_kws` 和 `weighted_score = raw_score × idf`，冷门行业被提及时自动放大权重（美容护理 idf 4.1 vs 社会服务 idf 1.0）。

**每月 NLP review 工作流**（第 1 个交易日，人工 + 脚本）：

1. **采样 1 周 corpus**：挑最近 5-7 个交易日，手动 `python -c "..."` 看原文（不运行在 pipeline 里，人眼判断够不够变化）
2. **跑 keyword audit**：对低覆盖 L1 逐词 `corpus.count(kw)`，识别死词 / 漏词（参考 `scripts/build_news_baseline.py` 的逐词统计风格）
3. **改词库**：直接编辑 `data/news/industry_keywords.json`，更新 `_meta.version` 为 `v(n+1)-YYYY-MM-DD`
4. **重算 baseline**：`python scripts/build_news_baseline.py --days 90`
5. **回填 mentions**：`python scripts/backfill_cctv_mentions.py --rebuild-mentions-only --start <训练起点>` — 用新词库 + 新 baseline 重算历史所有 mentions（不碰 tushare API）

**新接训练时的一次性 bootstrap**：
```bash
python scripts/backfill_cctv_mentions.py --start 2023-01-01  # 从训练起点一路拉到今天 + 算 mentions
```
脚本幂等：重跑不会拉重复 API（已有 raw 直接用库里），只重算 mentions。

**训练时的约定**：直接 join `cctv_news_mentions` 表，用当前词库 / 当前 baseline 的产物——**训练不跑 NLP**。Pipeline 的 `stage_s1c_news_ingest` 只处理 T 日实时 ingest，历史数据走 backfill 脚本。

### ML 训练方法论 (anchored WFV + warm-start 链)

**时间切分** (`src/floatshare/ml/config.py::DataConfig`):
```
train: 2018-01-01 ~ 2024-12-31  (7 年, anchor 固定 2018 — tushare cctv_news 起点)
val:   2025-01-01 ~ 2025-03-31  (3 月, 紧贴 train 尾)
test:  2025-04-01 ~ 2026-03-31  (1 年, 过一轮周期)
live:  2026-04+  (S4 daily warm-start 接管)
```

**两级训练**:
- **Cold-start** (季度手动): 从零训 30+ epoch, 产出 anchor ckpt — `floatshare-train-pop --epochs 30`
- **Warm-start** (每日自动): `stage_s4_train` warm 3 epoch, `val_window_days=60` 滚动 val, 防 drift
- 对齐 AQR / Man-AHL / DeepSeek-Math 等业界惯例 (anchored expanding window + chained warm-start)

**Walk-Forward Validation** (`scripts/walk_forward_eval.py`):
月度滚动, 每步 train 到 M-1, val 在 M. 两种模式:
- 默认 warm-start 链: 每步 3 epochs, 产线等价
- `--cold` cold retrain: 每步 30 epoch, 学术/论文基线

输出: `data/ml/wfv_results/<uuid>/metrics.csv` + `summary.json` (mean_auc / std_auc / n_steps).

**RL 算法选择**: PPO (`training/ppo.py`) 与 GRPO (`training/grpo.py`) 二选一:
- PPO: 经典 actor-critic + GAE, value head 估计 V(s) — 金融 V(s) 高方差, tuning 成本高
- **GRPO (推荐)**: critic-free, 每 state 采 G=8 actions 组内归一化 baseline — 更适合奖励稀疏的涨停任务
- 两者共享 MarketEnv, 数据管道一致, 可并行跑对比

### Feature flags 框架

`src/floatshare/observability/features.py` 是注册式 flag 系统：

- `register(name, description, impact=, category=, default=, requires=, conflicts=)`
- 开关方式优先级：`FLOATSHARE_FEATURE_<NAME>=1` > `FLOATSHARE_FEATURES=a,b,c` > `default`
- 集中登记在 `src/floatshare/application/feature_registry.py`（cli 和 web 都 `import` 这一处以触发注册）
- `validate_registry()` 启动查注册表自身（循环依赖、未登记引用）；`validate_enabled()` 查运行时启用集合（requires/conflicts）
- **注意**：`requires` 当前是同进程同 env 的强约束。web 和 sync 跑在不同进程时，跨进程 `requires` 语义不成立——加新 flag 做跨进程依赖要留心。
- 策略级私有 flag **不走**这里，直接写进 backtrader `params`，WFO 自动可搜。

### Domain schema & records

`domain/records/` 下是各类市场/财务数据的 `RecordSchema` Protocol 实现（`daily_price`, `daily_flow`, `daily_chip`, `daily_macro`, `quarterly`, `monthly`, `reference`, `event`, `book`, `ops`）。`domain/schema.py` 定义 OHLCV 必需/可选列与 normalize 工具。新增数据类型时在这里加 `RecordSchema`，同步器 (`application/data_syncer.py`) 按 schema 驱动存 SQLite。

### Web 看板

`src/floatshare/web/app.py` 只组装 4 个 tab（账户/策略/同步/开关），实际 UI 拆到 `layouts/` + `callbacks/`。web 启动时会 `import strategies` 把项目根的策略注册进来。

## Conventions

- Python 3.11+（`requires-python = ">=3.11"`；mypy/pyright 都按 `py311` 配）
- `backtrader` 用本地 `stubs/backtrader/__init__.pyi` 提供类型，无需 `ignore_missing_imports`
- ruff 配 `target-version = "py311"` 和 `UP` 规则——**必须用新式 type hint**（但用户全局偏好 `Optional[X]` 而非 `X | None`，以个人偏好为准，ruff 不强制）
- `__init__.py` 的 `F401` 已豁免；re-export 用 `from .x import y as y` 显式标注
- `tests/` 目录豁免 `ARG`、`PLR2004`、`S101`
- `data/`、`logs/`、`notebooks/`、`prettycli/` 被 ruff 全量排除

## 可选依赖组

`[dev]` / `[tushare]` / `[web]` / `[tune]` / `[all]`。Tushare、optuna、dash 都是可选——没装对应 extras 时相关 feature 应该优雅降级或给清晰错误。

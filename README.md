# FloatShare

个人 A 股量化交易框架 — 数据同步、策略回测、组合跟踪、可视化看板。

## 特性

- **多源数据** 带降级链：`Cache → SQLite → Tushare → AKShare → EastMoney`，失败静默切下一个
- **backtrader 回测引擎**，策略用 `@register` 一行装饰即登记
- **DDD / 六边形分层**，`import-linter` 契约强制单向依赖，结构回归会在 CI 挂掉
- **Feature flag 框架**：env 开关 + 依赖/互斥约束 + 启动校验
- **超参 WFO**：Optuna + walk-forward，策略写个 `search_space` 即可参与
- **Dash 看板**：账户 / 回测 / 同步进度 / 功能开关 4 个 tab
- **每日自动同步**：macOS launchd plist，增量拉取 + Bark 推送

## 安装

要求 Python 3.11+。

```bash
git clone <repository-url>
cd floatshare
git submodule update --init --recursive    # prettycli

python -m venv .venv
source .venv/bin/activate

pip install -e ".[all]"                     # dev + tushare + web + tune
# 或按需: pip install -e ".[dev]"  /  ".[web]"  /  ".[tushare]"
```

配置环境变量（可选）：

```bash
cp .env.example .env
# 填 TUSHARE_TOKEN（付费数据源）
# 填 FLOATSHARE_NOTIFY_URLS（apprise 告警，示例见文件注释）
```

启用 git hooks 防 secret 入库：

```bash
git config core.hooksPath .githooks
```

## 快速上手

### 命令行入口

```bash
floatshare-sync --all-stocks --end 2026-04-17     # 增量同步到 SQLite
floatshare-backtest --strategy ma_cross \         # 跑策略
    --codes 600000.SH 000001.SZ \
    --start 2023-01-01 --end 2024-12-01 \
    --capital 1000000 --params short_period=5 long_period=20
floatshare-healthcheck                             # DB 体检
floatshare-web --port 8050                         # 启看板
```

### 写一个策略

策略放在项目根 `strategies/` 下，继承 `backtrader.Strategy`，用 `@register` 登记：

```python
# strategies/my_strategy.py
import backtrader as bt
from floatshare import register

@register("my_strategy")
class MyStrategy(bt.Strategy):
    params = (("period", 20), ("position_pct", 0.9))

    @classmethod
    def search_space(cls, trial):        # 可选，WFO 会用
        return {
            "period": trial.suggest_int("period", 5, 60),
            "position_pct": trial.suggest_float("position_pct", 0.3, 1.0),
        }

    def __init__(self):
        self.sma = {d._name: bt.indicators.SMA(d.close, period=self.p.period) for d in self.datas}

    def next(self):
        for d in self.datas:
            ...
```

CLI 和看板会通过 `discover("strategies")` 自动 import，触发 `@register` 副作用。

### 用 Python API

```python
from floatshare import create_default_loader, run_backtest, discover, get

discover()                               # 扫描 strategies/
loader = create_default_loader()         # 带降级链 + 本地 writeback
data = loader.get_daily("600000.SH", start=date(2023,1,1), end=date(2024,12,1))

strategy_cls = get("ma_cross")
result = run_backtest(strategy_cls, data, initial_capital=1_000_000)
result.print_summary()
```

## 架构

分层严格单向流（`pyproject.toml` 里 `import-linter` 契约强制，`lint-imports` 检查）：

```
cli | web                    ← 入口
    ↓
application                  ← 用例编排（组合根，唯一可 import infrastructure 的层）
    ↓
infrastructure               ← 适配器实现（data_sources / storage / broker）
    ↓
factors | registry | analytics
    ↓
interfaces | observability   ← Protocol/ABC 端口 + 日志告警 feature-flags
    ↓
domain                       ← 纯领域（enums / schema / records / trading）
```

- `domain/` 零 `floatshare` 内部依赖，由 forbidden-modules 契约显式保证
- 上层拿数据源走 `interfaces.data_source.*` 的 Protocol + 依赖注入，不直接 import 具体实现
- `application.create_default_loader()` 是唯一的组合根，把 infrastructure 装配成 `DataLoader`

### Feature flags

全局开关登记在 `src/floatshare/application/feature_registry.py`（cli 和 web 都 import 一次触发注册）：

```bash
export FLOATSHARE_FEATURE_AUTO_ANALYZE=1                   # 单独开
export FLOATSHARE_FEATURES=auto_analyze,verbose_section    # 多开
python -c "from floatshare.observability import features; features.print_flags()"
```

策略私有的开关不走这里 — 直接写进 backtrader `params`，WFO 自动可搜。

## 开发

### 质量门

```bash
ruff check src tests strategies
ruff format --check src tests strategies
mypy src/floatshare strategies
pyright                                    # Pylance 同核
lint-imports                               # 分层架构契约
pytest                                     # 全量测试
pytest tests/test_application/test_backtest.py::test_name
pytest -k "sync and not slow"
```

### 每日 sync（macOS launchd）

```bash
bash scripts/install-daily-sync.sh     # 安装 launchd agent
bash scripts/daily-sync.sh             # 手动跑一次
tail -f logs/daily-sync-$(date +%F).log
```

入口脚本分两步：`floatshare-sync --all-stocks --end $TODAY` → `floatshare-sync --include top_list top_inst --start $TODAY --end $TODAY`。

## 可选依赖组

| Extra | 用途 |
| --- | --- |
| `dev` | pytest / ruff / mypy / pyright / import-linter |
| `tushare` | Tushare 数据源（需 token） |
| `web` | Dash 看板 |
| `tune` | Optuna WFO |
| `all` | 上面四个全装 |

未装对应 extras 时相关功能会优雅降级或给出清晰错误。

## 许可证

MIT

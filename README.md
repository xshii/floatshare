# FloatShare

股票量化交易系统

## 项目简介

FloatShare 是一个功能完整的 A 股量化交易框架，提供数据管理、策略开发、回测分析、风险控制等核心功能。

## 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  数据管理    │────▶│   策略系统   │────▶│  交易执行   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   回测系统   │     │  风险管理   │     │  账户管理   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  监控告警 + 绩效分析     │
              └─────────────────────────┘
```

## 核心模块

| 模块 | 路径 | 功能 |
|------|------|------|
| 数据管理 | `src/data/` | 多数据源支持、数据清洗、存储 |
| 策略系统 | `src/strategy/` | 策略基类、因子库、信号生成 |
| 回测系统 | `src/backtest/` | 回测引擎、撮合、报告生成 |
| 交易执行 | `src/execution/` | 订单管理、持仓管理、券商接口 |
| 风险管理 | `src/risk/` | 风控限制、敞口计算 |
| 账户管理 | `src/account/` | 组合管理、交易记录 |
| 绩效分析 | `src/analysis/` | 绩效指标、归因分析、可视化 |
| 监控告警 | `src/monitor/` | 健康检查、告警、日志 |

## 快速开始

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd floatshare

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 或使用 pyproject.toml
pip install -e .
```

### 初始化数据库

```bash
python scripts/init_db.py
```

### 获取数据

```bash
# 获取示例股票数据
python scripts/fetch_data.py --source akshare --codes 000001.SZ 600000.SH

# 获取更多股票
python scripts/fetch_data.py --source akshare --all --start 2023-01-01
```

### 运行回测

```bash
# 使用均线交叉策略回测
python scripts/run_backtest.py \
    --strategy ma_cross \
    --codes 000001.SZ 600000.SH \
    --start 2023-01-01 \
    --end 2024-12-01 \
    --capital 1000000 \
    --params short_period=5 long_period=20

# 使用 DualThrust 策略
python scripts/run_backtest.py \
    --strategy dual_thrust \
    --codes 000001.SZ \
    --start 2023-01-01
```

## 开发自定义策略

```python
from src.strategy.base import Strategy, Signal, StrategyContext
from src.strategy.registry import StrategyRegistry

@StrategyRegistry.register("my_strategy")
class MyStrategy(Strategy):
    name = "MyStrategy"

    def init(self, context: StrategyContext) -> None:
        self.log("策略初始化")

    def handle_data(self, context, data) -> list:
        signals = []
        # 实现你的策略逻辑
        return signals
```

## 数据源

支持以下数据源：

- **AKShare** (免费): 默认数据源，无需注册
- **Tushare** (需Token): 更稳定，需要在 `.env` 中配置 `TUSHARE_TOKEN`

## 目录结构

```
floatshare/
├── config/           # 配置模块
├── src/              # 核心源码
│   ├── data/         # 数据管理
│   ├── strategy/     # 策略系统
│   ├── backtest/     # 回测系统
│   ├── execution/    # 交易执行
│   ├── risk/         # 风险管理
│   ├── account/      # 账户管理
│   ├── analysis/     # 绩效分析
│   └── monitor/      # 监控告警
├── strategies/       # 策略实现
├── scripts/          # 脚本工具
├── tests/            # 测试
├── data/             # 数据目录
├── logs/             # 日志目录
└── notebooks/        # Jupyter笔记本
```

## 运行测试

```bash
pytest tests/ -v
```

## 依赖

本项目使用 git submodule 管理部分依赖：

- [prettycli](https://github.com/xshii/prettycli) - CLI 美化工具

```bash
git submodule update --init --recursive
```

## 许可证

MIT

## 贡献

欢迎贡献代码和提出建议！

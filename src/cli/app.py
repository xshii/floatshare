"""FloatShare CLI 主入口

使用 Typer 构建命令行接口
"""

import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# 创建 Typer 应用
app = typer.Typer(
    name="floatshare",
    help="FloatShare - A股量化交易框架",
    add_completion=False,
)

console = Console()


# ============================================================
# 数据命令组
# ============================================================

data_app = typer.Typer(help="数据管理命令")
app.add_typer(data_app, name="data")


@data_app.command("sync")
def data_sync(
    source: str = typer.Option("akshare", "--source", "-s", help="数据源"),
    start_date: str = typer.Option(None, "--start", help="开始日期 (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, "--end", help="结束日期 (YYYY-MM-DD)"),
    codes: Optional[List[str]] = typer.Option(None, "--code", "-c", help="股票代码"),
    priority: str = typer.Option("hs300", "--priority", "-p", help="优先级: hs300/zz500/all"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="是否断点续传"),
):
    """同步股票数据"""
    from src.data.syncer import DataSyncer, SyncPriority

    console.print(f"[bold blue]开始数据同步[/bold blue]")
    console.print(f"  数据源: {source}")
    console.print(f"  日期范围: {start_date or '1990-01-01'} ~ {end_date or '今天'}")

    # 解析日期
    s_date = date.fromisoformat(start_date) if start_date else None
    e_date = date.fromisoformat(end_date) if end_date else None

    # 解析优先级
    priority_map = {
        "hs300": SyncPriority.HS300,
        "zz500": SyncPriority.ZZ500,
        "zz1000": SyncPriority.ZZ1000,
        "all": SyncPriority.ALL,
    }
    priorities = [priority_map.get(priority, SyncPriority.HS300)]

    syncer = DataSyncer(source=source)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("同步中...", total=None)

        def on_progress(completed: int, total: int, code: str):
            progress.update(task, description=f"[{completed}/{total}] {code}")

        syncer.on_progress(on_progress)

        result = syncer.sync_daily(
            priorities=priorities,
            start_date=s_date,
            end_date=e_date,
            resume=resume,
        )

    # 显示结果
    table = Table(title="同步结果")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")

    table.add_row("总数", str(result["total"]))
    table.add_row("成功", str(result["completed"]))
    table.add_row("失败", str(result["failed"]))
    table.add_row("跳过", str(result["skipped"]))
    table.add_row("新增数据", f"{result['total_rows']} 条")

    console.print(table)


@data_app.command("status")
def data_status():
    """查看同步状态"""
    from src.data.syncer import DataSyncer

    syncer = DataSyncer()
    progress = syncer.get_sync_progress()

    if progress is None:
        console.print("[yellow]没有进行中的同步任务[/yellow]")
        return

    table = Table(title="同步进度")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")

    for key, value in progress.items():
        table.add_row(key, str(value))

    console.print(table)


@data_app.command("clear")
def data_clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="确认清除"),
):
    """清除同步状态"""
    if not confirm:
        if not typer.confirm("确定要清除同步状态吗？"):
            raise typer.Abort()

    from src.data.syncer import DataSyncer

    syncer = DataSyncer()
    syncer.clear_state()
    console.print("[green]同步状态已清除[/green]")


@data_app.command("health")
def data_health():
    """检查数据源健康状态"""
    console.print("[bold]检查数据源健康状态...[/bold]")

    sources = ["akshare", "baostock"]
    results = []

    for source in sources:
        try:
            from src.data.loader import DataLoader
            loader = DataLoader(source=source)

            # 测试获取数据
            from datetime import date, timedelta
            end = date.today()
            start = end - timedelta(days=7)

            df = loader.get_daily("000001.SZ", start, end)

            if not df.empty:
                results.append((source, "✅ 正常", f"{len(df)} 条"))
            else:
                results.append((source, "⚠️ 空数据", "-"))

        except Exception as e:
            results.append((source, "❌ 失败", str(e)[:30]))

    table = Table(title="数据源状态")
    table.add_column("数据源", style="cyan")
    table.add_column("状态", style="green")
    table.add_column("详情")

    for source, status, detail in results:
        table.add_row(source, status, detail)

    console.print(table)


# ============================================================
# 回测命令组
# ============================================================

backtest_app = typer.Typer(help="回测命令")
app.add_typer(backtest_app, name="backtest")


@backtest_app.command("run")
def backtest_run(
    strategy: str = typer.Argument(..., help="策略名称"),
    start_date: str = typer.Option("2020-01-01", "--start", "-s", help="开始日期"),
    end_date: str = typer.Option(None, "--end", "-e", help="结束日期"),
    capital: float = typer.Option(1_000_000, "--capital", "-c", help="初始资金"),
    benchmark: str = typer.Option("000300.SH", "--benchmark", "-b", help="基准指数"),
):
    """运行回测"""
    console.print(f"[bold blue]开始回测[/bold blue]")
    console.print(f"  策略: {strategy}")
    console.print(f"  日期: {start_date} ~ {end_date or '今天'}")
    console.print(f"  初始资金: {capital:,.0f}")
    console.print(f"  基准: {benchmark}")

    # TODO: 实现回测逻辑
    console.print("[yellow]回测功能开发中...[/yellow]")


@backtest_app.command("list")
def backtest_list():
    """列出可用策略"""
    from src.strategy.registry import StrategyRegistry

    strategies = StrategyRegistry.list_strategies()

    if not strategies:
        console.print("[yellow]没有注册的策略[/yellow]")
        return

    table = Table(title="可用策略")
    table.add_column("名称", style="cyan")
    table.add_column("描述")

    for name, info in strategies.items():
        table.add_row(name, info.get("description", "-"))

    console.print(table)


# ============================================================
# 系统命令
# ============================================================


@app.command("init")
def init_db():
    """初始化数据库"""
    from src.data.storage.database import DatabaseStorage

    console.print("[bold]初始化数据库...[/bold]")

    storage = DatabaseStorage()
    storage.init_tables()

    console.print("[green]数据库初始化完成[/green]")


@app.command("version")
def version():
    """显示版本信息"""
    console.print("[bold]FloatShare[/bold] v0.1.0")
    console.print("A股量化交易框架")


@app.command("config")
def show_config():
    """显示当前配置"""
    try:
        from config.base import get_settings

        settings = get_settings()

        table = Table(title="当前配置")
        table.add_column("配置项", style="cyan")
        table.add_column("值", style="green")

        table.add_row("数据目录", str(settings.data_dir))
        table.add_row("日志目录", str(settings.logging.log_dir))
        table.add_row("日志级别", settings.logging.level.value)
        table.add_row("主数据源", settings.data_source.primary_source)
        table.add_row("缓存后端", settings.cache.backend)
        table.add_row("初始资金", f"{settings.backtest.initial_capital:,.0f}")

        console.print(table)

    except Exception as e:
        console.print(f"[red]加载配置失败: {e}[/red]")


# ============================================================
# 入口点
# ============================================================


def main():
    """CLI 入口点"""
    app()


if __name__ == "__main__":
    main()

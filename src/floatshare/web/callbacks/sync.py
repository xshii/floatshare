"""同步/健康 tab callbacks — 进度卡 + 日志 + 表行数 + 日热图 + 手动拉取。"""

from __future__ import annotations

import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html

from floatshare.application.sync_progress import SyncProgress, SyncStatus
from floatshare.web.components import ANSI_RE, feedback, fmt_duration
from floatshare.web.data import daily_status_cells, table_counts

# severity → Bootstrap CSS 类 (Cookbook style: 数据语义在 dataclass, UI 映射在这)
_SEVERITY_CSS: dict[str, str] = {
    "ok": "table-success",
    "warn": "table-warning",
    "bad": "table-danger",
    "empty": "table-light",
}

# SyncStatus → (Bootstrap color, label) — 单一映射表，Cookbook Recipe 8.10 风格
_STATUS_THEME: dict[SyncStatus, tuple[str, str]] = {
    SyncStatus.RUNNING: ("info", "🔄 进行中"),
    SyncStatus.DONE: ("success", "✅ 完成"),
    SyncStatus.FAILED: ("danger", "⚠️ 完成 (有失败)"),
    SyncStatus.STALE: ("warning", "💀 中断 (pid 已消失)"),
}


@callback(
    Output("sync-progress-card", "children"),
    Output("sync-log", "children"),
    Input("sync-tick", "n_intervals"),
)
def update_sync_progress(_n: int | None) -> tuple:
    """快 tick (5s) — 进度卡 + 日志，sync 运行时高频更新。"""
    return _render_progress_card(SyncProgress.read()), _tail_today_log()


@callback(
    Output("sync-counts", "children"),
    Output("counts-updated-at", "children"),
    Input("counts-refresh", "n_clicks"),
)
def update_sync_counts(n_clicks: int | None) -> tuple:
    """初始加载读 snapshot (instant)；点刷新按钮强制重算 (~25s)。"""
    from floatshare.application.db_snapshot import COUNTS_SNAPSHOT_PATH

    if n_clicks:
        from floatshare.application.db_snapshot import refresh_counts_snapshot
        from floatshare.web import data as _d

        refresh_counts_snapshot(_d.DB)
        _d._counts_cache = None
    df = table_counts()
    table = dbc.Table.from_dataframe(df, striped=True, size="sm")
    # 用 snapshot 文件 mtime 作为"真实数据时间"，比 callback 触发时间更准
    if COUNTS_SNAPSHOT_PATH.exists():
        mtime = datetime.fromtimestamp(COUNTS_SNAPSHOT_PATH.stat().st_mtime)
        badge = f"@ {mtime.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        badge = "@ —"
    return table, badge


@callback(
    Output("daily-status-grid", "children"),
    Input("daily-status-refresh", "n_clicks"),
)
def update_daily_status(_n: int | None) -> Any:
    """近一周 × 6 个 daily 表的 code 覆盖数热图。"""
    cells = daily_status_cells(days=7)
    if not cells:
        return dbc.Alert("无数据", color="secondary")

    days = sorted({c.day for c in cells})
    tables = sorted({c.table for c in cells})
    grid = {(c.table, c.day): c for c in cells}

    head = html.Thead(
        html.Tr(
            [
                html.Th("表"),
                *(html.Th(d.strftime("%m-%d")) for d in days),
            ]
        )
    )
    body_rows = []
    for t in tables:
        row_cells = [html.Td(t, className="font-monospace")]
        for d in days:
            cell = grid.get((t, d))
            if cell is None:
                row_cells.append(html.Td("—", className="table-light text-end"))
                continue
            css = _SEVERITY_CSS[cell.severity]
            text = f"{cell.code_count:,}" if cell.code_count else "—"
            row_cells.append(html.Td(text, className=f"{css} text-end font-monospace"))
        body_rows.append(html.Tr(row_cells))
    return dbc.Table(
        [head, html.Tbody(body_rows)], bordered=True, hover=True, size="sm", className="small"
    )


@callback(
    Output("pull-status", "children"),
    Input("pull-trigger", "n_clicks"),
    State("pull-start", "date"),
    State("pull-end", "date"),
    State("pull-types", "value"),
    prevent_initial_call=True,
)
def trigger_manual_pull(
    _n: int | None,
    start: str | None,
    end: str | None,
    types: list[str] | None,
) -> Any:
    """点 [📥 拉取] → 后台 subprocess.Popen 跑 floatshare-sync。"""
    if not start or not end:
        return feedback("请填写起止日期", "warning")
    cmd = ["floatshare-sync", "--all-stocks", "--start", start, "--end", end]
    if types:
        cmd += ["--include", *types]
    log_path = Path(f"logs/manual-sync-{datetime.now():%Y%m%d-%H%M%S}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # 脱离 web 进程组，web 重启不影响
        )
    except Exception as exc:
        return feedback(f"启动失败: {exc}")
    return dbc.Alert(
        f"✓ 已启动 PID {proc.pid} → 日志 {log_path.name}; 进度看上方进度卡 (5s 自动刷新)",
        color="success",
        className="mb-0",
    )


def _tail_today_log() -> str:
    """优先读今日 daily-sync 日志，没有则尝试 sync_wave_a.log。strip ANSI 颜色码。"""
    candidates = [
        Path(f"logs/daily-sync-{date.today().isoformat()}.log"),
        Path("logs/sync_wave_a.log"),
    ]
    log_path = next((p for p in candidates if p.exists()), None)
    if log_path is None:
        return "(无 sync 日志)"
    try:
        result = subprocess.run(
            ["tail", "-50", str(log_path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return ANSI_RE.sub("", result.stdout)
    except Exception:
        return "(读取日志失败)"


def _render_progress_card(p: SyncProgress | None) -> Any:
    """根据 SyncProgress 渲染顶部进度卡。"""
    if p is None:
        return dbc.Alert(
            "尚未触发任何 sync。每天 19:00 自动启动 (launchd)。",
            color="secondary",
            className="mb-0",
        )

    # 用 effective_status: 如果文件停在 RUNNING 但 pid 已死 → STALE
    eff = p.effective_status
    status_enum = SyncStatus(eff) if eff in SyncStatus._value2member_map_ else SyncStatus.RUNNING
    color, status_label = _STATUS_THEME.get(status_enum, ("secondary", eff))
    eta = fmt_duration(p.eta_seconds) if p.eta_seconds else "—"
    started = p.started_at[11:19] if p.started_at else "?"
    finished = p.finished_at[11:19] if p.finished_at else ""

    is_alive = p.is_running  # 已综合 status + pid 存活
    progress_bar = dbc.Progress(
        value=p.percent,
        label=f"{p.percent}% ({p.codes_done}/{p.codes_total})",
        color=color,
        animated=is_alive,
        striped=is_alive,
        style={"height": "24px"},
    )

    meta_row = dbc.Row(
        [
            dbc.Col([html.Small("状态"), html.Div(status_label, className="fw-bold")], width=2),
            dbc.Col([html.Small("阶段"), html.Div(p.stage)], width=2),
            dbc.Col(
                [
                    html.Small("当前 code"),
                    html.Div(p.current_code or "-", className="font-monospace"),
                ],
                width=2,
            ),
            dbc.Col([html.Small("已运行"), html.Div(fmt_duration(p.elapsed_seconds))], width=2),
            dbc.Col([html.Small("预计剩余"), html.Div(eta)], width=2),
            dbc.Col(
                [
                    html.Small("错误"),
                    html.Div(str(p.errors), className="text-danger fw-bold" if p.errors else ""),
                ],
                width=2,
            ),
        ],
        className="g-2 mb-2",
    )

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Span(f"开始 {started}", className="text-muted small me-2"),
                            html.Span(f"完成 {finished}", className="text-muted small")
                            if finished
                            else "",
                        ]
                    ),
                    meta_row,
                    progress_bar,
                ]
            ),
        ],
        color=color,
        outline=True,
    )

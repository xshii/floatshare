"""跑板块接力策略回测 — 10w 初始资金 / T+1 / 0.1% 滑点 / A 股标准费率。

数据: SW L1 31 行业 index_daily (假设可零成本 ETF 跟踪)
窗口: 2022-01-01 ~ 2024-12-31  (避开 EDA 训练期 2018-2021, 减少 in-sample bias)

输出:
  - 控制台 print: 总盈亏/年化/最大回撤/Sharpe/换手
  - notebooks/output/sector_relay_metrics.json
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

# 让 strategies/ 顶层目录可被 import (notebooks/ 不在 src 下)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import text

from floatshare.application.backtest import run_backtest
from floatshare.domain.trading import TradingConfig
from floatshare.infrastructure.storage.database import DatabaseStorage
from floatshare.registry import discover, get
from strategies.sector_relay import SW_L1

OUT = Path("notebooks/output")
OUT.mkdir(parents=True, exist_ok=True)


# 指数点位 → ETF-等价价格的归一化系数
# 申万指数 ~3000 点 → 仿真 3 元 / 沪深300 ~4000 → 4 元 (跟实际 ETF 价位吻合)
# 让 100 股一手 + 10w 资金能跑通仿真，逻辑上 = "假装在交易对应的行业 ETF"
_INDEX_PRICE_DIVISOR = 1000.0


def load_sw_l1_panel(start: date, end: date) -> pd.DataFrame:
    """从 index_daily 拼出 31 行业的 OHLCV panel, 价格归一化到 ETF 量级。"""
    db = DatabaseStorage()
    codes = "', '".join(SW_L1)
    df = pd.read_sql(
        text(f"""
        SELECT code, trade_date, open, high, low, close, volume, amount
        FROM index_daily
        WHERE code IN ('{codes}')
          AND date(trade_date) >= :start
          AND date(trade_date) <= :end
        ORDER BY code, trade_date
    """),
        db.engine,
        params={"start": start.isoformat(), "end": end.isoformat()},
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # 价格归一化 (volume/amount 保持原值, 不影响信号)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] / _INDEX_PRICE_DIVISOR
    return df


def main() -> None:
    discover("strategies")  # 触发 @register
    strategy_cls = get("sector_relay")
    if strategy_cls is None:
        raise SystemExit("✗ 找不到 sector_relay 策略")

    # OOS 测试期 (避免 in-sample bias — EDA 是用 2018-2024 全期算的, 严格说也不算 OOS)
    start = date(2022, 1, 1)
    end = date(2024, 12, 31)

    print(f"=== 加载 SW L1 31 行业 {start} ~ {end} ===")
    data = load_sw_l1_panel(start, end)
    print(f"  rows: {len(data):,}, codes: {data['code'].nunique()}")
    if data.empty:
        raise SystemExit("✗ index_daily 无数据, 先跑 floatshare-sync --include index_daily")

    print("\n=== 跑回测 (100k / T+1 / 0.1% slippage / 万 3 佣金 + 印花 + 过户费) ===")
    cfg = TradingConfig(
        commission_rate=0.0003,  # 万 3 (券商佣金率)
        stamp_duty=0.0005,  # 千 0.5 (2023 年 8 月降税后, 仅卖方)
        transfer_fee=0.00001,  # 沪市过户费 (万 0.1, 双边)
        min_commission=5.0,  # 单笔最低 5 元
        slippage=0.001,  # 千 1 双边 (实战 0.1-0.3% 取中)
    )
    result = run_backtest(
        strategy_cls=strategy_cls,
        data=data,
        initial_capital=100_000,
        trading_config=cfg,
        start_date=start,
        end_date=end,
    )

    print("\n=== 回测结果 ===")
    result.print_summary()

    n_trades = len(result.trades) if not result.trades.empty else 0
    print(f"  交易次数: {n_trades}")

    # 统计换手
    if not result.trades.empty:
        buy_value = result.trades[result.trades["action"] == "buy"]["value"].sum()
        turnover = abs(buy_value) / 100_000  # 总买入金额 / 初始资金 = 总换手倍数
        print(f"  总换手 (buy notional / cap): {turnover:.1f}x")
        print(f"  年化换手: {turnover / 3:.1f}x  (3 年期)")

    # 落盘 metrics
    metrics = {
        "window": f"{start} ~ {end}",
        "initial_capital": 100_000,
        "final_value": result.final_value,
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "n_trades": n_trades,
    }
    out_path = OUT / "sector_relay_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    print(f"\n  ✓ 写入 {out_path}")


if __name__ == "__main__":
    main()

"""行业 Lead-Lag 关系探索 — 找"红旗手 + 跟随者"模式。

输出:
1. 28 申万 L1 行业 × 12 年日收益矩阵
2. 滞后相关热图 (lag = 1, 3, 5, 10, 20 天)
3. 每个行业的 top-5 跟随者 → 沉淀为 LEAD_LAG_MAP dict
4. 涨停联动 lift (大涨日条件下，N 日后跟随板块异动概率提升)

用法:
    python notebooks/sector_lead_lag.py
    # 输出文件:
    #   notebooks/output/sector_corr_lag5.png
    #   notebooks/output/lead_lag_map.json
    #   notebooks/output/contagion_lift.csv
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from floatshare.infrastructure.storage.database import DatabaseStorage

OUT_DIR = Path("notebooks/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 申万 L1 代码 → 中文名 (从 industry 表查也可，写死简单)
SW_L1: dict[str, str] = {
    "801010.SI": "农林牧渔",
    "801030.SI": "基础化工",
    "801040.SI": "钢铁",
    "801050.SI": "有色金属",
    "801080.SI": "电子",
    "801110.SI": "家用电器",
    "801120.SI": "食品饮料",
    "801130.SI": "纺织服饰",
    "801140.SI": "轻工制造",
    "801150.SI": "医药生物",
    "801160.SI": "公用事业",
    "801170.SI": "交通运输",
    "801180.SI": "房地产",
    "801200.SI": "商贸零售",
    "801210.SI": "社会服务",
    "801230.SI": "综合",
    "801710.SI": "建筑材料",
    "801720.SI": "建筑装饰",
    "801730.SI": "电力设备",
    "801740.SI": "国防军工",
    "801750.SI": "计算机",
    "801760.SI": "传媒",
    "801770.SI": "通信",
    "801780.SI": "银行",
    "801790.SI": "非银金融",
    "801880.SI": "汽车",
    "801890.SI": "机械设备",
    "801950.SI": "煤炭",
    "801960.SI": "石油石化",
    "801970.SI": "环保",
    "801980.SI": "美容护理",
}


# ============================================================================
# 1. 读数据 → wide-format 收益矩阵
# ============================================================================


def load_returns(start: date, end: date) -> pd.DataFrame:
    """从 index_daily 读 SW L1 收益, wide-format (date × industry_name)。"""
    db = DatabaseStorage()
    codes = "', '".join(SW_L1)
    query = text(f"""
        SELECT code, trade_date, close
        FROM index_daily
        WHERE code IN ('{codes}')
          AND date(trade_date) >= :start
          AND date(trade_date) <= :end
        ORDER BY code, trade_date
    """)
    df = pd.read_sql(query, db.engine, params={"start": start.isoformat(), "end": end.isoformat()})
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # rename code → name (中文更可读)
    df["name"] = df["code"].map(SW_L1)
    wide_close = df.pivot(index="trade_date", columns="name", values="close")
    return wide_close.pct_change().dropna(how="all")


# ============================================================================
# 2. 滞后相关矩阵 — 找 lead-lag pair
# ============================================================================


def lag_corr_matrix(returns: pd.DataFrame, lag: int) -> pd.DataFrame:
    """corr( A_t, B_{t+lag} ) — A 是 leader, B 是 follower。

    返回 DataFrame: index=leader, columns=follower, value=correlation。
    高值 = A 涨 N 天后 B 跟涨。
    """
    leaders = returns
    followers = returns.shift(-lag)  # follower 时间向前移 lag 天
    aligned = leaders.iloc[:-lag] if lag > 0 else leaders
    followers = followers.iloc[:-lag] if lag > 0 else followers
    # 对每对 (a, b) 算相关
    n = len(returns.columns)
    cols = list(returns.columns)
    out = np.zeros((n, n))
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            out[i, j] = aligned[a].corr(followers[b])
    return pd.DataFrame(out, index=cols, columns=cols)


def top_followers(corr: pd.DataFrame, k: int = 5) -> dict[str, list[str]]:
    """每个行业 (leader) 的 top-k follower (排除自己)。"""
    out: dict[str, list[str]] = {}
    for leader in corr.index:
        row = corr.loc[leader].drop(leader, errors="ignore")
        out[leader] = row.nlargest(k).index.tolist()
    return out


# ============================================================================
# 3. 涨停联动 lift — 大涨日条件下后续涨幅提升
# ============================================================================


def contagion_lift(
    returns: pd.DataFrame,
    threshold: float = 0.02,
    lag: int = 3,
) -> pd.DataFrame:
    """计算 lift = P(B 上涨 > thr | A 大涨过去) / P(B 上涨 > thr 无条件)。

    > 1.5 显著正联动；< 0.7 反向。
    """
    n = len(returns.columns)
    cols = list(returns.columns)
    out = np.zeros((n, n))
    base_prob: dict[str, float] = {b: float((returns[b] > threshold).mean()) for b in cols}
    for i, a in enumerate(cols):
        big_up_a = returns[a] > threshold
        # A 大涨日的索引位置
        idx = np.where(big_up_a.to_numpy())[0]
        # 这些日期的 lag 天后 B 收益
        future_idx = idx + lag
        future_idx = future_idx[future_idx < len(returns)]
        for j, b in enumerate(cols):
            future_b = returns[b].to_numpy()[future_idx]
            cond_prob = (future_b > threshold).mean() if len(future_b) else 0
            base = base_prob[b]
            out[i, j] = cond_prob / base if base > 0 else 0
    return pd.DataFrame(out, index=cols, columns=cols)


# ============================================================================
# 主流程
# ============================================================================


def main() -> None:
    start = date(2018, 1, 1)
    end = date.today()
    print(f"=== 加载 {start} ~ {end} 的 SW L1 31 行业日收益 ===")
    returns = load_returns(start, end)
    if returns.empty:
        print("⚠️  index_daily 表无数据，请先跑 floatshare-sync --include index_daily")
        return
    print(f"  shape: {returns.shape}, 行业数 {returns.shape[1]}")

    print("\n=== 滞后相关矩阵 + top-5 跟随者 (lag = 5 天) ===")
    corr5 = lag_corr_matrix(returns, lag=5)
    followers = top_followers(corr5, k=5)
    for leader in list(SW_L1.values())[:8]:
        if leader in followers:
            print(f"  {leader:6s} → {' / '.join(followers[leader])}")
    print(f"  ... (共 {len(followers)} 个行业，详情见 lead_lag_map.json)")

    # 沉淀 LEAD_LAG_MAP 到 JSON (策略代码读它)
    map_path = OUT_DIR / "lead_lag_map_lag5.json"
    map_path.write_text(json.dumps(followers, ensure_ascii=False, indent=2))
    print(f"\n  ✓ 写入 {map_path}")

    # 多个 lag 的相关矩阵存 csv (后续画热图用)
    for lag in (1, 3, 5, 10, 20):
        corr = lag_corr_matrix(returns, lag=lag)
        corr.to_csv(OUT_DIR / f"corr_lag{lag}.csv")
    print(f"  ✓ 5 个滞后矩阵写入 {OUT_DIR}/corr_lag*.csv")

    print("\n=== 涨停联动 lift (条件大涨阈值 2%, lag 3 天) ===")
    lift = contagion_lift(returns, threshold=0.02, lag=3)
    # 找最强 5 对 (leader → follower lift)
    pairs = []
    for leader in lift.index:
        for follower in lift.columns:
            if leader == follower:
                continue
            pairs.append((leader, follower, float(lift.loc[leader, follower])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("  最强联动 top-10:")
    for leader, follower, val in pairs[:10]:
        print(f"    {leader:6s} → {follower:6s}  lift={val:.2f}")
    lift.to_csv(OUT_DIR / "contagion_lift_lag3.csv")
    print(f"\n  ✓ {OUT_DIR}/contagion_lift_lag3.csv")

    print("\n=== 完成 — 接下来 ===")
    print("  cat notebooks/output/lead_lag_map_lag5.json   # 看 lookup 表")
    print("  open notebooks/output/contagion_lift_lag3.csv # Numbers/Excel 看")


if __name__ == "__main__":
    main()

"""数据集 — 把 OHLCV 表 → MarketCube tensor (cached)。

MarketCube 是 PPO env 的底层数据结构, 保存:
    dates       — (n_days,) trade_date 数组
    token_meta  — (n_tokens,) 含 token_id / token_type / industry_id
    features    — (n_days, n_tokens, F) 标准化特征 (含当日)
    prices      — (n_days, n_tokens) close 价格 (算 reward)
    traded      — (n_days, n_tokens) bool: 当日是否有成交 (停盘 = False)

Phase 1: tokens 仅含 31 个 SW L1 行业指数
Phase 2: tokens 含 31 行业 + N 股票
Phase 3: tokens 仅 N 股票 (抓涨停, 含 opens/highs/lows/hit_labels)

时间切分按 train/val/test 三段, 不重叠. 缓存到 .npz 加速重跑.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from floatshare.ml.data.loader import load_industry_map, load_panel
from floatshare.ml.data.universe import select_universe
from floatshare.ml.features import FEATURE_COLS, N_FEATURES, compute_features
from floatshare.ml.normalize import cross_sectional_zscore

if TYPE_CHECKING:
    from floatshare.ml.config import DataConfig

SW_L1_CODES: tuple[str, ...] = (
    "801010.SI",
    "801030.SI",
    "801040.SI",
    "801050.SI",
    "801080.SI",
    "801110.SI",
    "801120.SI",
    "801130.SI",
    "801140.SI",
    "801150.SI",
    "801160.SI",
    "801170.SI",
    "801180.SI",
    "801200.SI",
    "801210.SI",
    "801230.SI",
    "801710.SI",
    "801720.SI",
    "801730.SI",
    "801740.SI",
    "801750.SI",
    "801760.SI",
    "801770.SI",
    "801780.SI",
    "801790.SI",
    "801880.SI",
    "801890.SI",
    "801950.SI",
    "801960.SI",
    "801970.SI",
    "801980.SI",
)
N_INDUSTRIES = len(SW_L1_CODES)
assert N_INDUSTRIES == 31

# 指数没有的列 — 用 NaN 占位, cross_sectional_zscore 后变 0
_INDEX_MISSING_COLS: tuple[str, ...] = (
    "pe_ttm",
    "pb",
    "turnover_rate",
    "total_mv",
    "net_mf_amount",
    "buy_lg_amount",
    "buy_elg_amount",
    "buy_sm_amount",
    "buy_md_amount",
    "sell_sm_amount",
    "sell_md_amount",
    "sell_lg_amount",
    "sell_elg_amount",
)


@dataclass(frozen=True, slots=True)
class TokenMeta:
    """每个 token 的元信息."""

    token_id: str  # '801770.SI' (industry) 或 '300308.SZ' (stock)
    token_type: int  # 0 = industry, 1 = stock
    industry_id: int  # 0..30, 行业自己 or 股票父行业


@dataclass(slots=True)
class MarketCube:
    """完整训练数据的张量化表示 (一个时间窗口)."""

    dates: np.ndarray  # (n_days,) datetime64
    tokens: list[TokenMeta]  # 长 n_tokens
    features: np.ndarray  # (n_days, n_tokens, F) float32
    prices: np.ndarray  # (n_days, n_tokens) float32 (close)
    traded: np.ndarray  # (n_days, n_tokens) bool
    # Phase 3 抓涨停专属 (build 时按需填):
    opens: np.ndarray | None = None  # (n_days, n_tokens) 开盘价 — label 用
    highs: np.ndarray | None = None  # (n_days, n_tokens) 最高价 — 一字板检测
    lows: np.ndarray | None = None  # (n_days, n_tokens) 最低价 — 全天封死判定
    hit_labels: np.ndarray | None = None  # (n_days, n_tokens) int8: 1/0/-1

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def n_industries(self) -> int:
        return sum(1 for t in self.tokens if t.token_type == 0)


def build_cube(
    cfg: DataConfig,
    start: str,
    end: str,
    phase: Literal[1, 2, 3],
    universe: list[str] | None = None,
) -> MarketCube:
    """构建一个时间区间的 MarketCube (含缓存).

    phase 1: 仅 31 行业指数
    phase 2: 31 行业 + N 股票 (双层 tokens)
    phase 3: 仅 N 股票 (抓涨停, 含 opens/highs/hit_labels)
    """
    cache_path = _cache_path(cfg, start, end, phase, universe)
    if cfg.use_cache and cache_path.exists():
        return _load_cache(cache_path)
    cube = _build_cube_uncached(cfg, start, end, phase, universe)
    if cfg.use_cache:
        _save_cache(cube, cache_path)
    return cube


def _build_cube_uncached(
    cfg: DataConfig,
    start: str,
    end: str,
    phase: Literal[1, 2, 3],
    universe: list[str] | None,
) -> MarketCube:
    industry_tokens, industry_panel, industry_feats_z = (
        _build_industry_part(cfg.db_path, start, end)
        if phase in (1, 2)
        else ([], pd.DataFrame(), pd.DataFrame())
    )
    stock_tokens, stock_panel, stock_feats_z = (
        _build_stock_part(cfg, start, end, universe)
        if phase in (2, 3)
        else ([], pd.DataFrame(), pd.DataFrame())
    )

    all_tokens = industry_tokens + stock_tokens
    panel = _concat_nonempty([industry_panel, stock_panel])
    feats_z = _concat_nonempty([industry_feats_z, stock_feats_z])

    cube = _assemble_cube(all_tokens, panel, feats_z, with_ohl=(phase == 3))
    if phase == 3 and cube.opens is not None and cube.highs is not None:
        from floatshare.ml.labels import make_hit_labels

        cube.hit_labels = make_hit_labels(cube.opens, cube.highs, lows=cube.lows)
    return cube


# --- Industry / Stock branches -----------------------------------------------


def _build_industry_part(
    db_path: str,
    start: str,
    end: str,
) -> tuple[list[TokenMeta], pd.DataFrame, pd.DataFrame]:
    panel = _load_index_as_panel(db_path, list(SW_L1_CODES), start, end)
    feats = compute_features(panel)
    feats_z = cross_sectional_zscore(_ensure_reset(feats))
    tokens = [TokenMeta(token_id=c, token_type=0, industry_id=i) for i, c in enumerate(SW_L1_CODES)]
    return tokens, panel, feats_z


def _build_stock_part(
    cfg: DataConfig,
    start: str,
    end: str,
    universe: list[str] | None,
) -> tuple[list[TokenMeta], pd.DataFrame, pd.DataFrame]:
    if universe is None:
        universe = select_universe(cfg, as_of_date=end)
    ind_map = load_industry_map(cfg.db_path, level=1).set_index("code")["ind_code"]
    sw_to_idx = {c: i for i, c in enumerate(SW_L1_CODES)}
    tokens: list[TokenMeta] = []
    for code in universe:
        ind_code = ind_map.get(code)
        ind_idx = sw_to_idx.get(str(ind_code), 0) if ind_code else 0
        tokens.append(TokenMeta(token_id=code, token_type=1, industry_id=ind_idx))
    panel = load_panel(cfg.db_path, universe, start, end)
    feats = compute_features(panel)
    feats_z = cross_sectional_zscore(_ensure_reset(feats))
    return tokens, panel, feats_z


def _concat_nonempty(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [df for df in dfs if not df.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _ensure_reset(df: pd.DataFrame) -> pd.DataFrame:
    """compute_features 的 trade_date 有时是列, 有时是 index — 统一拿成列."""
    return df if "trade_date" in df.columns else df.reset_index()


def _load_index_as_panel(
    db_path: str,
    index_codes: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """把指数日线伪装成股票 panel 喂给 compute_features (复用特征逻辑).

    指数没有 daily_basic / moneyflow → 那些列填 NaN, 后续 normalize 填 0.
    """
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30.0})
    placeholders = ",".join(f":c{i}" for i in range(len(index_codes)))
    params: dict[str, object] = {f"c{i}": c for i, c in enumerate(index_codes)}
    params.update({"start": start, "end": end})
    sql = text(f"""
        SELECT code, trade_date, open, high, low, close, volume, amount
        FROM index_daily
        WHERE code IN ({placeholders})
          AND trade_date >= :start AND trade_date <= :end
        ORDER BY code, trade_date
    """)
    df = pd.read_sql(sql, engine, params=params)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="ISO8601")
    for col in _INDEX_MISSING_COLS:
        df[col] = np.nan
    return df


# --- Cube assembly -----------------------------------------------------------


def _assemble_cube(
    tokens: list[TokenMeta],
    panel: pd.DataFrame,
    feats_z: pd.DataFrame,
    with_ohl: bool = False,
) -> MarketCube:
    """从 long-format → 3D tensor (n_days, n_tokens, ...).

    with_ohl=True: 额外填 opens/highs/lows (Phase 3 抓涨停 label 用).
    """
    if panel.empty or feats_z.empty:
        raise ValueError("空 panel/特征, 检查 universe 与时间范围")
    panel = _ensure_datetime(panel.copy())
    feats_z = _ensure_datetime(feats_z.copy())

    dates = np.sort(panel["trade_date"].unique())
    code_to_idx = {t.token_id: i for i, t in enumerate(tokens)}
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    n_days, n_tokens = len(dates), len(tokens)

    prices, traded, opens, highs, lows = _init_price_arrays(n_days, n_tokens, with_ohl)
    _fill_prices_and_ohl(panel, code_to_idx, date_to_idx, prices, traded, opens, highs, lows)

    features = np.zeros((n_days, n_tokens, N_FEATURES), dtype=np.float32)
    _fill_features(feats_z, code_to_idx, date_to_idx, features)

    return MarketCube(
        dates=dates,
        tokens=tokens,
        features=features,
        prices=prices,
        traded=traded,
        opens=opens,
        highs=highs,
        lows=lows,
    )


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="ISO8601")
    return df


def _init_price_arrays(
    n_days: int,
    n_tokens: int,
    with_ohl: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    prices = np.full((n_days, n_tokens), np.nan, dtype=np.float32)
    traded = np.zeros((n_days, n_tokens), dtype=bool)
    opens: np.ndarray | None = None
    highs: np.ndarray | None = None
    lows: np.ndarray | None = None
    if with_ohl:
        opens = np.full((n_days, n_tokens), np.nan, dtype=np.float32)
        highs = np.full((n_days, n_tokens), np.nan, dtype=np.float32)
        lows = np.full((n_days, n_tokens), np.nan, dtype=np.float32)
    return prices, traded, opens, highs, lows


def _fill_prices_and_ohl(
    panel: pd.DataFrame,
    code_to_idx: dict[str, int],
    date_to_idx: dict[pd.Timestamp, int],
    prices: np.ndarray,
    traded: np.ndarray,
    opens: np.ndarray | None,
    highs: np.ndarray | None,
    lows: np.ndarray | None,
) -> None:
    for row in panel.itertuples(index=False):
        ti = code_to_idx.get(row.code)
        di = date_to_idx.get(pd.Timestamp(row.trade_date))
        if ti is None or di is None:
            continue
        if pd.notna(row.close):
            prices[di, ti] = row.close
            traded[di, ti] = True
        if opens is not None and pd.notna(row.open):
            opens[di, ti] = row.open
        if highs is not None and pd.notna(row.high):
            highs[di, ti] = row.high
        if lows is not None and pd.notna(row.low):
            lows[di, ti] = row.low


def _fill_features(
    feats_z: pd.DataFrame,
    code_to_idx: dict[str, int],
    date_to_idx: dict[pd.Timestamp, int],
    features: np.ndarray,
) -> None:
    feat_arr = feats_z[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
    codes = feats_z["code"].to_numpy()
    fdates = feats_z["trade_date"].to_numpy()
    for k in range(len(feats_z)):
        ti = code_to_idx.get(str(codes[k]))
        di = date_to_idx.get(pd.Timestamp(fdates[k]))
        if ti is None or di is None:
            continue
        features[di, ti, :] = feat_arr[k]


# --- Cache -------------------------------------------------------------------


def _cache_path(
    cfg: DataConfig,
    start: str,
    end: str,
    phase: Literal[1, 2, 3],
    universe: list[str] | None,
) -> Path:
    key = json.dumps(
        {
            "start": start,
            "end": end,
            "phase": phase,
            "universe": sorted(universe) if universe else None,
            "universe_mode": cfg.universe_mode,
            "top_mv_n": cfg.top_mv_n,
            "n_features": N_FEATURES,  # 维度变了 → 不命中旧 cache, 自动重建
        },
        sort_keys=True,
    )
    # MD5 仅用于 cache key 哈希, 不涉及安全 — usedforsecurity=False 显式声明 (S324)
    h = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]
    p = Path(cfg.cache_dir) / f"cube_{phase}_{start[:7]}_{end[:7]}_{h}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _save_cache(cube: MarketCube, path: Path) -> None:
    extra: dict = {}
    for name in ("opens", "highs", "lows", "hit_labels"):
        arr = getattr(cube, name)
        if arr is not None:
            extra[name] = arr
    np.savez_compressed(
        path,
        dates=cube.dates,
        token_ids=np.array([t.token_id for t in cube.tokens]),
        token_types=np.array([t.token_type for t in cube.tokens], dtype=np.int8),
        industry_ids=np.array([t.industry_id for t in cube.tokens], dtype=np.int8),
        features=cube.features,
        prices=cube.prices,
        traded=cube.traded,
        **extra,
    )


def _load_cache(path: Path) -> MarketCube:
    z = np.load(path, allow_pickle=False)
    tokens = [
        TokenMeta(str(tid), int(tt), int(iid))
        for tid, tt, iid in zip(z["token_ids"], z["token_types"], z["industry_ids"], strict=False)
    ]
    keys = set(z.files)
    return MarketCube(
        dates=z["dates"],
        tokens=tokens,
        features=z["features"],
        prices=z["prices"],
        traded=z["traded"],
        opens=z["opens"] if "opens" in keys else None,
        highs=z["highs"] if "highs" in keys else None,
        lows=z["lows"] if "lows" in keys else None,
        hit_labels=z["hit_labels"] if "hit_labels" in keys else None,
    )

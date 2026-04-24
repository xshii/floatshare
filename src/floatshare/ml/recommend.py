"""Phase 3 抓涨停 — 今日 top-K 选股 + Bark 推送, trainer 无关.

用途: Pop / GRPO 训完后调 `push_today_picks(ckpt_path, device)` 出当日选股 +
通过 observability.notify 推到手机. 锁屏预览用"人设化"掩饰段 (颜文字 + 闲聊) 替代
明显股票 token 字样.

调用方:
    - `ml/train_pop.py` main() 末尾 (Pop cold/warm 都用)
    - `ml/train_grpo.py` 末尾 (GRPO warm 用, 同格式)
    - 手动运维 one-shot: `python -m floatshare.ml.recommend --ckpt xxx.pt`
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from floatshare.ml.config import DataConfig
from floatshare.ml.data.dataset import build_cube
from floatshare.ml.data.universe import select_per_industry_top_k
from floatshare.ml.model.agent import load_ckpt
from floatshare.ml.types import PopActionOut
from floatshare.observability import logger, notify

# DB 路径以 DataConfig 为真相源, 避免散落在各文件的字符串 "data/floatshare.db" 不一致.
_DEFAULT_DB_PATH = DataConfig().db_path


def _build_today_picks(
    ckpt_path: Path,
    device: torch.device,
    top_k: int,
    db_path: str,
) -> tuple[date, list[tuple[str, float]]] | None:
    """读 ckpt + 算今日 universe + top-K, 返回 (trade_date, [(code, p_hit), ...]).

    None = 跳过 (ckpt 缺 / universe 空 / cube 短)."""
    if not ckpt_path.exists():
        logger.warning(f"[push_picks] ckpt 不存在: {ckpt_path}, 跳过")
        return None

    model = load_ckpt(str(ckpt_path)).to(device)
    model.eval()
    seq_len = model.cfg.seq_len

    today = date.today().strftime("%Y-%m-%d")
    universe = select_per_industry_top_k(db_path, today)
    if not universe:
        logger.warning("[push_picks] universe 空 (可能 today snapshot 未就绪), 跳过")
        return None

    # Lookback: seq_len 是**交易日**, 转 calendar 要 × 1.5 (含周末) + buffer
    lookback_cal = int(seq_len * 1.5) + 60
    cube_start = (pd.Timestamp(today) - pd.Timedelta(days=lookback_cal)).strftime("%Y-%m-%d")
    cube = build_cube(DataConfig(), cube_start, today, phase=3, universe=universe)
    if cube.n_days < seq_len:
        logger.warning(f"[push_picks] cube 不够长 ({cube.n_days} < {seq_len}), 跳过")
        return None

    feats = torch.from_numpy(cube.features).to(device)
    t_last = cube.n_days - 1
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        window = (
            feats[t_last - seq_len + 1 : t_last + 1].unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        )
        n = window.shape[1]
        out = cast(
            PopActionOut,
            model(
                window,
                torch.ones(1, n, dtype=torch.long, device=device),
                torch.zeros(1, n, dtype=torch.long, device=device),
                torch.ones(1, n, dtype=torch.bool, device=device),
            ),
        )
        p_hit = out.p_hit.float().cpu().numpy()[0]

    top_idx = np.argsort(-p_hit)[:top_k]
    token_ids = [tk.token_id for tk in cube.tokens]
    picks = [(token_ids[i], float(p_hit[i])) for i in top_idx]
    # cube.dates 是有效 datetime64, Timestamp().date() 返 date 不会 NaT;
    # pyright 从 stub 推出 NaTType | date, 这里实际不可能是 NaT, 强制收紧
    trade_date = cast(date, pd.Timestamp(cube.dates[t_last]).date())
    return trade_date, picks


def _format_camo_body(trade_date: date, picks: list[tuple[str, float]]) -> str:
    """构造 Bark body: 前 ~90 字掩饰段 (锁屏预览看到的), picks 拼后面.

    6 位 code 去掉交易所后缀, 降低一眼认出是股票的概率.
    tag 每日按日期 hash 不同, 增加 "真监控" 可信度.
    """
    # md5 纯装饰用 (不做安全用途), usedforsecurity=False 显式告知 linter
    tag = hashlib.md5(str(trade_date).encode(), usedforsecurity=False).hexdigest()[:6]
    camo = (
        f"〜(꒪꒳꒪)〜 今日小日记 ~ 猫猫今天吃了三条鱼 /ᐠ｡ꞈ｡ᐟ\\ "
        f"窗外花开了呢 ✿ 温度 21°C 心情不错 (๑•̀ㅂ•́)و✧ "
        f"session_id={tag} 加油鸭~ 🦆 · "
    )
    picks_str = " ".join(f"{c[:6]}({p:.2f})" for c, p in picks)
    return camo + picks_str


def push_today_picks(
    ckpt_path: str | Path,
    device: torch.device,
    *,
    top_k: int = 10,
    db_path: str = _DEFAULT_DB_PATH,
) -> None:
    """训完 / 离线用: 读 ckpt → 今日 top-K → 掩饰 body → Bark push.

    失败只告警, 不抛异常 (caller 典型场景 = 训练尾部, 推送失败不该让训练结果 status 变 CRASHED).
    """
    result = _build_today_picks(Path(ckpt_path), device, top_k, db_path)
    if result is None:
        return
    trade_date, picks = result
    body = _format_camo_body(trade_date, picks)
    notify(title="日常小记 (*´꒳`*)", body=body)
    picks_str = " ".join(f"{c[:6]}({p:.2f})" for c, p in picks)
    logger.info(f"[push_picks] {trade_date} pushed (camo): {picks_str}")

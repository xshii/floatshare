"""ML pipeline stage 内部 helper — ckpt 路径发现、通用 subprocess 封装等."""

from __future__ import annotations

from pathlib import Path

_CKPT_DIR = Path("data/ml/ckpts")


def find_best_ckpt(ckpt_dir: Path = _CKPT_DIR) -> Path | None:
    """扫 ckpt 目录找最新 best checkpoint.

    约定文件名形如 `phase3_pretrain_best.pt` 或 `v{N}-best.pt`. 取 mtime 最晚的.
    None = 目录不存在或空.
    """
    if not ckpt_dir.exists():
        return None
    candidates = sorted(
        (p for p in ckpt_dir.iterdir() if p.suffix == ".pt" and p.stat().st_size >= 1024),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None

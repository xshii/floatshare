"""训练性能 profiling — 分段 timer + torch.profiler 包装.

用途:
    1. SectionTimer: 手动 `with timer.section("name")` 标记代码段, 累计多 batch 统计,
       打印 per-section mean / pct. MPS/CUDA 会自动 synchronize() 拿真实 kernel 时间.
    2. torch_profile_run: 可选产出 chrome://tracing 兼容 JSON (op-level 深度分析).

    PopTrainer 接受一个可选 _timer 参数 (默认 None → 零开销 nullcontext),
    profile 模式下注入 SectionTimer 实例给各阶段打点.
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from floatshare.observability import logger

if TYPE_CHECKING:
    from torch.profiler import profile as TorchProfile

    from floatshare.ml.training.pop import PopTrainer


class SectionTimer:
    """累积分段耗时 — section() 是上下文管理器, MPS/CUDA 会 sync 后再 stop clock.

    典型用法:
        timer = SectionTimer(device)
        for batch in batches:
            with timer.section("data_prep"):
                ...
            with timer.section("forward"):
                ...
        print(timer.format_report())
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self._device = device
        self._sections: defaultdict[str, list[float]] = defaultdict(list)
        # start/stop 用: 按 name stack, 支持同名重入 (forward hook 可能递归)
        self._open: defaultdict[str, list[float]] = defaultdict(list)

    def _sync(self) -> None:
        """等 device 上的 kernel 都完成, 确保 clock 反映真实耗时."""
        if self._device is None:
            return
        if self._device.type == "mps":
            torch.mps.synchronize()
        elif self._device.type == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._sections[name].append(time.perf_counter() - t0)

    def start(self, name: str) -> None:
        """非 with 版本 — 用于 forward hook 这种拿不到 context manager 的场景."""
        self._sync()
        self._open[name].append(time.perf_counter())

    def stop(self, name: str) -> None:
        self._sync()
        if not self._open[name]:
            return  # 没 start 过, 丢弃 (防御性)
        t0 = self._open[name].pop()
        self._sections[name].append(time.perf_counter() - t0)

    def report(self) -> dict[str, dict[str, float]]:
        """{section: {count, sum_s, mean_ms, pct}}, pct 相对所有 section 总时间."""
        total = sum(sum(v) for v in self._sections.values())
        out: dict[str, dict[str, float]] = {}
        for name, times in self._sections.items():
            s = sum(times)
            out[name] = {
                "count": float(len(times)),
                "sum_s": s,
                "mean_ms": s / max(len(times), 1) * 1000.0,
                "pct": s / total * 100.0 if total > 0 else 0.0,
            }
        return out

    def format_report(self) -> str:
        r = self.report()
        if not r:
            return "(empty profile)"
        lines = [
            f"{'section':20} {'count':>6} {'sum(s)':>8} {'mean(ms)':>10} {'pct(%)':>7}",
            "-" * 60,
        ]
        for name, stats in sorted(r.items(), key=lambda kv: -kv[1]["sum_s"]):
            lines.append(
                f"{name:20} "
                f"{stats['count']:>6.0f} "
                f"{stats['sum_s']:>8.2f} "
                f"{stats['mean_ms']:>10.2f} "
                f"{stats['pct']:>7.1f}"
            )
        return "\n".join(lines)


def maybe_section(timer: SectionTimer | None, name: str) -> contextlib.AbstractContextManager[None]:
    """零开销 helper: 无 timer 时返回 nullcontext, 有 timer 调 section()."""
    return timer.section(name) if timer is not None else contextlib.nullcontext()


_DEFAULT_LEAF_NAMES: tuple[str, ...] = (
    "time_attn",
    "stock_attn",
    "ffn",
    "norm_t",
    "norm_s",
    "norm_ff",
    "feat_proj",
    "final_norm",
    "head",
    "value_head",
)


def install_module_timings(
    model: nn.Module,
    timer: SectionTimer,
    leaf_names: tuple[str, ...] = _DEFAULT_LEAF_NAMES,
    *,
    prefix: str = "mod/",
) -> Callable[[], None]:
    """给 model 里末段 name (leaf component) 命中 leaf_names 的子模块装 forward hooks.

    只匹配 `name.rsplit(".", 1)[-1] in leaf_names`, 避免 attention 内部 q_proj/k_proj
    那种深层 Linear 产生噪声 sections (内部 Linear 的耗时已被父模块 attn 覆盖).

    内部用 timer.start/stop 兼容 PyTorch hook 协议 (hook 不能持 context manager).
    name 前缀 `mod/` 跟顶层 section (data_prep/forward/...) 区分.

    注意: 每 hook 都会触发 _sync (MPS/CUDA), 强制序列化 pipeline — profile 模式下
    forward 总时间会被 inflated (约 2-3×), 但 section 间**相对**比例仍可信.

    Returns: cleanup — 调用后移除全部 hook.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in model.named_modules():
        if not name:  # top-level
            continue
        last = name.rsplit(".", 1)[-1]
        if last not in leaf_names:
            continue
        tag = prefix + name

        def _pre(_m: nn.Module, _inputs: object, tag: str = tag) -> None:
            timer.start(tag)

        def _post(_m: nn.Module, _inputs: object, _output: object, tag: str = tag) -> None:
            timer.stop(tag)

        handles.append(module.register_forward_pre_hook(_pre))
        handles.append(module.register_forward_hook(_post))

    def cleanup() -> None:
        for h in handles:
            h.remove()

    return cleanup


@contextmanager
def torch_profile_run(
    output_dir: Path,
    *,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
) -> Iterator[TorchProfile]:
    """torch.profiler 包装 — 产 chrome trace JSON 到 output_dir, Perfetto 打开.

    用法:
        with torch_profile_run(Path("data/ml/traces/pop")) as prof:
            for _ in range(wait + warmup + active):
                train_one_batch(...)
                prof.step()
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    # MPS 目前 ProfilerActivity.MPS 在 PyTorch 里还不稳; CUDA 有完整支持
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)
    on_trace_ready = torch.profiler.tensorboard_trace_handler(str(output_dir))
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        yield prof


def run_profile(
    trainer: PopTrainer,
    *,
    n_batches: int = 5,
    module_level: bool = True,
) -> SectionTimer:
    """对 trainer 跑 n_batches 个 batch 分段打点, 返回 SectionTimer.

    会:
      - 构建 train_ctx (含 cube GPU preload + 模板预分配)
      - 跑 scheduler / warmup 等 boilerplate
      - 跑 (1 warmup + n_batches timed) — 第 1 个 batch 不计 (MPS kernel cache)
      - module_level=True: 额外装 forward hook 产每个 block / attn / ffn 细粒度 breakdown
      - 打印 report 到 logger
      - **不 save ckpt, 不进 metrics.db**, 作为独立工具使用.
    """
    timer = SectionTimer(trainer.device)
    trainer._timer = timer  # type: ignore[attr-defined]  # 注入给 _train_one_batch

    logger.info(
        f"[profile] building train ctx (this includes cube load & .to({trainer.device}))..."
    )
    with timer.section("build_ctx"):
        ctx = trainer._build_train_ctx()
    total_steps = trainer._total_steps(ctx)
    trainer.scheduler = trainer._build_scheduler(total_steps)

    rng = np.random.default_rng(0)
    valid = ctx.valid_starts.copy()
    rng.shuffle(valid)

    seq_len = trainer.model_cfg.seq_len
    bd = ctx.batch_days
    logger.info(f"[profile] n_batches={n_batches} (+1 warmup), batch_days={bd}, seq_len={seq_len}")

    undo_hooks: Callable[[], None] | None = None
    n_total = n_batches + 1  # 第 1 个 warmup, 不算入 section stats
    for i in range(n_total):
        batch_idx = valid[i * bd : (i + 1) * bd]
        if len(batch_idx) < 2:
            break
        if i == 0:
            # warmup — 不打点 (MPS kernel cache / lazy graph 构建)
            trainer._timer = None
            trainer._train_one_batch(batch_idx, ctx, seq_len)
            trainer._timer = timer
            # warmup 后才装 hook — 避免 MPS 首次 kernel compile 污染模块级耗时
            if module_level:
                undo_hooks = install_module_timings(trainer.model, timer)
        else:
            trainer._train_one_batch(batch_idx, ctx, seq_len)

    if undo_hooks is not None:
        undo_hooks()

    logger.info("[profile] bottleneck report:\n" + timer.format_report())
    return timer

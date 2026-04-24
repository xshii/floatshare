"""ml/ 全部超参 — 单点真相, 用 dataclass 表达。

CLI / 测试 / 训练循环都从这里读, 避免散落在各处的 magic number。
所有 dataclass 都 frozen=True, slots=True — 不可变 + 内存友好。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Transformer 架构超参。同一份配置同时支持 Phase 1 / Phase 2。"""

    # === 核心维度 ===
    embed_dim: int = 64  # token / 时序步 embedding 维度 D
    ff_dim: int = 256  # FFN 中间维度 (典型 4×D)
    head_dim: int | None = None  # None → embed_dim // n_heads

    # === 注意力 ===
    n_heads: int = 4  # multi-head 数
    n_layers_temporal: int = 3  # 单 token 内的时序 encoder 层数
    n_layers_cross: int = 2  # 跨 token 的 attention 层数

    # === 序列 ===
    seq_len: int = 60  # T (历史天数)
    n_features: int = 39  # F (per-day 特征维度, 见 features.py; 37 旧 + 2 新闻)

    # === 标的 ===
    n_industries: int = 31  # SW L1 行业数
    universe_size: int = 300  # Phase 2 股票池大小

    # === 正则 ===
    dropout: float = 0.1
    attn_dropout: float = 0.0
    rms_eps: float = 1e-6

    # === Encoder 模式 ===
    # single_axis (旧): TemporalEncoder (per-token 时序) + CrossTokenEncoder (跨 token)
    # dual_axis  (新): DualAxisBlock 每层交替 Time-attn + Stock-attn, 全程 (B,N,T,D)
    encoder_mode: Literal["single_axis", "dual_axis"] = "dual_axis"
    n_dual_layers: int = 3  # dual_axis 下用的 block 层数

    # === Sparse stock-attn (dual_axis 性能优化) ===
    # T=60 时, stock-attn 是 O(T·N²) 瓶颈; 稀疏化能省 3-5×.
    # active steps = {0, every, 2·every, ...} ∪ {T - last_dense, ..., T-1}
    # 默认: every=10, last_dense=5 → {0/10/20/30/40/50} ∪ {55..59} = 11 次, 省 5× (vs 60 次)
    # (旧 default every=5,last_dense=10 = 20 次; M4+SDPA 后压力转到 MHA FLOPs, 激进稀疏化更好)
    stock_attn_every: int = 10
    stock_attn_last_dense: int = 5

    # === 阶段 ===
    # 1: 行业 pick (IndustryHead)
    # 2: 行业 + 个股双层 (HierarchicalHead)
    # 3: 抓涨停 (FlatStockHead — 截面 + P(hit) 二分类)
    phase: Literal[1, 2, 3] = 1
    pretrained_backbone: str | None = None  # 加载上一阶段 backbone ckpt


@dataclass(frozen=True, slots=True)
class DataConfig:
    """数据准备超参。"""

    db_path: str = "data/floatshare.db"

    # === 时间切分 (按时间, 不是随机) — prod-oriented, 对齐业界 WFV + warm-start 链 ===
    # Anchor 固定 2018-01-01: tushare cctv_news 数据起点, 避免 2014-2017 news 特征全 0 伪信号
    # Train 吃到 2024-09: 6.75 年历史, 学到注册制成熟期 / 量化监管 / AI 行情等最新 regime
    # Val 紧贴 train 尾 6 个月 (~120 交易日): AUC 稳定需要, 小于 seq_len=60 会崩
    # Test 1 整年: 2025-04~2026-03, 覆盖一轮周期足够评估
    # 2026-04+ 起 S4 daily warm-start 接手 (stage_s4_train + find_best_ckpt)
    # 每季度 / 半年人工跑 scripts/walk_forward_eval.py 做 sanity check + 定期冷重训 anchor
    train_start: str = "2018-01-01"
    train_end: str = "2024-09-30"
    val_start: str = "2024-10-01"
    val_end: str = "2025-03-31"
    test_start: str = "2025-04-01"
    test_end: str = "2026-03-31"

    # === Universe ===
    # 默认 top_mv: index_weight 表只有近期数据, 历史训练集走动态市值排名更稳
    universe_mode: Literal["hs300", "zz500", "top_mv"] = "top_mv"
    top_mv_n: int = 300  # universe_mode='top_mv' 时取 top N 流通市值

    # === 缓存 ===
    cache_dir: str = "data/ml"
    use_cache: bool = True


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO 算法超参。"""

    # === Reward 公式 ===
    reward_horizon: int = 5  # 持仓 K 天后算 reward
    turnover_penalty: float = 0.001  # λ_turn ~ commission_rate
    entropy_bonus: float = 0.05  # β_ent — 调高鼓励探索 (Dirichlet 易坍缩)

    # Phase 2 only: 行业 timing 项权重 (1-α 是 selection alpha, α 是 timing)
    industry_timing_weight: float = 0.3  # γ in Phase 2 公式

    # === PPO 标准超参 ===
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 1.0  # K 天即终止, 不需要 discount
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64

    # === Rollout ===
    rollout_days: int = 250  # 每次 rollout 收集 1 年 episode


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    """GRPO 算法超参 (DeepSeek-Math 2024).

    相比 PPO:
        - 无 value head / GAE (critic-free)
        - 每个 state 采 G 个 action, 组内 mean/std 归一化做 advantage
        - 更适合金融 RL (V(s) 难估) + 天然适合"一天 G 个 portfolio 比较"
    """

    # === Reward 公式 (与 PPO 一致, 复用 MarketEnv) ===
    reward_horizon: int = 5
    turnover_penalty: float = 0.001
    entropy_bonus: float = 0.02  # GRPO 比 PPO 略低, 组内归一化已降方差

    # === Group sampling ===
    group_size: int = 8  # G: 每 state 采样 action 数 (4-16 典型范围)
    adv_eps: float = 1e-6  # 组 std 分母 smoothing

    # === Clip / KL ===
    clip_ratio: float = 0.2  # 同 PPO
    kl_coef: float = 0.04  # GRPO 特有: 对 reference policy 的 KL 约束 (防漂)
    max_grad_norm: float = 0.5
    update_epochs: int = 1  # GRPO 典型 1 epoch (PPO 要 4), 组内 baseline 省梯度

    # === Rollout ===
    rollout_days: int = 250  # 同 PPO


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """训练循环超参。"""

    device: Literal["mps", "cuda", "cpu"] = "mps"
    seed: int = 42

    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-5

    early_stop_patience: int = 10  # val sharpe 不再提升 patience 个 epoch 停

    # === 输出 ===
    ckpt_dir: str = "data/ml/ckpts"
    log_dir: str = "data/ml/logs"
    signal_csv: str = "data/ml/signals/daily.csv"

    # === Reward benchmark ===
    market_baseline: Literal["equal_sw", "hs300"] = "equal_sw"


@dataclass(frozen=True, slots=True)
class FullConfig:
    """所有 config 一站式入口 (yaml 反序列化目标)。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

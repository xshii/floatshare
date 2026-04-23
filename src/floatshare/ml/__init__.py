"""ml/ — Transformer + PPO 量化交易 agent。

架构 (统一 Phase 1 行业 pick + Phase 2 行业+个股 pick):
    Encoder backbone (RMSNorm Transformer) ──┬── IndustryHead (Phase 1)
                                              └── HierarchicalHead (Phase 2)

Phase 1 → Phase 2 时, backbone ckpt 复用; 仅切换 head + tokens 集合。

子模块:
    config       — 所有超参数 (ModelConfig / TrainConfig / PPOConfig / DataConfig)
    features     — 23 维指标计算 (复用 web/indicators.py 的纯函数)
    normalize    — cross-sectional z-score + clip + nan-fill
    data/        — DB 读取 + universe 选股 + 滑窗 dataset
    model/       — RMSNorm + Attention + Encoder + Heads + ActorCritic
    rl/          — MarketEnv (reward 公式) + rollout + PPO update
    train        — CLI 训练入口
    eval         — IC/RankIC/Sharpe + signal CSV 输出

Reward 公式:
    Phase 1: R = Σ_ind w_ind * (r_ind - r_market) - λ_turn * |Δw| + β_ent * H(w)
    Phase 2: R = Σ_i w_i * (r_i - r_industry(i))
               + γ * Σ_ind w_ind_total * (r_ind - r_market)
               - λ_turn * |Δw| + β_ent * H(w)
"""

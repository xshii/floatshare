"""RL 训练 — MarketEnv + Rollout + PPO update。

env     — 接 ActorCritic, 输出 state/action/reward; 支持 Phase 1 / Phase 2
rollout — 批量收集 episodes
ppo     — 标准 PPO update (clip surrogate + GAE + entropy bonus)
"""

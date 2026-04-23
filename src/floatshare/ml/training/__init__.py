"""训练器 — 消除 PPO / 监督的 epoch 循环 / 早停 / ckpt save 重复.

base : BaseTrainer — 通用骨架 (optimizer + epoch 循环 + 早停 + save)
pop  : PopTrainer  — Phase 3 监督预训 (BCE + EMA + warmup + pos_weight)
ppo  : PPOTrainer  — Phase 1/2 PPO (rollout + clipped surrogate + GAE)
grpo : GRPOTrainer — Phase 1/2 GRPO (critic-free, 组内归一化 baseline)
"""

from floatshare.ml.training.base import BaseTrainer as BaseTrainer
from floatshare.ml.training.grpo import GRPOTrainer as GRPOTrainer
from floatshare.ml.training.pop import PopTrainer as PopTrainer
from floatshare.ml.training.ppo import PPOTrainer as PPOTrainer

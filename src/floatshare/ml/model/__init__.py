"""Transformer + PPO 模型组件。

norm      — RMSNorm
attention — 可配置 MultiHeadAttention (含 mask)
encoder   — TemporalEncoder (per-token) + CrossTokenEncoder (跨 token)
heads     — IndustryHead (Phase 1) + HierarchicalHead (Phase 2)
agent     — ActorCritic 顶层封装 (backbone + 两个 head 可热插拔)
"""

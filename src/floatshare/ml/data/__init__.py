"""数据准备 — 从 SQLite 到 Transformer 输入 tensor。

loader   — 读 OHLCV / daily_basic / moneyflow / industry / sw_index
universe — 选股池 (沪深300 / 中证500 / top-N 流通市值)
dataset  — 滑窗 + 时间切分 + 张量化 + 缓存
"""

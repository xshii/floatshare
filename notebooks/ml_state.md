# ML 工作交接 — Phase 3 抓涨停 Transformer

> **作用**：`/clear` 重启后第一句问 "继续 ML 工作" 时，Claude 读这个文件就能立刻接着干。
> **禁止**：在这里写代码细节（看 `src/floatshare/ml/`）、CLAUDE.md 已说的项目约定、临时进度。

---

## 任务

D 日决策 → D+1 开盘买 → D+2 开盘卖 → 目标涨幅 ≥ 5%。
Stage A 监督预训 (BCE on hit_label) → Stage B PPO 微调（未启动）。

代码全在 `src/floatshare/ml/`，入口 `train_pop.py`。

---

## 关键设计决策（**为什么**这样，code 看不出）

### Universe — `data/universe.py:select_per_industry_top_k`
- **每行业 top-15** 而不是全市场 top-N：避免风格集中（top-N 流通市值 → 全是金融周期）
- **主板 only** (`60/00 前缀`)：创业板/科创板涨跌停 20%，跟主板 10% 分布完全不同，不能混训
- **快照固定到 train_end**：避免幸存者偏差（拿训练末日 universe 训整段是次优但实用妥协；月度 rolling 是 TODO）
- **打分 = 0.5·z(turnover_ma20) + 0.35·z(vola_60d) + 0.15·z(-log(circ_mv))**：换手 > 波动 > 偏小盘
- **circ_mv sweet spot [20亿, 500亿]**：抓涨停股的实证范围
- **`turnover_rate > 0` 仅排停牌不排一字板**（一字板由 labels 那层管，`open==high==low`）

### Label — `labels.py:make_hit_labels`
- **threshold 5% / 持 1 天**：A 股 T+1 最短可行；超过 5% 是显著超额
- **一字板判定 = `open == high == low`**：T 字板 / 破板能买，不能误伤；浮点容差 1e-4
- **NaN / 一字板 → label=-1 mask 掉**：避免乐观偏差

### Features — `features.py` 37 维
代码里看 `FEATURE_COLS`。**为什么是这些**：
- 全部 `shift(1)`：D 日特征 = D-1 收盘后可见信息，零前视
- 涨停板 6 维（`days_since_limit` 等）：纯抓涨停任务专属
- 多尺度 6 维（流通盘/换手/波动各两个尺度）：让模型自己学规模 × 活跃度交互
- 价量匹配 3 维（`price_vol_match5`、`inflow_5d_sum`、`vol_ret_corr60`）：单独的量和价网络可学，但联合关系（"价升量增" vs "价升量缩"）短窗口很关键，提前算好减负担
- **不加涨停日量比**：用户明确"预测高涨幅 ≠ 追涨停"

### 训练范式 — `train_pop.py:train_supervised`
- **`pos_weight = sqrt(neg/pos) ≈ 4.93`**：raw 24x 太狠会过度预测正，sqrt 是业界经验缓和
- **EMA decay 0.995**：raw model 抖动大，eval / save 都用 EMA
- **Warmup 5% + cosine to 0.1×base_lr**：稳定起步 + 收尾精调
- **Label smoothing 0.05**：缓解过拟合（正样本 → 0.95, 负样本 → 0.05）
- **Early stop patience=5**（按每 2 epoch eval 一次算）
- **`valid_starts` 排除最后 reward_horizon=3 天**：避免拿到 NaN label

### 模型 — `model/encoder.py:DualAxisBlock`
- **dual_axis 模式**（iTransformer 风格）：每层交替 Time-attn (per stock) + Stock-attn (per timestep)，全程 (B,N,T,D)
- **Sparse stock-attn**：`stock_attn_every=5, last_dense=10` → 60 步只在 ~20 步做 stock-attn，省 3×（原 O(T·N²) 是 GPU 瓶颈）

---

## 当前进度

- **v6 baseline**：AUC **0.652**（top-300 流通市值 + 27 维特征 + 无 EMA/warmup/pos_weight）
- **v7**：34 维 + 460 股行业 + EMA + warmup + pos_weight + label_smooth + sparse stock-attn
  - 在 E02 意外中断（原因未知，进程没了、log 无错误）
  - E02 `val_auc=0.659 val_p@10=0.168`（已超 v6 → 架构升级有效）
  - ckpt 备份: `data/ml/ckpts/phase3_pretrain_v7_best.{pt,json}`
  - log: `logs/train_v7_20260421-081500.log`
- **v8 跑中** (2026-04-21 08:41 启动, PID 66549)：37 维（v7 + 价量匹配 3 维）
  - log: `logs/train_v8_20260421-084100.log`
  - 30 epoch / dual_axis / MPS，预计 ~11:00 完成（或早停）
  - 注：v7 单独 34 维完整结果缺失，v8 只能跟 v6 0.652 综合对比，34→37 特征纯净 A/B 放弃

---

## TODO（按优先级）

1. **等 v7 完成 → 看 vs v6 0.652 提升**
2. **启 v8（37 维）跟 v7 A/B**（同样 30 epoch / device=mps）
3. **回测 lookahead 修复**：选 top-K 时不预筛 traded，执行 D+1 时一字板/停盘股变现金（避开"只算可买股 top-K"的隐式前视）
4. **Train/Val 时间缓冲**：`val_start` 加 60 天 gap（防特征 rolling 泄漏）
5. **除权日 mask**（用户明确要的，未实现）：分红 / 扩股期间不交易，避开分红税
6. **混合精度 bf16**：MPS 支持后能再快 1.5×
7. **Stage B PPO finetune**：A 跑稳后接 B（reward_horizon=5, turnover_penalty=0.001）
8. **Monthly rolling universe**：月初重选 460 股（消幸存者偏差，长期项）

---

## 重要路径速查

- 训练入口：`src/floatshare/ml/train_pop.py` (CLI 薄壳) → `training/pop.py::PopTrainer.fit()`
- PPO 训练：`src/floatshare/ml/train.py` → `training/ppo.py::PPOTrainer.fit()`
- 评估 API：`src/floatshare/ml/evaluation/` — `classifier_metrics` (监督) / `run_deterministic_rollout` (env 推理)
- ckpt 输出：`data/ml/ckpts/phase3_pretrain_best.pt`
- cube 缓存：`data/ml/cube_3_*.npz`（key 含 n_features 自动失效）
- 训练 log：`logs/train_v{7,8}_*.log`
- venv 激活：`source .venv/bin/activate`（不要用 `.venv/bin/python` 前缀）

## 2026-04-21 架构重构记录

为避免 "函数太大 → AI 写错" 风险做了完整架构重构（跟 v8 训练并行, 不影响进程中模型）:

- 大函数全部 < 50 行（from 4 个 >100 行 + 13 个 >50 行 → 0 个 >100 + 4 个 >50 都是 CLI main）
- `features.py` 改 FeatureSpec 注册制: `FEATURE_GROUPS` tuple 声明 10 组 helper, `FEATURE_COLS` 自动推导. 每个 `_feat_*` docstring 含 "**手动推导**" 段, 逐个特征写明 D 日能看到什么、shift(1) 后安全性 — 反前视审计可一眼过
- `model/heads.py` 引入 `Head` 基类 + `HEAD_REGISTRY[phase] → HeadWiring(cls, wrap)`, `ActorCritic.forward` 消除 isinstance 链
- 新 `training/` 包: `BaseTrainer` / `PopTrainer` / `PPOTrainer`, 消除 `train.py` vs `train_pop.py` 的 200 行 epoch/早停/ckpt 重复. `train.py` / `train_pop.py` 降为 thin CLI
- 新 `evaluation/` 包: `metrics.py` (AUC/Sharpe/…) + `cube.py` (批量推理) + `env.py` (step 推理), eval.py / Trainer eval / train_pop eval 全共用
- `ppo_update` / `_assemble_cube` / `select_per_industry_top_k` / `make_hit_labels` 全拆 helpers
- 质量门: ruff 0 errors, mypy 0 errors (32 files), pyright 与 baseline HEAD 持平 (12 errors 全是 pandas-stubs), lint-imports 4 contracts KEPT, pytest 247/247 passed

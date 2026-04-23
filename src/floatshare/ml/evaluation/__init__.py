"""评估工具 — 共享于 training / offline eval.

metrics : pure-numpy 指标 (AUC / sharpe / drawdown / top-K precision)
cube    : cube-based batched 推理 (监督 AUC/P@K, 不走 env)
env     : env-based step 推理 (PPO sharpe, ckpt 评估, signal 生成)
"""

from floatshare.ml.evaluation.cube import classifier_metrics as classifier_metrics
from floatshare.ml.evaluation.env import (
    decode_eval_weights as decode_eval_weights,
)
from floatshare.ml.evaluation.env import (
    run_deterministic_rollout as run_deterministic_rollout,
)
from floatshare.ml.evaluation.metrics import (
    compute_max_drawdown as compute_max_drawdown,
)
from floatshare.ml.evaluation.metrics import (
    compute_sharpe as compute_sharpe,
)
from floatshare.ml.evaluation.metrics import (
    compute_turnover_avg as compute_turnover_avg,
)
from floatshare.ml.evaluation.metrics import (
    rank_based_auc as rank_based_auc,
)
from floatshare.ml.evaluation.metrics import (
    top_k_precision as top_k_precision,
)

#!/usr/bin/env bash
#
# 每日 Pop warm-start + 今日 top-K 推送.
#
# 策略:
#   1. 确定 warm-start 源 ckpt: 优先 pop_warm_latest.pt, fallback pop_anchor.pt
#   2. 若两者都没, 跳过 (必须先手动 cold-start 产 anchor)
#   3. 跑 3 epoch warm-start, 覆盖 pop_warm_latest.pt (不动 anchor)
#   4. 训完自动 push_today_picks (Bark)
#
# 日志: logs/daily-train-YYYY-MM-DD.log
#
# launchd 调用 (见 scripts/com.floatshare.daily-train.plist).

set -euo pipefail

PROJECT_ROOT="/Users/gakki/dev/floatshare"
cd "$PROJECT_ROOT"

# Lock: mkdir 原子性 (macOS 无 flock). 上一轮没跑完直接跳过, 避免 MPS 并发 OOM.
LOCK_DIR="/tmp/floatshare-daily-train.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[daily-train] 前一轮还在跑 (lock $LOCK_DIR), 跳过本次" >&2
  exit 0
fi
# shellcheck disable=SC2064  # 故意立即展开 $LOCK_DIR (不需要延迟求值)
trap "rmdir $LOCK_DIR" EXIT INT TERM

# shellcheck source=/dev/null
source .venv/bin/activate

TODAY=$(date +%Y-%m-%d)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/daily-train-${TODAY}.log"
mkdir -p "$LOG_DIR"

CKPT_DIR="data/ml/ckpts"
ANCHOR="${CKPT_DIR}/pop_anchor.pt"
WARM="${CKPT_DIR}/pop_warm_latest.pt"

{
  echo "===== Daily train started at $(date '+%Y-%m-%d %H:%M:%S') ====="

  # 1. 选 warm source
  if [ -f "$WARM" ]; then
    RESUME="$WARM"
    SOURCE_TAG="warm_latest"
  elif [ -f "$ANCHOR" ]; then
    RESUME="$ANCHOR"
    SOURCE_TAG="anchor (warm 链首轮)"
  else
    echo "ERROR: 没有 ckpt 可 warm-start — 先跑 cold-start 产 pop_anchor.pt"
    echo "  floatshare-train-pop --seed 42 --ckpt-out pop_anchor.pt --no-push-picks"
    exit 1
  fi
  echo "warm source: $RESUME ($SOURCE_TAG)"

  # 2. Pop warm-start 3 epoch, 到昨天 (today 的 daily_basic 等可能还没完全就绪)
  YESTERDAY=$(date -v-1d +%Y-%m-%d)
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    floatshare-train-pop \
      --seed 42 \
      --resume-from "$RESUME" \
      --ckpt-out pop_warm_latest.pt \
      --epochs 3 \
      --end "$YESTERDAY" \
      --eval-every 1 \
      --push-top-k 10 \
      --note "daily warm-start from $(basename "$RESUME") @ $TODAY" \
    || { echo "ERROR: train exit $?"; exit 1; }

  # 3. Drift alert: warm AUC 比 anchor 掉 > 5% 就 Bark 告警 (metrics.db 读)
  python scripts/drift_check.py || echo "drift check exit $?, 不阻塞"

  echo "===== Daily train ended at $(date '+%Y-%m-%d %H:%M:%S') ====="
} >> "$LOG_FILE" 2>&1

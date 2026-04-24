#!/usr/bin/env bash
#
# 每日数据同步入口 — 由 launchd / cron 调用。
#
# 行为:
#   1. 增量拉所有 per-code + batch 表 (watermark 自动决定起点, end=今天)
#   2. 单独补当日 top_list / top_inst (按日批量，需要显式 start/end)
#   3. 完成后通过 floatshare-sync 内置 notify 推 Bark
#
# 日志: logs/daily-sync-YYYY-MM-DD.log (按天滚动)
#
# 手动测试: bash scripts/daily-sync.sh

set -euo pipefail

PROJECT_ROOT="/Users/gakki/dev/floatshare"
cd "$PROJECT_ROOT"

# shellcheck source=/dev/null
source .venv/bin/activate

TODAY=$(date +%Y-%m-%d)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/daily-sync-${TODAY}.log"
mkdir -p "$LOG_DIR"

{
  echo "===== Daily sync started at $(date '+%Y-%m-%d %H:%M:%S') ====="

  YESTERDAY=$(date -v-7d +%Y-%m-%d)  # 7 天回溯, 容错周末/补漏

  # 1. 非财务表增量: by-date 6 个快表 + 其它 per-code (但不含 chip_dist)
  # 显式排掉:
  #   - 6 个 one-shot 财务表: income/balancesheet/cashflow/fina_indicator/
  #     stk_holder_number/dividend, 走 [2] 的 bulk VIP 路径 ~3600× 加速
  #   - chip_dist: 老实现 per-code 5500 股/天 ~15min, tushare 无 bulk API, 走 [3]
  #     的 "按训练 universe 拉" 路径 (~465 股, ~2min)
  echo "--- [1/4] 非财务表增量 (by-date + per-code, window=$YESTERDAY..$TODAY) ---"
  floatshare-sync --all-stocks --by-date --start "$YESTERDAY" --end "$TODAY" \
    --include lifecycle index_weight industry concept \
              raw_daily adj_factor daily_basic chip_perf moneyflow margin_detail \
              forecast moneyflow_hsgt index_daily cn_cpi cn_ppi shibor fx_daily \
              broker_picks \
    || echo "main sync exit $?"

  # 2. 财务 6 表 bulk (tushare VIP income_vip/balancesheet_vip/cashflow_vip/
  # fina_indicator_vip + stk_holdernumber/dividend, 按 ann_date 全市场一把)
  echo "--- [2/4] 财务 bulk (6 表, bulk VIP, window=$YESTERDAY..$TODAY) ---"
  python scripts/sync_financials_bulk.py --start "$YESTERDAY" --end "$TODAY" \
    || echo "financials bulk sync exit $?"

  # 3. chip_dist 只对训练 universe 拉 (~465 股, per-industry top-15)
  # tushare 强制 per-code, 只能减量不能 bulk. universe 按今日 snapshot 动态算.
  echo "--- [3/4] chip_dist for training universe (~465 股) ---"
  UNIVERSE=$(python -c "
from floatshare.ml.data.universe import select_per_industry_top_k
codes = select_per_industry_top_k('data/floatshare.db', '$TODAY')
print(' '.join(codes))
" 2>/dev/null)
  if [ -n "$UNIVERSE" ]; then
    # shellcheck disable=SC2086  # 故意 word-split UNIVERSE
    floatshare-sync --codes $UNIVERSE --include chip_dist \
      --start "$YESTERDAY" --end "$TODAY" \
      || echo "chip_dist universe sync exit $?"
  else
    echo "universe empty, skip chip_dist (可能 today snapshot 数据不全)"
  fi

  # 4. 当日龙虎榜 (按日批量)
  echo "--- [4/4] 当日龙虎榜 ---"
  floatshare-sync --include top_list top_inst --start "$TODAY" --end "$TODAY" \
    || echo "top_list sync exit $?"

  echo "===== Daily sync ended at $(date '+%Y-%m-%d %H:%M:%S') ====="
} >> "$LOG_FILE" 2>&1

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

  # 1. 全表增量 (raw_daily/adj_factor/daily_basic/moneyflow/margin_detail/chip_*/
  #    income/balancesheet/cashflow/fina_indicator/stk_holder_number/dividend/
  #    forecast + lifecycle/industry/moneyflow_hsgt/cn_cpi/cn_ppi/shibor/fx_daily/index_weight)
  # top_list/top_inst 在 main 里因 `if start and end` 被跳过，下一步单独跑
  #
  # --by-date: 6 个高频日表 (raw_daily/daily_basic/moneyflow/margin_detail/
  # chip_perf/adj_factor) 走 trade_date 全市场维度, 1 次 API 拿 5500 票,
  # 比 per-code 提速 ~1000x. 财报类/chip_dist 仍走 per-code.
  YESTERDAY=$(date -v-7d +%Y-%m-%d)  # 7 天回溯, 容错周末/补漏
  echo "--- [1/2] 全表增量 (by-date 加速, window=$YESTERDAY..$TODAY) ---"
  floatshare-sync --all-stocks --by-date --start "$YESTERDAY" --end "$TODAY" \
    || echo "main sync exit $?"

  # 2. 当日龙虎榜 (按日批量)
  echo "--- [2/2] 当日龙虎榜 ---"
  floatshare-sync --include top_list top_inst --start "$TODAY" --end "$TODAY" \
    || echo "top_list sync exit $?"

  echo "===== Daily sync ended at $(date '+%Y-%m-%d %H:%M:%S') ====="
} >> "$LOG_FILE" 2>&1

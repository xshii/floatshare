#!/usr/bin/env bash
#
# 安装 / 卸载 floatshare 每日 sync 的 launchd job。
#
# 用法:
#   bash scripts/install-daily-sync.sh           # 安装并启动
#   bash scripts/install-daily-sync.sh uninstall # 停止并卸载
#   bash scripts/install-daily-sync.sh status    # 查看状态
#   bash scripts/install-daily-sync.sh test      # 立即触发一次 (不等 20:00)

set -euo pipefail

LABEL="com.floatshare.daily-sync"
PROJECT_ROOT="/Users/gakki/dev/floatshare"
SRC_PLIST="${PROJECT_ROOT}/scripts/${LABEL}.plist"
DST_PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"

action="${1:-install}"

case "$action" in
  install)
    echo "→ 安装 ${LABEL} (每日 20:00 触发)"
    chmod +x "${PROJECT_ROOT}/scripts/daily-sync.sh"
    cp "$SRC_PLIST" "$DST_PLIST"
    launchctl unload "$DST_PLIST" 2>/dev/null || true
    launchctl load -w "$DST_PLIST"
    echo "✓ 已安装. 查看状态: bash $0 status"
    ;;

  uninstall)
    echo "→ 卸载 ${LABEL}"
    launchctl unload -w "$DST_PLIST" 2>/dev/null || true
    rm -f "$DST_PLIST"
    echo "✓ 已卸载"
    ;;

  status)
    if launchctl list | grep -q "$LABEL"; then
      echo "✓ 已加载:"
      launchctl list | grep "$LABEL"
      echo ""
      echo "下次触发: 今天/明天 20:00"
      echo "日志: ${PROJECT_ROOT}/logs/daily-sync-*.log"
    else
      echo "✗ 未安装. 运行: bash $0 install"
    fi
    ;;

  test)
    echo "→ 立即触发一次 (调试用，不等 20:00)"
    launchctl start "$LABEL"
    echo "✓ 已触发。查看日志:"
    echo "  tail -f ${PROJECT_ROOT}/logs/daily-sync-$(date +%Y-%m-%d).log"
    ;;

  *)
    echo "用法: $0 [install|uninstall|status|test]"
    exit 1
    ;;
esac

#!/bin/bash
# FloatShare 生产环境安装脚本

set -e

# 配置
APP_NAME="floatshare"
APP_DIR="/opt/floatshare"
LOG_DIR="/var/log/floatshare"
USER="floatshare"

echo "=== FloatShare 安装脚本 ==="

# 检查 root 权限
if [ "$EUID" -ne 0 ]; then
    echo "请使用 root 权限运行"
    exit 1
fi

# 创建用户
if ! id "$USER" &>/dev/null; then
    echo "创建用户 $USER..."
    useradd -r -s /bin/false "$USER"
fi

# 创建目录
echo "创建目录..."
mkdir -p "$APP_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$APP_DIR/data"

# 复制文件
echo "复制文件..."
cp -r src "$APP_DIR/"
cp -r config "$APP_DIR/"
cp -r deploy "$APP_DIR/"
cp requirements.txt "$APP_DIR/"

# 创建虚拟环境
echo "创建 Python 虚拟环境..."
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

# 设置权限
echo "设置权限..."
chown -R "$USER:$USER" "$APP_DIR"
chown -R "$USER:$USER" "$LOG_DIR"
chmod +x "$APP_DIR/deploy/scripts/"*.py
chmod +x "$APP_DIR/deploy/scripts/"*.sh

# 安装 systemd 服务
echo "安装 systemd 服务..."
cp "$APP_DIR/deploy/systemd/floatshare-health.service" /etc/systemd/system/
cp "$APP_DIR/deploy/systemd/floatshare-health.timer" /etc/systemd/system/
systemctl daemon-reload

# 启用定时器
echo "启用健康检查定时器..."
systemctl enable floatshare-health.timer
systemctl start floatshare-health.timer

echo ""
echo "=== 安装完成 ==="
echo ""
echo "目录: $APP_DIR"
echo "日志: $LOG_DIR"
echo ""
echo "常用命令:"
echo "  # 手动运行健康检查"
echo "  systemctl start floatshare-health.service"
echo ""
echo "  # 查看定时器状态"
echo "  systemctl status floatshare-health.timer"
echo ""
echo "  # 查看日志"
echo "  tail -f $LOG_DIR/health.log"
echo ""
echo "  # 查看最新检查结果"
echo "  cat $LOG_DIR/health_latest.json"

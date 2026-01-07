"""告警管理"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime
from enum import Enum
import json


class AlertLevel(Enum):
    """告警级别"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型"""

    SYSTEM = "system"  # 系统告警
    STRATEGY = "strategy"  # 策略告警
    RISK = "risk"  # 风控告警
    TRADE = "trade"  # 交易告警
    DATA = "data"  # 数据告警


@dataclass
class Alert:
    """告警"""

    level: AlertLevel
    type: AlertType
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    data: Optional[Dict[str, Any]] = None
    acknowledged: bool = False

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "level": self.level.value,
            "type": self.type.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "acknowledged": self.acknowledged,
        }

    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self._alerts: List[Alert] = []
        self._handlers: Dict[AlertLevel, List[Callable]] = {
            AlertLevel.INFO: [],
            AlertLevel.WARNING: [],
            AlertLevel.ERROR: [],
            AlertLevel.CRITICAL: [],
        }
        self._max_alerts = 1000

    def add_handler(self, level: AlertLevel, handler: Callable) -> None:
        """
        添加告警处理器

        Args:
            level: 告警级别
            handler: 处理函数，接收Alert参数
        """
        self._handlers[level].append(handler)

    def send(
        self,
        level: AlertLevel,
        type_: AlertType,
        title: str,
        message: str,
        source: str = "",
        data: Optional[Dict] = None,
    ) -> Alert:
        """发送告警"""
        alert = Alert(
            level=level,
            type=type_,
            title=title,
            message=message,
            source=source,
            data=data,
        )

        self._alerts.append(alert)

        # 限制告警数量
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]

        # 触发处理器
        for handler in self._handlers[level]:
            try:
                handler(alert)
            except Exception as e:
                print(f"告警处理器异常: {e}")

        return alert

    def info(self, title: str, message: str, **kwargs) -> Alert:
        """发送INFO级别告警"""
        return self.send(AlertLevel.INFO, AlertType.SYSTEM, title, message, **kwargs)

    def warning(self, title: str, message: str, **kwargs) -> Alert:
        """发送WARNING级别告警"""
        return self.send(AlertLevel.WARNING, AlertType.SYSTEM, title, message, **kwargs)

    def error(self, title: str, message: str, **kwargs) -> Alert:
        """发送ERROR级别告警"""
        return self.send(AlertLevel.ERROR, AlertType.SYSTEM, title, message, **kwargs)

    def critical(self, title: str, message: str, **kwargs) -> Alert:
        """发送CRITICAL级别告警"""
        return self.send(AlertLevel.CRITICAL, AlertType.SYSTEM, title, message, **kwargs)

    def risk_alert(self, title: str, message: str, **kwargs) -> Alert:
        """发送风控告警"""
        return self.send(AlertLevel.WARNING, AlertType.RISK, title, message, **kwargs)

    def trade_alert(self, title: str, message: str, **kwargs) -> Alert:
        """发送交易告警"""
        return self.send(AlertLevel.INFO, AlertType.TRADE, title, message, **kwargs)

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        type_: Optional[AlertType] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """获取告警"""
        alerts = self._alerts

        if level:
            alerts = [a for a in alerts if a.level == level]

        if type_:
            alerts = [a for a in alerts if a.type == type_]

        return alerts[-limit:]

    def get_unacknowledged(self) -> List[Alert]:
        """获取未确认告警"""
        return [a for a in self._alerts if not a.acknowledged]

    def acknowledge(self, alert: Alert) -> None:
        """确认告警"""
        alert.acknowledged = True

    def acknowledge_all(self) -> int:
        """确认所有告警"""
        count = 0
        for alert in self._alerts:
            if not alert.acknowledged:
                alert.acknowledged = True
                count += 1
        return count

    def clear(self) -> None:
        """清空告警"""
        self._alerts.clear()


# 预定义处理器
def console_handler(alert: Alert) -> None:
    """控制台输出处理器"""
    level_colors = {
        AlertLevel.INFO: "\033[94m",  # 蓝色
        AlertLevel.WARNING: "\033[93m",  # 黄色
        AlertLevel.ERROR: "\033[91m",  # 红色
        AlertLevel.CRITICAL: "\033[95m",  # 紫色
    }
    reset = "\033[0m"

    color = level_colors.get(alert.level, "")
    print(f"{color}[{alert.level.value.upper()}] {alert.title}: {alert.message}{reset}")


def file_handler(log_path: str) -> Callable:
    """文件输出处理器工厂"""
    def handler(alert: Alert) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(alert.to_json() + "\n")
    return handler

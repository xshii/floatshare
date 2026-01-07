"""健康检查"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum


class HealthStatus(Enum):
    """健康状态"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """健康检查结果"""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict] = None


class HealthChecker:
    """健康检查器"""

    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, HealthCheckResult] = {}

    def register_check(self, name: str, check_func: Callable) -> None:
        """
        注册健康检查

        Args:
            name: 检查名称
            check_func: 检查函数，返回 (status, message, details)
        """
        self._checks[name] = check_func

    def run_check(self, name: str) -> HealthCheckResult:
        """运行单个检查"""
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="检查未注册",
                timestamp=datetime.now(),
            )

        try:
            result = self._checks[name]()
            if isinstance(result, tuple):
                status, message, details = result
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                message = "OK" if result else "Failed"
                details = None

            check_result = HealthCheckResult(
                name=name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                details=details,
            )
        except Exception as e:
            check_result = HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"检查异常: {str(e)}",
                timestamp=datetime.now(),
            )

        self._results[name] = check_result
        return check_result

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """运行所有检查"""
        for name in self._checks:
            self.run_check(name)
        return self._results

    def get_overall_status(self) -> HealthStatus:
        """获取整体状态"""
        if not self._results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in self._results.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_summary(self) -> Dict:
        """获取摘要"""
        return {
            "overall_status": self.get_overall_status().value,
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "timestamp": result.timestamp.isoformat(),
                }
                for name, result in self._results.items()
            },
        }


# 预定义检查函数
def check_data_source(loader) -> tuple:
    """检查数据源连接"""
    try:
        # 尝试获取股票列表
        df = loader.get_stock_list()
        if not df.empty:
            return HealthStatus.HEALTHY, "数据源正常", {"stock_count": len(df)}
        else:
            return HealthStatus.WARNING, "数据源返回空数据", None
    except Exception as e:
        return HealthStatus.CRITICAL, f"数据源异常: {e}", None


def check_database(storage) -> tuple:
    """检查数据库连接"""
    try:
        # 尝试简单查询
        storage.execute("SELECT 1")
        return HealthStatus.HEALTHY, "数据库正常", None
    except Exception as e:
        return HealthStatus.CRITICAL, f"数据库异常: {e}", None


def check_broker(broker) -> tuple:
    """检查券商连接"""
    if broker.is_connected():
        balance = broker.get_balance()
        return HealthStatus.HEALTHY, "券商连接正常", balance
    else:
        return HealthStatus.CRITICAL, "券商未连接", None

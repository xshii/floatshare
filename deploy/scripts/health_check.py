#!/usr/bin/env python3
"""
生产环境数据源健康检查

用途：
- 定期检测各数据源API是否可用
- 记录响应时间和成功率
- 失败时发送告警

使用：
    # 单次运行
    python deploy/scripts/health_check.py

    # cron 定时任务 (每小时检测)
    0 * * * * cd /app && python deploy/scripts/health_check.py >> /var/log/floatshare/health.log 2>&1

    # systemd timer (推荐)
    见 deploy/systemd/health-check.timer
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """检查结果"""
    source: str
    check_type: str
    success: bool
    response_time: float
    error: Optional[str] = None
    data_count: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class DataSourceHealthChecker:
    """数据源健康检查器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.results: List[CheckResult] = []

        # 测试用的股票代码
        self.test_codes = self.config.get("test_codes", [
            "000001.SZ",  # 平安银行
            "600000.SH",  # 浦发银行
        ])

        # 告警配置
        self.alert_webhook = self.config.get("alert_webhook")
        self.alert_email = self.config.get("alert_email")

    def check_akshare(self) -> List[CheckResult]:
        """检查 AKShare 数据源"""
        results = []

        try:
            import akshare as ak

            # 检查日线数据接口
            for code in self.test_codes:
                ticker = code.split(".")[0]
                start = time.time()
                try:
                    end_date = date.today()
                    start_date = end_date - timedelta(days=7)
                    df = ak.stock_zh_a_hist(
                        symbol=ticker,
                        period="daily",
                        start_date=start_date.strftime("%Y%m%d"),
                        end_date=end_date.strftime("%Y%m%d"),
                        adjust=""
                    )
                    elapsed = time.time() - start
                    results.append(CheckResult(
                        source="akshare",
                        check_type=f"daily_{code}",
                        success=True,
                        response_time=elapsed,
                        data_count=len(df),
                    ))
                    logger.info(f"[akshare] 日线 {code}: {len(df)} 条, {elapsed:.2f}s")
                except Exception as e:
                    results.append(CheckResult(
                        source="akshare",
                        check_type=f"daily_{code}",
                        success=False,
                        response_time=time.time() - start,
                        error=str(e),
                    ))
                    logger.error(f"[akshare] 日线 {code} 失败: {e}")

        except ImportError:
            logger.error("[akshare] 未安装")
            results.append(CheckResult(
                source="akshare",
                check_type="import",
                success=False,
                response_time=0,
                error="akshare not installed",
            ))

        return results

    def check_tushare(self) -> List[CheckResult]:
        """检查 Tushare 数据源"""
        results = []

        try:
            import tushare as ts

            token = os.environ.get("TUSHARE_TOKEN") or self.config.get("tushare_token")
            if not token:
                logger.warning("[tushare] 未配置 TUSHARE_TOKEN，跳过")
                return results

            pro = ts.pro_api(token)

            # 检查1: 股票列表
            start = time.time()
            try:
                df = pro.stock_basic(exchange='', list_status='L')
                elapsed = time.time() - start
                results.append(CheckResult(
                    source="tushare",
                    check_type="stock_list",
                    success=True,
                    response_time=elapsed,
                    data_count=len(df),
                ))
                logger.info(f"[tushare] 股票列表: {len(df)} 只, {elapsed:.2f}s")
            except Exception as e:
                results.append(CheckResult(
                    source="tushare",
                    check_type="stock_list",
                    success=False,
                    response_time=time.time() - start,
                    error=str(e),
                ))
                logger.error(f"[tushare] 股票列表失败: {e}")

            # 检查2: 日线数据
            for code in self.test_codes[:1]:
                start = time.time()
                try:
                    end_date = date.today().strftime("%Y%m%d")
                    start_date = (date.today() - timedelta(days=7)).strftime("%Y%m%d")
                    df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
                    elapsed = time.time() - start
                    results.append(CheckResult(
                        source="tushare",
                        check_type=f"daily_{code}",
                        success=True,
                        response_time=elapsed,
                        data_count=len(df),
                    ))
                    logger.info(f"[tushare] 日线 {code}: {len(df)} 条, {elapsed:.2f}s")
                except Exception as e:
                    results.append(CheckResult(
                        source="tushare",
                        check_type=f"daily_{code}",
                        success=False,
                        response_time=time.time() - start,
                        error=str(e),
                    ))
                    logger.error(f"[tushare] 日线 {code} 失败: {e}")

        except ImportError:
            logger.warning("[tushare] 未安装")

        return results

    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        logger.info("=" * 50)
        logger.info("开始数据源健康检查")
        logger.info("=" * 50)

        start_time = time.time()

        # 执行各数据源检查
        self.results.extend(self.check_akshare())
        self.results.extend(self.check_tushare())

        total_time = time.time() - start_time

        # 统计
        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
        failed = total - success

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_checks": total,
            "success": success,
            "failed": failed,
            "success_rate": f"{100 * success / total:.1f}%" if total > 0 else "N/A",
            "total_time": f"{total_time:.2f}s",
            "results": [asdict(r) for r in self.results],
        }

        logger.info("=" * 50)
        logger.info(f"检查完成: {success}/{total} 成功, 耗时 {total_time:.2f}s")
        logger.info("=" * 50)

        # 如果有失败，发送告警
        if failed > 0:
            self._send_alert(summary)

        return summary

    def _send_alert(self, summary: Dict):
        """发送告警"""
        failed_checks = [r for r in self.results if not r.success]

        message = f"⚠️ FloatShare 数据源告警\n"
        message += f"时间: {summary['timestamp']}\n"
        message += f"失败: {len(failed_checks)} 项\n\n"

        for r in failed_checks:
            message += f"- [{r.source}] {r.check_type}: {r.error}\n"

        logger.warning(f"告警内容:\n{message}")

        # Webhook 告警 (钉钉/飞书/企业微信等)
        if self.alert_webhook:
            try:
                import requests
                requests.post(self.alert_webhook, json={
                    "msgtype": "text",
                    "text": {"content": message}
                }, timeout=10)
                logger.info("Webhook 告警已发送")
            except Exception as e:
                logger.error(f"Webhook 告警发送失败: {e}")

    def save_results(self, output_path: str):
        """保存结果到文件"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="数据源健康检查")
    parser.add_argument("--output", "-o", help="输出结果到JSON文件")
    parser.add_argument("--webhook", help="告警 Webhook URL")
    parser.add_argument("--config", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = json.load(f)

    if args.webhook:
        config["alert_webhook"] = args.webhook

    # 运行检查
    checker = DataSourceHealthChecker(config)
    summary = checker.run_all_checks()

    # 保存结果
    if args.output:
        checker.save_results(args.output)

    # 输出到 stdout (方便 cron 日志)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 返回码: 有失败则返回 1
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()

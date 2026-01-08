"""任务调度模块

基于 APScheduler 提供定时任务能力：
- 定时数据同步
- 定时策略回测
- Cron 表达式支持
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobEvent
from apscheduler.executors.pool import ProcessPoolExecutor, ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskScheduler:
    """
    任务调度器

    Example:
        scheduler = TaskScheduler()

        # 每天 0:00 同步数据
        scheduler.add_daily_job("sync_data", sync_function, hour=0, minute=0)

        # 每 5 分钟检查一次
        scheduler.add_interval_job("health_check", check_function, minutes=5)

        scheduler.start()
    """

    def __init__(
        self,
        max_workers: int = 10,
        process_workers: int = 2,
    ):
        """
        初始化调度器

        Args:
            max_workers: 线程池大小
            process_workers: 进程池大小
        """
        executors = {
            "default": ThreadPoolExecutor(max_workers),
            "processpool": ProcessPoolExecutor(process_workers),
        }

        job_defaults = {
            "coalesce": True,  # 错过的任务只执行一次
            "max_instances": 1,  # 同一任务最多同时运行一个实例
            "misfire_grace_time": 60,  # 错过执行时间的容忍秒数
        }

        self.scheduler = BackgroundScheduler(
            executors=executors,
            job_defaults=job_defaults,
        )

        self._job_history: Dict[str, List[Dict[str, Any]]] = {}

        # 注册事件监听
        self.scheduler.add_listener(
            self._on_job_executed,
            EVENT_JOB_EXECUTED,
        )
        self.scheduler.add_listener(
            self._on_job_error,
            EVENT_JOB_ERROR,
        )

    def start(self) -> None:
        """启动调度器"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("任务调度器已启动")

    def shutdown(self, wait: bool = True) -> None:
        """关闭调度器"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("任务调度器已关闭")

    def add_cron_job(
        self,
        job_id: str,
        func: Callable,
        cron_expr: str,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        executor: str = "default",
    ) -> str:
        """
        添加 Cron 任务

        Args:
            job_id: 任务 ID
            func: 任务函数
            cron_expr: Cron 表达式 (分 时 日 月 周)
            args: 位置参数
            kwargs: 关键字参数
            executor: 执行器 ("default" 或 "processpool")

        Returns:
            任务 ID
        """
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError("Cron 表达式格式: 分 时 日 月 周")

        trigger = CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )

        self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            args=args or (),
            kwargs=kwargs or {},
            executor=executor,
            replace_existing=True,
        )

        logger.info(f"添加 Cron 任务: {job_id}, 表达式: {cron_expr}")
        return job_id

    def add_daily_job(
        self,
        job_id: str,
        func: Callable,
        hour: int = 0,
        minute: int = 0,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
    ) -> str:
        """
        添加每日任务

        Args:
            job_id: 任务 ID
            func: 任务函数
            hour: 执行小时 (0-23)
            minute: 执行分钟 (0-59)
        """
        trigger = CronTrigger(hour=hour, minute=minute)

        self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            args=args or (),
            kwargs=kwargs or {},
            replace_existing=True,
        )

        logger.info(f"添加每日任务: {job_id}, 时间: {hour:02d}:{minute:02d}")
        return job_id

    def add_interval_job(
        self,
        job_id: str,
        func: Callable,
        seconds: int = 0,
        minutes: int = 0,
        hours: int = 0,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
    ) -> str:
        """
        添加间隔任务

        Args:
            job_id: 任务 ID
            func: 任务函数
            seconds: 间隔秒数
            minutes: 间隔分钟
            hours: 间隔小时
        """
        trigger = IntervalTrigger(
            seconds=seconds,
            minutes=minutes,
            hours=hours,
        )

        self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            args=args or (),
            kwargs=kwargs or {},
            replace_existing=True,
        )

        interval = timedelta(seconds=seconds, minutes=minutes, hours=hours)
        logger.info(f"添加间隔任务: {job_id}, 间隔: {interval}")
        return job_id

    def remove_job(self, job_id: str) -> None:
        """移除任务"""
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"移除任务: {job_id}")
        except Exception as e:
            logger.warning(f"移除任务失败: {job_id}, {e}")

    def pause_job(self, job_id: str) -> None:
        """暂停任务"""
        self.scheduler.pause_job(job_id)
        logger.info(f"暂停任务: {job_id}")

    def resume_job(self, job_id: str) -> None:
        """恢复任务"""
        self.scheduler.resume_job(job_id)
        logger.info(f"恢复任务: {job_id}")

    def run_job_now(self, job_id: str) -> None:
        """立即执行任务"""
        job = self.scheduler.get_job(job_id)
        if job:
            job.modify(next_run_time=datetime.now())
            logger.info(f"立即执行任务: {job_id}")

    def get_jobs(self) -> List[Dict[str, Any]]:
        """获取所有任务信息"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time,
                "trigger": str(job.trigger),
            })
        return jobs

    def get_job_history(self, job_id: str) -> List[Dict[str, Any]]:
        """获取任务执行历史"""
        return self._job_history.get(job_id, [])

    def _on_job_executed(self, event: JobEvent) -> None:
        """任务执行完成回调"""
        self._record_job_event(event.job_id, JobStatus.COMPLETED)
        logger.info(f"任务执行完成: {event.job_id}")

    def _on_job_error(self, event: JobEvent) -> None:
        """任务执行错误回调"""
        self._record_job_event(
            event.job_id,
            JobStatus.FAILED,
            error=str(event.exception),
        )
        logger.error(f"任务执行失败: {event.job_id}, 错误: {event.exception}")

    def _record_job_event(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
    ) -> None:
        """记录任务事件"""
        if job_id not in self._job_history:
            self._job_history[job_id] = []

        self._job_history[job_id].append({
            "timestamp": datetime.now().isoformat(),
            "status": status.value,
            "error": error,
        })

        # 只保留最近 100 条记录
        if len(self._job_history[job_id]) > 100:
            self._job_history[job_id] = self._job_history[job_id][-100:]


# ============================================================
# 预定义任务
# ============================================================


def create_data_sync_scheduler(
    sync_func: Callable,
    hour: int = 18,
    minute: int = 0,
) -> TaskScheduler:
    """
    创建数据同步调度器

    默认每天 18:00 执行（A股收盘后）
    """
    scheduler = TaskScheduler()
    scheduler.add_daily_job("daily_sync", sync_func, hour=hour, minute=minute)
    return scheduler


def create_backtest_scheduler(
    backtest_func: Callable,
    cron_expr: str = "0 2 * * 6",  # 每周六凌晨 2 点
) -> TaskScheduler:
    """
    创建回测调度器

    默认每周六执行
    """
    scheduler = TaskScheduler()
    scheduler.add_cron_job("weekly_backtest", backtest_func, cron_expr)
    return scheduler


# ============================================================
# 全局调度器
# ============================================================


_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """获取全局调度器"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


def set_scheduler(scheduler: TaskScheduler) -> None:
    """设置全局调度器"""
    global _scheduler
    _scheduler = scheduler

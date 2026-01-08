"""异步数据加载器

提供异步数据获取能力：
- 异步 HTTP 请求
- 并发数据获取
- 异步数据库操作
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class AsyncDataSource(ABC):
    """异步数据源基类"""

    @abstractmethod
    async def fetch_daily(
        self,
        code: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取日线数据"""
        pass

    @abstractmethod
    async def fetch_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        pass

    async def close(self) -> None:
        """关闭连接"""
        pass


class AsyncHttpClient:
    """异步 HTTP 客户端"""

    def __init__(
        self,
        timeout: int = 30,
        max_connections: int = 100,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_connections = max_connections
        self.headers = headers or {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=self.max_connections)
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector,
                headers=self.headers,
            )
        return self._session

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """GET 请求"""
        session = await self._get_session()
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.text()

    async def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """GET 请求返回 JSON"""
        session = await self._get_session()
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> str:
        """POST 请求"""
        session = await self._get_session()
        async with session.post(url, data=data, json=json) as response:
            response.raise_for_status()
            return await response.text()

    async def close(self) -> None:
        """关闭会话"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "AsyncHttpClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()


class AsyncDataLoader:
    """异步数据加载器"""

    def __init__(
        self,
        sources: Optional[List[AsyncDataSource]] = None,
        max_concurrent: int = 10,
    ):
        self.sources = sources or []
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_daily(
        self,
        code: str,
        start_date: date,
        end_date: date,
    ) -> Tuple[pd.DataFrame, str]:
        """
        获取日线数据，依次尝试各数据源

        Returns:
            (DataFrame, source_name) 元组
        """
        async with self._semaphore:
            for source in self.sources:
                try:
                    df = await source.fetch_daily(code, start_date, end_date)
                    if not df.empty:
                        return df, source.__class__.__name__
                except Exception as e:
                    logger.warning(f"{source.__class__.__name__} 获取 {code} 失败: {e}")
                    continue

            raise RuntimeError(f"所有数据源获取 {code} 失败")

    async def fetch_batch(
        self,
        codes: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据

        Returns:
            {code: DataFrame} 字典
        """
        tasks = [
            self.fetch_daily(code, start_date, end_date)
            for code in codes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for code, result in zip(codes, results):
            if isinstance(result, Exception):
                logger.error(f"获取 {code} 失败: {result}")
            else:
                df, _ = result
                data[code] = df

        return data

    async def close(self) -> None:
        """关闭所有数据源"""
        for source in self.sources:
            await source.close()


async def run_async(coro):
    """运行异步协程的辅助函数"""
    try:
        loop = asyncio.get_running_loop()
        return await coro
    except RuntimeError:
        return asyncio.run(coro)


def sync_wrapper(async_func):
    """将异步函数包装为同步函数"""
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
            # 已在事件循环中，创建任务
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, async_func(*args, **kwargs))
                return future.result()
        except RuntimeError:
            # 不在事件循环中，直接运行
            return asyncio.run(async_func(*args, **kwargs))
    return wrapper

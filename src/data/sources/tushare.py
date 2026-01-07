"""Tushare数据源适配器"""

from datetime import date
from typing import List, Optional
import pandas as pd

from ..loader import BaseDataSource
from config.settings import settings


class TushareSource(BaseDataSource):
    """Tushare数据源"""

    def __init__(self, token: Optional[str] = None):
        self.token = token or settings.TUSHARE_TOKEN
        self._pro = None

    @property
    def pro(self):
        """延迟初始化tushare pro接口"""
        if self._pro is None:
            try:
                import tushare as ts

                ts.set_token(self.token)
                self._pro = ts.pro_api()
            except ImportError:
                raise ImportError("请先安装tushare: pip install tushare")
            except Exception as e:
                raise RuntimeError(f"Tushare初始化失败: {e}")
        return self._pro

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        df = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
        df = df.rename(
            columns={
                "ts_code": "code",
                "symbol": "ticker",
            }
        )
        return df

    def get_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        adj: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取日线数据

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adj: 复权类型 (None-不复权[默认], qfq-前复权, hfq-后复权)

        Returns:
            DataFrame，价格默认为不复权，包含 adj_factor 列
        """
        start_str = start_date.strftime("%Y%m%d") if start_date else None
        end_str = end_date.strftime("%Y%m%d") if end_date else None

        # 获取不复权行情数据
        df = self.pro.daily(ts_code=code, start_date=start_str, end_date=end_str)

        if df.empty:
            return df

        # 重命名列
        df = df.rename(
            columns={
                "ts_code": "code",
                "vol": "volume",
            }
        )

        # 转换日期格式
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # 获取复权因子
        adj_df = self.pro.adj_factor(ts_code=code, start_date=start_str, end_date=end_str)
        if not adj_df.empty:
            adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])
            df = df.merge(adj_df[["trade_date", "adj_factor"]], on="trade_date", how="left")
            df["adj_factor"] = df["adj_factor"].fillna(1.0)
        else:
            df["adj_factor"] = 1.0

        df = df.sort_values("trade_date")

        # 如果请求复权数据，动态计算
        if adj:
            df = self._adjust_price(df, adj)

        return df.reset_index(drop=True)

    def _adjust_price(self, df: pd.DataFrame, adj: str) -> pd.DataFrame:
        """复权处理（动态计算）"""
        price_cols = ["open", "high", "low", "close"]

        if adj == "hfq":
            # 后复权：直接乘以复权因子
            for col in price_cols:
                df[col] = df[col] * df["adj_factor"]
        elif adj == "qfq":
            # 前复权：归一化到最新价格
            latest_factor = df["adj_factor"].iloc[-1]
            for col in price_cols:
                df[col] = df[col] * df["adj_factor"] / latest_factor

        return df

    def get_minute(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """获取分钟线数据（需要更高权限）"""
        # Tushare分钟线需要积分，这里返回空DataFrame
        return pd.DataFrame()

    def get_index_daily(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """获取指数日线数据"""
        start_str = start_date.strftime("%Y%m%d") if start_date else None
        end_str = end_date.strftime("%Y%m%d") if end_date else None

        df = self.pro.index_daily(ts_code=code, start_date=start_str, end_date=end_str)

        if not df.empty:
            df = df.rename(columns={"ts_code": "code", "vol": "volume"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date")

        return df.reset_index(drop=True)

    def get_financial(
        self,
        code: str,
        report_type: str = "quarterly",
    ) -> pd.DataFrame:
        """获取财务数据"""
        # 获取利润表
        income = self.pro.income(ts_code=code)

        # 获取资产负债表
        balance = self.pro.balancesheet(ts_code=code)

        # 获取财务指标
        indicator = self.pro.fina_indicator(ts_code=code)

        # 合并数据
        if income.empty:
            return pd.DataFrame()

        df = income[["ts_code", "ann_date", "end_date", "revenue", "n_income"]].copy()
        df = df.rename(columns={"ts_code": "code", "n_income": "net_profit"})

        if not indicator.empty:
            indicator = indicator[["end_date", "eps", "bps", "roe", "roa"]]
            df = df.merge(indicator, on="end_date", how="left")

        return df

    def get_trade_calendar(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """获取交易日历"""
        start_str = start_date.strftime("%Y%m%d") if start_date else None
        end_str = end_date.strftime("%Y%m%d") if end_date else None

        df = self.pro.trade_cal(
            exchange="SSE", start_date=start_str, end_date=end_str, is_open=1
        )

        dates = pd.to_datetime(df["cal_date"]).dt.date.tolist()
        return sorted(dates)

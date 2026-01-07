"""数据清洗模块"""

from typing import Optional, List
import pandas as pd
import numpy as np


class DataCleaner:
    """数据清洗器"""

    @staticmethod
    def clean_daily_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗日线数据

        处理内容:
        - 去除重复数据
        - 处理缺失值
        - 处理异常值
        - 排序
        """
        if df.empty:
            return df

        df = df.copy()

        # 去重
        df = df.drop_duplicates(subset=["code", "trade_date"], keep="last")

        # 按日期排序
        df = df.sort_values("trade_date")

        # 处理缺失值
        df = DataCleaner._fill_missing_prices(df)

        # 处理异常值
        df = DataCleaner._handle_outliers(df)

        return df.reset_index(drop=True)

    @staticmethod
    def _fill_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
        """填充缺失的价格数据"""
        price_cols = ["open", "high", "low", "close"]

        for col in price_cols:
            if col in df.columns:
                # 使用前值填充
                df[col] = df[col].fillna(method="ffill")
                # 再用后值填充剩余的
                df[col] = df[col].fillna(method="bfill")

        # 成交量和成交额缺失填0
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)
        if "amount" in df.columns:
            df["amount"] = df["amount"].fillna(0)

        return df

    @staticmethod
    def _handle_outliers(
        df: pd.DataFrame, std_threshold: float = 5.0
    ) -> pd.DataFrame:
        """处理异常值"""
        price_cols = ["open", "high", "low", "close"]

        for col in price_cols:
            if col not in df.columns:
                continue

            # 计算收益率
            returns = df[col].pct_change()

            # 找出超过阈值的异常值
            mean_ret = returns.mean()
            std_ret = returns.std()

            if std_ret > 0:
                outliers = np.abs(returns - mean_ret) > std_threshold * std_ret

                # 将异常值替换为前值
                if outliers.any():
                    df.loc[outliers, col] = np.nan
                    df[col] = df[col].fillna(method="ffill")

        return df

    @staticmethod
    def adjust_price(
        df: pd.DataFrame, adj_factor: pd.Series, method: str = "qfq"
    ) -> pd.DataFrame:
        """
        复权处理

        Args:
            df: 价格数据
            adj_factor: 复权因子
            method: qfq-前复权, hfq-后复权
        """
        df = df.copy()
        price_cols = ["open", "high", "low", "close"]

        if method == "qfq":
            # 前复权：以最新价格为基准向前调整
            factor = adj_factor / adj_factor.iloc[-1]
        elif method == "hfq":
            # 后复权：以最早价格为基准向后调整
            factor = adj_factor / adj_factor.iloc[0]
        else:
            return df

        for col in price_cols:
            if col in df.columns:
                df[col] = df[col] * factor

        return df

    @staticmethod
    def resample(
        df: pd.DataFrame, freq: str = "W", agg_dict: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        重采样（日线转周线/月线）

        Args:
            df: 日线数据
            freq: W-周线, M-月线
            agg_dict: 聚合方式
        """
        if agg_dict is None:
            agg_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }

        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")

        # 只聚合存在的列
        valid_agg = {k: v for k, v in agg_dict.items() if k in df.columns}

        resampled = df.resample(freq).agg(valid_agg)
        resampled = resampled.dropna(how="all")

        return resampled.reset_index()

    @staticmethod
    def merge_data(
        price_df: pd.DataFrame,
        other_df: pd.DataFrame,
        on: List[str] = ["code", "trade_date"],
        how: str = "left",
    ) -> pd.DataFrame:
        """合并数据"""
        return pd.merge(price_df, other_df, on=on, how=how)

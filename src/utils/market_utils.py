"""市场工具函数

提供股票代码市场后缀处理、日期转换等通用功能
"""

from datetime import date
from typing import Optional
import pandas as pd


def add_market_suffix(ticker: str) -> str:
    """
    添加市场后缀

    Args:
        ticker: 纯数字股票代码 (如 "000001")

    Returns:
        带市场后缀的代码 (如 "000001.SZ")
    """
    if ticker.startswith("6"):
        return f"{ticker}.SH"
    elif ticker.startswith(("0", "3")):
        return f"{ticker}.SZ"
    elif ticker.startswith(("4", "8")):
        return f"{ticker}.BJ"
    return ticker


def remove_market_suffix(code: str) -> str:
    """
    移除市场后缀

    Args:
        code: 带市场后缀的代码 (如 "000001.SZ")

    Returns:
        纯数字代码 (如 "000001")
    """
    return code.split(".")[0] if "." in code else code


def get_market(code: str) -> Optional[str]:
    """
    获取市场类型

    Args:
        code: 股票代码（带或不带后缀）

    Returns:
        市场类型: "SH", "SZ", "BJ" 或 None
    """
    if "." in code:
        return code.split(".")[-1].upper()

    ticker = code
    if ticker.startswith("6"):
        return "SH"
    elif ticker.startswith(("0", "3")):
        return "SZ"
    elif ticker.startswith(("4", "8")):
        return "BJ"
    return None


def format_date_str(d: Optional[date], fmt: str = "%Y%m%d", default: str = "") -> str:
    """
    格式化日期为字符串

    Args:
        d: 日期对象
        fmt: 格式字符串
        default: 日期为空时的默认值

    Returns:
        格式化后的日期字符串
    """
    return d.strftime(fmt) if d else default


def ensure_datetime(df: pd.DataFrame, column: str = "trade_date") -> pd.DataFrame:
    """
    确保DataFrame的日期列为datetime类型

    Args:
        df: DataFrame
        column: 日期列名

    Returns:
        处理后的DataFrame
    """
    if column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column])
    return df


def apply_adjustment(
    df: pd.DataFrame,
    adj: Optional[str],
    price_cols: list = None
) -> pd.DataFrame:
    """
    应用复权调整

    Args:
        df: 包含 adj_factor 列的DataFrame
        adj: 复权类型 (None-不复权, "qfq"-前复权, "hfq"-后复权)
        price_cols: 需要调整的价格列

    Returns:
        调整后的DataFrame
    """
    if not adj or "adj_factor" not in df.columns or df.empty:
        return df

    if price_cols is None:
        price_cols = ["open", "high", "low", "close"]

    df = df.copy()

    if adj == "hfq":
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col] * df["adj_factor"]
    elif adj == "qfq":
        latest_factor = df["adj_factor"].iloc[-1]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col] * df["adj_factor"] / latest_factor

    return df

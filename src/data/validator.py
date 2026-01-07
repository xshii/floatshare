"""数据校验模块

用于检测数据异常，防止脏数据入库：
- 价格异常（涨跌幅超限）
- 缺失值检测
- 数据类型校验
- 逻辑校验（高>低，量价匹配）
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """校验结果"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    rows_checked: int
    rows_invalid: int

    @property
    def summary(self) -> str:
        if self.valid:
            return f"校验通过: {self.rows_checked} 行"
        return f"校验失败: {self.rows_invalid}/{self.rows_checked} 行异常"


class DataValidator:
    """数据校验器"""

    def __init__(
        self,
        max_pct_change: float = 20.0,
        max_price: float = 10000.0,
        min_price: float = 0.01,
        allow_zero_volume: bool = True,
    ):
        """
        Args:
            max_pct_change: 最大涨跌幅限制 (%)，A股一般10%，ST股5%，科创板20%
            max_price: 最大价格限制
            min_price: 最小价格限制
            allow_zero_volume: 是否允许零成交量（停牌时为0）
        """
        self.max_pct_change = max_pct_change
        self.max_price = max_price
        self.min_price = min_price
        self.allow_zero_volume = allow_zero_volume

    def validate(self, df: pd.DataFrame, code: str = "") -> ValidationResult:
        """
        校验 DataFrame 数据

        Args:
            df: 日线数据 DataFrame
            code: 股票代码（用于日志）

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        invalid_rows = set()

        if df.empty:
            return ValidationResult(
                valid=True,
                errors=[],
                warnings=["数据为空"],
                rows_checked=0,
                rows_invalid=0,
            )

        rows_checked = len(df)

        # 1. 必要列检查
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            errors.append(f"缺少必要列: {missing_cols}")
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                rows_checked=rows_checked,
                rows_invalid=rows_checked,
            )

        # 2. 价格范围检查
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            # 检查过高价格
            high_mask = df[col] > self.max_price
            if high_mask.any():
                invalid_rows.update(df[high_mask].index.tolist())
                warnings.append(f"{code} {col} 价格过高: {df[high_mask][col].tolist()}")

            # 检查过低价格（排除0，可能是停牌）
            low_mask = (df[col] < self.min_price) & (df[col] > 0)
            if low_mask.any():
                invalid_rows.update(df[low_mask].index.tolist())
                warnings.append(f"{code} {col} 价格过低: {df[low_mask][col].tolist()}")

        # 3. 高低价逻辑检查
        logic_mask = df["high"] < df["low"]
        if logic_mask.any():
            invalid_rows.update(df[logic_mask].index.tolist())
            errors.append(f"{code} 最高价<最低价: {logic_mask.sum()} 行")

        # 4. 开收价范围检查
        open_out = (df["open"] > df["high"]) | (df["open"] < df["low"])
        close_out = (df["close"] > df["high"]) | (df["close"] < df["low"])
        if open_out.any():
            invalid_rows.update(df[open_out].index.tolist())
            errors.append(f"{code} 开盘价超出高低范围: {open_out.sum()} 行")
        if close_out.any():
            invalid_rows.update(df[close_out].index.tolist())
            errors.append(f"{code} 收盘价超出高低范围: {close_out.sum()} 行")

        # 5. 涨跌幅检查
        if "pct_change" in df.columns:
            pct_mask = df["pct_change"].abs() > self.max_pct_change
            if pct_mask.any():
                invalid_rows.update(df[pct_mask].index.tolist())
                warnings.append(
                    f"{code} 涨跌幅超过 {self.max_pct_change}%: "
                    f"{df[pct_mask]['pct_change'].tolist()}"
                )
        else:
            # 自己计算涨跌幅
            if len(df) > 1:
                pct = df["close"].pct_change() * 100
                pct_mask = pct.abs() > self.max_pct_change
                if pct_mask.any():
                    invalid_rows.update(df[pct_mask].index.tolist())
                    warnings.append(
                        f"{code} 计算涨跌幅超过 {self.max_pct_change}%: "
                        f"{pct[pct_mask].tolist()}"
                    )

        # 6. 成交量检查
        if "volume" in df.columns and not self.allow_zero_volume:
            zero_vol = df["volume"] == 0
            if zero_vol.any():
                warnings.append(f"{code} 零成交量: {zero_vol.sum()} 行")

        # 7. 负数检查
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                neg_mask = df[col] < 0
                if neg_mask.any():
                    invalid_rows.update(df[neg_mask].index.tolist())
                    errors.append(f"{code} {col} 存在负数: {neg_mask.sum()} 行")

        # 8. 缺失值检查
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            for col, count in null_counts.items():
                if count > 0:
                    warnings.append(f"{code} {col} 有 {count} 个缺失值")

        is_valid = len(errors) == 0
        rows_invalid = len(invalid_rows)

        result = ValidationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            rows_checked=rows_checked,
            rows_invalid=rows_invalid,
        )

        if not is_valid:
            logger.warning(f"数据校验失败 {code}: {errors}")
        elif warnings:
            logger.info(f"数据校验警告 {code}: {warnings}")

        return result

    def filter_valid(
        self, df: pd.DataFrame, code: str = ""
    ) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        过滤掉异常数据，返回有效数据

        Args:
            df: 原始数据
            code: 股票代码

        Returns:
            (过滤后的数据, 校验结果)
        """
        if df.empty:
            return df, ValidationResult(True, [], [], 0, 0)

        valid_mask = pd.Series(True, index=df.index)

        # 价格范围
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                valid_mask &= (df[col] >= self.min_price) & (df[col] <= self.max_price)

        # 高低价逻辑
        valid_mask &= df["high"] >= df["low"]

        # 开收价在高低范围内
        valid_mask &= (df["open"] >= df["low"]) & (df["open"] <= df["high"])
        valid_mask &= (df["close"] >= df["low"]) & (df["close"] <= df["high"])

        # 非负数
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                valid_mask &= df[col] >= 0

        filtered_df = df[valid_mask].copy()
        removed_count = len(df) - len(filtered_df)

        result = ValidationResult(
            valid=removed_count == 0,
            errors=[f"过滤掉 {removed_count} 行异常数据"] if removed_count > 0 else [],
            warnings=[],
            rows_checked=len(df),
            rows_invalid=removed_count,
        )

        if removed_count > 0:
            logger.warning(f"{code} 过滤掉 {removed_count} 行异常数据")

        return filtered_df, result


# 便捷函数
def validate_daily(df: pd.DataFrame, code: str = "") -> ValidationResult:
    """校验日线数据"""
    validator = DataValidator(max_pct_change=22.0)  # 科创板/北交所 20% + 余量
    return validator.validate(df, code)


def filter_daily(df: pd.DataFrame, code: str = "") -> Tuple[pd.DataFrame, ValidationResult]:
    """过滤日线异常数据"""
    validator = DataValidator(max_pct_change=22.0)
    return validator.filter_valid(df, code)

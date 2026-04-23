"""季频 — 财务三表 + 衍生指标 + 股东户数 + 盈利预测。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from floatshare.domain.records._meta import FieldSpec


@dataclass(frozen=True, slots=True)
class Income:
    """完整利润表 (Tushare income)。"""

    TABLE: ClassVar[str] = "income"
    PK: ClassVar[tuple[str, ...]] = ("code", "end_date", "report_type")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "revenue": FieldSpec("营业收入", primary=True, unit="元"),
        "n_income": FieldSpec("净利润", primary=True, unit="元"),
        "total_revenue": FieldSpec("营业总收入", unit="元"),
        "operate_profit": FieldSpec("营业利润", unit="元"),
        "total_profit": FieldSpec("利润总额", unit="元"),
        "basic_eps": FieldSpec("基本 EPS", unit="元"),
    }

    code: str
    end_date: str
    report_type: str
    ann_date: str | None = None
    f_ann_date: str | None = None
    end_type: str | None = None
    total_revenue: float | None = None
    revenue: float | None = None
    total_cogs: float | None = None
    oper_cost: float | None = None
    sell_exp: float | None = None
    admin_exp: float | None = None
    fin_exp: float | None = None
    operate_profit: float | None = None
    total_profit: float | None = None
    income_tax: float | None = None
    n_income: float | None = None
    n_income_attr_p: float | None = None
    basic_eps: float | None = None
    diluted_eps: float | None = None
    ebit: float | None = None
    ebitda: float | None = None


@dataclass(frozen=True, slots=True)
class Balancesheet:
    """资产负债表 (Tushare balancesheet)。"""

    TABLE: ClassVar[str] = "balancesheet"
    PK: ClassVar[tuple[str, ...]] = ("code", "end_date", "report_type")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "total_assets": FieldSpec("资产总计", primary=True, unit="元"),
        "total_liab": FieldSpec("负债合计", primary=True, unit="元"),
        "monetary_cap": FieldSpec("货币资金", unit="元"),
        "fix_assets": FieldSpec("固定资产", unit="元"),
        "inventories": FieldSpec("存货", unit="元"),
    }

    code: str
    end_date: str
    report_type: str
    ann_date: str | None = None
    f_ann_date: str | None = None
    total_assets: float | None = None
    total_cur_assets: float | None = None
    total_nca: float | None = None
    monetary_cap: float | None = None
    accounts_receiv: float | None = None
    inventories: float | None = None
    fix_assets: float | None = None
    intan_assets: float | None = None
    total_liab: float | None = None
    total_cur_liab: float | None = None
    total_ncl: float | None = None
    accounts_pay: float | None = None
    st_borr: float | None = None
    lt_borr: float | None = None
    total_hldr_eqy_inc_min_int: float | None = None
    total_hldr_eqy_exc_min_int: float | None = None


@dataclass(frozen=True, slots=True)
class Cashflow:
    """现金流量表 (Tushare cashflow)。"""

    TABLE: ClassVar[str] = "cashflow"
    PK: ClassVar[tuple[str, ...]] = ("code", "end_date", "report_type")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "n_cashflow_act": FieldSpec("经营现金流净额", primary=True, unit="元"),
        "n_cashflow_inv_act": FieldSpec("投资现金流净额", primary=True, unit="元"),
        "n_cash_flows_fnc_act": FieldSpec("筹资现金流净额", unit="元"),
        "free_cashflow": FieldSpec("自由现金流", unit="元"),
    }

    code: str
    end_date: str
    report_type: str
    ann_date: str | None = None
    f_ann_date: str | None = None
    n_cashflow_act: float | None = None
    c_inf_fr_operate_a: float | None = None
    c_paid_to_for_empl: float | None = None
    c_paid_for_taxes: float | None = None
    n_cashflow_inv_act: float | None = None
    c_pay_acq_const_fiolta: float | None = None
    n_cash_flows_fnc_act: float | None = None
    c_paid_dvd_pft: float | None = None
    free_cashflow: float | None = None


@dataclass(frozen=True, slots=True)
class FinaIndicator:
    """财务衍生指标 (Tushare fina_indicator，精选 ~25 字段)。"""

    TABLE: ClassVar[str] = "fina_indicator"
    PK: ClassVar[tuple[str, ...]] = ("code", "end_date")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "roe": FieldSpec("ROE", primary=True, unit="%"),
        "netprofit_margin": FieldSpec("净利率", primary=True, unit="%"),
        "grossprofit_margin": FieldSpec("毛利率", primary=True, unit="%"),
        "debt_to_assets": FieldSpec("资产负债率", primary=True, unit="%"),
        "netprofit_yoy": FieldSpec("净利同比", unit="%"),
        "or_yoy": FieldSpec("营收同比", unit="%"),
        "roa": FieldSpec("ROA", unit="%"),
        "roic": FieldSpec("ROIC", unit="%"),
        "current_ratio": FieldSpec("流动比率"),
        "quick_ratio": FieldSpec("速动比率"),
    }

    code: str
    end_date: str
    ann_date: str | None = None
    eps: float | None = None
    dt_eps: float | None = None
    bps: float | None = None
    revenue_ps: float | None = None
    ocfps: float | None = None
    netprofit_margin: float | None = None
    grossprofit_margin: float | None = None
    profit_to_gr: float | None = None
    roe: float | None = None
    roe_waa: float | None = None
    roe_dt: float | None = None
    roa: float | None = None
    roic: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    debt_to_assets: float | None = None
    assets_turn: float | None = None
    inv_turn: float | None = None
    ar_turn: float | None = None
    basic_eps_yoy: float | None = None
    netprofit_yoy: float | None = None
    or_yoy: float | None = None
    roe_yoy: float | None = None
    tr_yoy: float | None = None
    q_netprofit_yoy: float | None = None


@dataclass(frozen=True, slots=True)
class StkHolderNumber:
    """股东户数 (Tushare stk_holdernumber，季度披露)。"""

    TABLE: ClassVar[str] = "stk_holder_number"
    PK: ClassVar[tuple[str, ...]] = ("code", "end_date")
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "holder_num": FieldSpec("股东户数", primary=True, unit="户"),
    }

    code: str
    end_date: str
    ann_date: str | None = None
    holder_num: float | None = None


@dataclass(frozen=True, slots=True)
class EarningsForecast:
    """券商盈利预测 (Tushare report_rc)。"""

    TABLE: ClassVar[str] = "earnings_forecast"
    PK: ClassVar[tuple[str, ...]] = ("code", "report_date", "org_name", "author_name")

    code: str
    report_date: str
    org_name: str
    author_name: str
    report_title: str | None = None
    report_type: str | None = None
    classify: str | None = None
    quarter: str | None = None
    op_rt: float | None = None
    op_pr: float | None = None
    tp: float | None = None
    np: float | None = None
    eps: float | None = None
    pe: float | None = None
    rd: float | None = None
    roe: float | None = None
    ev_ebitda: float | None = None
    rating: str | None = None
    max_price: float | None = None
    min_price: float | None = None

"""月频 — CPI、PPI、券商月度金股。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from floatshare.domain.records._meta import FieldSpec


@dataclass(frozen=True, slots=True)
class CnCpi:
    """中国居民消费价格指数 (Tushare cn_cpi，月频)。"""

    TABLE: ClassVar[str] = "cn_cpi"
    PK: ClassVar[tuple[str, ...]] = ("month",)
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "nt_yoy": FieldSpec("全国同比", primary=True, unit="%"),
        "nt_mom": FieldSpec("全国环比", primary=True, unit="%"),
        "nt_val": FieldSpec("全国当月值"),
        "town_yoy": FieldSpec("城市同比", unit="%"),
        "cnt_yoy": FieldSpec("农村同比", unit="%"),
    }

    month: str
    nt_val: float | None = None
    nt_yoy: float | None = None
    nt_mom: float | None = None
    nt_accu: float | None = None
    town_val: float | None = None
    town_yoy: float | None = None
    town_mom: float | None = None
    town_accu: float | None = None
    cnt_val: float | None = None
    cnt_yoy: float | None = None
    cnt_mom: float | None = None
    cnt_accu: float | None = None


@dataclass(frozen=True, slots=True)
class CnPpi:
    """中国工业生产者出厂价格指数 (Tushare cn_ppi，月频)。"""

    TABLE: ClassVar[str] = "cn_ppi"
    PK: ClassVar[tuple[str, ...]] = ("month",)
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "ppi_yoy": FieldSpec("PPI 同比", primary=True, unit="%"),
        "ppi_mom": FieldSpec("PPI 环比", primary=True, unit="%"),
        "ppi_mp_yoy": FieldSpec("生产资料同比", unit="%"),
        "ppi_cg_yoy": FieldSpec("生活资料同比", unit="%"),
        "ppi_accu": FieldSpec("累计同比", unit="%"),
    }

    month: str
    ppi_yoy: float | None = None
    ppi_mp_yoy: float | None = None
    ppi_mp_qm_yoy: float | None = None
    ppi_mp_rm_yoy: float | None = None
    ppi_mp_p_yoy: float | None = None
    ppi_cg_yoy: float | None = None
    ppi_cg_f_yoy: float | None = None
    ppi_cg_c_yoy: float | None = None
    ppi_cg_adu_yoy: float | None = None
    ppi_cg_dcg_yoy: float | None = None
    ppi_mom: float | None = None
    ppi_accu: float | None = None


@dataclass(frozen=True, slots=True)
class BrokerPicks:
    """券商月度金股 (Tushare broker_recommend)。"""

    TABLE: ClassVar[str] = "broker_picks"
    PK: ClassVar[tuple[str, ...]] = ("month", "broker", "code")

    month: str
    broker: str
    code: str
    name: str | None = None

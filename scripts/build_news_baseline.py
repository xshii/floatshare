"""生成 data/news/industry_baseline.json — 过去 N 天联播行业 mention 基线.

用途:
    - extract_industry_mentions 读此文件算 IDF 权重, 冷门行业被提及时加大权重
    - 版本化 + 每月重算, 跟踪信号基线漂移

用法:
    python scripts/build_news_baseline.py                  # 默认 90 天至今
    python scripts/build_news_baseline.py --days 90
    python scripts/build_news_baseline.py --end 2026-04-22 --days 90
    python scripts/build_news_baseline.py --output data/news/industry_baseline.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from floatshare.infrastructure.data_sources.tushare import TushareSource
from floatshare.infrastructure.nlp import load_industry_keywords


def _compute_baseline(end: date, days: int) -> dict[str, object]:
    """扫 [end-days+1 .. end] 区间的 cctv_news, 统计每个 L1 的 mentions 计数.

    idf = log(N_days / (days_mentioned + 1)) + 1   — 平滑 idf, 最小值 1.
    mentioned 判定与 extract_industry_mentions 一致 (读当前 keywords.json 阈值).
    """
    kd = load_industry_keywords()
    ts = TushareSource()

    days_with_data = 0
    days_mentioned: Counter[str] = Counter()  # l1_code → 出现天数
    mentions_total: Counter[str] = Counter()  # l1_code → 命中词总数

    start = end - timedelta(days=days - 1)
    empty_days: list[str] = []
    cur = start
    while cur <= end:
        df = ts.get_cctv_news_by_date(cur)
        if df.empty:
            empty_days.append(cur.isoformat())
            cur += timedelta(days=1)
            continue
        days_with_data += 1
        full = "\n".join(str(c) for c in df["content"].tolist() if c)

        for l1_code, (_name, kws) in kd.entries.items():
            # 这里用 substring 逐词 count — 与 extract 的 hit 统计口径一致 (kw in text).
            # 不走 mention 阈值, 为的是捕捉真实命中频率, 由调用方自己算 idf.
            hits_this_day = sum(1 for kw in kws if kw in full)
            if hits_this_day > 0:
                days_mentioned[l1_code] += 1
                mentions_total[l1_code] += hits_this_day
        cur += timedelta(days=1)

    industries: dict[str, dict[str, object]] = {}
    for l1_code, (name, _kws) in kd.entries.items():
        dm = days_mentioned[l1_code]
        mt = mentions_total[l1_code]
        idf = math.log(days_with_data / (dm + 1)) + 1.0  # 平滑: +1 避免 0 分母, +1 提底
        industries[l1_code] = {
            "name": name,
            "days_mentioned": dm,
            "mentions_total": mt,
            "mentions_per_day": round(mt / days_with_data, 3) if days_with_data else 0.0,
            "idf": round(idf, 4),
        }

    return {
        "_meta": {
            "source_start": start.isoformat(),
            "source_end": end.isoformat(),
            "source_days_total": days,
            "source_days_with_data": days_with_data,
            "source_empty_days": empty_days,
            "generated_at": date.today().isoformat(),
            "keyword_dict_version": "loaded_from_industry_keywords.json",
        },
        "industries": industries,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=90, help="窗口长度 (默认 90 天)")
    p.add_argument("--end", type=str, default=date.today().isoformat(), help="窗口末端 YYYY-MM-DD")
    p.add_argument(
        "--output",
        type=str,
        default="data/news/industry_baseline.json",
        help="输出路径",
    )
    args = p.parse_args()

    end = date.fromisoformat(args.end)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    baseline = _compute_baseline(end, args.days)
    out.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = baseline["_meta"]
    ind = baseline["industries"]
    assert isinstance(ind, dict)
    print(f"✓ baseline written → {out}")
    print(f"  window: {meta['source_start']} .. {meta['source_end']}")  # type: ignore[index]
    print(
        f"  days_with_data: {meta['source_days_with_data']} / {meta['source_days_total']}"  # type: ignore[index]
    )
    print(f"  L1 covered: {sum(1 for v in ind.values() if v['days_mentioned'] > 0)} / {len(ind)}")
    top_idf = sorted(ind.items(), key=lambda kv: -kv[1]["idf"])[:5]
    print("  top-5 idf (最冷门):")
    for l1, v in top_idf:
        print(f"    {l1} {v['name']:6s} idf={v['idf']:.3f}  days={v['days_mentioned']}")


if __name__ == "__main__":
    main()

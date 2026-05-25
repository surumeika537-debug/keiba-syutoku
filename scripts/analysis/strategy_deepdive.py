"""戦略別に各次元 (grade / racecourse / 出走頭数帯 / 1番人気オッズ帯 / 三連単払戻帯) で
ROIや的中数を集計する深掘り分析。

出力 (CSV: data/processed/strategy_deepdive.csv) は long-format:
  strategy, dimension, group_key, races, hits, investment_yen, payout_yen, profit_yen, roi, hit_rate

注: dimension='payout_band' のみ「的中レースだけ」を payout_yen の帯で集計し、
   races / investment_yen / hit_rate は当該帯の戦略パフォーマンスではなく
   ヒット分布として扱う (詳細は CSV を参照)。

Usage:
    python scripts/analysis/strategy_deepdive.py --jra-only --exclude-steeplechase --grades G1 G2 G3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.config import PROCESSED_DIR
from src.utils import force_utf8_stdout, setup_logger
from scripts.backtest.simple_roi import (
    build_strategies,
    compute_detail,
    load_data,
)

force_utf8_stdout()
log = setup_logger("strategy_deepdive")


# ---- bucketers ------------------------------------------------------------

FIELD_SIZE_BANDS = [
    ("~10", lambda n: n <= 10),
    ("11-13", lambda n: 11 <= n <= 13),
    ("14-16", lambda n: 14 <= n <= 16),
    ("17-18", lambda n: 17 <= n <= 18),
]
FAV_ODDS_BANDS = [
    ("1.0-1.9", lambda o: 1.0 <= o < 2.0),
    ("2.0-2.9", lambda o: 2.0 <= o < 3.0),
    ("3.0-4.9", lambda o: 3.0 <= o < 5.0),
    ("5.0+",    lambda o: o >= 5.0),
]
PAYOUT_BANDS = [
    ("0-10000",      lambda p: p <= 10_000),
    ("10001-50000",  lambda p: 10_000 < p <= 50_000),
    ("50001-100000", lambda p: 50_000 < p <= 100_000),
    ("100001+",      lambda p: p > 100_000),
]


def assign_band(value, bands, na_label="(NaN)"):
    if pd.isna(value):
        return na_label
    for label, pred in bands:
        try:
            if pred(value):
                return label
        except TypeError:
            return na_label
    return "(other)"


# ---- aggregation -----------------------------------------------------------

LOW_SAMPLE_THRESHOLD = 30


def _sample_warning(races) -> str:
    if races is None or pd.isna(races):
        return ""
    return "LOW_SAMPLE" if int(races) < LOW_SAMPLE_THRESHOLD else ""


def aggregate(detail: pd.DataFrame, dimension: str, group_col: str) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for (strategy, group_key), sub in detail.groupby(["strategy", group_col], dropna=False):
        cost = int(sub["investment_yen"].sum())
        payout = int(sub["payout_yen"].sum())
        hits = int(sub["hit_ticket"].notna().sum())
        races = int(sub["race_id"].nunique())
        rows.append({
            "strategy": strategy,
            "dimension": dimension,
            "group_key": "(NaN)" if pd.isna(group_key) else group_key,
            "races": races,
            "hits": hits,
            "investment_yen": cost,
            "payout_yen": payout,
            "profit_yen": payout - cost,
            "roi": (payout - cost) / cost if cost else 0.0,
            "hit_rate": hits / races if races else 0.0,
            "sample_warning": _sample_warning(races),
        })
    return pd.DataFrame(rows)


def aggregate_payout_band(detail: pd.DataFrame) -> pd.DataFrame:
    """Distribution of HITS across payout bands (per strategy).

    races / investment_yen / hit_rate fields are filled with NaN since they don't
    apply to this slicing (the band is defined by the payout amount, which only
    exists when the bet hit)."""
    if detail.empty:
        return pd.DataFrame()
    hits_only = detail[detail["hit_ticket"].notna()].copy()
    hits_only["payout_band"] = hits_only["payout_yen"].apply(
        lambda p: assign_band(p, PAYOUT_BANDS))
    rows = []
    for (strategy, band), sub in hits_only.groupby(["strategy", "payout_band"], dropna=False):
        rows.append({
            "strategy": strategy,
            "dimension": "payout_band",
            "group_key": band,
            "races": pd.NA,
            "hits": int(len(sub)),
            "investment_yen": pd.NA,
            "payout_yen": int(sub["payout_yen"].sum()),
            "profit_yen": pd.NA,
            "roi": pd.NA,
            "hit_rate": pd.NA,
            # payout_band rows describe the hit distribution; no ROI-quality flag applies.
            "sample_warning": "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grades", nargs="+", default=None)
    ap.add_argument("--jra-only", action="store_true")
    ap.add_argument("--exclude-steeplechase", action="store_true")
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--strategy-d-min-popularity", type=int, default=5)
    args = ap.parse_args()

    races, entries, payouts = load_data(
        grades=tuple(args.grades) if args.grades else None,
        date_from=args.date_from,
        date_to=args.date_to,
        jra_only=args.jra_only,
        exclude_steeplechase=args.exclude_steeplechase,
    )
    log.info("after filters: races=%d entries=%d payouts=%d", len(races), len(entries), len(payouts))

    strategies = build_strategies(args.strategy_d_min_popularity)
    detail = compute_detail(races, entries, payouts, strategies)
    if detail.empty:
        print("(no data — relax filters)")
        return

    # enrich detail with field_size and 1番人気オッズ
    field_size = entries.groupby("race_id").size().rename("field_size")
    fav_odds = (entries[entries["popularity"] == 1]
                .drop_duplicates("race_id")
                .set_index("race_id")["win_odds"]
                .rename("fav_odds"))
    detail = detail.merge(field_size, left_on="race_id", right_index=True, how="left")
    detail = detail.merge(fav_odds, left_on="race_id", right_index=True, how="left")
    detail["field_size_band"] = detail["field_size"].apply(lambda n: assign_band(n, FIELD_SIZE_BANDS))
    detail["fav_odds_band"] = detail["fav_odds"].apply(lambda o: assign_band(o, FAV_ODDS_BANDS))

    parts = [
        aggregate(detail, "grade", "grade"),
        aggregate(detail, "racecourse", "racecourse"),
        aggregate(detail, "field_size_band", "field_size_band"),
        aggregate(detail, "fav_odds_band", "fav_odds_band"),
        aggregate_payout_band(detail),
    ]
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["strategy", "dimension", "group_key"], kind="stable").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "strategy_deepdive.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("wrote %s (%d rows)", path, len(out))

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

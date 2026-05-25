"""D9 (strategy D + plus候補のみ + D6 negative filter) を component に分解する。

Components:
  niigata_4g3   アイビスサマーD / 関屋記念 / 新潟記念 / レパードS (新潟G3)
  march_g2      race_date が3月 AND grade=G2
  hanshin_g1    racecourse=阪神 AND grade=G1
  september_g3  race_date が9月 AND grade=G3

重複: 新潟記念 (9月開催) は niigata_4g3 と september_g3 の両方にマッチする。
各 component の集計は overlap 重複計上 OK。D9_total_unique は unique race 集計。

出力:
  data/processed/d9_component_summary.csv
  data/processed/d9_ablation_summary.csv
  data/processed/d9_hits.csv
  data/processed/d9_component_by_year.csv

Usage:
    python scripts/analysis/d9_deepdive.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.utils import force_utf8_stdout, setup_logger
from scripts.backtest.strategy_d_variants import (
    BOOTSTRAP_SEED,
    NIIGATA_4GIII_PATTERNS,
    apply_variant,
    collapse_to_race,
    collect_subsets,
    compute_stats,
    load_data,
)

force_utf8_stdout()
log = setup_logger("d9_deepdive")


# ---- component tagging ----------------------------------------------------

def tag_components(per_race: pd.DataFrame) -> pd.DataFrame:
    per_race = per_race.copy()
    per_race["month"] = per_race["race_date"].apply(
        lambda d: int(pd.Timestamp(d).month) if pd.notna(d) else None
    )
    name = per_race["race_name"].astype(str)
    per_race["comp_niigata_4g3"] = (
        name.apply(lambda n: any(p in n for p in NIIGATA_4GIII_PATTERNS))
        & (per_race["racecourse"] == "新潟")
        & (per_race["grade"] == "G3")
    )
    per_race["comp_march_g2"] = (per_race["month"] == 3) & (per_race["grade"] == "G2")
    per_race["comp_hanshin_g1"] = (per_race["racecourse"] == "阪神") & (per_race["grade"] == "G1")
    per_race["comp_september_g3"] = (per_race["month"] == 9) & (per_race["grade"] == "G3")
    return per_race


def make_component_labels(per_race: pd.DataFrame) -> pd.Series:
    def labels(row):
        out = []
        if row["comp_niigata_4g3"]:    out.append("niigata_4g3")
        if row["comp_march_g2"]:       out.append("march_g2")
        if row["comp_hanshin_g1"]:     out.append("hanshin_g1")
        if row["comp_september_g3"]:   out.append("september_g3")
        return "|".join(out)
    return per_race.apply(labels, axis=1)


def stats_with_avg_hit_payout(sub: pd.DataFrame, rng) -> dict:
    s = compute_stats(sub, rng)
    s["avg_hit_payout"] = s["payout_yen"] / s["hits"] if s["hits"] else 0
    return s


# ---- main -----------------------------------------------------------------

COMPONENTS = [
    ("niigata_4g3",   "comp_niigata_4g3"),
    ("march_g2",      "comp_march_g2"),
    ("hanshin_g1",    "comp_hanshin_g1"),
    ("september_g3",  "comp_september_g3"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    log.info("loading data (JRA flat G1/G2/G3, hardcoded for D9 deepdive)")
    races, entries, payouts = load_data(
        grades=("G1", "G2", "G3"), jra_only=True, exclude_steeplechase=True
    )
    log.info("filtered races=%d", len(races))

    base = collect_subsets(races, entries, payouts)
    d9_subsets = apply_variant(base, "D9")
    per_race = collapse_to_race(d9_subsets)
    per_race = tag_components(per_race)
    per_race["component_labels"] = make_component_labels(per_race)
    log.info("D9 per_race=%d (hits=%d)", len(per_race), int(per_race["hit"].sum()))

    # ---- 作業2: component summary
    comp_rows = []
    for label, col in COMPONENTS:
        sub = per_race[per_race[col]]
        comp_rows.append({"component": label, **stats_with_avg_hit_payout(sub, rng)})
    comp_rows.append({"component": "D9_total_unique",
                      **stats_with_avg_hit_payout(per_race, rng)})
    comp_df = pd.DataFrame(comp_rows)

    # ---- 作業3: ablation
    ablations = [
        ("D9_all",                    per_race),
        ("D9_without_niigata_4g3",    per_race[~per_race["comp_niigata_4g3"]]),
        ("D9_without_march_g2",       per_race[~per_race["comp_march_g2"]]),
        ("D9_without_hanshin_g1",     per_race[~per_race["comp_hanshin_g1"]]),
        ("D9_without_september_g3",   per_race[~per_race["comp_september_g3"]]),
        ("D9_only_niigata_4g3",       per_race[per_race["comp_niigata_4g3"]]),
        ("D9_only_non_niigata",       per_race[~per_race["comp_niigata_4g3"]]),
    ]
    abl_rows = []
    for label, sub in ablations:
        abl_rows.append({"variant": label, **stats_with_avg_hit_payout(sub, rng)})
    abl_df = pd.DataFrame(abl_rows)

    # ---- 作業4: hits CSV
    distance_map = base.drop_duplicates("race_id").set_index("race_id")["distance"]
    winning_map = (d9_subsets[d9_subsets["hit"]]
                   .drop_duplicates("race_id")
                   .set_index("race_id")["winning_ticket"])
    hits = per_race[per_race["hit"] > 0].copy()
    hits["distance"] = hits["race_id"].map(distance_map)
    hits["ticket"] = hits["race_id"].map(winning_map)
    hits["profit_yen"] = hits["payout"] - hits["cost"]
    hits_df = (hits.rename(columns={"payout": "payout_yen", "cost": "investment_yen"})
               [["race_id", "race_date", "race_name", "grade", "racecourse",
                 "distance", "field_size", "component_labels", "ticket",
                 "payout_yen", "investment_yen", "profit_yen"]]
               .sort_values("race_date", kind="stable"))

    # ---- 作業5: component × year
    year_rows = []
    for label, col in COMPONENTS:
        sub = per_race[per_race[col]]
        for year, ysub in sub.groupby("year", dropna=False):
            cost = int(ysub["cost"].sum())
            payout = int(ysub["payout"].sum())
            hits_n = int(ysub["hit"].sum())
            year_rows.append({
                "component": label,
                "year": int(year) if pd.notna(year) else None,
                "races": len(ysub),
                "investment_yen": cost,
                "hits": hits_n,
                "payout_yen": payout,
                "roi": (payout - cost) / cost if cost else 0.0,
            })
    year_df = pd.DataFrame(year_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(PROCESSED_DIR / "d9_component_summary.csv",
                   index=False, encoding="utf-8-sig")
    abl_df.to_csv(PROCESSED_DIR / "d9_ablation_summary.csv",
                  index=False, encoding="utf-8-sig")
    hits_df.to_csv(PROCESSED_DIR / "d9_hits.csv",
                   index=False, encoding="utf-8-sig")
    year_df.to_csv(PROCESSED_DIR / "d9_component_by_year.csv",
                   index=False, encoding="utf-8-sig")
    log.info("wrote 4 CSVs")

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.4f}".format)

    # reorder columns for readability
    metric_cols = ["races", "tickets", "investment_yen", "hits", "payout_yen", "profit_yen",
                   "roi", "hit_rate", "avg_hit_payout", "max_payout_yen",
                   "max_losing_streak", "avg_tickets_per_race",
                   "bootstrap_ci_low", "bootstrap_ci_high", "sample_warning"]

    print("\n=== COMPONENT SUMMARY (重複計上、D9_total_unique は unique) ===")
    print(comp_df[["component"] + metric_cols].to_string(index=False))

    print("\n=== ABLATION SUMMARY ===")
    print(abl_df[["variant"] + metric_cols].to_string(index=False))

    print("\n=== D9 HITS (30 races) ===")
    print(hits_df.to_string(index=False))

    print("\n=== COMPONENT × YEAR (ROI matrix) ===")
    if not year_df.empty:
        pivot = year_df.pivot(index="year", columns="component", values="roi")
        ordered = [c for c, _ in COMPONENTS if c in pivot.columns]
        pivot = pivot[ordered]
        print(pivot.to_string())

    print("\n=== COMPONENT × YEAR (race count matrix) ===")
    if not year_df.empty:
        pivot2 = year_df.pivot(index="year", columns="component", values="races").fillna(0).astype(int)
        ordered = [c for c, _ in COMPONENTS if c in pivot2.columns]
        pivot2 = pivot2[ordered]
        print(pivot2.to_string())


if __name__ == "__main__":
    main()

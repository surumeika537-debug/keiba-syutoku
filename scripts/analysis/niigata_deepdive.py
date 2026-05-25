"""新潟 × strategy D の解剖。

「JRA平地G1/G2/G3 × 競馬場=新潟」だけを取り出し、strategy D
(1番人気・2番人気 + 5番人気以下 odds10-30 の馬1頭 の三連単BOX) を
race-level と (race, dark_horse) subset-level の両方で再集計する。

特徴量:
  distance / surface / field_size / month / fav_odds_band (race-level)
  frame_band of dark horse                                 (subset-level)

統計:
  各セルに races / tickets / hits / total_investment / total_payout
            roi / hit_rate / avg|median|max hit_payout / sample_warning

Bootstrap:
  全体ROIの 95% CI を race 単位 1000回 resample で算出。

1000m仮説:
  distance=1000 のレースを別集計し、全ROIへの寄与率を出す。

出力:
  data/processed/niigata_deepdive.csv
  data/processed/niigata_hits.csv

Usage:
    python scripts/analysis/niigata_deepdive.py
"""
from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.config import JRA_RACECOURSES, PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("niigata_deepdive")

# strategy D parameters (must match simple_roi.py defaults)
DARK_MIN_POP = 5
DARK_ODDS_MIN = 10.0
DARK_ODDS_MAX = 30.0
TICKET_COST = 100
N_TICKETS_PER_SUBSET = 6  # 3-horse box → 3! = 6 permutations

RACECOURSE = "新潟"
LOW_SAMPLE_THRESHOLD = 30
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


# ---- bucketers ------------------------------------------------------------

def distance_band(d) -> str:
    if pd.isna(d): return "(NaN)"
    d = int(d)
    if d == 1000: return "1000"
    if d == 1200: return "1200"
    if d == 1400: return "1400"
    if d == 1600: return "1600"
    if d == 1800: return "1800"
    if d >= 2000: return "2000+"
    return "その他"

def field_size_band(n) -> str:
    if pd.isna(n): return "(NaN)"
    n = int(n)
    if n <= 10: return "~10"
    if n <= 13: return "11-13"
    if n <= 16: return "14-16"
    return "17-18"

def fav_odds_band(o) -> str:
    if pd.isna(o): return "(NaN)"
    if o < 2.0: return "1.0-1.9"
    if o < 3.0: return "2.0-2.9"
    if o < 5.0: return "3.0-4.9"
    return "5.0+"

def frame_band(f) -> str:
    if pd.isna(f): return "(NaN)"
    f = int(f)
    if f <= 2: return "1-2枠"
    if f <= 4: return "3-4枠"
    if f <= 6: return "5-6枠"
    return "7-8枠"


# ---- data load + subset enumeration ---------------------------------------

def load_niigata_data():
    engine = get_engine()
    races = pd.read_sql("SELECT * FROM races", engine, parse_dates=["race_date"])
    entries = pd.read_sql("SELECT * FROM entries", engine)
    payouts = pd.read_sql("SELECT * FROM payouts", engine)

    # JRA flat G1-G3 × Niigata
    races = races[
        races["grade"].isin({"G1", "G2", "G3"})
        & races["racecourse"].isin(JRA_RACECOURSES)
        & (races["surface"] != "障害")
        & (races["racecourse"] == RACECOURSE)
    ].copy()
    keep = set(races["race_id"])
    entries = entries[entries["race_id"].isin(keep)].copy()
    payouts = payouts[payouts["race_id"].isin(keep)].copy()

    entries["popularity"] = pd.to_numeric(entries["popularity"], errors="coerce").astype("Int64")
    entries["horse_number"] = pd.to_numeric(entries["horse_number"], errors="coerce").astype("Int64")
    entries["frame_number"] = pd.to_numeric(entries["frame_number"], errors="coerce").astype("Int64")
    entries["win_odds"] = pd.to_numeric(entries["win_odds"], errors="coerce")
    entries = entries.dropna(subset=["popularity", "horse_number"])
    return races, entries, payouts


def collect_subsets(races: pd.DataFrame, entries: pd.DataFrame, payouts: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, dark_horse_candidate). This is the atomic unit of strategy D bets."""
    meta = races.set_index("race_id")[
        ["race_date", "race_name", "distance", "surface", "field_size", "grade"]
    ]
    tri = (payouts[payouts["bet_type"] == "三連単"]
           .drop_duplicates("race_id").set_index("race_id"))

    rows = []
    for rid, grp in entries.groupby("race_id", sort=False):
        if rid not in meta.index:
            continue
        p1_row = grp[grp["popularity"] == 1].head(1)
        p2_row = grp[grp["popularity"] == 2].head(1)
        if p1_row.empty or p2_row.empty:
            continue
        p1 = int(p1_row["horse_number"].iloc[0])
        p2 = int(p2_row["horse_number"].iloc[0])
        p1_odds_raw = p1_row["win_odds"].iloc[0]
        p1_odds = float(p1_odds_raw) if pd.notna(p1_odds_raw) else None

        quals = grp[
            (grp["popularity"] >= DARK_MIN_POP)
            & (grp["win_odds"].between(DARK_ODDS_MIN, DARK_ODDS_MAX))
        ]
        if quals.empty:
            continue

        true_combo = str(tri.loc[rid, "combination"]) if rid in tri.index else None
        true_payout = int(tri.loc[rid, "payout_yen"] or 0) if rid in tri.index else 0

        m = meta.loc[rid]
        race_date = m["race_date"]
        month = pd.Timestamp(race_date).month if pd.notna(race_date) else None

        for _, q in quals.iterrows():
            qhn = int(q["horse_number"])
            if qhn in (p1, p2):
                continue
            tickets = [f"{a}-{b}-{c}" for a, b, c in permutations([p1, p2, qhn], 3)]
            hit = bool(true_combo and (true_combo in tickets))
            rows.append({
                "race_id": rid,
                "race_date": race_date,
                "race_name": m["race_name"],
                "grade": m["grade"],
                "distance": int(m["distance"]) if pd.notna(m["distance"]) else None,
                "surface": m["surface"],
                "field_size": int(m["field_size"]) if pd.notna(m["field_size"]) else None,
                "month": month,
                "fav_odds": p1_odds,
                "p1_horse": p1,
                "p2_horse": p2,
                "dark_horse": qhn,
                "dark_frame": int(q["frame_number"]) if pd.notna(q["frame_number"]) else None,
                "dark_pop": int(q["popularity"]),
                "dark_odds": float(q["win_odds"]),
                "n_tickets": N_TICKETS_PER_SUBSET,
                "investment_yen": N_TICKETS_PER_SUBSET * TICKET_COST,
                "hit": hit,
                "payout_yen": true_payout if hit else 0,
                "winning_ticket": true_combo if hit else None,
                "actual_trifecta": true_combo,
            })
    return pd.DataFrame(rows)


# ---- aggregation ----------------------------------------------------------

OUT_COLS = [
    "dimension", "group_key", "races", "tickets", "hits",
    "total_investment", "total_payout", "roi", "hit_rate",
    "avg_hit_payout", "median_hit_payout", "max_hit_payout",
    "sample_warning",
]


def _agg_row(dimension, group_key, races, tickets, cost, hits, payout, hit_payouts):
    return {
        "dimension": dimension,
        "group_key": "(NaN)" if pd.isna(group_key) else group_key,
        "races": int(races),
        "tickets": int(tickets),
        "hits": int(hits),
        "total_investment": int(cost),
        "total_payout": int(payout),
        "roi": (payout - cost) / cost if cost else 0.0,
        "hit_rate": hits / races if races else 0.0,
        "avg_hit_payout": float(hit_payouts.mean()) if len(hit_payouts) else 0.0,
        "median_hit_payout": float(hit_payouts.median()) if len(hit_payouts) else 0.0,
        "max_hit_payout": int(hit_payouts.max()) if len(hit_payouts) else 0,
        "sample_warning": "LOW_SAMPLE" if races < LOW_SAMPLE_THRESHOLD else "",
    }


def aggregate_race_level(subsets: pd.DataFrame, group_col: str, dimension: str) -> list[dict]:
    """Collapse subsets to race-level first (race-shared dims like distance, surface, ...)."""
    if subsets.empty:
        return []
    per_race = (subsets.groupby("race_id", sort=False)
                .agg(group_val=(group_col, "first"),
                     tickets=("n_tickets", "sum"),
                     cost=("investment_yen", "sum"),
                     # at most one subset can hit, but to be safe use max
                     hit_any=("hit", "max"),
                     payout=("payout_yen", "max"))
                .reset_index())
    rows = []
    for k, sub in per_race.groupby("group_val", dropna=False):
        races_n = len(sub)
        hits = int(sub["hit_any"].sum())
        cost = int(sub["cost"].sum())
        payout = int(sub["payout"].sum())
        hit_payouts = sub.loc[sub["hit_any"] > 0, "payout"]
        rows.append(_agg_row(dimension, k, races_n,
                             int(sub["tickets"].sum()), cost, hits, payout, hit_payouts))
    return rows


def aggregate_subset_level(subsets: pd.DataFrame, group_col: str, dimension: str) -> list[dict]:
    """Each subset (race × dark_horse) is a betting decision; useful for per-dark-horse dims like frame."""
    if subsets.empty:
        return []
    rows = []
    for k, sub in subsets.groupby(group_col, dropna=False):
        races_n = len(sub)  # subset count (here 'races' = decisions, not unique races)
        hits = int(sub["hit"].sum())
        cost = int(sub["investment_yen"].sum())
        payout = int(sub["payout_yen"].sum())
        hit_payouts = sub.loc[sub["hit"], "payout_yen"]
        rows.append(_agg_row(dimension, k, races_n,
                             int(sub["n_tickets"].sum()), cost, hits, payout, hit_payouts))
    return rows


# ---- bootstrap ------------------------------------------------------------

def bootstrap_overall_roi(subsets: pd.DataFrame, n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Race-level resample with replacement. Each draw aggregates that race's full subset block."""
    rng = np.random.default_rng(seed)
    race_blocks = list(subsets.groupby("race_id", sort=False))
    n = len(race_blocks)
    if n == 0:
        return float("nan"), float("nan")
    costs = np.array([blk["investment_yen"].sum() for _, blk in race_blocks], dtype=float)
    payouts = np.array([blk["payout_yen"].sum() for _, blk in race_blocks], dtype=float)
    rois = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        c = costs[idx].sum()
        p = payouts[idx].sum()
        rois[i] = (p - c) / c if c else 0.0
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    args = ap.parse_args()

    races, entries, payouts = load_niigata_data()
    log.info("新潟 JRA 平地 G1-G3: races=%d entries=%d payouts=%d",
             len(races), len(entries), len(payouts))

    subsets = collect_subsets(races, entries, payouts)
    log.info("strategy D 適用subsets: %d (across %d races)",
             len(subsets), subsets["race_id"].nunique() if len(subsets) else 0)

    # add bucket columns
    subsets["distance_band"] = subsets["distance"].apply(distance_band)
    subsets["field_size_band"] = subsets["field_size"].apply(field_size_band)
    subsets["fav_odds_band"] = subsets["fav_odds"].apply(fav_odds_band)
    subsets["frame_band"] = subsets["dark_frame"].apply(frame_band)
    subsets["month_str"] = subsets["month"].apply(lambda m: f"{int(m):02d}" if pd.notna(m) else "(NaN)")

    # ---- overall row + bootstrap CI
    total_races = subsets["race_id"].nunique()
    total_subsets = len(subsets)
    total_tickets = int(subsets["n_tickets"].sum())
    total_cost = int(subsets["investment_yen"].sum())
    # at most one subset per race can hit; aggregate per race
    per_race_hits = subsets.groupby("race_id")["hit"].max()
    per_race_payout = subsets.groupby("race_id")["payout_yen"].max()
    total_hits = int(per_race_hits.sum())
    total_payout = int(per_race_payout.sum())
    overall_roi = (total_payout - total_cost) / total_cost if total_cost else 0.0
    overall_hit_rate = total_hits / total_races if total_races else 0.0

    log.info("bootstrapping %d resamples...", args.bootstrap)
    ci_lo, ci_hi = bootstrap_overall_roi(subsets, n_resamples=args.bootstrap)

    rows = [{
        "dimension": "overall",
        "group_key": "ALL",
        "races": total_races,
        "tickets": total_tickets,
        "hits": total_hits,
        "total_investment": total_cost,
        "total_payout": total_payout,
        "roi": overall_roi,
        "hit_rate": overall_hit_rate,
        "avg_hit_payout": float(per_race_payout[per_race_hits > 0].mean()) if total_hits else 0.0,
        "median_hit_payout": float(per_race_payout[per_race_hits > 0].median()) if total_hits else 0.0,
        "max_hit_payout": int(per_race_payout[per_race_hits > 0].max()) if total_hits else 0,
        "sample_warning": "LOW_SAMPLE" if total_races < LOW_SAMPLE_THRESHOLD else "",
    }]

    # ---- breakdowns
    rows.extend(aggregate_race_level(subsets,   "distance_band",   "distance"))
    rows.extend(aggregate_race_level(subsets,   "surface",         "surface"))
    rows.extend(aggregate_race_level(subsets,   "field_size_band", "field_size"))
    rows.extend(aggregate_race_level(subsets,   "month_str",       "month"))
    rows.extend(aggregate_race_level(subsets,   "fav_odds_band",   "fav_odds"))
    rows.extend(aggregate_subset_level(subsets, "frame_band",      "frame"))

    # ---- 1000m hypothesis (separate cell + contribution)
    sub1000 = subsets[subsets["distance"] == 1000]
    if len(sub1000):
        per_race_1k = sub1000.groupby("race_id").agg(
            tickets=("n_tickets", "sum"),
            cost=("investment_yen", "sum"),
            hit=("hit", "max"),
            payout=("payout_yen", "max"),
        )
        races_1k = len(per_race_1k)
        cost_1k = int(per_race_1k["cost"].sum())
        payout_1k = int(per_race_1k["payout"].sum())
        hits_1k = int(per_race_1k["hit"].sum())
        hit_payouts_1k = per_race_1k.loc[per_race_1k["hit"] > 0, "payout"]
        roi_1k = (payout_1k - cost_1k) / cost_1k if cost_1k else 0.0
        contribution_payout = payout_1k / total_payout if total_payout else 0.0
        contribution_cost = cost_1k / total_cost if total_cost else 0.0
        rows.append({
            "dimension": "1000m_only",
            "group_key": f"contrib_payout={contribution_payout:.1%} cost={contribution_cost:.1%}",
            "races": races_1k,
            "tickets": int(per_race_1k["tickets"].sum()),
            "hits": hits_1k,
            "total_investment": cost_1k,
            "total_payout": payout_1k,
            "roi": roi_1k,
            "hit_rate": hits_1k / races_1k if races_1k else 0.0,
            "avg_hit_payout": float(hit_payouts_1k.mean()) if hits_1k else 0.0,
            "median_hit_payout": float(hit_payouts_1k.median()) if hits_1k else 0.0,
            "max_hit_payout": int(hit_payouts_1k.max()) if hits_1k else 0,
            "sample_warning": "LOW_SAMPLE" if races_1k < LOW_SAMPLE_THRESHOLD else "",
        })

    # ---- "all niigata except 1000m" — robustness check
    sub_non1k = subsets[subsets["distance"] != 1000]
    if len(sub_non1k):
        per_race_n = sub_non1k.groupby("race_id").agg(
            tickets=("n_tickets", "sum"),
            cost=("investment_yen", "sum"),
            hit=("hit", "max"),
            payout=("payout_yen", "max"),
        )
        races_n = len(per_race_n)
        cost_n = int(per_race_n["cost"].sum())
        payout_n = int(per_race_n["payout"].sum())
        hits_n = int(per_race_n["hit"].sum())
        hit_payouts_n = per_race_n.loc[per_race_n["hit"] > 0, "payout"]
        rows.append({
            "dimension": "exclude_1000m",
            "group_key": "non-1000m only",
            "races": races_n,
            "tickets": int(per_race_n["tickets"].sum()),
            "hits": hits_n,
            "total_investment": cost_n,
            "total_payout": payout_n,
            "roi": (payout_n - cost_n) / cost_n if cost_n else 0.0,
            "hit_rate": hits_n / races_n if races_n else 0.0,
            "avg_hit_payout": float(hit_payouts_n.mean()) if hits_n else 0.0,
            "median_hit_payout": float(hit_payouts_n.median()) if hits_n else 0.0,
            "max_hit_payout": int(hit_payouts_n.max()) if hits_n else 0,
            "sample_warning": "LOW_SAMPLE" if races_n < LOW_SAMPLE_THRESHOLD else "",
        })

    out = pd.DataFrame(rows)[OUT_COLS]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    deepdive_path = PROCESSED_DIR / "niigata_deepdive.csv"
    out.to_csv(deepdive_path, index=False, encoding="utf-8-sig")
    log.info("wrote %s (%d rows)", deepdive_path, len(out))

    # ---- hits CSV
    hit_subsets = subsets[subsets["hit"]].copy()
    # aggregate per race (one trifecta hit per race, regardless of which subset)
    hit_per_race = hit_subsets.drop_duplicates("race_id").copy()
    fav_pop_lookup = entries[entries["popularity"] == 1].drop_duplicates("race_id").set_index("race_id")["popularity"]
    hits_csv = hit_per_race[[
        "race_id", "race_date", "race_name", "distance", "field_size",
        "winning_ticket", "payout_yen", "fav_odds",
    ]].rename(columns={"fav_odds": "top1_odds"})
    hits_csv["top1_popularity"] = 1  # by construction
    hits_csv = hits_csv.sort_values("race_date").reset_index(drop=True)
    hits_path = PROCESSED_DIR / "niigata_hits.csv"
    hits_csv.to_csv(hits_path, index=False, encoding="utf-8-sig")
    log.info("wrote %s (%d hits)", hits_path, len(hits_csv))

    # ---- print summary
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    print()
    print("=== OVERALL NIIGATA (strategy D) ===")
    print(f"races={total_races}  tickets={total_tickets}  hits={total_hits}  hit_rate={overall_hit_rate:.1%}")
    print(f"investment={total_cost:,}  payout={total_payout:,}  ROI={overall_roi:+.1%}")
    print(f"Bootstrap {args.bootstrap}x  95% CI ROI: [{ci_lo:+.1%}, {ci_hi:+.1%}]")
    if ci_lo < 0 < ci_hi:
        print("→ CI が 0 を跨ぐ。統計的にはまだ不確実 (偶然の可能性残る)。")
    elif ci_lo > 0:
        print("→ CI が 0 より上。統計的に有意なプラス。")
    else:
        print("→ CI が 0 より下。統計的に有意なマイナス。")

    print()
    print("=== BREAKDOWN ===")
    print(out.to_string(index=False))

    print()
    print(f"=== HIT RACES ({len(hits_csv)}) ===")
    if len(hits_csv):
        print(hits_csv.to_string(index=False))


if __name__ == "__main__":
    main()

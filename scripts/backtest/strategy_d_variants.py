"""strategy D の改良版 variants を比較する。

ベース D: 1番人気・2番人気 + (popularity>=5 AND odds 10-30) の馬 1頭 で三連単BOX (6枚)。

Variants:
  D0  現行 D (filter なし)
  D1  dark horse 3-4枠 を全 race で除外
  D2  17-18頭 race × dark horse 3-4枠 のみ除外
  D3  京都 G1 race を除外
  D4  中山 G1 race を除外
  D5  小倉 G3 race を除外
  D6  D2 + D3 + D4 + D5 (複合 negative filter)
  D7  新潟4GIII (アイビスサマーD / 関屋記念 / 新潟記念 / レパードS) のみ買う
  D8  D6 を基本にしつつ、新潟4GIII は filter 無し (carve-out)
  D9  D6 + 「プラス候補」(新潟4GIII / 3月G2 / 阪神G1 / 9月G3) のみ買う

各 variant に以下を出す:
  races / tickets / investment_yen / hits / payout_yen / profit_yen
  roi / hit_rate / max_payout_yen / max_losing_streak / avg_tickets_per_race
  bootstrap_ci_low / bootstrap_ci_high / sample_warning

Bootstrap: 1000 回 race 単位 resample。races >= 30 のみ。

出力:
  data/processed/strategy_d_variants_summary.csv
  data/processed/strategy_d_variants_by_year.csv
  data/processed/strategy_d_variants_hits.csv
  data/processed/strategy_d_variant_delta.csv
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
log = setup_logger("strategy_d_variants")

DARK_MIN_POP = 5
DARK_ODDS_MIN = 10.0
DARK_ODDS_MAX = 30.0
TICKET_COST = 100
N_TICKETS_PER_SUBSET = 6
LOW_SAMPLE_THRESHOLD = 30
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

NIIGATA_4GIII_PATTERNS = ("アイビスサマー", "関屋記念", "新潟記念", "レパード")


# ---- bucketers ------------------------------------------------------------

def field_size_band(n):
    if pd.isna(n): return "(NaN)"
    n = int(n)
    if n <= 10: return "~10"
    if n <= 13: return "11-13"
    if n <= 16: return "14-16"
    return "17-18"

def frame_band(f):
    if pd.isna(f): return "(NaN)"
    f = int(f)
    if f <= 2: return "1-2枠"
    if f <= 4: return "3-4枠"
    if f <= 6: return "5-6枠"
    return "7-8枠"


# ---- race-classification predicates ---------------------------------------

def _mask_niigata_4giii(df: pd.DataFrame) -> pd.Series:
    name_match = df["race_name"].astype(str).apply(
        lambda n: any(pat in n for pat in NIIGATA_4GIII_PATTERNS)
    )
    return name_match & (df["racecourse"] == "新潟") & (df["grade"] == "G3")


def _mask_plus_candidates(df: pd.DataFrame) -> pd.Series:
    """新潟4GIII | 3月G2 | 阪神G1 | 9月G3"""
    return (
        _mask_niigata_4giii(df)
        | ((df["month"] == 3) & (df["grade"] == "G2"))
        | ((df["racecourse"] == "阪神") & (df["grade"] == "G1"))
        | ((df["month"] == 9) & (df["grade"] == "G3"))
    )


def _mask_d6_bad(df: pd.DataFrame) -> pd.Series:
    """D6 が除外する条件 (subset-level OR race-level)。"""
    bad_subset = (df["field_size_band"] == "17-18") & (df["frame_band"] == "3-4枠")
    bad_race = (
        ((df["racecourse"] == "京都") & (df["grade"] == "G1"))
        | ((df["racecourse"] == "中山") & (df["grade"] == "G1"))
        | ((df["racecourse"] == "小倉") & (df["grade"] == "G3"))
    )
    return bad_subset | bad_race


# ---- variant filters ------------------------------------------------------

def apply_variant(subsets: pd.DataFrame, variant: str) -> pd.DataFrame:
    df = subsets
    if variant == "D0":
        return df
    if variant == "D1":
        return df[df["frame_band"] != "3-4枠"]
    if variant == "D2":
        bad = (df["field_size_band"] == "17-18") & (df["frame_band"] == "3-4枠")
        return df[~bad]
    if variant == "D3":
        bad = (df["racecourse"] == "京都") & (df["grade"] == "G1")
        return df[~bad]
    if variant == "D4":
        bad = (df["racecourse"] == "中山") & (df["grade"] == "G1")
        return df[~bad]
    if variant == "D5":
        bad = (df["racecourse"] == "小倉") & (df["grade"] == "G3")
        return df[~bad]
    if variant == "D6":
        return df[~_mask_d6_bad(df)]
    if variant == "D7":
        return df[_mask_niigata_4giii(df)]
    if variant == "D8":
        # for non-Niigata-4-GIII races, apply D6 filter; for Niigata-4-GIII, keep all
        n4g = _mask_niigata_4giii(df)
        keep = n4g | (~_mask_d6_bad(df))
        return df[keep]
    if variant == "D9":
        return df[_mask_plus_candidates(df) & ~_mask_d6_bad(df)]
    raise ValueError(f"unknown variant: {variant}")


VARIANT_DESCRIPTIONS = {
    "D0": "現行 D (filter なし)",
    "D1": "dark horse 3-4枠 除外",
    "D2": "17-18頭 × dark horse 3-4枠 除外",
    "D3": "京都 G1 race 除外",
    "D4": "中山 G1 race 除外",
    "D5": "小倉 G3 race 除外",
    "D6": "D2 + D3 + D4 + D5 (複合 negative filter)",
    "D7": "新潟4GIII のみ",
    "D8": "D6 + 新潟4GIII は filter 無し (carve-out)",
    "D9": "D6 + プラス候補 (新潟4GIII / 3月G2 / 阪神G1 / 9月G3) のみ",
}


# ---- data load + subsets --------------------------------------------------

def load_data(grades, jra_only, exclude_steeplechase):
    engine = get_engine()
    races = pd.read_sql("SELECT * FROM races", engine, parse_dates=["race_date"])
    entries = pd.read_sql("SELECT * FROM entries", engine)
    payouts = pd.read_sql("SELECT * FROM payouts", engine)
    if grades:
        races = races[races["grade"].isin(grades)]
    if jra_only:
        races = races[races["racecourse"].isin(JRA_RACECOURSES)]
    if exclude_steeplechase:
        races = races[races["surface"] != "障害"]
    keep = set(races["race_id"])
    entries = entries[entries["race_id"].isin(keep)].copy()
    payouts = payouts[payouts["race_id"].isin(keep)].copy()
    entries["popularity"] = pd.to_numeric(entries["popularity"], errors="coerce").astype("Int64")
    entries["horse_number"] = pd.to_numeric(entries["horse_number"], errors="coerce").astype("Int64")
    entries["frame_number"] = pd.to_numeric(entries["frame_number"], errors="coerce").astype("Int64")
    entries["win_odds"] = pd.to_numeric(entries["win_odds"], errors="coerce")
    entries = entries.dropna(subset=["popularity", "horse_number"])
    return races, entries, payouts


def collect_subsets(races, entries, payouts) -> pd.DataFrame:
    meta = races.set_index("race_id")[
        ["race_date", "race_name", "racecourse", "grade", "distance", "surface", "field_size"]
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
        rd = m["race_date"]
        month = int(pd.Timestamp(rd).month) if pd.notna(rd) else None
        year = int(pd.Timestamp(rd).year) if pd.notna(rd) else None
        for _, q in quals.iterrows():
            qhn = int(q["horse_number"])
            if qhn in (p1, p2):
                continue
            tickets = [f"{a}-{b}-{c}" for a, b, c in permutations([p1, p2, qhn], 3)]
            hit = bool(true_combo and (true_combo in tickets))
            rows.append({
                "race_id": rid,
                "race_date": rd,
                "year": year,
                "month": month,
                "race_name": m["race_name"],
                "racecourse": m["racecourse"],
                "grade": m["grade"],
                "distance": int(m["distance"]) if pd.notna(m["distance"]) else None,
                "surface": m["surface"],
                "field_size": int(m["field_size"]) if pd.notna(m["field_size"]) else None,
                "fav_odds": p1_odds,
                "dark_horse": qhn,
                "dark_frame": int(q["frame_number"]) if pd.notna(q["frame_number"]) else None,
                "n_tickets": N_TICKETS_PER_SUBSET,
                "investment_yen": N_TICKETS_PER_SUBSET * TICKET_COST,
                "hit": hit,
                "payout_yen": true_payout if hit else 0,
                "winning_ticket": true_combo if hit else None,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["field_size_band"] = df["field_size"].apply(field_size_band)
    df["frame_band"] = df["dark_frame"].apply(frame_band)
    return df


def collapse_to_race(subsets: pd.DataFrame) -> pd.DataFrame:
    if subsets.empty:
        return pd.DataFrame(columns=[
            "race_id", "race_date", "year", "race_name", "racecourse", "grade",
            "field_size", "n_subsets", "n_tickets", "cost", "hit", "payout",
        ])
    g = subsets.groupby("race_id", sort=False).agg(
        race_date=("race_date", "first"),
        year=("year", "first"),
        race_name=("race_name", "first"),
        racecourse=("racecourse", "first"),
        grade=("grade", "first"),
        field_size=("field_size", "first"),
        n_subsets=("dark_horse", "count"),
        n_tickets=("n_tickets", "sum"),
        cost=("investment_yen", "sum"),
        hit=("hit", "max"),
        payout=("payout_yen", "max"),
    ).reset_index()
    g["hit"] = g["hit"].astype(int)
    return g


# ---- stats helpers --------------------------------------------------------

def longest_loss_streak(per_race: pd.DataFrame) -> int:
    if per_race.empty:
        return 0
    ordered = per_race.sort_values("race_date", kind="stable", na_position="last")
    cur = best = 0
    for h in ordered["hit"]:
        if h == 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def bootstrap_ci(costs: np.ndarray, payouts: np.ndarray, rng, n=BOOTSTRAP_N):
    if len(costs) < LOW_SAMPLE_THRESHOLD:
        return None, None
    idx = rng.integers(0, len(costs), size=(n, len(costs)))
    sc = costs[idx].sum(axis=1)
    sp = payouts[idx].sum(axis=1)
    valid = sc > 0
    rois = np.zeros(n)
    rois[valid] = (sp[valid] - sc[valid]) / sc[valid]
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def compute_stats(per_race: pd.DataFrame, rng) -> dict:
    races = len(per_race)
    tickets = int(per_race["n_tickets"].sum()) if races else 0
    cost = int(per_race["cost"].sum()) if races else 0
    payout = int(per_race["payout"].sum()) if races else 0
    hits = int(per_race["hit"].sum()) if races else 0
    max_payout = int(per_race.loc[per_race["hit"] > 0, "payout"].max()) if hits else 0
    streak = longest_loss_streak(per_race)
    avg_tix = tickets / races if races else 0.0
    if races > 0:
        costs_arr = per_race["cost"].to_numpy(dtype=float)
        pays_arr = per_race["payout"].to_numpy(dtype=float)
        ci_lo, ci_hi = bootstrap_ci(costs_arr, pays_arr, rng)
    else:
        ci_lo = ci_hi = None
    return {
        "races": races,
        "tickets": tickets,
        "investment_yen": cost,
        "hits": hits,
        "payout_yen": payout,
        "profit_yen": payout - cost,
        "roi": (payout - cost) / cost if cost else 0.0,
        "hit_rate": hits / races if races else 0.0,
        "max_payout_yen": max_payout,
        "max_losing_streak": streak,
        "avg_tickets_per_race": avg_tix,
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "sample_warning": "LOW_SAMPLE" if races < LOW_SAMPLE_THRESHOLD else "",
    }


# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grades", nargs="+", default=None)
    ap.add_argument("--jra-only", action="store_true")
    ap.add_argument("--exclude-steeplechase", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    log.info("loading data...")
    races, entries, payouts = load_data(
        grades=tuple(args.grades) if args.grades else None,
        jra_only=args.jra_only,
        exclude_steeplechase=args.exclude_steeplechase,
    )
    log.info("filtered races=%d", len(races))
    base_subsets = collect_subsets(races, entries, payouts)
    log.info("base subsets=%d (across %d races)",
             len(base_subsets), base_subsets["race_id"].nunique())

    summary_rows = []
    year_rows = []
    hits_rows = []
    stats_by_variant = {}

    for variant, desc in VARIANT_DESCRIPTIONS.items():
        filtered = apply_variant(base_subsets, variant)
        per_race = collapse_to_race(filtered)
        stats = compute_stats(per_race, rng)
        stats_by_variant[variant] = stats

        summary_rows.append({"variant": variant, "description": desc, **stats})
        log.info("%s: races=%d tickets=%d ROI=%+.1f%% hits=%d",
                 variant, stats["races"], stats["tickets"], stats["roi"] * 100, stats["hits"])

        # year-level
        if not per_race.empty:
            for yr, sub in per_race.groupby("year", dropna=False):
                ys = compute_stats(sub, rng)
                year_rows.append({"variant": variant,
                                  "year": int(yr) if pd.notna(yr) else None, **ys})

        # hits
        for _, r in per_race[per_race["hit"] > 0].iterrows():
            hits_rows.append({
                "variant": variant,
                "race_id": r["race_id"],
                "race_date": r["race_date"],
                "race_name": r["race_name"],
                "racecourse": r["racecourse"],
                "grade": r["grade"],
                "field_size": r["field_size"],
                "n_tickets": r["n_tickets"],
                "investment_yen": r["cost"],
                "payout_yen": r["payout"],
                "profit_yen": r["payout"] - r["cost"],
            })

    summary_df = pd.DataFrame(summary_rows)
    year_df = pd.DataFrame(year_rows)
    hits_df = pd.DataFrame(hits_rows)

    # delta from D0
    d0 = stats_by_variant["D0"]
    delta_rows = []
    for variant in VARIANT_DESCRIPTIONS:
        if variant == "D0":
            continue
        s = stats_by_variant[variant]
        delta_rows.append({
            "variant": variant,
            "description": VARIANT_DESCRIPTIONS[variant],
            "races_d0": d0["races"],
            "races_variant": s["races"],
            "races_dropped": d0["races"] - s["races"],
            "tickets_dropped": d0["tickets"] - s["tickets"],
            "investment_dropped": d0["investment_yen"] - s["investment_yen"],
            "payout_dropped": d0["payout_yen"] - s["payout_yen"],
            "profit_change_vs_d0": s["profit_yen"] - d0["profit_yen"],
            "roi_d0": d0["roi"],
            "roi_variant": s["roi"],
            "roi_improvement_pp": (s["roi"] - d0["roi"]) * 100,
        })
    delta_df = pd.DataFrame(delta_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(PROCESSED_DIR / "strategy_d_variants_summary.csv",
                      index=False, encoding="utf-8-sig")
    year_df.to_csv(PROCESSED_DIR / "strategy_d_variants_by_year.csv",
                   index=False, encoding="utf-8-sig")
    hits_df.to_csv(PROCESSED_DIR / "strategy_d_variants_hits.csv",
                   index=False, encoding="utf-8-sig")
    delta_df.to_csv(PROCESSED_DIR / "strategy_d_variant_delta.csv",
                    index=False, encoding="utf-8-sig")
    log.info("wrote 4 CSVs")

    # ---- stdout
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n=== SUMMARY (D0 - D9) ===")
    print(summary_df.to_string(index=False))
    print("\n=== DELTA FROM D0 ===")
    print(delta_df.to_string(index=False))
    print("\n=== BY YEAR — ROI matrix ===")
    if not year_df.empty:
        pivot = year_df.pivot(index="year", columns="variant", values="roi")
        ordered = [v for v in VARIANT_DESCRIPTIONS if v in pivot.columns]
        pivot = pivot[ordered]
        print(pivot.to_string())
    print("\n=== BY YEAR — race count matrix ===")
    if not year_df.empty:
        pivot2 = year_df.pivot(index="year", columns="variant", values="races").fillna(0).astype(int)
        ordered = [v for v in VARIANT_DESCRIPTIONS if v in pivot2.columns]
        pivot2 = pivot2[ordered]
        print(pivot2.to_string())


if __name__ == "__main__":
    main()

"""strategy D を全国JRA平地G1/G2/G3で条件分解する。

新潟の +45% が「新潟という競馬場固有」か「多頭数/夏/G3/穴馬構造」の一般法則かを
検証する。

出力 (4 CSV):
  condition_deepdive_summary.csv     単一次元ROI (8 dimensions)
  condition_deepdive_grid.csv        複合条件グリッド (2-3 次元の組合せ)
  niigata_hypothesis_tests.csv       7 Niigata 仮説 + 7 comparisons
  frame_signal_tests.csv             dark horse 枠帯 × {全国/新潟/新潟以外/17-18頭}

Race-level: 1番人気+2番人気+dark horse のうち、その race 内で 1 つでも当たれば hit。
Subset-level: 各 (race, dark_horse) ペア毎を独立な賭けと見なす。frame 分析用。

Bootstrap: races >= 30 のセルのみ 1000 回 race(subset)単位 resample で 95% CI。

Usage:
    python scripts/analysis/condition_deepdive.py
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
log = setup_logger("condition_deepdive")

# strategy D parameters
DARK_MIN_POP = 5
DARK_ODDS_MIN = 10.0
DARK_ODDS_MAX = 30.0
TICKET_COST = 100
N_TICKETS_PER_SUBSET = 6

LOW_SAMPLE_THRESHOLD = 30
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

NIIGATA = "新潟"


# ---- bucketers ------------------------------------------------------------

def distance_band(d):
    if pd.isna(d): return "(NaN)"
    d = int(d)
    if d == 1000: return "1000"
    if d == 1200: return "1200"
    if d == 1400: return "1400"
    if d == 1600: return "1600"
    if d == 1800: return "1800"
    if d >= 2000: return "2000+"
    return "その他"

def field_size_band(n):
    if pd.isna(n): return "(NaN)"
    n = int(n)
    if n <= 10: return "~10"
    if n <= 13: return "11-13"
    if n <= 16: return "14-16"
    return "17-18"

def fav_odds_band(o):
    if pd.isna(o): return "(NaN)"
    if o < 2.0: return "1.0-1.9"
    if o < 3.0: return "2.0-2.9"
    if o < 5.0: return "3.0-4.9"
    return "5.0+"

def frame_band(f):
    if pd.isna(f): return "(NaN)"
    f = int(f)
    if f <= 2: return "1-2枠"
    if f <= 4: return "3-4枠"
    if f <= 6: return "5-6枠"
    return "7-8枠"


# ---- data load + subset enumeration ---------------------------------------

def load_data():
    engine = get_engine()
    races = pd.read_sql("SELECT * FROM races", engine, parse_dates=["race_date"])
    entries = pd.read_sql("SELECT * FROM entries", engine)
    payouts = pd.read_sql("SELECT * FROM payouts", engine)
    races = races[
        races["grade"].isin({"G1", "G2", "G3"})
        & races["racecourse"].isin(JRA_RACECOURSES)
        & (races["surface"] != "障害")
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


def collect_subsets(races, entries, payouts) -> pd.DataFrame:
    """One row per (race, dark_horse_candidate)."""
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

        for _, q in quals.iterrows():
            qhn = int(q["horse_number"])
            if qhn in (p1, p2):
                continue
            tickets = [f"{a}-{b}-{c}" for a, b, c in permutations([p1, p2, qhn], 3)]
            hit = bool(true_combo and (true_combo in tickets))
            rows.append({
                "race_id": rid,
                "race_date": rd,
                "race_name": m["race_name"],
                "racecourse": m["racecourse"],
                "grade": m["grade"],
                "distance": int(m["distance"]) if pd.notna(m["distance"]) else None,
                "surface": m["surface"],
                "field_size": int(m["field_size"]) if pd.notna(m["field_size"]) else None,
                "month": month,
                "fav_odds": p1_odds,
                "dark_horse": qhn,
                "dark_frame": int(q["frame_number"]) if pd.notna(q["frame_number"]) else None,
                "dark_pop": int(q["popularity"]),
                "dark_odds": float(q["win_odds"]),
                "n_tickets": N_TICKETS_PER_SUBSET,
                "investment_yen": N_TICKETS_PER_SUBSET * TICKET_COST,
                "hit": hit,
                "payout_yen": true_payout if hit else 0,
                "true_combo": true_combo,
            })
    df = pd.DataFrame(rows)
    df["distance_band"] = df["distance"].apply(distance_band)
    df["field_size_band"] = df["field_size"].apply(field_size_band)
    df["fav_odds_band"] = df["fav_odds"].apply(fav_odds_band)
    df["frame_band"] = df["dark_frame"].apply(frame_band)
    df["month_str"] = df["month"].apply(lambda m: f"{int(m):02d}" if pd.notna(m) else "(NaN)")
    return df


def collapse_to_race(subsets: pd.DataFrame) -> pd.DataFrame:
    """One row per race (multiple dark horses collapsed; hit is OR across subsets)."""
    g = subsets.groupby("race_id", sort=False).agg(
        race_date=("race_date", "first"),
        race_name=("race_name", "first"),
        racecourse=("racecourse", "first"),
        grade=("grade", "first"),
        distance=("distance", "first"),
        distance_band=("distance_band", "first"),
        surface=("surface", "first"),
        field_size=("field_size", "first"),
        field_size_band=("field_size_band", "first"),
        month=("month", "first"),
        month_str=("month_str", "first"),
        fav_odds=("fav_odds", "first"),
        fav_odds_band=("fav_odds_band", "first"),
        n_subsets=("dark_horse", "count"),
        n_tickets=("n_tickets", "sum"),
        cost=("investment_yen", "sum"),
        hit=("hit", "max"),
        payout=("payout_yen", "max"),
    ).reset_index()
    g["hit"] = g["hit"].astype(int)
    return g


# ---- aggregation core -----------------------------------------------------

def _bootstrap_ci(costs: np.ndarray, payouts: np.ndarray, rng, n=BOOTSTRAP_N):
    if len(costs) < LOW_SAMPLE_THRESHOLD:
        return None, None
    idx = rng.integers(0, len(costs), size=(n, len(costs)))
    sc = costs[idx].sum(axis=1)
    sp = payouts[idx].sum(axis=1)
    valid = sc > 0
    rois = np.zeros(n)
    rois[valid] = (sp[valid] - sc[valid]) / sc[valid]
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def _cell_stats(sub: pd.DataFrame, race_level: bool, rng) -> dict:
    if race_level:
        cost_col, pay_col, hit_col = "cost", "payout", "hit"
    else:
        cost_col, pay_col, hit_col = "investment_yen", "payout_yen", "hit"

    n = len(sub)
    cost = int(sub[cost_col].sum())
    payout = int(sub[pay_col].sum())
    hits = int(sub[hit_col].astype(int).sum())
    max_payout = int(sub.loc[sub[hit_col].astype(bool), pay_col].max()) if hits else 0
    costs_arr = sub[cost_col].to_numpy(dtype=float)
    pays_arr = sub[pay_col].to_numpy(dtype=float)
    ci_lo, ci_hi = _bootstrap_ci(costs_arr, pays_arr, rng=rng)
    return {
        "races": n,
        "tickets": int(sub["n_tickets"].sum()),
        "hits": hits,
        "investment_yen": cost,
        "payout_yen": payout,
        "profit_yen": payout - cost,
        "roi": (payout - cost) / cost if cost else 0.0,
        "hit_rate": hits / n if n else 0.0,
        "max_payout_yen": max_payout,
        "sample_warning": "LOW_SAMPLE" if n < LOW_SAMPLE_THRESHOLD else "",
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
    }


def aggregate_breakdown(detail, group_col, dim_label, rng, race_level=True):
    rows = []
    for k, sub in detail.groupby(group_col, dropna=False):
        row = {"dimension": dim_label, "group_key": "(NaN)" if pd.isna(k) else k}
        row.update(_cell_stats(sub, race_level=race_level, rng=rng))
        rows.append(row)
    return rows


def aggregate_grid(per_race, group_cols, dim_label, rng):
    rows = []
    for key, sub in per_race.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {"dimension": dim_label}
        for col, val in zip(group_cols, key):
            row[col] = "(NaN)" if pd.isna(val) else val
        row.update(_cell_stats(sub, race_level=True, rng=rng))
        rows.append(row)
    return rows


# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    log.info("loading data...")
    races, entries, payouts = load_data()
    log.info("JRA flat G1-G3: %d races", len(races))
    subsets = collect_subsets(races, entries, payouts)
    per_race = collapse_to_race(subsets)
    log.info("strategy D applicable: %d races, %d subsets", len(per_race), len(subsets))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ===========================================================================
    # 作業1: single-dimension summary
    # ===========================================================================
    summary_rows = []
    for col, dim in [
        ("racecourse", "racecourse"),
        ("month_str", "month"),
        ("grade", "grade"),
        ("field_size_band", "field_size"),
        ("distance_band", "distance"),
        ("surface", "surface"),
        ("fav_odds_band", "fav_odds"),
    ]:
        summary_rows.extend(aggregate_breakdown(per_race, col, dim, rng, race_level=True))
    # frame_band は subset-level
    summary_rows.extend(aggregate_breakdown(subsets, "frame_band", "frame", rng, race_level=False))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(PROCESSED_DIR / "condition_deepdive_summary.csv",
                      index=False, encoding="utf-8-sig")
    log.info("wrote summary (%d rows)", len(summary_df))

    # ===========================================================================
    # 作業2: composite grids
    # ===========================================================================
    grids = [
        (["racecourse", "field_size_band"],            "racecourse,field_size"),
        (["racecourse", "month_str"],                  "racecourse,month"),
        (["racecourse", "grade"],                      "racecourse,grade"),
        (["field_size_band", "month_str"],             "field_size,month"),
        (["field_size_band", "grade"],                 "field_size,grade"),
        (["month_str", "grade"],                       "month,grade"),
        (["racecourse", "field_size_band", "month_str"], "racecourse,field_size,month"),
        (["racecourse", "field_size_band", "grade"],     "racecourse,field_size,grade"),
        (["racecourse", "month_str", "grade"],           "racecourse,month,grade"),
    ]
    grid_rows = []
    for group_cols, dim_label in grids:
        grid_rows.extend(aggregate_grid(per_race, group_cols, dim_label, rng))
    # uniform schema
    for r in grid_rows:
        for col in ["racecourse", "field_size_band", "month_str", "grade"]:
            r.setdefault(col, "")

    metric_cols = ["races", "tickets", "hits", "investment_yen", "payout_yen", "profit_yen",
                   "roi", "hit_rate", "max_payout_yen", "sample_warning",
                   "bootstrap_ci_low", "bootstrap_ci_high"]
    grid_df = pd.DataFrame(grid_rows)[
        ["dimension", "racecourse", "field_size_band", "month_str", "grade"] + metric_cols
    ]
    grid_df.to_csv(PROCESSED_DIR / "condition_deepdive_grid.csv",
                   index=False, encoding="utf-8-sig")
    log.info("wrote grid (%d rows)", len(grid_df))

    # ===========================================================================
    # 作業3+4: Niigata hypothesis tests + non-Niigata comparisons
    # ===========================================================================
    is_niigata = per_race["racecourse"] == NIIGATA
    is_1718 = per_race["field_size_band"] == "17-18"
    is_summer = per_race["month"].isin([7, 8, 9])
    is_g3 = per_race["grade"] == "G3"
    all_mask = pd.Series(True, index=per_race.index)

    tests = [
        ("A", "新潟 × 17-18頭",              is_niigata & is_1718),
        ("B", "新潟 × 7-9月",                is_niigata & is_summer),
        ("C", "新潟 × G3",                   is_niigata & is_g3),
        ("D", "新潟 × 17-18 × 7-9月",        is_niigata & is_1718 & is_summer),
        ("E", "新潟 × 17-18 × G3",           is_niigata & is_1718 & is_g3),
        ("F", "新潟 × 7-9月 × G3",           is_niigata & is_summer & is_g3),
        ("G", "新潟 × 17-18 × 7-9月 × G3",   is_niigata & is_1718 & is_summer & is_g3),
        # comparisons / baselines
        ("H", "新潟 (全体)",                  is_niigata),
        ("I", "非新潟 × 17-18頭",             ~is_niigata & is_1718),
        ("J", "非新潟 × 7-9月",               ~is_niigata & is_summer),
        ("K", "非新潟 × G3",                  ~is_niigata & is_g3),
        ("L", "非新潟 × 17-18 × 7-9 × G3",   ~is_niigata & is_1718 & is_summer & is_g3),
        ("M", "非新潟 (全体)",                ~is_niigata),
        ("N", "全国 (全体)",                  all_mask),
    ]
    hypo_rows = []
    for label, name, mask in tests:
        sub = per_race[mask]
        row = {"label": label, "hypothesis": name}
        row.update(_cell_stats(sub, race_level=True, rng=rng))
        hit_sub = sub[sub["hit"] > 0]
        row["hit_race_ids"] = ";".join(hit_sub["race_id"].astype(str).tolist())
        row["hit_race_names"] = ";".join(hit_sub["race_name"].astype(str).tolist())
        hypo_rows.append(row)
    hypo_df = pd.DataFrame(hypo_rows)
    hypo_df.to_csv(PROCESSED_DIR / "niigata_hypothesis_tests.csv",
                   index=False, encoding="utf-8-sig")
    log.info("wrote hypothesis tests (%d rows)", len(hypo_df))

    # ===========================================================================
    # 作業5: frame signal tests
    # ===========================================================================
    contexts = [
        ("全国", subsets),
        ("新潟", subsets[subsets["racecourse"] == NIIGATA]),
        ("非新潟", subsets[subsets["racecourse"] != NIIGATA]),
        ("17-18頭限定", subsets[subsets["field_size_band"] == "17-18"]),
    ]
    frame_rows = []
    for ctx_name, ctx_subsets in contexts:
        for band in ["1-2枠", "3-4枠", "5-6枠", "7-8枠"]:
            sub = ctx_subsets[ctx_subsets["frame_band"] == band]
            row = {"context": ctx_name, "frame_band": band}
            row.update(_cell_stats(sub, race_level=False, rng=rng))
            frame_rows.append(row)
    frame_df = pd.DataFrame(frame_rows)
    frame_df.to_csv(PROCESSED_DIR / "frame_signal_tests.csv",
                    index=False, encoding="utf-8-sig")
    log.info("wrote frame signal tests (%d rows)", len(frame_df))

    # ===========================================================================
    # stdout summary
    # ===========================================================================
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== 作業1: SINGLE-DIM SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== 作業3+4: NIIGATA HYPOTHESIS TESTS ===")
    print(hypo_df.drop(columns=["hit_race_ids", "hit_race_names"]).to_string(index=False))
    print()
    print("--- hit race names per hypothesis ---")
    for _, r in hypo_df.iterrows():
        if r["hits"]:
            print(f"  [{r['label']}] {r['hypothesis']}  hits={int(r['hits'])}:")
            for nm in (r["hit_race_names"] or "").split(";"):
                if nm:
                    print(f"      {nm}")

    print("\n=== 作業5: FRAME SIGNAL TESTS ===")
    print(frame_df.to_string(index=False))

    # show top non-LOW_SAMPLE grid cells by ROI
    valid_grid = grid_df[(grid_df["sample_warning"] == "") & (grid_df["races"] >= LOW_SAMPLE_THRESHOLD)]
    print("\n=== TOP-15 GRID CELLS (races>=30) by ROI ===")
    print(valid_grid.sort_values("roi", ascending=False).head(15).to_string(index=False))
    print("\n=== BOTTOM-10 GRID CELLS (races>=30) by ROI ===")
    print(valid_grid.sort_values("roi").head(10).to_string(index=False))


if __name__ == "__main__":
    main()

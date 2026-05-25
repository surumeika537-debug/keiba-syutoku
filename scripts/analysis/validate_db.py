"""Validate the contents of data/db/keiba.sqlite — sanity checks on ingest+parse output.

Usage:
    python scripts/analysis/validate_db.py
    python scripts/analysis/validate_db.py --jra-only --exclude-steeplechase --grades G1 G2 G3
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.config import JRA_RACECOURSES
from src.database import get_engine
from src.utils import BET_TYPE_CANONICAL_SET, force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("validate")

TRIFECTA_LABELS = ("三連単",)  # post-normalization there's only one label
EXPECTED_GRADES = {"G1", "G2", "G3"}
TRIFECTA_COMBO_PAT = re.compile(r"^\d+-\d+-\d+$")


def section(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, 60 - len(title)))


def fmt_pct(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{num}/{den} ({num/den:.1%})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-anomalies", type=int, default=20,
                    help="how many anomalous race_ids to print per category")
    ap.add_argument("--grades", nargs="+", default=None, help="filter by grade (e.g. G1 G2 G3)")
    ap.add_argument("--jra-only", action="store_true", help="keep only JRA (10場) races")
    ap.add_argument("--exclude-steeplechase", action="store_true", help="drop surface==障害")
    args = ap.parse_args()

    engine = get_engine()
    races_all = pd.read_sql("SELECT * FROM races", engine, parse_dates=["race_date"])
    entries_all = pd.read_sql("SELECT * FROM entries", engine)
    payouts_all = pd.read_sql("SELECT * FROM payouts", engine)

    # ---- apply filters
    races = races_all.copy()
    if args.grades:
        races = races[races["grade"].isin(args.grades)]
    if args.jra_only:
        races = races[races["racecourse"].isin(JRA_RACECOURSES)]
    if args.exclude_steeplechase:
        races = races[races["surface"] != "障害"]

    keep = set(races["race_id"])
    entries = entries_all[entries_all["race_id"].isin(keep)].copy()
    payouts = payouts_all[payouts_all["race_id"].isin(keep)].copy()

    if any([args.grades, args.jra_only, args.exclude_steeplechase]):
        section("0. filters applied")
        print(f"  grades              : {args.grades or '(none)'}")
        print(f"  jra_only            : {args.jra_only}")
        print(f"  exclude_steeplechase: {args.exclude_steeplechase}")
        print(f"  races  : {len(races_all):>6}  ->  {len(races):>6}")
        print(f"  entries: {len(entries_all):>6}  ->  {len(entries):>6}")
        print(f"  payouts: {len(payouts_all):>6}  ->  {len(payouts):>6}")

    # ---- row counts
    section("1-3. row counts")
    print(f"races   : {len(races):>6}")
    print(f"entries : {len(entries):>6}")
    print(f"payouts : {len(payouts):>6}")
    if races.empty:
        print("\n(no races match filters — run ingest + parse first, or relax filters)")
        return

    # ---- year / grade / field size
    section("4. races per year")
    by_year = (races.assign(year=races["race_date"].dt.year)
               .groupby("year", dropna=False).size().reset_index(name="races")
               .sort_values("year", na_position="last"))
    print(by_year.to_string(index=False))

    section("5. races per grade")
    by_grade = races["grade"].fillna("(none)").value_counts(dropna=False).reset_index()
    by_grade.columns = ["grade", "races"]
    print(by_grade.to_string(index=False))
    unexpected = set(races["grade"].dropna().unique()) - EXPECTED_GRADES
    if unexpected:
        print(f"  ! unexpected grade values: {sorted(unexpected)}")

    section("6. field size")
    field = entries.groupby("race_id").size()
    print(f"races with entries : {len(field)}")
    if len(field):
        print(f"avg field size     : {field.mean():.2f}")
        print(f"min / max          : {field.min()} / {field.max()}")

    # ---- trifecta coverage
    section("7. trifecta payouts")
    tri = payouts[payouts["bet_type"].isin(TRIFECTA_LABELS)]
    tri_races = tri["race_id"].nunique()
    print(f"races with 三連単 payout : {fmt_pct(tri_races, len(races))}")

    # ---- missingness
    section("8. win_odds (REAL) NULL rate")
    odds_num = pd.to_numeric(entries["win_odds"], errors="coerce")
    miss_odds = int(odds_num.isna().sum())
    print(f"NULL/invalid odds : {fmt_pct(miss_odds, len(entries))}")

    section("9. popularity NULL rate")
    pop = pd.to_numeric(entries["popularity"], errors="coerce")
    miss_pop = int(pop.isna().sum())
    print(f"NULL/invalid popularity : {fmt_pct(miss_pop, len(entries))}")

    section("10. finish_status distribution")
    status_dist = entries["finish_status"].fillna("(NULL)").value_counts(dropna=False).reset_index()
    status_dist.columns = ["finish_status", "entries"]
    print(status_dist.to_string(index=False))
    # finish_position numeric check
    fp_num = pd.to_numeric(entries["finish_position"], errors="coerce")
    no_position_with_完走 = int(((entries["finish_status"] == "完走") & fp_num.isna()).sum())
    has_position_without_完走 = int((fp_num.notna() & (entries["finish_status"] != "完走")).sum())
    if no_position_with_完走 or has_position_without_完走:
        print(f"  WARN: finish_position/status mismatch: 完走_no_pos={no_position_with_完走}, "
              f"pos_no_完走={has_position_without_完走}")

    # ---- bet_type coverage
    section("11. bet_type distribution")
    bt = payouts["bet_type"].fillna("(NULL)").value_counts().reset_index()
    bt.columns = ["bet_type", "rows"]
    print(bt.to_string(index=False))
    unknown_bt = sorted(set(payouts["bet_type"].dropna()) - set(BET_TYPE_CANONICAL_SET))
    if unknown_bt:
        print(f"  ! non-canonical bet_type values present: {unknown_bt}")

    # ---- trifecta combination format
    section("12. 三連単 combination format")
    bad_combo = tri[~tri["combination"].astype(str).str.match(TRIFECTA_COMBO_PAT)]
    print(f"malformed trifecta combination rows : {len(bad_combo)} / {len(tri)}")
    if len(bad_combo):
        print(bad_combo[["race_id", "bet_type", "combination", "payout_yen"]]
              .head(args.max_anomalies).to_string(index=False))

    # ---- per-race duplicate checks
    section("13. duplicate horse_number within race (must be 0)")
    hn_dup = (entries.groupby(["race_id", "horse_number"]).size()
              .reset_index(name="n"))
    hn_dup = hn_dup[hn_dup["n"] > 1]
    print(f"duplicate (race_id, horse_number) rows : {len(hn_dup)}")
    if len(hn_dup):
        print(hn_dup.head(args.max_anomalies).to_string(index=False))

    section("14. duplicate popularity within race (warning — dead-heat possible)")
    pop_nonnull = entries.dropna(subset=["popularity"])
    pop_dup = (pop_nonnull.groupby(["race_id", "popularity"]).size()
               .reset_index(name="n"))
    pop_dup = pop_dup[pop_dup["n"] > 1]
    print(f"races with duplicate popularity : {pop_dup['race_id'].nunique()}")
    if len(pop_dup):
        print(pop_dup.head(args.max_anomalies).to_string(index=False))

    # ---- top-3 completeness (allowing dead-heats: positions are tied, e.g. 1,2,2,4 is valid)
    section("15. races with fewer than 3 horses in top-3 (anomaly; dead-heat OK)")
    fp_int = pd.to_numeric(entries["finish_position"], errors="coerce").astype("Int64")
    top3_counts = (entries.assign(_pos=fp_int).dropna(subset=["_pos"])
                   .groupby("race_id")["_pos"].apply(lambda s: int((s <= 3).sum())))
    incomplete_ids = set(top3_counts[top3_counts < 3].index)
    races_with_any_pos = set(top3_counts.index)
    no_pos_ids = set(races["race_id"]) - races_with_any_pos
    incomplete_total = sorted(incomplete_ids | no_pos_ids)
    print(f"count : {len(incomplete_total)}")
    if incomplete_total:
        sub = races[races["race_id"].isin(incomplete_total)][["race_id", "race_date", "race_name", "grade"]]
        print(sub.head(args.max_anomalies).to_string(index=False))

    # ---- 0-entry / missing-trifecta anomalies
    section("16. races with 0 entries (anomaly)")
    empty_races = races[~races["race_id"].isin(set(entries["race_id"].unique()))]
    print(f"count : {len(empty_races)}")
    if len(empty_races):
        print(empty_races[["race_id", "race_date", "race_name", "grade"]]
              .head(args.max_anomalies).to_string(index=False))

    section("17. races missing 三連単 payout (anomaly)")
    missing_tri = races[~races["race_id"].isin(set(tri["race_id"].unique()))]
    print(f"count : {len(missing_tri)}")
    if len(missing_tri):
        print(missing_tri[["race_id", "race_date", "race_name", "grade"]]
              .head(args.max_anomalies).to_string(index=False))

    # ---- verdict
    section("verdict (advisory)")
    issues = []
    if tri_races / max(1, len(races)) < 0.95:
        issues.append(f"low 三連単 coverage ({tri_races}/{len(races)})")
    if races["grade"].isna().mean() > 0.05:
        issues.append(f"grade missing rate {races['grade'].isna().mean():.1%}")
    if len(entries):
        if miss_odds / len(entries) > 0.05:
            issues.append(f"win_odds missing rate {miss_odds/len(entries):.1%}")
        if miss_pop / len(entries) > 0.05:
            issues.append(f"popularity missing rate {miss_pop/len(entries):.1%}")
    if len(hn_dup):
        issues.append(f"{len(hn_dup)} duplicate (race_id,horse_number) rows")
    if len(bad_combo):
        issues.append(f"{len(bad_combo)} malformed 三連単 combinations")
    if unknown_bt:
        issues.append(f"non-canonical bet_type: {unknown_bt}")
    if incomplete_total:
        issues.append(f"{len(incomplete_total)} races without complete top3 finish positions")
    if not issues:
        print("OK - no major data quality issues detected")
    else:
        for s in issues:
            print(f"WARN : {s}")


if __name__ == "__main__":
    main()

"""odds_snapshots テーブルの品質チェック。

以下を snapshot_time_label 別に出力:
  - races               この snapshot を持つ race 数
  - horses              総 row 数
  - coverage_pct        D9-eligible universe (JRA 平地 G1/G2/G3) に対するカバレッジ
  - popularity_null_pct
  - win_odds_null_pct
  - duplicate_rows      (race_id, label, horse_number) の重複行数
  - missing_horses_avg  per race 平均 (entries にいるのに snapshot にいない)
  - races_with_missing  上記が >0 のレース数

Usage:
    python scripts/analysis/validate_snapshot_coverage.py
    python scripts/analysis/validate_snapshot_coverage.py --max-anomalies 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.config import JRA_RACECOURSES, PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("validate_snapshot")


def section(title: str):
    print()
    print(f"--- {title} " + "-" * max(0, 60 - len(title)))


def fmt_pct(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{num}/{den} ({num/den:.1%})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-anomalies", type=int, default=10)
    args = ap.parse_args()

    engine = get_engine()
    races_all = pd.read_sql("SELECT * FROM races", engine, parse_dates=["race_date"])
    entries_all = pd.read_sql("SELECT race_id, horse_number FROM entries", engine)
    snaps_all = pd.read_sql("SELECT * FROM odds_snapshots", engine, parse_dates=["captured_at", "created_at"])

    # D9-eligible universe (= JRA flat G1-G3) for coverage denominator
    d9_universe = races_all[
        races_all["grade"].isin({"G1", "G2", "G3"})
        & races_all["racecourse"].isin(JRA_RACECOURSES)
        & (races_all["surface"] != "障害")
    ]
    universe_size = len(d9_universe)

    print(f"odds_snapshots total rows : {len(snaps_all)}")
    print(f"distinct races covered    : {snaps_all['race_id'].nunique()}")
    print(f"D9-eligible universe size : {universe_size}")
    print(f"snapshot labels present   : {sorted(snaps_all['snapshot_time_label'].unique().tolist())}")
    print(f"sources present           : {sorted(snaps_all['source'].dropna().unique().tolist())}")

    if snaps_all.empty:
        print("\n(odds_snapshots is empty — nothing to validate)")
        return

    summary_rows = []
    anomalies_per_label = {}
    for label, grp in snaps_all.groupby("snapshot_time_label"):
        section(f"label = {label!r}")
        races_with_snap = grp["race_id"].nunique()
        n_horses = len(grp)
        pop_null = int(grp["popularity"].isna().sum())
        odds_null = int(grp["win_odds"].isna().sum())
        # duplicates: row count > 1 per (race_id, horse_number)
        dup = (grp.groupby(["race_id", "horse_number"]).size().reset_index(name="n"))
        dup = dup[dup["n"] > 1]
        # missing horses per race: races where # entries > # snapshot rows for that race
        entries_per_race = (entries_all[entries_all["race_id"].isin(grp["race_id"])]
                            .groupby("race_id").size())
        snap_per_race = grp.groupby("race_id").size()
        missing_per_race = (entries_per_race - snap_per_race).fillna(entries_per_race).clip(lower=0)
        missing_per_race = missing_per_race[missing_per_race > 0]

        coverage_pct = races_with_snap / universe_size if universe_size > 0 else 0.0
        print(f"races_with_snapshot   : {races_with_snap}")
        print(f"horses (total rows)   : {n_horses}")
        print(f"coverage              : {races_with_snap}/{universe_size} ({coverage_pct:.1%})")
        print(f"popularity NULL       : {fmt_pct(pop_null, n_horses)}")
        print(f"win_odds NULL         : {fmt_pct(odds_null, n_horses)}")
        print(f"duplicate (race,horse): {len(dup)} rows")
        print(f"races missing horses  : {len(missing_per_race)} "
              f"(avg missing/race = {missing_per_race.mean() if len(missing_per_race) else 0:.1f})")
        if len(missing_per_race):
            print("  sample races with missing horses:")
            for rid, m in missing_per_race.head(args.max_anomalies).items():
                print(f"    {rid}: {int(m)} horse(s) missing")
        if len(dup):
            print("  sample duplicate rows:")
            print(dup.head(args.max_anomalies).to_string(index=False))

        summary_rows.append({
            "snapshot_time_label": label,
            "races": races_with_snap,
            "horses": n_horses,
            "coverage_pct": round(coverage_pct, 4),
            "popularity_null_pct": round(pop_null / n_horses, 4) if n_horses else 0,
            "win_odds_null_pct": round(odds_null / n_horses, 4) if n_horses else 0,
            "duplicate_rows": len(dup),
            "races_with_missing_horses": len(missing_per_race),
            "avg_missing_per_race": round(float(missing_per_race.mean()), 2) if len(missing_per_race) else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "odds_snapshot_coverage.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("wrote %s", out_path)

    section("verdict")
    issues = []
    for r in summary_rows:
        if r["duplicate_rows"]:
            issues.append(f"{r['snapshot_time_label']}: {r['duplicate_rows']} duplicate rows")
        if r["popularity_null_pct"] > 0.05:
            issues.append(f"{r['snapshot_time_label']}: popularity NULL "
                          f"{r['popularity_null_pct']:.1%}")
        if r["win_odds_null_pct"] > 0.05:
            issues.append(f"{r['snapshot_time_label']}: win_odds NULL "
                          f"{r['win_odds_null_pct']:.1%}")
        if r["races_with_missing_horses"] > 0:
            issues.append(f"{r['snapshot_time_label']}: {r['races_with_missing_horses']} "
                          f"races missing horses (avg {r['avg_missing_per_race']})")
    if not issues:
        print("OK - no major snapshot coverage issues")
    else:
        for i in issues:
            print(f"WARN : {i}")


if __name__ == "__main__":
    main()

"""odds_snapshots の 2 つの snapshot_time_label 間で odds/popularity 変動を分析。

通常使用例: --label-a 30min --label-b final  で「30分前 → 確定」のドリフトを見る。

出力:
  - per-race CSV (data/processed/odds_drift_report.csv)
      race_id, race_date, race_name, n_horses_compared,
      avg_odds_drift_pct, max_odds_drift_pct,
      avg_pop_drift, max_pop_drift,
      dark_candidates_a, dark_candidates_b,
      dark_in_both, dark_added_in_b, dark_removed_in_b, dark_churn_count,
      p1_horse_a, p1_horse_b, p1_reversal_flag,
      p2_horse_a, p2_horse_b, p2_reversal_flag,
      late_steam_horses, late_fade_horses
  - stdout summary

定義:
  - late steam: snapshot_b で odds が 20% 以上下がった (= 人気上昇) 馬
  - late fade : snapshot_b で odds が 20% 以上上がった (= 人気低下) 馬
  - dark candidate: popularity>=5 AND 10 <= win_odds <= 30 (strategy D 条件)
  - dark churn: 候補集合の対称差 = added + removed
  - p1/p2 reversal: a と b で 1番人気 (もしくは 2番人気) が別の馬

Usage:
    python scripts/analysis/odds_drift_report.py
    python scripts/analysis/odds_drift_report.py --label-a 60min --label-b 10min
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("odds_drift")

STEAM_FADE_THRESHOLD = 0.20  # 20% odds change → "late steam" / "late fade"
DARK_POP_MIN = 5
DARK_ODDS_MIN = 10.0
DARK_ODDS_MAX = 30.0


def _dark_set(df_race: pd.DataFrame, suffix: str) -> set:
    """Return set of horse_numbers that qualify as dark in this snapshot."""
    pop_col = f"popularity_{suffix}"
    odds_col = f"win_odds_{suffix}"
    sub = df_race[
        (df_race[pop_col].notna())
        & (df_race[odds_col].notna())
        & (df_race[pop_col] >= DARK_POP_MIN)
        & (df_race[odds_col] >= DARK_ODDS_MIN)
        & (df_race[odds_col] <= DARK_ODDS_MAX)
    ]
    return set(int(x) for x in sub["horse_number"].astype(int).tolist())


def _horse_at_popularity(df_race: pd.DataFrame, suffix: str, target_pop: int):
    pop_col = f"popularity_{suffix}"
    sub = df_race[df_race[pop_col] == target_pop]
    if sub.empty:
        return None
    return int(sub.iloc[0]["horse_number"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-a", default="30min", help="earlier snapshot label")
    ap.add_argument("--label-b", default="final", help="later snapshot label")
    args = ap.parse_args()

    engine = get_engine()
    snaps = pd.read_sql(
        text("SELECT race_id, snapshot_time_label, horse_number, popularity, win_odds "
             "FROM odds_snapshots WHERE snapshot_time_label IN (:la, :lb)"),
        engine, params={"la": args.label_a, "lb": args.label_b},
    )
    races_meta = pd.read_sql("SELECT race_id, race_date, race_name FROM races", engine,
                              parse_dates=["race_date"])

    if snaps.empty:
        log.error("no rows in odds_snapshots for labels (%s, %s)",
                  args.label_a, args.label_b)
        sys.exit(1)

    # split into a / b and merge on (race_id, horse_number)
    snap_a = (snaps[snaps["snapshot_time_label"] == args.label_a]
              .rename(columns={"popularity": "popularity_a", "win_odds": "win_odds_a"})
              [["race_id", "horse_number", "popularity_a", "win_odds_a"]])
    snap_b = (snaps[snaps["snapshot_time_label"] == args.label_b]
              .rename(columns={"popularity": "popularity_b", "win_odds": "win_odds_b"})
              [["race_id", "horse_number", "popularity_b", "win_odds_b"]])
    merged = snap_a.merge(snap_b, on=["race_id", "horse_number"], how="inner")
    if merged.empty:
        log.error("no overlapping (race_id, horse_number) rows between labels %s and %s",
                  args.label_a, args.label_b)
        sys.exit(1)

    merged["popularity_a"] = pd.to_numeric(merged["popularity_a"], errors="coerce")
    merged["popularity_b"] = pd.to_numeric(merged["popularity_b"], errors="coerce")
    merged["win_odds_a"] = pd.to_numeric(merged["win_odds_a"], errors="coerce")
    merged["win_odds_b"] = pd.to_numeric(merged["win_odds_b"], errors="coerce")

    # per-horse drift fields
    merged["odds_drift_pct"] = (merged["win_odds_b"] - merged["win_odds_a"]) / merged["win_odds_a"]
    merged["pop_drift"] = merged["popularity_b"] - merged["popularity_a"]
    merged["is_late_steam"] = (merged["odds_drift_pct"] <= -STEAM_FADE_THRESHOLD).fillna(False)
    merged["is_late_fade"] = (merged["odds_drift_pct"] >= STEAM_FADE_THRESHOLD).fillna(False)

    # per race aggregation
    race_rows = []
    for race_id, g in merged.groupby("race_id", sort=False):
        # both snapshots must have non-null odds for drift calc; skip otherwise
        valid = g.dropna(subset=["win_odds_a", "win_odds_b"])
        n_horses = len(valid)
        if n_horses == 0:
            continue
        avg_odds_drift = float(valid["odds_drift_pct"].abs().mean())
        max_odds_drift = float(valid["odds_drift_pct"].abs().max())
        valid_pop = g.dropna(subset=["popularity_a", "popularity_b"])
        avg_pop_drift = float(valid_pop["pop_drift"].abs().mean()) if len(valid_pop) else float("nan")
        max_pop_drift = int(valid_pop["pop_drift"].abs().max()) if len(valid_pop) else 0

        dark_a = _dark_set(g, "a")
        dark_b = _dark_set(g, "b")
        added = dark_b - dark_a
        removed = dark_a - dark_b
        in_both = dark_a & dark_b

        p1_a = _horse_at_popularity(g, "a", 1)
        p1_b = _horse_at_popularity(g, "b", 1)
        p2_a = _horse_at_popularity(g, "a", 2)
        p2_b = _horse_at_popularity(g, "b", 2)

        steam_horses = sorted(g.loc[g["is_late_steam"], "horse_number"].astype(int).tolist())
        fade_horses = sorted(g.loc[g["is_late_fade"], "horse_number"].astype(int).tolist())

        race_rows.append({
            "race_id": race_id,
            "n_horses_compared": n_horses,
            "avg_odds_drift_pct": round(avg_odds_drift, 4),
            "max_odds_drift_pct": round(max_odds_drift, 4),
            "avg_pop_drift": round(avg_pop_drift, 3),
            "max_pop_drift": max_pop_drift,
            "dark_candidates_a": ";".join(str(x) for x in sorted(dark_a)),
            "dark_candidates_b": ";".join(str(x) for x in sorted(dark_b)),
            "dark_in_both": len(in_both),
            "dark_added_in_b": ";".join(str(x) for x in sorted(added)),
            "dark_removed_in_b": ";".join(str(x) for x in sorted(removed)),
            "dark_churn_count": len(added) + len(removed),
            "p1_horse_a": p1_a,
            "p1_horse_b": p1_b,
            "p1_reversal_flag": (p1_a is not None and p1_b is not None and p1_a != p1_b),
            "p2_horse_a": p2_a,
            "p2_horse_b": p2_b,
            "p2_reversal_flag": (p2_a is not None and p2_b is not None and p2_a != p2_b),
            "late_steam_horses": ";".join(str(x) for x in steam_horses),
            "late_fade_horses": ";".join(str(x) for x in fade_horses),
        })

    out_df = pd.DataFrame(race_rows)
    out_df = out_df.merge(races_meta, on="race_id", how="left")
    # reorder
    front = ["race_id", "race_date", "race_name"]
    rest = [c for c in out_df.columns if c not in front]
    out_df = out_df[front + rest].sort_values("race_date", kind="stable")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "odds_drift_report.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("wrote %s (%d rows)", out_path, len(out_df))

    # ---- summary
    n_races = len(out_df)
    if n_races == 0:
        return
    avg_odds_drift = float(out_df["avg_odds_drift_pct"].mean())
    avg_pop_drift = float(out_df["avg_pop_drift"].mean())
    avg_dark_churn = float(out_df["dark_churn_count"].mean())
    n_p1_rev = int(out_df["p1_reversal_flag"].sum())
    n_p2_rev = int(out_df["p2_reversal_flag"].sum())
    n_with_steam = int((out_df["late_steam_horses"].str.len() > 0).sum())
    n_with_fade = int((out_df["late_fade_horses"].str.len() > 0).sum())

    print()
    print(f"=== ODDS DRIFT REPORT  ({args.label_a} → {args.label_b}) ===")
    print(f"  races compared             : {n_races}")
    print(f"  avg per-horse odds drift   : {avg_odds_drift:.1%} (|odds_b - odds_a| / odds_a)")
    print(f"  avg per-horse pop drift    : {avg_pop_drift:.2f} (|pop_b - pop_a|)")
    print(f"  avg dark-candidate churn   : {avg_dark_churn:.2f} horses/race")
    print(f"  races with p1 reversal     : {n_p1_rev} ({n_p1_rev/n_races:.1%})")
    print(f"  races with p2 reversal     : {n_p2_rev} ({n_p2_rev/n_races:.1%})")
    print(f"  races with late_steam horse: {n_with_steam} ({n_with_steam/n_races:.1%})")
    print(f"  races with late_fade horse : {n_with_fade} ({n_with_fade/n_races:.1%})")

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 28)
    print()
    cols = ["race_date", "race_name", "n_horses_compared",
            "avg_odds_drift_pct", "avg_pop_drift",
            "dark_churn_count", "p1_reversal_flag", "p2_reversal_flag",
            "late_steam_horses", "late_fade_horses"]
    print(out_df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

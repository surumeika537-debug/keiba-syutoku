"""E_D9_P3_CAP4 の買い目を DB から自動生成して CSV に保存する。

対象指定:
  --date YYYY-MM-DD     : その日のレースだけ
  --year YYYY           : その年のレース全部
  --race-id ID          : 個別の race_id 指定
  --snapshot-time STR   : odds snapshot のラベル (default "final")
                          現状の DB は確定odds のみ。将来 "30min" 等の
                          発走前 snapshot に差し替える前提のため、
                          ラベルだけ CSV に記録しておく。
  (date/year/race-id は複数指定可、OR 結合)

出力 (data/processed/live_tickets_{suffix}_{snapshot}.csv) は 1 行 = 1 ticket。
追加メタ列: snapshot_time / odds_source / generated_at / rule_name

Usage:
    python scripts/live/generate_tickets.py --date 2025-09-07
    python scripts/live/generate_tickets.py --year 2025 --snapshot-time final
    python scripts/live/generate_tickets.py --race-id 202504030811 --snapshot-time 30min
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
from scripts.backtest.simple_roi import (
    d9_component_labels,
    is_d6_excluded_race,
    is_d9_candidate_race,
    select_d9_p3_cap4_darks,
    _horse_by_popularity,
)

force_utf8_stdout()
log = setup_logger("live.generate_tickets")

STAKE_PER_TICKET = 100
RULE_NAME = "E_D9_P3_CAP4"
# Source identifier when we successfully loaded per-horse odds from odds_snapshots.
SOURCE_DB_SNAPSHOT = "db_snapshot"
# When odds_snapshots has no row for the (race, label), we fall back to entries (final odds).
SOURCE_DB_FALLBACK = "db_final_fallback"


def load_filtered(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (races, entries_with_final_odds, snapshot_odds).
    snapshot_odds is empty DataFrame if --snapshot-time not provided or no rows match."""
    from sqlalchemy import text
    engine = get_engine()
    races = pd.read_sql("SELECT * FROM races", engine, parse_dates=["race_date"])
    entries = pd.read_sql("SELECT * FROM entries", engine)

    # JRA flat G1/G2/G3 baseline filter (D9 only applies here)
    races = races[
        races["grade"].isin({"G1", "G2", "G3"})
        & races["racecourse"].isin(JRA_RACECOURSES)
        & (races["surface"] != "障害")
    ].copy()

    # selection filters
    mask = pd.Series(False, index=races.index)
    if args.date:
        mask |= races["race_date"] == pd.Timestamp(args.date)
    if args.year:
        mask |= races["race_date"].dt.year == int(args.year)
    if args.race_id:
        mask |= races["race_id"].astype(str).isin(args.race_id)
    if not (args.date or args.year or args.race_id):
        # no selector → all
        mask = pd.Series(True, index=races.index)
    races = races[mask].copy()

    keep = set(races["race_id"])
    entries = entries[entries["race_id"].isin(keep)].copy()
    entries["popularity"] = pd.to_numeric(entries["popularity"], errors="coerce").astype("Int64")
    entries["horse_number"] = pd.to_numeric(entries["horse_number"], errors="coerce").astype("Int64")
    entries["frame_number"] = pd.to_numeric(entries["frame_number"], errors="coerce").astype("Int64")
    entries["win_odds"] = pd.to_numeric(entries["win_odds"], errors="coerce")
    entries = entries.dropna(subset=["popularity", "horse_number"])

    # snapshot odds (if a snapshot label was requested)
    snapshot_df = pd.DataFrame(columns=["race_id", "horse_number", "popularity", "win_odds"])
    if getattr(args, "snapshot_time", None):
        snapshot_df = pd.read_sql(
            text("SELECT race_id, horse_number, popularity, win_odds, source "
                 "FROM odds_snapshots WHERE snapshot_time_label = :label"),
            engine, params={"label": args.snapshot_time},
        )
        snapshot_df = snapshot_df[snapshot_df["race_id"].isin(keep)]
    return races, entries, snapshot_df


def merge_odds_for_race(race_entries: pd.DataFrame,
                         snapshot_for_race: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """If snapshot rows exist for this race, override popularity/win_odds per horse.
    Returns (merged_entries, effective_odds_source).
    """
    if snapshot_for_race is None or snapshot_for_race.empty:
        return race_entries, SOURCE_DB_FALLBACK
    out = race_entries.copy()
    snap_map = snapshot_for_race.set_index("horse_number")[["popularity", "win_odds"]]
    # vectorized override using map; fall back to existing value if missing
    snap_pop = out["horse_number"].map(snap_map["popularity"])
    snap_odds = out["horse_number"].map(snap_map["win_odds"])
    out["popularity"] = snap_pop.combine_first(out["popularity"])
    out["win_odds"] = snap_odds.combine_first(out["win_odds"])
    # ensure int dtype after the override
    out["popularity"] = pd.to_numeric(out["popularity"], errors="coerce").astype("Int64")
    return out, SOURCE_DB_SNAPSHOT


def build_tickets_for_race(race_row: pd.Series, race_entries: pd.DataFrame,
                            snapshot_time: str, odds_source: str,
                            generated_at: str) -> list[dict]:
    """Apply E_D9_P3_CAP4 to a single race; return list of ticket-row dicts."""
    rd = race_row["race_date"]
    race_meta = {
        "race_id": race_row["race_id"],
        "race_name": race_row.get("race_name"),
        "race_date": rd,
        "racecourse": race_row.get("racecourse"),
        "grade": race_row.get("grade"),
        "surface": race_row.get("surface"),
        "distance": race_row.get("distance"),
        "month": int(pd.Timestamp(rd).month) if pd.notna(rd) else None,
        "field_size": len(race_entries),
    }
    if not is_d9_candidate_race(race_meta):
        return []
    if is_d6_excluded_race(race_meta):
        return []
    p1 = _horse_by_popularity(race_entries, 1)
    p2 = _horse_by_popularity(race_entries, 2)
    if p1 is None or p2 is None:
        return []
    p1_row = race_entries[race_entries["horse_number"] == p1].iloc[0]
    p2_row = race_entries[race_entries["horse_number"] == p2].iloc[0]
    darks = select_d9_p3_cap4_darks(race_entries, race_meta, max_darks=4)
    if darks.empty:
        return []

    components = d9_component_labels(race_meta)
    rows = []
    for _, dk in darks.iterrows():
        d_num = int(dk["horse_number"])
        for first, second in [(p1, p2), (p2, p1)]:
            ticket = f"{first}-{second}-{d_num}"
            rows.append({
                "race_id": race_row["race_id"],
                "race_date": rd,
                "race_name": race_row.get("race_name"),
                "grade": race_row.get("grade"),
                "racecourse": race_row.get("racecourse"),
                "surface": race_row.get("surface"),
                "distance": race_row.get("distance"),
                "field_size": race_meta["field_size"],
                "component_labels": components,
                "p1_horse_number": p1,
                "p1_horse_name": p1_row.get("horse_name"),
                "p1_odds": float(p1_row.get("win_odds")) if pd.notna(p1_row.get("win_odds")) else None,
                "p2_horse_number": p2,
                "p2_horse_name": p2_row.get("horse_name"),
                "p2_odds": float(p2_row.get("win_odds")) if pd.notna(p2_row.get("win_odds")) else None,
                "dark_horse_number": d_num,
                "dark_horse_name": dk.get("horse_name"),
                "dark_popularity": int(dk.get("popularity")),
                "dark_odds": float(dk.get("win_odds")),
                "dark_frame_number": int(dk.get("frame_number")) if pd.notna(dk.get("frame_number")) else None,
                "ticket": ticket,
                "stake_yen": STAKE_PER_TICKET,
                "reason": f"{RULE_NAME} [{components}] dark_rank={int(dk['dark_rank'])} (odds {float(dk['win_odds']):.1f}, pop {int(dk['popularity'])})",
                "snapshot_time": snapshot_time,
                "odds_source": odds_source,
                "generated_at": generated_at,
                "rule_name": RULE_NAME,
            })
    return rows


def main():
    import datetime as _dt
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--race-id", nargs="+", default=None,
                    help="one or more race_ids to include")
    ap.add_argument("--snapshot-time", default="final",
                    help='odds snapshot label (e.g. "final", "30min", "1h"). '
                         "currently uses DB final odds regardless; the label is "
                         "recorded for future use with pre-race snapshots.")
    ap.add_argument("--output", type=Path, default=None,
                    help="output CSV path (default: data/processed/live_tickets_<suffix>_<snapshot>.csv)")
    args = ap.parse_args()

    snapshot_time = args.snapshot_time
    generated_at = _dt.datetime.now().isoformat(timespec="seconds")

    log.info("loading data with filters: date=%s year=%s race_id=%s snapshot=%s",
             args.date, args.year, args.race_id, snapshot_time)
    races, entries, snapshot_df = load_filtered(args)
    log.info("candidate races after filters: %d  snapshot rows available: %d",
             len(races), len(snapshot_df))

    snap_by_race = (snapshot_df.groupby("race_id")
                    if not snapshot_df.empty else {})
    snap_lookup = {rid: g for rid, g in snap_by_race} if snap_by_race else {}

    all_rows: list[dict] = []
    races_with_tickets = 0
    n_used_snapshot = 0
    n_used_fallback = 0
    for _, r in races.sort_values("race_date", kind="stable").iterrows():
        race_entries = entries[entries["race_id"] == r["race_id"]]
        snap_for_race = snap_lookup.get(r["race_id"])
        merged_entries, effective_source = merge_odds_for_race(race_entries, snap_for_race)
        rows = build_tickets_for_race(r, merged_entries,
                                       snapshot_time=snapshot_time,
                                       odds_source=effective_source,
                                       generated_at=generated_at)
        if rows:
            all_rows.extend(rows)
            races_with_tickets += 1
            if effective_source == SOURCE_DB_SNAPSHOT:
                n_used_snapshot += 1
            else:
                n_used_fallback += 1
    log.info("odds_source breakdown: db_snapshot=%d races, db_final_fallback=%d races",
             n_used_snapshot, n_used_fallback)

    if not all_rows:
        log.warning("no D9-eligible races in selection; CSV not written")
        return

    df = pd.DataFrame(all_rows)
    if args.output:
        out_path = args.output
    else:
        suffix = args.date or (str(args.year) if args.year else
                                args.race_id[0] if args.race_id else "all")
        out_path = PROCESSED_DIR / f"live_tickets_{suffix}_{snapshot_time}.csv"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("wrote %s (%d tickets / %d races)", out_path, len(df), races_with_tickets)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 60)

    print()
    print(f"=== Generated {len(df)} tickets across {races_with_tickets} races ===")
    # show condensed view: 1 line per race (race_date / race_name / dark horses / tickets)
    summary = (df.groupby("race_id")
               .agg(race_date=("race_date", "first"),
                    race_name=("race_name", "first"),
                    racecourse=("racecourse", "first"),
                    grade=("grade", "first"),
                    component_labels=("component_labels", "first"),
                    n_tickets=("ticket", "count"),
                    total_stake=("stake_yen", "sum"))
               .reset_index()
               .sort_values("race_date"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

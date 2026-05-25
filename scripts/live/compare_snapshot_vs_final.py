"""generate_tickets.py が出す snapshot CSV (発走前 odds) と final CSV (確定 odds) を比較。

各レースについて以下を出力:
  - n_tickets_snapshot / n_tickets_final / common_count / *_only_count / *_only_tickets
  - p1_horse_snapshot / p1_horse_final + p1_changed
  - p1_odds_snapshot  / p1_odds_final
  - p2_horse_snapshot / p2_horse_final + p2_changed
  - p2_odds_snapshot  / p2_odds_final
  - p1_popularity_drift_for_snapshot_horse  (snapshot 時 p1 だった馬の final popularity - 1)
  - p2_popularity_drift_for_snapshot_horse
  - darks_snapshot / darks_final (順序付き ;-join)
  - darks_added / darks_removed / darks_reordered (set同じだが順位差)
  - ticket_hit_drift   (snapshot だと当たり, final だと外れ あるいは逆)
  - hit_changed_direction  (gained_in_snapshot / lost_in_snapshot / both_hit / both_miss)
  - eligibility_status (both / snapshot_only / final_only)

両 CSV が完全に同じなら、全 _changed_flag = False、added/removed/reordered 空、
hit_drift=False になる。

出力: data/processed/snapshot_vs_final_diff.csv (1 row = 1 race)

Usage:
    python scripts/live/compare_snapshot_vs_final.py \
        --snapshot data/processed/live_tickets_2025_30min.csv \
        --final    data/processed/live_tickets_2025_final.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sqlalchemy import text

from src.config import PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("live.compare")


def _read_tickets(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error("tickets CSV not found: %s", path)
        sys.exit(1)
    df = pd.read_csv(path, dtype={"race_id": str})
    return df


def _race_first_value(grp: pd.DataFrame, col: str):
    if col not in grp.columns:
        return None
    s = grp[col].dropna()
    return s.iloc[0] if not s.empty else None


def _race_summary(grp: pd.DataFrame) -> dict:
    # Preserve dark ordering: tickets are emitted in dark_rank order by generate_tickets,
    # so the unique darks list as encountered is already in odds-rank order.
    darks_ordered = []
    seen = set()
    if "dark_horse_number" in grp.columns:
        for x in grp["dark_horse_number"].dropna().astype(int).tolist():
            if x not in seen:
                seen.add(x)
                darks_ordered.append(int(x))
    return {
        "race_date": _race_first_value(grp, "race_date"),
        "race_name": _race_first_value(grp, "race_name"),
        "racecourse": _race_first_value(grp, "racecourse"),
        "grade": _race_first_value(grp, "grade"),
        "p1_horse_number": _race_first_value(grp, "p1_horse_number"),
        "p1_odds": _race_first_value(grp, "p1_odds"),
        "p2_horse_number": _race_first_value(grp, "p2_horse_number"),
        "p2_odds": _race_first_value(grp, "p2_odds"),
        "tickets": set(grp["ticket"].astype(str).tolist()) if "ticket" in grp.columns else set(),
        "darks": set(darks_ordered),
        "darks_ordered": darks_ordered,
        "snapshot_time": _race_first_value(grp, "snapshot_time"),
    }


def _load_db_lookups(race_ids: set) -> tuple[dict, dict, dict, dict]:
    """Per-race lookups from DB:
       - trifecta_combo:   race_id -> '1-5-3' (actual outcome)
       - trifecta_payout:  race_id -> int
       - final_pop_map:    race_id -> {horse_number: popularity_in_final}
       - final_label_map:  race_id -> set of labels in odds_snapshots
    """
    if not race_ids:
        return {}, {}, {}, {}
    engine = get_engine()
    placeholders = ",".join([f":r{i}" for i in range(len(race_ids))])
    params = {f"r{i}": rid for i, rid in enumerate(race_ids)}
    tri = pd.read_sql(
        text(f"SELECT race_id, combination, payout_yen FROM payouts "
             f"WHERE bet_type = '三連単' AND race_id IN ({placeholders})"),
        engine, params=params,
    ).drop_duplicates("race_id")
    combo_by = tri.set_index("race_id")["combination"].astype(str).to_dict()
    payout_by = tri.set_index("race_id")["payout_yen"].to_dict()
    entries = pd.read_sql(
        text(f"SELECT race_id, horse_number, popularity FROM entries "
             f"WHERE race_id IN ({placeholders})"),
        engine, params=params,
    )
    final_pop_map = {}
    for rid, g in entries.groupby("race_id"):
        final_pop_map[str(rid)] = dict(zip(g["horse_number"].astype(int),
                                            g["popularity"].astype("Int64")))
    return combo_by, payout_by, final_pop_map, {}


def _join_set(s: set) -> str:
    return ";".join(str(x) for x in sorted(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, required=True,
                    help="path to pre-race snapshot live_tickets_*.csv")
    ap.add_argument("--final", type=Path, required=True,
                    help="path to final-odds live_tickets_*.csv")
    ap.add_argument("--output", type=Path,
                    default=PROCESSED_DIR / "snapshot_vs_final_diff.csv")
    args = ap.parse_args()

    snap_df = _read_tickets(args.snapshot)
    final_df = _read_tickets(args.final)
    log.info("snapshot rows=%d (%s), final rows=%d (%s)",
             len(snap_df), args.snapshot.name, len(final_df), args.final.name)

    snap_by_race = {rid: _race_summary(grp) for rid, grp in snap_df.groupby("race_id", sort=False)}
    final_by_race = {rid: _race_summary(grp) for rid, grp in final_df.groupby("race_id", sort=False)}

    all_race_ids = sorted(set(snap_by_race) | set(final_by_race))
    combo_by, payout_by, final_pop_map, _ = _load_db_lookups(set(all_race_ids))

    def _gv(d, k):
        return d[k] if d else None

    def _join_list(xs):
        return ";".join(str(x) for x in xs)

    rows = []
    for rid in all_race_ids:
        s = snap_by_race.get(rid)
        f = final_by_race.get(rid)
        in_snap = s is not None
        in_final = f is not None
        if in_snap and in_final:
            eligibility = "both"
        elif in_snap:
            eligibility = "snapshot_only"
        else:
            eligibility = "final_only"

        snap_tickets = s["tickets"] if s else set()
        final_tickets = f["tickets"] if f else set()
        common = snap_tickets & final_tickets
        snap_only = snap_tickets - final_tickets
        final_only = final_tickets - snap_tickets

        snap_darks = s["darks"] if s else set()
        final_darks = f["darks"] if f else set()
        snap_darks_ord = s["darks_ordered"] if s else []
        final_darks_ord = f["darks_ordered"] if f else []
        darks_added = final_darks - snap_darks
        darks_removed = snap_darks - final_darks
        # reorder: same set, different order
        darks_reordered = (snap_darks == final_darks
                           and snap_darks_ord != final_darks_ord
                           and bool(snap_darks))

        p1_snap = _gv(s, "p1_horse_number")
        p1_final = _gv(f, "p1_horse_number")
        p2_snap = _gv(s, "p2_horse_number")
        p2_final = _gv(f, "p2_horse_number")
        p1_changed = (in_snap and in_final and p1_snap != p1_final)
        p2_changed = (in_snap and in_final and p2_snap != p2_final)

        # popularity drift: take the snapshot's p1 horse and look up its FINAL popularity
        pop_map = final_pop_map.get(rid, {})
        def _drift_for(snap_horse):
            if snap_horse is None or pd.isna(snap_horse):
                return None
            try:
                hn = int(snap_horse)
            except (TypeError, ValueError):
                return None
            final_pop = pop_map.get(hn)
            if final_pop is None or pd.isna(final_pop):
                return None
            return int(final_pop) - 1  # delta from "snapshot was 1番人気"
        p1_pop_drift = _drift_for(p1_snap) if in_snap else None
        # p2 drift: snapshot's p2 horse - its final popularity vs 2
        def _drift_p2(snap_horse):
            if snap_horse is None or pd.isna(snap_horse):
                return None
            try:
                hn = int(snap_horse)
            except (TypeError, ValueError):
                return None
            final_pop = pop_map.get(hn)
            if final_pop is None or pd.isna(final_pop):
                return None
            return int(final_pop) - 2
        p2_pop_drift = _drift_p2(p2_snap) if in_snap else None

        # ticket hit drift: did the actual 三連単 combo land differently
        # for snapshot's tickets vs final's tickets?
        true_combo = combo_by.get(rid)
        if true_combo:
            snap_hit = true_combo in snap_tickets
            final_hit = true_combo in final_tickets
        else:
            snap_hit = final_hit = False
        ticket_hit_drift = (snap_hit != final_hit) if true_combo else False
        if not true_combo:
            hit_direction = "no_result_in_db"
        elif snap_hit and final_hit:
            hit_direction = "both_hit"
        elif snap_hit and not final_hit:
            hit_direction = "gained_in_snapshot"   # snapshot 採用なら勝ち, final なら負け
        elif final_hit and not snap_hit:
            hit_direction = "lost_in_snapshot"     # snapshot 採用なら負け, final なら勝ち
        else:
            hit_direction = "both_miss"

        race_date = _gv(s, "race_date") if in_snap else _gv(f, "race_date")
        race_name = _gv(s, "race_name") if in_snap else _gv(f, "race_name")

        rows.append({
            "race_id": rid,
            "race_date": race_date,
            "race_name": race_name,
            "eligibility_status": eligibility,
            "snapshot_snapshot_time": _gv(s, "snapshot_time"),
            "final_snapshot_time": _gv(f, "snapshot_time"),
            "n_tickets_snapshot": len(snap_tickets),
            "n_tickets_final": len(final_tickets),
            "common_count": len(common),
            "snapshot_only_count": len(snap_only),
            "final_only_count": len(final_only),
            "snapshot_only_tickets": _join_set(snap_only),
            "final_only_tickets": _join_set(final_only),
            "p1_horse_snapshot": p1_snap,
            "p1_horse_final": p1_final,
            "p1_changed": p1_changed,
            "p1_pop_drift_for_snapshot_horse": p1_pop_drift,
            "p1_odds_snapshot": _gv(s, "p1_odds"),
            "p1_odds_final": _gv(f, "p1_odds"),
            "p2_horse_snapshot": p2_snap,
            "p2_horse_final": p2_final,
            "p2_changed": p2_changed,
            "p2_pop_drift_for_snapshot_horse": p2_pop_drift,
            "p2_odds_snapshot": _gv(s, "p2_odds"),
            "p2_odds_final": _gv(f, "p2_odds"),
            "darks_snapshot": _join_list(snap_darks_ord),
            "darks_final": _join_list(final_darks_ord),
            "darks_added": _join_set(darks_added),
            "darks_removed": _join_set(darks_removed),
            "darks_reordered": darks_reordered,
            "actual_trifecta_combo": true_combo or "",
            "snapshot_would_hit": snap_hit,
            "final_would_hit": final_hit,
            "ticket_hit_drift": ticket_hit_drift,
            "hit_direction": hit_direction,
        })

    out_df = pd.DataFrame(rows).sort_values(["race_date", "race_id"], kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    log.info("wrote %s (%d rows)", args.output, len(out_df))

    # ---- summary
    n_total = len(out_df)
    n_both = int((out_df["eligibility_status"] == "both").sum())
    n_snap_only = int((out_df["eligibility_status"] == "snapshot_only").sum())
    n_final_only = int((out_df["eligibility_status"] == "final_only").sum())
    n_p1_changed = int(out_df["p1_changed"].sum())
    n_p2_changed = int(out_df["p2_changed"].sum())
    n_dark_drift = int((out_df["darks_added"].str.len() + out_df["darks_removed"].str.len() > 0).sum())
    n_dark_reorder = int(out_df["darks_reordered"].sum())
    n_ticket_diff = int((out_df["snapshot_only_count"] + out_df["final_only_count"] > 0).sum())
    n_hit_drift = int(out_df["ticket_hit_drift"].sum())
    n_gained_snap = int((out_df["hit_direction"] == "gained_in_snapshot").sum())
    n_lost_snap = int((out_df["hit_direction"] == "lost_in_snapshot").sum())
    print()
    print("=== SNAPSHOT vs FINAL DIFF ===")
    print(f"  races compared            : {n_total}")
    print(f"  eligible in both          : {n_both}")
    print(f"  snapshot_only             : {n_snap_only}")
    print(f"  final_only                : {n_final_only}")
    print(f"  races with p1 changed     : {n_p1_changed}")
    print(f"  races with p2 changed     : {n_p2_changed}")
    print(f"  races with dark drift     : {n_dark_drift}")
    print(f"  races with dark reorder   : {n_dark_reorder}")
    print(f"  races with ticket diff    : {n_ticket_diff}")
    print(f"  races with hit drift      : {n_hit_drift}")
    print(f"    gained_in_snapshot      : {n_gained_snap}")
    print(f"    lost_in_snapshot        : {n_lost_snap}")
    if n_total == 0:
        return

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 36)
    print()
    cols_show = ["race_date", "race_name", "eligibility_status",
                 "n_tickets_snapshot", "n_tickets_final",
                 "common_count", "snapshot_only_count", "final_only_count",
                 "p1_changed", "p2_changed",
                 "p1_pop_drift_for_snapshot_horse", "p2_pop_drift_for_snapshot_horse",
                 "darks_added", "darks_removed", "darks_reordered",
                 "ticket_hit_drift", "hit_direction"]
    print(out_df[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()

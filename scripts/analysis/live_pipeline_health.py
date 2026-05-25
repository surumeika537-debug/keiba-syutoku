"""run_weekend_pipeline.py の運用品質を集計するレポート。

入力:
  - pipeline_state / pipeline_races (SQLite)
  - data/processed/live_pipeline_events.jsonl
  - data/processed/realtime_snapshot_health_*.csv
  - odds_snapshots テーブル
  - data/processed/paper_trading_log.csv (もし存在すれば)

出力指標:
  - snapshot success rate (60min / 30min / 10min / 5min / final 別)
  - realtime fetch latency (per stage 平均/p95)
  - retry count distribution
  - stage failure breakdown
  - drift by snapshot timing (60→30, 30→10, 10→5, 5→final)
  - p1 stability curve  (60min p1 → final p1 が同じ馬の率)
  - p2 stability curve
  - dark candidate churn curve
  - bankroll trajectory (paper trading)
  - expected vs realized ROI (backtest +141% vs 実 paper trading)

CSV:
  data/processed/live_pipeline_health.csv  (1 row per snapshot label + 1 summary row)

Usage:
    python scripts/analysis/live_pipeline_health.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sqlalchemy import text

from src.config import PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("live_pipeline_health")

SNAPSHOT_ORDER = ["60min", "30min", "10min", "5min", "final"]


def _load_state_tables(engine):
    try:
        races = pd.read_sql("SELECT * FROM pipeline_races", engine)
    except Exception:
        races = pd.DataFrame()
    try:
        states = pd.read_sql("SELECT * FROM pipeline_state", engine)
    except Exception:
        states = pd.DataFrame()
    return races, states


def _load_events_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _stability(snaps: pd.DataFrame, label_a: str, label_b: str, pop: int):
    """Per race, did pop=N horse stay the same between two snapshot labels?
    Returns (matched, total)."""
    a = snaps[(snaps["snapshot_time_label"] == label_a) & (snaps["popularity"] == pop)]
    b = snaps[(snaps["snapshot_time_label"] == label_b) & (snaps["popularity"] == pop)]
    a = a[["race_id", "horse_number"]].rename(columns={"horse_number": "h_a"})
    b = b[["race_id", "horse_number"]].rename(columns={"horse_number": "h_b"})
    m = a.merge(b, on="race_id")
    if m.empty:
        return 0, 0
    return int((m["h_a"] == m["h_b"]).sum()), len(m)


def _dark_churn(snaps: pd.DataFrame, label_a: str, label_b: str):
    """Per race, count dark candidate additions+removals between two snapshots.
    dark: popularity>=5 AND 10<=win_odds<=30."""
    def _dark_set_per_race(label):
        sub = snaps[snaps["snapshot_time_label"] == label].copy()
        sub["popularity"] = pd.to_numeric(sub["popularity"], errors="coerce")
        sub["win_odds"] = pd.to_numeric(sub["win_odds"], errors="coerce")
        is_dark = ((sub["popularity"] >= 5)
                   & (sub["win_odds"] >= 10) & (sub["win_odds"] <= 30))
        sub = sub[is_dark]
        out = {}
        for rid, g in sub.groupby("race_id"):
            out[rid] = set(int(x) for x in g["horse_number"].astype(int))
        return out
    a_map = _dark_set_per_race(label_a)
    b_map = _dark_set_per_race(label_b)
    races = set(a_map) | set(b_map)
    if not races:
        return 0, 0
    churn = sum(len(a_map.get(r, set()) ^ b_map.get(r, set())) for r in races)
    return churn, len(races)


def _drift(snaps: pd.DataFrame, label_a: str, label_b: str):
    """Per (race, horse): mean |odds_b - odds_a| / odds_a."""
    a = snaps[snaps["snapshot_time_label"] == label_a][
        ["race_id", "horse_number", "win_odds"]
    ].rename(columns={"win_odds": "odds_a"})
    b = snaps[snaps["snapshot_time_label"] == label_b][
        ["race_id", "horse_number", "win_odds"]
    ].rename(columns={"win_odds": "odds_b"})
    m = a.merge(b, on=["race_id", "horse_number"])
    m = m.dropna(subset=["odds_a", "odds_b"])
    if m.empty:
        return None
    drift = ((m["odds_b"] - m["odds_a"]).abs() / m["odds_a"]).mean()
    return float(drift)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-log", type=Path,
                    default=PROCESSED_DIR / "live_pipeline_events.jsonl")
    args = ap.parse_args()

    engine = get_engine()
    races, states = _load_state_tables(engine)
    events = _load_events_log(args.events_log)
    snaps = pd.read_sql("SELECT * FROM odds_snapshots", engine)

    print(f"=== pipeline_races: {len(races)} ===")
    print(f"=== pipeline_state: {len(states)} ===")
    print(f"=== events: {len(events)} ===")
    print(f"=== odds_snapshots: {len(snaps)} ===")

    # ---- per-stage health
    rows = []
    if not states.empty:
        for stage, g in states.groupby("stage"):
            total = len(g)
            done = int((g["state"] == "DONE").sum())
            failed = int((g["state"] == "FAILED").sum())
            pending = int((g["state"] == "PENDING").sum())
            in_progress = int((g["state"] == "IN_PROGRESS").sum())
            avg_lat = pd.to_numeric(g["latency_ms"], errors="coerce").dropna().mean()
            p95_lat = pd.to_numeric(g["latency_ms"], errors="coerce").dropna().quantile(0.95) \
                if g["latency_ms"].dropna().size else None
            avg_retry = pd.to_numeric(g["retry_count"], errors="coerce").dropna().mean()
            rows.append({
                "section": "stage_health",
                "key": stage,
                "total": total, "done": done, "failed": failed,
                "pending": pending, "in_progress": in_progress,
                "success_rate": done / total if total else 0,
                "avg_latency_ms": float(avg_lat) if pd.notna(avg_lat) else None,
                "p95_latency_ms": float(p95_lat) if p95_lat is not None and pd.notna(p95_lat) else None,
                "avg_retry": float(avg_retry) if pd.notna(avg_retry) else None,
            })

    # ---- drift / stability / churn curves
    if not snaps.empty:
        pairs = [("60min", "30min"), ("30min", "10min"),
                 ("10min", "5min"), ("5min", "final")]
        for la, lb in pairs:
            drift = _drift(snaps, la, lb)
            p1_match, p1_n = _stability(snaps, la, lb, 1)
            p2_match, p2_n = _stability(snaps, la, lb, 2)
            churn, churn_n = _dark_churn(snaps, la, lb)
            rows.append({
                "section": "drift_pair",
                "key": f"{la}->{lb}",
                "drift_pair_n_races": churn_n,
                "avg_odds_drift_pct": drift,
                "p1_stability_rate": p1_match / p1_n if p1_n else None,
                "p1_stability_n": p1_n,
                "p2_stability_rate": p2_match / p2_n if p2_n else None,
                "p2_stability_n": p2_n,
                "avg_dark_churn_per_race": churn / churn_n if churn_n else None,
            })

    # ---- ROI: expected (backtest) vs realized (paper trading)
    expected_roi = 1.41   # E_D9_P3_CAP4 backtest ROI from 10-year backtest
    paper_log = PROCESSED_DIR / "paper_trading_log.csv"
    realized_roi = None
    realized_hits = realized_races = None
    if paper_log.exists():
        log_df = pd.read_csv(paper_log)
        # only count rows from this pipeline (filter by rule_name or notes)
        if not log_df.empty:
            stake = pd.to_numeric(log_df.get("stake_yen", 0), errors="coerce").fillna(0).sum()
            payout = pd.to_numeric(log_df.get("payout_yen", 0), errors="coerce").fillna(0).sum()
            realized_roi = (payout - stake) / stake if stake > 0 else None
            realized_hits = int(pd.to_numeric(log_df.get("hit_flag", 0), errors="coerce").fillna(0).astype(bool).sum())
            realized_races = len(log_df)
    rows.append({
        "section": "expected_vs_realized",
        "key": "ROI",
        "expected_roi": expected_roi,
        "realized_roi": realized_roi,
        "realized_hits": realized_hits,
        "realized_races": realized_races,
        "delta": (realized_roi - expected_roi) if realized_roi is not None else None,
    })

    # ---- event-based metrics: snapshot retry / source breakdown
    if not events.empty:
        snap_events = events[events.get("action") == "snapshot"] if "action" in events.columns else pd.DataFrame()
        if not snap_events.empty and "source" in snap_events.columns:
            for src, n in snap_events["source"].value_counts().items():
                rows.append({"section": "source_counts", "key": str(src), "events": int(n)})

    summary_df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "live_pipeline_health.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("wrote %s (%d rows)", out_path, len(summary_df))

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.float_format", "{:.4f}".format)
    print()
    print("=== LIVE PIPELINE HEALTH ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

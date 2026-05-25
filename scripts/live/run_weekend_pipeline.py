"""自動 paper trading pipeline orchestrator (D9 + E_D9_P3_CAP4 専用)。

各 race を以下の state machine で処理する:

  DISCOVERED
    → SNAP_60   (発走 60 分前 odds + tickets 生成)
    → SNAP_30   (発走 30 分前 odds + tickets 生成)
    → SNAP_10   (発走 10 分前 odds + tickets 生成)
    → SNAP_5    (発走  5 分前 odds + tickets 生成)  ← この snapshot を実投票想定
    → RESULT    (final odds + 結果取得 + paper trading log 更新)
    → COMPLETE  (snapshot vs final 差分、drift report 更新、equity 確定)

state は SQLite の `pipeline_state` / `pipeline_races` 表に persist され、
SIGINT/再起動後も中断地点から resume。lock file で並列実行を防止。

Modes:
  --mode scheduler   長時間 daemon。次の event まで sleep。SIGINT で graceful 終了。
  --mode single-run  due な stage を全部処理して exit (cron 推奨)。
  --mode dry-run     何をやるか log だけ、network/DB 書き込みなし。

Mock テスト用:
  --race-id <ID> --mock-race-start <ISO> --immediate-mode

通常運用:
  --race-ids-file <CSV with race_id, race_start_time columns>

Health log は `data/processed/live_pipeline_events.jsonl` に逐次 append される。
集計は `scripts/analysis/live_pipeline_health.py` で。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sqlalchemy import text

from src.config import PROCESSED_DIR, JRA_RACECOURSES
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger

# inline reuse of existing modules
from scripts.ingest.fetch_odds_snapshots import (
    SOURCE_PLACEHOLDER,
    fetch_odds_for_race,
    fetch_odds_via_placeholder,
)

force_utf8_stdout()
log = setup_logger("live.pipeline")


# ============================================================================
#  Configuration
# ============================================================================
STAGE_SNAPSHOTS = ("SNAP_60", "SNAP_30", "SNAP_10", "SNAP_5")
STAGES = STAGE_SNAPSHOTS + ("RESULT", "COMPLETE")
SNAPSHOT_MINUTES_BEFORE = {"SNAP_60": 60, "SNAP_30": 30, "SNAP_10": 10, "SNAP_5": 5}
SNAPSHOT_LABEL_BY_STAGE = {"SNAP_60": "60min", "SNAP_30": "30min", "SNAP_10": "10min", "SNAP_5": "5min"}
RESULT_DELAY_MINUTES = 30   # wait this long after race start for results to settle
COMPLETE_DELAY_MINUTES = 35

DEFAULT_LOCK_FILE = PROCESSED_DIR / ".pipeline.lock"
DEFAULT_EVENTS_LOG = PROCESSED_DIR / "live_pipeline_events.jsonl"
LOCK_STALE_HOURS = 4

DEFAULT_BANKROLL = 100_000
STAKE_PER_TICKET = 100
RULE_NAME = "E_D9_P3_CAP4"


# ============================================================================
#  Lock
# ============================================================================
class PipelineLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def acquire(self, force: bool = False):
        if self.path.exists() and not force:
            try:
                data = json.loads(self.path.read_text())
                started = dt.datetime.fromisoformat(data["started_at"])
                age_h = (dt.datetime.now() - started).total_seconds() / 3600
                if age_h < LOCK_STALE_HOURS:
                    raise RuntimeError(
                        f"pipeline already running (pid={data.get('pid')}, "
                        f"started={data.get('started_at')}). "
                        f"Use --force-lock to override, or wait."
                    )
                log.warning("found stale lock (%.1fh old), overwriting", age_h)
            except (json.JSONDecodeError, KeyError):
                log.warning("found corrupt lock file, overwriting")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "pid": os.getpid(),
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown",
        }, indent=2))
        self.acquired = True
        log.info("acquired lock %s (pid=%d)", self.path, os.getpid())

    def release(self):
        if self.acquired and self.path.exists():
            try:
                self.path.unlink()
                log.info("released lock %s", self.path)
            except OSError as e:
                log.warning("failed to release lock %s: %s", self.path, e)
            self.acquired = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()


# ============================================================================
#  State tables (SQLite, additive — created on first use)
# ============================================================================
def _ensure_state_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_races (
                race_id TEXT PRIMARY KEY,
                race_start_time TEXT NOT NULL,
                rule_name TEXT,
                initial_bankroll INTEGER,
                bankroll_after REAL,
                discovered_at TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_state (
                race_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                state TEXT NOT NULL,
                scheduled_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                latency_ms INTEGER,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (race_id, stage)
            )
        """))


def _upsert_race(engine, race_id, race_start_time, initial_bankroll):
    now = dt.datetime.now().isoformat(timespec="seconds")
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO pipeline_races(race_id, race_start_time, rule_name,
                                          initial_bankroll, bankroll_after, discovered_at)
            VALUES (:rid, :rst, :rule, :bk, :bk, :now)
            ON CONFLICT(race_id) DO NOTHING
        """), {"rid": race_id, "rst": race_start_time.isoformat(),
                "rule": RULE_NAME, "bk": float(initial_bankroll), "now": now})


def _upsert_stage(engine, race_id, stage, state, **kw):
    now = dt.datetime.now().isoformat(timespec="seconds")
    params = {"rid": race_id, "stage": stage, "state": state, "now": now}
    fields = {"scheduled_at", "started_at", "completed_at",
              "latency_ms", "retry_count", "error_message"}
    cols = ["race_id", "stage", "state", "last_updated"]
    placeholders = [":rid", ":stage", ":state", ":now"]
    for f in fields:
        if f in kw and kw[f] is not None:
            v = kw[f]
            if isinstance(v, dt.datetime):
                v = v.isoformat(timespec="seconds")
            params[f] = v
            cols.append(f)
            placeholders.append(f":{f}")
    # build INSERT...ON CONFLICT DO UPDATE
    update_cols = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in ("race_id", "stage"))
    sql = (f"INSERT INTO pipeline_state ({', '.join(cols)}) "
           f"VALUES ({', '.join(placeholders)}) "
           f"ON CONFLICT(race_id, stage) DO UPDATE SET {update_cols}")
    with engine.begin() as conn:
        conn.execute(text(sql), params)


def _get_stage(engine, race_id, stage) -> dict | None:
    with engine.begin() as conn:
        r = conn.execute(text("SELECT * FROM pipeline_state WHERE race_id=:rid AND stage=:s"),
                         {"rid": race_id, "s": stage}).mappings().first()
        return dict(r) if r else None


def _list_pending(engine) -> list[dict]:
    with engine.begin() as conn:
        rs = conn.execute(text("""
            SELECT race_id, stage, state, scheduled_at, retry_count
            FROM pipeline_state
            WHERE state IN ('PENDING', 'FAILED')
            ORDER BY scheduled_at
        """)).mappings().all()
        return [dict(r) for r in rs]


# ============================================================================
#  Health event log (JSONL — one row per event)
# ============================================================================
def _log_event(events_path: Path, **kv):
    events_path.parent.mkdir(parents=True, exist_ok=True)
    kv["ts"] = dt.datetime.now().isoformat(timespec="seconds")
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kv, ensure_ascii=False, default=str) + "\n")


# ============================================================================
#  Per-stage worker functions (reuse existing modules; minimal new logic)
# ============================================================================
def _do_snapshot_stage(engine, race_id: str, snapshot_label: str,
                        dry_run: bool, debug_save: bool,
                        events_path: Path) -> dict:
    """Fetch realtime odds + insert into odds_snapshots."""
    from src.schemas import odds_snapshots as odds_tbl
    if dry_run:
        _log_event(events_path, race_id=race_id, stage=snapshot_label, action="snapshot",
                    dry_run=True)
        return {"horses": 0, "source": "dry_run"}
    rows, source, health = fetch_odds_for_race(race_id, snapshot_label, debug_save=debug_save)
    captured = None
    if health.get("official_datetime"):
        try:
            captured = dt.datetime.fromisoformat(health["official_datetime"])
        except ValueError:
            captured = dt.datetime.now()
    else:
        captured = dt.datetime.now()
    now = dt.datetime.now()
    db_rows = [{
        "race_id": race_id,
        "snapshot_time_label": snapshot_label,
        "captured_at": captured,
        "horse_number": int(r["horse_number"]),
        "popularity": r["popularity"],
        "win_odds": r["win_odds"],
        "source": source,
        "created_at": now,
    } for r in rows]
    # idempotent insert: drop existing then insert
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM odds_snapshots
            WHERE race_id = :rid AND snapshot_time_label = :label
        """), {"rid": race_id, "label": snapshot_label})
        conn.execute(odds_tbl.insert(), db_rows)
    _log_event(events_path, race_id=race_id, stage=snapshot_label, action="snapshot",
                source=source, horses=len(rows), latency_ms=health.get("json_latency_ms"),
                official_datetime=health.get("official_datetime"))
    return {"horses": len(rows), "source": source, "health": health}


def _generate_tickets_for_race(engine, race_id, snapshot_label, dry_run, events_path):
    """Run strategy E on this race with current snapshot data; write tickets CSV."""
    from scripts.backtest.simple_roi import (
        is_d9_candidate_race, is_d6_excluded_race,
        select_d9_p3_cap4_darks, d9_component_labels, _horse_by_popularity,
    )
    races = pd.read_sql(text("SELECT * FROM races WHERE race_id = :rid"),
                         engine, params={"rid": race_id}, parse_dates=["race_date"])
    if races.empty:
        log.warning("[%s] no entry in races table; skipping ticket generation", race_id)
        return {"tickets": 0, "n_darks": 0}
    race_row = races.iloc[0]
    entries = pd.read_sql(text("SELECT * FROM entries WHERE race_id = :rid"),
                          engine, params={"rid": race_id})
    entries["popularity"] = pd.to_numeric(entries["popularity"], errors="coerce").astype("Int64")
    entries["horse_number"] = pd.to_numeric(entries["horse_number"], errors="coerce").astype("Int64")
    entries["frame_number"] = pd.to_numeric(entries["frame_number"], errors="coerce").astype("Int64")
    entries["win_odds"] = pd.to_numeric(entries["win_odds"], errors="coerce")
    entries = entries.dropna(subset=["popularity", "horse_number"])

    # override with snapshot odds
    snap = pd.read_sql(text("""
        SELECT horse_number, popularity, win_odds
        FROM odds_snapshots
        WHERE race_id = :rid AND snapshot_time_label = :label
    """), engine, params={"rid": race_id, "label": snapshot_label})
    if not snap.empty:
        snap_map = snap.set_index("horse_number")[["popularity", "win_odds"]]
        snap_pop = entries["horse_number"].map(snap_map["popularity"])
        snap_odds = entries["horse_number"].map(snap_map["win_odds"])
        entries["popularity"] = snap_pop.combine_first(entries["popularity"])
        entries["popularity"] = pd.to_numeric(entries["popularity"], errors="coerce").astype("Int64")
        entries["win_odds"] = snap_odds.combine_first(entries["win_odds"])
        odds_source = "db_snapshot"
    else:
        odds_source = "db_final_fallback"

    rd = race_row["race_date"]
    race_meta = {
        "race_id": race_id,
        "race_name": race_row.get("race_name"),
        "race_date": rd,
        "racecourse": race_row.get("racecourse"),
        "grade": race_row.get("grade"),
        "surface": race_row.get("surface"),
        "distance": race_row.get("distance"),
        "month": int(pd.Timestamp(rd).month) if pd.notna(rd) else None,
        "field_size": len(entries),
    }
    if not is_d9_candidate_race(race_meta) or is_d6_excluded_race(race_meta):
        _log_event(events_path, race_id=race_id, stage=snapshot_label,
                    action="generate_tickets", tickets=0, odds_source=odds_source,
                    note="not D9-eligible")
        return {"tickets": 0, "n_darks": 0, "odds_source": odds_source}

    p1 = _horse_by_popularity(entries, 1)
    p2 = _horse_by_popularity(entries, 2)
    if p1 is None or p2 is None:
        _log_event(events_path, race_id=race_id, stage=snapshot_label,
                    action="generate_tickets", tickets=0, odds_source=odds_source,
                    note="missing p1/p2 in snapshot")
        return {"tickets": 0, "n_darks": 0, "odds_source": odds_source}
    darks = select_d9_p3_cap4_darks(entries, race_meta, max_darks=4)

    rows = []
    components = d9_component_labels(race_meta)
    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    for _, dk in darks.iterrows():
        d_num = int(dk["horse_number"])
        for first, second in [(p1, p2), (p2, p1)]:
            rows.append({
                "race_id": race_id, "race_date": rd, "race_name": race_row.get("race_name"),
                "grade": race_row.get("grade"), "racecourse": race_row.get("racecourse"),
                "snapshot_time": snapshot_label, "odds_source": odds_source,
                "generated_at": generated_at, "rule_name": RULE_NAME,
                "component_labels": components,
                "p1_horse_number": p1, "p2_horse_number": p2,
                "dark_horse_number": d_num, "dark_popularity": int(dk["popularity"]),
                "dark_odds": float(dk["win_odds"]),
                "dark_rank": int(dk["dark_rank"]),
                "ticket": f"{first}-{second}-{d_num}",
                "stake_yen": STAKE_PER_TICKET,
            })
    if dry_run:
        _log_event(events_path, race_id=race_id, stage=snapshot_label,
                    action="generate_tickets", tickets=len(rows), dry_run=True)
        return {"tickets": len(rows), "n_darks": int(len(darks)), "odds_source": odds_source}

    # write per-(race, snapshot) tickets CSV (incremental, for later record_result)
    out_dir = PROCESSED_DIR / "pipeline_tickets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tickets_{race_id}_{snapshot_label}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    _log_event(events_path, race_id=race_id, stage=snapshot_label,
                action="generate_tickets", tickets=len(rows), n_darks=int(len(darks)),
                odds_source=odds_source, file=str(out_path))
    return {"tickets": len(rows), "n_darks": int(len(darks)), "odds_source": odds_source}


def _do_result_stage(engine, race_id, dry_run, events_path):
    """Fetch final odds + record result + update equity. Uses the SNAP_5 tickets."""
    if dry_run:
        _log_event(events_path, race_id=race_id, stage="RESULT", action="record",
                    dry_run=True)
        return {}
    # 1. ensure 'final' snapshot exists (fetch if not)
    n = pd.read_sql(text("""
        SELECT COUNT(*) AS n FROM odds_snapshots
        WHERE race_id = :rid AND snapshot_time_label = 'final'
    """), engine, params={"rid": race_id})["n"].iloc[0]
    if n == 0:
        # fetch
        rows, source, health = fetch_odds_for_race(race_id, "final", debug_save=False)
        from src.schemas import odds_snapshots as odds_tbl
        now = dt.datetime.now()
        captured = dt.datetime.fromisoformat(health["official_datetime"]) \
            if health.get("official_datetime") else now
        db_rows = [{
            "race_id": race_id, "snapshot_time_label": "final",
            "captured_at": captured, "horse_number": int(r["horse_number"]),
            "popularity": r["popularity"], "win_odds": r["win_odds"],
            "source": source, "created_at": now,
        } for r in rows]
        with engine.begin() as conn:
            conn.execute(odds_tbl.insert(), db_rows)
        _log_event(events_path, race_id=race_id, stage="RESULT",
                    action="fetch_final", source=source, horses=len(rows))

    # 2. look up actual 三連単 combination from payouts
    tri = pd.read_sql(text("""
        SELECT combination, payout_yen FROM payouts
        WHERE race_id = :rid AND bet_type = '三連単'
        LIMIT 1
    """), engine, params={"rid": race_id})
    if tri.empty:
        # race not yet settled in DB — we'd need to re-fetch race result page + parse
        _log_event(events_path, race_id=race_id, stage="RESULT",
                    action="record", note="no payout in DB yet")
        return {"hit": False, "payout": 0, "note": "no_payout_in_db"}
    actual_combo = str(tri["combination"].iloc[0])
    raw_payout = int(tri["payout_yen"].iloc[0] or 0)

    # 3. load SNAP_5 tickets (this is the snapshot we would have actually bet from)
    ticket_path = PROCESSED_DIR / "pipeline_tickets" / f"tickets_{race_id}_5min.csv"
    if not ticket_path.exists():
        # fall back to closest available snapshot
        for fallback_label in ("10min", "30min", "60min", "final"):
            alt = PROCESSED_DIR / "pipeline_tickets" / f"tickets_{race_id}_{fallback_label}.csv"
            if alt.exists():
                ticket_path = alt
                break
    if not ticket_path.exists():
        _log_event(events_path, race_id=race_id, stage="RESULT",
                    action="record", note="no tickets generated")
        return {"hit": False, "payout": 0, "note": "no_tickets"}
    tickets = pd.read_csv(ticket_path)
    ticket_strs = set(tickets["ticket"].astype(str).tolist())
    hit = actual_combo in ticket_strs
    stake_total = int(tickets["stake_yen"].sum())
    payout = raw_payout if hit else 0
    profit = payout - stake_total

    # 4. update bankroll in pipeline_races
    with engine.begin() as conn:
        row = conn.execute(text("SELECT bankroll_after FROM pipeline_races WHERE race_id=:rid"),
                            {"rid": race_id}).first()
        if row:
            new_bk = float(row[0]) + profit
            conn.execute(text("UPDATE pipeline_races SET bankroll_after=:bk WHERE race_id=:rid"),
                          {"bk": new_bk, "rid": race_id})
        else:
            new_bk = None

    _log_event(events_path, race_id=race_id, stage="RESULT", action="record",
                hit=hit, actual_combo=actual_combo, payout=payout,
                stake_total=stake_total, profit=profit, bankroll_after=new_bk,
                ticket_file=str(ticket_path))
    return {"hit": hit, "payout": payout, "profit": profit, "bankroll_after": new_bk}


def _do_complete_stage(engine, race_id, dry_run, events_path):
    """Update drift report + paper trading equity CSV (append mode)."""
    if dry_run:
        _log_event(events_path, race_id=race_id, stage="COMPLETE", action="complete",
                    dry_run=True)
        return {}

    # 1. compute per-race snapshot stats (drift between SNAP_5 and final)
    pivots = {}
    for label in ("60min", "30min", "10min", "5min", "final"):
        df = pd.read_sql(text("""
            SELECT horse_number, popularity, win_odds FROM odds_snapshots
            WHERE race_id = :rid AND snapshot_time_label = :label
        """), engine, params={"rid": race_id, "label": label})
        if not df.empty:
            pivots[label] = df
    drifts = []
    if "5min" in pivots and "final" in pivots:
        m = pivots["5min"].merge(pivots["final"], on="horse_number", suffixes=("_5", "_f"))
        m["odds_drift"] = m["win_odds_f"] - m["win_odds_5"]
        m["pop_drift"] = m["popularity_f"] - m["popularity_5"]
        drifts.append({
            "race_id": race_id, "label_a": "5min", "label_b": "final",
            "n_horses": len(m),
            "avg_odds_drift_pct": float(((m["win_odds_f"] - m["win_odds_5"]) / m["win_odds_5"]).abs().mean()),
            "p1_horse_5min": _horse_at_pop(pivots["5min"], 1),
            "p1_horse_final": _horse_at_pop(pivots["final"], 1),
        })

    # 2. write paper trading equity row (append)
    paper_log = PROCESSED_DIR / "paper_trading_log.csv"
    races_meta = pd.read_sql(text("SELECT race_date, race_name FROM races WHERE race_id=:rid"),
                              engine, params={"rid": race_id})
    race_row = pd.read_sql(text("SELECT * FROM pipeline_races WHERE race_id=:rid"),
                            engine, params={"rid": race_id}).iloc[0]
    # find result info from event log (last RESULT event for this race)
    last_result = _last_event_for(events_path, race_id, "RESULT")
    new_row = {
        "race_id": race_id,
        "race_date": races_meta["race_date"].iloc[0] if not races_meta.empty else None,
        "race_name": races_meta["race_name"].iloc[0] if not races_meta.empty else None,
        "rule_name": RULE_NAME,
        "snapshot_time": "5min",
        "odds_source": "db_snapshot",
        "generated_at": "",
        "final_result_checked_at": dt.datetime.now().isoformat(timespec="seconds"),
        "stake_yen": last_result.get("stake_total") if last_result else 0,
        "hit_flag": bool(last_result.get("hit")) if last_result else False,
        "payout_yen": int(last_result.get("payout", 0)) if last_result else 0,
        "profit_yen": int(last_result.get("profit", 0)) if last_result else 0,
        "bankroll_after": float(race_row["bankroll_after"]),
        "notes": "auto from pipeline",
    }
    append_header = not paper_log.exists()
    pd.DataFrame([new_row]).to_csv(paper_log, mode="a", header=append_header,
                                     index=False, encoding="utf-8-sig")
    _log_event(events_path, race_id=race_id, stage="COMPLETE", action="complete",
                drifts=drifts, bankroll_after=float(race_row["bankroll_after"]))
    return {"drifts": drifts, "bankroll_after": float(race_row["bankroll_after"])}


def _horse_at_pop(df, pop):
    sub = df[df["popularity"] == pop]
    if sub.empty:
        return None
    return int(sub.iloc[0]["horse_number"])


def _last_event_for(events_path: Path, race_id: str, stage: str) -> dict:
    if not events_path.exists():
        return {}
    last = {}
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
                if ev.get("race_id") == race_id and ev.get("stage") == stage:
                    last = ev
            except json.JSONDecodeError:
                continue
    return last


# ============================================================================
#  Orchestrator
# ============================================================================
class WeekendPipeline:
    def __init__(self, args):
        self.args = args
        self.engine = get_engine()
        _ensure_state_tables(self.engine)
        self.interrupted = False
        self.events_path = args.events_log
        signal.signal(signal.SIGINT, self._sigint)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._sigint)

    def _sigint(self, sig, frame):
        log.warning("signal %s received — finishing current stage, then stopping", sig)
        self.interrupted = True

    # ---- discovery
    def discover(self) -> int:
        if self.args.reset_state:
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM pipeline_state"))
                conn.execute(text("DELETE FROM pipeline_races"))
            log.warning("--reset-state: cleared pipeline_state and pipeline_races")

        races: list[tuple[str, dt.datetime]] = []
        if self.args.race_id and self.args.mock_race_start:
            start = dt.datetime.fromisoformat(self.args.mock_race_start)
            for rid in self.args.race_id:
                races.append((rid, start))
        elif self.args.race_ids_file:
            df = pd.read_csv(self.args.race_ids_file, dtype={"race_id": str})
            for _, r in df.iterrows():
                races.append((str(r["race_id"]),
                              dt.datetime.fromisoformat(str(r["race_start_time"]))))
        else:
            log.warning("no races specified (provide --race-id+--mock-race-start or --race-ids-file)")

        for race_id, start_time in races:
            _upsert_race(self.engine, race_id, start_time, self.args.bankroll)
            for stage in STAGES:
                if _get_stage(self.engine, race_id, stage):
                    continue  # already exists (resume case)
                if stage in STAGE_SNAPSHOTS:
                    sched = start_time - dt.timedelta(minutes=SNAPSHOT_MINUTES_BEFORE[stage])
                elif stage == "RESULT":
                    sched = start_time + dt.timedelta(minutes=RESULT_DELAY_MINUTES)
                else:  # COMPLETE
                    sched = start_time + dt.timedelta(minutes=COMPLETE_DELAY_MINUTES)
                _upsert_stage(self.engine, race_id, stage, "PENDING", scheduled_at=sched)
        return len(races)

    # ---- execution
    def execute_one(self, race_id, stage) -> bool:
        existing = _get_stage(self.engine, race_id, stage)
        retry_count = (existing.get("retry_count", 0) if existing else 0)
        _upsert_stage(self.engine, race_id, stage, "IN_PROGRESS", started_at=dt.datetime.now())
        t0 = time.time()
        try:
            if stage in STAGE_SNAPSHOTS:
                label = SNAPSHOT_LABEL_BY_STAGE[stage]
                snap_result = _do_snapshot_stage(self.engine, race_id, label,
                                                   self.args.dry_run, self.args.debug_save,
                                                   self.events_path)
                _generate_tickets_for_race(self.engine, race_id, label,
                                             self.args.dry_run, self.events_path)
            elif stage == "RESULT":
                _do_result_stage(self.engine, race_id, self.args.dry_run, self.events_path)
            elif stage == "COMPLETE":
                _do_complete_stage(self.engine, race_id, self.args.dry_run, self.events_path)
            latency_ms = int((time.time() - t0) * 1000)
            _upsert_stage(self.engine, race_id, stage, "DONE",
                            completed_at=dt.datetime.now(), latency_ms=latency_ms,
                            retry_count=retry_count)
            log.info("[%s/%s] DONE in %dms", race_id, stage, latency_ms)
            return True
        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            log.exception("[%s/%s] FAILED: %s", race_id, stage, e)
            _upsert_stage(self.engine, race_id, stage, "FAILED",
                            latency_ms=latency_ms,
                            retry_count=retry_count + 1,
                            error_message=str(e))
            _log_event(self.events_path, race_id=race_id, stage=stage,
                        action="execute", error=str(e), retry_count=retry_count + 1)
            return False

    def run_single(self):
        n_done = 0
        while not self.interrupted:
            pending = _list_pending(self.engine)
            if not pending:
                log.info("nothing pending; done")
                break
            now = dt.datetime.now()
            due = [r for r in pending if self.args.immediate_mode
                   or (r["scheduled_at"] and
                       dt.datetime.fromisoformat(r["scheduled_at"]) <= now)]
            if not due:
                log.info("nothing due; next scheduled at %s", pending[0]["scheduled_at"])
                break
            for r in due:
                if self.interrupted:
                    break
                if self.args.max_stages and n_done >= self.args.max_stages:
                    log.info("hit --max-stages=%d, exiting (will resume on next run)",
                              self.args.max_stages)
                    return
                self.execute_one(r["race_id"], r["stage"])
                n_done += 1
        log.info("single-run done; executed %d stage(s)", n_done)

    def run_scheduler(self):
        log.info("scheduler mode: looping until all stages COMPLETE or SIGINT")
        while not self.interrupted:
            pending = _list_pending(self.engine)
            if not pending:
                log.info("no pending stages — scheduler exiting")
                break
            now = dt.datetime.now()
            next_due_at = min(dt.datetime.fromisoformat(p["scheduled_at"]) for p in pending)
            if next_due_at > now:
                wait_sec = min((next_due_at - now).total_seconds(), 60)
                log.info("sleeping %.0fs (next stage at %s)", wait_sec, next_due_at)
                end = time.time() + wait_sec
                while time.time() < end and not self.interrupted:
                    time.sleep(1)
                continue
            due = [r for r in pending if dt.datetime.fromisoformat(r["scheduled_at"]) <= now]
            for r in due:
                if self.interrupted:
                    break
                self.execute_one(r["race_id"], r["stage"])

    def run(self):
        n = self.discover()
        log.info("discovered %d races; events log = %s", n, self.events_path)
        if self.args.mode == "scheduler":
            self.run_scheduler()
        else:
            self.run_single()


# ============================================================================
#  CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["scheduler", "single-run", "dry-run"], default="single-run")
    ap.add_argument("--race-id", nargs="+", help="mock testing: race_id(s)")
    ap.add_argument("--mock-race-start", help="ISO 8601 race start time for mock testing")
    ap.add_argument("--race-ids-file", type=Path,
                    help="CSV with race_id, race_start_time columns")
    ap.add_argument("--bankroll", type=int, default=DEFAULT_BANKROLL)
    ap.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE)
    ap.add_argument("--events-log", type=Path, default=DEFAULT_EVENTS_LOG)
    ap.add_argument("--immediate-mode", action="store_true",
                    help="execute all due stages right now, ignoring scheduled_at")
    ap.add_argument("--max-stages", type=int, default=0,
                    help="for testing: exit after processing N stages (0=unlimited)")
    ap.add_argument("--reset-state", action="store_true",
                    help="wipe pipeline_state + pipeline_races first (destroys resume info)")
    ap.add_argument("--force-lock", action="store_true",
                    help="ignore an existing lock file")
    ap.add_argument("--debug-save", action="store_true",
                    help="passed through to fetch_odds_for_race")
    args = ap.parse_args()

    if args.mode == "dry-run":
        args.dry_run = True
    else:
        args.dry_run = False

    lock = PipelineLock(args.lock_file)
    try:
        lock.acquire(force=args.force_lock)
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(2)
    try:
        WeekendPipeline(args).run()
    finally:
        lock.release()


if __name__ == "__main__":
    main()

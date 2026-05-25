"""Parse all cached raw HTMLs into the SQLite DB and CSV.

Usage:
    python scripts/transform/parse_race_results.py
    python scripts/transform/parse_race_results.py --rebuild   # drop & re-insert

Safety guards:
  - if `--raw-dir` has 0 HTML files → ABORT (exit 2), no DB change
  - if parsed_races < `--min-races-threshold` (default 100) → ABORT, no DB change
  - if `--rebuild` is requested but FK-dependent table (odds_snapshots) has rows,
    we now delete dependent rows FIRST, then races/entries/payouts in the right
    order. The whole sequence is one transaction → rollback on any error.

Together these prevent the "empty rebuild wipes good data" failure mode.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sqlalchemy import delete, text
from tqdm import tqdm

from src.config import PROCESSED_DIR, RAW_RACE_RESULTS_DIR, SOURCE
from src.database import get_engine
from src.schemas import entries as entries_tbl
from src.schemas import metadata
from src.schemas import payouts as payouts_tbl
from src.schemas import races as races_tbl
from src.utils import (
    log_parse_failure,
    setup_logger,
    truncate_parse_failure_log,
)
from scripts.transform.base import BaseParser, ParsedRace

TRIFECTA_LABELS = ("三連単",)
DEFAULT_MIN_RACES_THRESHOLD = 100   # abort rebuild if parsed_races < this

log = setup_logger("transform")


def get_parser(source: str) -> BaseParser:
    if source == "netkeiba":
        from scripts.transform.netkeiba import NetkeibaParser
        return NetkeibaParser()
    raise ValueError(f"unknown source: {source!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--rebuild", action="store_true",
                    help="delete existing rows for parsed race_ids before inserting")
    ap.add_argument("--raw-dir", type=Path, default=RAW_RACE_RESULTS_DIR)
    ap.add_argument("--min-races-threshold", type=int, default=DEFAULT_MIN_RACES_THRESHOLD,
                    help=f"abort if parsed_races < this (default {DEFAULT_MIN_RACES_THRESHOLD}). "
                         "Set 0 to disable (e.g. for initial small-batch ingest).")
    ap.add_argument("--allow-empty-rebuild", action="store_true",
                    help="DANGER: allow --rebuild with zero raw HTMLs (wipes DB).")
    args = ap.parse_args()

    parser = get_parser(args.source)
    html_files = sorted(args.raw_dir.glob("*.html"))
    log.info("found %d raw HTML files in %s", len(html_files), args.raw_dir)

    # ---- SAFETY CHECK 1: empty raw dir
    if not html_files:
        if args.rebuild and not args.allow_empty_rebuild:
            log.error("ABORT: --rebuild with zero raw HTML files would wipe the DB. "
                       "If this is really what you want, pass --allow-empty-rebuild.")
            sys.exit(2)
        log.warning("no raw HTML files found; nothing to do")
        # don't even touch DB — early exit
        sys.exit(0)

    # Start with a clean failure log so this run's output is self-contained.
    truncate_parse_failure_log()

    races_rows: list[dict] = []
    entries_rows: list[dict] = []
    payouts_rows: list[dict] = []
    failed = 0
    warnings_count = 0

    for fp in tqdm(html_files, desc="parse"):
        race_id = fp.stem

        # stage 1: read file
        try:
            html = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.warning("file_read race_id=%s: %s", race_id, e)
            log_parse_failure(race_id, fp, "file_read", type(e).__name__, e)
            failed += 1
            continue

        # stage 2: parse
        try:
            parsed: ParsedRace = parser.parse(html, race_id)
        except Exception as e:
            log.warning("parser_exception race_id=%s: %s", race_id, e)
            log_parse_failure(race_id, fp, "parser_exception", type(e).__name__, e)
            failed += 1
            continue

        # stage 3: post-parse content checks (warnings — row still gets inserted)
        races_rows.append(parsed.race)
        entries_rows.extend(parsed.entries)
        payouts_rows.extend(parsed.payouts)

        if not parsed.entries:
            log_parse_failure(race_id, fp, "empty_entries", "DataMissing", "parsed 0 entries")
            warnings_count += 1
        if not parsed.payouts:
            log_parse_failure(race_id, fp, "empty_payouts", "DataMissing", "parsed 0 payouts")
            warnings_count += 1
        else:
            tri = [p for p in parsed.payouts if p.get("bet_type") in TRIFECTA_LABELS]
            if not tri:
                log_parse_failure(race_id, fp, "missing_trifecta", "DataMissing",
                                  "no 三連単 row in payouts")
                warnings_count += 1
        if not parsed.race.get("race_name"):
            log_parse_failure(race_id, fp, "missing_race_name", "DataMissing", "race_name is None")
            warnings_count += 1
        if not parsed.race.get("race_date"):
            log_parse_failure(race_id, fp, "missing_race_date", "DataMissing", "race_date is None")
            warnings_count += 1
        if not parsed.race.get("grade"):
            log_parse_failure(race_id, fp, "missing_grade", "DataMissing", "grade is None")
            warnings_count += 1

    log.info("parsed races=%d entries=%d payouts=%d (failed=%d, warnings=%d)",
             len(races_rows), len(entries_rows), len(payouts_rows), failed, warnings_count)
    if failed or warnings_count:
        log.info("see %s for details", PROCESSED_DIR / "parse_failures.log")

    # ---- SAFETY CHECK 2: too few parsed races
    threshold = args.min_races_threshold
    if threshold > 0 and len(races_rows) < threshold:
        log.error("ABORT: parsed_races=%d < min_threshold=%d. "
                   "Refusing to touch DB to avoid wiping good data. "
                   "If this is intentional (e.g. small initial batch), pass "
                   "--min-races-threshold 0.", len(races_rows), threshold)
        sys.exit(3)

    # processed CSV (snapshot for quick inspection / external tools)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(races_rows).to_csv(PROCESSED_DIR / "races.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(entries_rows).to_csv(PROCESSED_DIR / "entries.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(payouts_rows).to_csv(PROCESSED_DIR / "payouts.csv", index=False, encoding="utf-8-sig")
    log.info("wrote CSVs to %s", PROCESSED_DIR)

    # SQLite write — single transaction. On any error → automatic rollback (DB untouched).
    engine = get_engine()
    metadata.create_all(engine)
    race_ids = [r["race_id"] for r in races_rows]
    if not race_ids:
        log.info("no rows to write — DB untouched.")
        return
    try:
        with engine.begin() as conn:
            # Defer FK checks until commit so intra-transaction state doesn't
            # violate constraints (e.g. DELETE FROM races while odds_snapshots
            # still references those race_ids — fine if races are re-inserted
            # in the same transaction).
            conn.execute(text("PRAGMA defer_foreign_keys = ON"))
            if args.rebuild:
                # full wipe of the parse-managed tables only.
                # odds_snapshots is NOT wiped — its rows remain valid as long as
                # the corresponding race_id is re-inserted below. Dangling rows
                # (race_ids absent from the new parse) are pruned at the end.
                conn.execute(delete(payouts_tbl))
                conn.execute(delete(entries_tbl))
                conn.execute(delete(races_tbl))
            else:
                # incremental: only this batch
                conn.execute(delete(payouts_tbl).where(payouts_tbl.c.race_id.in_(race_ids)))
                conn.execute(delete(entries_tbl).where(entries_tbl.c.race_id.in_(race_ids)))
                conn.execute(delete(races_tbl).where(races_tbl.c.race_id.in_(race_ids)))
            if races_rows:
                conn.execute(races_tbl.insert(), races_rows)
            if entries_rows:
                conn.execute(entries_tbl.insert(), entries_rows)
            if payouts_rows:
                conn.execute(payouts_tbl.insert(), payouts_rows)
            # cleanup: prune odds_snapshots whose race_id is no longer in races
            # (only relevant for --rebuild + a smaller raw set than before)
            if args.rebuild:
                pruned = conn.execute(text(
                    "DELETE FROM odds_snapshots "
                    "WHERE race_id NOT IN (SELECT race_id FROM races)"
                )).rowcount
                if pruned:
                    log.warning("pruned %d odds_snapshots rows whose race_id "
                                  "is no longer present after rebuild", pruned)
        log.info("DB updated (races=%d entries=%d payouts=%d).",
                  len(races_rows), len(entries_rows), len(payouts_rows))
    except Exception as e:
        log.exception("DB write failed — TRANSACTION ROLLED BACK. DB unchanged.")
        sys.exit(4)


if __name__ == "__main__":
    main()

"""Orchestrate fetching G1/G2/G3 race result HTMLs for the target years.

Usage:
    python scripts/ingest/fetch_race_results.py --years 2023 2024 2025
    python scripts/ingest/fetch_race_results.py --years 2024 --race-ids-file ids.txt

Behaviour:
- Skips race_ids whose HTML is already on disk (the cache is the source of truth).
- Sleeps `FETCH_SLEEP_SECONDS` between network requests.
- The source is pluggable via env var KEIBA_SOURCE (default: netkeiba).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tqdm import tqdm

from src.config import SOURCE, TARGET_GRADES
from src.utils import setup_logger
from scripts.ingest.base import BaseFetcher

log = setup_logger("ingest")


def get_fetcher(source: str) -> BaseFetcher:
    """Plug new sources in here. Keeps the orchestrator free of source-specific imports."""
    if source == "netkeiba":
        from scripts.ingest.netkeiba import NetkeibaFetcher
        return NetkeibaFetcher()
    raise ValueError(f"unknown source: {source!r}")


def load_race_ids_from_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument(
        "--grades",
        nargs="+",
        default=list(TARGET_GRADES),
        help="grades to fetch (default: G1 G2 G3)",
    )
    ap.add_argument(
        "--race-ids-file",
        type=Path,
        default=None,
        help="optional: skip discovery and use race_ids from this file (one per line)",
    )
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many races (debug)")
    args = ap.parse_args()

    fetcher = get_fetcher(args.source)
    log.info("using source=%s, raw_dir=%s", fetcher.name, fetcher.race_html_path("X").parent)

    if args.race_ids_file:
        race_ids = load_race_ids_from_file(args.race_ids_file)
        log.info("loaded %d race_ids from %s", len(race_ids), args.race_ids_file)
    else:
        race_ids = []
        for year in args.years:
            ids = fetcher.discover_race_ids(year, tuple(args.grades))
            log.info("year=%d discovered %d race_ids", year, len(ids))
            race_ids.extend(ids)

    if args.limit:
        race_ids = race_ids[: args.limit]

    fetched = skipped = failed = 0
    for race_id in tqdm(race_ids, desc="fetch"):
        cache_path = fetcher.race_html_path(race_id)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            skipped += 1
            continue
        try:
            fetcher.fetch_race_html(race_id)
            fetched += 1
        except Exception as e:
            log.warning("fetch failed race_id=%s: %s", race_id, e)
            failed += 1

    log.info("done. fetched=%d skipped(cached)=%d failed=%d", fetched, skipped, failed)


if __name__ == "__main__":
    main()

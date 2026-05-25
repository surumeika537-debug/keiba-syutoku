"""Initialize the SQLite DB. By default ADDS only missing tables (existing data preserved).
Use --reset to drop all tables and recreate from scratch.

Usage:
    python scripts/init_db.py            # additive: add new tables, keep existing data
    python scripts/init_db.py --reset    # destructive: wipe and recreate everything
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import inspect

from src.config import DB_PATH
from src.database import get_engine
from src.schemas import metadata
from src.utils import setup_logger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true",
                    help="drop all existing tables before creating (DESTROYS data)")
    args = ap.parse_args()

    log = setup_logger()
    engine = get_engine()
    existing = set(inspect(engine).get_table_names())

    if args.reset:
        log.warning("--reset: dropping all existing tables (data will be lost)")
        metadata.drop_all(engine)
        existing = set()

    metadata.create_all(engine)
    after = set(inspect(engine).get_table_names())
    added = sorted(after - existing)
    if added:
        log.info("created tables: %s", ", ".join(added))
    else:
        log.info("no new tables to create (existing schema is up-to-date)")
    log.info("DB at %s", DB_PATH)
    log.info("Tables: %s", ", ".join(sorted(after)))


if __name__ == "__main__":
    main()

"""Project-wide configuration: paths, fetch policy, source selection."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_RACE_RESULTS_DIR = RAW_DIR / "race_results"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"

DB_PATH = DB_DIR / "keiba.sqlite"
DB_URL = f"sqlite:///{DB_PATH.as_posix()}"

# Fetch policy. Conservative defaults — be a good citizen and respect the source's ToS.
FETCH_SLEEP_SECONDS = float(os.environ.get("FETCH_SLEEP_SECONDS", "3.0"))
FETCH_TIMEOUT_SECONDS = float(os.environ.get("FETCH_TIMEOUT_SECONDS", "20.0"))
USER_AGENT = os.environ.get(
    "FETCH_USER_AGENT",
    "keiba-trifecta-analysis/0.1 (personal research; contact: you@example.com)",
)

# Which source implementation to use. Swappable.
SOURCE = os.environ.get("KEIBA_SOURCE", "netkeiba")

TARGET_GRADES = ("G1", "G2", "G3")

# JRA中央競馬の10競馬場名。NAR(地方)/海外を除くフィルタに使う。
JRA_RACECOURSES = frozenset({
    "札幌", "函館", "福島", "新潟", "東京",
    "中山", "中京", "京都", "阪神", "小倉",
})

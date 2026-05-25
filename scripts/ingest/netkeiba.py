"""netkeiba-backed fetcher.

NOTE on usage: this code is for personal research only. The user is responsible for confirming
that their use complies with the source site's Terms of Service and robots.txt. Defaults below
are intentionally conservative (multi-second sleep, cached file reuse, single-threaded).
"""
from __future__ import annotations

import re
from pathlib import Path

import requests

from src.config import (
    FETCH_TIMEOUT_SECONDS,
    RAW_RACE_RESULTS_DIR,
    USER_AGENT,
)
from src.utils import polite_sleep, setup_logger
from scripts.ingest.base import BaseFetcher

log = setup_logger("ingest.netkeiba")

RACE_URL = "https://db.netkeiba.com/race/{race_id}/"
# Search endpoint that returns the result list for a year filtered by grade(s).
SEARCH_URL = "https://db.netkeiba.com/"

GRADE_TO_PARAM = {"G1": "1", "G2": "2", "G3": "3"}


class NetkeibaFetcher(BaseFetcher):
    name = "netkeiba"

    def __init__(self, raw_dir: Path | None = None) -> None:
        self.raw_dir = raw_dir or RAW_RACE_RESULTS_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    # ---- paths -------------------------------------------------------------

    def race_html_path(self, race_id: str) -> Path:
        return self.raw_dir / f"{race_id}.html"

    # ---- discovery ---------------------------------------------------------

    def discover_race_ids(self, year: int, grades: tuple[str, ...]) -> list[str]:
        """Best-effort: hit the race_search endpoint with grade filters and paginate.

        netkeiba's search URL/structure can change. If discovery breaks, you can also
        feed an explicit list via `--race-ids-file` on the orchestrator CLI.
        """
        race_ids: list[str] = []
        grade_params = [("grade[]", GRADE_TO_PARAM[g]) for g in grades if g in GRADE_TO_PARAM]
        page = 1
        seen: set[str] = set()

        while True:
            params = [
                ("pid", "race_list"),
                ("word", ""),
                ("start_year", str(year)),
                ("start_mon", "1"),
                ("end_year", str(year)),
                ("end_mon", "12"),
                ("sort", "date"),
                ("list", "100"),
                ("page", str(page)),
                *grade_params,
            ]
            try:
                resp = self.session.get(SEARCH_URL, params=params, timeout=FETCH_TIMEOUT_SECONDS)
                resp.encoding = resp.apparent_encoding
            except requests.RequestException as e:
                log.warning("discover page=%d failed: %s", page, e)
                break

            polite_sleep()
            found = re.findall(r"/race/(\d{12})", resp.text)
            new_ids = [rid for rid in found if rid not in seen]
            if not new_ids:
                break
            seen.update(new_ids)
            race_ids.extend(new_ids)
            log.info("discover year=%d page=%d new=%d total=%d", year, page, len(new_ids), len(race_ids))
            page += 1
            if page > 30:  # hard safety stop
                break

        return race_ids

    # ---- per-race fetch ----------------------------------------------------

    def fetch_race_html(self, race_id: str) -> str:
        cache_path = self.race_html_path(race_id)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path.read_text(encoding="utf-8", errors="replace")

        url = RACE_URL.format(race_id=race_id)
        resp = self.session.get(url, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        # Prefer the meta charset declared inside the HTML — netkeiba mixes UTF-8 and EUC-JP
        # across pages and requests.apparent_encoding occasionally guesses wrong (mojibake).
        resp.encoding = _detect_encoding(resp.content) or resp.apparent_encoding
        cache_path.write_text(resp.text, encoding="utf-8")
        polite_sleep()
        return resp.text


_META_CHARSET_PAT = re.compile(rb"charset=[\"']?([\w-]+)", re.IGNORECASE)


def _detect_encoding(body: bytes) -> str | None:
    """Read <meta charset=...> from the first 2KB of the response body."""
    m = _META_CHARSET_PAT.search(body[:2048])
    return m.group(1).decode("ascii", errors="ignore") if m else None

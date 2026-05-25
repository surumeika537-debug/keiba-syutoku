"""odds_snapshots テーブルへ netkeiba から odds を取得して投入する。

Fallback chain (上から順に試行、成功した時点で確定):
  1. **netkeiba_realtime_json**
       https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=...&type=1&action=init
       JSON 直叩き。最も軽量・最速・schema安定。
       未走 race は realtime odds、過去 race は final odds を返す。
  2. **netkeiba_realtime_html**
       https://race.netkeiba.com/odds/index.html?race_id=...&type=b1
       HTML から (a) embedded JSON / (b) BS4 table 抽出。JS-driven のため後者は
       しばしば失敗。JSON API がブロックされた時の保険。
  3. **netkeiba_race_result_page**
       https://db.netkeiba.com/race/{race_id}/
       過去 race の確定結果ページ。realtime が全滅した時の最後の砦。
  4. **db_final_odds_PLACEHOLDER** (only with --use-placeholder)
       network なしのテスト用。entries table を転記。

JSON API schema (実観測):
  {"status":"result",
   "data":{"official_datetime":"YYYY-MM-DD HH:MM:SS",
           "odds":{"1":{"01":[odds, _, pop], "02":[...], ...}}},
   "update_count":"0", "reason":""}
  スクラッチ: odds="-3.0", popularity="9999"

挙動:
  - polite_sleep: request 毎に 2-4 秒ランダム
  - retry: timeout / 403 / parse fail → exponential backoff
  - block 検知: Cloudflare / CAPTCHA / "アクセスが集中" / 空 response
  - (race_id, snapshot_time_label) が既に存在する race は skip。--force で置換
  - 各 race の fetch 過程を realtime_snapshot_health_{label}.csv に記録
  - --debug-save で raw response を debug/realtime_responses/ に保存

CLI:
    python scripts/ingest/fetch_odds_snapshots.py --year 2025 --snapshot-time final
    python scripts/ingest/fetch_odds_snapshots.py --date 2026-08-02 --snapshot-time 30min
    python scripts/ingest/fetch_odds_snapshots.py --race-id 202504030411 --snapshot-time 10min --debug-save
    python scripts/ingest/fetch_odds_snapshots.py --year 2025 --snapshot-time final --use-placeholder
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import delete, text
from tqdm import tqdm

from src.config import FETCH_TIMEOUT_SECONDS, JRA_RACECOURSES, PROCESSED_DIR, USER_AGENT
from src.database import get_engine
from src.schemas import odds_snapshots as odds_snapshots_tbl
from src.utils import force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("ingest.fetch_odds_snapshots")


# ============================================================================
#  Source identifiers
# ============================================================================
SOURCE_REALTIME_JSON = "netkeiba_realtime_json"
SOURCE_REALTIME_HTML = "netkeiba_realtime_html"
SOURCE_NETKEIBA_RESULT_PAGE = "netkeiba_race_result_page"
SOURCE_PLACEHOLDER = "db_final_odds_PLACEHOLDER"

DEBUG_RESPONSE_DIR = Path("debug/realtime_responses")


# ============================================================================
#  URL templates (centralized so source-site changes only require touching these)
# ============================================================================
RACE_RESULT_URL_TMPL = "https://db.netkeiba.com/race/{race_id}/"
REALTIME_HTML_URL_TMPL = "https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b1"
# JSON odds API. type=1 = 単勝 (win) + 複勝 (place). action=init returns initial payload
# including official_datetime ("when the server captured these odds").
REALTIME_JSON_URL_TMPL = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init"


# ============================================================================
#  HTML parsing selectors / tokens (constants so we can fix easily on layout drift)
# ============================================================================
RESULT_TABLE_CLASS_CANDIDATES = ("race_table_01", "race_table_old", "RaceTable01")
HEADER_TEXT_HORSE_NUMBER = "馬番"
HEADER_TEXT_POPULARITY = "人気"
HEADER_TEXT_WIN_ODDS = "単勝"
# Tokens that should be treated as NULL when seen in odds/popularity cells
NULL_TOKENS = {"", "---", "--.-", "--", "－", "-", "...", "計算中"}
# Scratch / cancelled markers seen in the JSON API for non-runners
SCRATCH_ODDS_VALUES = {"-3.0", "-3", "-1.0", "-1", "0.0", "0"}
SCRATCH_POP_VALUES = {"9999", "-1", "0"}

# Indicators that we hit a Cloudflare challenge / CAPTCHA / rate-limit page,
# rather than the data we expected.
BLOCK_INDICATORS_BYTES = (
    b"Just a moment",
    b"cf-chl-bypass",
    b"cf-error-code",
    b"g-recaptcha",
    b"h-captcha",
    b"recaptcha",
    b"hcaptcha",
)
BLOCK_INDICATORS_TEXT = (
    "アクセスが集中",
    "しばらくお待ち",
    "メンテナンス",
    "ご迷惑をおかけ",
    "ただいま大変混雑",
)


# ============================================================================
#  Polite-fetch settings
# ============================================================================
SLEEP_MIN_SECONDS = 2.0
SLEEP_MAX_SECONDS = 4.0
MAX_RETRIES = 3                     # total attempts = MAX_RETRIES + 1
RETRY_BACKOFF_BASE_SECONDS = 2.0    # 2, 4, 8, ... seconds


# ============================================================================
#  Exceptions
# ============================================================================
class OddsFetchError(Exception):
    """Network / HTTP error during odds fetching."""


class NoOddsAvailable(OddsFetchError):
    """The page loaded but contained no parseable odds (race not yet released, JS-only, etc.)."""


class BlockDetected(OddsFetchError):
    """Bot block / captcha / Cloudflare challenge / maintenance page detected."""


# ============================================================================
#  HTTP session + retry
# ============================================================================
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT})
    return _session


def _polite_sleep():
    time.sleep(random.uniform(SLEEP_MIN_SECONDS, SLEEP_MAX_SECONDS))


_META_CHARSET_PAT = re.compile(rb"charset=[\"']?([\w-]+)", re.IGNORECASE)


def _detect_encoding(body: bytes) -> str | None:
    m = _META_CHARSET_PAT.search(body[:2048])
    return m.group(1).decode("ascii", errors="ignore") if m else None


def _http_get_with_retry(url: str) -> str:
    """GET with exponential backoff on timeout / 403 / network error."""
    last_err: str | None = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            backoff = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
            log.warning("retry %d/%d for %s after %.1fs (last error: %s)",
                        attempt, MAX_RETRIES, url, backoff, last_err)
            time.sleep(backoff)
        try:
            resp = _get_session().get(url, timeout=FETCH_TIMEOUT_SECONDS)
            if resp.status_code in (403, 429, 503):
                last_err = f"HTTP {resp.status_code}"
                continue
            resp.raise_for_status()
            resp.encoding = _detect_encoding(resp.content) or resp.apparent_encoding
            return resp.text
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    raise OddsFetchError(f"GET {url} failed after {MAX_RETRIES + 1} attempts: {last_err}")


# ============================================================================
#  Parsers
# ============================================================================
def _safe_int(s) -> int | None:
    if s is None:
        return None
    txt = str(s).strip()
    if txt in NULL_TOKENS:
        return None
    m = re.search(r"\d+", txt)
    return int(m.group(0)) if m else None


def _safe_float(s) -> float | None:
    if s is None:
        return None
    txt = str(s).replace(",", "").strip()
    if txt in NULL_TOKENS:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _find_result_table(soup: BeautifulSoup):
    for cls in RESULT_TABLE_CLASS_CANDIDATES:
        t = soup.find("table", class_=cls)
        if t:
            return t
    # heuristic fallback: largest table by row count
    tables = soup.find_all("table")
    return max(tables, key=lambda t: len(t.find_all("tr")), default=None)


def _parse_odds_from_result_page(html: str) -> list[dict]:
    """Return [{horse_number, popularity, win_odds}, ...] from a race-result page."""
    soup = BeautifulSoup(html, "lxml")
    table = _find_result_table(soup)
    if not table:
        raise NoOddsAvailable("no result table found")
    head = table.find("tr")
    if not head:
        raise NoOddsAvailable("no header row")
    headers = [c.get_text(strip=True) for c in head.find_all(["th", "td"])]

    def col_idx(name: str):
        for i, h in enumerate(headers):
            if name in h:
                return i
        return None

    c_num = col_idx(HEADER_TEXT_HORSE_NUMBER)
    c_pop = col_idx(HEADER_TEXT_POPULARITY)
    c_odds = col_idx(HEADER_TEXT_WIN_ODDS)
    if c_num is None or c_pop is None or c_odds is None:
        raise NoOddsAvailable(
            f"missing columns (馬番={c_num}, 人気={c_pop}, 単勝={c_odds}) headers={headers}"
        )

    rows: list[dict] = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all(["td", "th"])
        if len(tds) < max(c_num, c_pop, c_odds) + 1:
            continue
        hn = _safe_int(tds[c_num].get_text(" ", strip=True))
        if hn is None:
            continue
        rows.append({
            "horse_number": hn,
            "popularity": _safe_int(tds[c_pop].get_text(" ", strip=True)),
            "win_odds": _safe_float(tds[c_odds].get_text(" ", strip=True)),
        })
    if not rows:
        raise NoOddsAvailable("no horse rows parsed")
    return rows


# ----- block detection ------------------------------------------------------
def _detect_block(body: bytes, headers) -> str | None:
    """Return a block-type string if the response looks like an anti-bot page, else None."""
    sample = body[:8192]
    # binary indicators (Cloudflare etc.)
    for ind in BLOCK_INDICATORS_BYTES:
        if ind in sample:
            if b"cf-" in ind or b"Just a moment" in ind:
                return "cloudflare_challenge"
            if b"recaptcha" in ind or b"captcha" in ind:
                return "captcha"
    # text indicators (Japanese rate-limit / maintenance pages)
    try:
        text_sample = sample.decode("utf-8", errors="ignore")
        for ind in BLOCK_INDICATORS_TEXT:
            if ind in text_sample:
                return "rate_limit_or_maintenance"
    except Exception:
        pass
    return None


def _save_debug_response(race_id: str, snapshot_label: str, stage: str, body: bytes) -> None:
    """Save raw body under debug/realtime_responses/. stage ∈ {'json','html','race_result'}."""
    DEBUG_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    ext = "json" if stage == "json" else "html"
    path = DEBUG_RESPONSE_DIR / f"{race_id}_{snapshot_label}_{stage}.{ext}"
    path.write_bytes(body)


# ----- REALTIME: JSON API (PRIMARY) -----------------------------------------
def _parse_realtime_json_payload(payload: dict) -> list[dict]:
    """Parse netkeiba odds JSON.

    Observed schema (2026-05-24):
      {"status": "result",
       "data": {"official_datetime": "YYYY-MM-DD HH:MM:SS",
                "odds": {"1": {"01": [win_odds, _, popularity], ...}, "2": {...}}},
       "update_count": "0", "reason": ""}

    For scratched horses: odds == "-3.0", popularity == "9999".
    """
    if not isinstance(payload, dict):
        return []
    if payload.get("status") not in (None, "result", "success", "True", "true"):
        # explicit error from the API
        return []
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return []
    odds_block = data.get("odds", {})
    if not isinstance(odds_block, dict):
        return []
    # type=1 = 単勝 entries; key may be "1" (string) or 1 (int)
    win_dict = odds_block.get("1", odds_block.get(1))
    if not isinstance(win_dict, dict):
        return []

    rows = []
    for k, v in win_dict.items():
        try:
            hn = int(k)
        except (ValueError, TypeError):
            continue
        if not isinstance(v, list) or len(v) < 1:
            continue
        odds_raw = str(v[0]) if v[0] is not None else ""
        pop_raw = str(v[2]) if len(v) > 2 and v[2] is not None else ""
        # detect scratched horses
        if odds_raw in SCRATCH_ODDS_VALUES or pop_raw in SCRATCH_POP_VALUES:
            rows.append({"horse_number": hn, "win_odds": None, "popularity": None})
            continue
        rows.append({
            "horse_number": hn,
            "win_odds": _safe_float(odds_raw),
            "popularity": _safe_int(pop_raw),
        })
    return rows


def _fetch_via_json_api(race_id: str, health: dict, debug_save: bool) -> list[dict]:
    """Try the JSON odds API. Updates `health` in place. Returns rows or raises."""
    url = REALTIME_JSON_URL_TMPL.format(race_id=race_id)
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": REALTIME_HTML_URL_TMPL.format(race_id=race_id),
    }
    t0 = time.time()
    try:
        resp = _get_session().get(url, headers=headers, timeout=FETCH_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        health["json_status"] = f"http_error:{type(e).__name__}"
        raise OddsFetchError(f"JSON HTTP error: {e}")
    health["json_latency_ms"] = int((time.time() - t0) * 1000)
    health["json_http_status"] = resp.status_code
    body = resp.content

    if debug_save:
        _save_debug_response(race_id, health["snapshot_label"], "json", body)

    if resp.status_code in (403, 429, 503):
        health["json_status"] = f"http_{resp.status_code}"
        raise BlockDetected(f"JSON HTTP {resp.status_code}")
    if resp.status_code != 200:
        health["json_status"] = f"http_{resp.status_code}"
        raise OddsFetchError(f"JSON HTTP {resp.status_code}")

    block = _detect_block(body, resp.headers)
    if block:
        health["json_status"] = f"block:{block}"
        health["block_type"] = block
        raise BlockDetected(f"JSON page blocked: {block}")

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        # try JSONP-style wrap
        try:
            text_body = body.decode("utf-8", errors="replace")
            m = re.search(r"\{[\s\S]+\}", text_body)
            if m:
                data = json.loads(m.group(0))
            else:
                raise OddsFetchError("not JSON")
        except Exception as e:
            health["json_status"] = "not_json"
            raise OddsFetchError(f"JSON parse failed: {e}")

    # capture the server-side timestamp for captured_at
    try:
        ts = data.get("data", {}).get("official_datetime")
        if ts:
            health["official_datetime"] = ts
    except Exception:
        pass

    rows = _parse_realtime_json_payload(data)
    if not rows:
        health["json_status"] = "empty"
        raise NoOddsAvailable("JSON API returned no win-odds rows")

    health["json_status"] = "success"
    health["horses_fetched"] = len(rows)
    return rows


# ----- REALTIME: HTML PAGE (SECONDARY) --------------------------------------
EMBEDDED_JSON_PATTERNS = (
    re.compile(r"window\.__INITIAL_STATE__\s*=\s*(\{[\s\S]+?\})\s*;\s*</script>"),
    re.compile(r"window\.__NUXT__\s*=\s*(\{[\s\S]+?\})\s*;\s*</script>"),
    re.compile(r"var\s+oddsData\s*=\s*(\{[\s\S]+?\})\s*;"),
    re.compile(r"var\s+odds_data\s*=\s*(\{[\s\S]+?\})\s*;"),
)


def _try_embedded_json(html_text: str) -> list[dict]:
    for pat in EMBEDDED_JSON_PATTERNS:
        m = pat.search(html_text)
        if not m:
            continue
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        rows = _parse_realtime_json_payload(data)
        if rows:
            return rows
    return []


def _try_html_table(html_text: str) -> list[dict]:
    """Best-effort BS4 table scrape. The realtime page is JS-driven so this
    usually returns []; kept as a last-resort layer."""
    soup = BeautifulSoup(html_text, "lxml")
    for table in soup.find_all("table"):
        head = table.find("tr")
        if not head:
            continue
        headers = [c.get_text(strip=True) for c in head.find_all(["th", "td"])]
        if (any(HEADER_TEXT_HORSE_NUMBER in h for h in headers)
                and any(HEADER_TEXT_WIN_ODDS in h for h in headers)):
            try:
                return _parse_odds_from_result_page(str(table.parent))
            except NoOddsAvailable:
                continue
    return []


def _fetch_via_realtime_html(race_id: str, health: dict, debug_save: bool) -> list[dict]:
    url = REALTIME_HTML_URL_TMPL.format(race_id=race_id)
    t0 = time.time()
    try:
        resp = _get_session().get(url, timeout=FETCH_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        health["html_status"] = f"http_error:{type(e).__name__}"
        raise OddsFetchError(f"HTML HTTP error: {e}")
    health["html_latency_ms"] = int((time.time() - t0) * 1000)
    health["html_http_status"] = resp.status_code
    resp.encoding = _detect_encoding(resp.content) or resp.apparent_encoding
    body = resp.content

    if debug_save:
        _save_debug_response(race_id, health["snapshot_label"], "html", body)

    if resp.status_code in (403, 429, 503):
        health["html_status"] = f"http_{resp.status_code}"
        raise BlockDetected(f"HTML HTTP {resp.status_code}")
    if resp.status_code != 200:
        health["html_status"] = f"http_{resp.status_code}"
        raise OddsFetchError(f"HTML HTTP {resp.status_code}")

    block = _detect_block(body, resp.headers)
    if block:
        health["html_status"] = f"block:{block}"
        health["block_type"] = block
        raise BlockDetected(f"HTML page blocked: {block}")

    html_text = resp.text

    # (a) embedded JSON in <script>
    rows = _try_embedded_json(html_text)
    if rows:
        health["html_status"] = "embedded_json"
        health["horses_fetched"] = len(rows)
        return rows

    # (b) BS4 table scrape (rarely works on JS pages)
    rows = _try_html_table(html_text)
    if rows:
        health["html_status"] = "bs4_table"
        health["horses_fetched"] = len(rows)
        return rows

    health["html_status"] = "no_odds_in_html"
    raise NoOddsAvailable("realtime HTML page has no parseable odds "
                          "(JS-driven; the JSON API is the proper source)")


# ----- FALLBACK: race-result page -------------------------------------------
def _fetch_via_race_result_page(race_id: str, health: dict, debug_save: bool) -> list[dict]:
    url = RACE_RESULT_URL_TMPL.format(race_id=race_id)
    t0 = time.time()
    try:
        html = _http_get_with_retry(url)
    except OddsFetchError as e:
        health["race_result_status"] = f"http_error"
        raise
    health["race_result_latency_ms"] = int((time.time() - t0) * 1000)
    if debug_save:
        _save_debug_response(race_id, health["snapshot_label"], "race_result",
                              html.encode("utf-8", errors="replace"))
    rows = _parse_odds_from_result_page(html)
    health["race_result_status"] = "success"
    health["horses_fetched"] = len(rows)
    return rows


# ============================================================================
#  Public fetcher: 3-layer fallback chain
# ============================================================================
def fetch_odds_for_race(race_id: str, snapshot_time_label: str,
                          debug_save: bool = False) -> tuple[list[dict], str, dict]:
    """Try realtime JSON → realtime HTML → race result page. Returns (rows, source, health).

    `health` is a dict suitable for appending to a per-race health log CSV.
    Sleeps between attempts (polite_sleep) so a 3-layer fall-through costs
    roughly 6-12 seconds of wall clock + the actual response times.
    """
    health: dict = {
        "race_id": race_id,
        "snapshot_label": snapshot_time_label,
        "attempted_at": datetime.now().isoformat(timespec="seconds"),
        "json_latency_ms": None, "json_http_status": None, "json_status": None,
        "html_latency_ms": None, "html_http_status": None, "html_status": None,
        "race_result_latency_ms": None, "race_result_status": None,
        "final_source": None, "block_type": None,
        "horses_fetched": None, "official_datetime": None,
    }

    # ---- Layer 1: JSON API (primary)
    try:
        rows = _fetch_via_json_api(race_id, health, debug_save)
        _polite_sleep()
        health["final_source"] = SOURCE_REALTIME_JSON
        return rows, SOURCE_REALTIME_JSON, health
    except BlockDetected as e:
        log.debug("[%s] JSON API blocked: %s", race_id, e)
    except (NoOddsAvailable, OddsFetchError) as e:
        log.debug("[%s] JSON API failed: %s", race_id, e)
    _polite_sleep()

    # ---- Layer 2: realtime HTML (embedded JSON or BS4)
    try:
        rows = _fetch_via_realtime_html(race_id, health, debug_save)
        _polite_sleep()
        health["final_source"] = SOURCE_REALTIME_HTML
        return rows, SOURCE_REALTIME_HTML, health
    except BlockDetected as e:
        log.debug("[%s] realtime HTML blocked: %s", race_id, e)
    except (NoOddsAvailable, OddsFetchError) as e:
        log.debug("[%s] realtime HTML failed: %s", race_id, e)
    _polite_sleep()

    # ---- Layer 3: race result page (last resort, works for past races)
    try:
        rows = _fetch_via_race_result_page(race_id, health, debug_save)
        _polite_sleep()
        health["final_source"] = SOURCE_NETKEIBA_RESULT_PAGE
        return rows, SOURCE_NETKEIBA_RESULT_PAGE, health
    except (NoOddsAvailable, OddsFetchError) as e:
        health["final_source"] = None
        raise OddsFetchError(
            f"all 3 sources failed for {race_id} "
            f"(json={health['json_status']}, html={health['html_status']}, "
            f"race_result={health['race_result_status']})"
        )


def fetch_odds_via_placeholder(race_id: str, entries_for_race: pd.DataFrame) -> tuple[list[dict], str]:
    """Test-only: don't hit the network, just transcribe entries."""
    rows = []
    for _, e in entries_for_race.iterrows():
        hn = e["horse_number"]
        if pd.isna(hn):
            continue
        rows.append({
            "horse_number": int(hn),
            "popularity": int(e["popularity"]) if pd.notna(e["popularity"]) else None,
            "win_odds": float(e["win_odds"]) if pd.notna(e["win_odds"]) else None,
        })
    return rows, SOURCE_PLACEHOLDER


# ============================================================================
#  Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--race-id", nargs="+", default=None)
    ap.add_argument("--snapshot-time", required=True,
                    help='label e.g. "60min" / "30min" / "10min" / "5min" / "final"')
    ap.add_argument("--use-placeholder", action="store_true",
                    help="don't hit the network; transcribe entries as PLACEHOLDER source")
    ap.add_argument("--force", action="store_true",
                    help='replace existing rows for the same (race_id, snapshot_time_label)')
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on number of races to fetch (debug)")
    ap.add_argument("--debug-save", action="store_true",
                    help="save raw response bytes to debug/realtime_responses/")
    args = ap.parse_args()

    engine = get_engine()
    races = pd.read_sql("SELECT * FROM races", engine, parse_dates=["race_date"])
    entries = pd.read_sql("SELECT * FROM entries", engine)

    # JRA flat G1/G2/G3 universe (matches generate_tickets filter)
    races = races[
        races["grade"].isin({"G1", "G2", "G3"})
        & races["racecourse"].isin(JRA_RACECOURSES)
        & (races["surface"] != "障害")
    ].copy()
    mask = pd.Series(False, index=races.index)
    if args.date:
        mask |= races["race_date"] == pd.Timestamp(args.date)
    if args.year:
        mask |= races["race_date"].dt.year == int(args.year)
    if args.race_id:
        mask |= races["race_id"].astype(str).isin(args.race_id)
    if not (args.date or args.year or args.race_id):
        mask = pd.Series(True, index=races.index)
    races = races[mask].copy()

    target_ids = set(races["race_id"].astype(str))
    entries = entries[entries["race_id"].astype(str).isin(target_ids)].copy()
    entries["horse_number"] = pd.to_numeric(entries["horse_number"], errors="coerce").astype("Int64")
    entries["popularity"] = pd.to_numeric(entries["popularity"], errors="coerce").astype("Int64")
    entries["win_odds"] = pd.to_numeric(entries["win_odds"], errors="coerce")
    entries = entries.dropna(subset=["horse_number"])

    existing = pd.read_sql(
        text("SELECT DISTINCT race_id FROM odds_snapshots WHERE snapshot_time_label = :label"),
        engine, params={"label": args.snapshot_time},
    )
    existing_set = set(existing["race_id"].astype(str))
    to_replace = target_ids & existing_set if args.force else set()
    to_fetch = sorted((target_ids - existing_set) | to_replace)
    if args.limit:
        to_fetch = to_fetch[: args.limit]

    log.info("target races=%d  existing=%d  to_fetch=%d  (force=%s, placeholder=%s)",
             len(target_ids), len(target_ids & existing_set),
             len(to_fetch), args.force, args.use_placeholder)

    if not to_fetch:
        log.info("nothing to fetch")
        return

    if to_replace:
        with engine.begin() as conn:
            for rid in to_replace:
                conn.execute(delete(odds_snapshots_tbl)
                             .where(odds_snapshots_tbl.c.race_id == rid)
                             .where(odds_snapshots_tbl.c.snapshot_time_label == args.snapshot_time))
        log.info("removed existing rows for %d races (force)", len(to_replace))

    now = datetime.now()
    inserted = parse_failed = http_failed = 0
    source_counts: dict[str, int] = {}
    rows_buffer: list[dict] = []
    failures: list[dict] = []
    health_log: list[dict] = []

    for race_id in tqdm(to_fetch, desc="fetch_odds"):
        race_entries = entries[entries["race_id"] == race_id]
        try:
            if args.use_placeholder:
                rows, src = fetch_odds_via_placeholder(race_id, race_entries)
                captured = None
                health_log.append({
                    "race_id": race_id, "snapshot_label": args.snapshot_time,
                    "attempted_at": datetime.now().isoformat(timespec="seconds"),
                    "final_source": SOURCE_PLACEHOLDER,
                    "horses_fetched": len(rows),
                })
            else:
                rows, src, health = fetch_odds_for_race(race_id, args.snapshot_time,
                                                         debug_save=args.debug_save)
                health_log.append(health)
                # prefer server-side official_datetime when available
                if health.get("official_datetime"):
                    try:
                        captured = datetime.fromisoformat(health["official_datetime"])
                    except ValueError:
                        captured = datetime.now()
                else:
                    captured = datetime.now()
        except OddsFetchError as e:
            log.warning("ALL FALLBACKS failed race_id=%s: %s", race_id, e)
            http_failed += 1
            failures.append({"race_id": race_id, "stage": "all_layers", "error": str(e)})
            # still log the failed attempt for health analysis
            health_log.append({
                "race_id": race_id, "snapshot_label": args.snapshot_time,
                "attempted_at": datetime.now().isoformat(timespec="seconds"),
                "final_source": None, "error": str(e),
            })
            continue
        except Exception as e:
            log.warning("parse failed race_id=%s: %s", race_id, e)
            parse_failed += 1
            failures.append({"race_id": race_id, "stage": "parse", "error": str(e)})
            continue

        source_counts[src] = source_counts.get(src, 0) + 1
        for r in rows:
            rows_buffer.append({
                "race_id": race_id,
                "snapshot_time_label": args.snapshot_time,
                "captured_at": captured,
                "horse_number": int(r["horse_number"]),
                "popularity": r["popularity"],
                "win_odds": r["win_odds"],
                "source": src,
                "created_at": now,
            })
        inserted += 1

        # batch-commit every 500 rows to bound risk of full failure on long runs
        if len(rows_buffer) >= 500:
            with engine.begin() as conn:
                conn.execute(odds_snapshots_tbl.insert(), rows_buffer)
            rows_buffer.clear()

    if rows_buffer:
        with engine.begin() as conn:
            conn.execute(odds_snapshots_tbl.insert(), rows_buffer)

    log.info("done. races_inserted=%d parse_failed=%d http_failed=%d",
             inserted, parse_failed, http_failed)
    log.info("source breakdown: %s", source_counts)

    # always write the per-race health log
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if health_log:
        health_path = PROCESSED_DIR / f"realtime_snapshot_health_{args.snapshot_time}.csv"
        pd.DataFrame(health_log).to_csv(health_path, index=False, encoding="utf-8-sig")
        log.info("wrote health log to %s (%d rows)", health_path, len(health_log))

    if failures:
        pd.DataFrame(failures).to_csv(
            PROCESSED_DIR / f"odds_snapshot_failures_{args.snapshot_time}.csv",
            index=False, encoding="utf-8-sig",
        )
        log.info("wrote failures to data/processed/odds_snapshot_failures_%s.csv",
                 args.snapshot_time)


if __name__ == "__main__":
    main()

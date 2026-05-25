"""Small utilities shared across scripts."""
from __future__ import annotations

import logging
import random
import re
import sys
import time
from pathlib import Path

from src.config import FETCH_SLEEP_SECONDS


def setup_logger(name: str = "keiba", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def polite_sleep(base: float | None = None, jitter: float = 0.5) -> None:
    """Sleep base seconds + uniform(0, jitter). Always call between requests."""
    base = FETCH_SLEEP_SECONDS if base is None else base
    time.sleep(base + random.uniform(0, jitter))


def normalize_trifecta_combination(raw: str) -> str:
    """Convert various dash/space representations into 'h1-h2-h3' with integer numbers."""
    nums = re.findall(r"\d+", raw or "")
    return "-".join(str(int(n)) for n in nums)


def parse_int_safe(value) -> int | None:
    try:
        return int(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def parse_float_safe(value) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


# ---- domain-specific normalizers ----------------------------------------

# Source HTML often shortens 中止/取消/除外/失格 to a single character.
# The first character is enough to disambiguate.
_FINISH_STATUS_BY_FIRST_CHAR = {
    "中": "中止",
    "取": "取消",
    "除": "除外",
    "失": "失格",
}


def normalize_finish_position(raw) -> tuple[int | None, str | None]:
    """Split a result-table cell into (integer position, status).

    - "1", "12" → (1, '完走'), (12, '完走')
    - "1(降)" or trailing junk → (1, '完走')  (leading digits win)
    - "中" / "中止" → (None, '中止')
    - "取消" → (None, '取消')
    - empty / None / unknown char → (None, None)
    """
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s:
        return None, None
    # leading digits first
    import re as _re
    m = _re.match(r"^(\d+)", s)
    if m:
        return int(m.group(1)), "完走"
    return None, _FINISH_STATUS_BY_FIRST_CHAR.get(s[0])


# Canonical bet-type names + known aliases.
BET_TYPE_CANONICAL_SET = (
    "単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単",
)
_BET_TYPE_ALIASES = {
    "単勝": "単勝",
    "複勝": "複勝",
    "枠連": "枠連",
    "馬連": "馬連",
    "ワイド": "ワイド",
    "拡連複": "ワイド",
    "馬単": "馬単",
    "三連複": "三連複",
    "3連複": "三連複",
    "３連複": "三連複",
    "三連単": "三連単",
    "3連単": "三連単",
    "３連単": "三連単",
}


def normalize_bet_type(raw) -> str | None:
    """Map source labels to the canonical bet-type strings. Returns input unchanged
    if not recognized (so callers can spot oddities downstream)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return _BET_TYPE_ALIASES.get(s, s)


# ---- parse-failure log (JSONL) -----------------------------------------

import json as _json
from datetime import datetime as _datetime

_PARSE_FAILURE_LOG_PATH = None


def _failure_log_path():
    global _PARSE_FAILURE_LOG_PATH
    if _PARSE_FAILURE_LOG_PATH is None:
        from src.config import PROCESSED_DIR
        _PARSE_FAILURE_LOG_PATH = PROCESSED_DIR / "parse_failures.log"
    return _PARSE_FAILURE_LOG_PATH


def truncate_parse_failure_log() -> None:
    """Clear the failure log. Call once at the start of a parse run."""
    p = _failure_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")


def log_parse_failure(race_id, file_path, stage: str, error_type: str, error_message) -> None:
    """Append a JSONL record describing one parse-time issue. Cheap, append-only.

    Stages used: file_read, parser_exception, empty_entries, empty_payouts,
    missing_race_name, missing_race_date, missing_grade, missing_trifecta.
    """
    p = _failure_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": _datetime.now().isoformat(timespec="seconds"),
        "race_id": str(race_id),
        "file_path": str(file_path),
        "stage": stage,
        "error_type": error_type,
        "error_message": str(error_message)[:500],
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(record, ensure_ascii=False) + "\n")


def ensure_project_on_path() -> None:
    """Allow scripts to `from src.xxx import ...` when invoked as plain scripts."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def force_utf8_stdout() -> None:
    """Force UTF-8 stdout/stderr. On Windows the default console encoding (cp932) can't print
    many of the chars we use (三連単 / em-dash / etc.). Idempotent."""
    import io
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
                continue
            except Exception:
                pass
        if hasattr(stream, "buffer"):
            setattr(sys, stream_name, io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace"))

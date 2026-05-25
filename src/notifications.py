"""Telegram notification helper.

Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID / ENABLE_TELEGRAM from .env
(via python-dotenv) or from process env vars.

設計ルール (project spec):
  - token はログに **絶対** 出さない
  - 送信失敗で pipeline を止めない (返り値 False を返すだけ)
  - ENABLE_TELEGRAM=true の時だけ実送信
  - dedupe 対応 (data/processed/telegram_sent.json で重複防止)

Public API:
  send_telegram_message(text)                       — 単発送信 (dedupe 無し)
  send_telegram_once(text, race_id=..., snapshot_time=...,
                     strategy=..., notification_type=...)
                                                    — dedupe key で 1 回だけ
  already_sent(...)                                 — dedupe key の存在確認
  mark_sent(...)                                    — 送信フラグ手動マーク
  _get_settings()                                   — (enabled, token, chat_id)
                                                      (token は呼び出し元でも log 禁止)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except Exception:  # pragma: no cover
    JST = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEDUP_FILE = PROCESSED_DIR / "telegram_sent.json"
DEDUP_RETENTION_DAYS = 30

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_MAX_LEN = 4096      # Telegram API hard limit
DEFAULT_TIMEOUT_SEC = 10

log = logging.getLogger(__name__)


# ============================================================================
#  .env loader (idempotent)
# ============================================================================
def _load_dotenv_once() -> None:
    """Load .env from PROJECT_ROOT into os.environ. No-op if python-dotenv missing
    or .env not present. ``override=False`` so explicit env vars win.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


_load_dotenv_once()


def _now_jst() -> datetime:
    return datetime.now(JST) if JST else datetime.now().astimezone()


# ============================================================================
#  Settings (re-read every call so updated .env is picked up between runs)
# ============================================================================
def _get_settings() -> tuple[bool, str | None, str | None]:
    """Return (enabled, token, chat_id). Token may be None / chat_id may be None."""
    raw_enabled = os.environ.get("ENABLE_TELEGRAM", "false").strip().lower()
    enabled = raw_enabled in ("true", "1", "yes", "on")
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip() or None
    chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip() or None
    return enabled, token, chat_id


def _redact(text: str, token: str | None) -> str:
    """Make sure we never echo the token back into a log line."""
    if token and len(token) >= 8 and token in text:
        return text.replace(token, "<TELEGRAM_TOKEN_REDACTED>")
    return text


# ============================================================================
#  Send
# ============================================================================
def send_telegram_message(text: str) -> bool:
    """Send a Telegram message. Returns True iff actually delivered.

    Never raises. Returns False on:
      - ENABLE_TELEGRAM != 'true'
      - token / chat_id missing
      - requests not installed
      - network error
      - non-200 / API rejection

    Token は失敗ログにも出さない。
    """
    enabled, token, chat_id = _get_settings()
    if not enabled:
        log.debug("telegram: skipped (ENABLE_TELEGRAM is not true)")
        return False
    if not token or not chat_id:
        log.debug("telegram: skipped (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set)")
        return False
    if len(text) > TELEGRAM_MAX_LEN:
        text = text[:TELEGRAM_MAX_LEN - 30] + "\n…(truncated)"
    try:
        import requests
    except ImportError:
        log.warning("telegram: requests not installed")
        return False
    url = TELEGRAM_API_URL.format(token=token)
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": text,
                  "disable_web_page_preview": True},
            timeout=DEFAULT_TIMEOUT_SEC,
        )
    except requests.RequestException as e:
        log.warning("telegram: network error: %s", _redact(str(e), token))
        return False
    if r.status_code != 200:
        body = _redact((r.text or "")[:500], token)
        log.warning("telegram: HTTP %d: %s", r.status_code, body)
        return False
    try:
        data = r.json()
    except ValueError:
        log.warning("telegram: response not JSON")
        return False
    if not data.get("ok"):
        log.warning("telegram: API rejected: %s", _redact(str(data), token))
        return False
    return True


# ============================================================================
#  Dedup (data/processed/telegram_sent.json)
# ============================================================================
def _dedup_key(race_id: str, snapshot_time: str, strategy: str,
               notification_type: str) -> str:
    return f"{race_id}|{snapshot_time}|{strategy}|{notification_type}"


def _load_dedup() -> dict[str, str]:
    if not DEDUP_FILE.exists():
        return {}
    try:
        raw = json.loads(DEDUP_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_dedup(data: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DEDUP_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _prune_old(data: dict) -> dict:
    """Drop entries older than DEDUP_RETENTION_DAYS (== 30)."""
    cutoff = _now_jst() - timedelta(days=DEDUP_RETENTION_DAYS)
    out: dict[str, str] = {}
    for k, ts in data.items():
        if not isinstance(ts, str):
            continue
        try:
            dt = datetime.fromisoformat(ts)
            if JST and dt.tzinfo is None:
                dt = dt.replace(tzinfo=JST)
            if dt >= cutoff:
                out[k] = ts
        except (TypeError, ValueError):
            continue
    return out


def already_sent(race_id: str, snapshot_time: str, strategy: str,
                 notification_type: str) -> bool:
    """True if this key is in telegram_sent.json (regardless of when)."""
    return _dedup_key(race_id, snapshot_time, strategy, notification_type) in _load_dedup()


def mark_sent(race_id: str, snapshot_time: str, strategy: str,
              notification_type: str) -> None:
    data = _prune_old(_load_dedup())
    data[_dedup_key(race_id, snapshot_time, strategy, notification_type)] = (
        _now_jst().isoformat(timespec="seconds")
    )
    _save_dedup(data)


def send_telegram_once(text: str, *, race_id: str, snapshot_time: str,
                       strategy: str, notification_type: str) -> bool:
    """Send only if (race_id, snapshot_time, strategy, notification_type) not seen.

    マークは「実際に送信成功した時」のみ — 失敗時は再送可能性を残す。
    """
    if already_sent(race_id, snapshot_time, strategy, notification_type):
        log.debug("telegram: skipped (already sent: %s/%s/%s/%s)",
                   race_id, snapshot_time, strategy, notification_type)
        return False
    sent = send_telegram_message(text)
    if sent:
        mark_sent(race_id, snapshot_time, strategy, notification_type)
    return sent

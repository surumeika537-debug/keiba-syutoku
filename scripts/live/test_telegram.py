"""Manual Telegram send test.

Usage:
    python scripts/live/test_telegram.py
    python scripts/live/test_telegram.py --message "hello from VPS"

Exit code:
    0 = delivered
    1 = delivery failed (network / API rejection)
    2 = misconfigured (.env not set / ENABLE_TELEGRAM=false)

Token は画面に出さない (長さだけ表示)。
このスクリプトは dedupe を通さないので何度でも再実行可能。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.notifications import _get_settings, send_telegram_message  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--message", "-m", default="hello from keiba-syutoku")
    args = ap.parse_args()

    enabled, token, chat_id = _get_settings()

    print("=== Telegram test ===")
    print(f"  ENABLE_TELEGRAM    : {'true' if enabled else 'false (or unset)'}")
    print(f"  TELEGRAM_BOT_TOKEN : "
          f"{'set (len=%d, hidden)' % len(token) if token else 'NOT SET'}")
    print(f"  TELEGRAM_CHAT_ID   : {chat_id if chat_id else 'NOT SET'}")
    print()

    if not enabled:
        print("[FAIL] ENABLE_TELEGRAM is not 'true'.")
        print("  Edit .env to set ENABLE_TELEGRAM=true and try again.")
        return 2
    if not token:
        print("[FAIL] TELEGRAM_BOT_TOKEN is not set in .env.")
        return 2
    if not chat_id:
        print("[FAIL] TELEGRAM_CHAT_ID is not set in .env.")
        return 2

    print(f"sending: {args.message!r}")
    ok = send_telegram_message(args.message)
    if ok:
        print(f"[OK] delivered")
        return 0
    print("[FAIL] delivery failed (see WARNING lines above for cause)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

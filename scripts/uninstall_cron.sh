#!/usr/bin/env bash
# uninstall_cron.sh — remove the auto_paper_trading cron entry by marker.

set -euo pipefail

MARKER="# KeibaPaperTrading"
CURRENT="$(crontab -l 2>/dev/null || true)"

if ! echo "${CURRENT}" | grep -F -q "${MARKER}"; then
    echo "not installed (no '${MARKER}' marker in crontab)"
    exit 0
fi

# rebuild crontab without the marker line(s)
echo "${CURRENT}" | grep -F -v "${MARKER}" | crontab -
echo "==> removed cron entries with marker '${MARKER}'"
echo
echo "remaining crontab:"
crontab -l 2>/dev/null || echo "(empty)"

#!/usr/bin/env bash
# install_cron.sh — register the auto_paper_trading cron entry.
#
# 毎週 土日, 9:00〜18:59, 5 分おきに run_paper_trading_once.sh を kick。
# entry には "# KeibaPaperTrading" マーカーを付ける (uninstall 時に検出する)。
#
# IMPORTANT (timezone):
#   このスクリプトは system の cron daemon (= system timezone) を使う。
#   日本時間で動かしたいなら:
#     sudo timedatectl set-timezone Asia/Tokyo
#   で OS の TZ を JST にしておくこと。
#
# Usage:
#   bash scripts/install_cron.sh
#   crontab -l   # 確認

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${PROJECT_ROOT}/scripts/run_paper_trading_once.sh"

if [ ! -x "${RUNNER}" ]; then
    chmod +x "${RUNNER}" || {
        echo "ERROR: cannot make ${RUNNER} executable" >&2
        exit 1
    }
fi

MARKER="# KeibaPaperTrading"
CRON_SCHEDULE="*/5 9-18 * * 6,0"
CRON_LINE="${CRON_SCHEDULE} ${RUNNER} ${MARKER}"

# 既存 crontab 取得 (1件もなければ空)
CURRENT="$(crontab -l 2>/dev/null || true)"

if echo "${CURRENT}" | grep -F -q "${MARKER}"; then
    echo "==> already installed:"
    echo "${CURRENT}" | grep -F "${MARKER}"
    echo
    echo "(skip; uninstall first with scripts/uninstall_cron.sh)"
    exit 0
fi

# append + install
{
    if [ -n "${CURRENT}" ]; then echo "${CURRENT}"; fi
    echo "${CRON_LINE}"
} | crontab -

echo "==> installed:"
crontab -l | grep -F "${MARKER}"
echo
echo "  schedule : every 5 min, 09:00-18:59, Saturdays and Sundays"
echo "  runner   : ${RUNNER}"
echo
echo "WARNING: cron uses the SYSTEM timezone, not the script's TZ env."
echo "  current system TZ:"
echo "    $(timedatectl 2>/dev/null | grep -i 'time zone' || date +%Z)"
echo
echo "  → if not 'Asia/Tokyo', run:  sudo timedatectl set-timezone Asia/Tokyo"

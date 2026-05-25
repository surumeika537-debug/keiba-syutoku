#!/usr/bin/env bash
# deploy_check.sh — VPS 初期デプロイ後の総合 health check。
#
# 13 項目を順番に検査し、全部 PASS なら exit 0、1 つでも FAIL なら exit 1。
# 「本番運用に入って良い状態か」の最終ゲート。
#
# Usage:
#   bash scripts/deploy_check.sh
#   bash scripts/deploy_check.sh --quick      # dry-run / status を skip (高速)
#   bash scripts/deploy_check.sh --no-color
#
# 各項目は失敗しても続行する (= 全体像を把握するため)。
# Exit code:
#   0 = all PASS
#   1 = 1 つ以上 FAIL
#   2 = 1 つ以上 WARN (PASS だが要確認)
#
# 13 checks:
#   1. python3 が python3.10+ である
#   2. 仮想環境 (.venv または venv) が存在する
#   3. requirements が import 可能 (sqlalchemy, pandas, requests, tqdm, bs4)
#   4. data/db/keiba.sqlite が存在する
#   5. races テーブルに rows がある (>= MIN_RACES)
#   6. entries / payouts / odds_snapshots テーブルにも rows がある
#   7. system timezone が Asia/Tokyo
#   8. logs/ dir が書き込み可能
#   9. data/backups/ dir が書き込み可能
#  10. data/processed/ dir が書き込み可能
#  11. cron または systemd timer が install 済み
#  12. lock file が無い (or stale)
#  13. auto_paper_trading.py --dry-run / --status が exit 0

set -uo pipefail   # NOTE: -e は外す。1 個失敗しても続行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

QUICK=0
USE_COLOR=1
for arg in "$@"; do
    case "${arg}" in
        --quick) QUICK=1 ;;
        --no-color) USE_COLOR=0 ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -40
            exit 0
            ;;
    esac
done

if [ "${USE_COLOR}" -eq 1 ] && [ -t 1 ]; then
    C_PASS=$'\033[32m'; C_FAIL=$'\033[31m'; C_WARN=$'\033[33m'
    C_HEAD=$'\033[36;1m'; C_OFF=$'\033[0m'
else
    C_PASS=""; C_FAIL=""; C_WARN=""; C_HEAD=""; C_OFF=""
fi

MIN_RACES=100
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
FAILED_ITEMS=()
WARNED_ITEMS=()

pass() { printf "  %sPASS%s  %s\n" "${C_PASS}" "${C_OFF}" "$1"; PASS_COUNT=$((PASS_COUNT+1)); }
fail() { printf "  %sFAIL%s  %s\n" "${C_FAIL}" "${C_OFF}" "$1"; FAIL_COUNT=$((FAIL_COUNT+1)); FAILED_ITEMS+=("$1"); }
warn() { printf "  %sWARN%s  %s\n" "${C_WARN}" "${C_OFF}" "$1"; WARN_COUNT=$((WARN_COUNT+1)); WARNED_ITEMS+=("$1"); }
head() { printf "\n%s== %s ==%s\n" "${C_HEAD}" "$1" "${C_OFF}"; }

# venv activate (.venv 優先)
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi
PY="${PYTHON_BIN:-python}"

echo "========================================================"
echo "  keiba-syutoku VPS deploy check"
echo "  project root : ${PROJECT_ROOT}"
echo "  date         : $(date -Iseconds 2>/dev/null || date)"
echo "  hostname     : $(hostname 2>/dev/null || echo unknown)"
echo "========================================================"

# -----------------------------------------------------------
head "1. Python version (>= 3.10)"
PY_VER="$("${PY}" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "0.0")"
PY_MAJ="${PY_VER%%.*}"
PY_MIN="${PY_VER##*.}"
if [ "${PY_MAJ}" -ge 3 ] && [ "${PY_MIN}" -ge 10 ] 2>/dev/null; then
    pass "python ${PY_VER}"
else
    fail "python ${PY_VER} (need >= 3.10)"
fi

# -----------------------------------------------------------
head "2. Virtual environment"
if [ -d .venv ]; then
    pass "found .venv/"
elif [ -d venv ]; then
    pass "found venv/"
else
    fail "no .venv/ or venv/ directory (run: python3.11 -m venv .venv)"
fi

# -----------------------------------------------------------
head "3. Required packages importable"
MISSING=()
for pkg in sqlalchemy pandas requests tqdm bs4 dotenv; do
    if ! "${PY}" -c "import ${pkg}" >/dev/null 2>&1; then
        MISSING+=("${pkg}")
    fi
done
if [ ${#MISSING[@]} -eq 0 ]; then
    pass "sqlalchemy / pandas / requests / tqdm / bs4 / dotenv all importable"
else
    fail "missing packages: ${MISSING[*]} (run: pip install -r requirements.txt)"
fi

# -----------------------------------------------------------
head "4. DB file exists"
DB_FILE="data/db/keiba.sqlite"
if [ -f "${DB_FILE}" ]; then
    DB_SIZE_MB="$(du -m "${DB_FILE}" 2>/dev/null | awk '{print $1}')"
    pass "${DB_FILE} (${DB_SIZE_MB} MiB)"
else
    fail "${DB_FILE} not found (run: python scripts/init_db.py)"
fi

# -----------------------------------------------------------
head "5. races table populated (>= ${MIN_RACES})"
if [ -f "${DB_FILE}" ] && command -v sqlite3 >/dev/null 2>&1; then
    N_RACES="$(sqlite3 "${DB_FILE}" 'SELECT COUNT(*) FROM races;' 2>/dev/null || echo 0)"
    if [ "${N_RACES}" -ge "${MIN_RACES}" ] 2>/dev/null; then
        pass "races = ${N_RACES}"
    else
        fail "races = ${N_RACES} (< ${MIN_RACES})"
    fi
else
    warn "skip (sqlite3 CLI not found or DB missing)"
fi

# -----------------------------------------------------------
head "6. entries / payouts / odds_snapshots populated"
if [ -f "${DB_FILE}" ] && command -v sqlite3 >/dev/null 2>&1; then
    N_ENT="$(sqlite3 "${DB_FILE}" 'SELECT COUNT(*) FROM entries;' 2>/dev/null || echo 0)"
    N_PAY="$(sqlite3 "${DB_FILE}" 'SELECT COUNT(*) FROM payouts;' 2>/dev/null || echo 0)"
    N_ODD="$(sqlite3 "${DB_FILE}" 'SELECT COUNT(*) FROM odds_snapshots;' 2>/dev/null || echo 0)"
    if [ "${N_ENT}" -gt 0 ] && [ "${N_PAY}" -gt 0 ]; then
        pass "entries=${N_ENT}, payouts=${N_PAY}, odds_snapshots=${N_ODD}"
    else
        fail "entries=${N_ENT}, payouts=${N_PAY}, odds_snapshots=${N_ODD} (entries/payouts must be > 0)"
    fi
else
    warn "skip (sqlite3 CLI not found or DB missing)"
fi

# -----------------------------------------------------------
head "7. System timezone = Asia/Tokyo"
TZ_NOW="$(timedatectl 2>/dev/null | grep -i 'time zone' | awk '{print $3}' || echo unknown)"
if [ "${TZ_NOW}" = "Asia/Tokyo" ]; then
    pass "timedatectl reports Asia/Tokyo"
elif [ "${TZ_NOW}" = "unknown" ]; then
    # fallback: check `date`
    DATE_TZ="$(date +%Z)"
    if [ "${DATE_TZ}" = "JST" ]; then
        warn "timedatectl not available, but date says JST"
    else
        fail "timezone is ${DATE_TZ} (need JST). Run: sudo timedatectl set-timezone Asia/Tokyo"
    fi
else
    fail "timezone is ${TZ_NOW} (need Asia/Tokyo). Run: sudo timedatectl set-timezone Asia/Tokyo"
fi

# -----------------------------------------------------------
head "8. logs/ writable"
mkdir -p logs 2>/dev/null
if [ -w logs ]; then
    pass "logs/ writable"
else
    fail "logs/ not writable"
fi

# -----------------------------------------------------------
head "9. data/backups/ writable"
mkdir -p data/backups 2>/dev/null
if [ -w data/backups ]; then
    pass "data/backups/ writable"
else
    fail "data/backups/ not writable"
fi

# -----------------------------------------------------------
head "10. data/processed/ writable"
mkdir -p data/processed 2>/dev/null
if [ -w data/processed ]; then
    pass "data/processed/ writable"
else
    fail "data/processed/ not writable"
fi

# -----------------------------------------------------------
head "11. cron or systemd timer installed"
CRON_OK=0
TIMER_OK=0
if crontab -l 2>/dev/null | grep -F -q '# KeibaPaperTrading'; then
    CRON_OK=1
fi
if systemctl --user list-timers 2>/dev/null | grep -F -q 'keiba-paper.timer'; then
    TIMER_OK=1
fi
if [ "${CRON_OK}" -eq 1 ] && [ "${TIMER_OK}" -eq 1 ]; then
    warn "BOTH cron and systemd timer installed — disable one to avoid double-firing"
elif [ "${CRON_OK}" -eq 1 ]; then
    pass "cron entry installed"
elif [ "${TIMER_OK}" -eq 1 ]; then
    pass "systemd timer installed"
else
    fail "neither cron nor systemd timer installed (run install_cron.sh or install_systemd_timer.sh)"
fi

# -----------------------------------------------------------
head "12. lock file state"
LOCK="data/processed/.paper_trading.lock"
if [ ! -f "${LOCK}" ]; then
    pass "no lock present"
else
    # lock 中身を読んで stale 判定 (4h 経過なら stale)
    ACQ="$("${PY}" -c "import json;print(json.load(open('${LOCK}')).get('acquired_at',''))" 2>/dev/null || echo "")"
    if [ -z "${ACQ}" ]; then
        warn "lock file exists but unparseable (consider --force-unlock)"
    else
        AGE_SEC="$("${PY}" -c "
import sys
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    JST=ZoneInfo('Asia/Tokyo')
except Exception:
    JST=None
a=datetime.fromisoformat('${ACQ}')
n=datetime.now(JST) if JST else datetime.now().astimezone()
print(int((n-a).total_seconds()))
" 2>/dev/null || echo "0")"
        if [ "${AGE_SEC}" -ge 14400 ] 2>/dev/null; then
            warn "lock file age=${AGE_SEC}s (stale, will auto-release on next run)"
        else
            warn "lock file age=${AGE_SEC}s (another run may be in progress)"
        fi
    fi
fi

# -----------------------------------------------------------
head "13. auto_paper_trading.py --dry-run / --status"
if [ "${QUICK}" -eq 1 ]; then
    warn "skipped (--quick)"
else
    if "${PY}" scripts/live/auto_paper_trading.py --dry-run >/dev/null 2>&1; then
        if "${PY}" scripts/live/auto_paper_trading.py --status >/dev/null 2>&1; then
            pass "--dry-run / --status both exit 0"
        else
            fail "--status exit != 0"
        fi
    else
        fail "--dry-run exit != 0"
    fi
fi

# -----------------------------------------------------------
echo
echo "========================================================"
printf "  PASS=%d  WARN=%d  FAIL=%d\n" "${PASS_COUNT}" "${WARN_COUNT}" "${FAIL_COUNT}"
if [ "${FAIL_COUNT}" -gt 0 ]; then
    echo
    echo "  ${C_FAIL}FAILED:${C_OFF}"
    for it in "${FAILED_ITEMS[@]}"; do echo "    - ${it}"; done
fi
if [ "${WARN_COUNT}" -gt 0 ]; then
    echo
    echo "  ${C_WARN}WARNINGS:${C_OFF}"
    for it in "${WARNED_ITEMS[@]}"; do echo "    - ${it}"; done
fi
echo "========================================================"

if [ "${FAIL_COUNT}" -gt 0 ]; then
    echo "==> NOT READY for production. Fix the FAIL items above."
    exit 1
elif [ "${WARN_COUNT}" -gt 0 ]; then
    echo "==> READY but review WARN items."
    exit 2
else
    echo "==> READY for production."
    exit 0
fi

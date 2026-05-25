#!/usr/bin/env bash
# run_paper_trading_once.sh
#
# cron / systemd timer から kick される single-run wrapper。
# - project root に chdir
# - venv が有れば activate
# - python scripts/live/auto_paper_trading.py を実行
# - stdout → logs/paper_trading_YYYYMMDD.log
# - stderr → logs/errors_YYYYMMDD.log
#
# Usage:
#   bash scripts/run_paper_trading_once.sh
#
# Exit code: auto_paper_trading.py の exit code に従う。

set -euo pipefail

# project root = この script の親ディレクトリ
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# venv activate (.venv 優先、なければ venv)
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# python interpreter (venv が active なら自動でその python)
PYTHON_BIN="${PYTHON_BIN:-python}"

# logs dir
mkdir -p logs
TODAY="$(date +%Y%m%d)"
LOG_OUT="logs/paper_trading_${TODAY}.log"
LOG_ERR="logs/errors_${TODAY}.log"

# run (stdout/stderr 分離)
{
    echo "===== run start $(date -Iseconds) ====="
} >> "${LOG_OUT}"

# 環境変数 TZ も念のため JST に
export TZ="${TZ:-Asia/Tokyo}"

set +e
"${PYTHON_BIN}" scripts/live/auto_paper_trading.py --mode single-run \
    >> "${LOG_OUT}" 2>> "${LOG_ERR}"
RC=$?
set -e

{
    echo "===== run end   $(date -Iseconds)  exit=${RC} ====="
    echo
} >> "${LOG_OUT}"

exit "${RC}"

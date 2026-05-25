#!/usr/bin/env bash
# first_run_vps.sh — VPS 初回 setup の one-shot wrapper.
#
# 想定状況:
#   - git clone は終わっている
#   - python3.11 + venv module は install 済み
#   - .venv は作成済み + requirements も install 済み
#   - data/db/keiba.sqlite は配置済み (またはこれから fetch する)
#
# やること:
#   1. shell script (sh/timer/service) を chmod +x
#   2. logs/ data/backups/ data/processed/ dir を作成
#   3. .gitignore に含まれる artefact が誤って tracked されてないか warn
#   4. auto_paper_trading.py --dry-run (DB 不変)
#   5. auto_paper_trading.py --status
#   6. deploy_check.sh
#
# 各 step は失敗しても続行し、最後にサマリを出す。
# 全 step 成功なら exit 0、1 つでも失敗なら exit 1。
#
# Usage:
#   bash scripts/first_run_vps.sh
#   bash scripts/first_run_vps.sh --skip-deploy-check

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SKIP_DEPLOY=0
for arg in "$@"; do
    case "${arg}" in
        --skip-deploy-check) SKIP_DEPLOY=1 ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -30
            exit 0
            ;;
    esac
done

if [ -t 1 ]; then
    C_OK=$'\033[32m'; C_NG=$'\033[31m'; C_H=$'\033[36;1m'; C_OFF=$'\033[0m'
else
    C_OK=""; C_NG=""; C_H=""; C_OFF=""
fi

step() { printf "\n%s==> %s%s\n" "${C_H}" "$1" "${C_OFF}"; }
ok()   { printf "    %sOK%s   %s\n" "${C_OK}" "${C_OFF}" "$1"; }
ng()   { printf "    %sFAIL%s %s\n" "${C_NG}" "${C_OFF}" "$1"; }

OK_COUNT=0
NG_COUNT=0
FAILED=()

mark_ok() { OK_COUNT=$((OK_COUNT+1)); ok "$1"; }
mark_ng() { NG_COUNT=$((NG_COUNT+1)); FAILED+=("$1"); ng "$1"; }

# venv activate
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    echo "WARN: no virtualenv found (.venv / venv). Using system python." >&2
fi
PY="${PYTHON_BIN:-python}"

echo "============================================================"
echo "  keiba-syutoku  VPS first-run setup"
echo "  project root : ${PROJECT_ROOT}"
echo "  date         : $(date -Iseconds 2>/dev/null || date)"
echo "============================================================"

# -----------------------------------------------------------
step "1. shell scripts に chmod +x"
SCRIPTS_TO_CHMOD=(
    "scripts/run_paper_trading_once.sh"
    "scripts/install_cron.sh"
    "scripts/uninstall_cron.sh"
    "scripts/install_systemd_timer.sh"
    "scripts/deploy_check.sh"
    "scripts/first_run_vps.sh"
)
ALL_OK=1
for s in "${SCRIPTS_TO_CHMOD[@]}"; do
    if [ -f "${s}" ]; then
        chmod +x "${s}" 2>/dev/null || ALL_OK=0
    fi
done
if [ "${ALL_OK}" -eq 1 ]; then
    mark_ok "chmod +x all wrapper scripts"
else
    mark_ng "chmod +x failed on one or more scripts"
fi

# -----------------------------------------------------------
step "2. dir 作成 (logs / data/backups / data/processed)"
ALL_OK=1
for d in logs data/backups data/processed data/db data/raw/race_results; do
    mkdir -p "${d}" || ALL_OK=0
done
if [ "${ALL_OK}" -eq 1 ]; then
    mark_ok "logs/ data/backups/ data/processed/ data/db/ data/raw/race_results/ ensured"
else
    mark_ng "mkdir failed on one or more dirs"
fi

# -----------------------------------------------------------
step "3. .gitignore で wipe された artefact が tracked になってないか確認"
if command -v git >/dev/null 2>&1 && [ -d .git ]; then
    LEAKED="$(git ls-files | grep -E '^(data/backups/|data/processed/.paper_trading.lock|data/processed/pipeline_state.json|logs/|\.env$)' 2>/dev/null || true)"
    if [ -z "${LEAKED}" ]; then
        mark_ok "no operational artefact in git index"
    else
        mark_ng "tracked artefact in git index (should be in .gitignore): $(echo "${LEAKED}" | tr '\n' ' ')"
    fi
else
    mark_ok "skip (not a git repo)"
fi

# -----------------------------------------------------------
step "4. auto_paper_trading.py --dry-run (DB 不変)"
if "${PY}" scripts/live/auto_paper_trading.py --dry-run; then
    mark_ok "dry-run finished"
else
    mark_ng "dry-run exited non-zero"
fi

# -----------------------------------------------------------
step "5. auto_paper_trading.py --status"
if "${PY}" scripts/live/auto_paper_trading.py --status; then
    mark_ok "status finished"
else
    mark_ng "status exited non-zero"
fi

# -----------------------------------------------------------
step "6. deploy_check.sh (13 checks)"
if [ "${SKIP_DEPLOY}" -eq 1 ]; then
    mark_ok "skipped (--skip-deploy-check)"
else
    if bash scripts/deploy_check.sh; then
        mark_ok "deploy_check passed"
    else
        DC_RC=$?
        if [ "${DC_RC}" -eq 2 ]; then
            mark_ok "deploy_check passed with warnings (review above)"
        else
            mark_ng "deploy_check FAILED (rc=${DC_RC})"
        fi
    fi
fi

# -----------------------------------------------------------
echo
echo "============================================================"
printf "  steps OK=%d   FAIL=%d\n" "${OK_COUNT}" "${NG_COUNT}"
if [ "${NG_COUNT}" -gt 0 ]; then
    echo
    echo "  ${C_NG}FAILED steps:${C_OFF}"
    for it in "${FAILED[@]}"; do echo "    - ${it}"; done
fi
echo "============================================================"

if [ "${NG_COUNT}" -gt 0 ]; then
    cat <<EOF

==> first-run setup INCOMPLETE. Fix failures above, then run:
    bash scripts/first_run_vps.sh

==> common next steps if 4./5./6. failed:
    - DB が空 →  python scripts/ingest/fetch_race_results.py --years 2024 2025
                  python scripts/transform/parse_race_results.py --rebuild
    - venv 不在 → python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
    - tz が JST でない → sudo timedatectl set-timezone Asia/Tokyo

EOF
    exit 1
fi

cat <<EOF

==> first-run setup COMPLETE.

次にやること:
  # systemd timer をインストール (推奨)
  bash scripts/install_systemd_timer.sh

  # または cron 方式
  bash scripts/install_cron.sh

  # 状態確認
  python scripts/live/auto_paper_trading.py --status
  systemctl --user list-timers      # systemd の場合
  crontab -l                         # cron の場合

  # ヘッドレス VPS なら user unit を login なしで動かす:
  sudo loginctl enable-linger "\$(whoami)"

EOF
exit 0

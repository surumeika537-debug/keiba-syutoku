#!/usr/bin/env bash
# install_systemd_timer.sh — install user-level systemd units for paper trading.
#
# Why systemd over cron:
#   - survives VPS reboots cleanly (Persistent=true catches up missed runs)
#   - journalctl integration
#   - explicit timezone handling
#
# Installs to:
#   ~/.config/systemd/user/keiba-paper.service
#   ~/.config/systemd/user/keiba-paper.timer
#
# Requirements:
#   - systemd-based Linux
#   - `loginctl enable-linger <user>` recommended for headless VPS so user units
#     run without an active login session.
#
# Usage:
#   bash scripts/install_systemd_timer.sh                # 既存があれば確認 prompt
#   bash scripts/install_systemd_timer.sh --force        # 既存を上書き (確認なし)
#   bash scripts/install_systemd_timer.sh --uninstall    # 削除のみ
#
# Post-install:
#   systemctl --user list-timers
#   systemctl --user status keiba-paper.timer
#   journalctl --user -u keiba-paper.service -n 50 --no-pager
#   journalctl --user -u keiba-paper.service -f          # tail
#   systemctl --user start keiba-paper.service           # 手動 1 回 kick

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${PROJECT_ROOT}/deploy"
TARGET_DIR="${HOME}/.config/systemd/user"

FORCE=0
UNINSTALL=0
for arg in "$@"; do
    case "${arg}" in
        --force) FORCE=1 ;;
        --uninstall) UNINSTALL=1 ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -30
            exit 0
            ;;
    esac
done

if ! command -v systemctl >/dev/null 2>&1; then
    echo "ERROR: systemctl not found (this script is Linux/systemd only)" >&2
    exit 1
fi

SERVICE_DST="${TARGET_DIR}/keiba-paper.service"
TIMER_DST="${TARGET_DIR}/keiba-paper.timer"

# ---------- uninstall path ----------
if [ "${UNINSTALL}" -eq 1 ]; then
    echo "==> uninstalling keiba-paper.timer / keiba-paper.service"
    systemctl --user disable --now keiba-paper.timer 2>/dev/null || true
    rm -f "${SERVICE_DST}" "${TIMER_DST}"
    systemctl --user daemon-reload
    echo "==> done."
    exit 0
fi

# ---------- install path ----------
if [ ! -d "${DEPLOY_DIR}" ]; then
    echo "ERROR: ${DEPLOY_DIR} not found" >&2
    exit 1
fi

SERVICE_SRC="${DEPLOY_DIR}/keiba-paper.service"
TIMER_SRC="${DEPLOY_DIR}/keiba-paper.timer"
if [ ! -f "${SERVICE_SRC}" ] || [ ! -f "${TIMER_SRC}" ]; then
    echo "ERROR: deploy/keiba-paper.{service,timer} not found" >&2
    exit 1
fi

mkdir -p "${TARGET_DIR}"

# 既存 check → 上書き確認
if [ -f "${SERVICE_DST}" ] || [ -f "${TIMER_DST}" ]; then
    echo "==> existing unit files detected:"
    [ -f "${SERVICE_DST}" ] && echo "      ${SERVICE_DST}"
    [ -f "${TIMER_DST}"   ] && echo "      ${TIMER_DST}"
    if [ "${FORCE}" -ne 1 ]; then
        if [ -t 0 ]; then
            printf "    overwrite? [y/N] "
            read -r REPLY
            case "${REPLY}" in
                [yY]|[yY][eE][sS]) ;;
                *) echo "==> aborted (use --force to overwrite without prompt)"; exit 0 ;;
            esac
        else
            echo "==> non-interactive shell; refusing to overwrite (use --force)"
            exit 1
        fi
    fi
    echo "==> overwriting..."
    # 旧 timer を一旦止める (新しい unit で start し直す)
    systemctl --user stop keiba-paper.timer 2>/dev/null || true
fi

# substitute "%h/keiba-syutoku" → actual PROJECT_ROOT
sed "s|%h/keiba-syutoku|${PROJECT_ROOT}|g" "${SERVICE_SRC}" > "${SERVICE_DST}"
cp "${TIMER_SRC}" "${TIMER_DST}"

chmod +x "${PROJECT_ROOT}/scripts/run_paper_trading_once.sh" 2>/dev/null || true

echo "==> wrote units:"
echo "      ${SERVICE_DST}"
echo "      ${TIMER_DST}"

# daemon-reload はファイル書き換え後に必須
systemctl --user daemon-reload
systemctl --user enable --now keiba-paper.timer

# linger 推奨 warn
if ! loginctl show-user "$(whoami)" 2>/dev/null | grep -q 'Linger=yes'; then
    echo
    echo "WARN: linger is OFF for $(whoami)."
    echo "      Headless VPS では login session が無いと user unit が止まる。"
    echo "      → sudo loginctl enable-linger \"\$(whoami)\""
fi

echo
echo "==> installed and enabled."
echo
echo "==> current status:"
systemctl --user status keiba-paper.timer --no-pager || true
echo
echo "==> next scheduled runs:"
systemctl --user list-timers --no-pager | head -5

cat <<EOF

================================================================
post-install checklist:

  # 1. timer がスケジュール上に乗っているか
  systemctl --user list-timers | grep keiba-paper

  # 2. 直近の起動結果 (= service の最終 run)
  systemctl --user status keiba-paper.service --no-pager

  # 3. journalctl で履歴を見る
  journalctl --user -u keiba-paper.service -n 50 --no-pager
  journalctl --user -u keiba-paper.service -f         # tail mode

  # 4. 手動で 1 回 kick (テスト)
  systemctl --user start keiba-paper.service
  tail -n 20 logs/paper_trading_\$(date +%Y%m%d).log

  # 5. ヘッドレス VPS で持続させる (login session 不要にする)
  sudo loginctl enable-linger "\$(whoami)"

  # 6. timezone 確認 (Asia/Tokyo であること)
  timedatectl

  # uninstall:
  bash scripts/install_systemd_timer.sh --uninstall
================================================================
EOF

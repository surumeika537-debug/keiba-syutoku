"""VPS 常駐運用向け single-run orchestrator (Linux + cron / systemd 前提)。

このスクリプトは **daemon ではない**。1 回起動するごとに:
  1. lock 取得 (二重起動防止 / stale 自動解除)
  2. state 読込 → 何を実行すべきか判断
  3. DB backup (日次)
  4. fetch (12h 経過していれば)
  5. parse (毎回 — 安価で idempotent)
  6. generate_tickets (今日の race を対象)
  7. record_result (今日の tickets CSV を実 payout と照合)
  8. state 保存 + lock 解放 + exit
を実行する。

cron で 5 分おきに kick されることを想定。実行時間は通常数秒〜数分。

CLI:
  python scripts/live/auto_paper_trading.py                 # 通常実行 (mode=single-run)
  python scripts/live/auto_paper_trading.py --dry-run       # 何をするか log 出力のみ、DB 不変
  python scripts/live/auto_paper_trading.py --status        # 状態確認 (lock/state/log/cron)
  python scripts/live/auto_paper_trading.py --force-unlock  # stale lock を強制解除
  python scripts/live/auto_paper_trading.py --backup-db     # backup のみ強制実行

design notes:
  - JST 厳守 (ZoneInfo("Asia/Tokyo"))
  - state は data/processed/pipeline_state.json
  - lock は data/processed/.paper_trading.lock (JSON, 4h 経過で stale)
  - backup は data/backups/keiba_YYYYMMDD.sqlite (1日1回, 既存なら skip)
  - 既存 strategy E_D9_P3_CAP4 / fetch / parse / generate_tickets / record_result
    は subprocess 経由で呼ぶ (= 既存ロジック変更なし)
  - Windows でも動く (cross-platform path; cron/systemd 部分は Linux 専用)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except Exception:  # pragma: no cover
    JST = None  # falls back to local time

# allow `from src.notifications import ...` (this file is scripts/live/...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from src.notifications import send_telegram_once
    _telegram_available = True
except Exception:  # pragma: no cover — notifications never break the pipeline
    _telegram_available = False

# ============================================================================
#  Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BACKUPS_DIR = PROJECT_ROOT / "data" / "backups"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_FILE = PROJECT_ROOT / "data" / "db" / "keiba.sqlite"
STATE_FILE = PROCESSED_DIR / "pipeline_state.json"
LOCK_FILE = PROCESSED_DIR / ".paper_trading.lock"

LOCK_STALE_HOURS = 4
FETCH_COOLDOWN_HOURS = 12   # 直近 fetch から N 時間以内なら skip


# ============================================================================
#  Utility: tz-aware now
# ============================================================================
def now_jst() -> datetime:
    if JST is not None:
        return datetime.now(JST)
    return datetime.now().astimezone()


def now_iso() -> str:
    return now_jst().isoformat(timespec="seconds")


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None and JST is not None:
            dt = dt.replace(tzinfo=JST)
        return dt
    except (TypeError, ValueError):
        return None


def hours_since(iso: str | None) -> float | None:
    dt = parse_iso(iso)
    if dt is None:
        return None
    return (now_jst() - dt).total_seconds() / 3600


# ============================================================================
#  Logger (writes to stdout; cron wrapper captures to file)
# ============================================================================
def log(level: str, msg: str, *args) -> None:
    """Lightweight logger that writes to stdout/stderr only (cron wrapper redirects)."""
    line = f"[{now_iso()}] {level} {msg % args if args else msg}"
    if level in ("ERROR", "FATAL"):
        print(line, file=sys.stderr, flush=True)
    else:
        print(line, flush=True)


# ============================================================================
#  Telegram notify (never raises; dedupe by today+kind)
# ============================================================================
STRATEGY_LABEL = "E_D9_P3_CAP4"   # constant for this pipeline; informational tag in dedup key


def _notify(text: str, kind: str, *, snapshot_time: str = "pipeline",
            strategy: str = STRATEGY_LABEL) -> None:
    """Send a pipeline-level Telegram notification (deduped by date+kind).

    Failures are swallowed — notification must never break the pipeline.
    Token is never logged (handled inside src.notifications).
    """
    if not _telegram_available:
        return
    try:
        today = now_jst().strftime("%Y-%m-%d")
        send_telegram_once(
            text,
            race_id=today,
            snapshot_time=snapshot_time,
            strategy=strategy,
            notification_type=kind,
        )
    except Exception as e:
        log("WARN", "telegram notify failed: %s", e)


def _notify_error(label: str, detail: str = "") -> None:
    """Error notification with hash-suffixed dedup key so distinct errors get separate alerts,
    but the *same* error in the same day is only reported once.
    """
    body = f"⚠️ keiba pipeline FAILED: {label}"
    if detail:
        body += f"\n  detail: {detail[:300]}"
    body += f"\n  see logs/errors_{now_jst():%Y%m%d}.log"
    err_hash = hashlib.sha1((label + "|" + detail).encode("utf-8")).hexdigest()[:8]
    _notify(body, kind=f"error_{err_hash}")


# ============================================================================
#  State (JSON)
# ============================================================================
DEFAULT_STATE: dict = {
    "last_fetch":     None,
    "last_parse":     None,
    "last_generate":  None,
    "last_record":    None,
    "last_backup":    None,
    "last_success":   None,
    "last_error":     None,
    "current_lock":   False,
    "runs_total":     0,
    "runs_successful": 0,
    "runs_failed":    0,
    "history":        [],   # last N run summaries
}
HISTORY_MAX = 50


def load_state() -> dict:
    if not STATE_FILE.exists():
        return DEFAULT_STATE.copy()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        # merge with defaults to handle schema additions
        merged = DEFAULT_STATE.copy()
        merged.update(data)
        return merged
    except json.JSONDecodeError:
        log("WARN", "state file corrupt — resetting")
        return DEFAULT_STATE.copy()


def save_state(state: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    state["history"] = state.get("history", [])[-HISTORY_MAX:]
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")


# ============================================================================
#  Lock
# ============================================================================
class LockError(Exception):
    pass


class PaperTradingLock:
    """JSON lock file with PID + acquired_at + host. 4h stale auto-release."""

    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def _stale(self) -> bool:
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            dt = parse_iso(data.get("acquired_at"))
            if dt is None:
                return True
            return (now_jst() - dt).total_seconds() / 3600 > LOCK_STALE_HOURS
        except Exception:
            return True

    def acquire(self, force: bool = False) -> None:
        if self.path.exists():
            if force or self._stale():
                reason = "force" if force else "stale"
                log("WARN", "removing existing lock (%s): %s", reason, self.path)
                try:
                    self.path.unlink()
                except OSError as e:
                    raise LockError(f"could not remove stale lock: {e}")
            else:
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
                raise LockError(f"lock held by pid={data.get('pid')} "
                                f"since {data.get('acquired_at')} "
                                f"(< {LOCK_STALE_HOURS}h old; use --force-unlock to override)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        host = os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "unknown"
        self.path.write_text(json.dumps({
            "pid": os.getpid(),
            "acquired_at": now_iso(),
            "host": host,
        }, indent=2), encoding="utf-8")
        self.acquired = True
        log("INFO", "lock acquired (pid=%d, host=%s)", os.getpid(), host)

    def release(self) -> None:
        if self.acquired and self.path.exists():
            try:
                self.path.unlink()
                log("INFO", "lock released")
            except OSError as e:
                log("WARN", "lock release failed: %s", e)
            self.acquired = False


# ============================================================================
#  Step runners (subprocess wrappers around existing scripts)
# ============================================================================
def run_step(name: str, argv: list[str], dry_run: bool,
             retries: int = 1) -> tuple[bool, str]:
    """Execute a python step via the same interpreter. Returns (success, message)."""
    full = [sys.executable, *argv]
    if dry_run:
        log("INFO", "[DRY-RUN] would run: %s", " ".join(full))
        return True, "dry-run"
    for attempt in range(retries):
        if attempt > 0:
            log("WARN", "%s: retry %d/%d", name, attempt, retries - 1)
        try:
            r = subprocess.run(
                full, cwd=str(PROJECT_ROOT),
                capture_output=True, text=True,
                timeout=1800,  # 30 min hard cap per step
            )
        except subprocess.TimeoutExpired:
            log("ERROR", "%s: TIMEOUT after 30 min", name)
            return False, "timeout"
        except Exception as e:
            log("ERROR", "%s: subprocess error: %s", name, e)
            return False, f"exception: {e}"
        if r.returncode == 0:
            log("INFO", "%s OK (stdout %d lines)",
                  name, len(r.stdout.splitlines()))
            return True, "ok"
        log("ERROR", "%s exit %d. stderr tail:\n%s",
              name, r.returncode, "\n".join(r.stderr.splitlines()[-10:]))
    return False, f"exit {r.returncode}"


# ============================================================================
#  Decision logic — what to do this invocation
# ============================================================================
@dataclass
class RunPlan:
    do_backup:   bool = False
    do_fetch:    bool = False
    do_parse:    bool = False
    do_generate: bool = True   # always (idempotent, cheap)
    do_record:   bool = True   # always (idempotent, cheap)
    reasons:     list[str] = field(default_factory=list)


def decide_plan(state: dict, force_backup: bool = False) -> RunPlan:
    plan = RunPlan()
    # backup: once per day
    last_backup_iso = state.get("last_backup")
    last_backup_dt = parse_iso(last_backup_iso)
    if force_backup:
        plan.do_backup = True
        plan.reasons.append("backup: forced")
    elif last_backup_dt is None or last_backup_dt.date() < now_jst().date():
        plan.do_backup = True
        plan.reasons.append(f"backup: last={last_backup_iso or 'never'}")
    # fetch: cooldown
    hrs = hours_since(state.get("last_fetch"))
    if hrs is None or hrs > FETCH_COOLDOWN_HOURS:
        plan.do_fetch = True
        plan.reasons.append(
            f"fetch: last={hrs:.1f}h ago" if hrs is not None else "fetch: first run"
        )
    # parse: only if we ran fetch this turn (otherwise no new data)
    plan.do_parse = plan.do_fetch
    if plan.do_parse:
        plan.reasons.append("parse: because fetched")
    # generate / record are always attempted (no-op if no new races)
    return plan


# ============================================================================
#  Concrete steps
# ============================================================================
def step_backup_db(dry_run: bool) -> bool:
    today = now_jst().strftime("%Y%m%d")
    dst = BACKUPS_DIR / f"keiba_{today}.sqlite"
    if dst.exists():
        log("INFO", "backup: %s already exists, skip", dst)
        return True
    if not DB_FILE.exists():
        log("WARN", "backup: DB %s not found, skip", DB_FILE)
        return True
    if dry_run:
        log("INFO", "[DRY-RUN] would copy %s → %s", DB_FILE, dst)
        return True
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(DB_FILE, dst)
        log("INFO", "backup OK: %s (%d KB)", dst, dst.stat().st_size // 1024)
        return True
    except OSError as e:
        log("ERROR", "backup failed: %s", e)
        return False


def step_fetch(dry_run: bool) -> bool:
    current_year = now_jst().year
    ok, _ = run_step(
        "fetch_race_results",
        ["scripts/ingest/fetch_race_results.py", "--years", str(current_year)],
        dry_run, retries=2,
    )
    return ok


def step_parse(dry_run: bool) -> bool:
    # NOTE: NO --rebuild — auto pipeline must be INCREMENTAL.
    # --rebuild wipes races/entries/payouts. With weekly fetch returning only
    # the current year's races so far (e.g. 81 in early-June), a rebuild would
    # delete all historical data (multi-year, 1878 races) and replace with 81.
    # Incremental mode only touches the race_ids that were just parsed; the rest
    # of the DB is left intact.
    #
    # We also pass --min-races-threshold 0 because the safety guard (default 100)
    # is intended to protect against accidental --rebuild wipes; incremental mode
    # is safe regardless of batch size.
    ok, _ = run_step(
        "parse_race_results",
        ["scripts/transform/parse_race_results.py", "--min-races-threshold", "0"],
        dry_run,
    )
    return ok


def step_generate_today(dry_run: bool) -> tuple[bool, Path | None]:
    today = now_jst().strftime("%Y-%m-%d")
    ok, _ = run_step(
        "generate_tickets",
        ["scripts/live/generate_tickets.py", "--date", today, "--snapshot-time", "final"],
        dry_run,
    )
    # tickets CSV path that generate_tickets would write
    tickets_path = PROCESSED_DIR / f"live_tickets_{today}_final.csv"
    return ok, tickets_path if tickets_path.exists() else None


def step_record_today(tickets_csv: Path | None, dry_run: bool, bankroll: int) -> bool:
    if tickets_csv is None or not tickets_csv.exists():
        log("INFO", "record: no tickets CSV for today, skip")
        return True
    ok, _ = run_step(
        "record_result",
        ["scripts/live/record_result.py",
         "--tickets", str(tickets_csv),
         "--bankroll", str(bankroll)],
        dry_run,
    )
    return ok


# ============================================================================
#  Status command
# ============================================================================
def cmd_status() -> int:
    print(f"=== keiba-syutoku auto_paper_trading STATUS ===")
    print(f"now (JST)              : {now_iso()}")
    print(f"project_root           : {PROJECT_ROOT}")
    print()
    # lock
    if LOCK_FILE.exists():
        try:
            data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            age_h = hours_since(data.get("acquired_at"))
            print(f"lock                   : HELD (pid={data.get('pid')}, "
                  f"host={data.get('host')}, age={age_h:.2f}h)")
            if age_h is not None and age_h > LOCK_STALE_HOURS:
                print(f"                         ⚠ STALE (>{LOCK_STALE_HOURS}h) — "
                      f"will be auto-released on next run")
        except Exception as e:
            print(f"lock                   : CORRUPT ({e})")
    else:
        print(f"lock                   : free")
    # state
    state = load_state()
    print()
    print(f"runs_total             : {state.get('runs_total')}")
    print(f"runs_successful        : {state.get('runs_successful')}")
    print(f"runs_failed            : {state.get('runs_failed')}")
    for k in ("last_fetch", "last_parse", "last_generate", "last_record",
                "last_backup", "last_success", "last_error"):
        v = state.get(k)
        hrs = hours_since(v) if v else None
        suffix = f" ({hrs:.1f}h ago)" if hrs is not None else ""
        print(f"{k:<22} : {v}{suffix}")
    # logs
    print()
    if LOGS_DIR.exists():
        today_log = LOGS_DIR / f"paper_trading_{now_jst():%Y%m%d}.log"
        if today_log.exists():
            print(f"today's log            : {today_log} ({today_log.stat().st_size} B)")
            print(f"---- last 5 lines ----")
            lines = today_log.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-5:]:
                print(f"  {line}")
        else:
            print(f"today's log            : (none yet at {today_log})")
    else:
        print(f"logs dir               : (none)")
    # backups
    print()
    if BACKUPS_DIR.exists():
        backups = sorted(BACKUPS_DIR.glob("keiba_*.sqlite"))
        print(f"backups in {BACKUPS_DIR}: {len(backups)}")
        for b in backups[-3:]:
            print(f"  {b.name}  ({b.stat().st_size // 1024} KB)")
    else:
        print(f"backups                : (none)")
    # cron / systemd presence (linux only)
    print()
    print(f"--- scheduler ---")
    cron_out = ""
    try:
        cron_out = subprocess.run(["crontab", "-l"], capture_output=True, text=True,
                                    timeout=5).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        cron_out = "(crontab not available on this OS)"
    keiba_cron = [l for l in cron_out.splitlines() if "KeibaPaperTrading" in l]
    print(f"cron entries           : {len(keiba_cron)}")
    for l in keiba_cron:
        print(f"  {l}")
    try:
        sd = subprocess.run(
            ["systemctl", "--user", "list-timers", "keiba-paper.timer"],
            capture_output=True, text=True, timeout=5,
        )
        if sd.returncode == 0 and "keiba-paper.timer" in sd.stdout:
            print(f"systemd timer          : INSTALLED")
            for l in sd.stdout.splitlines():
                if "keiba-paper" in l or "NEXT" in l:
                    print(f"  {l}")
        else:
            print(f"systemd timer          : not installed")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"systemd timer          : (systemctl not available)")
    return 0


# ============================================================================
#  Main run
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["single-run", "dry-run"], default="single-run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force-unlock", action="store_true")
    ap.add_argument("--backup-db", action="store_true",
                    help="run a DB backup and exit (still respects 'already backed up today')")
    ap.add_argument("--bankroll", type=int, default=100_000)
    args = ap.parse_args()

    if args.status:
        return cmd_status()

    if args.force_unlock and not args.status:
        if LOCK_FILE.exists():
            try:
                LOCK_FILE.unlink()
                print("lock removed")
            except OSError as e:
                print(f"failed: {e}", file=sys.stderr)
                return 1
        else:
            print("no lock to remove")
        # if user passed only --force-unlock, exit without running pipeline
        if not (args.backup_db or args.dry_run):
            return 0

    dry_run = args.dry_run or args.mode == "dry-run"
    state = load_state()
    state["runs_total"] = state.get("runs_total", 0) + 1
    state["last_run_at"] = now_iso()
    state["current_lock"] = True

    # signal handler: clean lock on SIGTERM / SIGINT
    interrupted = {"v": False}
    def _sig(sig, frame):
        log("WARN", "signal %s received — finishing safely", sig)
        interrupted["v"] = True
    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)

    lock = PaperTradingLock(LOCK_FILE)
    success = False
    run_summary = {"started_at": now_iso(), "dry_run": dry_run}
    try:
        lock.acquire(force=args.force_unlock)

        plan = decide_plan(state, force_backup=args.backup_db)
        log("INFO", "plan: backup=%s fetch=%s parse=%s gen=%s record=%s",
              plan.do_backup, plan.do_fetch, plan.do_parse,
              plan.do_generate, plan.do_record)
        for r in plan.reasons:
            log("INFO", "  reason: %s", r)

        # ---- notify: pipeline start (deduped to once per day) ----
        if not dry_run:
            _notify(
                f"🏇 keiba paper trading: pipeline started\n"
                f"  date    : {now_jst():%Y-%m-%d (%a)}\n"
                f"  mode    : single-run\n"
                f"  plan    : backup={plan.do_backup} fetch={plan.do_fetch} "
                f"parse={plan.do_parse} gen={plan.do_generate} record={plan.do_record}",
                kind="start",
            )

        if plan.do_backup:
            ok = step_backup_db(dry_run)
            if ok and not dry_run:
                state["last_backup"] = now_iso()
            run_summary["backup"] = "ok" if ok else "fail"

        # ---- gated execution: if a critical upstream step fails,
        # ----                   downstream steps are SKIPPED (not silently OK).
        # ---- fetch  → parse → generate → record  (each gates the next)
        upstream_failed = False

        if plan.do_fetch and not interrupted["v"]:
            ok = step_fetch(dry_run)
            if ok and not dry_run:
                state["last_fetch"] = now_iso()
            run_summary["fetch"] = "ok" if ok else "fail"
            if not ok:
                state["last_error"] = "fetch failed"
                # fetch failure is recoverable (next run retries) → continue
                # but mark so we can decide if we trust new data
                if not dry_run:
                    log("WARN", "fetch failed — skipping parse (no new data to commit)")
                    _notify_error("fetch", "fetch_race_results exit != 0 — will retry next cron tick")
                    upstream_failed = True

        # parse only if (fetch ran AND succeeded) OR fetch wasn't due
        if plan.do_parse and not interrupted["v"] and not upstream_failed:
            ok = step_parse(dry_run)
            if ok and not dry_run:
                state["last_parse"] = now_iso()
            run_summary["parse"] = "ok" if ok else "fail"
            if not ok:
                state["last_error"] = "parse failed"
                if not dry_run:
                    log("ERROR", "parse failed — SKIPPING generate/record to avoid "
                                  "betting on stale/inconsistent data")
                    _notify_error("parse", "parse_race_results --rebuild failed — "
                                            "generate/record skipped")
                    upstream_failed = True
        elif plan.do_parse and upstream_failed:
            run_summary["parse"] = "skip_upstream_failed"

        tickets_csv = None
        if plan.do_generate and not interrupted["v"] and not upstream_failed:
            ok, tickets_csv = step_generate_today(dry_run)
            if ok and not dry_run:
                state["last_generate"] = now_iso()
            run_summary["generate"] = "ok" if ok else "fail"
            if ok and not dry_run:
                # success notify (deduped: once per day per snapshot/strategy)
                n_tickets = "?"
                tickets_label = "(no CSV — likely no JRA G1-G3 race today)"
                if tickets_csv and tickets_csv.exists():
                    try:
                        # count rows (minus header)
                        with open(tickets_csv, "r", encoding="utf-8-sig") as f:
                            n_tickets = max(0, sum(1 for _ in f) - 1)
                        tickets_label = f"{tickets_csv.name} ({n_tickets} tickets)"
                    except OSError:
                        tickets_label = tickets_csv.name
                _notify(
                    f"🎫 tickets generated\n"
                    f"  date     : {now_jst():%Y-%m-%d}\n"
                    f"  snapshot : final\n"
                    f"  strategy : {STRATEGY_LABEL}\n"
                    f"  file     : {tickets_label}",
                    kind="generate",
                    snapshot_time="final",
                )
            if not ok and not dry_run:
                state["last_error"] = "generate failed"
                _notify_error("generate", "generate_tickets failed for today")
                upstream_failed = True
        elif plan.do_generate and upstream_failed:
            run_summary["generate"] = "skip_upstream_failed"

        if plan.do_record and not interrupted["v"] and not upstream_failed:
            ok = step_record_today(tickets_csv, dry_run, args.bankroll)
            if ok and not dry_run:
                state["last_record"] = now_iso()
            run_summary["record"] = "ok" if ok else "fail"
            if ok and not dry_run:
                _notify(
                    f"📊 results recorded\n"
                    f"  date     : {now_jst():%Y-%m-%d}\n"
                    f"  strategy : {STRATEGY_LABEL}\n"
                    f"  log      : data/processed/paper_trading_log.csv",
                    kind="record",
                    snapshot_time="final",
                )
            if not ok and not dry_run:
                state["last_error"] = "record failed"
                _notify_error("record", "record_result failed for today")
        elif plan.do_record and upstream_failed:
            run_summary["record"] = "skip_upstream_failed"

        # overall success requires no upstream failures (in non-dry-run mode)
        success = (not upstream_failed) or dry_run

    except LockError as e:
        log("ERROR", "lock: %s", e)
        run_summary["error"] = f"lock: {e}"
        state["last_error"] = f"lock: {e}"
        # NOT notifying lock errors — they fire every 5 min on cron when a long
        # run is still in progress, would spam Telegram.
    except KeyboardInterrupt:
        log("WARN", "KeyboardInterrupt — exiting safely")
        run_summary["error"] = "KeyboardInterrupt"
    except Exception as e:
        log("ERROR", "unhandled: %s", e)
        run_summary["error"] = str(e)
        state["last_error"] = str(e)
        if not dry_run:
            _notify_error("unhandled", str(e))
    finally:
        lock.release()
        state["current_lock"] = False
        run_summary["finished_at"] = now_iso()
        run_summary["success"] = success
        state["history"].append(run_summary)
        if success:
            state["runs_successful"] = state.get("runs_successful", 0) + 1
            state["last_success"] = now_iso()
        else:
            state["runs_failed"] = state.get("runs_failed", 0) + 1
        save_state(state)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

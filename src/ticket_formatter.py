"""Format live_tickets CSV → human-readable Telegram message(s).

Used by auto_paper_trading.step_generate_today to send detailed buy-target
notifications. Reads the live_tickets_{date}_{snapshot}.csv produced by
generate_tickets.py and turns it into per-race Telegram messages.

Design:
  - One race = one block; multiple races chunked to stay under Telegram's
    4096-char per-message limit
  - Race start-time is APPROXIMATE — derived from the race number suffix of
    race_id using the JRA standard weekend schedule (no exact post_time in DB)
  - All ticket combinations for one race are listed inline (typical D9/P3/CAP4
    = 6-8 tickets per race, well within 4 KB)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

# ============================================================================
#  Approximate JRA weekend race start times
# ============================================================================
# Derived from standard JRA weekend race-day schedule. Actual times vary
# ±15 min by venue and day; user should treat these as DEADLINE GUIDES only.
JRA_APPROX_START_TIMES = {
    1:  "09:50",
    2:  "10:20",
    3:  "10:50",
    4:  "11:25",
    5:  "11:55",
    6:  "12:30",
    7:  "13:00",
    8:  "13:35",
    9:  "14:10",
    10: "14:45",
    11: "15:25",   # main race (重賞 typically here)
    12: "16:00",
}


def _approx_post_time(race_id: str) -> str:
    """race_id last 2 digits = race number; map to approximate JST start time.

    Returns "??:??" if race_id doesn't parse.
    """
    if not race_id or len(race_id) < 2:
        return "??:??"
    try:
        race_no = int(race_id[-2:])
        return JRA_APPROX_START_TIMES.get(race_no, "??:??")
    except (ValueError, TypeError):
        return "??:??"


def _safe(value, default: str = "?") -> str:
    """Stringify a CSV cell, treating empty/'nan'/None as default."""
    if value is None:
        return default
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return default
    return s


def _fmt_odds(value) -> str:
    s = _safe(value, "-")
    try:
        return f"{float(s):.1f}"
    except ValueError:
        return s


# ============================================================================
#  Per-race block builder
# ============================================================================
def _format_race_block(race_id: str, rows: list[dict]) -> str:
    """Build one race's notification block. Returns multi-line string."""
    head = rows[0]
    race_name   = _safe(head.get("race_name"), "(レース名不明)")
    grade       = _safe(head.get("grade"), "")
    racecourse  = _safe(head.get("racecourse"), "?")
    surface     = _safe(head.get("surface"), "")
    distance    = _safe(head.get("distance"), "")
    field_size  = _safe(head.get("field_size"), "?")
    post_time   = _approx_post_time(race_id)
    race_no     = race_id[-2:] if len(race_id) >= 2 else "??"

    grade_tag = f" {grade}" if grade and grade != "?" else ""
    surf_dist = f"{surface}{distance}m" if surface != "?" and distance != "?" else ""

    # 本命 + ダーク (= same across all rows for a given race)
    p1_no    = _safe(head.get("p1_horse_number"))
    p1_name  = _safe(head.get("p1_horse_name"), "")
    p1_odds  = _fmt_odds(head.get("p1_odds"))
    p2_no    = _safe(head.get("p2_horse_number"))
    p2_name  = _safe(head.get("p2_horse_name"), "")
    p2_odds  = _fmt_odds(head.get("p2_odds"))
    dk_no    = _safe(head.get("dark_horse_number"))
    dk_name  = _safe(head.get("dark_horse_name"), "")
    dk_pop   = _safe(head.get("dark_popularity"))
    dk_odds  = _fmt_odds(head.get("dark_odds"))

    # 全 tickets を縦に
    tickets = [_safe(r.get("ticket")) for r in rows]
    tickets = [t for t in tickets if t and t != "?"]
    # 1 点単価 × 点数 = 合計
    try:
        per_stake = int(float(_safe(rows[0].get("stake_yen"), "0")))
    except ValueError:
        per_stake = 0
    total_stake = per_stake * len(tickets)

    lines = []
    lines.append(f"▼ {racecourse} {race_no}R {race_name}{grade_tag}")
    if surf_dist:
        lines.append(f"   {surf_dist}  {field_size}頭立て")
    lines.append(f"   ⏰ 〜{post_time} (発走目安)")
    lines.append(f"   ◎ {p1_no} {p1_name}({p1_odds}倍)")
    lines.append(f"   ○ {p2_no} {p2_name}({p2_odds}倍)")
    lines.append(f"   ★ {dk_no} {dk_name} ({dk_pop}番人気, {dk_odds}倍)")
    lines.append(f"   買い目 ({len(tickets)}点 ¥{total_stake}):")
    # 3 tickets per row to keep compact
    for i in range(0, len(tickets), 3):
        chunk = ", ".join(tickets[i:i+3])
        lines.append(f"     {chunk}")
    return "\n".join(lines)


# ============================================================================
#  Public entry: parse CSV → list of Telegram-ready messages
# ============================================================================
TELEGRAM_MAX_LEN = 3800   # leave headroom under 4096 hard limit


def format_tickets_for_telegram(csv_path: Path, date_label: str,
                                snapshot_time: str,
                                rule_name: str | None = None) -> list[str]:
    """Read live_tickets CSV → return list of messages (each ≤ TELEGRAM_MAX_LEN).

    Empty CSV / parse failures → returns [] (caller skips notification).
    Multiple races may be split across messages if total size > TELEGRAM_MAX_LEN.
    """
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []
    if not rows:
        return []

    # group by race_id, preserving first-seen order
    by_race: dict[str, list[dict]] = defaultdict(list)
    race_order: list[str] = []
    for r in rows:
        rid = _safe(r.get("race_id"))
        if rid == "?":
            continue
        if rid not in by_race:
            race_order.append(rid)
        by_race[rid].append(r)

    if not by_race:
        return []

    rule = rule_name or _safe(rows[0].get("rule_name"), "E_D9_P3_CAP4")
    n_races = len(race_order)
    n_tickets = sum(len(by_race[rid]) for rid in race_order)
    try:
        per_stake = int(float(_safe(rows[0].get("stake_yen"), "0")))
    except ValueError:
        per_stake = 0
    total_stake = per_stake * n_tickets

    header = (
        f"🎫 {date_label} 買い目生成\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"strategy : {rule}\n"
        f"snapshot : {snapshot_time}\n"
        f"{n_races}R / {n_tickets}点 / ¥{total_stake}\n"
        f"━━━━━━━━━━━━━━━━"
    )

    # build per-race blocks
    blocks = [_format_race_block(rid, by_race[rid]) for rid in race_order]

    # pack blocks into messages, respecting TELEGRAM_MAX_LEN
    messages: list[str] = []
    current = header
    for block in blocks:
        candidate = current + "\n\n" + block
        if len(candidate) > TELEGRAM_MAX_LEN:
            messages.append(current)
            # subsequent messages get a short continuation header
            current = f"🎫 {date_label} (続き)\n━━━━━━━━━━━━━━━━\n\n{block}"
        else:
            current = candidate
    if current:
        messages.append(current)

    # footer (only attached to last message if there's room)
    footer = (
        "\n━━━━━━━━━━━━━━━━\n"
        "※ 発走目安は JRA 標準スケジュール推定。\n"
        "  正確な締切は netkeiba / JRA 公式で確認。"
    )
    if messages and len(messages[-1]) + len(footer) <= TELEGRAM_MAX_LEN:
        messages[-1] += footer
    else:
        messages.append(footer.strip())

    return messages

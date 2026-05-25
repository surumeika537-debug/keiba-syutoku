"""Simple trifecta ROI backtest with filtering + multi-axis breakdown.

Strategies (each ticket = 100 yen; hit iff ticket string == actual 三連単 combination):
  A. 1番人気を1着固定、2〜5番人気を2着/3着に流す
  B. 1〜3番人気の三連単ボックス
  C. 1番人気を1着固定、4〜8番人気を2着/3着に流す
  D. 1番人気・2番人気 + (N番人気以下 AND 単勝オッズ10〜30倍) の馬1頭を含む三連単ボックス
     N is configurable via --strategy-d-min-popularity (default 5).

Filters:
  --jra-only                 JRA10場のみ
  --exclude-steeplechase     surface == 障害 を除外
  --grades G1 G2 G3          グレード絞り込み
  --from / --to              日付範囲 (YYYY-MM-DD)

Outputs (all under data/processed/):
  backtest_summary.csv               全体 (group_key=ALL)
  backtest_summary_by_year.csv       strategy x year
  backtest_summary_by_grade.csv      strategy x grade
  backtest_summary_by_racecourse.csv strategy x racecourse
  backtest_summary_by_surface.csv    strategy x surface
  backtest_hits.csv                  的中レースの明細
"""
from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.config import JRA_RACECOURSES, PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("backtest")

TICKET_COST = 100
TRIFECTA_LABELS = ("三連単", "3連単")


# ---- ticket generators (pre-race info only) -------------------------------

def _horse_by_popularity(entries: pd.DataFrame, pop: int) -> int | None:
    row = entries.loc[entries["popularity"] == pop]
    if row.empty:
        return None
    return int(row["horse_number"].iloc[0])


def _horses_by_popularity_range(entries: pd.DataFrame, lo: int, hi: int) -> list[int]:
    sub = entries[entries["popularity"].between(lo, hi)]
    return [int(x) for x in sub["horse_number"].tolist()]


def tickets_A(entries: pd.DataFrame, race_meta: dict | None = None) -> list[str]:
    p1 = _horse_by_popularity(entries, 1)
    if p1 is None:
        return []
    others = [h for h in _horses_by_popularity_range(entries, 2, 5) if h != p1]
    return [f"{p1}-{h2}-{h3}" for h2, h3 in permutations(others, 2)]


def tickets_B(entries: pd.DataFrame, race_meta: dict | None = None) -> list[str]:
    horses = _horses_by_popularity_range(entries, 1, 3)
    if len(horses) < 3:
        return []
    return [f"{a}-{b}-{c}" for a, b, c in permutations(horses, 3)]


def tickets_C(entries: pd.DataFrame, race_meta: dict | None = None) -> list[str]:
    p1 = _horse_by_popularity(entries, 1)
    if p1 is None:
        return []
    others = [h for h in _horses_by_popularity_range(entries, 4, 8) if h != p1]
    return [f"{p1}-{h2}-{h3}" for h2, h3 in permutations(others, 2)]


def make_tickets_D(min_popularity: int) -> Callable[..., list[str]]:
    """Factory: returns a D-strategy ticket generator with the popularity threshold baked in."""

    def tickets_D(entries: pd.DataFrame, race_meta: dict | None = None) -> list[str]:
        p1 = _horse_by_popularity(entries, 1)
        p2 = _horse_by_popularity(entries, 2)
        if p1 is None or p2 is None:
            return []
        quals = entries[
            (entries["popularity"] >= min_popularity)
            & (entries["win_odds"].between(10.0, 30.0))
        ]
        cand_horses = [int(h) for h in quals["horse_number"].tolist() if int(h) not in (p1, p2)]
        out: list[str] = []
        for q in cand_horses:
            out.extend(f"{a}-{b}-{c}" for a, b, c in permutations([p1, p2, q], 3))
        return list(dict.fromkeys(out))

    return tickets_D


# -------- strategy E: D9 + P3 + cap4 (production rule, see docs in README) --

NIIGATA_4GIII_PATTERNS = ("アイビスサマー", "関屋記念", "新潟記念", "レパード")


def is_d9_candidate_race(race_meta: dict) -> bool:
    """D9 race-inclusion test: niigata_4g3 | march_g2 | hanshin_g1 | september_g3."""
    name = str(race_meta.get("race_name") or "")
    grade = race_meta.get("grade")
    rc = race_meta.get("racecourse")
    month = race_meta.get("month")
    return (
        (rc == "新潟" and grade == "G3" and any(p in name for p in NIIGATA_4GIII_PATTERNS))
        or (month == 3 and grade == "G2")
        or (rc == "阪神" and grade == "G1")
        or (month == 9 and grade == "G3")
    )


def d9_component_labels(race_meta: dict) -> str:
    name = str(race_meta.get("race_name") or "")
    grade = race_meta.get("grade")
    rc = race_meta.get("racecourse")
    month = race_meta.get("month")
    labels = []
    if rc == "新潟" and grade == "G3" and any(p in name for p in NIIGATA_4GIII_PATTERNS):
        labels.append("niigata_4g3")
    if month == 3 and grade == "G2":
        labels.append("march_g2")
    if rc == "阪神" and grade == "G1":
        labels.append("hanshin_g1")
    if month == 9 and grade == "G3":
        labels.append("september_g3")
    return "|".join(labels)


def is_d6_excluded_race(race_meta: dict) -> bool:
    """D6 race-level negative filter (drop entire race)."""
    grade = race_meta.get("grade")
    rc = race_meta.get("racecourse")
    return ((rc == "京都" and grade == "G1")
            or (rc == "中山" and grade == "G1")
            or (rc == "小倉" and grade == "G3"))


def select_d9_p3_cap4_darks(entries: pd.DataFrame, race_meta: dict,
                             max_darks: int = 4) -> pd.DataFrame:
    """Return ordered DataFrame of dark horses (max 4) per D9 P3 cap4 rules.
    Includes: horse_number, popularity, win_odds, frame_number, dark_rank (1-indexed)."""
    quals = entries[
        (entries["popularity"] >= 5)
        & (entries["win_odds"].between(10.0, 30.0))
    ].copy()
    # D6 subset filter: in 17-18 races, drop dark in frames 3-4
    fs = race_meta.get("field_size")
    if fs is not None and 17 <= int(fs) <= 18:
        quals = quals[~quals["frame_number"].isin([3, 4])]
    if quals.empty:
        return quals
    # remove p1/p2 from dark pool (should already not match by popularity, defensive)
    p1 = _horse_by_popularity(entries, 1)
    p2 = _horse_by_popularity(entries, 2)
    quals = quals[~quals["horse_number"].isin([p1, p2])]
    # order: win_odds asc, popularity asc, horse_number asc
    quals = quals.sort_values(["win_odds", "popularity", "horse_number"],
                              kind="stable").head(max_darks)
    quals = quals.reset_index(drop=True)
    quals["dark_rank"] = quals.index + 1
    return quals


def tickets_E(entries: pd.DataFrame, race_meta: dict | None = None) -> list[str]:
    """E_D9_P3_CAP4 — production rule. Empty list if race is not D9-eligible."""
    if race_meta is None:
        return []
    if not is_d9_candidate_race(race_meta):
        return []
    if is_d6_excluded_race(race_meta):
        return []
    p1 = _horse_by_popularity(entries, 1)
    p2 = _horse_by_popularity(entries, 2)
    if p1 is None or p2 is None:
        return []
    darks = select_d9_p3_cap4_darks(entries, race_meta, max_darks=4)
    if darks.empty:
        return []
    out: list[str] = []
    for d in darks["horse_number"].astype(int).tolist():
        # P3: dark fixed at 3rd, p1/p2 swap as 1st/2nd
        out.append(f"{p1}-{p2}-{d}")
        out.append(f"{p2}-{p1}-{d}")
    return out


def build_strategies(strategy_d_min_popularity: int) -> dict[str, Callable[..., list[str]]]:
    return {
        "A: 1番人気固定 / 2-5番人気を相手": tickets_A,
        "B: 1-3番人気ボックス": tickets_B,
        "C: 1番人気固定 / 4-8番人気を相手": tickets_C,
        f"D: 1,2番人気 + ({strategy_d_min_popularity}番人気以下 odds10-30) BOX":
            make_tickets_D(strategy_d_min_popularity),
        "E_D9_P3_CAP4": tickets_E,
        "F_D9_DYNAMIC_STATE": tickets_F,
    }


# -------- strategy F: E + dynamic state (market_stop + regime aware) --------
#
# F は E の race-eligibility に対し、市場 stress state と regime で「買うか/見送るか」
# を上書きする。bet サイジング (1.5% / 0.75% / 0.5% / 0.25% / 0%) の経済影響は
# scripts/backtest/strategy_f_backtest.py で variable-stake シミュレーション。
# このモジュール内では「state == HALT または regime == DARK_SUPPRESSED の race は
# 空 list を返す」= バイナリ skip 判定に絞る。
#
# 依存: market_stress_timeseries.csv が PROCESSED_DIR に存在すること
#   (scripts/analysis/market_stop_system.py を事前実行)

_MARKET_STATE_CACHE: dict[str, tuple[str, str]] | None = None


def _load_market_state_cache() -> dict[str, tuple[str, str]]:
    """Lazy-load (state, regime) per race_id from market_stress_timeseries.csv."""
    global _MARKET_STATE_CACHE
    if _MARKET_STATE_CACHE is not None:
        return _MARKET_STATE_CACHE
    from src.config import PROCESSED_DIR as _PROCESSED_DIR
    path = _PROCESSED_DIR / "market_stress_timeseries.csv"
    if not path.exists():
        log.warning("market_stress_timeseries.csv not found at %s — "
                     "F will fall through to E (no skip logic)", path)
        _MARKET_STATE_CACHE = {}
        return _MARKET_STATE_CACHE
    df = pd.read_csv(path, dtype={"race_id": str})
    _MARKET_STATE_CACHE = {
        str(r["race_id"]): (str(r["market_state"]), str(r["regime"]))
        for _, r in df.iterrows()
    }
    log.info("loaded market state cache: %d races", len(_MARKET_STATE_CACHE))
    return _MARKET_STATE_CACHE


def get_market_state_regime(race_id: str) -> tuple[str, str]:
    cache = _load_market_state_cache()
    return cache.get(str(race_id), ("GREEN", "NORMAL"))  # default = open / normal


F_SKIP_STATES = {"HALT"}
F_SKIP_REGIMES = {"DARK_SUPPRESSED"}


def tickets_F(entries: pd.DataFrame, race_meta: dict | None = None) -> list[str]:
    """F_D9_DYNAMIC_STATE — E + market_stop + regime filter (binary skip layer).

    Eligibility logic:
      1. E が空なら空 (= D9-ineligible)
      2. state == HALT なら空 (= circuit breaker)
      3. regime == DARK_SUPPRESSED なら空 (= no usable dark pool)
      4. それ以外は E のチケットをそのまま採用
    """
    e_tickets = tickets_E(entries, race_meta)
    if not e_tickets:
        return []
    if race_meta is None:
        return e_tickets
    state, regime = get_market_state_regime(race_meta.get("race_id"))
    if state in F_SKIP_STATES:
        return []
    if regime in F_SKIP_REGIMES:
        return []
    return e_tickets


# ---- data loading with filtering ------------------------------------------

def apply_race_filters(
    races: pd.DataFrame,
    *,
    grades: tuple[str, ...] | None,
    date_from: str | None,
    date_to: str | None,
    jra_only: bool,
    exclude_steeplechase: bool,
) -> pd.DataFrame:
    out = races.copy()
    if grades:
        out = out[out["grade"].isin(grades)]
    if date_from:
        out = out[out["race_date"] >= pd.Timestamp(date_from)]
    if date_to:
        out = out[out["race_date"] <= pd.Timestamp(date_to)]
    if jra_only:
        out = out[out["racecourse"].isin(JRA_RACECOURSES)]
    if exclude_steeplechase:
        out = out[out["surface"] != "障害"]
    return out


def load_data(
    *,
    grades: tuple[str, ...] | None,
    date_from: str | None,
    date_to: str | None,
    jra_only: bool,
    exclude_steeplechase: bool,
):
    engine = get_engine()
    races = pd.read_sql("SELECT * FROM races", engine, parse_dates=["race_date"])
    entries = pd.read_sql("SELECT * FROM entries", engine)
    payouts = pd.read_sql("SELECT * FROM payouts", engine)

    races = apply_race_filters(
        races,
        grades=grades,
        date_from=date_from,
        date_to=date_to,
        jra_only=jra_only,
        exclude_steeplechase=exclude_steeplechase,
    )

    keep = set(races["race_id"])
    entries = entries[entries["race_id"].isin(keep)].copy()
    payouts = payouts[payouts["race_id"].isin(keep)].copy()

    # win_odds is REAL in the DB; just ensure it's numeric (NaN where NULL).
    entries["win_odds"] = pd.to_numeric(entries["win_odds"], errors="coerce")
    entries["popularity"] = pd.to_numeric(entries["popularity"], errors="coerce").astype("Int64")
    entries["horse_number"] = pd.to_numeric(entries["horse_number"], errors="coerce").astype("Int64")
    entries = entries.dropna(subset=["popularity", "horse_number"])
    return races, entries, payouts


def trifecta_lookup(payouts: pd.DataFrame) -> dict[str, tuple[str, int]]:
    tri = payouts[payouts["bet_type"].isin(TRIFECTA_LABELS)]
    out: dict[str, tuple[str, int]] = {}
    for _, r in tri.iterrows():
        out.setdefault(r["race_id"], (str(r["combination"]), int(r["payout_yen"] or 0)))
    return out


# ---- per-race detail (single source of truth for all aggregations) --------

def compute_detail(
    races: pd.DataFrame,
    entries: pd.DataFrame,
    payouts: pd.DataFrame,
    strategies: dict[str, Callable[..., list[str]]],
) -> pd.DataFrame:
    tri = trifecta_lookup(payouts)
    meta = races.set_index("race_id")[
        ["race_date", "race_name", "grade", "racecourse", "surface"]
    ]
    # precompute per-race metadata for strategies that need it (E uses month, field_size, etc.)
    meta_dict = {}
    field_sizes = entries.groupby("race_id").size().to_dict()
    for rid, row in meta.iterrows():
        rd = row.get("race_date")
        meta_dict[rid] = {
            "race_id": rid,
            "race_name": row.get("race_name"),
            "race_date": rd,
            "racecourse": row.get("racecourse"),
            "grade": row.get("grade"),
            "surface": row.get("surface"),
            "month": int(pd.Timestamp(rd).month) if pd.notna(rd) else None,
            "field_size": field_sizes.get(rid),
        }

    rows: list[dict] = []
    for race_id, race_entries in entries.groupby("race_id", sort=False):
        true_combo, true_payout = tri.get(race_id, (None, 0))
        race_meta = meta_dict.get(race_id, {})
        for strategy, gen in strategies.items():
            tickets = gen(race_entries, race_meta)
            if not tickets:
                continue
            hit_ticket = true_combo if (true_combo and true_combo in tickets) else None
            cost = len(tickets) * TICKET_COST
            payout = true_payout if hit_ticket else 0
            rows.append({
                "race_id": race_id,
                "strategy": strategy,
                "n_tickets": len(tickets),
                "hit_ticket": hit_ticket,
                "investment_yen": cost,
                "payout_yen": payout,
                "profit_yen": payout - cost,
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail

    detail = detail.merge(meta, left_on="race_id", right_index=True, how="left")
    detail["year"] = detail["race_date"].dt.year
    return detail


# ---- aggregation ----------------------------------------------------------

AGG_COLUMNS = [
    "strategy", "group_key", "races", "tickets", "investment_yen",
    "hits", "payout_yen", "profit_yen", "roi", "hit_rate",
    "max_payout_yen", "max_losing_streak", "sample_warning",
]
LOW_SAMPLE_THRESHOLD = 30


def _sample_warning(races) -> str:
    """Flag groups where the ROI estimate is too noisy to trust."""
    if races is None or pd.isna(races):
        return ""
    return "LOW_SAMPLE" if int(races) < LOW_SAMPLE_THRESHOLD else ""


def _longest_loss_streak(sub: pd.DataFrame) -> int:
    """Within `sub` (one strategy x group), sort by race_date and find the longest
    consecutive miss run. Race_date NaT goes to the end."""
    ordered = sub.sort_values("race_date", kind="stable", na_position="last")
    cur = best = 0
    for hit in ordered["hit_ticket"]:
        if pd.isna(hit):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def aggregate_by(detail: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate detail rows by (strategy, group_col). Returns the canonical schema."""
    if detail.empty:
        return pd.DataFrame(columns=AGG_COLUMNS)

    rows = []
    for (strategy, group_key), sub in detail.groupby(["strategy", group_col], dropna=False):
        cost = int(sub["investment_yen"].sum())
        payout = int(sub["payout_yen"].sum())
        hits = int(sub["hit_ticket"].notna().sum())
        races = int(sub["race_id"].nunique())
        rows.append({
            "strategy": strategy,
            "group_key": "(none)" if pd.isna(group_key) else group_key,
            "races": races,
            "tickets": int(sub["n_tickets"].sum()),
            "investment_yen": cost,
            "hits": hits,
            "payout_yen": payout,
            "profit_yen": payout - cost,
            "roi": (payout - cost) / cost if cost else 0.0,
            "hit_rate": hits / races if races else 0.0,
            "max_payout_yen": int(sub["payout_yen"].max()) if hits else 0,
            "max_losing_streak": _longest_loss_streak(sub),
            "sample_warning": _sample_warning(races),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["strategy", "group_key"], kind="stable").reset_index(drop=True)


def build_hits(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    hits = detail[detail["hit_ticket"].notna()].copy()
    hits = hits.rename(columns={"hit_ticket": "ticket"})
    return hits[[
        "race_id", "race_date", "race_name", "grade",
        "strategy", "ticket", "payout_yen", "investment_yen", "profit_yen",
    ]].sort_values(["race_date", "strategy"], kind="stable")


# ---- main -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grades", nargs="+", default=None, help="filter by grade (e.g. G1 G2 G3)")
    ap.add_argument("--jra-only", action="store_true", help="keep only JRA (10場) races")
    ap.add_argument("--exclude-steeplechase", action="store_true", help="drop surface==障害")
    ap.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD")
    ap.add_argument("--strategy-d-min-popularity", type=int, default=5,
                    help="strategy D: minimum popularity for the dark-horse leg (default 5)")
    args = ap.parse_args()

    races, entries, payouts = load_data(
        grades=tuple(args.grades) if args.grades else None,
        date_from=args.date_from,
        date_to=args.date_to,
        jra_only=args.jra_only,
        exclude_steeplechase=args.exclude_steeplechase,
    )
    log.info("after filters: races=%d entries=%d payouts=%d", len(races), len(entries), len(payouts))

    strategies = build_strategies(args.strategy_d_min_popularity)
    detail = compute_detail(races, entries, payouts, strategies)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. overall (group_key = ALL)
    detail_overall = detail.copy()
    detail_overall["_all"] = "ALL"
    overall = aggregate_by(detail_overall, "_all")
    overall.to_csv(PROCESSED_DIR / "backtest_summary.csv", index=False, encoding="utf-8-sig")

    # 2-5. by year / grade / racecourse / surface
    by_year = aggregate_by(detail, "year")
    by_grade = aggregate_by(detail, "grade")
    by_course = aggregate_by(detail, "racecourse")
    by_surface = aggregate_by(detail, "surface")
    by_year.to_csv(PROCESSED_DIR / "backtest_summary_by_year.csv", index=False, encoding="utf-8-sig")
    by_grade.to_csv(PROCESSED_DIR / "backtest_summary_by_grade.csv", index=False, encoding="utf-8-sig")
    by_course.to_csv(PROCESSED_DIR / "backtest_summary_by_racecourse.csv", index=False, encoding="utf-8-sig")
    by_surface.to_csv(PROCESSED_DIR / "backtest_summary_by_surface.csv", index=False, encoding="utf-8-sig")

    # hits detail
    hits = build_hits(detail)
    hits.to_csv(PROCESSED_DIR / "backtest_hits.csv", index=False, encoding="utf-8-sig")

    log.info("wrote backtest_summary.csv (%d rows), by_year=%d by_grade=%d by_racecourse=%d by_surface=%d, hits=%d",
             len(overall), len(by_year), len(by_grade), len(by_course), len(by_surface), len(hits))

    if overall.empty:
        print("(no data — run ingest + parse first or relax filters)")
        return

    # Pretty stdout: overall + one quick view per dimension
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_colwidth", 50)

    print()
    print("=== OVERALL ===")
    print(overall.drop(columns="group_key").to_string(index=False))
    print()
    print("=== BY YEAR ===")
    print(by_year.to_string(index=False))
    print()
    print("=== BY GRADE ===")
    print(by_grade.to_string(index=False))


if __name__ == "__main__":
    main()

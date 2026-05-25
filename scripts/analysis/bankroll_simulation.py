"""D9 を時系列で実運用した場合の bankroll / drawdown シミュレーション。

5つの投資方式 × 4つの初期資金 = 20シナリオ を比較。

投資方式:
  A. fixed_100yen          各買い目 100yen 固定 (n_tickets × 100yen / race)
  B. fixed_bankroll_1pct   1レースに「初期資金の1%」を均等配分
  C. fixed_bankroll_2pct   初期資金の 2%
  D. fixed_bankroll_5pct   初期資金の 5%
  E. half_kelly_cap_2pct   過去D9実績から half-Kelly (≈4%) を上限 2% で適用
                           D9 集計: p=0.1604, b≈10.44 → full kelly 8%,
                           half kelly 4%, cap 2% → 結果 C と同等

bet 配分: 1レースの総 bet を tickets で均等割り。例: bet=1000 yen / 24枚
  = 1枚あたり 41.67yen。三連単の払戻は「100yen あたり X yen」表記なので、
  当たった 1枚の payout = (1枚 bet / 100) × raw_payout。

ruin: bankroll < 必要 bet になったレースは「skip」(賭けない)。skip が1件
  でも発生すると ruin_flag=True。bankroll が <= 0 になっても ruin_flag=True。

出力:
  data/processed/d9_bankroll_summary.csv
  data/processed/d9_bankroll_by_year.csv
  data/processed/d9_equity_curve.csv

Usage:
    python scripts/analysis/bankroll_simulation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.utils import force_utf8_stdout, setup_logger
from scripts.backtest.strategy_d_variants import (
    apply_variant,
    collapse_to_race,
    collect_subsets,
    load_data,
)
from scripts.analysis.d9_deepdive import make_component_labels, tag_components

force_utf8_stdout()
log = setup_logger("bankroll_sim")


# ---- bet-amount functions -------------------------------------------------

def bet_fn_100yen(race, initial, current):
    return float(race["cost"])  # = n_tickets × 100

def bet_fn_pct(pct):
    def fn(race, initial, current):
        return float(initial) * pct
    return fn

# E: half-kelly upper-bounded by 2%. Computed offline from D9 aggregates:
#   p = 30/187 = 0.1604, avg_payout_when_hit/avg_cost = 11.45 → b = 10.45
#   full kelly = (bp - q)/b = 0.0800 (8%); half kelly = 4%; capped at 2% → 2%.
# Implemented as min(half_kelly, 0.02). Since 0.04 > 0.02, equals C numerically.
HALF_KELLY_RAW = 0.04
HALF_KELLY_CAP = 0.02
HALF_KELLY_USED = min(HALF_KELLY_RAW, HALF_KELLY_CAP)

def bet_fn_half_kelly(race, initial, current):
    return float(initial) * HALF_KELLY_USED


METHODS = {
    "A_fixed_100yen":         bet_fn_100yen,
    "B_fixed_bankroll_1pct":  bet_fn_pct(0.01),
    "C_fixed_bankroll_2pct":  bet_fn_pct(0.02),
    "D_fixed_bankroll_5pct":  bet_fn_pct(0.05),
    "E_half_kelly_cap_2pct":  bet_fn_half_kelly,
}

INITIAL_BANKROLLS = [50_000, 100_000, 300_000, 500_000]


# ---- single simulation ----------------------------------------------------

def simulate(per_race: pd.DataFrame, method_name: str, bet_fn, initial: int):
    races = per_race.sort_values("race_date", kind="stable").reset_index(drop=True)
    bankroll = float(initial)
    peak = float(initial)
    min_bk = float(initial)
    max_dd_yen = 0.0
    max_dd_pct = 0.0
    streak_races = 0
    max_streak_races = 0
    max_streak_days = 0
    last_win_date = None
    first_race_date = pd.Timestamp(races["race_date"].iloc[0])
    ruin = False
    skipped_races = 0
    total_bet = 0.0
    total_payout = 0.0
    bet_amounts = []
    curve_rows = []

    for _, r in races.iterrows():
        intended_bet = float(bet_fn(r, initial, bankroll))
        race_date = pd.Timestamp(r["race_date"])

        if intended_bet > bankroll or intended_bet <= 0:
            # can't afford or invalid
            ruin = True
            skipped_races += 1
            bet = 0.0
            payout = 0.0
            profit = 0.0
        else:
            bet = intended_bet
            if r["hit"]:
                # The winning ticket is 1 of n_tickets; payout scales linearly with per-ticket bet.
                #   per_ticket_bet = bet / n_tickets
                #   payout = (per_ticket_bet / 100) * raw_payout
                #          = bet * raw_payout / (n_tickets * 100)
                #          = bet * raw_payout / cost          (since cost = n_tickets * 100)
                payout = bet * (float(r["payout"]) / float(r["cost"]))
                profit = payout - bet
            else:
                payout = 0.0
                profit = -bet
            total_bet += bet
            total_payout += payout
            bet_amounts.append(bet)

        bankroll += profit
        if bankroll > peak:
            peak = bankroll
        dd_yen = peak - bankroll
        dd_pct = dd_yen / peak if peak > 0 else 0.0
        if dd_yen > max_dd_yen:
            max_dd_yen = dd_yen
            max_dd_pct = dd_pct
        if bankroll < min_bk:
            min_bk = bankroll

        if r["hit"] and bet > 0:
            streak_races = 0
            ref = last_win_date or first_race_date
            days = (race_date - ref).days
            if days > max_streak_days:
                max_streak_days = days
            last_win_date = race_date
        elif bet > 0:
            streak_races += 1
            if streak_races > max_streak_races:
                max_streak_races = streak_races

        if bankroll <= 0:
            ruin = True

        curve_rows.append({
            "method": method_name,
            "initial_bankroll": initial,
            "race_date": r["race_date"],
            "race_id": r["race_id"],
            "race_name": r["race_name"],
            "component_labels": r["component_labels"],
            "tickets": int(r["n_tickets"]),
            "bet_amount": round(bet, 2),
            "payout": round(payout, 2),
            "profit": round(profit, 2),
            "bankroll_after": round(bankroll, 2),
            "drawdown_pct": round(dd_pct, 4),
        })

    # trailing dry-spell (after last win to end of period)
    if last_win_date is not None:
        tail = (pd.Timestamp(races["race_date"].iloc[-1]) - last_win_date).days
        if tail > max_streak_days:
            max_streak_days = tail

    summary = {
        "method": method_name,
        "initial_bankroll": int(initial),
        "final_bankroll": round(bankroll, 2),
        "total_profit": round(bankroll - initial, 2),
        "total_return_pct": (bankroll - initial) / initial if initial > 0 else 0.0,
        "max_drawdown_yen": round(max_dd_yen, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "max_losing_streak_races": int(max_streak_races),
        "max_losing_streak_days": int(max_streak_days),
        "ruin_flag": bool(ruin),
        "skipped_races": int(skipped_races),
        "min_bankroll": round(min_bk, 2),
        "avg_bet_per_race": round(float(np.mean(bet_amounts)), 2) if bet_amounts else 0.0,
        "max_bet_per_race": round(float(np.max(bet_amounts)), 2) if bet_amounts else 0.0,
        "total_bet": round(total_bet, 2),
        "total_payout": round(total_payout, 2),
        "roi": (total_payout - total_bet) / total_bet if total_bet > 0 else 0.0,
    }
    return summary, curve_rows


# ---- year aggregation -----------------------------------------------------

def aggregate_by_year(curve_df: pd.DataFrame, initial: int):
    """Per-year P&L for one (method, initial)."""
    if curve_df.empty:
        return []
    df = curve_df.copy()
    df["year"] = pd.to_datetime(df["race_date"]).dt.year
    rows = []
    running_start = float(initial)
    for year in sorted(df["year"].unique()):
        ysub = df[df["year"] == year].sort_values("race_date", kind="stable")
        end_bk = float(ysub["bankroll_after"].iloc[-1])
        # starting bankroll = end_bk - sum(profit) within year
        profit = float(ysub["profit"].sum())
        start_bk = end_bk - profit
        # within-year max drawdown
        peak = start_bk
        max_dd_pct = 0.0
        bk = start_bk
        for _, r in ysub.iterrows():
            bk += float(r["profit"])
            if bk > peak:
                peak = bk
            dd = (peak - bk) / peak if peak > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd
        races_n = int(len(ysub))
        bets_n = int((ysub["bet_amount"] > 0).sum())
        total_bet_y = float(ysub["bet_amount"].sum())
        total_payout_y = float(ysub["payout"].sum())
        hits = int((ysub["payout"] > 0).sum())
        rows.append({
            "year": int(year),
            "starting_bankroll": round(start_bk, 2),
            "ending_bankroll": round(end_bk, 2),
            "profit": round(profit, 2),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "races": races_n,
            "bets": bets_n,
            "hits": hits,
            "total_bet": round(total_bet_y, 2),
            "total_payout": round(total_payout_y, 2),
            "roi": (total_payout_y - total_bet_y) / total_bet_y if total_bet_y > 0 else 0.0,
        })
        running_start = end_bk
    return rows


# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args()

    log.info("loading D9 per-race detail (JRA flat G1/G2/G3)")
    races, entries, payouts = load_data(("G1", "G2", "G3"), True, True)
    base = collect_subsets(races, entries, payouts)
    d9 = apply_variant(base, "D9")
    per_race = collapse_to_race(d9)
    per_race = tag_components(per_race)
    per_race["component_labels"] = make_component_labels(per_race)
    log.info("D9 races=%d hits=%d", len(per_race), int(per_race["hit"].sum()))

    summaries = []
    year_rows = []
    all_curves = []

    for method_name, bet_fn in METHODS.items():
        for initial in INITIAL_BANKROLLS:
            summary, curve = simulate(per_race, method_name, bet_fn, initial)
            summaries.append(summary)
            all_curves.extend(curve)
            curve_df = pd.DataFrame(curve)
            for yr_row in aggregate_by_year(curve_df, initial):
                year_rows.append({"method": method_name, "initial_bankroll": int(initial), **yr_row})

    summary_df = pd.DataFrame(summaries)
    year_df = pd.DataFrame(year_rows)
    curve_df = pd.DataFrame(all_curves)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(PROCESSED_DIR / "d9_bankroll_summary.csv",
                      index=False, encoding="utf-8-sig")
    year_df.to_csv(PROCESSED_DIR / "d9_bankroll_by_year.csv",
                   index=False, encoding="utf-8-sig")
    curve_df.to_csv(PROCESSED_DIR / "d9_equity_curve.csv",
                    index=False, encoding="utf-8-sig")
    log.info("wrote 3 CSVs (summary=%d, year=%d, curve=%d)",
             len(summary_df), len(year_df), len(curve_df))

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.2f}".format)

    print("\n=== SUMMARY (5 methods × 4 initial) ===")
    cols = ["method", "initial_bankroll", "final_bankroll", "total_profit",
            "total_return_pct", "max_drawdown_yen", "max_drawdown_pct",
            "max_losing_streak_races", "max_losing_streak_days",
            "ruin_flag", "skipped_races", "min_bankroll",
            "avg_bet_per_race", "max_bet_per_race", "total_bet", "total_payout", "roi"]
    print(summary_df[cols].to_string(index=False))

    print("\n=== BY YEAR (compact: profit + max_dd_pct per method/initial/year) ===")
    pivot = year_df.pivot_table(
        index=["method", "year"],
        columns="initial_bankroll",
        values=["profit", "max_drawdown_pct"],
        aggfunc="first",
    )
    print(pivot.to_string())


if __name__ == "__main__":
    main()

"""D9 の実運用 risk-control ルール比較 (初期 100K, year-shuffle Monte Carlo)。

6 ルールを比較:
  R0  fixed_1pct                            毎レース 初期×1%
  R1  fixed_2pct                            毎レース 初期×2%
  R2  1pct_then_2pct_after_profit           bankroll > 1.3×initial で 2%, 以下 1%
  R3  1pct_then_2pct_dd_reset               R2 + 全時 peak から DD20%以上で 1% に戻す
  R4  stop_after_30pct_drawdown             1%、ただし全時 peak から DD30%以上なら
                                            その年の残り bet 停止 (翌年再開)
  R5  aggressive_but_safe                   通常 1%。
                                            (a) bankroll >= 1.5×initial かつ
                                                直近20レースROIプラスなら 2%
                                            (b) DD >= 20% で 1% に戻す
                                            (c) DD >= 35% でその年停止

state は trial 単位に追跡し、NumPy ベクトル化で 10K trials を並列処理。

出力 (3 CSV):
  data/processed/d9_risk_control_summary.csv
  data/processed/d9_risk_control_order_tests.csv
  data/processed/d9_risk_control_trials_sample.csv

Usage:
    python scripts/analysis/risk_control_simulation.py
    python scripts/analysis/risk_control_simulation.py --trials 50000
"""
from __future__ import annotations

import argparse
import sys
import time
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
log = setup_logger("risk_control")

INITIAL = 100_000
RECENT_WINDOW = 20
RNG_SEED = 20260524

RULES = [
    "R0_fixed_1pct",
    "R1_fixed_2pct",
    "R2_1pct_then_2pct_after_profit",
    "R3_1pct_then_2pct_dd_reset",
    "R4_stop_after_30pct_drawdown",
    "R5_aggressive_but_safe",
]


# ---- core vectorized simulation per rule ----------------------------------

def simulate_rule(
    rule: str,
    orders: np.ndarray,         # (trials, N) shuffled race indices
    years_seq: np.ndarray,      # (trials, N) year per race-position per trial
    costs: np.ndarray,          # (N,)
    payouts: np.ndarray,        # (N,)
    hits: np.ndarray,           # (N,) bool
    initial: int,
) -> dict:
    n_trials, N = orders.shape
    bankroll = np.full(n_trials, float(initial))
    peak = bankroll.copy()
    min_bk = bankroll.copy()
    max_dd = np.zeros(n_trials)
    ruin = np.zeros(n_trials, dtype=bool)
    streak = np.zeros(n_trials, dtype=np.int32)
    max_streak = np.zeros(n_trials, dtype=np.int32)
    bet_paused = np.zeros(n_trials, dtype=bool)
    current_year = np.full(n_trials, -1, dtype=np.int32)
    races_bet = np.zeros(n_trials, dtype=np.int32)
    races_skipped = np.zeros(n_trials, dtype=np.int32)
    total_bet = np.zeros(n_trials)
    total_payout = np.zeros(n_trials)

    # R5 recent-window state
    profit_hist = np.zeros((n_trials, RECENT_WINDOW))
    bet_hist = np.zeros((n_trials, RECENT_WINDOW))
    recent_profit = np.zeros(n_trials)
    recent_bet = np.zeros(n_trials)

    init_f = float(initial)

    for i in range(N):
        # ---- year change → reset year-scoped state (bet_paused)
        year_now = years_seq[:, i]
        year_changed = year_now != current_year
        bet_paused = np.where(year_changed, False, bet_paused)
        current_year = np.where(year_changed, year_now, current_year)

        # ---- current drawdown
        dd_pct = np.where(peak > 0, (peak - bankroll) / peak, 0.0)

        # ---- per-rule pause triggers (before bet sizing)
        if rule == "R4_stop_after_30pct_drawdown":
            bet_paused = bet_paused | (dd_pct >= 0.30)
        elif rule == "R5_aggressive_but_safe":
            bet_paused = bet_paused | (dd_pct >= 0.35)

        # ---- determine intended bet
        if rule == "R0_fixed_1pct":
            intended = np.full(n_trials, init_f * 0.01)
        elif rule == "R1_fixed_2pct":
            intended = np.full(n_trials, init_f * 0.02)
        elif rule == "R2_1pct_then_2pct_after_profit":
            intended = np.where(bankroll > init_f * 1.3, init_f * 0.02, init_f * 0.01)
        elif rule == "R3_1pct_then_2pct_dd_reset":
            cond = (bankroll > init_f * 1.3) & (dd_pct < 0.20)
            intended = np.where(cond, init_f * 0.02, init_f * 0.01)
        elif rule == "R4_stop_after_30pct_drawdown":
            intended = np.full(n_trials, init_f * 0.01)
        elif rule == "R5_aggressive_but_safe":
            recent_roi = np.where(recent_bet > 0, recent_profit / recent_bet, 0.0)
            cond_aggr = (bankroll >= init_f * 1.5) & (recent_roi > 0)
            intended = np.where(cond_aggr, init_f * 0.02, init_f * 0.01)
            intended = np.where(dd_pct >= 0.20, init_f * 0.01, intended)  # throttle

        # ---- actual bet: needs not-paused AND affordable
        can_afford = intended <= bankroll
        not_paused = ~bet_paused
        wants_to_bet = not_paused & (intended > 0)
        can_bet = wants_to_bet & can_afford
        # ruin: wanted to bet but couldn't afford
        ruin |= wants_to_bet & ~can_afford
        actual_bet = np.where(can_bet, intended, 0.0)

        # ---- race outcome (vectorized lookups by shuffled order)
        race_idx = orders[:, i]
        race_cost = costs[race_idx]
        race_payout = payouts[race_idx]
        race_hit = hits[race_idx]

        safe_cost = np.where(race_cost > 0, race_cost, 1.0)
        profit_hit = actual_bet * (race_payout / safe_cost) - actual_bet
        profit_miss = -actual_bet
        profit = np.where(can_bet,
                          np.where(race_hit, profit_hit, profit_miss),
                          0.0)
        payout_amt = np.where(can_bet & race_hit, actual_bet * (race_payout / safe_cost), 0.0)

        bankroll = bankroll + profit
        peak = np.maximum(peak, bankroll)
        min_bk = np.minimum(min_bk, bankroll)
        new_dd = np.where(peak > 0, (peak - bankroll) / peak, 0.0)
        max_dd = np.maximum(max_dd, new_dd)
        ruin |= (bankroll <= 0)

        # streak (only over actual bets)
        is_loss = can_bet & (profit < 0)
        streak = np.where(is_loss, streak + 1, np.where(can_bet, 0, streak))
        max_streak = np.maximum(max_streak, streak)

        races_bet += can_bet.astype(np.int32)
        races_skipped += (~can_bet).astype(np.int32)
        total_bet += actual_bet
        total_payout += payout_amt

        # ---- update recent window (used by R5; harmless otherwise)
        pos = i % RECENT_WINDOW
        recent_profit -= profit_hist[:, pos]
        recent_bet -= bet_hist[:, pos]
        profit_hist[:, pos] = profit
        bet_hist[:, pos] = actual_bet
        recent_profit += profit
        recent_bet += actual_bet

    return {
        "final_bankroll": bankroll,
        "min_bankroll": min_bk,
        "max_drawdown_pct": max_dd,
        "ruin_flag": ruin,
        "max_losing_streak_races": max_streak,
        "races_bet": races_bet,
        "races_skipped": races_skipped,
        "total_bet": total_bet,
        "total_payout": total_payout,
    }


# ---- helpers --------------------------------------------------------------

def build_year_index(per_race, years_sorted):
    return {y: np.where(per_race["year"].values == y)[0] for y in years_sorted}


def random_year_orders(year_indices, year_list, n_trials, rng):
    n_years = len(year_list)
    N = sum(len(year_indices[y]) for y in year_list)
    orders = np.zeros((n_trials, N), dtype=np.int32)
    year_perms = np.zeros((n_trials, n_years), dtype=np.int32)
    year_list_arr = np.array(year_list)
    for t in range(n_trials):
        perm = rng.permutation(year_list_arr)
        year_perms[t] = perm
        orders[t] = np.concatenate([year_indices[y] for y in perm])
    return orders, year_perms


def build_year_seq(orders, race_year):
    """Map shuffled order to year sequence per trial."""
    return race_year[orders]


def year_roi_ranking(per_race):
    g = per_race.groupby("year").agg(cost=("cost", "sum"), payout=("payout", "sum"))
    g["roi"] = (g["payout"] - g["cost"]) / g["cost"].where(g["cost"] > 0, 1)
    return g.sort_values("roi").index.tolist()


def fixed_order_indices(year_indices, year_list_in_order):
    return np.concatenate([year_indices[y] for y in year_list_in_order])


# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10_000)
    ap.add_argument("--trials-sample-size", type=int, default=1_000)
    args = ap.parse_args()

    log.info("loading D9 per-race detail (JRA flat G1/G2/G3)")
    races, entries, payouts = load_data(("G1", "G2", "G3"), True, True)
    base = collect_subsets(races, entries, payouts)
    d9 = apply_variant(base, "D9")
    per_race = collapse_to_race(d9)
    per_race = tag_components(per_race)
    per_race["component_labels"] = make_component_labels(per_race)
    per_race = per_race.sort_values("race_date", kind="stable").reset_index(drop=True)
    log.info("D9 races=%d hits=%d initial=%d", len(per_race), int(per_race["hit"].sum()), INITIAL)

    costs = per_race["cost"].to_numpy(dtype=float)
    payouts_arr = per_race["payout"].to_numpy(dtype=float)
    hits = per_race["hit"].astype(bool).to_numpy()
    race_year = per_race["year"].to_numpy(dtype=np.int32)
    year_list = sorted(set(race_year.tolist()))
    year_indices = build_year_index(per_race, year_list)

    worst_first = year_roi_ranking(per_race)
    fixed_orders = {
        "worst_first": worst_first,
        "best_first": list(reversed(worst_first)),
        "chronological": sorted(year_list),
        "reverse_chronological": list(reversed(sorted(year_list))),
    }

    # ===== Monte Carlo =====
    log.info("generating %d random year-orderings...", args.trials)
    rng = np.random.default_rng(RNG_SEED)
    t0 = time.time()
    orders, year_perms = random_year_orders(year_indices, year_list, args.trials, rng)
    years_seq = build_year_seq(orders, race_year)
    log.info("  done in %.2fs (orders shape %s)", time.time() - t0, orders.shape)

    summary_rows = []
    sample_rows = []
    for rule in RULES:
        t0 = time.time()
        res = simulate_rule(rule, orders, years_seq, costs, payouts_arr, hits, INITIAL)
        elapsed = time.time() - t0
        ret = (res["final_bankroll"] - INITIAL) / INITIAL
        log.info("MC %s: ruin=%.2f%% median_return=%+.1f%% (%.1fs)",
                 rule, res["ruin_flag"].mean() * 100, np.median(ret) * 100, elapsed)

        summary_rows.append({
            "rule": rule,
            "initial_bankroll": INITIAL,
            "trials": int(args.trials),
            "ruin_rate": float(res["ruin_flag"].mean()),
            "median_final_bankroll": float(np.median(res["final_bankroll"])),
            "p05_final_bankroll": float(np.percentile(res["final_bankroll"], 5)),
            "p95_final_bankroll": float(np.percentile(res["final_bankroll"], 95)),
            "median_total_return_pct": float(np.median(ret)),
            "p05_total_return_pct": float(np.percentile(ret, 5)),
            "p95_total_return_pct": float(np.percentile(ret, 95)),
            "median_max_drawdown_pct": float(np.median(res["max_drawdown_pct"])),
            "p95_max_drawdown_pct": float(np.percentile(res["max_drawdown_pct"], 95)),
            "p99_max_drawdown_pct": float(np.percentile(res["max_drawdown_pct"], 99)),
            "median_races_bet": float(np.median(res["races_bet"])),
            "median_races_skipped": float(np.median(res["races_skipped"])),
            "median_min_bankroll": float(np.median(res["min_bankroll"])),
            "p05_min_bankroll": float(np.percentile(res["min_bankroll"], 5)),
        })

        # trials sample
        S = min(args.trials_sample_size, args.trials)
        for t in range(S):
            sample_rows.append({
                "rule": rule,
                "initial_bankroll": INITIAL,
                "trial_id": t,
                "final_bankroll": float(res["final_bankroll"][t]),
                "total_return_pct": float(ret[t]),
                "max_drawdown_pct": float(res["max_drawdown_pct"][t]),
                "min_bankroll": float(res["min_bankroll"][t]),
                "ruin_flag": bool(res["ruin_flag"][t]),
                "max_losing_streak_races": int(res["max_losing_streak_races"][t]),
                "races_bet": int(res["races_bet"][t]),
                "races_skipped": int(res["races_skipped"][t]),
                "total_bet": float(res["total_bet"][t]),
                "total_payout": float(res["total_payout"][t]),
            })

    summary_df = pd.DataFrame(summary_rows)

    # ===== Fixed-order tests (1 trial each, but vectorized as trials=1) =====
    fixed_rows = []
    for order_name, year_order in fixed_orders.items():
        order_idx = fixed_order_indices(year_indices, year_order)
        orders_1 = order_idx[None, :]
        years_seq_1 = race_year[order_idx][None, :]
        for rule in RULES:
            res = simulate_rule(rule, orders_1, years_seq_1, costs, payouts_arr, hits, INITIAL)
            fixed_rows.append({
                "rule": rule,
                "order_type": order_name,
                "initial_bankroll": INITIAL,
                "final_bankroll": float(res["final_bankroll"][0]),
                "total_return_pct": float((res["final_bankroll"][0] - INITIAL) / INITIAL),
                "max_drawdown_pct": float(res["max_drawdown_pct"][0]),
                "min_bankroll": float(res["min_bankroll"][0]),
                "ruin_flag": bool(res["ruin_flag"][0]),
                "races_bet": int(res["races_bet"][0]),
                "races_skipped": int(res["races_skipped"][0]),
                "max_losing_streak": int(res["max_losing_streak_races"][0]),
                "total_bet": float(res["total_bet"][0]),
                "total_payout": float(res["total_payout"][0]),
            })
    fixed_df = pd.DataFrame(fixed_rows)
    sample_df = pd.DataFrame(sample_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(PROCESSED_DIR / "d9_risk_control_summary.csv",
                      index=False, encoding="utf-8-sig")
    fixed_df.to_csv(PROCESSED_DIR / "d9_risk_control_order_tests.csv",
                    index=False, encoding="utf-8-sig")
    sample_df.to_csv(PROCESSED_DIR / "d9_risk_control_trials_sample.csv",
                     index=False, encoding="utf-8-sig")
    log.info("wrote 3 CSVs (summary=%d, fixed=%d, sample=%d)",
             len(summary_df), len(fixed_df), len(sample_df))

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== MONTE CARLO SUMMARY (10K trials, initial=100K) ===")
    print(summary_df.to_string(index=False))

    print("\n=== FIXED-ORDER STRESS TESTS (initial=100K) ===")
    # pivot for readability
    print(fixed_df.to_string(index=False))


if __name__ == "__main__":
    main()

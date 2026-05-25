"""D9 の year-shuffle Monte Carlo + fixed-order stress tests.

実順序 (2016→2025) は 2016 が爆発年、2023 が崩壊年。順序の偶然で破産を免れた
可能性がある。本スクリプトは:

  1) 年単位で順序をランダム化した 10,000 trials を実行 (年内順序は保持)
  2) worst_first / best_first / chronological / reverse_chronological の
     固定4順序でも実行
  3) 各 method × initial について 破産率 / median return / p05/p95 を集計

methods: A (100yen) / B (1%) / C (2%) / D (5%)
initial: 50,000 / 100,000 / 300,000

ruin: bankroll < 必要 bet になったレースが1件でも発生したら ruin_flag=True

出力:
  data/processed/d9_monte_carlo_summary.csv   12 rows (method × initial)
  data/processed/d9_order_stress_tests.csv    48 rows (4 orders × 12 combos)
  data/processed/d9_monte_carlo_trials_sample.csv  最初の 1,000 trials × 12 combos

Usage:
    python scripts/analysis/bankroll_monte_carlo.py
    python scripts/analysis/bankroll_monte_carlo.py --trials 50000
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
log = setup_logger("mc")

INITIAL_BANKROLLS = [50_000, 100_000, 300_000]
METHODS = ["A_fixed_100yen", "B_fixed_bankroll_1pct",
           "C_fixed_bankroll_2pct", "D_fixed_bankroll_5pct"]
METHOD_PCT = {"B_fixed_bankroll_1pct": 0.01,
              "C_fixed_bankroll_2pct": 0.02,
              "D_fixed_bankroll_5pct": 0.05}
RNG_SEED = 20260524


# ---- pre-compute method arrays --------------------------------------------

def compute_bet_profit(costs, payouts, hits, method, initial):
    """Return (bet[N], profit[N]) arrays for this method × initial."""
    N = len(costs)
    if method == "A_fixed_100yen":
        bet = costs.astype(float)
    else:
        pct = METHOD_PCT[method]
        bet = np.full(N, float(initial) * pct)
    # payout per yen bet (when hit) = raw_payout / cost
    safe_cost = np.where(costs > 0, costs, 1)
    profit_if_hit = bet * (payouts / safe_cost) - bet
    profit_if_miss = -bet
    profit = np.where(hits, profit_if_hit, profit_if_miss)
    return bet, profit


# ---- vectorized simulation across trials -----------------------------------

def vectorized_simulate(bet_arr_2d, profit_arr_2d, initial):
    """Run `trials` independent simulations in parallel.

    bet_arr_2d / profit_arr_2d : shape (trials, N) in already-shuffled order.

    Returns per-trial arrays:
      final_bankroll, max_dd_pct, min_bankroll, ruin, max_losing_streak
    """
    trials, N = bet_arr_2d.shape
    bankroll = np.full(trials, float(initial))
    peak = bankroll.copy()
    min_bk = bankroll.copy()
    max_dd = np.zeros(trials)
    ruin = np.zeros(trials, dtype=bool)
    streak = np.zeros(trials, dtype=np.int32)
    max_streak = np.zeros(trials, dtype=np.int32)

    for i in range(N):
        b = bet_arr_2d[:, i]
        p = profit_arr_2d[:, i]
        can_bet = b <= bankroll
        ruin |= ~can_bet
        bankroll = np.where(can_bet, bankroll + p, bankroll)
        peak = np.maximum(peak, bankroll)
        min_bk = np.minimum(min_bk, bankroll)
        dd = np.where(peak > 0, (peak - bankroll) / peak, 0.0)
        max_dd = np.maximum(max_dd, dd)
        # streak: only count losses on races we actually bet
        is_loss = can_bet & (p <= 0)
        streak = np.where(is_loss, streak + 1, np.where(can_bet, 0, streak))
        max_streak = np.maximum(max_streak, streak)
        ruin |= (bankroll <= 0)
    return bankroll, max_dd, min_bk, ruin, max_streak


# ---- order generation ------------------------------------------------------

def build_year_index(per_race, years_sorted):
    """{year -> np.array of race indices (in chronological within-year order)}."""
    return {y: np.where(per_race["year"].values == y)[0] for y in years_sorted}


def random_year_orders(year_indices, year_list, n_trials, rng):
    """Return (orders [trials, N], year_perms [trials, n_years])."""
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


def fixed_order_indices(year_indices, year_list_in_order):
    """For a specific year ordering, return the concatenated race indices."""
    return np.concatenate([year_indices[y] for y in year_list_in_order])


def year_roi_ranking(per_race):
    """Return years sorted ascending by year-level ROI (worst first)."""
    g = per_race.groupby("year").agg(
        cost=("cost", "sum"), payout=("payout", "sum")
    )
    g["roi"] = (g["payout"] - g["cost"]) / g["cost"].where(g["cost"] > 0, 1)
    return g.sort_values("roi").index.tolist()


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
    log.info("D9 races=%d hits=%d", len(per_race), int(per_race["hit"].sum()))

    costs = per_race["cost"].to_numpy(dtype=float)
    payouts_arr = per_race["payout"].to_numpy(dtype=float)
    hits = per_race["hit"].astype(bool).to_numpy()
    years = per_race["year"].to_numpy(dtype=int)
    year_list = sorted(set(years.tolist()))
    year_indices = build_year_index(per_race, year_list)

    worst_first = year_roi_ranking(per_race)            # ascending ROI
    best_first = list(reversed(worst_first))            # descending ROI
    chronological = sorted(year_list)
    reverse_chrono = list(reversed(chronological))
    fixed_orders = {
        "worst_first": worst_first,
        "best_first": best_first,
        "chronological": chronological,
        "reverse_chronological": reverse_chrono,
    }
    log.info("year ROI ranking (worst→best): %s", worst_first)
    worst_year = worst_first[0]

    # ===== Monte Carlo =====
    rng = np.random.default_rng(RNG_SEED)
    log.info("generating %d random year-orderings...", args.trials)
    t0 = time.time()
    orders, year_perms = random_year_orders(year_indices, year_list, args.trials, rng)
    log.info("  done in %.2fs", time.time() - t0)

    # position of worst year in each shuffle
    worst_year_pos = np.argmax(year_perms == worst_year, axis=1)

    summary_rows = []
    sample_rows = []

    for method in METHODS:
        for initial in INITIAL_BANKROLLS:
            bet_arr, profit_arr = compute_bet_profit(costs, payouts_arr, hits, method, initial)
            # shuffled per trial
            bet_2d = bet_arr[orders]
            profit_2d = profit_arr[orders]

            t0 = time.time()
            final_bk, max_dd, min_bk, ruin, max_streak = vectorized_simulate(
                bet_2d, profit_2d, initial
            )
            total_return = (final_bk - initial) / initial
            log.info("MC method=%s initial=%d ruin=%.2f%% (%.1fs)",
                     method, initial, ruin.mean() * 100, time.time() - t0)

            summary_rows.append({
                "method": method,
                "initial_bankroll": int(initial),
                "trials": int(args.trials),
                "ruin_rate": float(ruin.mean()),
                "median_final_bankroll": float(np.median(final_bk)),
                "p05_final_bankroll": float(np.percentile(final_bk, 5)),
                "p95_final_bankroll": float(np.percentile(final_bk, 95)),
                "median_max_drawdown_pct": float(np.median(max_dd)),
                "p95_max_drawdown_pct": float(np.percentile(max_dd, 95)),
                "p99_max_drawdown_pct": float(np.percentile(max_dd, 99)),
                "median_total_return_pct": float(np.median(total_return)),
                "p05_total_return_pct": float(np.percentile(total_return, 5)),
                "p95_total_return_pct": float(np.percentile(total_return, 95)),
                "median_max_losing_streak": float(np.median(max_streak)),
                "p95_max_losing_streak": float(np.percentile(max_streak, 95)),
                "min_bankroll_overall": float(min_bk.min()),
            })

            # save first N trials as sample
            S = min(args.trials_sample_size, args.trials)
            for t in range(S):
                sample_rows.append({
                    "method": method,
                    "initial_bankroll": int(initial),
                    "trial_id": t,
                    "final_bankroll": float(final_bk[t]),
                    "total_return_pct": float(total_return[t]),
                    "max_drawdown_pct": float(max_dd[t]),
                    "min_bankroll": float(min_bk[t]),
                    "ruin_flag": bool(ruin[t]),
                    "longest_losing_streak_races": int(max_streak[t]),
                    "worst_year_position": int(worst_year_pos[t]),
                })

    summary_df = pd.DataFrame(summary_rows)
    sample_df = pd.DataFrame(sample_rows)

    # ===== Fixed-order stress tests =====
    fixed_rows = []
    for order_name, year_order in fixed_orders.items():
        order_idx = fixed_order_indices(year_indices, year_order)
        for method in METHODS:
            for initial in INITIAL_BANKROLLS:
                bet_arr, profit_arr = compute_bet_profit(costs, payouts_arr, hits, method, initial)
                # 1-trial simulate
                final_bk, max_dd, min_bk, ruin, max_streak = vectorized_simulate(
                    bet_arr[order_idx][None, :], profit_arr[order_idx][None, :], initial
                )
                fixed_rows.append({
                    "order_type": order_name,
                    "method": method,
                    "initial_bankroll": int(initial),
                    "final_bankroll": float(final_bk[0]),
                    "total_return_pct": float((final_bk[0] - initial) / initial),
                    "max_drawdown_pct": float(max_dd[0]),
                    "min_bankroll": float(min_bk[0]),
                    "ruin_flag": bool(ruin[0]),
                    "max_losing_streak": int(max_streak[0]),
                })
    fixed_df = pd.DataFrame(fixed_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(PROCESSED_DIR / "d9_monte_carlo_summary.csv",
                      index=False, encoding="utf-8-sig")
    fixed_df.to_csv(PROCESSED_DIR / "d9_order_stress_tests.csv",
                    index=False, encoding="utf-8-sig")
    sample_df.to_csv(PROCESSED_DIR / "d9_monte_carlo_trials_sample.csv",
                     index=False, encoding="utf-8-sig")
    log.info("wrote 3 CSVs (mc=%d, fixed=%d, sample=%d)",
             len(summary_df), len(fixed_df), len(sample_df))

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== MONTE CARLO SUMMARY (10K trials) ===")
    print(summary_df.to_string(index=False))

    print("\n=== FIXED-ORDER STRESS TESTS ===")
    print(fixed_df.to_string(index=False))

    print("\n=== year ROI ranking (worst-first) ===")
    g = per_race.groupby("year").agg(cost=("cost", "sum"), payout=("payout", "sum"),
                                     races=("race_id", "count"), hits=("hit", "sum"))
    g["roi"] = (g["payout"] - g["cost"]) / g["cost"]
    print(g.sort_values("roi").to_string())


if __name__ == "__main__":
    main()

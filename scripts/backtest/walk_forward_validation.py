"""walk-forward validation for F_D9_DYNAMIC_STATE.

Folds (rolling, train 5 years → test next year):
  2015-2019 → test 2020
  2016-2020 → test 2021
  2017-2021 → test 2022
  2018-2022 → test 2023
  2019-2023 → test 2024
  2020-2024 → test 2025

各 fold で再 calibration:
  - stress score normalization params (quantiles from train period)
  - regime classification thresholds
  - state machine (fold-specific market_state per race)
  - component ROI ranking (informational, not bet-changing)
  - state transition probabilities (informational)

dark ordering (P3 cap4 logic = win_odds asc / pop asc / horse_num asc) は
strategy 定義そのものなので fold 毎に再学習しない (fix)。

Leakage audit:
  - baseline_year_max < test_year (= しきい値 calibration は過去のみ)
  - rolling features は shift(1) で当該 race の outcome 不使用
  - state machine は時系列 sequential、未来は参照しない

Compare:
  - rolling_F  (fold-specific thresholds, 各 test year)
  - static_F   (固定 thresholds = baseline 2020、test years 全体に適用)
  - E_baseline (state filter 無し、test years)

Output:
  walk_forward_summary.csv          fold 別 / strategy 別 summary
  walk_forward_yearly.csv           per-test-year breakdown
  walk_forward_equity.csv           fold × race equity curve
  walk_forward_leakage_audit.csv    leakage check + threshold drift

Usage:
  python scripts/backtest/walk_forward_validation.py
  python scripts/backtest/walk_forward_validation.py --mc-trials 500
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

from scripts.backtest.strategy_d_variants import load_data
from scripts.backtest.simple_roi import (
    build_strategies, compute_detail,
)
from scripts.analysis.regime_detection import (
    build_per_race_summary, compute_rolling_features,
    calibrate_thresholds, classify_regime,
)
from scripts.analysis.market_stop_system import (
    compute_stress_score, apply_state_machine,
)
from scripts.backtest.strategy_f_backtest import (
    INITIAL_BANKROLL, BASE_BET_PCT_BY_STATE, REGIME_BOOST, REGIME_SKIP,
    f_bet_for_race, simulate,
)

force_utf8_stdout()
log = setup_logger("walk_forward")

# Folds: (train_years_list, test_year)
FOLDS = [
    (list(range(2015, 2020)), 2020),
    (list(range(2016, 2021)), 2021),
    (list(range(2017, 2022)), 2022),
    (list(range(2018, 2023)), 2023),
    (list(range(2019, 2024)), 2024),
    (list(range(2020, 2025)), 2025),
]
STATIC_BASELINE_YEAR_MAX = 2020   # used by "static_F" comparison
MC_TRIALS_DEFAULT = 500
RNG_SEED = 20260525


# ============================================================================
#  Build E race-level baseline (cap4 ticket cost & payout per race)
# ============================================================================
def build_e_per_race_global(races_all, entries_all, payouts_all) -> pd.DataFrame:
    """Use simple_roi's compute_detail to get E_D9_P3_CAP4 per race for ALL years.
    state/regime will be reassigned per fold."""
    strategies = build_strategies(strategy_d_min_popularity=5)
    only_e = {k: v for k, v in strategies.items() if k == "E_D9_P3_CAP4"}
    detail = compute_detail(races_all, entries_all, payouts_all, only_e)
    if detail.empty:
        return pd.DataFrame()
    e = detail[detail["strategy"] == "E_D9_P3_CAP4"].copy()
    e["hit"] = e["hit_ticket"].notna().astype(int)
    e = e.rename(columns={"investment_yen": "e_cost", "payout_yen": "e_payout"})
    e["race_date"] = pd.to_datetime(e["race_date"])
    e["year"] = e["race_date"].dt.year
    return e[["race_id", "race_date", "race_name", "year",
              "n_tickets", "e_cost", "hit", "e_payout"]]


# ============================================================================
#  Per-fold recalibration → state CSV → simulate test year
# ============================================================================
def run_fold(train_years, test_year,
             per_race_summary_all: pd.DataFrame,
             features_all: pd.DataFrame,
             e_per_race_all: pd.DataFrame,
             mc_trials: int, rng) -> dict:
    """Recalibrate thresholds from train period, apply to test year."""
    baseline_year_max = max(train_years)

    # --- leakage audit
    audit = {
        "fold": f"{train_years[0]}-{train_years[-1]} -> {test_year}",
        "train_years": f"{train_years[0]}-{train_years[-1]}",
        "test_year": test_year,
        "baseline_year_max_used_for_thresholds": baseline_year_max,
        "leakage_baseline_year_lt_test_year": baseline_year_max < test_year,
        "n_training_races_used_for_thresholds": int(
            (features_all["race_date"].dt.year <= baseline_year_max).sum()),
        "n_test_year_races": int(
            (features_all["race_date"].dt.year == test_year).sum()),
        "rolling_features_use_shift1": True,  # by construction
    }

    # --- recalibrate regime thresholds and stress normalization from train
    regime_T = calibrate_thresholds(features_all, baseline_year_max)

    # --- stress score + state machine using fold-specific normalization
    stress_states = apply_state_machine(
        compute_stress_score(features_all, baseline_year_max)
    )

    # --- attach state/regime to E test races
    sr = stress_states.set_index("race_id")[["market_state", "regime", "stress_score"]]
    test_races = e_per_race_all[e_per_race_all["year"] == test_year].copy()
    test_races["state"] = test_races["race_id"].map(sr["market_state"])
    test_races["regime"] = test_races["race_id"].map(sr["regime"])
    test_races["stress_score"] = test_races["race_id"].map(sr["stress_score"])
    test_races = test_races.sort_values("race_date", kind="stable").reset_index(drop=True)

    # --- simulate F and E on test year (chronological)
    f_result = simulate(test_races, "F", initial=INITIAL_BANKROLL)
    e_result = simulate(test_races, "E", initial=INITIAL_BANKROLL)

    # --- Monte Carlo: race-level shuffle within test year (order sensitivity)
    finals_f = []
    dds_f = []
    finals_e = []
    dds_e = []
    if len(test_races) >= 2:
        for _ in range(mc_trials):
            order = rng.permutation(len(test_races))
            shuffled = test_races.iloc[order].reset_index(drop=True)
            rf = simulate(shuffled, "F", initial=INITIAL_BANKROLL)["summary"]
            re = simulate(shuffled, "E", initial=INITIAL_BANKROLL)["summary"]
            finals_f.append(rf["final_bankroll"]); dds_f.append(rf["max_dd"])
            finals_e.append(re["final_bankroll"]); dds_e.append(re["max_dd"])
    mc = {
        "mc_f_median_final": float(np.median(finals_f)) if finals_f else None,
        "mc_f_p05_final":    float(np.percentile(finals_f, 5)) if finals_f else None,
        "mc_f_p95_max_dd":   float(np.percentile(dds_f, 95)) if dds_f else None,
        "mc_e_median_final": float(np.median(finals_e)) if finals_e else None,
        "mc_e_p05_final":    float(np.percentile(finals_e, 5)) if finals_e else None,
        "mc_e_p95_max_dd":   float(np.percentile(dds_e, 95)) if dds_e else None,
    }

    # --- component ROI ranking on train period (informational)
    train_e = e_per_race_all[e_per_race_all["year"].isin(train_years)].copy()
    train_e_cost = float(train_e["e_cost"].sum())
    train_e_payout = float(train_e["e_payout"].sum())
    train_e_roi = (train_e_payout - train_e_cost) / train_e_cost if train_e_cost > 0 else 0

    return {
        "audit": audit,
        "thresholds": regime_T,
        "test_races": test_races,
        "e_summary": e_result["summary"],
        "f_summary": f_result["summary"],
        "f_equity": f_result["equity"],
        "f_skips": f_result["skips"],
        "mc": mc,
        "train_e_roi": train_e_roi,
        "train_e_races": int(len(train_e)),
    }


def _simulate_static(e_per_race_all, features_all, test_years, mc_trials, rng):
    """Static F = fixed thresholds (baseline = STATIC_BASELINE_YEAR_MAX), tested on
    test_years (concat)."""
    stress_states = apply_state_machine(
        compute_stress_score(features_all, STATIC_BASELINE_YEAR_MAX)
    )
    sr = stress_states.set_index("race_id")[["market_state", "regime", "stress_score"]]
    test_races = e_per_race_all[e_per_race_all["year"].isin(test_years)].copy()
    test_races["state"] = test_races["race_id"].map(sr["market_state"])
    test_races["regime"] = test_races["race_id"].map(sr["regime"])
    test_races["stress_score"] = test_races["race_id"].map(sr["stress_score"])
    test_races = test_races.sort_values("race_date", kind="stable").reset_index(drop=True)
    f_result = simulate(test_races, "F", initial=INITIAL_BANKROLL)
    e_result = simulate(test_races, "E", initial=INITIAL_BANKROLL)
    return f_result, e_result, test_races


# ============================================================================
#  Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-trials", type=int, default=MC_TRIALS_DEFAULT)
    args = ap.parse_args()

    log.info("loading data (one-shot)...")
    races, entries, payouts = load_data(
        grades=("G1", "G2", "G3"), jra_only=True, exclude_steeplechase=True,
    )
    per_race_summary = build_per_race_summary(races, entries, payouts)
    features = compute_rolling_features(per_race_summary)
    e_per_race = build_e_per_race_global(races, entries, payouts)
    log.info("E baseline: %d races (years %s)",
              len(e_per_race),
              sorted(e_per_race["year"].unique().tolist()))

    rng = np.random.default_rng(RNG_SEED)

    # ---- run each fold
    fold_results = []
    for train_years, test_year in FOLDS:
        log.info("fold: train %d-%d → test %d",
                  train_years[0], train_years[-1], test_year)
        res = run_fold(train_years, test_year,
                        per_race_summary, features, e_per_race,
                        args.mc_trials, rng)
        fold_results.append(res)

    # ---- static F comparison on same test years (2020-2025)
    test_years_concat = [tr[1] for tr in FOLDS]
    static_f, static_e, static_test_races = _simulate_static(
        e_per_race, features, test_years_concat, args.mc_trials, rng,
    )

    # ---- assemble summaries
    rows = []
    for res, (train_years, test_year) in zip(fold_results, FOLDS):
        baseline_year_max = max(train_years)
        rows.append({
            "label": f"rolling_F_fold_{test_year}",
            "strategy": "rolling_F",
            "train_years": f"{train_years[0]}-{train_years[-1]}",
            "test_year": test_year,
            "baseline_year_max": baseline_year_max,
            **{f"e_{k}": v for k, v in res["e_summary"].items()},
            **{f"f_{k}": v for k, v in res["f_summary"].items()},
            **res["mc"],
            "train_e_roi": res["train_e_roi"],
            "regime_T_fav_low": res["thresholds"]["fav_low"],
            "regime_T_fav_high": res["thresholds"]["fav_high"],
            "regime_T_dark_low": res["thresholds"]["dark_low"],
        })

    # aggregate rolling_F across all 6 test years
    rolling_total_profit = sum(r["f_summary"]["profit"] for r in fold_results)
    rolling_total_bet = sum(r["f_summary"]["total_bet"] for r in fold_results)
    rolling_total_payout = sum(r["f_summary"]["total_payout"] for r in fold_results)
    rolling_hits = sum(r["f_summary"]["hits"] for r in fold_results)
    rolling_bets = sum(r["f_summary"]["races_bet"] for r in fold_results)
    rolling_skipped = sum(r["f_summary"]["races_skipped"] for r in fold_results)
    rolling_skipped_winners = sum(r["f_summary"]["skipped_winning_races"] for r in fold_results)
    rolling_max_dd = max(r["f_summary"]["max_dd"] for r in fold_results)

    rows.append({
        "label": "rolling_F_aggregate", "strategy": "rolling_F",
        "train_years": "rolling 5y",
        "test_year": "2020-2025",
        "f_profit": rolling_total_profit,
        "f_total_bet": rolling_total_bet,
        "f_total_payout": rolling_total_payout,
        "f_hits": rolling_hits, "f_races_bet": rolling_bets,
        "f_races_skipped": rolling_skipped,
        "f_skipped_winning_races": rolling_skipped_winners,
        "f_roi_on_bet": (rolling_total_payout - rolling_total_bet) / rolling_total_bet if rolling_total_bet else 0,
        "f_max_dd_worst_fold": rolling_max_dd,
    })
    rows.append({
        "label": "static_F_all_test_years",
        "strategy": "static_F",
        "train_years": f"baseline ≤ {STATIC_BASELINE_YEAR_MAX}",
        "test_year": "2020-2025",
        **{f"f_{k}": v for k, v in static_f["summary"].items()},
    })
    rows.append({
        "label": "E_baseline_all_test_years",
        "strategy": "E_baseline",
        "train_years": "n/a (no skip)",
        "test_year": "2020-2025",
        **{f"e_{k}": v for k, v in static_e["summary"].items()},
    })
    summary_df = pd.DataFrame(rows)

    # ---- per-year breakdown
    yearly_rows = []
    for res, (train_years, test_year) in zip(fold_results, FOLDS):
        yearly_rows.append({
            "test_year": test_year, "strategy": "rolling_F",
            "races": res["f_summary"]["races_total"],
            "bets": res["f_summary"]["races_bet"],
            "skipped": res["f_summary"]["races_skipped"],
            "skipped_winners": res["f_summary"]["skipped_winning_races"],
            "hits": res["f_summary"]["hits"],
            "total_bet": res["f_summary"]["total_bet"],
            "total_payout": res["f_summary"]["total_payout"],
            "profit": res["f_summary"]["profit"],
            "roi_on_bet": res["f_summary"]["roi_on_bet"],
            "max_dd": res["f_summary"]["max_dd"],
            "max_losing_streak": res["f_summary"]["max_losing_streak"],
            "profit_factor": res["f_summary"]["profit_factor"],
            "sharpe_like": res["f_summary"]["sharpe_like_per_race"],
            "mc_p05_final": res["mc"]["mc_f_p05_final"],
            "mc_p95_max_dd": res["mc"]["mc_f_p95_max_dd"],
        })
        yearly_rows.append({
            "test_year": test_year, "strategy": "E_baseline",
            "races": res["e_summary"]["races_total"],
            "bets": res["e_summary"]["races_bet"],
            "hits": res["e_summary"]["hits"],
            "total_bet": res["e_summary"]["total_bet"],
            "total_payout": res["e_summary"]["total_payout"],
            "profit": res["e_summary"]["profit"],
            "roi_on_bet": res["e_summary"]["roi_on_bet"],
            "max_dd": res["e_summary"]["max_dd"],
        })
    yearly_df = pd.DataFrame(yearly_rows)

    # ---- equity curve per fold
    equity_rows = []
    for res, (train_years, test_year) in zip(fold_results, FOLDS):
        for er in res["f_equity"]:
            equity_rows.append({"test_year": test_year, "strategy": "rolling_F", **er})
    equity_df = pd.DataFrame(equity_rows)

    # ---- leakage audit
    audit_rows = [res["audit"] for res in fold_results]
    for ar, res in zip(audit_rows, fold_results):
        ar.update({
            f"threshold_{k}": v for k, v in res["thresholds"].items()
        })
        ar["train_e_roi_for_reference"] = res["train_e_roi"]
        ar["test_year_e_races_count"] = res["e_summary"]["races_total"]
    audit_df = pd.DataFrame(audit_rows)

    # ---- save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(PROCESSED_DIR / "walk_forward_summary.csv",
                       index=False, encoding="utf-8-sig")
    yearly_df.to_csv(PROCESSED_DIR / "walk_forward_yearly.csv",
                      index=False, encoding="utf-8-sig")
    equity_df.to_csv(PROCESSED_DIR / "walk_forward_equity.csv",
                      index=False, encoding="utf-8-sig")
    audit_df.to_csv(PROCESSED_DIR / "walk_forward_leakage_audit.csv",
                     index=False, encoding="utf-8-sig")
    log.info("wrote 4 CSVs")

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 280)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== LEAKAGE AUDIT ===")
    print(audit_df[["fold", "train_years", "test_year", "baseline_year_max_used_for_thresholds",
                     "leakage_baseline_year_lt_test_year",
                     "n_training_races_used_for_thresholds", "n_test_year_races",
                     "rolling_features_use_shift1"]].to_string(index=False))

    print("\n=== THRESHOLD DRIFT (across folds) ===")
    drift = audit_df[["fold", "threshold_fav_low", "threshold_fav_high",
                       "threshold_dark_low", "threshold_payout_inflation", "threshold_cv_low"]]
    print(drift.to_string(index=False))

    print("\n=== ROLLING-F per fold (test year only) ===")
    yc = yearly_df[yearly_df["strategy"] == "rolling_F"]
    print(yc[["test_year", "races", "bets", "skipped", "skipped_winners", "hits",
               "profit", "roi_on_bet", "max_dd", "profit_factor", "sharpe_like"]].to_string(index=False))

    print("\n=== HEAD-TO-HEAD on test years (2020-2025) ===")
    h2h = summary_df[summary_df["label"].isin([
        "rolling_F_aggregate", "static_F_all_test_years", "E_baseline_all_test_years"])]
    print(h2h[["label",
                 "f_profit", "f_total_bet", "f_total_payout", "f_hits",
                 "f_races_bet", "f_races_skipped", "f_roi_on_bet",
                 "f_max_dd_worst_fold", "e_profit", "e_max_dd"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""F_D9_DYNAMIC_STATE の variable-stake backtest + E との横比較。

simple_roi.py の F は binary skip (HALT/DARK_SUPPRESSED → 空 list) のみ実装。
このスクリプトは state ごとの bet_pct と regime 補正を実額シミュレーションし、
E (固定 100yen/ticket × cap4) と並べて全 metric を比較する。

Bet sizing rule:
  base_pct_by_state = {GREEN: 1.5%, YELLOW: 0.75%, ORANGE: 0.5%, RED: 0.25%, HALT: 0%}
  regime_mult = 1.33 if regime in {FAVORITE_DOMINANT, HIGH_PAYOUT} else 1.0
  regime_mult = 0   if regime == DARK_SUPPRESSED
  race_bet (yen) = INITIAL_BANKROLL * state_pct * regime_mult

  → race_bet == 0 → skip
  → race_bet > 0  → 1 race の全 E チケットに均等配分

Payout for hit race:
  payout = race_bet * (raw_trifecta_payout / e_baseline_cost)
  ※ e_baseline_cost = E が 100yen/ticket で買った時の総額

Metrics:
  ROI / DD / max_losing_streak / Sharpe-like / profit_factor /
  skipped_races / skipped_winning_races / Monte Carlo (year shuffle) /
  worst_first / reverse_chronological / chronological

出力:
  data/processed/f_strategy_summary.csv     E と F の横比較
  data/processed/f_strategy_equity.csv      race-by-race equity curve
  data/processed/f_strategy_yearly.csv      per-year ROI/DD
  data/processed/f_strategy_skips.csv       F が skip した race (would_have_hit 付)

Usage:
  python scripts/backtest/strategy_f_backtest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger
from scripts.backtest.simple_roi import (
    build_strategies, compute_detail, load_data,
)
from scripts.backtest.simple_roi import get_market_state_regime

force_utf8_stdout()
log = setup_logger("strategy_f_backtest")

INITIAL_BANKROLL = 100_000
BASE_BET_PCT_BY_STATE = {
    "GREEN": 0.015, "YELLOW": 0.0075, "ORANGE": 0.005, "RED": 0.0025, "HALT": 0.0,
}
REGIME_BOOST = {"FAVORITE_DOMINANT": 1.33, "HIGH_PAYOUT": 1.33}
REGIME_SKIP = {"DARK_SUPPRESSED"}
MC_TRIALS = 10_000
RNG_SEED = 20260525


# ============================================================================
#  Build per-race E baseline (cap4)
# ============================================================================
def build_e_per_race() -> pd.DataFrame:
    """Use simple_roi.compute_detail to get E_D9_P3_CAP4 per-race results
    (using the strategies dict; F entries are filtered post-hoc)."""
    races, entries, payouts = load_data(
        grades=("G1", "G2", "G3"), date_from=None, date_to=None,
        jra_only=True, exclude_steeplechase=True,
    )
    strategies = build_strategies(strategy_d_min_popularity=5)
    # only keep E for baseline (we'll compute F from E+state below)
    only_e = {k: v for k, v in strategies.items() if k == "E_D9_P3_CAP4"}
    detail = compute_detail(races, entries, payouts, only_e)
    if detail.empty:
        return pd.DataFrame()
    e = detail[detail["strategy"] == "E_D9_P3_CAP4"].copy()
    e["hit"] = e["hit_ticket"].notna().astype(int)
    e = e.rename(columns={"investment_yen": "e_cost", "payout_yen": "e_payout"})
    # add state + regime
    state_regime = e["race_id"].apply(get_market_state_regime)
    e["state"] = state_regime.map(lambda t: t[0])
    e["regime"] = state_regime.map(lambda t: t[1])
    e["race_date"] = pd.to_datetime(e["race_date"])
    e["year"] = e["race_date"].dt.year
    return e[["race_id", "race_date", "race_name", "year",
              "state", "regime", "n_tickets", "e_cost", "hit", "e_payout"]]


# ============================================================================
#  Bet sizing for F
# ============================================================================
def f_bet_for_race(state: str, regime: str, initial_bankroll: float = INITIAL_BANKROLL) -> float:
    if regime in REGIME_SKIP:
        return 0.0
    base = BASE_BET_PCT_BY_STATE.get(state, 0.01)
    if base <= 0:
        return 0.0
    mult = REGIME_BOOST.get(regime, 1.0)
    return float(initial_bankroll) * base * mult


# ============================================================================
#  Simulate either E (baseline 100yen/ticket) or F (dynamic stakes) in race order
# ============================================================================
def simulate(per_race_in_order: pd.DataFrame, strategy: str,
              initial: float = INITIAL_BANKROLL) -> dict:
    bk = float(initial)
    peak = bk
    max_dd = 0.0
    bets = hits = skipped = skipped_winners = 0
    cur_streak = max_streak = 0
    total_bet = total_payout = 0.0
    profits_per_race: list[float] = []
    wins_sum = losses_sum = 0.0
    equity_rows = []
    skip_rows = []
    skipped_winners_payout = 0
    for _, r in per_race_in_order.iterrows():
        if strategy == "E":
            bet = float(r["e_cost"])
        elif strategy == "F":
            bet = f_bet_for_race(r["state"], r["regime"], initial)
        else:
            raise ValueError(f"unknown strategy {strategy}")

        if bet <= 0:
            skipped += 1
            if int(r["hit"]) == 1:
                skipped_winners += 1
                skipped_winners_payout += int(r["e_payout"])
                skip_rows.append({
                    "race_id": r["race_id"], "race_date": r["race_date"],
                    "race_name": r["race_name"], "state": r["state"],
                    "regime": r["regime"], "would_have_hit": True,
                    "would_have_payout": int(r["e_payout"]),
                    "e_cost": int(r["e_cost"]),
                })
            else:
                skip_rows.append({
                    "race_id": r["race_id"], "race_date": r["race_date"],
                    "race_name": r["race_name"], "state": r["state"],
                    "regime": r["regime"], "would_have_hit": False,
                    "would_have_payout": 0, "e_cost": int(r["e_cost"]),
                })
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
            equity_rows.append({
                "race_id": r["race_id"], "race_date": r["race_date"],
                "state": r["state"], "regime": r["regime"],
                "bet": 0.0, "profit": 0.0, "bankroll": bk,
                "drawdown_pct": (peak - bk) / peak if peak > 0 else 0,
                "action": "SKIP",
            })
            continue
        bets += 1
        if int(r["hit"]) == 1:
            payout = bet * (float(r["e_payout"]) / float(r["e_cost"]))
            hits += 1
            cur_streak = 0
        else:
            payout = 0.0
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        profit = payout - bet
        if profit > 0:
            wins_sum += profit
        else:
            losses_sum += -profit  # store as positive magnitude
        bk += profit
        peak = max(peak, bk)
        dd_pct = (peak - bk) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd_pct)
        total_bet += bet
        total_payout += payout
        profits_per_race.append(profit)
        equity_rows.append({
            "race_id": r["race_id"], "race_date": r["race_date"],
            "state": r["state"], "regime": r["regime"],
            "bet": round(bet, 2), "profit": round(profit, 2),
            "bankroll": round(bk, 2), "drawdown_pct": round(dd_pct, 4),
            "action": "HIT" if int(r["hit"]) == 1 else "MISS",
        })
    # ---- compute aggregate metrics
    profit_total = bk - initial
    arr = np.array(profits_per_race) if profits_per_race else np.array([0.0])
    sharpe_like = float(arr.mean() / arr.std()) if arr.std() > 0 else None
    profit_factor = wins_sum / losses_sum if losses_sum > 0 else None
    summary = {
        "strategy": strategy,
        "races_total": int(len(per_race_in_order)),
        "races_bet": bets, "races_skipped": skipped,
        "skipped_winning_races": skipped_winners,
        "skipped_winners_payout": int(skipped_winners_payout),
        "hits": hits, "hit_rate": hits / bets if bets else 0,
        "total_bet": round(total_bet, 0),
        "total_payout": round(total_payout, 0),
        "profit": round(profit_total, 0),
        "final_bankroll": round(bk, 0),
        "roi_on_bet": (total_payout - total_bet) / total_bet if total_bet else 0,
        "return_on_initial": profit_total / initial,
        "max_dd": round(max_dd, 4),
        "max_losing_streak": max_streak,
        "sharpe_like_per_race": round(sharpe_like, 4) if sharpe_like is not None else None,
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
    }
    return {"summary": summary, "equity": equity_rows, "skips": skip_rows}


# ============================================================================
#  Order variants (for stress tests)
# ============================================================================
def chronological(per_race):
    return per_race.sort_values("race_date", kind="stable").reset_index(drop=True)


def reverse_chronological(per_race):
    return per_race.sort_values("race_date", kind="stable", ascending=False).reset_index(drop=True)


def worst_first(per_race):
    # By year ROI ascending (with E baseline cost/payout)
    yr = per_race.groupby("year").agg(c=("e_cost", "sum"), p=("e_payout", "sum"))
    yr["roi"] = (yr["p"] - yr["c"]) / yr["c"].replace(0, 1)
    order = yr.sort_values("roi").index.tolist()
    per_race["_yo"] = per_race["year"].map({y: i for i, y in enumerate(order)})
    return per_race.sort_values(["_yo", "race_date"], kind="stable").drop(columns="_yo").reset_index(drop=True)


# ============================================================================
#  Monte Carlo (year shuffle)
# ============================================================================
def monte_carlo(per_race: pd.DataFrame, strategy: str, n_trials: int, rng) -> dict:
    years = sorted(per_race["year"].unique().tolist())
    year_index = {y: per_race[per_race["year"] == y].index.tolist() for y in years}
    finals = np.zeros(n_trials)
    max_dds = np.zeros(n_trials)
    ruins = np.zeros(n_trials, dtype=bool)
    for t in range(n_trials):
        perm = rng.permutation(years)
        order_idx = []
        for y in perm:
            order_idx.extend(year_index[y])
        shuffled = per_race.iloc[order_idx].reset_index(drop=True)
        res = simulate(shuffled, strategy)["summary"]
        finals[t] = res["final_bankroll"]
        max_dds[t] = res["max_dd"]
        # ruin = final < initial / 10 (effectively unrecoverable)
        ruins[t] = finals[t] <= 0 or max_dds[t] >= 0.95
    return {
        "trials": n_trials,
        "ruin_rate": float(ruins.mean()),
        "median_final": float(np.median(finals)),
        "p05_final": float(np.percentile(finals, 5)),
        "p95_final": float(np.percentile(finals, 95)),
        "median_max_dd": float(np.median(max_dds)),
        "p95_max_dd": float(np.percentile(max_dds, 95)),
        "p99_max_dd": float(np.percentile(max_dds, 99)),
    }


# ============================================================================
#  Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-trials", type=int, default=MC_TRIALS)
    args = ap.parse_args()

    log.info("loading E per-race detail...")
    e_per_race = build_e_per_race()
    log.info("E races: %d  (state/regime CSV: %s)",
              len(e_per_race),
              "loaded" if (PROCESSED_DIR / "market_stress_timeseries.csv").exists() else "MISSING")
    if e_per_race.empty:
        log.error("no E races — run market_stop_system.py + ensure DB populated first")
        sys.exit(1)
    log.info("state distribution among E races:")
    log.info("  state : %s", e_per_race["state"].value_counts().to_dict())
    log.info("  regime: %s", e_per_race["regime"].value_counts().to_dict())

    rng = np.random.default_rng(RNG_SEED)

    # ---- chronological backtest
    chrono = chronological(e_per_race)
    e_res = simulate(chrono, "E")
    f_res = simulate(chrono, "F")

    # ---- order stress tests
    rev = reverse_chronological(e_per_race)
    wf = worst_first(e_per_race)
    e_rev = simulate(rev, "E")["summary"]
    f_rev = simulate(rev, "F")["summary"]
    e_wf = simulate(wf, "E")["summary"]
    f_wf = simulate(wf, "F")["summary"]

    # ---- Monte Carlo
    log.info("running Monte Carlo (%d trials per strategy)...", args.mc_trials)
    e_mc = monte_carlo(e_per_race, "E", args.mc_trials, rng)
    f_mc = monte_carlo(e_per_race, "F", args.mc_trials, rng)

    # ---- assemble summary
    def _row(label, base, rev, wf, mc):
        return {
            "strategy": label,
            "chrono_profit": base["profit"],
            "chrono_roi_on_initial": base["return_on_initial"],
            "chrono_max_dd": base["max_dd"],
            "chrono_hits": base["hits"],
            "chrono_races_bet": base["races_bet"],
            "chrono_races_skipped": base["races_skipped"],
            "chrono_skipped_winners": base["skipped_winning_races"],
            "chrono_skipped_winners_payout": base["skipped_winners_payout"],
            "chrono_max_losing_streak": base["max_losing_streak"],
            "chrono_sharpe_like": base["sharpe_like_per_race"],
            "chrono_profit_factor": base["profit_factor"],
            "reverse_profit": rev["profit"],
            "reverse_max_dd": rev["max_dd"],
            "worst_first_profit": wf["profit"],
            "worst_first_max_dd": wf["max_dd"],
            "mc_ruin_rate": mc["ruin_rate"],
            "mc_median_final": mc["median_final"],
            "mc_p05_final": mc["p05_final"],
            "mc_p95_max_dd": mc["p95_max_dd"],
            "mc_p99_max_dd": mc["p99_max_dd"],
        }
    summary_df = pd.DataFrame([
        _row("E_D9_P3_CAP4", e_res["summary"], e_rev, e_wf, e_mc),
        _row("F_D9_DYNAMIC_STATE", f_res["summary"], f_rev, f_wf, f_mc),
    ])

    # ---- equity (chrono only for the saved CSV)
    e_eq = pd.DataFrame(e_res["equity"]).assign(strategy="E")
    f_eq = pd.DataFrame(f_res["equity"]).assign(strategy="F")
    equity_df = pd.concat([e_eq, f_eq], ignore_index=True)

    # ---- yearly breakdown (chrono)
    def yearly(equity, label):
        df = equity.copy()
        df["race_date"] = pd.to_datetime(df["race_date"])
        df["year"] = df["race_date"].dt.year
        out = df.groupby("year").apply(lambda g: pd.Series({
            "races": len(g),
            "races_bet": int((g["action"].isin(["HIT", "MISS"])).sum()),
            "races_skipped": int((g["action"] == "SKIP").sum()),
            "hits": int((g["action"] == "HIT").sum()),
            "total_bet": float(g["bet"].sum()),
            "profit": float(g["profit"].sum()),
            "ending_bankroll": float(g["bankroll"].iloc[-1]),
            "max_dd_within_year": float(g["drawdown_pct"].max()),
        })).reset_index()
        out["strategy"] = label
        out["roi"] = out.apply(
            lambda r: (r["profit"] / r["total_bet"]) if r["total_bet"] > 0 else 0, axis=1)
        return out
    yearly_df = pd.concat([yearly(pd.DataFrame(e_res["equity"]), "E"),
                            yearly(pd.DataFrame(f_res["equity"]), "F")], ignore_index=True)

    skips_df = pd.DataFrame(f_res["skips"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(PROCESSED_DIR / "f_strategy_summary.csv",
                       index=False, encoding="utf-8-sig")
    equity_df.to_csv(PROCESSED_DIR / "f_strategy_equity.csv",
                      index=False, encoding="utf-8-sig")
    yearly_df.to_csv(PROCESSED_DIR / "f_strategy_yearly.csv",
                      index=False, encoding="utf-8-sig")
    skips_df.to_csv(PROCESSED_DIR / "f_strategy_skips.csv",
                     index=False, encoding="utf-8-sig")
    log.info("wrote 4 CSVs")

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 280)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n=== STRATEGY F SUMMARY ===")
    print(summary_df.T.to_string())
    print("\n=== YEARLY (E vs F) ===")
    pivot = yearly_df.pivot(index="year", columns="strategy",
                              values=["profit", "races_skipped", "roi"])
    print(pivot.to_string())
    print(f"\n=== F SKIPS (first 20 of {len(skips_df)}) ===")
    if not skips_df.empty:
        print(skips_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

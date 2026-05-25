"""market stop system — survival 優先で 2023 型崩壊を回避する circuit breaker。

設計優先順位:
  1. catastrophic year (2023 型) を完全回避
  2. max drawdown 抑制
  3. ROI 最大化 (← 上 2 つの後)

レイヤー:
  1. stress score (0-100, 複数 feature の weighted sum)
       - fav_win_rate (低いほど stress)
       - fav_top3_rate (低いほど stress)
       - dark_top3_rate (中庸からの乖離 = U-shape stress)
       - payout_inflation
       - top3_pop_entropy
       - chaos_index
       - chaos persistence (rolling CHAOTIC 比率)
  2. state machine: GREEN / YELLOW / ORANGE / RED / HALT
       - 閾値: 30 / 50 / 70 / 85
       - RED 連続 3 race → HALT 強制発動
       - HALT は最低 10 race 持続
       - HALT 解除条件: 終了後 stress < 50
  3. bet multiplier:
       - GREEN 1.0  / YELLOW 0.75 / ORANGE 0.5 / RED 0.25 / HALT 0.0
  4. replay backtest:
       - fixed_1pct / fixed_2pct / regime_aware / market_stop
       - saved_losses, missed_profits, ruin, DD 比較
  5. 2023 deep-dive

出力 CSV:
  market_stress_timeseries.csv     per-race stress + components + state
  market_stop_events.csv           state-change events
  market_state_transition.csv      6x6 transition matrix (probabilities)
  market_stop_backtest.csv         4 strategies 比較
  2023_market_stop_breakdown.csv   2022-07 〜 2023-12 月次

Usage:
    python scripts/analysis/market_stop_system.py
    python scripts/analysis/market_stop_system.py --baseline-year-max 2021
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
    apply_variant, collapse_to_race, collect_subsets, load_data,
)
from scripts.analysis.regime_detection import (
    REGIME_BET_PCT, build_per_race_summary, compute_rolling_features,
    classify_regime, calibrate_thresholds, build_e_per_race,
)

force_utf8_stdout()
log = setup_logger("market_stop")


# ============================================================================
#  Config
# ============================================================================
INITIAL_BANKROLL = 100_000
BASE_BET_PCT = 0.01   # 1% as 100% bet multiplier baseline

STATE_THRESHOLDS = [    # (max_score_exclusive, state)
    (30, "GREEN"),
    (50, "YELLOW"),
    (70, "ORANGE"),
    (85, "RED"),
    (101, "HALT"),
]
STATES = ["GREEN", "YELLOW", "ORANGE", "RED", "HALT"]
BET_MULTIPLIERS = {
    "GREEN": 1.00, "YELLOW": 0.75, "ORANGE": 0.50, "RED": 0.25, "HALT": 0.00,
}

# Persistence rules
RED_TO_HALT_CONSECUTIVE = 3   # 3 consecutive RED → escalate to HALT
HALT_MIN_DURATION = 10        # HALT lasts at least 10 races
HALT_EXIT_SCORE = 50          # must be below this to exit HALT (after min duration)


# ============================================================================
#  Stress score
# ============================================================================
def _norm(value, lo, hi, invert=False):
    """Map value to 0-100 stress score."""
    if pd.isna(value):
        return 50.0
    if hi <= lo:
        return 50.0
    x = (value - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))
    if invert:
        x = 1 - x
    return x * 100


def _u_shape(value, ideal, scale):
    """U-shape stress: ideal value = 0 stress, distance increases stress."""
    if pd.isna(value):
        return 50.0
    return min(100, abs(value - ideal) / scale * 100)


def compute_stress_score(features: pd.DataFrame, baseline_year_max: int) -> pd.DataFrame:
    """Compute per-race 0-100 stress score from rolling features.

    Calibrate normalization parameters from baseline (≤ baseline_year_max).
    """
    base = features[features["race_date"].dt.year <= baseline_year_max].copy()
    base = base.dropna(subset=["fav_win_rate_w30", "dark_top3_rate_w30"])

    # baseline quantiles for normalization
    def q(col, p):
        if col not in base.columns or base[col].dropna().empty:
            return None
        return float(base[col].quantile(p))

    N = {
        "fav_win_lo":    q("fav_win_rate_w30", 0.10) or 0.20,
        "fav_win_hi":    q("fav_win_rate_w30", 0.90) or 0.40,
        "fav_top3_lo":   q("fav_top3_rate_w30", 0.10) or 0.45,
        "fav_top3_hi":   q("fav_top3_rate_w30", 0.90) or 0.70,
        "dark_top3_ideal": q("dark_top3_rate_w30", 0.50) or 0.42,
        "dark_top3_scale": (q("dark_top3_rate_w30", 0.90) or 0.55) - (q("dark_top3_rate_w30", 0.10) or 0.30),
        "infl_lo":       q("payout_inflation_w30v90", 0.10) or 0.80,
        "infl_hi":       q("payout_inflation_w30v90", 0.90) or 1.60,
        "entropy_lo":    q("top3_pop_entropy_w30", 0.10) or 0.5,
        "entropy_hi":    q("top3_pop_entropy_w30", 0.90) or 3.5,
        "chaos_lo":      q("chaos_index_w30", 0.10) or -1.0,
        "chaos_hi":      q("chaos_index_w30", 0.90) or 3.0,
    }
    log.info("normalization params: %s", {k: round(v, 3) for k, v in N.items()})

    # weighted feature → 0-100 stress
    weights = {
        "s_fav_win":      0.22,   # fav 弱 → 高 stress (CHAOTIC sign)
        "s_fav_top3":     0.10,
        "s_dark_extreme": 0.20,   # dark が極端 (chaos or suppressed) → 高 stress
        "s_inflation":    0.13,   # payout 高騰 → 高 stress (異常市場)
        "s_entropy":      0.13,   # top3 popularity ばらつき → 高 stress
        "s_chaos":        0.12,   # top3_pop_avg - 2 → 高 stress
        "s_chaos_persist": 0.10,  # 直近 CHAOTIC 比率 → 高 stress
    }

    # need regime for chaos_persist (re-use regime_detection)
    T = calibrate_thresholds(features, baseline_year_max)
    feat_with_regime = features.copy()
    feat_with_regime["regime"] = feat_with_regime.apply(
        lambda r: classify_regime(r, T), axis=1
    )
    # rolling CHAOTIC ratio (previous 10 races, shift to avoid leak)
    chaotic_flag = (feat_with_regime["regime"] == "CHAOTIC").astype(int)
    chaos_persist_10 = chaotic_flag.rolling(10).mean().shift(1).fillna(0)

    rows = []
    for i, r in features.iterrows():
        comp = {
            "s_fav_win":      _norm(r.get("fav_win_rate_w30"),
                                     N["fav_win_lo"], N["fav_win_hi"], invert=True),
            "s_fav_top3":     _norm(r.get("fav_top3_rate_w30"),
                                     N["fav_top3_lo"], N["fav_top3_hi"], invert=True),
            "s_dark_extreme": _u_shape(r.get("dark_top3_rate_w30"),
                                        N["dark_top3_ideal"], N["dark_top3_scale"] / 2),
            "s_inflation":    _norm(r.get("payout_inflation_w30v90"),
                                     N["infl_lo"], N["infl_hi"]),
            "s_entropy":      _norm(r.get("top3_pop_entropy_w30"),
                                     N["entropy_lo"], N["entropy_hi"]),
            "s_chaos":        _norm(r.get("chaos_index_w30"),
                                     N["chaos_lo"], N["chaos_hi"]),
            "s_chaos_persist": float(chaos_persist_10.iloc[i]) * 100,
        }
        stress = sum(comp[k] * w for k, w in weights.items())
        row = {
            "race_id": r["race_id"],
            "race_date": r["race_date"],
            "regime": feat_with_regime.iloc[i]["regime"],
            "stress_score": round(stress, 2),
            **{k: round(v, 1) for k, v in comp.items()},
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
#  State machine
# ============================================================================
def _raw_state(score: float) -> str:
    for upper, name in STATE_THRESHOLDS:
        if score < upper:
            return name
    return "HALT"


def apply_state_machine(stress: pd.DataFrame) -> pd.DataFrame:
    df = stress.sort_values("race_date", kind="stable").reset_index(drop=True).copy()
    states: list[str] = []
    halt_remaining = 0
    red_consec = 0
    for _, r in df.iterrows():
        score = r["stress_score"]
        raw = _raw_state(score)
        if halt_remaining > 0:
            current = "HALT"
            halt_remaining -= 1
            if halt_remaining == 0 and score >= HALT_EXIT_SCORE:
                # extend until stress drops
                halt_remaining = 1
        else:
            current = raw
            if current == "RED":
                red_consec += 1
                if red_consec >= RED_TO_HALT_CONSECUTIVE:
                    current = "HALT"
                    halt_remaining = HALT_MIN_DURATION - 1
                    red_consec = 0
            elif current == "HALT":
                halt_remaining = HALT_MIN_DURATION - 1
            else:
                red_consec = 0
        states.append(current)
    df["market_state"] = states
    df["bet_multiplier"] = df["market_state"].map(BET_MULTIPLIERS)
    return df


# ============================================================================
#  State change events
# ============================================================================
def extract_stop_events(stress_states: pd.DataFrame) -> pd.DataFrame:
    df = stress_states.sort_values("race_date", kind="stable").reset_index(drop=True)
    events = []
    prev = None
    for _, r in df.iterrows():
        state = r["market_state"]
        if state != prev:
            events.append({
                "race_id": r["race_id"], "race_date": r["race_date"],
                "stress_score": r["stress_score"], "regime": r["regime"],
                "from_state": prev or "(start)", "to_state": state,
            })
            prev = state
    return pd.DataFrame(events)


def state_transition_matrix(stress_states: pd.DataFrame) -> pd.DataFrame:
    seq = stress_states.sort_values("race_date")["market_state"].tolist()
    matrix = pd.DataFrame(0, index=STATES, columns=STATES, dtype=int)
    for a, b in zip(seq[:-1], seq[1:]):
        matrix.at[a, b] += 1
    return matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0).round(4)


# ============================================================================
#  Replay backtest
# ============================================================================
def replay_strategy(stress_states: pd.DataFrame, e_per_race: pd.DataFrame,
                     strategy: str) -> dict:
    merged = e_per_race.merge(
        stress_states[["race_id", "stress_score", "market_state", "bet_multiplier", "regime"]],
        on="race_id", how="left",
    ).sort_values("race_date", kind="stable").reset_index(drop=True)
    bk = float(INITIAL_BANKROLL)
    peak = bk
    max_dd = 0.0
    bets = hits = skipped = 0
    total_bet = total_payout = 0.0
    saved_losses = missed_profits = 0.0
    cur_streak = max_streak = 0
    state_bet_counts = {s: 0 for s in STATES}

    for _, r in merged.iterrows():
        # determine multiplier per strategy
        if strategy == "fixed_1pct":
            mult = 1.0
        elif strategy == "fixed_2pct":
            mult = 2.0
        elif strategy == "regime_aware":
            mult = REGIME_BET_PCT.get(r.get("regime", "INITIAL"), 0.01) / BASE_BET_PCT
        elif strategy == "market_stop":
            mult = float(r.get("bet_multiplier", 1.0))
        else:
            mult = 1.0
        bet = INITIAL_BANKROLL * BASE_BET_PCT * mult

        # baseline counterfactual (fixed_1pct)
        baseline_bet = INITIAL_BANKROLL * BASE_BET_PCT
        if r["hit"]:
            baseline_profit = baseline_bet * (r["payout"] / r["cost"]) - baseline_bet
        else:
            baseline_profit = -baseline_bet

        state_bet_counts[r.get("market_state", "GREEN")] = state_bet_counts.get(r.get("market_state", "GREEN"), 0) + (1 if bet > 0 else 0)

        if bet <= 0:
            skipped += 1
            if baseline_profit < 0:
                saved_losses += -baseline_profit
            else:
                missed_profits += baseline_profit
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
            continue
        bets += 1
        if r["hit"]:
            payout = bet * (r["payout"] / r["cost"])
            hits += 1
            cur_streak = 0
        else:
            payout = 0
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        profit = payout - bet
        # compare to baseline (only if multiplier shrinks the bet)
        if mult < 1.0:
            delta = baseline_profit - profit
            if delta > 0:
                missed_profits += delta
            else:
                saved_losses += -delta
        bk += profit
        peak = max(peak, bk)
        max_dd = max(max_dd, (peak - bk) / peak if peak > 0 else 0)
        total_bet += bet
        total_payout += payout
    return {
        "strategy": strategy,
        "races_total": len(merged),
        "races_bet": bets, "races_skipped": skipped, "hits": hits,
        "hit_rate": round(hits / bets, 4) if bets else 0,
        "total_bet": int(round(total_bet)),
        "total_payout": int(round(total_payout)),
        "profit": int(round(bk - INITIAL_BANKROLL)),
        "final_bankroll": int(round(bk)),
        "roi_on_bet": round((total_payout - total_bet) / total_bet, 4) if total_bet else 0,
        "return_on_initial": round((bk - INITIAL_BANKROLL) / INITIAL_BANKROLL, 4),
        "max_dd": round(max_dd, 4),
        "max_losing_streak": max_streak,
        "saved_losses": int(round(saved_losses)),
        "missed_profits": int(round(missed_profits)),
    }


# ============================================================================
#  2023 deep-dive
# ============================================================================
def deep_dive_2023(stress_states: pd.DataFrame, e_per_race: pd.DataFrame) -> pd.DataFrame:
    df = stress_states[(stress_states["race_date"] >= "2022-07-01")
                        & (stress_states["race_date"] <= "2023-12-31")].copy()
    df["year_month"] = pd.to_datetime(df["race_date"]).dt.to_period("M").astype(str)
    monthly = df.groupby("year_month").agg(
        races=("race_id", "count"),
        avg_stress=("stress_score", "mean"),
        max_stress=("stress_score", "max"),
        green_n=("market_state", lambda s: (s == "GREEN").sum()),
        yellow_n=("market_state", lambda s: (s == "YELLOW").sum()),
        orange_n=("market_state", lambda s: (s == "ORANGE").sum()),
        red_n=("market_state", lambda s: (s == "RED").sum()),
        halt_n=("market_state", lambda s: (s == "HALT").sum()),
    )
    e_in_period = e_per_race[(e_per_race["race_date"] >= "2022-07-01")
                                & (e_per_race["race_date"] <= "2023-12-31")].copy()
    e_in_period["year_month"] = pd.to_datetime(e_in_period["race_date"]).dt.to_period("M").astype(str)
    e_in_period = e_in_period.merge(
        stress_states[["race_id", "market_state"]], on="race_id", how="left"
    )
    e_monthly = e_in_period.groupby("year_month").agg(
        e_races=("race_id", "count"),
        e_hits=("hit", "sum"),
        e_invest=("cost", "sum"),
        e_payout=("payout", "sum"),
        e_halted_n=("market_state", lambda s: (s == "HALT").sum()),
        e_red_n=("market_state", lambda s: (s == "RED").sum()),
    )
    e_monthly["e_roi"] = ((e_monthly["e_payout"] - e_monthly["e_invest"])
                          / e_monthly["e_invest"].replace(0, 1))
    out = monthly.join(e_monthly, how="outer").fillna(0).reset_index()
    out.rename(columns={"index": "year_month"}, inplace=True)
    return out


# ============================================================================
#  Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-year-max", type=int, default=2020)
    args = ap.parse_args()

    log.info("loading data...")
    races, entries, payouts = load_data(("G1", "G2", "G3"),
                                          jra_only=True, exclude_steeplechase=True)
    log.info("races=%d entries=%d", len(races), len(entries))

    per_race = build_per_race_summary(races, entries, payouts)
    features = compute_rolling_features(per_race)
    log.info("features computed: %d races", len(features))

    # ---- 1: stress score
    stress = compute_stress_score(features, args.baseline_year_max)
    # ---- 2: state machine
    stress_states = apply_state_machine(stress)
    stress_states.to_csv(PROCESSED_DIR / "market_stress_timeseries.csv",
                          index=False, encoding="utf-8-sig")
    log.info("wrote market_stress_timeseries.csv")

    # ---- 3: events + transition
    events = extract_stop_events(stress_states)
    events.to_csv(PROCESSED_DIR / "market_stop_events.csv",
                   index=False, encoding="utf-8-sig")
    transition = state_transition_matrix(stress_states)
    transition.to_csv(PROCESSED_DIR / "market_state_transition.csv", encoding="utf-8-sig")

    # ---- 4: replay backtest
    e_per_race = build_e_per_race(races, entries, payouts)
    e_per_race["race_date"] = pd.to_datetime(e_per_race["race_date"])
    results = [replay_strategy(stress_states, e_per_race, s)
               for s in ("fixed_1pct", "fixed_2pct", "regime_aware", "market_stop")]
    backtest = pd.DataFrame(results)
    backtest.to_csv(PROCESSED_DIR / "market_stop_backtest.csv",
                     index=False, encoding="utf-8-sig")

    # ---- 5: 2023 deep-dive
    dd = deep_dive_2023(stress_states, e_per_race)
    dd.to_csv(PROCESSED_DIR / "2023_market_stop_breakdown.csv",
               index=False, encoding="utf-8-sig")

    # ---- stdout
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== STATE DISTRIBUTION (all races) ===")
    print(stress_states["market_state"].value_counts().to_string())
    print(f"\nstress_score quantiles: "
           f"p25={stress_states['stress_score'].quantile(0.25):.1f}, "
           f"p50={stress_states['stress_score'].quantile(0.5):.1f}, "
           f"p75={stress_states['stress_score'].quantile(0.75):.1f}, "
           f"p95={stress_states['stress_score'].quantile(0.95):.1f}, "
           f"max={stress_states['stress_score'].max():.1f}")

    print(f"\n=== STOP EVENTS ({len(events)}) ===")
    print(events.to_string(index=False))

    print("\n=== TRANSITION MATRIX (% by row) ===")
    print(transition.to_string())

    print("\n=== BACKTEST (4 strategies) ===")
    print(backtest.to_string(index=False))

    print("\n=== 2023 DEEP-DIVE (2022-07 〜 2023-12) ===")
    print(dd.to_string(index=False))


if __name__ == "__main__":
    main()

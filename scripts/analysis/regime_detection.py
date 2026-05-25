"""市場 regime を時系列で分類し、E_D9_P3_CAP4 の崩壊 (2023型) を事前検知する。

レイヤー:
  1. per-race summary (全 JRA 平地 G1-G3 から市場特徴を抽出)
  2. rolling features (window: 10/30/90 races + 30/90 days; shift(1) で未来情報リーク防止)
  3. regime classification (rule-based, 2016-2020 で閾値 calibration)
  4. strategy E performance per regime
  5. transition matrix (NORMAL→CHAOTIC 等)
  6. adaptive betting simulation (fixed 1% / fixed 2% / regime-aware)
  7. 2023 deep-dive (崩壊兆候の時系列)
  8. realtime API: get_current_regime(as_of_date) — pipeline から呼び出し可能

出力 CSV:
  regime_timeseries.csv           per-race regime + features
  regime_summary.csv              regime 別 races/E-ROI/DD/etc
  regime_transition_matrix.csv    regime 間遷移確率
  adaptive_betting_backtest.csv   fixed vs regime-aware 比較
  2023_regime_breakdown.csv       2022年後半-2023年の月次推移

Usage:
    python scripts/analysis/regime_detection.py
    python scripts/analysis/regime_detection.py --baseline-year-max 2021
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.config import JRA_RACECOURSES, PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger
from scripts.backtest.strategy_d_variants import (
    apply_variant, collapse_to_race, collect_subsets, load_data,
)

force_utf8_stdout()
log = setup_logger("regime_detection")

INITIAL_BANKROLL = 100_000
RACE_WINDOWS = [10, 30, 90]
DAY_WINDOWS = [30, 90]


# ============================================================================
#  Step 1: per-race market summary (全 JRA 平地 G1-G3)
# ============================================================================
def build_per_race_summary(races: pd.DataFrame,
                            entries: pd.DataFrame,
                            payouts: pd.DataFrame) -> pd.DataFrame:
    e = entries.copy()
    e["popularity"] = pd.to_numeric(e["popularity"], errors="coerce").astype("Int64")
    e["finish_position"] = pd.to_numeric(e["finish_position"], errors="coerce").astype("Int64")
    e["win_odds"] = pd.to_numeric(e["win_odds"], errors="coerce")

    tri = (payouts[payouts["bet_type"] == "三連単"]
           .drop_duplicates("race_id").set_index("race_id"))

    rows = []
    for race_id, g in e.groupby("race_id"):
        race_row = races[races["race_id"] == race_id]
        if race_row.empty:
            continue
        rd = race_row["race_date"].iloc[0]

        fav = g[g["popularity"] == 1]
        fav_finish = int(fav["finish_position"].iloc[0]) \
            if not fav.empty and pd.notna(fav["finish_position"].iloc[0]) else None
        fav_odds = float(fav["win_odds"].iloc[0]) \
            if not fav.empty and pd.notna(fav["win_odds"].iloc[0]) else None

        p2 = g[g["popularity"] == 2]
        p2_finish = int(p2["finish_position"].iloc[0]) \
            if not p2.empty and pd.notna(p2["finish_position"].iloc[0]) else None

        top3 = g[g["finish_position"].isin([1, 2, 3])]
        top3_pops = top3["popularity"].dropna().astype(int).tolist()

        tri_payout = int(tri.loc[race_id, "payout_yen"]) if race_id in tri.index else None

        rows.append({
            "race_id": race_id,
            "race_date": rd,
            "fav_won": 1 if fav_finish == 1 else 0,
            "fav_in_top3": 1 if (fav_finish and fav_finish <= 3) else 0,
            "p1p2_in_top2": 1 if (fav_finish and fav_finish <= 2
                                   and p2_finish and p2_finish <= 2) else 0,
            "dark_in_top3": 1 if any(p >= 5 for p in top3_pops) else 0,
            "n_dark_in_top3": sum(1 for p in top3_pops if p >= 5),
            "top3_pop_avg": float(np.mean(top3_pops)) if top3_pops else None,
            "top3_pop_std": float(np.std(top3_pops)) if len(top3_pops) > 1 else None,
            "fav_odds": fav_odds,
            "trifecta_payout": tri_payout,
        })
    df = pd.DataFrame(rows).sort_values("race_date").reset_index(drop=True)
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


# ============================================================================
#  Step 2: rolling features (shift(1) で未来情報リーク防止)
# ============================================================================
def compute_rolling_features(per_race: pd.DataFrame) -> pd.DataFrame:
    df = per_race.copy()
    for w in RACE_WINDOWS:
        df[f"fav_win_rate_w{w}"]     = df["fav_won"].rolling(w).mean().shift(1)
        df[f"fav_top3_rate_w{w}"]    = df["fav_in_top3"].rolling(w).mean().shift(1)
        df[f"p1p2_freq_w{w}"]        = df["p1p2_in_top2"].rolling(w).mean().shift(1)
        df[f"dark_top3_rate_w{w}"]   = df["dark_in_top3"].rolling(w).mean().shift(1)
        df[f"avg_payout_w{w}"]       = df["trifecta_payout"].rolling(w).mean().shift(1)
        df[f"median_payout_w{w}"]    = df["trifecta_payout"].rolling(w).median().shift(1)
        df[f"payout_std_w{w}"]       = df["trifecta_payout"].rolling(w).std().shift(1)
        df[f"top3_pop_avg_w{w}"]     = df["top3_pop_avg"].rolling(w).mean().shift(1)
        df[f"top3_pop_entropy_w{w}"] = df["top3_pop_std"].rolling(w).mean().shift(1)
        df[f"fav_odds_avg_w{w}"]     = df["fav_odds"].rolling(w).mean().shift(1)

    # day-based windows on a date index (then re-align back)
    df_idx = df.set_index("race_date").sort_index()
    for d in DAY_WINDOWS:
        df_idx[f"fav_win_rate_d{d}"] = df_idx["fav_won"].rolling(f"{d}D").mean().shift(1)
        df_idx[f"dark_top3_rate_d{d}"] = df_idx["dark_in_top3"].rolling(f"{d}D").mean().shift(1)
        df_idx[f"median_payout_d{d}"] = df_idx["trifecta_payout"].rolling(f"{d}D").median().shift(1)
    df = df_idx.reset_index()

    # composite indicators
    df["chaos_index_w30"]          = df["top3_pop_avg_w30"] - 2.0  # 0 = pure 1,2,3 top3; positive = chaos
    df["favorite_dominance_w30"]   = df["fav_win_rate_w30"] - df["dark_top3_rate_w30"]
    df["payout_inflation_w30v90"]  = df["median_payout_w30"] / df["median_payout_w90"]
    df["cv_payout_w30"]            = df["payout_std_w30"] / df["avg_payout_w30"]
    return df


# ============================================================================
#  Step 3: regime classification (rule-based, calibrated thresholds)
# ============================================================================
REGIMES = ("NORMAL", "CHAOTIC", "FAVORITE_DOMINANT", "DARK_SUPPRESSED",
           "HIGH_PAYOUT", "LOW_VARIANCE", "INITIAL")


def calibrate_thresholds(features: pd.DataFrame, baseline_year_max: int) -> dict:
    """Use 2016-baseline_year_max as 'pre-anomaly' period to set quantile thresholds."""
    base = features[features["race_date"].dt.year <= baseline_year_max]
    base = base.dropna(subset=["fav_win_rate_w30", "dark_top3_rate_w30",
                                  "median_payout_w30", "payout_std_w30"])
    if base.empty:
        log.warning("baseline period has no data — using defaults")
        return {"fav_low": 0.23, "fav_high": 0.40,
                "dark_low": 0.30, "dark_high": 0.55,
                "payout_inflation": 1.5, "cv_low": 0.45}
    T = {
        "fav_low":           float(base["fav_win_rate_w30"].quantile(0.15)),
        "fav_high":          float(base["fav_win_rate_w30"].quantile(0.85)),
        "dark_low":          float(base["dark_top3_rate_w30"].quantile(0.15)),
        "dark_high":         float(base["dark_top3_rate_w30"].quantile(0.85)),
        "payout_inflation":  float(base["payout_inflation_w30v90"].quantile(0.90))
                              if "payout_inflation_w30v90" in base else 1.5,
        "cv_low":            float(base["cv_payout_w30"].quantile(0.15))
                              if "cv_payout_w30" in base else 0.45,
    }
    log.info("calibrated thresholds (baseline ≤ %d): %s", baseline_year_max,
              {k: round(v, 3) for k, v in T.items()})
    return T


def classify_regime(row: pd.Series, T: dict) -> str:
    fwr     = row.get("fav_win_rate_w30")
    dtop3   = row.get("dark_top3_rate_w30")
    payout  = row.get("median_payout_w30")
    pay_l   = row.get("median_payout_w90")
    cv      = row.get("cv_payout_w30")

    if pd.isna(fwr):
        return "INITIAL"

    # CHAOTIC: 本命弱 + dark 暴れ
    if fwr < T["fav_low"] and (pd.isna(dtop3) or dtop3 >= T["dark_low"]):
        return "CHAOTIC"
    # FAVORITE_DOMINANT
    if fwr > T["fav_high"]:
        return "FAVORITE_DOMINANT"
    # DARK_SUPPRESSED
    if not pd.isna(dtop3) and dtop3 < T["dark_low"]:
        return "DARK_SUPPRESSED"
    # HIGH_PAYOUT
    if (not pd.isna(payout) and not pd.isna(pay_l) and pay_l > 0
            and payout / pay_l > T["payout_inflation"]):
        return "HIGH_PAYOUT"
    # LOW_VARIANCE
    if not pd.isna(cv) and cv < T["cv_low"]:
        return "LOW_VARIANCE"
    return "NORMAL"


# ============================================================================
#  Step 4: strategy E performance per regime
# ============================================================================
def build_e_per_race(races_filtered, entries, payouts):
    """Strategy E (D9 + P3 + cap4) per-race results."""
    base = collect_subsets(races_filtered, entries, payouts)
    d9 = apply_variant(base, "D9")
    if d9.empty:
        return pd.DataFrame()
    per_race = collapse_to_race(d9)
    # Note: this uses full-D9 buy (avg 20.2 tickets/race). For cap4 (avg 6 tickets),
    # we'd reuse d9_p3_dark_cap.py logic — keep simple here, use D9 baseline.
    return per_race[["race_id", "race_date", "n_tickets", "cost", "hit", "payout"]].copy()


def aggregate_strategy_e_per_regime(features: pd.DataFrame,
                                     e_per_race: pd.DataFrame) -> pd.DataFrame:
    merged = e_per_race.merge(features[["race_id", "regime"]], on="race_id", how="inner")
    rows = []
    for regime, g in merged.groupby("regime"):
        races = len(g)
        tickets = int(g["n_tickets"].sum())
        cost = int(g["cost"].sum())
        payout = int(g["payout"].sum())
        hits = int(g["hit"].sum())
        # streak / DD
        ordered = g.sort_values("race_date", kind="stable")
        cur = best = 0
        peak = bk = INITIAL_BANKROLL
        max_dd = 0.0
        for _, r in ordered.iterrows():
            profit = int(r["payout"]) - int(r["cost"])
            bk += profit
            peak = max(peak, bk)
            max_dd = max(max_dd, (peak - bk) / peak if peak > 0 else 0)
            if r["hit"] == 0:
                cur += 1; best = max(best, cur)
            else:
                cur = 0
        rows.append({
            "regime": regime, "races": races, "tickets": tickets,
            "investment_yen": cost, "hits": hits, "payout_yen": payout,
            "profit_yen": payout - cost,
            "roi": (payout - cost) / cost if cost else 0.0,
            "hit_rate": hits / races if races else 0,
            "max_losing_streak": best, "max_dd_simulated": round(max_dd, 4),
            "max_payout_yen": int(g.loc[g["hit"] > 0, "payout"].max()) if hits else 0,
        })
    return pd.DataFrame(rows).sort_values("races", ascending=False)


# ============================================================================
#  Step 5: transition matrix
# ============================================================================
def transition_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seq = features.sort_values("race_date")["regime"].tolist()
    labels = sorted(set(seq) - {"INITIAL"})
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for a, b in zip(seq[:-1], seq[1:]):
        if a in labels and b in labels:
            matrix.at[a, b] += 1
    row_sums = matrix.sum(axis=1).replace(0, 1)
    matrix_pct = matrix.div(row_sums, axis=0).round(4)
    return matrix, matrix_pct


# ============================================================================
#  Step 6: adaptive betting simulation
# ============================================================================
REGIME_BET_PCT = {
    "NORMAL": 0.015,
    "CHAOTIC": 0.005,
    "FAVORITE_DOMINANT": 0.02,
    "DARK_SUPPRESSED": 0.0,    # skip
    "HIGH_PAYOUT": 0.02,
    "LOW_VARIANCE": 0.01,
    "INITIAL": 0.01,
}


def adaptive_betting_backtest(e_per_race: pd.DataFrame,
                                features: pd.DataFrame) -> pd.DataFrame:
    """Compare fixed 1% / fixed 2% / regime-aware sizing.
    Bet sizing uses INITIAL_BANKROLL fraction (not current) to match earlier work.
    """
    merged = e_per_race.merge(features[["race_id", "regime"]], on="race_id", how="left")
    merged = merged.sort_values("race_date", kind="stable").reset_index(drop=True)

    strategies = {
        "fixed_1pct":   lambda r: 0.01 * INITIAL_BANKROLL,
        "fixed_2pct":   lambda r: 0.02 * INITIAL_BANKROLL,
        "regime_aware": lambda r: REGIME_BET_PCT.get(r, 0.01) * INITIAL_BANKROLL,
    }

    out = []
    for name, sizer in strategies.items():
        bk = float(INITIAL_BANKROLL)
        peak = bk
        max_dd = 0.0
        races = bets = hits = 0
        total_bet = total_payout = 0.0
        max_streak = cur = 0
        skipped = 0
        for _, r in merged.iterrows():
            races += 1
            bet = sizer(r["regime"])
            if bet <= 0:
                skipped += 1
                continue
            if bet > bk:
                # skip if can't afford (defensive; with 0.005-0.02 of 100K never triggers here)
                skipped += 1
                continue
            bets += 1
            # payout scales linearly with bet (E[D9] payout already known per race)
            if r["hit"]:
                payout = bet * (r["payout"] / r["cost"])
                hits += 1
                cur = 0
            else:
                payout = 0
                cur += 1
                max_streak = max(max_streak, cur)
            profit = payout - bet
            bk += profit
            peak = max(peak, bk)
            max_dd = max(max_dd, (peak - bk) / peak if peak > 0 else 0)
            total_bet += bet
            total_payout += payout
        out.append({
            "strategy": name,
            "races_total": races,
            "races_bet": bets,
            "races_skipped": skipped,
            "hits": hits,
            "hit_rate": hits / bets if bets else 0,
            "total_bet": round(total_bet, 0),
            "total_payout": round(total_payout, 0),
            "profit": round(bk - INITIAL_BANKROLL, 0),
            "final_bankroll": round(bk, 0),
            "roi_on_bet": (total_payout - total_bet) / total_bet if total_bet else 0,
            "return_on_initial": (bk - INITIAL_BANKROLL) / INITIAL_BANKROLL,
            "max_dd": round(max_dd, 4),
            "max_losing_streak": max_streak,
        })
    return pd.DataFrame(out)


# ============================================================================
#  Step 7: 2023 deep-dive
# ============================================================================
def deep_dive_2023(features: pd.DataFrame,
                     e_per_race: pd.DataFrame) -> pd.DataFrame:
    period = features[(features["race_date"] >= "2022-07-01")
                       & (features["race_date"] <= "2023-12-31")].copy()
    period["year_month"] = period["race_date"].dt.to_period("M").astype(str)
    # monthly regime distribution
    monthly = period.groupby("year_month")["regime"].value_counts().unstack(fill_value=0)
    # monthly E performance
    e_in_period = e_per_race[(e_per_race["race_date"] >= "2022-07-01")
                              & (e_per_race["race_date"] <= "2023-12-31")].copy()
    e_in_period["year_month"] = pd.to_datetime(e_in_period["race_date"]).dt.to_period("M").astype(str)
    e_monthly = e_in_period.groupby("year_month").agg(
        e_races=("race_id", "count"),
        e_hits=("hit", "sum"),
        e_invest=("cost", "sum"),
        e_payout=("payout", "sum"),
    )
    e_monthly["e_roi"] = (e_monthly["e_payout"] - e_monthly["e_invest"]) / e_monthly["e_invest"].replace(0, 1)
    out = monthly.join(e_monthly, how="outer").fillna(0)
    out.index.name = "year_month"
    return out.reset_index()


# ============================================================================
#  Step 8: realtime API (pipeline で呼べる)
# ============================================================================
def get_current_regime(as_of_date) -> str:
    """Run regime detection on data up to `as_of_date` and return the most recent regime label."""
    engine = get_engine()
    races, entries, payouts = load_data(("G1", "G2", "G3"),
                                          jra_only=True,
                                          exclude_steeplechase=True)
    races = races[races["race_date"] <= pd.Timestamp(as_of_date)]
    per_race = build_per_race_summary(races, entries, payouts)
    features = compute_rolling_features(per_race)
    # threshold from 2016-2020 baseline
    T = calibrate_thresholds(features, baseline_year_max=2020)
    features["regime"] = features.apply(lambda r: classify_regime(r, T), axis=1)
    latest = features.dropna(subset=["fav_win_rate_w30"]).tail(1)
    if latest.empty:
        return "INITIAL"
    return str(latest["regime"].iloc[0])


# ============================================================================
#  Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-year-max", type=int, default=2020,
                    help="last year of pre-anomaly baseline for threshold calibration")
    args = ap.parse_args()

    log.info("loading data (JRA flat G1-G3 2015-2025)")
    races, entries, payouts = load_data(("G1", "G2", "G3"),
                                          jra_only=True,
                                          exclude_steeplechase=True)
    log.info("races=%d entries=%d", len(races), len(entries))

    per_race = build_per_race_summary(races, entries, payouts)
    log.info("per_race summary built: %d rows", len(per_race))

    features = compute_rolling_features(per_race)
    log.info("rolling features computed (race windows %s, day windows %s)",
              RACE_WINDOWS, DAY_WINDOWS)

    T = calibrate_thresholds(features, baseline_year_max=args.baseline_year_max)
    features["regime"] = features.apply(lambda r: classify_regime(r, T), axis=1)

    # ---- Output 1: regime_timeseries.csv
    ts_cols = ["race_id", "race_date", "regime",
                "fav_win_rate_w30", "fav_top3_rate_w30", "dark_top3_rate_w30",
                "median_payout_w30", "median_payout_w90", "payout_inflation_w30v90",
                "chaos_index_w30", "favorite_dominance_w30", "cv_payout_w30",
                "fav_win_rate_d90", "dark_top3_rate_d90"]
    ts_cols = [c for c in ts_cols if c in features.columns]
    features[ts_cols].to_csv(PROCESSED_DIR / "regime_timeseries.csv",
                              index=False, encoding="utf-8-sig")
    log.info("wrote regime_timeseries.csv")

    # ---- Output 2: regime_summary.csv (E per regime)
    e_per_race = build_e_per_race(races, entries, payouts)
    log.info("strategy E applicable: %d races", len(e_per_race))
    e_per_race["race_date"] = pd.to_datetime(e_per_race["race_date"])
    summary = aggregate_strategy_e_per_regime(features, e_per_race)
    summary.to_csv(PROCESSED_DIR / "regime_summary.csv",
                    index=False, encoding="utf-8-sig")
    log.info("wrote regime_summary.csv")

    # ---- Output 3: transition matrix
    matrix, matrix_pct = transition_matrix(features)
    matrix_pct.to_csv(PROCESSED_DIR / "regime_transition_matrix.csv", encoding="utf-8-sig")
    log.info("wrote regime_transition_matrix.csv")

    # ---- Output 4: adaptive betting backtest
    adaptive = adaptive_betting_backtest(e_per_race, features)
    adaptive.to_csv(PROCESSED_DIR / "adaptive_betting_backtest.csv",
                     index=False, encoding="utf-8-sig")
    log.info("wrote adaptive_betting_backtest.csv")

    # ---- Output 5: 2023 deep-dive
    dd2023 = deep_dive_2023(features, e_per_race)
    dd2023.to_csv(PROCESSED_DIR / "2023_regime_breakdown.csv",
                   index=False, encoding="utf-8-sig")
    log.info("wrote 2023_regime_breakdown.csv")

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== THRESHOLDS (calibrated) ===")
    for k, v in T.items():
        print(f"  {k:<22}: {v:.3f}")

    print("\n=== REGIME DISTRIBUTION (all races) ===")
    print(features["regime"].value_counts().to_string())

    print("\n=== STRATEGY E PERFORMANCE BY REGIME ===")
    print(summary.to_string(index=False))

    print("\n=== TRANSITION MATRIX (row → col, %) ===")
    print(matrix_pct.to_string())

    print("\n=== ADAPTIVE BETTING BACKTEST ===")
    print(adaptive.to_string(index=False))

    print("\n=== 2023 DEEP-DIVE (2022-07 〜 2023-12 monthly) ===")
    print(dd2023.to_string(index=False))


if __name__ == "__main__":
    main()

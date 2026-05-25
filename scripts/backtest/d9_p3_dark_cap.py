"""P3 を dark horse の上位 N 頭で打ち切る (cap1〜cap5) ROI / 破産率 比較。

P3 は「dark 3 着固定、p1/p2 を 1 着 2 着で swap」= 2 perms / dark。
dark 候補が複数いる場合、以下の順序でランキングして上位 N 頭を採用:

  1) 単勝オッズ 昇順 (低オッズ = "最も低い穴")
  2) 同値時 人気 昇順
  3) さらに同値時 馬番 昇順

各 cap での 1 race 最大 ticket 数 = 2 × cap (cap5 で 10、cap1 で 2)。

評価:
  bet = 100 yen / ticket 固定
  hit = true 三連単 combo (a-b-c) が、{a,b}={p1,p2} かつ c ∈ 上位 N dark のとき
  payout = raw_payout (1 ticket 当たる)

Bootstrap CI: race 単位 resample 1000 回、races >= 30 のみ。
Monte Carlo: 100K initial、年単位 shuffle 10K trials。

出力:
  data/processed/d9_p3_dark_cap_summary.csv
  data/processed/d9_p3_dark_cap_hits.csv
  data/processed/d9_p3_dark_cap_monte_carlo.csv
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
    apply_variant, collapse_to_race, collect_subsets, load_data,
)
from scripts.analysis.d9_deepdive import make_component_labels, tag_components

force_utf8_stdout()
log = setup_logger("p3_dark_cap")

TICKET_COST = 100
LOW_SAMPLE_THRESHOLD = 30
BOOTSTRAP_N = 1000
INITIAL = 100_000
MC_TRIALS = 10_000
RNG_SEED = 20260524

CAPS = [1, 2, 3, 4, 5]


# ---- helpers --------------------------------------------------------------

def enrich_subsets(d9_subsets, entries_df):
    """Add dark_pop / dark_odds from entries."""
    e = entries_df.copy()
    e["popularity"] = pd.to_numeric(e["popularity"], errors="coerce").astype("Int64")
    e["horse_number"] = pd.to_numeric(e["horse_number"], errors="coerce").astype("Int64")
    e["win_odds"] = pd.to_numeric(e["win_odds"], errors="coerce")
    meta = (e[["race_id", "horse_number", "popularity", "win_odds"]]
            .rename(columns={"horse_number": "dark_horse",
                             "popularity": "dark_pop",
                             "win_odds": "dark_odds"}))
    return d9_subsets.merge(meta, on=["race_id", "dark_horse"], how="left")


def sorted_darks_per_race(d9_enriched: pd.DataFrame) -> dict[str, list[int]]:
    """For each race, return dark horse numbers sorted by (odds, pop, horse_num) ascending."""
    out = {}
    for rid, sub in d9_enriched.groupby("race_id", sort=False):
        ordered = sub.sort_values(["dark_odds", "dark_pop", "dark_horse"], kind="stable")
        out[rid] = ordered["dark_horse"].astype(int).tolist()
    return out


def winning_dark_position(per_race_meta, sorted_darks_map, p1_p2_map, tri_combo):
    """Per race, return the rank (0-indexed) of the winning dark in the sorted list,
    or None if the race isn't a P3-pattern hit (either {a,b} != {p1,p2}, or c not a dark)."""
    out = {}
    for rid in per_race_meta:
        pp = p1_p2_map.get(rid)
        true = tri_combo.get(rid)
        darks = sorted_darks_map.get(rid, [])
        if pp is None or not true or not darks:
            out[rid] = None
            continue
        p1, p2 = pp
        try:
            parts = [int(x) for x in str(true).split("-")]
        except ValueError:
            out[rid] = None
            continue
        if len(parts) != 3:
            out[rid] = None
            continue
        a, b, c = parts
        if {a, b} != {p1, p2}:
            out[rid] = None
            continue
        if c in darks:
            out[rid] = darks.index(c)  # 0-indexed
        else:
            out[rid] = None
    return out


# ---- per-cap per-race evaluation ------------------------------------------

def evaluate_cap(cap, per_race_meta, sorted_darks_map, winning_pos_map,
                 tri_payout):
    """Per-race outcome for this cap. Returns DataFrame."""
    rows = []
    for rid, info in per_race_meta.items():
        darks = sorted_darks_map.get(rid, [])
        n_darks_used = min(cap, len(darks))
        n_tickets = n_darks_used * 2  # 2 perms per dark
        cost = n_tickets * TICKET_COST
        wpos = winning_pos_map.get(rid)
        hit = wpos is not None and wpos < cap
        payout = int(tri_payout.get(rid, 0) or 0) if hit else 0
        rows.append({
            "cap": cap,
            "race_id": rid,
            "race_date": info["race_date"],
            "race_name": info["race_name"],
            "racecourse": info["racecourse"],
            "grade": info["grade"],
            "year": info["year"],
            "field_size": info["field_size"],
            "component_labels": info["component_labels"],
            "n_darks_available": len(darks),
            "n_darks_used": n_darks_used,
            "n_tickets": n_tickets,
            "cost": cost,
            "winning_dark_position": wpos,  # None if no P3 hit possible
            "hit": int(hit),
            "payout": payout,
            "profit": payout - cost,
        })
    return pd.DataFrame(rows)


def bootstrap_ci(costs, payouts, rng, n=BOOTSTRAP_N):
    if len(costs) < LOW_SAMPLE_THRESHOLD:
        return None, None
    idx = rng.integers(0, len(costs), size=(n, len(costs)))
    sc = costs[idx].sum(axis=1)
    sp = payouts[idx].sum(axis=1)
    valid = sc > 0
    rois = np.zeros(n)
    rois[valid] = (sp[valid] - sc[valid]) / sc[valid]
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def longest_loss_streak(per_race: pd.DataFrame) -> int:
    if per_race.empty:
        return 0
    ordered = per_race.sort_values("race_date", kind="stable", na_position="last")
    cur = best = 0
    for h in ordered["hit"]:
        if h == 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summarize(per_race: pd.DataFrame, cap: int, rng) -> dict:
    races = len(per_race)
    tickets = int(per_race["n_tickets"].sum())
    cost = int(per_race["cost"].sum())
    payout = int(per_race["payout"].sum())
    hits = int(per_race["hit"].sum())
    max_payout = int(per_race.loc[per_race["hit"] > 0, "payout"].max()) if hits else 0
    avg_hit_payout = payout / hits if hits else 0
    ci_lo, ci_hi = bootstrap_ci(
        per_race["cost"].to_numpy(dtype=float),
        per_race["payout"].to_numpy(dtype=float),
        rng,
    )
    return {
        "cap": cap,
        "max_tickets_per_race": cap * 2,
        "races": races,
        "tickets": tickets,
        "avg_tickets_per_race": tickets / races if races else 0.0,
        "investment_yen": cost,
        "hits": hits,
        "payout_yen": payout,
        "profit_yen": payout - cost,
        "roi": (payout - cost) / cost if cost else 0.0,
        "hit_rate": hits / races if races else 0.0,
        "avg_hit_payout": avg_hit_payout,
        "max_payout_yen": max_payout,
        "max_losing_streak": longest_loss_streak(per_race),
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "sample_warning": "LOW_SAMPLE" if races < LOW_SAMPLE_THRESHOLD else "",
    }


# ---- Monte Carlo ----------------------------------------------------------

def vectorized_simulate(bet_arr, profit_arr, orders, initial):
    trials, N = orders.shape
    bet_2d = bet_arr[orders]
    profit_2d = profit_arr[orders]
    bankroll = np.full(trials, float(initial))
    peak = bankroll.copy()
    min_bk = bankroll.copy()
    max_dd = np.zeros(trials)
    ruin = np.zeros(trials, dtype=bool)
    for i in range(N):
        b = bet_2d[:, i]
        p = profit_2d[:, i]
        can_bet = b <= bankroll
        ruin |= ~can_bet
        bankroll = np.where(can_bet, bankroll + p, bankroll)
        peak = np.maximum(peak, bankroll)
        min_bk = np.minimum(min_bk, bankroll)
        dd = np.where(peak > 0, (peak - bankroll) / peak, 0.0)
        max_dd = np.maximum(max_dd, dd)
        ruin |= (bankroll <= 0)
    return bankroll, max_dd, min_bk, ruin


def year_roi_ranking(per_race):
    g = per_race.groupby("year").agg(cost=("cost", "sum"), payout=("payout", "sum"))
    g["roi"] = (g["payout"] - g["cost"]) / g["cost"].where(g["cost"] > 0, 1)
    return g.sort_values("roi").index.tolist()


def run_mc(per_race, rng):
    if per_race.empty:
        return None
    per = per_race.sort_values("race_date", kind="stable").reset_index(drop=True)
    costs = per["cost"].to_numpy(dtype=float)
    payouts = per["payout"].to_numpy(dtype=float)
    profits = payouts - costs
    years = per["year"].to_numpy(dtype=np.int32)
    year_list = sorted(set(years.tolist()))
    year_indices = {y: np.where(years == y)[0] for y in year_list}
    N = len(per)
    orders = np.zeros((MC_TRIALS, N), dtype=np.int32)
    yarr = np.array(year_list)
    for t in range(MC_TRIALS):
        perm = rng.permutation(yarr)
        orders[t] = np.concatenate([year_indices[y] for y in perm])
    final_bk, max_dd, min_bk, ruin = vectorized_simulate(costs, profits, orders, INITIAL)
    # worst-first
    wf_order = year_roi_ranking(per)
    wf_idx = np.concatenate([year_indices[y] for y in wf_order])[None, :]
    wf_final, wf_dd, wf_min, wf_ruin = vectorized_simulate(costs, profits, wf_idx, INITIAL)
    return {
        "ruin_rate": float(ruin.mean()),
        "median_final_bankroll": float(np.median(final_bk)),
        "p05_final_bankroll": float(np.percentile(final_bk, 5)),
        "p95_final_bankroll": float(np.percentile(final_bk, 95)),
        "median_max_drawdown_pct": float(np.median(max_dd)),
        "p95_max_drawdown_pct": float(np.percentile(max_dd, 95)),
        "p99_max_drawdown_pct": float(np.percentile(max_dd, 99)),
        "median_min_bankroll": float(np.median(min_bk)),
        "p05_min_bankroll": float(np.percentile(min_bk, 5)),
        "worst_first_final_bankroll": float(wf_final[0]),
        "worst_first_max_dd_pct": float(wf_dd[0]),
        "worst_first_ruin_flag": bool(wf_ruin[0]),
    }


# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args()
    rng_boot = np.random.default_rng(RNG_SEED)
    rng_mc = np.random.default_rng(RNG_SEED + 1)

    log.info("loading D9 base...")
    races_df, entries_df, payouts_df = load_data(("G1", "G2", "G3"), True, True)
    base = collect_subsets(races_df, entries_df, payouts_df)
    d9 = apply_variant(base, "D9")
    d9 = enrich_subsets(d9, entries_df)
    log.info("D9 subsets=%d", len(d9))

    per_race_d9 = collapse_to_race(d9)
    per_race_d9 = tag_components(per_race_d9)
    per_race_d9["component_labels"] = make_component_labels(per_race_d9)
    per_race_meta = {
        r["race_id"]: {
            "race_date": r["race_date"],
            "race_name": r["race_name"],
            "racecourse": r["racecourse"],
            "grade": r["grade"],
            "year": r["year"],
            "field_size": r["field_size"],
            "component_labels": r["component_labels"],
        }
        for _, r in per_race_d9.iterrows()
    }

    # p1, p2 maps
    e = entries_df.copy()
    e["popularity"] = pd.to_numeric(e["popularity"], errors="coerce").astype("Int64")
    e["horse_number"] = pd.to_numeric(e["horse_number"], errors="coerce").astype("Int64")
    p1_map = (e[e["popularity"] == 1].drop_duplicates("race_id")
              .set_index("race_id")["horse_number"]).to_dict()
    p2_map = (e[e["popularity"] == 2].drop_duplicates("race_id")
              .set_index("race_id")["horse_number"]).to_dict()
    p1_p2_map = {rid: (int(p1_map[rid]), int(p2_map[rid]))
                 for rid in p1_map if rid in p2_map}

    # trifecta lookup
    tri = (payouts_df[payouts_df["bet_type"] == "三連単"]
           .drop_duplicates("race_id").set_index("race_id"))
    tri_combo = tri["combination"].astype(str).to_dict()
    tri_payout = tri["payout_yen"].to_dict()

    # precompute sorted darks per race and winning dark position
    sorted_darks_map = sorted_darks_per_race(d9)
    winning_pos_map = winning_dark_position(per_race_meta, sorted_darks_map,
                                            p1_p2_map, tri_combo)
    log.info("races with P3-possible hit (winning dark in any rank): %d / %d",
             sum(1 for v in winning_pos_map.values() if v is not None),
             len(winning_pos_map))

    # ---- evaluate each cap
    summary_rows = []
    hits_rows = []
    per_race_by_cap = {}
    mc_summary_rows = []

    for cap in CAPS:
        t0 = time.time()
        per = evaluate_cap(cap, per_race_meta, sorted_darks_map, winning_pos_map, tri_payout)
        per_race_by_cap[cap] = per
        s = summarize(per, cap, rng_boot)
        summary_rows.append(s)
        log.info("cap=%d: races=%d tickets=%d avg=%.1f ROI=%+.1f%% hits=%d (%.1fs)",
                 cap, s["races"], s["tickets"], s["avg_tickets_per_race"],
                 s["roi"] * 100, s["hits"], time.time() - t0)

        for _, r in per[per["hit"] > 0].iterrows():
            hits_rows.append({
                "cap": cap,
                "race_id": r["race_id"],
                "race_date": r["race_date"],
                "race_name": r["race_name"],
                "racecourse": r["racecourse"],
                "grade": r["grade"],
                "field_size": r["field_size"],
                "component_labels": r["component_labels"],
                "winning_dark_position": r["winning_dark_position"],
                "n_tickets": r["n_tickets"],
                "investment_yen": r["cost"],
                "payout_yen": r["payout"],
                "profit_yen": r["profit"],
            })

    summary_df = pd.DataFrame(summary_rows)

    # ---- missed hits vs cap5
    cap5_per = per_race_by_cap[5]
    cap5_hit_set = set(cap5_per[cap5_per["hit"] > 0]["race_id"])
    cap5_payout_map = cap5_per.set_index("race_id")["payout"].to_dict()
    missed_rows = []
    for cap in CAPS:
        per = per_race_by_cap[cap]
        cap_hit_set = set(per[per["hit"] > 0]["race_id"])
        kept = cap_hit_set
        missed_vs_5 = cap5_hit_set - cap_hit_set
        missed_rows.append({
            "cap": cap,
            "kept_hits": len(kept),
            "missed_hits_vs_cap5": len(missed_vs_5),
            "kept_payout_yen": int(per["payout"].sum()),
            "missed_payout_yen_vs_cap5": int(sum(cap5_payout_map[r] for r in missed_vs_5)),
        })
    missed_df = pd.DataFrame(missed_rows)
    summary_df = summary_df.merge(missed_df, on="cap", how="left")

    # ---- MC per cap
    log.info("running MC for each cap (100K initial)...")
    for cap in CAPS:
        per = per_race_by_cap[cap]
        t0 = time.time()
        mc = run_mc(per, rng_mc)
        log.info("MC cap=%d: ruin=%.2f%% median_final=%.0f worst_first_ruin=%s (%.1fs)",
                 cap, mc["ruin_rate"] * 100, mc["median_final_bankroll"],
                 mc["worst_first_ruin_flag"], time.time() - t0)
        mc_summary_rows.append({"cap": cap, **mc})
    mc_df = pd.DataFrame(mc_summary_rows)

    # ---- save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(PROCESSED_DIR / "d9_p3_dark_cap_summary.csv",
                      index=False, encoding="utf-8-sig")
    pd.DataFrame(hits_rows).to_csv(PROCESSED_DIR / "d9_p3_dark_cap_hits.csv",
                                   index=False, encoding="utf-8-sig")
    mc_df.to_csv(PROCESSED_DIR / "d9_p3_dark_cap_monte_carlo.csv",
                 index=False, encoding="utf-8-sig")
    log.info("wrote 3 CSVs")

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== P3 DARK-CAP SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== MONTE CARLO @ 100K (each cap) ===")
    print(mc_df.to_string(index=False))

    # extra: distribution of winning_dark_position
    pos_dist = pd.Series([v for v in winning_pos_map.values() if v is not None]).value_counts().sort_index()
    print("\n=== winning_dark_position distribution (P3 hits only) ===")
    print(pos_dist.to_string())


if __name__ == "__main__":
    main()

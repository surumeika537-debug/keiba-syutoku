"""D9 の買い目を 1レース 10点以内に圧縮するルールを比較。

JRA の最低 bet が 100yen/ticket なので、「1レース 1,000yen 運用 = 最大10点」が現実上限。
D9 の full 買い目 (18-30点) では足りないため、何らかの pruning が必要。

Pruning rules:
  P0  full_d9                       baseline = 全買い目 (10点制約なし)
  P1  top10_by_low_odds             3頭オッズ合計 (= dark_odds) 昇順に10点
  P2  top10_by_popularity_sum       3頭人気合計 (= dark_pop) 昇順に10点
  P3  top10_darkhorse_3rd           dark=3着固定, 1/2番人気を1着2着swap. dark候補は
                                    odds昇順に最大5頭 → 最大10点
  P4  top10_darkhorse_2nd_or_3rd    1/2番人気 1着, dark 2着or3着 (= dark不1着) の
                                    4perms/dark を dark_pop昇順で最大10点
  P5  exclude_bad_frame_then_top10  dark 3-4枠は除外。残りから dark_pop昇順10点
  P6  component_weighted_top10      component別にperm filter:
                                      niigata_4g3:    dark 1着以外 (=4perms/dark)
                                      march_g2:      1番人気 1着 (=2perms/dark)
                                      hanshin_g1:    1/2番人気 1着 (=4perms/dark)
                                      september_g3:  dark 3着固定 (=2perms/dark)
                                    残りから dark_pop昇順10点。複数componentマッチは
                                    優先順 niigata > march > hanshin > september

評価:
  per-race: 選択した10点以内に true_combo が含まれていれば hit (= raw_payout 払戻)
  bet/ticket = 100yen 固定
  cost = n_tickets × 100

出力 (3 CSV):
  data/processed/d9_ticket_pruning_summary.csv
  data/processed/d9_ticket_pruning_hits.csv         (rule × race-level hits)
  data/processed/d9_ticket_pruning_monte_carlo.csv  (rule × MC summary @ 100K initial)
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.utils import force_utf8_stdout, setup_logger
from scripts.backtest.strategy_d_variants import (
    NIIGATA_4GIII_PATTERNS,
    apply_variant,
    collapse_to_race,
    collect_subsets,
    load_data,
)
from scripts.analysis.d9_deepdive import make_component_labels, tag_components

force_utf8_stdout()
log = setup_logger("ticket_pruning")

MAX_TICKETS = 10
TICKET_COST = 100
LOW_SAMPLE_THRESHOLD = 30
BOOTSTRAP_N = 1000
INITIAL = 100_000
MC_TRIALS = 10_000
RNG_SEED = 20260524


# ---- ticket-level enumeration ---------------------------------------------

def frame_band(f):
    if pd.isna(f): return "(NaN)"
    f = int(f)
    if f <= 2: return "1-2枠"
    if f <= 4: return "3-4枠"
    if f <= 6: return "5-6枠"
    return "7-8枠"


def build_ticket_table(d9_subsets: pd.DataFrame, p1_p2_map: dict) -> pd.DataFrame:
    """For each D9 subset (= 1 race × 1 dark horse), produce 6 permutation tickets."""
    rows = []
    for _, s in d9_subsets.iterrows():
        rid = s["race_id"]
        pp = p1_p2_map.get(rid)
        if pp is None:
            continue
        p1, p2 = pp
        d = int(s["dark_horse"])
        for perm in permutations((p1, p2, d), 3):
            rows.append({
                "race_id": rid,
                "ticket": f"{perm[0]}-{perm[1]}-{perm[2]}",
                "perm0": perm[0],
                "perm1": perm[1],
                "perm2": perm[2],
                "dark": d,
                "dark_odds": float(s["dark_odds"]),
                "dark_pop": int(s["dark_pop"]),
                "dark_frame": int(s["dark_frame"]) if pd.notna(s["dark_frame"]) else -1,
                "dark_pos": perm.index(d),  # 0=1st, 1=2nd, 2=3rd
                "p1_first": perm[0] == p1,
                "p2_first": perm[0] == p2,
                "p1_p2_first": perm[0] in (p1, p2),
            })
    df = pd.DataFrame(rows)
    df["frame_band"] = df["dark_frame"].apply(frame_band)
    return df


# ---- pruning rules --------------------------------------------------------

def select_top_n_by(filtered: pd.DataFrame, sort_cols: list, n: int) -> pd.DataFrame:
    return filtered.sort_values(sort_cols, kind="stable").head(n)


COMPONENT_PRIORITY = ["niigata_4g3", "march_g2", "hanshin_g1", "september_g3"]


def primary_component(comp_labels: str) -> str:
    if not comp_labels:
        return ""
    parts = set(comp_labels.split("|"))
    for c in COMPONENT_PRIORITY:
        if c in parts:
            return c
    return ""


def prune_p0(tickets: pd.DataFrame, race_info: dict) -> pd.DataFrame:
    return tickets  # baseline: keep all

def prune_p1(tickets: pd.DataFrame, race_info: dict) -> pd.DataFrame:
    # tie-break: dark_pop ascending, then ticket string
    return select_top_n_by(tickets, ["dark_odds", "dark_pop", "ticket"], MAX_TICKETS)

def prune_p2(tickets: pd.DataFrame, race_info: dict) -> pd.DataFrame:
    return select_top_n_by(tickets, ["dark_pop", "dark_odds", "ticket"], MAX_TICKETS)

def prune_p3(tickets: pd.DataFrame, race_info: dict) -> pd.DataFrame:
    # dark fixed at 3rd → 2 perms per dark (p1-p2-dark, p2-p1-dark)
    sub = tickets[tickets["dark_pos"] == 2]
    if sub.empty:
        return sub
    # rank darks by odds ascending, keep top 5 darks (2 perms each = 10)
    dark_order = (sub.drop_duplicates("dark")
                  .sort_values(["dark_odds", "dark_pop", "dark"], kind="stable")
                  ["dark"].head(5).tolist())
    out = sub[sub["dark"].isin(dark_order)].copy()
    # order rows by the dark-priority
    out["_dark_rank"] = out["dark"].map({d: i for i, d in enumerate(dark_order)})
    return out.sort_values(["_dark_rank", "ticket"], kind="stable").drop(columns="_dark_rank").head(MAX_TICKETS)

def prune_p4(tickets: pd.DataFrame, race_info: dict) -> pd.DataFrame:
    # p1 or p2 as 1st AND dark as 2nd or 3rd (= dark not 1st)
    sub = tickets[tickets["dark_pos"] != 0]
    return select_top_n_by(sub, ["dark_pop", "dark_odds", "ticket"], MAX_TICKETS)

def prune_p5(tickets: pd.DataFrame, race_info: dict) -> pd.DataFrame:
    sub = tickets[tickets["frame_band"] != "3-4枠"]
    return select_top_n_by(sub, ["dark_pop", "dark_odds", "ticket"], MAX_TICKETS)

def prune_p6(tickets: pd.DataFrame, race_info: dict) -> pd.DataFrame:
    comp = race_info.get("component", "")
    if comp == "niigata_4g3":
        sub = tickets[tickets["dark_pos"] != 0]               # dark 2/3着
    elif comp == "march_g2":
        sub = tickets[tickets["p1_first"]]                    # p1 1着
    elif comp == "hanshin_g1":
        sub = tickets[tickets["p1_p2_first"]]                 # p1 or p2 1着
    elif comp == "september_g3":
        sub = tickets[tickets["dark_pos"] == 2]               # dark 3着
    else:
        sub = tickets  # fallback (shouldn't happen for D9 races)
    return select_top_n_by(sub, ["dark_pop", "dark_odds", "ticket"], MAX_TICKETS)


RULES = {
    "P0_full_d9":                       prune_p0,
    "P1_top10_by_low_odds":             prune_p1,
    "P2_top10_by_popularity_sum":       prune_p2,
    "P3_top10_darkhorse_3rd":           prune_p3,
    "P4_top10_darkhorse_2nd_or_3rd":    prune_p4,
    "P5_exclude_bad_frame_then_top10":  prune_p5,
    "P6_component_weighted_top10":      prune_p6,
}


# ---- evaluation -----------------------------------------------------------

def evaluate_rule(rule_name, prune_fn, ticket_df, race_meta, tri_combo, tri_payout):
    """Return per-race result DataFrame for this rule."""
    rows = []
    for rid, sub in ticket_df.groupby("race_id", sort=False):
        info = race_meta[rid]
        selected = prune_fn(sub, info)
        n = len(selected)
        true = tri_combo.get(rid)
        payout = int(tri_payout.get(rid, 0) or 0) if true and (true in set(selected["ticket"])) else 0
        hit = payout > 0
        rows.append({
            "rule": rule_name,
            "race_id": rid,
            "race_date": info["race_date"],
            "race_name": info["race_name"],
            "racecourse": info["racecourse"],
            "grade": info["grade"],
            "year": info["year"],
            "field_size": info["field_size"],
            "component_labels": info["component_labels"],
            "primary_component": info["component"],
            "n_tickets": n,
            "cost": n * TICKET_COST,
            "hit": int(hit),
            "payout": payout,
            "profit": payout - n * TICKET_COST,
            "winning_ticket": true if hit else None,
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
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def summarize_rule(per_race: pd.DataFrame, rule: str, rng):
    races = len(per_race)
    tickets = int(per_race["n_tickets"].sum())
    cost = int(per_race["cost"].sum())
    payout = int(per_race["payout"].sum())
    hits = int(per_race["hit"].sum())
    max_payout = int(per_race.loc[per_race["hit"] > 0, "payout"].max()) if hits else 0
    ci_lo, ci_hi = bootstrap_ci(
        per_race["cost"].to_numpy(dtype=float),
        per_race["payout"].to_numpy(dtype=float),
        rng,
    )
    return {
        "rule": rule,
        "races": races,
        "tickets": tickets,
        "investment_yen": cost,
        "hits": hits,
        "payout_yen": payout,
        "profit_yen": payout - cost,
        "roi": (payout - cost) / cost if cost else 0.0,
        "hit_rate": hits / races if races else 0.0,
        "avg_tickets_per_race": tickets / races if races else 0.0,
        "max_payout_yen": max_payout,
        "max_losing_streak": longest_loss_streak(per_race),
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "sample_warning": "LOW_SAMPLE" if races < LOW_SAMPLE_THRESHOLD else "",
    }


# ---- Monte Carlo (year shuffle, 100K initial, 100yen/ticket) --------------

def vectorized_simulate(bet_arr, profit_arr, orders, initial):
    """Run trials in parallel. bet_arr/profit_arr are (N,) per-race; orders is (trials, N)."""
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


def year_roi_ranking(per_race: pd.DataFrame):
    g = per_race.groupby("year").agg(cost=("cost", "sum"), payout=("payout", "sum"))
    g["roi"] = (g["payout"] - g["cost"]) / g["cost"].where(g["cost"] > 0, 1)
    return g.sort_values("roi").index.tolist()


def run_mc_for_rule(rule_per_race: pd.DataFrame, rng_mc):
    """Year-shuffle MC at initial=100K, 100yen/ticket. Returns summary dict."""
    if rule_per_race.empty:
        return None
    per = rule_per_race.sort_values("race_date", kind="stable").reset_index(drop=True)
    costs = per["cost"].to_numpy(dtype=float)
    payouts = per["payout"].to_numpy(dtype=float)
    profits = payouts - costs  # since hit -> raw payout, miss -> 0
    years = per["year"].to_numpy(dtype=np.int32)
    year_list = sorted(set(years.tolist()))
    year_indices = {y: np.where(years == y)[0] for y in year_list}

    N = len(per)
    orders = np.zeros((MC_TRIALS, N), dtype=np.int32)
    year_list_arr = np.array(year_list)
    for t in range(MC_TRIALS):
        perm = rng_mc.permutation(year_list_arr)
        orders[t] = np.concatenate([year_indices[y] for y in perm])

    final_bk, max_dd, min_bk, ruin = vectorized_simulate(costs, profits, orders, INITIAL)

    # worst_first scenario
    worst_year_order = year_roi_ranking(per)
    worst_order_idx = np.concatenate([year_indices[y] for y in worst_year_order])[None, :]
    wf_final, wf_dd, wf_min, wf_ruin = vectorized_simulate(costs, profits, worst_order_idx, INITIAL)

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

    log.info("loading D9 base (JRA flat G1/G2/G3)")
    races_df, entries_df, payouts_df = load_data(("G1", "G2", "G3"), True, True)
    base = collect_subsets(races_df, entries_df, payouts_df)
    d9 = apply_variant(base, "D9")
    # enrich with dark_pop / dark_odds (not in strategy_d_variants.collect_subsets output)
    entries_df["popularity"] = pd.to_numeric(entries_df["popularity"], errors="coerce").astype("Int64")
    entries_df["horse_number"] = pd.to_numeric(entries_df["horse_number"], errors="coerce").astype("Int64")
    entries_df["win_odds"] = pd.to_numeric(entries_df["win_odds"], errors="coerce")
    dark_meta = (entries_df[["race_id", "horse_number", "popularity", "win_odds"]]
                 .rename(columns={"horse_number": "dark_horse",
                                  "popularity": "dark_pop",
                                  "win_odds": "dark_odds"}))
    d9 = d9.merge(dark_meta, on=["race_id", "dark_horse"], how="left")
    log.info("D9 subsets=%d (with dark_pop/dark_odds enrichment)", len(d9))

    # collapse to per-race for race-level meta
    per_race = collapse_to_race(d9)
    per_race = tag_components(per_race)
    per_race["component_labels"] = make_component_labels(per_race)
    per_race["primary_component"] = per_race["component_labels"].apply(primary_component)
    race_meta = {
        r["race_id"]: {
            "race_date": r["race_date"],
            "race_name": r["race_name"],
            "racecourse": r["racecourse"],
            "grade": r["grade"],
            "year": r["year"],
            "field_size": r["field_size"],
            "component_labels": r["component_labels"],
            "component": r["primary_component"],
        }
        for _, r in per_race.iterrows()
    }

    # p1, p2 horse numbers per race (entries already typed above)
    p1_map = (entries_df[entries_df["popularity"] == 1]
              .drop_duplicates("race_id").set_index("race_id")["horse_number"]).to_dict()
    p2_map = (entries_df[entries_df["popularity"] == 2]
              .drop_duplicates("race_id").set_index("race_id")["horse_number"]).to_dict()
    p1_p2_map = {rid: (int(p1_map[rid]), int(p2_map[rid]))
                 for rid in p1_map if rid in p2_map}

    # trifecta lookup
    tri = (payouts_df[payouts_df["bet_type"] == "三連単"]
           .drop_duplicates("race_id").set_index("race_id"))
    tri_combo = tri["combination"].astype(str).to_dict()
    tri_payout = tri["payout_yen"].to_dict()

    # build ticket-level table once
    ticket_df = build_ticket_table(d9, p1_p2_map)
    log.info("ticket table size=%d", len(ticket_df))

    # ---- evaluate each rule
    summary_rows = []
    hits_rows = []
    rule_per_race = {}
    mc_summary_rows = []
    log.info("evaluating %d rules...", len(RULES))
    for rule_name, prune_fn in RULES.items():
        t0 = time.time()
        per = evaluate_rule(rule_name, prune_fn, ticket_df, race_meta, tri_combo, tri_payout)
        rule_per_race[rule_name] = per
        s = summarize_rule(per, rule_name, rng_boot)
        summary_rows.append(s)
        log.info("%s: races=%d tickets=%d ROI=%+.1f%% hits=%d (%.1fs)",
                 rule_name, s["races"], s["tickets"], s["roi"] * 100, s["hits"], time.time() - t0)

        # hits CSV: collect hit races
        hit_races = per[per["hit"] > 0].copy()
        for _, r in hit_races.iterrows():
            hits_rows.append({
                "rule": rule_name,
                "race_id": r["race_id"],
                "race_date": r["race_date"],
                "race_name": r["race_name"],
                "racecourse": r["racecourse"],
                "grade": r["grade"],
                "field_size": r["field_size"],
                "component_labels": r["component_labels"],
                "n_tickets": r["n_tickets"],
                "winning_ticket": r["winning_ticket"],
                "payout_yen": r["payout"],
                "investment_yen": r["cost"],
                "profit_yen": r["profit"],
            })

    summary_df = pd.DataFrame(summary_rows)

    # ---- missed hits analysis (vs P0)
    p0_per = rule_per_race["P0_full_d9"]
    p0_hit_set = set(p0_per[p0_per["hit"] > 0]["race_id"])
    p0_payout_map = p0_per.set_index("race_id")["payout"].to_dict()
    missed_rows = []
    for rule_name in RULES:
        per = rule_per_race[rule_name]
        rule_hit_set = set(per[per["hit"] > 0]["race_id"])
        kept = p0_hit_set & rule_hit_set
        missed = p0_hit_set - rule_hit_set
        missed_rows.append({
            "rule": rule_name,
            "p0_hits": len(p0_hit_set),
            "kept_hits": len(kept),
            "missed_hits": len(missed),
            "kept_payout_yen": int(sum(p0_payout_map[r] for r in kept)),
            "missed_payout_yen": int(sum(p0_payout_map[r] for r in missed)),
        })
    missed_df = pd.DataFrame(missed_rows)
    # merge into summary
    summary_df = summary_df.merge(missed_df, on="rule", how="left")

    # ---- Monte Carlo for each rule
    log.info("running MC for each rule (100K initial, 10K trials)...")
    for rule_name in RULES:
        per = rule_per_race[rule_name]
        t0 = time.time()
        mc = run_mc_for_rule(per, rng_mc)
        if mc is None:
            continue
        log.info("MC %s: ruin=%.2f%% median_final=%.0f worst_first_ruin=%s (%.1fs)",
                 rule_name, mc["ruin_rate"] * 100, mc["median_final_bankroll"],
                 mc["worst_first_ruin_flag"], time.time() - t0)
        mc_summary_rows.append({"rule": rule_name, **mc})
    mc_df = pd.DataFrame(mc_summary_rows)

    # ---- save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(PROCESSED_DIR / "d9_ticket_pruning_summary.csv",
                      index=False, encoding="utf-8-sig")
    pd.DataFrame(hits_rows).to_csv(PROCESSED_DIR / "d9_ticket_pruning_hits.csv",
                                   index=False, encoding="utf-8-sig")
    mc_df.to_csv(PROCESSED_DIR / "d9_ticket_pruning_monte_carlo.csv",
                 index=False, encoding="utf-8-sig")
    log.info("wrote 3 CSVs (summary=%d, hits=%d, mc=%d)",
             len(summary_df), len(hits_rows), len(mc_df))

    # ---- stdout
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== PRUNING SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== MONTE CARLO @ 100K (each pruning rule) ===")
    print(mc_df.to_string(index=False))


if __name__ == "__main__":
    main()

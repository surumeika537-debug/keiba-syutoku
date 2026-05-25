"""generate_tickets.py の出力 CSV と実際の payouts を突き合わせて paper trading log を作る。

入力:
  --tickets    generate_tickets.py が出した CSV (1行 = 1 ticket)
  --bankroll   初期 bankroll (default 100,000)
  --output     paper trading log の出力先 (default data/processed/paper_trading_log.csv)

出力 CSV (1 row = 1 race) 列:
  race_id, race_date, race_name,
  rule_name, snapshot_time, odds_source, generated_at, final_result_checked_at,
  total_tickets, total_stake_yen,
  tickets_bought, ticket (winning if hit),
  stake_yen, hit_flag, payout_yen, profit_yen,
  cumulative_profit_yen, bankroll_after, notes

Usage:
    python scripts/live/record_result.py \
        --tickets data/processed/live_tickets_2025_final.csv --bankroll 100000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.config import PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("live.record_result")


def load_trifecta_results() -> tuple[dict, dict]:
    """Return (combo_by_race, payout_by_race) for 三連単 results currently in DB."""
    engine = get_engine()
    payouts = pd.read_sql(
        "SELECT race_id, bet_type, combination, payout_yen FROM payouts "
        "WHERE bet_type = '三連単'",
        engine,
    )
    payouts = payouts.drop_duplicates("race_id")
    combo_by = payouts.set_index("race_id")["combination"].astype(str).to_dict()
    payout_by = payouts.set_index("race_id")["payout_yen"].to_dict()
    return combo_by, payout_by


def main():
    import datetime as _dt
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickets", type=Path, required=True,
                    help="path to live_tickets_*.csv from generate_tickets.py")
    ap.add_argument("--bankroll", type=int, default=100_000,
                    help="starting bankroll (default 100,000)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not args.tickets.exists():
        log.error("tickets CSV not found: %s", args.tickets)
        sys.exit(1)

    tickets = pd.read_csv(args.tickets, dtype={"race_id": str})
    log.info("loaded %d ticket rows from %s", len(tickets), args.tickets)

    # metadata propagation (with backward-compat defaults if columns missing)
    def first_meta(col, default=""):
        if col in tickets.columns and not tickets[col].empty:
            v = tickets[col].dropna().head(1)
            return str(v.iloc[0]) if not v.empty else default
        return default
    csv_rule_name = first_meta("rule_name", default="E_D9_P3_CAP4")
    csv_snapshot_time = first_meta("snapshot_time", default="final")
    csv_odds_source = first_meta("odds_source", default="db_final_odds")
    csv_generated_at = first_meta("generated_at", default="")
    final_result_checked_at = _dt.datetime.now().isoformat(timespec="seconds")

    combo_by, payout_by = load_trifecta_results()

    # group by race; preserve race-level metadata
    bankroll = float(args.bankroll)
    cumulative_profit = 0.0
    log_rows = []

    # process races in date order so cumulative makes sense
    for race_id, grp in (tickets.sort_values(["race_date", "race_id"], kind="stable")
                                  .groupby("race_id", sort=False)):
        race_id = str(race_id)
        race_tickets = grp["ticket"].astype(str).tolist()
        stake = int(grp["stake_yen"].sum())
        total_tickets = int(len(race_tickets))
        race_date = grp["race_date"].iloc[0]
        race_name = grp["race_name"].iloc[0]
        # row-level snapshot/source falls back to CSV-level if missing per row
        row_snapshot = str(grp.get("snapshot_time", pd.Series([csv_snapshot_time])).iloc[0])
        row_odds_src = str(grp.get("odds_source", pd.Series([csv_odds_source])).iloc[0])
        row_generated = str(grp.get("generated_at", pd.Series([csv_generated_at])).iloc[0])
        row_rule = str(grp.get("rule_name", pd.Series([csv_rule_name])).iloc[0])

        true_combo = combo_by.get(race_id)
        if true_combo is None:
            note = "no 三連単 payout in DB (race not yet settled?)"
            hit = False
            payout = 0
            winning_ticket = ""
        else:
            hit = true_combo in race_tickets
            payout = int(payout_by.get(race_id, 0) or 0) if hit else 0
            winning_ticket = true_combo if hit else ""
            note = "hit" if hit else f"miss (actual: {true_combo})"

        profit = payout - stake
        bankroll += profit
        cumulative_profit += profit

        log_rows.append({
            "race_id": race_id,
            "race_date": race_date,
            "race_name": race_name,
            "rule_name": row_rule,
            "snapshot_time": row_snapshot,
            "odds_source": row_odds_src,
            "generated_at": row_generated,
            "final_result_checked_at": final_result_checked_at,
            "total_tickets": total_tickets,
            "total_stake_yen": stake,
            "tickets_bought": ";".join(race_tickets),
            "ticket": winning_ticket,
            "stake_yen": stake,
            "hit_flag": bool(hit),
            "payout_yen": payout,
            "profit_yen": profit,
            "cumulative_profit_yen": int(cumulative_profit),
            "bankroll_after": round(bankroll, 2),
            "notes": note,
        })

    out_df = pd.DataFrame(log_rows)
    out_path = args.output or (PROCESSED_DIR / "paper_trading_log.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("wrote %s (%d rows)", out_path, len(out_df))

    # ---- terminal summary
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 60)

    n_races = len(out_df)
    n_hits = int(out_df["hit_flag"].sum())
    total_stake = int(out_df["stake_yen"].sum())
    total_payout = int(out_df["payout_yen"].sum())
    final = round(bankroll, 2)
    roi = (total_payout - total_stake) / total_stake if total_stake > 0 else 0.0
    hit_rate = n_hits / n_races if n_races else 0.0
    print()
    print("=== PAPER TRADING SUMMARY ===")
    print(f"  races            : {n_races}")
    print(f"  hits             : {n_hits} ({hit_rate:.1%})")
    print(f"  total_stake      : ¥{total_stake:,}")
    print(f"  total_payout     : ¥{total_payout:,}")
    print(f"  profit           : ¥{int(total_payout - total_stake):+,}")
    print(f"  ROI              : {roi:+.1%}")
    print(f"  initial_bankroll : ¥{args.bankroll:,}")
    print(f"  final_bankroll   : ¥{int(final):,}")
    print()
    print(out_df[["race_date", "race_name", "stake_yen", "hit_flag",
                  "payout_yen", "profit_yen", "bankroll_after"]].to_string(index=False))


if __name__ == "__main__":
    main()

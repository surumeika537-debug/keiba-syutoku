"""年別に「1番人気の挙動」を分解する分析スクリプト。

戦略Aや戦略B (1番人気を本命に置く戦略) のROIが年によって大きく振れる原因が、
1番人気の実力／オッズの変化なのか、相手 (2-5番人気) の入線パターンなのか
を切り分けるために用意。

出力 (CSV: data/processed/favorite_breakdown_by_year.csv) のカラム:
  year
  race_count
  top1_win_rate          : 1番人気が1着になった率
  top1_top2_rate         : 1番人気が2着以内に入った率
  top1_top3_rate         : 1番人気が3着以内に入った率
  top1_avg_odds          : 1番人気の平均単勝オッズ
  avg_trifecta_when_top1_wins  : 1番人気が1着のレースの三連単払戻平均
  avg_trifecta_when_top1_busts : 1番人気が3着以下/未走の三連単払戻平均
  pop2to5_in_2nd_rate    : 2着に2-5番人気が入った率
  pop2to5_in_3rd_rate    : 3着に2-5番人気が入った率
  strategy_a_natural_rate: 1着=1番人気 かつ 2着3着両方が2-5番人気 のレース率 (=戦略A当選条件)

Usage:
    python scripts/analysis/favorite_breakdown.py --jra-only --exclude-steeplechase --grades G1 G2 G3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.config import JRA_RACECOURSES, PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("favorite_breakdown")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grades", nargs="+", default=None)
    ap.add_argument("--jra-only", action="store_true")
    ap.add_argument("--exclude-steeplechase", action="store_true")
    args = ap.parse_args()

    engine = get_engine()
    races = pd.read_sql("SELECT * FROM races", engine, parse_dates=["race_date"])
    entries = pd.read_sql("SELECT * FROM entries", engine)
    payouts = pd.read_sql("SELECT * FROM payouts", engine)

    if args.grades:
        races = races[races["grade"].isin(args.grades)]
    if args.jra_only:
        races = races[races["racecourse"].isin(JRA_RACECOURSES)]
    if args.exclude_steeplechase:
        races = races[races["surface"] != "障害"]
    keep = set(races["race_id"])
    entries = entries[entries["race_id"].isin(keep)].copy()
    payouts = payouts[payouts["race_id"].isin(keep)].copy()

    races["year"] = races["race_date"].dt.year
    entries["popularity"] = pd.to_numeric(entries["popularity"], errors="coerce").astype("Int64")
    entries["finish_position"] = pd.to_numeric(entries["finish_position"], errors="coerce").astype("Int64")
    entries["win_odds"] = pd.to_numeric(entries["win_odds"], errors="coerce")

    # 1番人気の行 (race_idで一意化)
    p1 = (entries[entries["popularity"] == 1]
          .sort_values("horse_number")
          .drop_duplicates("race_id", keep="first")
          .set_index("race_id"))[["win_odds", "finish_position"]]
    p1.columns = ["top1_odds", "top1_finish"]

    # 各レースの1着/2着/3着の馬の popularity
    top3 = entries[entries["finish_position"].isin([1, 2, 3])]
    pos_to_pop = (top3.pivot_table(index="race_id", columns="finish_position",
                                   values="popularity", aggfunc="first")
                  .rename(columns={1: "pop_at_1", 2: "pop_at_2", 3: "pop_at_3"}))

    # 三連単払戻
    tri = (payouts[payouts["bet_type"] == "三連単"]
           .drop_duplicates("race_id").set_index("race_id")["payout_yen"])

    # マージ
    df = races.set_index("race_id")[["year"]].join(p1).join(pos_to_pop).join(tri.rename("trifecta_payout"))

    rows = []
    for year, g in df.groupby("year", dropna=True):
        rc = len(g)
        win_n = (g["top1_finish"] == 1).sum()
        top2_n = (g["top1_finish"] <= 2).sum()
        top3_n = (g["top1_finish"] <= 3).sum()
        bust_mask = g["top1_finish"].isna() | (g["top1_finish"] > 3)
        pop2to5_in_2 = g["pop_at_2"].between(2, 5).sum()
        pop2to5_in_3 = g["pop_at_3"].between(2, 5).sum()
        strategy_a_n = (
            (g["pop_at_1"] == 1)
            & (g["pop_at_2"].between(2, 5))
            & (g["pop_at_3"].between(2, 5))
        ).sum()
        rows.append({
            "year": int(year),
            "race_count": rc,
            "top1_win_rate": win_n / rc if rc else 0.0,
            "top1_top2_rate": top2_n / rc if rc else 0.0,
            "top1_top3_rate": top3_n / rc if rc else 0.0,
            "top1_avg_odds": float(g["top1_odds"].mean()),
            "avg_trifecta_when_top1_wins":
                float(g.loc[g["top1_finish"] == 1, "trifecta_payout"].mean())
                if (g["top1_finish"] == 1).any() else float("nan"),
            "avg_trifecta_when_top1_busts":
                float(g.loc[bust_mask, "trifecta_payout"].mean())
                if bust_mask.any() else float("nan"),
            "pop2to5_in_2nd_rate": pop2to5_in_2 / rc if rc else 0.0,
            "pop2to5_in_3rd_rate": pop2to5_in_3 / rc if rc else 0.0,
            "strategy_a_natural_rate": strategy_a_n / rc if rc else 0.0,
        })

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "favorite_breakdown_by_year.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("wrote %s (%d rows)", path, len(out))

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

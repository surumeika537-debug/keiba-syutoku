"""fetch_odds_snapshots.py が出力する realtime_snapshot_health_*.csv を集計。

評価項目:
  - realtime_json success rate
  - realtime_html success rate
  - race_result_page fallback rate
  - avg response time per layer
  - parse fail reason breakdown
  - block detection count (cloudflare / captcha / rate_limit)
  - empty odds rate (horses_fetched == 0)
  - races with missing favorites (popularity 1 or 2 が NULL のレース)

出力: data/processed/realtime_snapshot_health.csv (1 row per snapshot_label)

Usage:
    python scripts/analysis/realtime_snapshot_health.py
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.config import PROCESSED_DIR
from src.database import get_engine
from src.utils import force_utf8_stdout, setup_logger

force_utf8_stdout()
log = setup_logger("realtime_health")


SOURCE_REALTIME_JSON = "netkeiba_realtime_json"
SOURCE_REALTIME_HTML = "netkeiba_realtime_html"
SOURCE_RACE_RESULT = "netkeiba_race_result_page"
SOURCE_PLACEHOLDER = "db_final_odds_PLACEHOLDER"


def _load_all_health() -> pd.DataFrame:
    files = sorted(glob.glob(str(PROCESSED_DIR / "realtime_snapshot_health_*.csv")))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        try:
            d = pd.read_csv(f)
            parts.append(d)
        except Exception as e:
            log.warning("could not read %s: %s", f, e)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def _missing_favorites_count(snapshot_label: str) -> int:
    """How many races have NULL popularity for rank 1 OR 2 in odds_snapshots?"""
    engine = get_engine()
    from sqlalchemy import text
    df = pd.read_sql(
        text("SELECT race_id, popularity FROM odds_snapshots "
             "WHERE snapshot_time_label = :label"),
        engine, params={"label": snapshot_label},
    )
    if df.empty:
        return 0
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    by_race = df.groupby("race_id")["popularity"].apply(
        lambda s: set(int(x) for x in s.dropna().astype(int))
    )
    missing = by_race[~by_race.apply(lambda s: {1, 2}.issubset(s))]
    return int(len(missing))


def _safe_pct(num, den):
    if not den:
        return 0.0
    return float(num) / float(den)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args()

    df = _load_all_health()
    if df.empty:
        log.error("no realtime_snapshot_health_*.csv files found in %s", PROCESSED_DIR)
        sys.exit(1)
    log.info("loaded %d health rows across %d labels",
             len(df), df["snapshot_label"].nunique())

    summary_rows = []
    for label, g in df.groupby("snapshot_label"):
        total = len(g)
        json_success = int((g.get("json_status", "") == "success").sum())
        html_success = int(g.get("html_status", pd.Series(dtype=str))
                            .isin(["embedded_json", "bs4_table"]).sum())
        race_result_success = int((g.get("race_result_status", "") == "success").sum())
        final_source_counts = g.get("final_source", pd.Series(dtype=str)
                                  ).fillna("FAILED").value_counts().to_dict()

        # latency means (only over rows that recorded a number)
        def _mean(col):
            s = pd.to_numeric(g.get(col, pd.Series(dtype=float)), errors="coerce")
            return float(s.dropna().mean()) if s.dropna().size else None

        # fail reasons (json_status / html_status / race_result_status)
        json_fail_counts = g[g.get("json_status", "") != "success"].get(
            "json_status", pd.Series(dtype=str)).fillna("").value_counts().to_dict()
        html_fail_counts = g[~g.get("html_status", pd.Series(dtype=str))
                              .isin(["embedded_json", "bs4_table"])].get(
            "html_status", pd.Series(dtype=str)).fillna("").value_counts().to_dict()

        # block detection
        block_counts = g.get("block_type", pd.Series(dtype=str)
                              ).dropna().value_counts().to_dict()

        # empty odds: horses_fetched == 0 or NaN
        hf = pd.to_numeric(g.get("horses_fetched", pd.Series(dtype=float)), errors="coerce")
        empty_n = int(hf.fillna(0).eq(0).sum())

        # missing favorites in odds_snapshots
        missing_fav = _missing_favorites_count(label)

        summary_rows.append({
            "snapshot_label": label,
            "races_total": total,
            "json_success_count": json_success,
            "json_success_rate": _safe_pct(json_success, total),
            "html_success_count": html_success,
            "html_success_rate": _safe_pct(html_success, total),
            "race_result_fallback_count": race_result_success,
            "race_result_fallback_rate": _safe_pct(race_result_success, total),
            "fully_failed_count": int(g.get("final_source", pd.Series(dtype=str)
                                            ).isna().sum()),
            "avg_json_latency_ms": _mean("json_latency_ms"),
            "avg_html_latency_ms": _mean("html_latency_ms"),
            "avg_race_result_latency_ms": _mean("race_result_latency_ms"),
            "block_detection_count": int(g.get("block_type", pd.Series(dtype=str)
                                                ).dropna().shape[0]),
            "block_breakdown": "; ".join(f"{k}:{v}" for k, v in block_counts.items()),
            "empty_odds_count": empty_n,
            "empty_odds_rate": _safe_pct(empty_n, total),
            "missing_favorites_races": missing_fav,
            "final_source_breakdown": "; ".join(f"{k}:{v}" for k, v in final_source_counts.items()),
            "json_fail_reasons": "; ".join(f"{k}:{v}" for k, v in json_fail_counts.items() if k),
            "html_fail_reasons": "; ".join(f"{k}:{v}" for k, v in html_fail_counts.items() if k),
        })

    summary_df = pd.DataFrame(summary_rows)
    out_path = PROCESSED_DIR / "realtime_snapshot_health.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("wrote %s (%d rows)", out_path, len(summary_df))

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.float_format", "{:.3f}".format)
    print()
    print("=== REALTIME SNAPSHOT HEALTH ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

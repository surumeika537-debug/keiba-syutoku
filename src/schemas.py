"""SQLAlchemy table definitions. Kept Core-style (Table objects) for easy use with pandas to_sql/read_sql."""
from __future__ import annotations

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
)

metadata = MetaData()

races = Table(
    "races",
    metadata,
    Column("race_id", String, primary_key=True),
    Column("race_date", Date, nullable=True, index=True),
    Column("race_name", String, nullable=True),
    Column("grade", String, nullable=True, index=True),     # G1 / G2 / G3
    Column("racecourse", String, nullable=True, index=True),
    Column("surface", String, nullable=True),               # 芝 / ダート
    Column("distance", Integer, nullable=True),             # meters
    Column("track_condition", String, nullable=True),       # 良 / 稍重 / 重 / 不良
    Column("field_size", Integer, nullable=True),
)

entries = Table(
    "entries",
    metadata,
    Column("race_id", String, ForeignKey("races.race_id"), primary_key=True),
    Column("horse_number", Integer, primary_key=True),
    Column("frame_number", Integer, nullable=True),
    Column("horse_name", String, nullable=True),
    Column("jockey", String, nullable=True),
    Column("popularity", Integer, nullable=True, index=True),
    Column("win_odds", Float, nullable=True),                # REAL; "---" 等は NULL
    Column("finish_position", Integer, nullable=True),       # 完走時のみ INT, 中止/取消/除外/失格 は NULL
    Column("finish_status", String, nullable=True),          # 完走 / 中止 / 取消 / 除外 / 失格 / NULL
)

payouts = Table(
    "payouts",
    metadata,
    Column("race_id", String, ForeignKey("races.race_id"), primary_key=True),
    Column("bet_type", String, primary_key=True),       # 単勝 / 複勝 / 馬連 / 馬単 / 三連複 / 三連単 / etc.
    Column("combination", String, primary_key=True),    # e.g. "1-5-3"
    Column("payout_yen", Integer, nullable=True),
)


# Time-series odds snapshot table. Captures (popularity, win_odds) at named time points
# leading up to the race so paper trading / realtime decisions can be tested against
# real pre-race conditions instead of post-race final odds.
#
# snapshot_time_label: free-form, but use one of:
#   "60min" / "30min" / "10min" / "5min" / "final"
# source: identifier of the data origin, e.g. "netkeiba_realtime", "db_final_odds",
#         "db_final_odds_PLACEHOLDER" (when we backfilled snapshots from final odds
#         because no realtime source exists yet for that label).
odds_snapshots = Table(
    "odds_snapshots",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("race_id", String, ForeignKey("races.race_id"), nullable=False),
    Column("snapshot_time_label", String, nullable=False),
    Column("captured_at", DateTime, nullable=True),      # wall-clock time the odds were observed
    Column("horse_number", Integer, nullable=False),
    Column("popularity", Integer, nullable=True),
    Column("win_odds", Float, nullable=True),
    Column("source", String, nullable=True),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("race_id", "snapshot_time_label", "horse_number",
                     name="uq_odds_snapshot_per_horse"),
)


# --- indices -----------------------------------------------------------------
# Most single-column lookups are already covered:
#   - races.race_id            (PK)
#   - races.race_date          (index=True on the Column)
#   - races.grade              (index=True)
#   - races.racecourse         (index=True)
#   - entries.race_id          (left edge of composite PK)
#   - entries.popularity       (index=True)
#   - payouts.race_id          (left edge of composite PK (race_id,bet_type,combination))
#   - payouts.(race_id,bet_type) — also covered by that composite PK's leading prefix
#
# What is NOT yet covered: composite (entries.race_id, entries.popularity), which is the
# hottest pattern in backtest/analysis (gen tickets per race by popularity range).
Index("ix_entries_race_popularity", entries.c.race_id, entries.c.popularity)

# Also keep a date index on entries via the join surface — speeds up year/grade scans.
Index("ix_payouts_race_bet_type", payouts.c.race_id, payouts.c.bet_type)

# Hot lookups on odds_snapshots: by (race_id, label) for full-race snapshot reads,
# and by (race_id, horse_number) for per-horse drift comparison.
Index("ix_odds_snapshots_race_label", odds_snapshots.c.race_id,
      odds_snapshots.c.snapshot_time_label)
Index("ix_odds_snapshots_race_horse", odds_snapshots.c.race_id,
      odds_snapshots.c.horse_number)


ALL_TABLES = [races, entries, payouts, odds_snapshots]

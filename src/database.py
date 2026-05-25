"""SQLAlchemy engine + helpers. Thin wrapper so other modules don't deal with engine lifecycle."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.config import DB_URL, DB_DIR


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(DB_URL, future=True)

        # SQLite needs PRAGMA foreign_keys=ON per connection to enforce FKs.
        @event.listens_for(_engine, "connect")
        def _enable_fk(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, future=True)
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _SessionLocal is not None
    sess = _SessionLocal()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()

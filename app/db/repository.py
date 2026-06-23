"""Repository helpers for persisting and querying predictions.

The engine is created lazily from ``PULSELEDGER_DATABASE_URL`` (default: a
local SQLite file). All public functions return plain dicts so callers never
touch detached ORM instances.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import Prediction

_engine = None
_SessionLocal: Optional[sessionmaker] = None

DEFAULT_URL = "sqlite:///./pulseledger.db"


def get_database_url() -> str:
    return os.getenv("PULSELEDGER_DATABASE_URL", DEFAULT_URL)


def is_configured() -> bool:
    return _SessionLocal is not None


def configure(url: Optional[str] = None) -> None:
    """Create the engine + session factory and ensure tables exist."""
    global _engine, _SessionLocal
    url = url or get_database_url()
    connect_args = (
        {"check_same_thread": False} if url.startswith("sqlite") else {}
    )
    _engine = create_engine(url, future=True, connect_args=connect_args)
    _SessionLocal = sessionmaker(
        bind=_engine, expire_on_commit=False, future=True
    )
    Base.metadata.create_all(_engine)


def reset() -> None:
    """Dispose the engine and clear cached state (used by tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def _session_scope():
    if _SessionLocal is None:
        configure()
    assert _SessionLocal is not None
    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_prediction(record: Dict[str, Any]) -> None:
    """Insert a prediction; no-op if the prediction_id already exists."""
    with _session_scope() as session:
        exists = session.scalar(
            select(Prediction.id).where(
                Prediction.prediction_id == record["prediction_id"]
            )
        )
        if exists is not None:
            return
        session.add(Prediction(**record))


def list_predictions(limit: int = 25) -> List[Dict[str, Any]]:
    with _session_scope() as session:
        rows = session.scalars(
            select(Prediction)
            .order_by(Prediction.created_at.desc(), Prediction.id.desc())
            .limit(limit)
        ).all()
        return [row.to_summary() for row in rows]


def get_prediction(prediction_id: str) -> Optional[Dict[str, Any]]:
    with _session_scope() as session:
        row = session.scalar(
            select(Prediction).where(Prediction.prediction_id == prediction_id)
        )
        return row.to_full() if row else None


def count_predictions() -> int:
    with _session_scope() as session:
        return session.scalar(select(func.count(Prediction.id))) or 0

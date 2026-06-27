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
from app.db.models import Prediction, User

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
    # Managed Postgres (Render/Heroku) hands out a "postgres://" scheme that
    # SQLAlchemy 2.0 no longer accepts; normalize to the modern dialect URL.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
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


def list_predictions(
    limit: int = 25, user_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    with _session_scope() as session:
        query = select(Prediction)
        if user_id is not None:
            query = query.where(Prediction.user_id == user_id)
        rows = session.scalars(
            query.order_by(
                Prediction.created_at.desc(), Prediction.id.desc()
            ).limit(limit)
        ).all()
        return [row.to_summary() for row in rows]


def get_prediction(
    prediction_id: str, user_id: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    with _session_scope() as session:
        query = select(Prediction).where(
            Prediction.prediction_id == prediction_id
        )
        if user_id is not None:
            query = query.where(Prediction.user_id == user_id)
        row = session.scalar(query)
        return row.to_full() if row else None


def count_predictions(user_id: Optional[int] = None) -> int:
    with _session_scope() as session:
        query = select(func.count(Prediction.id))
        if user_id is not None:
            query = query.where(Prediction.user_id == user_id)
        return session.scalar(query) or 0


def sustainability_totals(
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate persisted energy/carbon/duration across a user's history.

    The metrics live in a JSON column, so we sum in Python rather than rely on
    backend-specific JSON SQL. Returns the latest measurement method/region so
    the dashboard can show how the figures were derived.
    """
    with _session_scope() as session:
        query = select(Prediction.sustainability).where(
            Prediction.sustainability.is_not(None)
        )
        if user_id is not None:
            query = query.where(Prediction.user_id == user_id)
        query = query.order_by(
            Prediction.created_at.desc(), Prediction.id.desc()
        )
        rows = session.scalars(query).all()

    total_energy = total_carbon = total_duration = 0.0
    count = 0
    method = region = grid_source = None
    for metrics in rows:
        if not isinstance(metrics, dict):
            continue
        count += 1
        total_energy += float(metrics.get("energy_kwh") or 0.0)
        total_carbon += float(metrics.get("carbon_emissions") or 0.0)
        total_duration += float(metrics.get("duration_seconds") or 0.0)
        if method is None:  # first row is the most recent
            method = metrics.get("method")
            region = metrics.get("region")
            grid_source = metrics.get("grid_source")

    return {
        "count": count,
        "total_energy_kwh": total_energy,
        "total_carbon_kg": total_carbon,
        "total_duration_seconds": total_duration,
        "method": method,
        "region": region,
        "grid_source": grid_source,
    }


# ── Users ────────────────────────────────────────────────────────────────


def create_user(
    email: str,
    hashed_password: str,
    full_name: str = "",
    role: str = "analyst",
) -> Dict[str, Any]:
    with _session_scope() as session:
        user = User(
            email=email.lower().strip(),
            hashed_password=hashed_password,
            full_name=full_name,
            role=role,
        )
        session.add(user)
        session.flush()
        return user.to_public()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Return the full record (incl. hash) for login verification."""
    with _session_scope() as session:
        user = session.scalar(
            select(User).where(User.email == email.lower().strip())
        )
        if not user:
            return None
        record = user.to_public()
        record["hashed_password"] = user.hashed_password
        return record


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with _session_scope() as session:
        user = session.get(User, user_id)
        return user.to_public() if user else None


def ensure_user(
    email: str,
    hashed_password: str,
    full_name: str = "",
    role: str = "analyst",
) -> None:
    """Create a user only if the email isn't already registered."""
    if get_user_by_email(email) is None:
        create_user(email, hashed_password, full_name, role)

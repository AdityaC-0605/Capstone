"""ORM models for PulseLedger persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    """An application user (analyst). Owns the assessments they score."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(120), default="")
    role: Mapped[str] = mapped_column(String(32), default="analyst")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def to_public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
        }


class Prediction(Base):
    """A persisted credit-risk assessment and its explanation."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    prediction_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True
    )
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    processing_time_ms: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32), default="")
    api_key_prefix: Mapped[str] = mapped_column(String(20), default="")
    application: Mapped[Dict[str, Any]] = mapped_column(JSON)
    explanation: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    sustainability: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    def _created_iso(self) -> str:
        value = self.created_at
        if value is None:
            return ""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    def to_summary(self) -> Dict[str, Any]:
        """Compact record for history lists."""
        app = self.application or {}
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self._created_iso(),
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "loan_amount": app.get("loan_amount"),
            "loan_purpose": app.get("loan_purpose"),
        }

    def to_full(self) -> Dict[str, Any]:
        """Full record for reconstructing a detail view."""
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self._created_iso(),
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "application": self.application or {},
            "explanation": self.explanation,
            "sustainability_metrics": self.sustainability,
        }

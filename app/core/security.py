"""Password hashing and JWT helpers for user authentication.

Self-contained (bcrypt + python-jose) so the auth layer stays small and
independent of the larger RBAC scaffolding in ``app.core.auth``.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
from jose import JWTError, jwt

JWT_SECRET = os.getenv("PULSELEDGER_JWT_SECRET", "dev-insecure-change-me")
JWT_ALGORITHM = "HS256"
TOKEN_TTL_HOURS = int(os.getenv("PULSELEDGER_JWT_TTL_HOURS", "168"))  # 7 days


def hash_password(password: str) -> str:
    # bcrypt only uses the first 72 bytes; truncate to avoid errors.
    raw = password.encode("utf-8")[:72]
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(
            password.encode("utf-8")[:72], hashed.encode("utf-8")
        )
    except Exception:
        return False


def create_access_token(
    subject: int,
    email: str = "",
    role: str = "analyst",
    expires_hours: Optional[int] = None,
) -> str:
    now = datetime.now(timezone.utc)
    ttl = expires_hours if expires_hours is not None else TOKEN_TTL_HOURS
    payload = {
        "sub": str(subject),
        "email": email,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=ttl),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None

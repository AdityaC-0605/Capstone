"""Shared test fixtures.

Every test runs against a fresh, isolated SQLite database so persistence
state never leaks between tests.
"""

import pytest

from app.db import repository


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PULSELEDGER_DATABASE_URL", f"sqlite:///{db_path}")
    repository.reset()
    repository.configure()
    yield
    repository.reset()

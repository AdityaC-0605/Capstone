"""Tests for the durable persistence layer (SQLAlchemy repository)."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

from app.api.inference_service import APIConfig, create_inference_service
from app.db import repository

SAMPLE = {
    "prediction_id": "pred_test123",
    "risk_score": 0.42,
    "risk_level": "medium",
    "confidence": 0.78,
    "processing_time_ms": 12.3,
    "model_version": "1.0.0",
    "api_key_prefix": "sk-test-ab",
    "application": {
        "age": 35,
        "loan_amount": 25000,
        "loan_purpose": "debt_consolidation",
    },
    "explanation": {"summary": "ok"},
    "sustainability": {"energy_kwh": 0.0004},
}


def test_save_list_get_count():
    repository.save_prediction(SAMPLE)
    assert repository.count_predictions() == 1

    items = repository.list_predictions(10)
    assert len(items) == 1
    assert items[0]["prediction_id"] == "pred_test123"
    assert items[0]["loan_amount"] == 25000

    full = repository.get_prediction("pred_test123")
    assert full is not None
    assert full["application"]["age"] == 35
    assert full["explanation"]["summary"] == "ok"


def test_save_is_idempotent():
    repository.save_prediction(SAMPLE)
    repository.save_prediction(SAMPLE)
    assert repository.count_predictions() == 1


def test_get_missing_returns_none():
    assert repository.get_prediction("pred_does_not_exist") is None


def test_survives_engine_recreation():
    """Data persists across an engine reset (i.e. a process restart)."""
    repository.save_prediction(SAMPLE)
    url = os.environ["PULSELEDGER_DATABASE_URL"]

    repository.reset()
    repository.configure(url)

    assert repository.count_predictions() == 1
    assert repository.get_prediction("pred_test123") is not None


def test_prediction_persisted_and_fetchable_by_id():
    config = APIConfig()
    config.enable_rate_limiting = False
    service = create_inference_service(config)
    client = TestClient(service.get_app())
    headers = {
        "Authorization": f"Bearer {service.api_key_manager.default_key}"
    }
    payload = {
        "application": {
            "age": 35,
            "income": 65000,
            "employment_length": 5,
            "debt_to_income_ratio": 0.30,
            "credit_score": 720,
            "loan_amount": 25000,
            "loan_purpose": "debt_consolidation",
            "home_ownership": "rent",
            "verification_status": "verified",
        },
        "include_explanation": True,
        "track_sustainability": False,
        "explanation_type": "shap",
    }

    created = client.post("/predict", json=payload, headers=headers)
    assert created.status_code == 200
    pid = created.json()["prediction_id"]

    # Persisted record is independently in the repository.
    assert repository.get_prediction(pid) is not None

    # And fetchable through the API by id.
    fetched = client.get(f"/predict/{pid}", headers=headers)
    assert fetched.status_code == 200
    data = fetched.json()["data"]
    assert data["prediction_id"] == pid
    assert data["application"]["credit_score"] == 720

    # Unknown id -> 404.
    missing = client.get("/predict/pred_unknown", headers=headers)
    assert missing.status_code == 404

"""The live-preview endpoint scores without persisting or explaining."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.inference_service import APIConfig, create_inference_service
from app.db import repository

PAYLOAD = {
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
    "include_explanation": False,
    "track_sustainability": False,
    "explanation_type": "shap",
}


def _client():
    config = APIConfig()
    config.enable_rate_limiting = False
    service = create_inference_service(config)
    return service, TestClient(service.get_app())


def test_preview_returns_score_without_persisting():
    service, client = _client()
    headers = {
        "Authorization": f"Bearer {service.api_key_manager.default_key}"
    }

    resp = client.post("/predict/preview", json=PAYLOAD, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high", "very_high"}
    assert 0.0 <= body["confidence"] <= 1.0

    # Preview must not write to the durable store or rolling history.
    assert repository.count_predictions() == 0
    history = client.get("/predict/history", headers=headers).json()
    assert history["count"] == 0


def test_preview_requires_auth():
    _, client = _client()
    resp = client.post(
        "/predict/preview",
        json=PAYLOAD,
        headers={"Authorization": "Bearer nope"},
    )
    assert resp.status_code == 401

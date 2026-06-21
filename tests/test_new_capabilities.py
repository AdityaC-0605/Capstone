"""Tests for the capabilities added in v1.1:

- Real federated-learning endpoint on the main API
- Real fairness/bias-audit endpoint on the main API
- Inference hardening: Pydantic v2 validation, persistent API key,
  server-side prediction history, and Prometheus metrics.
"""

from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from app.api.inference_service import (
    APIConfig,
    APIKeyManager,
    create_inference_service,
)
from app.api.main import app as main_api_app

SAMPLE_APPLICATION = {
    "age": 35,
    "income": 65000,
    "employment_length": 5,
    "debt_to_income_ratio": 0.30,
    "credit_score": 720,
    "loan_amount": 25000,
    "loan_purpose": "debt_consolidation",
    "home_ownership": "rent",
    "verification_status": "verified",
}


def _inference_client():
    config = APIConfig()
    config.enable_rate_limiting = False
    service = create_inference_service(config)
    client = TestClient(service.get_app())
    api_key = service.api_key_manager.default_key
    return service, client, {"Authorization": f"Bearer {api_key}"}


def test_federated_run_returns_real_metrics():
    client = TestClient(main_api_app)
    response = client.post(
        "/api/v1/federated/run",
        json={
            "number_of_clients": 2,
            "aggregation_rounds": 2,
            "local_epochs": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]

    assert len(data["round_metrics"]) == 2
    assert 0.0 < data["best_val_loss"] < 5.0
    first = data["round_metrics"][0]
    for field in ("average_val_loss", "average_val_accuracy"):
        assert field in first
    assert data["wall_time_seconds"] >= 0


def test_federated_run_rejects_out_of_range_params():
    client = TestClient(main_api_app)
    # 1 client is below the ge=2 bound -> validation error.
    response = client.post(
        "/api/v1/federated/run",
        json={"number_of_clients": 1, "aggregation_rounds": 2, "local_epochs": 1},
    )
    assert response.status_code == 422


def test_fairness_audit_reports_violations():
    client = TestClient(main_api_app)
    response = client.get(
        "/api/v1/fairness/audit", params={"samples": 400, "bias_strength": 1.5}
    )
    assert response.status_code == 200
    report = response.json()["data"]["report"]
    assert "summary" in report
    assert "total_tests" in report["summary"]


def test_prediction_history_and_metrics():
    _, client, headers = _inference_client()

    # No predictions yet.
    empty = client.get("/predict/history", headers=headers)
    assert empty.status_code == 200
    assert empty.json()["count"] == 0

    payload = {
        "application": SAMPLE_APPLICATION,
        "include_explanation": False,
        "track_sustainability": False,
        "explanation_type": "shap",
    }
    assert client.post("/predict", json=payload, headers=headers).status_code == 200

    history = client.get("/predict/history", headers=headers).json()
    assert history["count"] == 1
    assert history["total_served"] == 1
    assert history["items"][0]["risk_level"] in {
        "low",
        "medium",
        "high",
        "very_high",
    }

    metrics = client.get("/metrics").text
    assert "pulseledger_inference_predictions_total 1" in metrics


def test_pydantic_v2_validation_rejects_bad_enum():
    _, client, headers = _inference_client()
    bad = {
        "application": {**SAMPLE_APPLICATION, "loan_purpose": "yacht"},
        "include_explanation": False,
        "track_sustainability": False,
        "explanation_type": "shap",
    }
    assert client.post("/predict", json=bad, headers=headers).status_code == 422


def test_invalid_api_key_rejected():
    _, client, _ = _inference_client()
    payload = {
        "application": SAMPLE_APPLICATION,
        "include_explanation": False,
        "track_sustainability": False,
        "explanation_type": "shap",
    }
    response = client.post(
        "/predict",
        json=payload,
        headers={"Authorization": "Bearer not-a-real-key"},
    )
    assert response.status_code == 401


def test_api_key_persists_across_instances(tmp_path):
    key_file = tmp_path / "api_key.txt"
    # Ensure no env override interferes with the file-persistence path.
    previous = os.environ.pop("PULSELEDGER_API_KEY", None)
    try:
        first = APIKeyManager(key_path=str(key_file))
        second = APIKeyManager(key_path=str(key_file))
        assert first.default_key == second.default_key
        assert key_file.read_text(encoding="utf-8").strip() == first.default_key
    finally:
        if previous is not None:
            os.environ["PULSELEDGER_API_KEY"] = previous

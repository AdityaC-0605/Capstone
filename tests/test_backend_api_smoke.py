"""Backend API smoke tests."""

from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from app.api.inference_service import APIConfig, create_inference_service
from app.api.main import app as main_api_app
from app.sustainability.sustainability_monitor import SustainabilityMonitor


def test_main_api_endpoints():
    client = TestClient(main_api_app)

    root = client.get("/")
    health = client.get("/health")
    ready = client.get("/ready")
    status = client.get("/api/v1/status")

    assert root.status_code == 200
    assert health.status_code == 200
    assert ready.status_code == 200
    assert status.status_code == 200
    assert status.json()["data"]["features"]


def test_inference_api_prediction_flow():
    config = APIConfig()
    config.enable_rate_limiting = False

    service = create_inference_service(config)
    client = TestClient(service.get_app())
    api_key = next(iter(service.api_key_manager.api_keys.keys()))
    headers = {"Authorization": f"Bearer {api_key}"}

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
        "track_sustainability": True,
        "explanation_type": "shap",
    }

    root = client.get("/")
    health = client.get("/health")
    model_info = client.get("/model/info", headers=headers)
    api_key_info = client.get("/api-key/info", headers=headers)
    prediction = client.post("/predict", json=payload, headers=headers)

    assert root.status_code == 200
    assert health.status_code == 200
    assert model_info.status_code == 200
    assert api_key_info.status_code == 200
    assert prediction.status_code == 200

    body = prediction.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high", "very_high"}
    assert body["explanation"]["summary"]
    assert body["explanation"]["top_factors"]


def test_sustainability_monitor_tracking():
    monitor = SustainabilityMonitor()
    exp_id = monitor.start_experiment_tracking("backend-smoke")
    report = monitor.stop_experiment_tracking(exp_id)

    assert report["experiment_id"] == "backend-smoke"
    assert report["energy_kwh"] > 0
    assert report["carbon_emissions"] > 0

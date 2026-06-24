"""Server-side sustainability aggregation + the real NAS endpoints."""

from __future__ import annotations

import time

import pytest
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
    "track_sustainability": True,
}


def _client():
    config = APIConfig()
    config.enable_rate_limiting = False
    service = create_inference_service(config)
    return service, TestClient(service.get_app())


def _key_headers(service):
    return {"Authorization": f"Bearer {service.api_key_manager.default_key}"}


def test_repository_aggregates_scoped_totals():
    repository.save_prediction(
        {
            "prediction_id": "p1",
            "user_id": 1,
            "risk_score": 0.2,
            "risk_level": "low",
            "confidence": 0.9,
            "processing_time_ms": 5.0,
            "model_version": "v",
            "api_key_prefix": "k",
            "application": {},
            "explanation": None,
            "sustainability": {
                "energy_kwh": 0.001,
                "carbon_emissions": 0.0004,
                "duration_seconds": 0.5,
                "method": "cpu-time",
                "region": "US",
            },
        }
    )
    repository.save_prediction(
        {
            "prediction_id": "p2",
            "user_id": 1,
            "risk_score": 0.6,
            "risk_level": "high",
            "confidence": 0.8,
            "processing_time_ms": 5.0,
            "model_version": "v",
            "api_key_prefix": "k",
            "application": {},
            "explanation": None,
            "sustainability": {
                "energy_kwh": 0.003,
                "carbon_emissions": 0.0011,
                "duration_seconds": 1.5,
                "method": "cpu-time",
                "region": "US",
            },
        }
    )
    # A different user's row must not bleed into user 1's totals.
    repository.save_prediction(
        {
            "prediction_id": "p3",
            "user_id": 2,
            "risk_score": 0.4,
            "risk_level": "medium",
            "confidence": 0.7,
            "processing_time_ms": 5.0,
            "model_version": "v",
            "api_key_prefix": "k",
            "application": {},
            "explanation": None,
            "sustainability": {
                "energy_kwh": 99.0,
                "carbon_emissions": 99.0,
                "duration_seconds": 99.0,
            },
        }
    )

    totals = repository.sustainability_totals(user_id=1)
    assert totals["count"] == 2
    assert abs(totals["total_energy_kwh"] - 0.004) < 1e-9
    assert abs(totals["total_carbon_kg"] - 0.0015) < 1e-9
    assert abs(totals["total_duration_seconds"] - 2.0) < 1e-9
    assert totals["method"] == "cpu-time"
    assert totals["region"] == "US"


def test_summary_endpoint_reflects_a_scored_prediction():
    service, client = _client()
    headers = _key_headers(service)

    client.post("/predict", json=PAYLOAD, headers=headers)

    resp = client.get("/sustainability/summary", headers=headers)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["count"] >= 1
    assert data["total_energy_kwh"] >= 0.0
    assert data["method"] in {"cpu-time", "wall-clock", "codecarbon", "mock"}


def test_summary_requires_auth():
    _, client = _client()
    resp = client.get(
        "/sustainability/summary",
        headers={"Authorization": "Bearer nope"},
    )
    assert resp.status_code == 401


def test_nas_run_is_real_and_returns_scored_candidates():
    pytest.importorskip("torch")
    service, client = _client()
    headers = _key_headers(service)

    started = client.post("/sustainability/nas", headers=headers)
    assert started.status_code == 200
    assert started.json()["status"] == "running"

    result = None
    for _ in range(60):
        status = client.get(
            "/sustainability/nas/status", headers=headers
        ).json()
        if status["state"] in {"done", "error"}:
            result = status["result"]
            break
        time.sleep(0.5)

    assert result is not None, "NAS did not finish in time"
    assert result["status"] == "done", result
    assert result["configs_tested"] == 6
    assert len(result["candidates"]) >= 1
    first = result["candidates"][0]
    assert 0.0 <= first["auc"] <= 1.0
    assert first["carbon_cost"] > 0
    assert first["precision"] in {"fp32", "fp16", "int8"}

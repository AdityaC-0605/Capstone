"""The live fairness audit runs over the model's real persisted decisions."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.inference_service import APIConfig, create_inference_service


def _client():
    config = APIConfig()
    config.enable_rate_limiting = False
    service = create_inference_service(config)
    return service, TestClient(service.get_app())


def _headers(service):
    return {"Authorization": f"Bearer {service.api_key_manager.default_key}"}


def _payload(age: int) -> dict:
    return {
        "application": {
            "age": age,
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
    }


def test_live_audit_reports_insufficient_without_history():
    service, client = _client()
    resp = client.get("/fairness/audit/live", headers=_headers(service))
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["mode"] == "insufficient"


def test_live_audit_audits_real_decisions_by_age_band():
    service, client = _client()
    headers = _headers(service)

    # Score across two age bands so age has two well-sized groups.
    for i in range(40):
        age = 22 if i % 2 == 0 else 55
        r = client.post("/predict", json=_payload(age), headers=headers)
        assert r.status_code == 200

    resp = client.get("/fairness/audit/live", headers=headers)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["mode"] == "live", data
    assert data["audited"]["n_decisions"] >= 40
    attrs = {a["attribute"] for a in data["audited"]["attributes"]}
    assert "age" in attrs
    # The report carries the demographic-parity result shape the UI renders.
    assert "summary" in data["report"]
    assert data["report"]["summary"]["total_tests"] >= 1
    # Label-dependent metrics must be honestly flagged unavailable.
    assert "unavailable" in data["audited"]["label_dependent_metrics"].lower()


def test_live_audit_requires_auth():
    _, client = _client()
    resp = client.get(
        "/fairness/audit/live",
        headers={"Authorization": "Bearer nope"},
    )
    assert resp.status_code == 401

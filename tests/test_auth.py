"""Tests for user authentication and per-user (multi-tenant) scoping."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.inference_service import APIConfig, create_inference_service


def _client():
    config = APIConfig()
    config.enable_rate_limiting = False
    service = create_inference_service(config)
    return service, TestClient(service.get_app())


def _register(client, email, password="password123", name="Analyst"):
    return client.post(
        "/auth/register",
        json={"email": email, "password": password, "full_name": name},
    )


def _payload(loan_amount=25000):
    return {
        "application": {
            "age": 35,
            "income": 65000,
            "employment_length": 5,
            "debt_to_income_ratio": 0.30,
            "credit_score": 720,
            "loan_amount": loan_amount,
            "loan_purpose": "debt_consolidation",
            "home_ownership": "rent",
            "verification_status": "verified",
        },
        "include_explanation": False,
        "track_sustainability": False,
        "explanation_type": "shap",
    }


def test_register_login_and_me():
    _, client = _client()
    reg = _register(client, "a@example.com")
    assert reg.status_code == 200
    token = reg.json()["access_token"]
    assert reg.json()["user"]["email"] == "a@example.com"

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["data"]["email"] == "a@example.com"


def test_duplicate_registration_conflicts():
    _, client = _client()
    assert _register(client, "dup@example.com").status_code == 200
    assert _register(client, "dup@example.com").status_code == 409


def test_login_success_and_failure():
    _, client = _client()
    _register(client, "b@example.com", password="password123")

    ok = client.post(
        "/auth/login",
        json={"email": "b@example.com", "password": "password123"},
    )
    assert ok.status_code == 200 and ok.json()["access_token"]

    bad = client.post(
        "/auth/login",
        json={"email": "b@example.com", "password": "wrongpass"},
    )
    assert bad.status_code == 401

    unknown = client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "whatever"},
    )
    assert unknown.status_code == 401


def test_seeded_demo_user_can_log_in():
    _, client = _client()
    resp = client.post(
        "/auth/login",
        json={"email": "demo@pulseledger.app", "password": "demo12345"},
    )
    assert resp.status_code == 200


def test_predictions_are_scoped_per_user():
    _, client = _client()
    token_a = _register(client, "ua@example.com").json()["access_token"]
    token_b = _register(client, "ub@example.com").json()["access_token"]
    head_a = {"Authorization": f"Bearer {token_a}"}
    head_b = {"Authorization": f"Bearer {token_b}"}

    first = client.post("/predict", json=_payload(25000), headers=head_a)
    assert first.status_code == 200
    client.post("/predict", json=_payload(26000), headers=head_a)
    client.post("/predict", json=_payload(27000), headers=head_b)

    hist_a = client.get("/predict/history", headers=head_a).json()
    hist_b = client.get("/predict/history", headers=head_b).json()
    assert hist_a["count"] == 2
    assert hist_b["count"] == 1

    # User B cannot read user A's assessment by id.
    a_id = first.json()["prediction_id"]
    assert client.get(f"/predict/{a_id}", headers=head_b).status_code == 404
    assert client.get(f"/predict/{a_id}", headers=head_a).status_code == 200


def test_invalid_bearer_rejected():
    _, client = _client()
    resp = client.post(
        "/predict",
        json=_payload(),
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401

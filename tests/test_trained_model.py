"""Tests for serving the trained credit-risk model from the registry."""

from __future__ import annotations

from app.models.runtime_credit_model import (
    LightweightCreditRiskModel,
    _load_trained_model,
)

HIGH_RISK = {
    "age": 23,
    "income": 28000,
    "employment_length": 1,
    "debt_to_income_ratio": 0.58,
    "credit_score": 560,
    "loan_amount": 26000,
    "loan_purpose": "medical",
    "home_ownership": "rent",
    "verification_status": "not_verified",
}
LOW_RISK = {
    "age": 46,
    "income": 120000,
    "employment_length": 12,
    "debt_to_income_ratio": 0.14,
    "credit_score": 790,
    "loan_amount": 9000,
    "loan_purpose": "home_improvement",
    "home_ownership": "own",
    "verification_status": "verified",
}


def test_trained_artifact_loads_from_registry():
    # The committed artifact should load (sklearn pinned in requirements).
    assert _load_trained_model() is not None


def test_model_source_is_trained():
    model = LightweightCreditRiskModel()
    assert model.model_source == "trained"


def test_scores_directional_and_bounded():
    model = LightweightCreditRiskModel()
    high = model.predict(HIGH_RISK)["prediction"]
    low = model.predict(LOW_RISK)["prediction"]
    assert 0.0 <= low < high <= 1.0
    assert high > 0.6
    assert low < 0.3


def test_predict_proba_is_normalized():
    model = LightweightCreditRiskModel()
    proba = model.predict_proba(HIGH_RISK)
    assert len(proba[0]) == 2
    assert abs(sum(proba[0]) - 1.0) < 1e-6

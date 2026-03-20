"""Shared runtime credit risk model used by backend inference services."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


def _normalized_text(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value).strip().lower()


def normalize_credit_application(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize user inputs into a consistent runtime schema."""
    return {
        "age": int(float(data.get("age", 35))),
        "income": float(data.get("income", 60000.0)),
        "employment_length": int(float(data.get("employment_length", 5))),
        "debt_to_income_ratio": float(data.get("debt_to_income_ratio", 0.30)),
        "credit_score": int(float(data.get("credit_score", 680))),
        "loan_amount": float(data.get("loan_amount", 15000.0)),
        "loan_purpose": _normalized_text(data.get("loan_purpose"), "other"),
        "home_ownership": _normalized_text(data.get("home_ownership"), "rent"),
        "verification_status": _normalized_text(
            data.get("verification_status"), "not_verified"
        ),
    }


def compute_credit_risk_score(data: Mapping[str, Any]) -> float:
    """Compute a lightweight credit risk score on a 0-1 scale."""
    app = normalize_credit_application(data)

    age = app["age"]
    income = max(app["income"], 1.0)
    employment_length = app["employment_length"]
    debt_to_income_ratio = float(
        np.clip(app["debt_to_income_ratio"], 0.0, 1.0)
    )
    credit_score = int(np.clip(app["credit_score"], 300, 850))
    loan_amount = max(app["loan_amount"], 1000.0)
    loan_purpose = app["loan_purpose"]
    home_ownership = app["home_ownership"]
    verification_status = app["verification_status"]

    score = 0.5

    # Stronger credit profiles reduce risk, weaker profiles increase it.
    score -= (credit_score - 680) * 0.0012
    score += (debt_to_income_ratio - 0.35) * 0.55

    loan_to_income_ratio = loan_amount / income
    score += (loan_to_income_ratio - 0.35) * 0.45

    score -= min(employment_length, 15) * 0.015

    if age < 25:
        score += 0.06
    elif age > 60:
        score += 0.02

    income_scale = min(income / 120000.0, 1.0)
    score -= income_scale * 0.08

    purpose_adjustments = {
        "debt_consolidation": 0.03,
        "home_improvement": -0.01,
        "major_purchase": 0.02,
        "medical": 0.05,
        "vacation": 0.04,
        "wedding": 0.03,
        "moving": 0.02,
        "other": 0.01,
    }
    score += purpose_adjustments.get(loan_purpose, 0.01)

    home_adjustments = {
        "rent": 0.03,
        "own": -0.03,
        "mortgage": -0.01,
        "other": 0.01,
    }
    score += home_adjustments.get(home_ownership, 0.0)

    verification_adjustments = {
        "not_verified": 0.03,
        "verified": -0.02,
        "source_verified": -0.03,
    }
    score += verification_adjustments.get(verification_status, 0.0)

    return float(np.clip(score, 0.02, 0.98))


def compute_prediction_confidence(score: float) -> float:
    distance_from_boundary = abs(float(score) - 0.5)
    return float(np.clip(0.72 + distance_from_boundary * 0.45, 0.72, 0.97))


class LightweightCreditRiskModel:
    """Applicant-sensitive runtime model for inference and explainability."""

    def predict(self, data: Mapping[str, Any]) -> Dict[str, float]:
        normalized = normalize_credit_application(data)
        score = compute_credit_risk_score(normalized)
        return {
            "prediction": score,
            "confidence": compute_prediction_confidence(score),
        }

    def predict_proba(self, data: Mapping[str, Any]):
        score = self.predict(data)["prediction"]
        return [[1.0 - score, score]]

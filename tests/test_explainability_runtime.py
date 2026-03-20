"""Regression tests for runtime credit-risk explainability."""

from app.api.inference_service import LightweightCreditRiskModel
from app.explainability.explanation_service import ExplainerService


def test_explanation_reflects_credit_risk_inputs():
    model = LightweightCreditRiskModel()
    explainer = ExplainerService(model)

    high_risk = {
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
    low_risk = {
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

    high_pred = model.predict(high_risk)
    low_pred = model.predict(low_risk)

    high_exp = explainer.explain_prediction(high_risk, high_pred)
    low_exp = explainer.explain_prediction(low_risk, low_pred)

    assert high_pred["prediction"] > low_pred["prediction"]
    assert high_exp["prediction"] == high_pred["prediction"]
    assert low_exp["prediction"] == low_pred["prediction"]
    assert any(
        abs(value) > 0.0 for value in high_exp["feature_importance"].values()
    )
    assert any(
        abs(value) > 0.0 for value in low_exp["feature_importance"].values()
    )
    assert high_exp["feature_importance"] != low_exp["feature_importance"]
    assert high_exp["risk_level"] in {"low", "medium", "high", "very_high"}
    assert low_exp["risk_level"] in {"low", "medium", "high", "very_high"}
    assert high_exp["summary"]
    assert low_exp["summary"]
    assert high_exp["top_factors"]
    assert low_exp["top_factors"]

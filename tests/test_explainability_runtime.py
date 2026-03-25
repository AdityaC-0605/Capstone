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

    # ── Original assertions (backward compatibility) ──
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

    # ── Risk threshold context ──
    assert "risk_threshold_context" in high_exp
    assert "Score" in high_exp["risk_threshold_context"]
    assert "risk_thresholds" in high_exp
    assert len(high_exp["risk_thresholds"]) == 4

    # ── Recommendations ──
    assert "recommendations" in high_exp
    assert isinstance(high_exp["recommendations"], list)
    assert len(high_exp["recommendations"]) > 0
    for rec in high_exp["recommendations"]:
        assert "feature" in rec
        assert "type" in rec
        assert rec["type"] in {"action_needed", "preserve", "informational"}
        assert "recommendation" in rec

    # High-risk should have action_needed recs; low-risk should have preserve recs
    high_types = {r["type"] for r in high_exp["recommendations"]}
    low_types = {r["type"] for r in low_exp["recommendations"]}
    assert "action_needed" in high_types
    assert "preserve" in low_types

    # Low-risk preserve recs should have "advisory" field
    for rec in low_exp["recommendations"]:
        if rec["type"] == "preserve":
            assert "advisory" in rec
            assert "Do NOT change" in rec["advisory"]

    # ── Counterfactual ──
    assert "counterfactual" in high_exp
    assert high_exp["counterfactual"]["needed"] is True
    assert len(high_exp["counterfactual"]["changes"]) > 0
    for feature, change in high_exp["counterfactual"]["changes"].items():
        assert "current_value" in change
        assert "suggested_target" in change
        assert "action" in change

    assert low_exp["counterfactual"]["needed"] is False
    assert len(low_exp["counterfactual"]["changes"]) == 0

    # ── Risk groups ──
    assert "risk_groups" in high_exp
    assert isinstance(high_exp["risk_groups"], dict)
    expected_groups = {
        "financial_strength",
        "debt_burden",
        "stability",
        "loan_context",
    }
    assert set(high_exp["risk_groups"].keys()) == expected_groups
    for group_key, group in high_exp["risk_groups"].items():
        assert "label" in group
        assert "impact" in group
        assert group["impact"] in {"risk_increase", "risk_decrease", "neutral"}
        assert "total_contribution" in group
        assert "features" in group
        assert isinstance(group["features"], list)

    # ── Confidence ──
    assert "confidence" in high_exp
    assert high_exp["confidence"]["level"] in {"low", "medium", "high"}
    assert 0.0 <= high_exp["confidence"]["score"] <= 1.0
    assert "reason" in high_exp["confidence"]
    # High-risk profile with strong signals should have high confidence
    assert high_exp["confidence"]["level"] == "high"

    # ── Methodology ──
    assert "methodology" in high_exp
    assert "method" in high_exp["methodology"]
    assert "description" in high_exp["methodology"]
    assert "interpretation" in high_exp["methodology"]
    assert "baseline" in high_exp["methodology"]
    assert "baseline_values" in high_exp["methodology"]["baseline"]
    assert len(high_exp["methodology"]["baseline"]["baseline_values"]) > 0

    # ── Top factors structure ──
    for factor in high_exp["top_factors"]:
        assert "feature" in factor
        assert "label" in factor
        assert "value" in factor
        assert "impact" in factor
        assert "contribution" in factor
        assert "magnitude" in factor
        assert factor["magnitude"] in {"strong", "moderate", "minor"}
        assert "benchmark_context" in factor
        assert "description" in factor
        # Grammar check: no "is strong " in descriptions
        assert "is strong " not in factor["description"]
        assert "is moderate " not in factor["description"]
        assert "is minor " not in factor["description"]

    # ── Summary narrative style ──
    assert "This applicant presents a" in high_exp["summary"]
    assert "primary driver" in high_exp["summary"]

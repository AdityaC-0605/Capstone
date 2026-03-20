"""Utility helpers for API-safe explainability."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

FEATURE_ORDER: List[str] = [
    "age",
    "income",
    "employment_length",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "loan_purpose",
    "home_ownership",
    "verification_status",
]

CATEGORY_MAPS: Dict[str, Dict[str, float]] = {
    "loan_purpose": {
        "debt_consolidation": 0.0,
        "home_improvement": 1.0,
        "major_purchase": 2.0,
        "medical": 3.0,
        "vacation": 4.0,
        "wedding": 5.0,
        "moving": 6.0,
        "other": 7.0,
    },
    "home_ownership": {"rent": 0.0, "own": 1.0, "mortgage": 2.0, "other": 3.0},
    "verification_status": {
        "not_verified": 0.0,
        "verified": 1.0,
        "source_verified": 2.0,
    },
}

REVERSE_CATEGORY_MAPS: Dict[str, Dict[int, str]] = {
    key: {int(v): k for k, v in values.items()}
    for key, values in CATEGORY_MAPS.items()
}


DEFAULT_INPUT: Dict[str, Any] = {
    "age": 35,
    "income": 60000.0,
    "employment_length": 5,
    "debt_to_income_ratio": 0.30,
    "credit_score": 680,
    "loan_amount": 15000.0,
    "loan_purpose": "other",
    "home_ownership": "rent",
    "verification_status": "not_verified",
}


def prediction_to_float(prediction_result: Any) -> float:
    """Extract a numeric prediction score from model output."""
    if isinstance(prediction_result, Mapping):
        if "prediction" in prediction_result:
            return float(prediction_result["prediction"])
        if "score" in prediction_result:
            return float(prediction_result["score"])

    if isinstance(prediction_result, (list, tuple, np.ndarray)):
        arr = np.asarray(prediction_result).reshape(-1)
        if arr.size:
            return float(arr[0])

    try:
        return float(prediction_result)
    except Exception:
        return 0.5


def risk_level_from_score(score: float) -> str:
    if score < 0.25:
        return "low"
    if score < 0.5:
        return "medium"
    if score < 0.75:
        return "high"
    return "very_high"


def _encode_value(feature: str, value: Any) -> float:
    if feature in CATEGORY_MAPS:
        if value is None:
            value = DEFAULT_INPUT[feature]
        return float(CATEGORY_MAPS[feature].get(str(value).lower(), 0.0))

    if value is None:
        value = DEFAULT_INPUT[feature]

    try:
        return float(value)
    except Exception:
        return float(DEFAULT_INPUT[feature])


def encode_feature_dict(data: Mapping[str, Any]) -> np.ndarray:
    """Encode request payload into numeric vector with safe defaults."""
    return np.array(
        [
            _encode_value(feature, data.get(feature))
            for feature in FEATURE_ORDER
        ],
        dtype=np.float32,
    )


def decode_feature_vector(vector: Sequence[float]) -> Dict[str, Any]:
    """Decode numeric vector back into model input dict."""
    out: Dict[str, Any] = {}
    for idx, feature in enumerate(FEATURE_ORDER):
        val = float(vector[idx])
        if feature in REVERSE_CATEGORY_MAPS:
            out[feature] = REVERSE_CATEGORY_MAPS[feature].get(
                int(round(val)), DEFAULT_INPUT[feature]
            )
        else:
            out[feature] = val
    return out


def predict_score(model: Any, input_data: Mapping[str, Any]) -> float:
    """Predict score with single-model and ensemble-style fallbacks."""
    if model is None:
        return 0.5

    # Primary model API
    if hasattr(model, "predict"):
        try:
            return prediction_to_float(model.predict(dict(input_data)))
        except Exception:
            pass

    # Ensemble fallback (weighted average if available)
    submodels = getattr(model, "models", None)
    if submodels:
        if isinstance(submodels, Mapping):
            items = list(submodels.items())
            model_list = [m for _, m in items]
            names = [k for k, _ in items]
        else:
            model_list = list(submodels)
            names = [str(i) for i in range(len(model_list))]

        weights_obj = getattr(model, "weights", None)
        if isinstance(weights_obj, Mapping):
            raw_weights = [float(weights_obj.get(name, 1.0)) for name in names]
        elif isinstance(weights_obj, Sequence):
            raw_weights = [float(w) for w in weights_obj[: len(model_list)]]
            if len(raw_weights) < len(model_list):
                raw_weights += [1.0] * (len(model_list) - len(raw_weights))
        else:
            raw_weights = [1.0] * len(model_list)

        denom = sum(raw_weights) or 1.0
        score = 0.0
        for w, submodel in zip(raw_weights, model_list):
            if hasattr(submodel, "predict"):
                try:
                    score += (w / denom) * prediction_to_float(
                        submodel.predict(dict(input_data))
                    )
                except Exception:
                    continue
        return float(score) if score else 0.5

    return 0.5


def sorted_feature_importance(
    feature_names: Sequence[str],
    contributions: Sequence[float],
    max_features: int,
) -> Dict[str, float]:
    pairs = list(zip(feature_names, contributions))
    pairs.sort(key=lambda item: abs(float(item[1])), reverse=True)
    top = pairs[:max_features]
    return {name: float(value) for name, value in top}


def humanize_feature_name(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def describe_feature_impact(
    feature_name: str,
    value: Any,
    contribution: float,
    input_data: Mapping[str, Any],
) -> str:
    direction = (
        "increasing"
        if contribution > 0
        else "reducing" if contribution < 0 else "having little effect on"
    )

    if feature_name == "credit_score":
        return f"Credit score of {int(float(value))} is {direction} the predicted risk."
    if feature_name == "debt_to_income_ratio":
        return f"Debt-to-income ratio of {float(value):.0%} is {direction} the predicted risk."
    if feature_name == "loan_amount":
        income = max(float(input_data.get("income", 1.0)), 1.0)
        loan_to_income = float(value) / income
        return (
            f"Loan amount of ${float(value):,.0f} "
            f"({loan_to_income:.0%} of income) is {direction} the predicted risk."
        )
    if feature_name == "income":
        return f"Annual income of ${float(value):,.0f} is {direction} the predicted risk."
    if feature_name == "employment_length":
        years = int(float(value))
        return f"Employment length of {years} years is {direction} the predicted risk."
    if feature_name == "age":
        return f"Applicant age of {int(float(value))} is {direction} the predicted risk."
    if feature_name == "loan_purpose":
        return (
            f"Loan purpose '{str(value)}' is {direction} the predicted risk."
        )
    if feature_name == "home_ownership":
        return f"Home ownership status '{str(value)}' is {direction} the predicted risk."
    if feature_name == "verification_status":
        return f"Verification status '{str(value)}' is {direction} the predicted risk."

    return f"{humanize_feature_name(feature_name)} is {direction} the predicted risk."


def build_ranked_explanation_factors(
    input_data: Mapping[str, Any],
    feature_importance: Mapping[str, float],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        feature_importance.items(),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )

    factors: List[Dict[str, Any]] = []
    for feature_name, contribution in ranked[:limit]:
        value = input_data.get(feature_name, DEFAULT_INPUT.get(feature_name))
        factors.append(
            {
                "feature": feature_name,
                "label": humanize_feature_name(feature_name),
                "value": value,
                "impact": (
                    "risk_increase"
                    if contribution > 0
                    else "risk_decrease" if contribution < 0 else "neutral"
                ),
                "contribution": float(contribution),
                "description": describe_feature_impact(
                    feature_name, value, float(contribution), input_data
                ),
            }
        )

    return factors


def build_prediction_summary(
    prediction: float, risk_level: str, factors: Sequence[Mapping[str, Any]]
) -> str:
    if not factors:
        return (
            f"The applicant is predicted as {risk_level} risk "
            f"with a score of {prediction:.3f}."
        )

    largest_factor = factors[0]
    top_increase = next(
        (
            factor
            for factor in factors
            if factor.get("impact") == "risk_increase"
        ),
        None,
    )
    top_decrease = next(
        (
            factor
            for factor in factors
            if factor.get("impact") == "risk_decrease"
        ),
        None,
    )

    summary = (
        f"The applicant is predicted as {risk_level} risk with a score of "
        f"{prediction:.3f}. The largest overall effect comes from "
        f"{largest_factor['label'].lower()}, which is "
        f"{str(largest_factor['impact']).replace('_', ' ')}."
    )

    if top_increase is not None and top_decrease is not None:
        return (
            f"{summary} The main risk increase comes from "
            f"{top_increase['label'].lower()}, while the main risk decrease "
            f"comes from {top_decrease['label'].lower()}."
        )

    if top_increase is not None:
        return (
            f"{summary} The main risk increase comes from "
            f"{top_increase['label'].lower()}."
        )

    if top_decrease is not None:
        return (
            f"{summary} The main risk decrease comes from "
            f"{top_decrease['label'].lower()}."
        )

    return summary

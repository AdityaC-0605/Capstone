"""Utility helpers for API-safe explainability."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

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

# ── Risk-level thresholds (single source of truth) ──────────────────────

RISK_THRESHOLDS: List[Dict[str, Any]] = [
    {"ceiling": 0.25, "label": "low", "tag": "Low Risk"},
    {"ceiling": 0.50, "label": "medium", "tag": "Medium Risk"},
    {"ceiling": 0.75, "label": "high", "tag": "High Risk"},
    {"ceiling": 1.01, "label": "very_high", "tag": "Very High Risk"},
]

# ── Feature benchmark context for richer explanations ───────────────────

FEATURE_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "credit_score": {
        "excellent": 750,
        "good": 700,
        "fair": 650,
        "poor": 550,
        "unit": "points",
        "direction": "higher_is_better",
        "description": (
            "Measures overall creditworthiness based on repayment history, "
            "credit utilization, and account age."
        ),
    },
    "debt_to_income_ratio": {
        "low": 0.20,
        "moderate": 0.36,
        "high": 0.50,
        "unit": "ratio",
        "direction": "lower_is_better",
        "description": (
            "Proportion of gross monthly income committed to debt payments. "
            "Lenders typically prefer ratios below 36%."
        ),
    },
    "income": {
        "low": 30000,
        "median": 60000,
        "high": 100000,
        "unit": "USD/year",
        "direction": "higher_is_better",
        "description": "Annual gross income; higher values reduce perceived risk.",
    },
    "loan_amount": {
        "small": 5000,
        "medium": 15000,
        "large": 30000,
        "unit": "USD",
        "direction": "lower_is_better",
        "description": (
            "Requested loan value. Assessed relative to income to gauge "
            "repayment capacity."
        ),
    },
    "employment_length": {
        "short": 2,
        "moderate": 5,
        "long": 10,
        "unit": "years",
        "direction": "higher_is_better",
        "description": (
            "Duration of current employment. Longer tenure signals income "
            "stability."
        ),
    },
    "age": {
        "young": 25,
        "prime": 40,
        "senior": 60,
        "unit": "years",
        "direction": "neutral",
        "description": (
            "Applicant's age; very young or very senior profiles carry "
            "modestly higher risk."
        ),
    },
    "loan_purpose": {
        "unit": "category",
        "direction": "contextual",
        "description": (
            "Purpose of the loan. Medical and vacation loans carry higher risk "
            "premiums; home improvement loans are viewed more favorably."
        ),
    },
    "home_ownership": {
        "unit": "category",
        "direction": "contextual",
        "description": (
            "Housing status. Homeowners are associated with lower risk; renters "
            "carry a slight risk premium."
        ),
    },
    "verification_status": {
        "unit": "category",
        "direction": "contextual",
        "description": (
            "Whether the applicant's income has been independently verified. "
            "Verified or source-verified status reduces risk."
        ),
    },
}

# ── Feature grouping for risk categories ────────────────────────────────

FEATURE_GROUPS: Dict[str, Dict[str, Any]] = {
    "financial_strength": {
        "label": "Financial Strength",
        "features": ["income", "credit_score"],
        "description": (
            "Core financial health indicators reflecting earning capacity "
            "and credit history."
        ),
    },
    "debt_burden": {
        "label": "Debt Burden",
        "features": ["debt_to_income_ratio", "loan_amount"],
        "description": (
            "Current and requested debt obligations relative to the "
            "applicant's ability to repay."
        ),
    },
    "stability": {
        "label": "Stability & Verification",
        "features": ["employment_length", "verification_status", "age"],
        "description": (
            "Indicators of income stability, tenure, and whether "
            "financial information has been independently confirmed."
        ),
    },
    "loan_context": {
        "label": "Loan Context",
        "features": ["loan_purpose", "home_ownership"],
        "description": (
            "Contextual factors about the loan's purpose and the "
            "applicant's housing situation."
        ),
    },
}

# ── Counterfactual target thresholds ────────────────────────────────────
# These are the "safe" values that would move an applicant toward lower risk.

COUNTERFACTUAL_TARGETS: Dict[str, Dict[str, Any]] = {
    "credit_score": {
        "target": 700,
        "direction": "increase",
        "label": "Increase to \u2265 700 (good range)",
    },
    "debt_to_income_ratio": {
        "target": 0.35,
        "direction": "decrease",
        "label": "Reduce to \u2264 35%",
    },
    "income": {
        "target": 60000,
        "direction": "increase",
        "label": "Increase to \u2265 $60,000/year",
    },
    "loan_amount": {
        "target": 15000,
        "direction": "decrease",
        "label": "Reduce to \u2264 $15,000",
    },
    "employment_length": {
        "target": 5,
        "direction": "increase",
        "label": "Extend to \u2265 5 years",
    },
    "verification_status": {
        "target": "verified",
        "direction": "change",
        "label": "Get income independently verified",
    },
    "home_ownership": {
        "target": "own",
        "direction": "change",
        "label": "Transition from renting to homeownership (long-term)",
    },
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
    """Determine the categorical risk level from a 0-1 score."""
    for threshold in RISK_THRESHOLDS:
        if score < threshold["ceiling"]:
            return str(threshold["label"])
    return "very_high"


def risk_threshold_context(score: float) -> str:
    """Return a short sentence explaining where the score falls."""
    for idx, threshold in enumerate(RISK_THRESHOLDS):
        if score < threshold["ceiling"]:
            floor = RISK_THRESHOLDS[idx - 1]["ceiling"] if idx > 0 else 0.0
            return (
                f"Score {score:.3f} falls in the {threshold['tag']} band "
                f"({floor:.2f}\u2013{threshold['ceiling']:.2f})."
            )
    last = RISK_THRESHOLDS[-1]
    floor = RISK_THRESHOLDS[-2]["ceiling"] if len(RISK_THRESHOLDS) > 1 else 0.0
    return (
        f"Score {score:.3f} falls in the {last['tag']} band "
        f"({floor:.2f}\u2013{last['ceiling']:.2f})."
    )


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


# ── Benchmark-aware feature descriptions ────────────────────────────────


def _credit_score_context(value: float) -> str:
    """Return a contextual bracket label for the credit score."""
    v = int(float(value))
    benchmarks = FEATURE_BENCHMARKS["credit_score"]
    if v >= benchmarks["excellent"]:
        return "excellent"
    if v >= benchmarks["good"]:
        return "good"
    if v >= benchmarks["fair"]:
        return "fair"
    return "poor"


def _dti_context(value: float) -> str:
    """Return a contextual label for the debt-to-income ratio."""
    v = float(value)
    benchmarks = FEATURE_BENCHMARKS["debt_to_income_ratio"]
    if v <= benchmarks["low"]:
        return "low"
    if v <= benchmarks["moderate"]:
        return "moderate"
    return "high"


def _employment_context(value: float) -> str:
    """Return a contextual label for employment length."""
    v = int(float(value))
    benchmarks = FEATURE_BENCHMARKS["employment_length"]
    if v >= benchmarks["long"]:
        return "long"
    if v >= benchmarks["moderate"]:
        return "moderate"
    return "short"


def _magnitude_label(contribution: float) -> str:
    """Return a human-friendly magnitude label (noun/adjective form)."""
    mag = abs(contribution)
    if mag > 0.08:
        return "strong"
    if mag > 0.03:
        return "moderate"
    return "minor"


_ADVERB_MAP = {"strong": "strongly", "moderate": "moderately", "minor": "slightly"}


def _magnitude_adverb(contribution: float) -> str:
    """Return a grammatically correct adverb form for descriptions."""
    return _ADVERB_MAP.get(_magnitude_label(contribution), "slightly")


def describe_feature_impact(
    feature_name: str,
    value: Any,
    contribution: float,
    input_data: Mapping[str, Any],
) -> str:
    """Return a detailed, benchmark-aware description of a feature's impact."""
    direction = (
        "increasing"
        if contribution > 0
        else "reducing" if contribution < 0 else "having little effect on"
    )
    strength = _magnitude_adverb(contribution)

    if feature_name == "credit_score":
        bracket = _credit_score_context(value)
        return (
            f"Credit score of {int(float(value))} ({bracket} range, "
            f"benchmark \u2265750 for excellent) is {strength} {direction} "
            f"the predicted risk (contribution: {contribution:+.4f})."
        )

    if feature_name == "debt_to_income_ratio":
        bracket = _dti_context(value)
        return (
            f"Debt-to-income ratio of {float(value):.0%} ({bracket}; "
            f"lenders prefer \u226436%) is {strength} {direction} "
            f"the predicted risk (contribution: {contribution:+.4f})."
        )

    if feature_name == "loan_amount":
        income = max(float(input_data.get("income", 1.0)), 1.0)
        loan_to_income = float(value) / income
        return (
            f"Loan amount of ${float(value):,.0f} "
            f"({loan_to_income:.0%} of income) is {strength} {direction} "
            f"the predicted risk (contribution: {contribution:+.4f})."
        )

    if feature_name == "income":
        median = FEATURE_BENCHMARKS["income"]["median"]
        relative = "above" if float(value) > median else "below"
        return (
            f"Annual income of ${float(value):,.0f} ({relative} the "
            f"${median:,.0f} median) is {strength} {direction} "
            f"the predicted risk (contribution: {contribution:+.4f})."
        )

    if feature_name == "employment_length":
        years = int(float(value))
        bracket = _employment_context(value)
        return (
            f"Employment length of {years} year{'s' if years != 1 else ''} "
            f"({bracket} tenure) is {strength} {direction} "
            f"the predicted risk (contribution: {contribution:+.4f})."
        )

    if feature_name == "age":
        age_val = int(float(value))
        note = ""
        if age_val < 25:
            note = " (younger applicants carry a modest risk premium)"
        elif age_val > 60:
            note = " (senior applicants carry a slight risk premium)"
        return (
            f"Applicant age of {age_val}{note} is {strength} {direction} "
            f"the predicted risk (contribution: {contribution:+.4f})."
        )

    if feature_name == "loan_purpose":
        purpose_risk: Dict[str, str] = {
            "medical": "higher-risk purpose",
            "vacation": "higher-risk purpose",
            "debt_consolidation": "moderate-risk purpose",
            "wedding": "moderate-risk purpose",
            "moving": "moderate-risk purpose",
            "major_purchase": "moderate-risk purpose",
            "home_improvement": "lower-risk purpose",
            "other": "neutral purpose",
        }
        label = purpose_risk.get(str(value).lower(), "unclassified purpose")
        return (
            f"Loan purpose '{str(value)}' ({label}) is {strength} "
            f"{direction} the predicted risk (contribution: {contribution:+.4f})."
        )

    if feature_name == "home_ownership":
        ownership_note: Dict[str, str] = {
            "own": "homeowners are associated with lower risk",
            "mortgage": "mortgage holders carry slightly lower risk",
            "rent": "renters carry a modest risk premium",
            "other": "non-standard housing status",
        }
        note = ownership_note.get(str(value).lower(), "")
        return (
            f"Home ownership status '{str(value)}' ({note}) is {strength} "
            f"{direction} the predicted risk (contribution: {contribution:+.4f})."
        )

    if feature_name == "verification_status":
        ver_note: Dict[str, str] = {
            "verified": "income independently confirmed",
            "source_verified": "income source confirmed",
            "not_verified": "income not independently verified",
        }
        note = ver_note.get(str(value).lower(), "")
        return (
            f"Verification status '{str(value)}' ({note}) is {strength} "
            f"{direction} the predicted risk (contribution: {contribution:+.4f})."
        )

    return (
        f"{humanize_feature_name(feature_name)} is {strength} {direction} "
        f"the predicted risk (contribution: {contribution:+.4f})."
    )


def build_ranked_explanation_factors(
    input_data: Mapping[str, Any],
    feature_importance: Mapping[str, float],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        feature_importance.items(),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )

    factors: List[Dict[str, Any]] = []
    for feature_name, contribution in ranked[:limit]:
        value = input_data.get(feature_name, DEFAULT_INPUT.get(feature_name))
        benchmark = FEATURE_BENCHMARKS.get(feature_name, {})
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
                "magnitude": _magnitude_label(contribution),
                "benchmark_context": benchmark.get("description", ""),
                "description": describe_feature_impact(
                    feature_name, value, float(contribution), input_data
                ),
            }
        )

    return factors


# ── 1. Actionable recommendations ──────────────────────────────────────

_RECOMMENDATION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "credit_score": {
        "risk_increase": (
            "Consider improving credit score by paying down revolving "
            "balances, disputing inaccuracies, and maintaining on-time "
            "payment history. Target \u2265700 for favorable terms."
        ),
        "risk_decrease": (
            "Strong credit score is a protective factor. Continue "
            "maintaining low utilization and on-time payments."
        ),
    },
    "debt_to_income_ratio": {
        "risk_increase": (
            "Reduce debt-to-income ratio below 36% by paying down existing "
            "debts or increasing income before applying. Consider "
            "consolidating high-interest debt."
        ),
        "risk_decrease": (
            "Low debt burden is a positive signal. Maintain this ratio "
            "by avoiding new debt before loan closing."
        ),
    },
    "income": {
        "risk_increase": (
            "Low income relative to the loan increases risk. Consider "
            "adding a co-borrower, documenting supplemental income, or "
            "requesting a smaller loan amount."
        ),
        "risk_decrease": (
            "Strong income is a protective factor. Ensure documentation "
            "is current and verified."
        ),
    },
    "loan_amount": {
        "risk_increase": (
            "High loan-to-income ratio elevates risk. Consider reducing "
            "the requested amount or making a larger down payment to "
            "lower required borrowing."
        ),
        "risk_decrease": (
            "Conservative loan size relative to income is favorable. "
            "This improves approval odds and interest rates."
        ),
    },
    "employment_length": {
        "risk_increase": (
            "Short employment history increases risk. If possible, wait "
            "until reaching 2+ years at current employer before applying, "
            "or provide additional income documentation."
        ),
        "risk_decrease": (
            "Long employment tenure signals stability. Ensure current "
            "employment verification is up-to-date."
        ),
    },
    "age": {
        "risk_increase": (
            "Age-related risk premium is minor and non-actionable. "
            "Focus on strengthening other financial factors."
        ),
        "risk_decrease": "Age is a neutral-to-positive factor here.",
    },
    "loan_purpose": {
        "risk_increase": (
            "Certain loan purposes carry higher risk premiums. If "
            "flexible, consider categorizing under a lower-risk purpose "
            "such as home improvement or debt consolidation."
        ),
        "risk_decrease": (
            "Loan purpose is viewed favorably by the model."
        ),
    },
    "home_ownership": {
        "risk_increase": (
            "Renting carries a slight risk premium. Homeownership is "
            "a long-term goal that improves creditworthiness."
        ),
        "risk_decrease": (
            "Homeownership is a strong protective factor indicating "
            "financial stability and collateral."
        ),
    },
    "verification_status": {
        "risk_increase": (
            "Unverified income increases perceived risk. Provide "
            "pay-stubs, tax returns, or bank statements to get "
            "income verified or source-verified."
        ),
        "risk_decrease": (
            "Verified income status reduces risk. Maintain "
            "documentation for future applications."
        ),
    },
}


def build_recommendations(
    factors: Sequence[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    """Generate actionable recommendations from top factors.

    For risk-increasing factors: actionable advice to reduce risk.
    For risk-decreasing factors: preserve advisory (do not change).
    """
    recommendations: List[Dict[str, str]] = []
    for factor in factors:
        feature = str(factor.get("feature", ""))
        impact = str(factor.get("impact", "neutral"))
        templates = _RECOMMENDATION_TEMPLATES.get(feature)
        if not templates:
            continue
        advice = templates.get(impact, "")
        if not advice:
            continue

        if impact == "risk_increase":
            rec_type = "action_needed"
        elif impact == "risk_decrease":
            rec_type = "preserve"
        else:
            rec_type = "informational"

        entry: Dict[str, str] = {
            "feature": feature,
            "label": humanize_feature_name(feature),
            "type": rec_type,
            "recommendation": advice,
        }

        # For protective factors, add an explicit do-not-change advisory
        if impact == "risk_decrease":
            entry["advisory"] = (
                f"Do NOT change: {humanize_feature_name(feature)} is currently "
                f"a protective factor reducing your credit risk."
            )

        recommendations.append(entry)
    return recommendations


# ── 2. Counterfactual explanations ──────────────────────────────────────


def build_counterfactual(
    input_data: Mapping[str, Any],
    feature_importance: Mapping[str, float],
    risk_level: str,
) -> Dict[str, Any]:
    """Suggest minimal changes to move toward the next-lower risk band."""
    if risk_level == "low":
        return {
            "needed": False,
            "message": (
                "The applicant is already in the low-risk band. "
                "No changes are needed."
            ),
            "changes": {},
        }

    changes: Dict[str, Dict[str, Any]] = {}

    # Only suggest changes for features that increase risk
    risk_drivers = sorted(
        (
            (f, c)
            for f, c in feature_importance.items()
            if c > 0.01
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )

    for feature, contribution in risk_drivers:
        target_info = COUNTERFACTUAL_TARGETS.get(feature)
        if not target_info:
            continue

        current_val = input_data.get(feature, DEFAULT_INPUT.get(feature))
        target_val = target_info["target"]
        direction = target_info["direction"]

        needs_change = False
        if direction == "increase" and isinstance(target_val, (int, float)):
            needs_change = float(current_val) < float(target_val)
        elif direction == "decrease" and isinstance(target_val, (int, float)):
            needs_change = float(current_val) > float(target_val)
        elif direction == "change":
            needs_change = str(current_val).lower() != str(target_val).lower()

        if needs_change:
            changes[feature] = {
                "current_value": current_val,
                "suggested_target": target_val,
                "action": target_info["label"],
                "estimated_impact": _magnitude_label(contribution),
            }

    return {
        "needed": bool(changes),
        "message": (
            f"To move toward a lower risk band, consider adjusting "
            f"the following {len(changes)} factor(s)."
            if changes
            else "No actionable counterfactual changes identified."
        ),
        "changes": changes,
    }


# ── 3. Feature grouping into risk categories ────────────────────────────


def build_risk_groups(
    input_data: Mapping[str, Any],
    feature_importance: Mapping[str, float],
) -> Dict[str, Dict[str, Any]]:
    """Group features into human-understandable risk categories."""
    groups: Dict[str, Dict[str, Any]] = {}

    for group_key, group_meta in FEATURE_GROUPS.items():
        group_features = group_meta["features"]
        group_contributions: List[Tuple[str, float, Any]] = []

        for feat in group_features:
            contrib = float(feature_importance.get(feat, 0.0))
            val = input_data.get(feat, DEFAULT_INPUT.get(feat))
            group_contributions.append((feat, contrib, val))

        total_contrib = sum(c for _, c, _ in group_contributions)
        avg_contrib = total_contrib / max(len(group_contributions), 1)

        if avg_contrib > 0.03:
            group_impact = "risk_increase"
        elif avg_contrib < -0.03:
            group_impact = "risk_decrease"
        else:
            group_impact = "neutral"

        groups[group_key] = {
            "label": group_meta["label"],
            "description": group_meta["description"],
            "impact": group_impact,
            "total_contribution": round(total_contrib, 4),
            "features": [
                {
                    "feature": feat,
                    "label": humanize_feature_name(feat),
                    "value": val,
                    "contribution": round(contrib, 4),
                }
                for feat, contrib, val in group_contributions
            ],
        }

    return groups


# ── 4. Explanation confidence ───────────────────────────────────────────


def compute_explanation_confidence(
    feature_importance: Mapping[str, float],
) -> Dict[str, Any]:
    """Assess how confident the explanation is based on contribution patterns."""
    contributions = [float(c) for c in feature_importance.values()]
    if not contributions:
        return {"level": "low", "score": 0.0, "reason": "No contributions available."}

    abs_contribs = [abs(c) for c in contributions]
    max_contrib = max(abs_contribs)
    total_magnitude = sum(abs_contribs)

    # Count features with meaningful contributions (> 0.02)
    meaningful_count = sum(1 for c in abs_contribs if c > 0.02)

    # Check for conflicting signals (both strong increase and decrease)
    strong_increases = sum(1 for c in contributions if c > 0.05)
    strong_decreases = sum(1 for c in contributions if c < -0.05)
    has_conflict = strong_increases >= 1 and strong_decreases >= 1

    # Compute confidence score (0-1)
    conf_score = 0.0

    # Reward: strong dominant features
    if max_contrib > 0.08:
        conf_score += 0.35
    elif max_contrib > 0.04:
        conf_score += 0.20

    # Reward: multiple meaningful contributors
    if meaningful_count >= 3:
        conf_score += 0.30
    elif meaningful_count >= 2:
        conf_score += 0.20
    elif meaningful_count >= 1:
        conf_score += 0.10

    # Reward: total magnitude (overall signal strength)
    if total_magnitude > 0.3:
        conf_score += 0.25
    elif total_magnitude > 0.15:
        conf_score += 0.15

    # Penalty: conflicting signals
    if has_conflict:
        conf_score -= 0.15

    conf_score = max(0.0, min(1.0, conf_score))

    if conf_score >= 0.65:
        level = "high"
        reason = (
            f"Strong, consistent signal from {meaningful_count} meaningful "
            f"feature(s) with total magnitude {total_magnitude:.3f}."
        )
    elif conf_score >= 0.35:
        level = "medium"
        reason = (
            f"Moderate signal from {meaningful_count} feature(s)."
            + (" Mixed risk signals present." if has_conflict else "")
        )
    else:
        level = "low"
        reason = (
            f"Weak signal; only {meaningful_count} feature(s) show "
            f"meaningful contributions (total magnitude {total_magnitude:.3f})."
        )

    return {
        "level": level,
        "score": round(conf_score, 2),
        "reason": reason,
    }


# ── 5. Narrative summary (analyst-style) ───────────────────────────────


def build_prediction_summary(
    prediction: float,
    risk_level: str,
    factors: Sequence[Mapping[str, Any]],
) -> str:
    """Build an analyst-style narrative summary of the prediction."""
    threshold_ctx = risk_threshold_context(prediction)
    risk_tag = risk_level.replace("_", " ").title()

    if not factors:
        return (
            f"Predicted risk: {risk_tag} (score {prediction:.3f}). "
            f"{threshold_ctx} "
            "No individual feature contributions could be computed."
        )

    # Classify factors by impact
    increases = [f for f in factors if f.get("impact") == "risk_increase"]
    decreases = [f for f in factors if f.get("impact") == "risk_decrease"]
    largest = factors[0]

    # Analyst-style narrative
    parts: List[str] = []

    # Opening assessment
    if risk_level in ("very_high", "high"):
        parts.append(
            f"This applicant presents a {risk_tag} credit risk profile "
            f"(score {prediction:.3f}). {threshold_ctx}"
        )
    elif risk_level == "medium":
        parts.append(
            f"This applicant presents a {risk_tag} credit risk profile "
            f"(score {prediction:.3f}). {threshold_ctx} The profile shows "
            f"a mix of risk-elevating and risk-mitigating factors."
        )
    else:
        parts.append(
            f"This applicant presents a {risk_tag} credit risk profile "
            f"(score {prediction:.3f}). {threshold_ctx}"
        )

    # Dominant driver
    parts.append(
        f"The primary driver is {largest['label'].lower()}, which is "
        f"{largest['impact'].replace('_', ' ')} with a "
        f"{largest.get('magnitude', 'notable')} effect "
        f"(contribution: {largest['contribution']:+.4f})."
    )

    # Risk drivers detail
    if increases:
        driver_details = []
        for f in increases[:3]:
            val = f.get("value", "")
            if isinstance(val, float):
                val_str = f"{val:,.2f}" if val > 100 else f"{val:.2%}"
            else:
                val_str = str(val)
            driver_details.append(
                f"{f['label'].lower()} ({val_str})"
            )
        parts.append(
            f"Key risk drivers: {', '.join(driver_details)}."
        )

    # Protective factors detail
    if decreases:
        shield_details = []
        for f in decreases[:3]:
            val = f.get("value", "")
            if isinstance(val, float):
                val_str = f"{val:,.2f}" if val > 100 else f"{val:.2%}"
            else:
                val_str = str(val)
            shield_details.append(
                f"{f['label'].lower()} ({val_str})"
            )
        parts.append(
            f"Key protective factors: {', '.join(shield_details)}."
        )

    return " ".join(parts)


# ── Methodology note ───────────────────────────────────────────────────


def build_methodology_note(shap_used: bool) -> Dict[str, Any]:
    """Return a concise explanation of how contributions were calculated."""
    baseline_profile = {
        "description": (
            "Contributions are measured relative to a baseline applicant "
            "profile. Positive values mean the applicant's feature pushes "
            "risk ABOVE the baseline; negative values push it BELOW."
        ),
        "baseline_values": {
            humanize_feature_name(k): v
            for k, v in DEFAULT_INPUT.items()
        },
    }

    if shap_used:
        return {
            "method": "SHAP (KernelExplainer)",
            "description": (
                "Feature contributions were computed using SHAP "
                "(SHapley Additive exPlanations), a game-theoretic approach "
                "that assigns each feature a marginal contribution to the "
                "prediction. A baseline of representative applicant profiles "
                "is used as the reference point."
            ),
            "interpretation": (
                "Positive contributions increase predicted risk; "
                "negative contributions decrease it. The sum of all "
                "contributions plus the baseline prediction equals the "
                "final score."
            ),
            "baseline": baseline_profile,
        }
    return {
        "method": "Perturbation-based (leave-one-out)",
        "description": (
            "Feature contributions were estimated by individually "
            "perturbing each feature to its baseline value and measuring "
            "the change in the prediction. This is faster than SHAP but "
            "does not account for feature interactions."
        ),
        "interpretation": (
            "Positive contributions increase predicted risk; "
            "negative contributions decrease it."
        ),
        "baseline": baseline_profile,
    }

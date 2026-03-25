"""Explainability package for credit-risk API inference."""

from .config import ExplainabilityConfig
from .explanation_service import ExplainerService
from .shap_explainer import SHAPExplainer
from .utils import (
    FEATURE_BENCHMARKS,
    FEATURE_GROUPS,
    RISK_THRESHOLDS,
    build_counterfactual,
    build_methodology_note,
    build_recommendations,
    build_risk_groups,
    compute_explanation_confidence,
    risk_threshold_context,
)

__all__ = [
    "ExplainabilityConfig",
    "ExplainerService",
    "FEATURE_BENCHMARKS",
    "FEATURE_GROUPS",
    "RISK_THRESHOLDS",
    "SHAPExplainer",
    "build_counterfactual",
    "build_methodology_note",
    "build_recommendations",
    "build_risk_groups",
    "compute_explanation_confidence",
    "risk_threshold_context",
]


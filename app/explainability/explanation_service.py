"""API-facing explanation service for credit risk inference."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import ExplainabilityConfig
from .shap_explainer import SHAPExplainer
from .utils import (
    build_prediction_summary,
    build_ranked_explanation_factors,
    prediction_to_float,
    risk_level_from_score,
)


class ExplainerService:
    """Reusable service that caches explainer initialization."""

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[ExplainabilityConfig] = None,
    ) -> None:
        self.config = config or ExplainabilityConfig()
        self._model: Optional[Any] = None
        self._shap_explainer: Optional[SHAPExplainer] = None

        if model is not None:
            self.set_model(model)

    def set_model(self, model: Any) -> None:
        if model is self._model:
            return

        self._model = model
        self._shap_explainer = SHAPExplainer(model=model, config=self.config)

    def explain_prediction(
        self,
        input_data: Dict[str, Any],
        prediction_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prediction = prediction_to_float(prediction_result or {})
        risk_level = risk_level_from_score(prediction)

        if self._shap_explainer is None:
            top_factors = []
            return {
                "prediction": prediction,
                "risk_level": risk_level,
                "feature_importance": {},
                "top_factors": top_factors,
                "summary": build_prediction_summary(
                    prediction, risk_level, top_factors
                ),
            }

        explanation = self._shap_explainer.explain(
            input_data=input_data, prediction=prediction
        )
        explanation.setdefault("risk_level", risk_level)
        explanation.setdefault(
            "top_factors",
            build_ranked_explanation_factors(
                input_data,
                explanation.get("feature_importance", {}),
                limit=min(3, self.config.max_features_to_display),
            ),
        )
        explanation.setdefault(
            "summary",
            build_prediction_summary(
                prediction,
                explanation["risk_level"],
                explanation["top_factors"],
            ),
        )
        return explanation

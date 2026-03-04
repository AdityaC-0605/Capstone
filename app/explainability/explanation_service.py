"""API-facing explanation service for credit risk inference."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import ExplainabilityConfig
from .shap_explainer import SHAPExplainer
from .utils import prediction_to_float, risk_level_from_score


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

        if self._shap_explainer is None:
            return {
                "prediction": prediction,
                "risk_level": risk_level_from_score(prediction),
                "feature_importance": {},
            }

        return self._shap_explainer.explain(input_data=input_data, prediction=prediction)

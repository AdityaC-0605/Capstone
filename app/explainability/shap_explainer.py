"""Lightweight SHAP explainer for inference-time API usage."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .config import ExplainabilityConfig
from .utils import (
    FEATURE_ORDER,
    decode_feature_vector,
    encode_feature_dict,
    predict_score,
    risk_level_from_score,
    sorted_feature_importance,
)

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """SHAP-first explainer with safe lightweight fallback."""

    def __init__(self, model: Any, config: Optional[ExplainabilityConfig] = None):
        self.model = model
        self.config = config or ExplainabilityConfig()
        self.feature_names = list(FEATURE_ORDER)
        self._background = np.tile(
            np.zeros((1, len(self.feature_names)), dtype=np.float32),
            (self.config.background_sample_size, 1),
        )
        self._explainer = None
        self._shap_ready = False
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        if not SHAP_AVAILABLE:
            return

        try:
            # Validate model callable on at least one row before constructing SHAP object.
            _ = self._predict_matrix(self._background[:1])
            self._explainer = shap.KernelExplainer(
                self._predict_matrix,
                self._background,
            )
            self._shap_ready = True
        except Exception:
            self._explainer = None
            self._shap_ready = False

    def _predict_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        preds = []
        for row in x_matrix:
            decoded = decode_feature_vector(row)
            preds.append(predict_score(self.model, decoded))
        return np.asarray(preds, dtype=np.float32)

    def _fallback_contributions(
        self,
        encoded_input: np.ndarray,
        prediction: float,
    ) -> np.ndarray:
        ref = self._background.mean(axis=0)
        contrib = np.zeros_like(encoded_input, dtype=np.float32)

        for idx in range(encoded_input.shape[0]):
            perturbed = encoded_input.copy()
            perturbed[idx] = ref[idx]
            perturbed_pred = float(self._predict_matrix(perturbed.reshape(1, -1))[0])
            contrib[idx] = float(prediction - perturbed_pred)

        return contrib

    def explain(
        self,
        input_data: Dict[str, Any],
        prediction: Optional[float] = None,
    ) -> Dict[str, Any]:
        encoded = encode_feature_dict(input_data)
        pred = (
            float(prediction)
            if prediction is not None
            else float(self._predict_matrix(encoded.reshape(1, -1))[0])
        )

        if self._shap_ready and self._explainer is not None:
            try:
                try:
                    shap_values = self._explainer.shap_values(
                        encoded.reshape(1, -1),
                        nsamples=self.config.shap_nsamples,
                        silent=True,
                    )
                except TypeError:
                    shap_values = self._explainer.shap_values(
                        encoded.reshape(1, -1),
                        nsamples=self.config.shap_nsamples,
                    )
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                contributions = np.asarray(shap_values).reshape(-1).astype(np.float32)
            except Exception:
                contributions = self._fallback_contributions(encoded, pred)
        else:
            contributions = self._fallback_contributions(encoded, pred)

        feature_importance = sorted_feature_importance(
            self.feature_names,
            contributions,
            self.config.max_features_to_display,
        )

        return {
            "prediction": pred,
            "risk_level": risk_level_from_score(pred),
            "feature_importance": feature_importance,
        }

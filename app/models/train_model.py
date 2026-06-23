"""Train the served credit-risk model.

Generates a synthetic-but-realistic labelled cohort, encodes it with the same
``encode_feature_dict`` used by the SHAP explainer (so attributions are
consistent), trains a gradient-boosted classifier, and saves it to the model
registry. Run with:  ``python -m app.models.train_model``
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from app.explainability.utils import FEATURE_ORDER, encode_feature_dict

REGISTRY_DIR = Path("model_registry")
MODEL_PATH = REGISTRY_DIR / "credit_risk_model.joblib"
REGISTRY_JSON = REGISTRY_DIR / "registry.json"

PURPOSES = [
    "debt_consolidation",
    "home_improvement",
    "major_purchase",
    "medical",
    "vacation",
    "wedding",
    "moving",
    "other",
]
HOMES = ["rent", "own", "mortgage", "other"]
VERIFS = ["not_verified", "verified", "source_verified"]

PURPOSE_RISK = {
    "debt_consolidation": 0.30,
    "home_improvement": -0.10,
    "major_purchase": 0.20,
    "medical": 0.55,
    "vacation": 0.45,
    "wedding": 0.30,
    "moving": 0.20,
    "other": 0.10,
}
HOME_RISK = {"rent": 0.30, "own": -0.35, "mortgage": -0.10, "other": 0.10}
VERIF_RISK = {
    "not_verified": 0.35,
    "verified": -0.25,
    "source_verified": -0.40,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_cohort(n: int, seed: int = 7):
    """Sample applicants and Bernoulli default labels from a risk DGP."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        income = float(np.clip(rng.lognormal(11.0, 0.5), 18000, 250000))
        loan_amount = float(rng.uniform(2000, 45000))
        rows.append(
            {
                "age": int(rng.integers(21, 70)),
                "income": income,
                "employment_length": int(rng.integers(0, 30)),
                "debt_to_income_ratio": float(rng.uniform(0.05, 0.70)),
                "credit_score": int(rng.integers(520, 820)),
                "loan_amount": loan_amount,
                "loan_purpose": rng.choice(PURPOSES),
                "home_ownership": rng.choice(HOMES),
                "verification_status": rng.choice(VERIFS),
            }
        )

    logits = []
    for app in rows:
        lti = app["loan_amount"] / max(app["income"], 1.0)
        z = -0.7
        z += -(app["credit_score"] - 680) / 100.0 * 1.6
        z += (app["debt_to_income_ratio"] - 0.35) * 4.5
        z += (lti - 0.30) * 2.2
        z += -min(app["employment_length"], 15) * 0.06
        z += 0.5 if app["age"] < 25 else 0.0
        z += -min(app["income"] / 120000.0, 1.0) * 0.6
        z += PURPOSE_RISK[app["loan_purpose"]]
        z += HOME_RISK[app["home_ownership"]]
        z += VERIF_RISK[app["verification_status"]]
        logits.append(z)

    probs = _sigmoid(np.array(logits))
    rng_labels = np.random.default_rng(seed + 1)
    labels = (rng_labels.random(n) < probs).astype(int)
    features = np.vstack([encode_feature_dict(app) for app in rows])
    return features, labels


def main() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train = generate_cohort(5000, seed=7)
    x_test, y_test = generate_cohort(1500, seed=99)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.06,
        subsample=0.9,
        random_state=42,
    )
    model.fit(x_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    print(f"Holdout ROC-AUC: {auc:.3f}  (default rate {y_train.mean():.2f})")

    joblib.dump(
        {"model": model, "feature_order": list(FEATURE_ORDER)}, MODEL_PATH
    )

    registry = {}
    if REGISTRY_JSON.exists():
        try:
            registry = json.loads(REGISTRY_JSON.read_text())
        except Exception:
            registry = {}
    registry["credit_risk_model"] = {
        "artifact": MODEL_PATH.name,
        "type": "GradientBoostingClassifier",
        "features": list(FEATURE_ORDER),
        "holdout_roc_auc": round(float(auc), 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "framework": "scikit-learn",
    }
    REGISTRY_JSON.write_text(json.dumps(registry, indent=2))
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()

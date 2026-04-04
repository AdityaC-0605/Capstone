# Sustainable Credit Risk AI Backend

Backend-only repository for credit risk prediction with three primary capabilities:

- **Explainable credit risk inference** — SHAP-powered explanations with actionable recommendations, counterfactual suggestions, feature grouping, confidence scoring, and analyst-style narrative summaries
- **Federated learning simulation**
- **Sustainability and carbon-aware experimentation**

## What Is Included

- `app/api/` FastAPI services for health, status, and credit risk inference
- `app/explainability/` SHAP-based and fallback explanation logic
- `app/federated/` local federated learning simulation utilities
- `app/sustainability/` carbon-aware NAS, monitoring, and evaluation helpers
- `app/core/` configuration, logging, security, auth, encryption, GDPR support
- `tests/` lightweight backend smoke tests

Frontend-specific files have been removed so a separate frontend can be built cleanly against these APIs.

## Frontend Preview

A new standalone frontend lives in `frontend/`.

Start the backend first:

```bash
./start_backend.sh
```

Then serve the frontend from the repo root:

```bash
python3 -m http.server 4173 -d frontend
```

Open:

```text
http://localhost:4173
```

Paste the `sk-test-...` API key from the inference log into the UI to run live predictions. If the backend is offline, the frontend can still render a polished demo state locally.

## Quick Start

```bash
cd /Users/aditya/Documents/MJ
source venv/bin/activate
```

Run the backend services:

```bash
./start_backend.sh
```

This starts:

- Main API: `http://localhost:8000`
- Inference API: `http://localhost:8001`

## Useful Checks

System bootstrap:

```bash
python main.py
```

Backend smoke tests:

```bash
python -m pytest -q tests/
```

Main API health:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready
curl -s http://localhost:8000/api/v1/status
```

Inference API health:

```bash
curl -s http://localhost:8001/health
```

## Inference API Example

Start the backend, copy the generated API key from the inference log, then call:

```bash
curl -s http://localhost:8001/predict \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "application": {
      "age": 35,
      "income": 65000,
      "employment_length": 5,
      "debt_to_income_ratio": 0.30,
      "credit_score": 720,
      "loan_amount": 25000,
      "loan_purpose": "debt_consolidation",
      "home_ownership": "rent",
      "verification_status": "verified"
    },
    "include_explanation": true,
    "track_sustainability": true,
    "explanation_type": "shap"
  }'
```

## Feature Commands

Explainability smoke test:

```bash
python -m pytest -q tests/test_explainability_runtime.py
```

Manual explainability test (prints full JSON output):

```bash
./venv/bin/python -c "
import json
from app.models.runtime_credit_model import LightweightCreditRiskModel
from app.explainability.explanation_service import ExplainerService

model = LightweightCreditRiskModel()
explainer = ExplainerService(model)

sample = {
    'age': 23, 'income': 28000, 'employment_length': 1,
    'debt_to_income_ratio': 0.58, 'credit_score': 560,
    'loan_amount': 26000, 'loan_purpose': 'medical',
    'home_ownership': 'rent', 'verification_status': 'not_verified',
}

pred = model.predict(sample)
exp = explainer.explain_prediction(sample, pred)
print(json.dumps(exp, indent=2, default=str))
"
```

Federated learning simulation:

```bash
python - <<'PY'
from app.federated.utils import run_federated_simulation
from app.federated.config import FLConfig

result = run_federated_simulation(
    FLConfig(number_of_clients=3, aggregation_rounds=3, local_epochs=2)
)
print(result["best_val_loss"])
PY
```

Carbon-aware NAS:

```bash
python -m app.sustainability.run_nas
python -m app.sustainability.run_nas_german
```

## Project Structure

```text
MJ/
├── app/
│   ├── api/
│   ├── core/
│   ├── explainability/
│   ├── federated/
│   ├── models/
│   ├── services/
│   └── sustainability/
├── config/
├── infrastructure/
├── model_registry/
├── models/
├── tests/
├── main.py
├── start_backend.sh
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Explainability Output Schema

The `explanation` field in the inference response contains:

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `float` | Risk score (0–1) |
| `risk_level` | `string` | `low` \| `medium` \| `high` \| `very_high` |
| `risk_threshold_context` | `string` | Which risk band the score falls into |
| `risk_thresholds` | `array` | All risk bands with their ceilings |
| `feature_importance` | `object` | Feature → contribution mapping |
| `top_factors` | `array` | Top 5 factors with description, magnitude, benchmark context |
| `recommendations` | `array` | Actionable advice per factor (`action_needed` or `preserve`) |
| `counterfactual` | `object` | Minimal changes to move to a lower risk band |
| `risk_groups` | `object` | Features grouped into categories (financial strength, debt burden, stability, loan context) |
| `confidence` | `object` | Explanation confidence (`low` \| `medium` \| `high` with score and reason) |
| `methodology` | `object` | How contributions were computed (SHAP or perturbation) + baseline profile |
| `summary` | `string` | Analyst-style narrative summary |

## Notes For Frontend Integration

- Treat `http://localhost:8000` as the system/status API.
- Treat `http://localhost:8001` as the credit risk inference API.
- Use `explanation.summary` for a quick overview and `explanation.top_factors` for detailed breakdowns.
- Display `explanation.recommendations` to show users what to improve (`action_needed`) and what to maintain (`preserve`).
- Use `explanation.counterfactual` to show specific targets (e.g., "Increase credit score to ≥ 700").
- Show `explanation.risk_groups` for a grouped view of risk categories.
- Use `explanation.confidence.level` to indicate how reliable the explanation is.
- Reference `explanation.methodology.baseline.baseline_values` to show the comparison baseline.

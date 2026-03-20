# Sustainable Credit Risk AI Backend

Backend-only repository for credit risk prediction with three primary capabilities:

- Explainable credit risk inference
- Federated learning simulation
- Sustainability and carbon-aware experimentation

## What Is Included

- `app/api/` FastAPI services for health, status, and credit risk inference
- `app/explainability/` SHAP-based and fallback explanation logic
- `app/federated/` local federated learning simulation utilities
- `app/sustainability/` carbon-aware NAS, monitoring, and evaluation helpers
- `app/core/` configuration, logging, security, auth, encryption, GDPR support
- `tests/` lightweight backend smoke tests

Frontend-specific files have been removed so a separate frontend can be built cleanly against these APIs.

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

## Notes For Frontend Integration

- Treat `http://localhost:8000` as the system/status API.
- Treat `http://localhost:8001` as the credit risk inference API.
- Use the inference response `explanation.summary` and `explanation.top_factors` directly in the frontend.

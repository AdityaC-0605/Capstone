# PulseLedger — Sustainable Credit Risk AI

A complete, full-stack application providing carbon-aware, federated, and explainable credit risk intelligence.

## 🌟 Core Capabilities

- **Explainable Credit Risk Inference** — SHAP-powered explanations featuring actionable recommendations, counterfactual suggestions, feature risk grouping, confidence scoring, and dynamic analyst-style narrative summaries.
- **Federated Learning Simulation** — Distributed client coordination visualization demonstrating privacy-preserving model aggregation.
- **Sustainability Operations** — Carbon-aware AI telemetry, tracking the carbon footprint and energy consumption of live operational AI models.
- **"Mercury Noir" Command Center** — A premium, unified React/Next.js frontend engineered with a custom warm-industrial design system indicating live operational intelligence.

---

## 🚀 Quick Start

The project consists of a Python FastAPI backend and a Next.js frontend. You will need two terminal windows to run both simultaneously.

### 1. Start the Backend

```bash
# In the root 'MJ/' directory
source venv/bin/activate
./start_backend.sh
```

This starts:
- **Main Engine API:** `http://localhost:8000`
- **Inference Engine API:** `http://localhost:8001`
*(Note: Keep track of the `sk-test-...` API key printed in the console for inference)*

### 2. Start the Frontend

```bash
# In a new terminal window
cd frontend
npm install
npm run dev
```

Open your browser to:
[http://localhost:3000](http://localhost:3000)

*Within the PulseLedger Studio UI, navigate to Settings (gear icon) and submit the `sk-test-...` bearer key to begin live polling.*

---

## 🛠️ Project Structure

```text
PulseLedger/
├── app/                  # FastAPI Backend Services
│   ├── api/              # Core endpoints
│   ├── explainability/   # SHAP explainer & narrative generator
│   ├── federated/        # FL simulation orchestration
│   ├── models/           # Lightweight credit models
│   └── sustainability/   # Carbon-aware NAS and tracking
├── frontend/             # Next.js Application
│   ├── app/              # Next App Router (Dashboard, Studio, Federated, Sustainability)
│   ├── components/       # Mercury Noir UI component library
│   ├── lib/              # Formatting, Types, API Connectors
│   └── store/            # Zustand global state (usePulseStore)
├── tests/                # Backend smoke/unit tests
├── main.py               # Main API launchbed
├── start_backend.sh      # Launch script
└── pyproject.toml / package.json
```

---

## 📊 Developer Commands

### Backend Verification
System bootstrap & smoke tests:
```bash
python main.py
python -m pytest -q tests/
```

Manual Explainability Terminal Test:
```bash
python -m pytest -q tests/test_explainability_runtime.py
```

### Checking API Health
```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8001/health
```

### CLI Inference Example
Submit a manual trace directly to the Inference Engine (Requires valid API key):
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

### Run Backend Simulations
Trigger the automated Federated simulation pipeline locally:
```bash
python - <<'PY'
from app.federated.utils import run_federated_simulation
from app.federated.config import FLConfig

result = run_federated_simulation(
    FLConfig(number_of_clients=3, aggregation_rounds=3, local_epochs=2)
)
print("Best Validation Loss:", result["best_val_loss"])
PY
```

Trigger Neural Architecture Search (NAS) tuning:
```bash
python -m app.sustainability.run_nas
python -m app.sustainability.run_nas_german
```

---

## 🧠 Explainability Subsystem Output Schema

The `explanation` payload returned by the inference runtime is consumed by the PulseLedger frontend to hydrate visual widgets. Its structural map includes:

| Return Field | Type | Function |
|-------|------|-------------|
| `prediction` | `float` | Base inferred risk score (0.0 to 1.0) |
| `risk_level` | `string` | Categorization boundary (`low` \| `medium` \| `high` \| `very_high`) |
| `risk_threshold_context` | `string` | Plain-text threshold narrative |
| `feature_importance` | `object` | Key-value mapping of feature impacts |
| `top_factors` | `array` | Top 5 risk-affecting factors spanning description, directional magnitude, and benchmark context |
| `recommendations` | `array` | Computed actions required by analyst or user (`action_needed` / `preserve`) |
| `counterfactual` | `object` | Target perturbations calculated to demote the risk to a lower band |
| `risk_groups` | `object` | Thematic aggregation (Financial Strength, Debt Burden, Stability, Loan Profile) |
| `confidence` | `object` | Scoring mechanism on the viability of the specific explainability branch |
| `methodology` | `object` | Baseline profile metadata defining standard SHAP vs perturbation origins |
| `summary` | `string` | NLP-stylized auto-generated narrative summary synthesizing the inference context |

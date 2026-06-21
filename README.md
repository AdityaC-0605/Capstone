# PulseLedger — Sustainable Credit Risk AI

A complete, full-stack application providing carbon-aware, federated, and explainable credit risk intelligence.

## 🌟 Core Capabilities

- **Explainable Credit Risk Inference** — SHAP-powered explanations featuring actionable recommendations, counterfactual suggestions, feature risk grouping, confidence scoring, and dynamic analyst-style narrative summaries. The Studio now visualizes raw SHAP attributions as a signed horizontal bar chart.
- **Federated Learning (live)** — A real multi-client FedAvg simulation runs on the backend and returns genuine round-by-round validation loss **and** accuracy. The frontend renders the real convergence curve (no more client-side `Math.random()` placeholders).
- **Fairness Auditing (live)** — Demographic-parity, equal-opportunity, equalized-odds, calibration and treatment-equality metrics computed by the real bias detector, with severity grading and remediation recommendations.
- **Sustainability Operations** — Carbon-aware AI telemetry, tracking the carbon footprint and energy consumption of live operational AI models.
- **"Mercury Noir" Command Center** — A premium, unified React/Next.js frontend engineered with a custom warm-industrial design system indicating live operational intelligence. Accessible by default (honors `prefers-reduced-motion`, dialog semantics, toast notifications).

## ✨ What's New in v1.1

| Area | Change |
|------|--------|
| **Federated learning** | Now genuinely computed on the backend (`POST /api/v1/federated/run`) and wired into the UI — real FedAvg, ~0.4s for a small run. |
| **Fairness audit** | New real bias-detection endpoint (`GET /api/v1/fairness/audit`). |
| **API key** | **Persists across restarts** (env → file → generated). Previously a new key was minted on every boot, silently invalidating the token saved in the UI. |
| **Rate limiting** | Actually enforced on `/predict` and `/predict/batch` (it was wired but never applied). |
| **Pydantic** | Validators migrated from deprecated v1 `@validator` to v2 `@field_validator`. |
| **Observability** | `GET /predict/history` (server-side rolling history) and `GET /metrics` (Prometheus) on both services. |
| **Frontend UX** | Toast notifications, SHAP attribution chart, reduced-motion + dialog a11y, friendlier network-error messages, removed dead code. |
| **CORS** | Configurable via `PULSELEDGER_ALLOWED_ORIGINS` (defaults to `*` for local dev). |
| **Tests** | Added `tests/test_new_capabilities.py` covering every new endpoint and the hardening behavior. |

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

## 🌐 Platform API (v1.1)

Beyond inference (port 8001), the **main API on port 8000** now exposes real ML capabilities:

```bash
# Real multi-client FedAvg simulation (no API key required)
curl -s -X POST http://localhost:8000/api/v1/federated/run \
  -H "Content-Type: application/json" \
  -d '{"number_of_clients": 4, "aggregation_rounds": 4, "local_epochs": 2}'
# -> { "data": { "round_metrics": [...], "best_val_loss": ..., "wall_time_seconds": ... } }

# Real fairness / bias audit over a deterministic synthetic cohort
curl -s "http://localhost:8000/api/v1/fairness/audit?samples=1000&bias_strength=1.5"
# -> demographic parity, equalized odds, calibration metrics + recommendations

# Prometheus metrics (both services)
curl -s http://localhost:8000/metrics
curl -s http://localhost:8001/metrics
```

On the inference engine (port 8001):

```bash
# Server-side rolling prediction history
curl -s "http://localhost:8001/predict/history?limit=25" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Environment Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `PULSELEDGER_API_KEY` | Pin the inference bearer key (highest precedence) | _generated_ |
| `PULSELEDGER_API_KEY_FILE` | Where the key is persisted/loaded | `keys/api_key.txt` |
| `PULSELEDGER_ALLOWED_ORIGINS` | Comma-separated CORS allow-list for the main API | `*` |
| `ENVIRONMENT` | `development` enables hot reload | `development` |

> The inference API key now **persists across restarts**, so the bearer token you save in the Studio UI keeps working. The key file is git-ignored.

---

## 🗺️ Feature Roadmap — toward a complete product

High-impact features to take PulseLedger from an excellent demo to a production platform:

1. **Persistence layer** — wire the configured PostgreSQL (`DatabaseConfig` already exists) + SQLAlchemy/Alembic so predictions, audit logs and API keys survive beyond memory. Add a queryable `/predictions/{id}` history.
2. **Fairness dashboard** — a dedicated frontend page on the live `/api/v1/fairness/audit` endpoint (the backend is ready) with a bias heatmap and "before/after mitigation" comparison.
3. **Batch / CSV scoring** — a Studio "bulk" mode that uploads a CSV and fans out to the existing `/predict/batch` endpoint with a downloadable results table.
4. **Real-time telemetry** — WebSocket/SSE stream of live predictions + carbon so the dashboard updates without polling.
5. **Real NAS runs** — replace the simulated NAS table on the Sustainability page with the existing `app.sustainability.run_nas` pipeline (bounded/async).
6. **Auth & multi-tenancy** — the JWT/RBAC framework in `app/core/auth.py` is built but unused; gate the UI behind login and scope keys per user.
7. **Model registry integration** — load/serve the trained artifacts in `model_registry/` instead of the lightweight formula model, with versioning and rollback.
8. **CI quality gates** — run the test suite + `next build` + `tsc` on every push; add Playwright E2E and an axe accessibility audit.

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

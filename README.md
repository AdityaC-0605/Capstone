# PulseLedger — Explainable Credit-Risk Intelligence

A full-stack platform for credit-risk scoring that ships **a defensible reason with every number** — SHAP explanations, federated learning, fairness auditing, and carbon-aware telemetry, behind a polished marketing site and a real application workspace.

- **Backend** — two FastAPI services (platform + inference) with real ML: SHAP explanations, FedAvg federated learning, and a bias detector.
- **Frontend** — a Next.js 14 app in a custom **"Ledger"** design system: an animated landing page plus a sidebar-driven workspace.

---

## 🌟 Core Capabilities

- **Explainable inference** — Every score returns SHAP feature attributions, a written analyst narrative, counterfactuals to a lower risk band, risk-group breakdowns, and confidence context. The UI renders signed SHAP attributions as a horizontal bar chart.
- **Federated learning (live)** — A real multi-client FedAvg simulation runs on the backend and returns genuine round-by-round validation loss **and** accuracy; the UI plots the real convergence curve.
- **Fairness auditing (live)** — Demographic parity, equal opportunity, equalized odds, calibration and treatment equality across protected groups, with severity grading and remediation recommendations, on its own dashboard.
- **Carbon-aware operations** — Energy, emissions, and latency are tracked alongside predictions and aggregated into a session footprint with an eco-score verdict.
- **Accounts & multi-tenancy** — email/password auth with JWT sessions; each analyst sees only their own assessments. A one-click demo account is seeded for instant access.
- **"The Ledger" experience** — A light, editorial, institutional design (bone paper + evergreen ink, Fraunces / IBM Plex) — accessible by default (honors `prefers-reduced-motion`, dialog semantics, keyboard focus, toasts).

---

## 🏗️ Architecture

```text
┌─────────────────────────────┐        ┌──────────────────────────────────────┐
│  Next.js frontend (:3000)   │        │  FastAPI backend                       │
│                             │        │                                        │
│  /            Landing       │──────► │  Platform API (:8000)                  │
│  /dashboard   Workspace     │  HTTP  │   /health  /ready  /api/v1/status      │
│  /assessments New / List /  │ ─────► │   POST /api/v1/federated/run  (FedAvg) │
│               Detail        │        │   GET  /api/v1/fairness/audit (bias)   │
│  /federated   FedAvg        │        │   /metrics                             │
│  /fairness    Bias audit    │        │                                        │
│  /sustainability  Carbon    │──────► │  Inference API (:8001)                 │
│  /settings    Connect       │  HTTP  │   POST /predict        (SHAP, rate-ltd)│
│                             │ ─────► │   POST /predict/batch                  │
│  Zustand store + localStorage│       │   GET  /predict/history  /metrics      │
└─────────────────────────────┘        └──────────────────────────────────────┘
```

The platform API lazy-imports its heavy ML dependencies (torch / numpy) **inside** the endpoints, so health checks and startup stay instant and the service degrades gracefully if an optional dependency is missing.

---

## 🚀 Quick Start

Two services back the UI; the helper script launches both.

### 1. Backend

```bash
# from the repo root
source venv/bin/activate          # or: python -m venv venv && pip install -r requirements.txt
./start_backend.sh
```

This starts the **Platform API** on `http://localhost:8000` and the **Inference API** on `http://localhost:8001`, and prints the inference bearer key (`sk-test-…`). The key now **persists** to `keys/api_key.txt`, so it stays stable across restarts.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev            # http://localhost:3000  (or: npm run build && npm start)
```

### 3. Sign in

Open **http://localhost:3000** → **Launch app** → on the login screen click **Use demo account** (or register your own). The workspace is gated behind login; your session token authorizes scoring and your assessments are scoped to your account.

> **Demo credentials:** `demo@pulseledger.app` / `demo12345`. Federated learning and the fairness audit call the platform API directly and need no credentials.

### Or run the whole stack in Docker

```bash
docker compose up --build   # Postgres + both APIs + frontend → http://localhost:3000
```

See **[DEPLOY.md](DEPLOY.md)** for Docker Compose, Render (one-click Blueprint), and Vercel.

---

## 🎨 Frontend — "The Ledger"

A deliberate, institutional design rather than a template — the front door of a serious risk instrument.

| Element | Choice |
|---------|--------|
| **Palette** | Bone paper `#F4F1E8` · warm ink `#1A1714` · deep evergreen `#1C4634`, with ochre `#B07A2C` and oxblood `#A3392A` as a functional risk scale |
| **Type** | **Fraunces** (serif display) · **IBM Plex Sans** (body) · **IBM Plex Mono** (every number) |
| **Motion** | Scroll-reveals, an animated node-network hero canvas, count-up metrics — all gated on `prefers-reduced-motion` |

**Information architecture** — the marketing site and the app are split via a Next.js route group so each has its own chrome:

- `/` — **Landing**: sticky nav, animated hero with a live assessment specimen, capability sections, workflow, metrics, CTA.
- `/dashboard` — **Workspace** (sidebar shell): KPIs, recent assessments, system health, quick actions.
- `/assessments` · `/assessments/new` · `/assessments/[id]` — the credit-scoring flow: queue → focused form → full explained result.
- `/federated` — run a live FedAvg round; watch the convergence curve + round-history ledger.
- `/fairness` — run a live bias audit; per-metric table, protected-group breakdown, recommendations.
- `/sustainability` — **Telemetry / Experiments** tabs (session footprint, eco-score, NAS preview).
- `/settings` — connect to the backend and paste the bearer key.

State lives in a Zustand store persisted to `localStorage` (assessment history survives reloads within a session).

---

## 🌐 API Reference

### Platform API — `:8000` (no key required)

```bash
# Real multi-client FedAvg simulation
curl -s -X POST http://localhost:8000/api/v1/federated/run \
  -H "Content-Type: application/json" \
  -d '{"number_of_clients": 4, "aggregation_rounds": 4, "local_epochs": 2}'
# -> { "data": { "round_metrics": [...], "best_val_loss": ..., "wall_time_seconds": ... } }

# Real fairness / bias audit over a deterministic synthetic cohort
curl -s "http://localhost:8000/api/v1/fairness/audit?samples=1000&bias_strength=1.5"
# -> demographic parity, equalized odds, calibration + recommendations

# Prometheus-style metrics
curl -s http://localhost:8000/metrics
```

### Inference API — `:8001` (bearer key required)

```bash
# Single prediction with full SHAP explanation
curl -s http://localhost:8001/predict \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "application": {
      "age": 35, "income": 65000, "employment_length": 5,
      "debt_to_income_ratio": 0.30, "credit_score": 720,
      "loan_amount": 25000, "loan_purpose": "debt_consolidation",
      "home_ownership": "rent", "verification_status": "verified"
    },
    "include_explanation": true,
    "track_sustainability": true,
    "explanation_type": "shap"
  }'

# Durable prediction history (persists across restarts)
curl -s "http://localhost:8001/predict/history?limit=25" -H "Authorization: Bearer YOUR_API_KEY"

# A single persisted assessment by id (full record incl. explanation)
curl -s "http://localhost:8001/predict/pred_abc123" -H "Authorization: Bearer YOUR_API_KEY"
```

**Accounts** (inference API). Login returns a session JWT; use it as the bearer
for `/predict*` — a logged-in user only sees their own assessments. A legacy
service API key still works for machine access.

```bash
curl -s -X POST http://localhost:8001/auth/login -H "Content-Type: application/json" \
  -d '{"email":"demo@pulseledger.app","password":"demo12345"}'
# -> { "access_token": "<jwt>", "token_type": "bearer", "user": {...} }
curl -s http://localhost:8001/auth/me -H "Authorization: Bearer <access_token>"

# Batch scoring (up to 100 applications) and metrics
curl -s -X POST http://localhost:8001/predict/batch -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" -d '{"applications": [ ... ]}'
curl -s http://localhost:8001/metrics
```

Interactive docs: `http://localhost:8000/docs` and `http://localhost:8001/docs`.

### Environment configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `PULSELEDGER_API_KEY` | Pin the inference bearer key (highest precedence) | _generated_ |
| `PULSELEDGER_API_KEY_FILE` | Where the key is persisted/loaded | `keys/api_key.txt` |
| `PULSELEDGER_DATABASE_URL` | SQLAlchemy DB URL — users + assessments persist here | `sqlite:///./pulseledger.db` |
| `PULSELEDGER_JWT_SECRET` | Secret for signing session tokens — **set in production** | `dev-insecure-change-me` |
| `PULSELEDGER_DEMO_PASSWORD` | Seeded demo-account password | `demo12345` |
| `PULSELEDGER_ALLOWED_ORIGINS` | Comma-separated CORS allow-list for the platform API | `*` |
| `ENVIRONMENT` | `development` enables hot reload | `development` |

The frontend's backend URLs and bearer key are configured in **Settings** and persisted client-side; no `.env` is required for the UI.

---

## 🧪 Testing, Linting & CI

```bash
# Backend tests (13 tests covering both APIs + hardening)
source venv/bin/activate
python -m pytest -q tests/

# Match the GitHub Actions lint gates exactly
black --check app/ tests/        # line-length 79
isort --check-only app/ tests/   # profile = black
flake8 app/ tests/

# Frontend
cd frontend
npx tsc --noEmit                 # type check
npm run build                    # production build (all routes)
```

The app bootstraps its tables automatically on startup (idempotent
`create_all`). For managed schema changes, Alembic is configured:

```bash
alembic upgrade head             # apply migrations (uses PULSELEDGER_DATABASE_URL)
alembic revision --autogenerate -m "describe change"
```

GitHub Actions (`.github/workflows/ci.yml`) runs Black, isort, flake8, pytest (mypy and bandit are advisory). Tooling versions are pinned in `.pre-commit-config.yaml` — pin them in CI too to avoid style drift from unpinned installs.

---

## 🗂️ Project Structure

```text
PulseLedger/
├── app/                       # FastAPI backend
│   ├── api/
│   │   ├── main.py            # Platform API (:8000) — federated, fairness, status, metrics
│   │   └── inference_service.py  # Inference API (:8001) — predict, history, auth, metrics
│   ├── explainability/        # SHAP explainer + analyst-narrative generator
│   ├── federated/             # FedAvg client/server/utils
│   ├── services/              # bias_detector (live) + compliance/ingestion modules
│   ├── sustainability/        # carbon tracking + carbon-aware NAS research
│   ├── models/                # runtime model (trained loader + formula) + train_model.py
│   ├── db/                    # SQLAlchemy base, ORM models (User, Prediction), repository
│   └── core/                  # config, logging, security (JWT), auth/RBAC, GDPR scaffolding
├── frontend/
│   ├── app/
│   │   ├── page.tsx           # Landing (/)
│   │   ├── login/page.tsx     # Auth — sign in / register (/login)
│   │   ├── layout.tsx, globals.css
│   │   └── (app)/             # App route group (sidebar shell, login-gated)
│   │       ├── dashboard/  assessments/{,new,batch,[id]}/
│   │       ├── federated/  fairness/  sustainability/  settings/
│   ├── components/            # Ledger design-system components
│   ├── lib/                   # api client, types, formatters, utils
│   └── store/                 # Zustand stores (pulse, auth, toast)
├── migrations/                # Alembic migration environment + versions
├── tests/                     # backend pytest suite (+ conftest isolated DB)
├── start_backend.sh           # launches both APIs
├── alembic.ini · main.py      # migration config · platform API launcher
└── pyproject.toml · requirements.txt
```

---

## 🧠 Explanation Payload Schema

The `explanation` object returned by `/predict` drives the UI's explainability widgets:

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `float` | Risk score, 0.0–1.0 |
| `risk_level` | `string` | `low` \| `medium` \| `high` \| `very_high` |
| `risk_threshold_context` | `string` | Plain-text threshold narrative |
| `feature_importance` | `object` | Feature → SHAP value mapping |
| `top_factors` | `array` | Leading factors with direction, magnitude, and benchmark context |
| `recommendations` | `array` | Suggested actions (`action_needed` / `preserve`) |
| `counterfactual` | `object` | Minimal changes to reach a lower band |
| `risk_groups` | `object` | Thematic aggregation (debt burden, stability, loan profile, …) |
| `confidence` | `object` | Level, score, and reasoning |
| `methodology` | `object` | Baseline profile + method metadata |
| `summary` | `string` | Auto-generated analyst narrative |

---

## 🗺️ Roadmap

Done since v1.0: ✅ real federated endpoint · ✅ live fairness **dashboard** · ✅ persistent API key · ✅ enforced rate limiting · ✅ Pydantic v2 · ✅ `/metrics` + `/predict/history` · ✅ landing site + app workspace · ✅ **durable persistence** (SQLite/Postgres + Alembic) · ✅ **accounts & multi-tenancy** (JWT auth, per-user scoping) · ✅ **bulk CSV scoring** · ✅ **deploy configs** (Docker, Compose, Render — see [DEPLOY.md](DEPLOY.md)) · ✅ **trained model served** from the registry.

Toward production:

1. **Extend persistence** — predictions and users are durably stored; next, persist the audit log and rotate the JWT secret via a secrets manager.
2. **RBAC, admin & SSO** — accounts and per-user scoping exist; add role-based admin views and SSO/OAuth providers.
3. **Model registry depth** — the trained model is served from the registry; next add versioning, rollback, and champion/challenger comparison.
4. **Real NAS runs** — replace the simulated NAS preview with the `app.sustainability.run_nas` pipeline (bounded/async).
5. **Real-time telemetry** — WebSocket/SSE stream so the dashboard updates without polling.

---

## ⚠️ Notes & Honest Scope

- **Persistence** — users and assessments are durably stored server-side (SQLite by default, Postgres via `PULSELEDGER_DATABASE_URL`) and survive restarts; the assessment detail page falls back to the server when a record isn't in the browser's local history. In-process `/metrics` counters remain memory-only.
- **Auth hardening** — the demo password is intentionally public and the default `PULSELEDGER_JWT_SECRET` is insecure. Set a real secret (and change or disable the demo account) before exposing the app publicly.
- **Illustrative content** — the landing page's headline stats and the "trusted by" row are marketing placeholders; the NAS table on Sustainability is a clearly-labelled **preview**. Everything in the app workspace (scoring, explanations, federated, fairness) is computed for real.
- **Model** — inference serves a trained gradient-boosted classifier from `model_registry/credit_risk_model.joblib` (ROC-AUC ≈ 0.85 on held-out synthetic data), with a transparent formula as an automatic fallback if the artifact is missing. SHAP explains whichever is active. Retrain with `python -m app.models.train_model`; disable with `PULSELEDGER_USE_TRAINED_MODEL=0`.

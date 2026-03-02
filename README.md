# 🌿 Sustainable Credit Risk AI

A credit risk assessment system that's **carbon-aware** — it finds the most efficient model architecture that maintains performance while minimizing environmental impact.

---

## What This Project Does

| Feature | Description |
|---------|-------------|
| **Carbon-Aware NAS** | Searches for the best model (architecture × exit level × precision) that reduces carbon cost by ~87% while retaining 99%+ performance |
| **Federated Learning** | Privacy-preserving distributed training using FedAvg across multiple simulated clients |
| **Credit Risk Prediction** | Full inference API with risk scoring, explainability (SHAP), and sustainability tracking |
| **Bias & Compliance** | FCRA, ECOA, GDPR compliance checks + fairness metrics across protected attributes |
| **Streamlit Dashboard** | Interactive UI to run NAS, simulate FL, predict risk, and monitor energy usage |

---

## Quick Start

### 1. Setup

```bash
# Clone and enter the project
cd /Users/aditya/Documents/MJ

# Activate virtual environment
source venv/bin/activate
```

### 2. Verify everything works

```bash
python main.py
```

You should see:
```
INFO - Starting Sustainable Credit Risk AI System
INFO - Environment: development
INFO - System initialization complete
```

### 3. Run the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser. The dashboard has 6 pages:

| Page | What it does |
|------|-------------|
| 🏠 Dashboard | Overview of datasets, system architecture, and top-level KPIs |
| 🔬 Carbon-Aware NAS | Run NAS experiments on Bank or German Credit datasets |
| 🤝 Federated Learning | Simulate distributed training with configurable clients/rounds |
| 🎯 Credit Risk Prediction | Input applicant data and get risk score + explanation |
| 📊 Sustainability Monitor | Track energy consumption and carbon emissions |
| 🛡️ System Health | Check API endpoints, security status, and component readiness |

---

## Run Individual Features (Terminal)

### Carbon-Aware NAS

```bash
# Bank dataset (~1-2 min)
python -m app.sustainability.run_nas

# German Credit dataset (~2-3 min)
python -m app.sustainability.run_nas_german
```

**Expected output:** A table of model configurations ranked by carbon cost, showing AUC, KS statistic, Brier score, and a research comparison table.

### Baseline Model Results

```bash
python -m app.sustainability.results
```

### Federated Learning Simulation

```bash
python -c "
from app.federated.utils import run_federated_simulation
from app.federated.config import FLConfig

result = run_federated_simulation(
    config=FLConfig(number_of_clients=3, aggregation_rounds=5, local_epochs=2)
)
print(f'Rounds: {len(result[\"round_metrics\"])}, Best loss: {result[\"best_val_loss\"]:.6f}')
"
```

### Start the REST APIs

```bash
./start_backend.sh
```

This starts:
- **Main API** → http://localhost:8000 (health checks, status)
- **Inference API** → http://localhost:8001 (predictions)

Health checks:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
MJ/
├── streamlit_app.py          ← Streamlit dashboard (start here!)
├── main.py                   ← CLI entry point
├── start_backend.sh          ← Starts both REST APIs
├── requirements.txt          ← Python dependencies
├── pyproject.toml            ← Build config, pytest, linting
├── Makefile                  ← Dev shortcuts (test, lint, format)
│
├── app/
│   ├── api/                  ← FastAPI endpoints
│   │   ├── main.py           ← Health, readiness, status routes
│   │   └── inference_service.py ← Prediction API with auth & rate limiting
│   │
│   ├── core/                 ← Shared infrastructure
│   │   ├── config.py         ← YAML config loader
│   │   ├── logging.py        ← Structured + audit logging
│   │   ├── auth.py           ← JWT, API keys, RBAC, MFA
│   │   ├── encryption.py     ← AES-256 encryption + key management
│   │   ├── anonymization.py  ← PII detection, data masking
│   │   ├── gdpr_compliance.py ← Consent tracking, right to erasure
│   │   ├── security_manager.py ← Security orchestrator
│   │   └── interfaces.py     ← Abstract base classes
│   │
│   ├── sustainability/       ← Carbon-aware ML (main research module)
│   │   ├── run_nas.py        ← NAS runner for Bank dataset
│   │   ├── run_nas_german.py ← NAS runner for German Credit dataset
│   │   ├── carbon_aware_nas.py ← Core NAS algorithm
│   │   ├── scalable_mlp.py   ← Multi-exit MLP with adjustable width
│   │   ├── metrics.py        ← Shared KS statistic function
│   │   ├── results.py        ← Baseline model evaluation
│   │   ├── research_table.py ← Research comparison table generator
│   │   ├── sustainability_report.py ← Carbon reduction summary
│   │   ├── plot_pareto.py    ← Pareto frontier visualization
│   │   ├── Bank_data.csv     ← Bank marketing dataset
│   │   └── german_data.csv   ← German credit dataset
│   │
│   ├── federated/            ← Federated learning
│   │   ├── server.py         ← FedAvg server
│   │   ├── client.py         ← Local training client
│   │   ├── aggregation.py    ← Model aggregation strategies
│   │   └── utils.py          ← Simulation runner
│   │
│   ├── explainability/       ← Model explanations
│   │   └── shap_explainer.py ← SHAP-based feature importance
│   │
│   ├── services/             ← Business logic
│   │   ├── bias_detector.py  ← Fairness metrics (demographic parity, etc.)
│   │   ├── bias_mitigation.py ← Reweighting, adversarial debiasing
│   │   ├── regulatory_compliance.py ← FCRA, ECOA, GDPR checks
│   │   ├── compliance_documentation.py ← Report generator
│   │   └── ingestion.py      ← Data loading + validation
│   │
│   └── data/
│       └── cross_validation.py ← CV strategies + result analysis
│
├── config/
│   └── base.yaml             ← Default configuration
│
└── tests/
    └── test_imports.py        ← Smoke tests for all modules
```

---

## Key Results

### Bank Dataset
| Metric | Full FP32 | Our Approach |
|--------|-----------|-------------|
| AUC | 0.9996 | 0.9982 |
| Carbon Cost | 512.52 | 64.87 |
| **Carbon Reduction** | — | **87.3%** |
| **Performance Retained** | — | **99.9%** |

### German Credit Dataset
| Metric | Full FP32 | Our Approach |
|--------|-----------|-------------|
| AUC | 0.8086 | 0.8129 |
| Carbon Cost | 512.25 | 64.60 |
| **Carbon Reduction** | — | **87.4%** |
| **Performance Retained** | — | **100.5%** |

---

## Tech Stack

- **Python 3.11** · PyTorch · scikit-learn · pandas · NumPy
- **FastAPI** + Uvicorn (REST APIs)
- **Streamlit** (Dashboard)
- **Flower** (Federated Learning framework)
- **SHAP** (Explainability)

---

## Useful Commands

```bash
# Activate environment
source venv/bin/activate

# Run dashboard
streamlit run streamlit_app.py

# Run NAS experiments
python -m app.sustainability.run_nas
python -m app.sustainability.run_nas_german

# Start APIs
./start_backend.sh

# Run tests
python -m pytest tests/ -v

# Lint & format
make lint
make format

# Stop APIs
pkill -f "uvicorn app.api.main:app"
pkill -f "run_inference_service"
```

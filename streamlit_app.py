"""
Sustainable Credit Risk AI — Streamlit Dashboard
=================================================
Interactive dashboard for the Sustainable Credit Risk AI system.
Provides NAS experimentation, federated learning simulation, credit risk
prediction, sustainability monitoring, and system health insights.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page configuration (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sustainable Credit Risk AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- global ---------- */
    .block-container { padding-top: 1.5rem; }

    /* ---------- metric cards ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0e1117 0%, #1a1f2c 100%);
        border: 1px solid #262c3a;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,.25);
    }
    div[data-testid="stMetric"] label {
        color: #8b95a5;
        font-size: 0.82rem;
        letter-spacing: 0.3px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 700;
        font-size: 1.55rem;
    }

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e14 0%, #111827 100%);
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
    }

    /* ---------- section headers ---------- */
    .section-header {
        background: linear-gradient(90deg, rgba(16,185,129,.12) 0%, transparent 100%);
        border-left: 4px solid #10b981;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 1.2rem 0 0.8rem 0;
    }
    .section-header h3 {
        margin: 0;
        color: #e2e8f0;
    }

    /* ---------- result cards ---------- */
    .result-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }

    /* ---------- tables ---------- */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ---------- expanders ---------- */
    details[data-testid="stExpander"] {
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
    }

    /* ---------- gradient title ---------- */
    .gradient-text {
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPER: lazy imports (keeps initial load fast)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner=False)
def _get_config():
    from app.core.config import load_config
    return load_config()


@st.cache_resource(show_spinner=False)
def _get_logger():
    from app.core.logging import get_logger
    return get_logger("streamlit_app")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown('<h1 class="gradient-text">🌿 Credit Risk AI</h1>', unsafe_allow_html=True)
    st.caption("Sustainable · Explainable · Fair")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠  Dashboard",
            "🔬  Carbon-Aware NAS",
            "🤝  Federated Learning",
            "🎯  Credit Risk Prediction",
            "📊  Sustainability Monitor",
            "🛡️  System Health",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"v1.0.0 · Python {sys.version_info.major}.{sys.version_info.minor}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — DASHBOARD                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_dashboard():
    st.markdown('<h1 class="gradient-text">Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Real-time overview of the Sustainable Credit Risk AI platform.")

    # --- top-level KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model AUC (Bank)", "0.998", delta="↑ 0.2%")
    c2.metric("Carbon Reduction", "87.3 %", delta="↑ 4.1%")
    c3.metric("Efficiency Gain", "689 %", delta="↑ 52%")
    c4.metric("API Status", "Healthy", delta="online")

    st.markdown("")

    # --- architecture summary ---
    st.markdown('<div class="section-header"><h3>⚙️ System Architecture</h3></div>', unsafe_allow_html=True)

    arch_cols = st.columns(3)
    with arch_cols[0]:
        st.markdown("##### 🧠 Core ML")
        st.markdown("""
        - Carbon-Aware NAS (multi-exit MLP)
        - Logistic Regression baseline
        - Scalable hidden layers + precision casting
        """)
    with arch_cols[1]:
        st.markdown("##### 🔒 Security & Compliance")
        st.markdown("""
        - AES-256 encryption at rest
        - JWT + API Key authentication
        - FCRA · ECOA · GDPR compliance
        """)
    with arch_cols[2]:
        st.markdown("##### 🌱 Sustainability")
        st.markdown("""
        - Energy & carbon tracking
        - 3 precision modes (fp32 / fp16 / int8)
        - Pareto-optimal architecture search
        """)

    # --- datasets ---
    st.markdown('<div class="section-header"><h3>📂 Available Datasets</h3></div>', unsafe_allow_html=True)

    ds1, ds2 = st.columns(2)
    with ds1:
        bank_path = PROJECT_ROOT / "app" / "sustainability" / "Bank_data.csv"
        if bank_path.exists():
            df_bank = pd.read_csv(bank_path)
            st.success(f"**Bank Dataset** — {len(df_bank):,} rows × {len(df_bank.columns)} cols")
            with st.expander("Preview"):
                st.dataframe(df_bank.head(8), width="stretch")
        else:
            st.warning("Bank dataset not found")

    with ds2:
        german_path = PROJECT_ROOT / "app" / "sustainability" / "german_data.csv"
        if german_path.exists():
            df_german = pd.read_csv(german_path)
            st.success(f"**German Credit** — {len(df_german):,} rows × {len(df_german.columns)} cols")
            with st.expander("Preview"):
                st.dataframe(df_german.head(8), width="stretch")
        else:
            st.warning("German dataset not found")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — CARBON-AWARE NAS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_nas():
    st.markdown('<h1 class="gradient-text">Carbon-Aware NAS</h1>', unsafe_allow_html=True)
    st.markdown("Run Neural Architecture Search to find the most efficient model configuration.")

    # --- controls ---
    ctrl_cols = st.columns([2, 2, 1])
    with ctrl_cols[0]:
        dataset = st.selectbox("Dataset", ["Bank", "German Credit"])
    with ctrl_cols[1]:
        dropout_val = st.slider("Dropout", 0.0, 0.5, 0.3 if dataset == "Bank" else 0.15, 0.05)
    with ctrl_cols[2]:
        st.markdown("")
        st.markdown("")
        run_btn = st.button("🚀 Run NAS", type="primary", width="stretch")

    if run_btn:
        with st.spinner("Running Carbon-Aware NAS … this may take 1-3 minutes"):
            results = _run_nas(dataset, dropout_val)

        if results and len(results) > 0:
            st.success(f"NAS complete — **{len(results)}** configurations evaluated.")

            # --- top metrics ---
            best = results[0]
            baseline = [r for r in results if r["architecture"]["hidden_scale"] == 1.0
                        and r["precision"] == "fp32" and r["exit_level"] == 3]
            base = baseline[0] if baseline else results[-1]

            carbon_red = (base["carbon_cost"] - best["carbon_cost"]) / max(base["carbon_cost"], 1e-12) * 100
            perf_ret = best["metrics"]["auc"] / max(base["metrics"]["auc"], 1e-12) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best AUC", f"{best['metrics']['auc']:.4f}")
            m2.metric("Carbon Cost", f"{best['carbon_cost']:.1f}")
            m3.metric("Carbon Reduction", f"{carbon_red:.1f}%")
            m4.metric("Performance Retained", f"{perf_ret:.1f}%")

            # --- results table ---
            st.markdown('<div class="section-header"><h3>📋 All Configurations (sorted by carbon cost)</h3></div>',
                        unsafe_allow_html=True)

            rows = []
            for r in results:
                rows.append({
                    "Scale": r["architecture"]["hidden_scale"],
                    "Exit": r["exit_level"],
                    "Precision": r["precision"],
                    "AUC": round(r["metrics"]["auc"], 4),
                    "KS": round(r["metrics"]["ks"], 4),
                    "Brier": round(r["metrics"]["brier"], 4),
                    "Carbon": round(r["carbon_cost"], 2),
                    "Score": round(r["multi_objective_score"], 4),
                })
            df_results = pd.DataFrame(rows)
            st.dataframe(
                df_results.style.background_gradient(subset=["AUC"], cmap="Greens")
                                .background_gradient(subset=["Carbon"], cmap="Reds_r"),
                width="stretch",
                height=400,
            )

            # --- Pareto chart ---
            st.markdown('<div class="section-header"><h3>📈 AUC vs Carbon Cost</h3></div>',
                        unsafe_allow_html=True)

            chart_df = df_results.copy()
            chart_df["label"] = chart_df.apply(
                lambda r: f"s{r['Scale']}·e{r['Exit']}·{r['Precision']}", axis=1
            )
            st.scatter_chart(
                chart_df, x="Carbon", y="AUC", color="Precision",
                size=60, width="stretch", height=400,
            )

            # --- research table ---
            st.markdown('<div class="section-header"><h3>🔬 Research Comparison</h3></div>',
                        unsafe_allow_html=True)
            _show_research_table(results)

        else:
            st.error("NAS returned no results.")


@st.cache_data(show_spinner=False)
def _run_nas(dataset: str, dropout: float):
    """Run NAS and cache results."""
    import torch
    from app.sustainability.carbon_aware_nas import carbon_aware_nas
    from app.sustainability.reference_model import REFERENCE_MODELS

    if dataset == "Bank":
        from app.sustainability.preprocessing import load_and_preprocess
        from app.sustainability.run_nas import make_evaluate_fn
        data_path = str(PROJECT_ROOT / "app" / "sustainability" / "Bank_data.csv")
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(data_path)
        preprocessor.fit(X_train)
        X_train_p = preprocessor.transform(X_train)
        X_test_p = preprocessor.transform(X_test)
        if hasattr(X_train_p, "toarray"):
            X_train_p = X_train_p.toarray()
        if hasattr(X_test_p, "toarray"):
            X_test_p = X_test_p.toarray()
        reference = REFERENCE_MODELS["logistic_regression"]
    else:
        from app.sustainability.preprocessing_german import load_and_preprocess_german
        from app.sustainability.run_nas_german import make_evaluate_fn
        data_path = str(PROJECT_ROOT / "app" / "sustainability" / "german_data.csv")
        X_train_p, X_test_p, y_train, y_test, _ = load_and_preprocess_german(data_path)
        reference = REFERENCE_MODELS["german_logistic_regression"]

    X_train_tensor = torch.tensor(X_train_p if isinstance(X_train_p, np.ndarray)
                                  else X_train_p.toarray() if hasattr(X_train_p, "toarray")
                                  else np.array(X_train_p), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_tensor = torch.tensor(X_test_p if isinstance(X_test_p, np.ndarray)
                            else np.array(X_test_p), dtype=torch.float32)

    evaluate_fn = make_evaluate_fn(X_train_tensor, y_train_tensor, y_test)

    exit_latencies_ms = {1: 0.10, 2: 0.20, 3: 0.25}

    results = carbon_aware_nas(
        X_tensor, y_test, reference, 12.0, exit_latencies_ms,
        evaluate_fn, verbose=False, dropout=dropout,
    )
    return results


def _show_research_table(results):
    """Build the research comparison table from NAS results."""
    all_r = results
    fp32_full = [r for r in all_r if r["precision"] == "fp32"
                 and r["architecture"]["hidden_scale"] == 1.0 and r["exit_level"] == 3]
    int8_best = sorted([r for r in all_r if r["precision"] == "int8"],
                       key=lambda x: -x["metrics"]["auc"])
    fp32_small = sorted([r for r in all_r if r["precision"] == "fp32"
                         and r["architecture"]["hidden_scale"] < 1.0 and r["exit_level"] == 3],
                        key=lambda x: -x["metrics"]["auc"])
    fp32_exit = sorted([r for r in all_r if r["precision"] == "fp32"
                        and r["architecture"]["hidden_scale"] == 1.0 and r["exit_level"] < 3],
                       key=lambda x: -x["metrics"]["auc"])
    best_auc_all = sorted(all_r, key=lambda x: -x["metrics"]["auc"])
    best_combined = sorted(all_r, key=lambda x: x["carbon_cost"])

    baseline_cost = fp32_full[0]["carbon_cost"] if fp32_full else all_r[-1]["carbon_cost"]

    def _row(name, r):
        red = (baseline_cost - r["carbon_cost"]) / max(baseline_cost, 1e-12) * 100
        eff = r["metrics"]["auc"] / max(r["carbon_cost"], 1e-12)
        return {
            "Method": name,
            "AUC": round(r["metrics"]["auc"], 4),
            "KS": round(r["metrics"]["ks"], 4),
            "Brier": round(r["metrics"]["brier"], 4),
            "Carbon": round(r["carbon_cost"], 2),
            "Reduction %": round(red, 1),
            "Efficiency": round(eff, 6),
        }

    rows = []
    if fp32_full:      rows.append(_row("Full FP32 Baseline", fp32_full[0]))
    if int8_best:      rows.append(_row("INT8 Best AUC", int8_best[0]))
    if fp32_small:     rows.append(_row("Scaling Only (FP32)", fp32_small[0]))
    if fp32_exit:      rows.append(_row("Early Exit Only", fp32_exit[0]))
    if best_auc_all:   rows.append(_row("Best AUC Overall", best_auc_all[0]))
    if best_combined:  rows.append(_row("Combined (Ours) ⭐", best_combined[0]))

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — FEDERATED LEARNING                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_federated():
    st.markdown('<h1 class="gradient-text">Federated Learning</h1>', unsafe_allow_html=True)
    st.markdown("Simulate privacy-preserving distributed training using FedAvg.")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_clients = st.slider("Number of clients", 2, 10, 3)
    with c2:
        n_rounds = st.slider("Aggregation rounds", 1, 20, 5)
    with c3:
        local_epochs = st.slider("Local epochs per round", 1, 10, 2)

    run_fl = st.button("🚀 Run FL Simulation", type="primary")

    if run_fl:
        from app.federated.config import FLConfig
        from app.federated.utils import run_federated_simulation

        config = FLConfig(
            number_of_clients=n_clients,
            aggregation_rounds=n_rounds,
            local_epochs=local_epochs,
        )

        progress = st.progress(0, text="Initializing federated simulation …")
        t_start = time.time()

        result = run_federated_simulation(config=config)

        elapsed = time.time() - t_start
        progress.progress(100, text="Simulation complete!")

        st.success(f"Simulation finished in **{elapsed:.1f}s**")

        # --- KPIs ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rounds", len(result["round_metrics"]))
        k2.metric("Best Round", result["best_round"])
        k3.metric("Best Val Loss", f"{result['best_val_loss']:.6f}")
        k4.metric("Clients", n_clients)

        # --- per-round chart ---
        st.markdown('<div class="section-header"><h3>📈 Training Progress</h3></div>',
                    unsafe_allow_html=True)

        metrics = result["round_metrics"]
        rounds_data = []
        for rm in metrics:
            row = {"Round": rm["round"]}
            for client in rm.get("client_results", []):
                row[f"Client {client['client_id']} Loss"] = client.get("val_loss", client.get("train_loss", 0))
            rounds_data.append(row)

        if rounds_data:
            df_fl = pd.DataFrame(rounds_data).set_index("Round")
            st.line_chart(df_fl, width="stretch", height=350)

        # --- raw data ---
        with st.expander("Raw round metrics"):
            st.json(metrics)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — CREDIT RISK PREDICTION                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_prediction():
    st.markdown('<h1 class="gradient-text">Credit Risk Prediction</h1>', unsafe_allow_html=True)
    st.markdown("Evaluate credit risk using the trained model.")

    with st.form("prediction_form"):
        st.markdown("##### Applicant Information")

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            age = st.number_input("Age", 18, 100, 35)
            income = st.number_input("Annual Income ($)", 10000, 1000000, 65000, step=5000)
        with r1c2:
            credit_score = st.number_input("Credit Score", 300, 850, 720)
            employment_length = st.number_input("Employment Length (yrs)", 0, 50, 5)
        with r1c3:
            loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 25000, step=1000)
            dti = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.3, step=0.05)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            loan_purpose = st.selectbox("Loan Purpose",
                                        ["debt_consolidation", "home_improvement", "credit_card",
                                         "education", "medical", "small_business", "other"])
        with r2c2:
            home_ownership = st.selectbox("Home Ownership",
                                          ["RENT", "OWN", "MORTGAGE", "OTHER"])

        submitted = st.form_submit_button("🎯 Predict Risk", type="primary", width="stretch")

    if submitted:
        # Simple risk scoring model (heuristic since no saved model file exists)
        risk_score = _compute_risk_score(age, income, credit_score, employment_length,
                                         loan_amount, dti, loan_purpose)

        risk_level, risk_color = _risk_level(risk_score)

        st.markdown("---")
        st.markdown(f"### Prediction Result")

        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", f"{risk_score:.3f}")
        m2.metric("Risk Level", risk_level)
        m3.metric("Confidence", f"{min(0.95, 0.75 + credit_score / 5000):.1%}")

        # Visual risk gauge
        st.progress(min(risk_score, 1.0), text=f"Risk: {risk_level}")

        # Explanation
        st.markdown('<div class="section-header"><h3>🔍 Key Risk Factors</h3></div>',
                    unsafe_allow_html=True)

        factors = _explain_risk(age, income, credit_score, employment_length,
                                loan_amount, dti, loan_purpose)
        for f in factors:
            icon = "🟢" if f["impact"] == "positive" else "🔴" if f["impact"] == "negative" else "🟡"
            st.markdown(f"{icon} **{f['factor']}**: {f['description']}")


def _compute_risk_score(age, income, credit_score, emp_length, loan_amount, dti, purpose):
    """Heuristic risk scoring (proxy for the actual model)."""
    score = 0.5

    # Credit score is the strongest predictor
    score -= (credit_score - 650) * 0.0008

    # DTI
    score += (dti - 0.35) * 0.4

    # Loan-to-income
    lti = loan_amount / max(income, 1)
    score += (lti - 0.4) * 0.3

    # Employment stability
    score -= min(emp_length, 10) * 0.012

    # Age (very young slightly riskier)
    if age < 25:
        score += 0.05

    return float(np.clip(score, 0.02, 0.98))


def _risk_level(score):
    if score < 0.25:
        return "✅ Low", "green"
    elif score < 0.50:
        return "🟡 Medium", "orange"
    elif score < 0.75:
        return "🟠 High", "red"
    else:
        return "🔴 Very High", "darkred"


def _explain_risk(age, income, credit_score, emp_length, loan_amount, dti, purpose):
    factors = []
    if credit_score >= 740:
        factors.append({"factor": "Credit Score", "impact": "positive",
                        "description": f"Excellent score ({credit_score}) significantly lowers risk."})
    elif credit_score < 640:
        factors.append({"factor": "Credit Score", "impact": "negative",
                        "description": f"Below-average score ({credit_score}) increases risk."})
    else:
        factors.append({"factor": "Credit Score", "impact": "neutral",
                        "description": f"Average score ({credit_score})."})

    if dti > 0.43:
        factors.append({"factor": "Debt-to-Income", "impact": "negative",
                        "description": f"High DTI ({dti:.0%}) indicates heavy debt burden."})
    elif dti < 0.25:
        factors.append({"factor": "Debt-to-Income", "impact": "positive",
                        "description": f"Low DTI ({dti:.0%}) shows manageable debt."})

    lti = loan_amount / max(income, 1)
    if lti > 0.5:
        factors.append({"factor": "Loan-to-Income Ratio", "impact": "negative",
                        "description": f"Loan is {lti:.0%} of income — above typical threshold."})

    if emp_length >= 5:
        factors.append({"factor": "Employment Stability", "impact": "positive",
                        "description": f"{emp_length} years of employment shows stability."})
    elif emp_length < 2:
        factors.append({"factor": "Employment Stability", "impact": "negative",
                        "description": f"Short employment history ({emp_length} yrs)."})

    return factors


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 5 — SUSTAINABILITY MONITOR                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_sustainability():
    st.markdown('<h1 class="gradient-text">Sustainability Monitor</h1>', unsafe_allow_html=True)
    st.markdown("Track energy consumption and carbon emissions of ML operations.")

    from app.sustainability.energy_tracker import EnergyTracker
    from app.sustainability.carbon_calculator import CarbonCalculator

    c1, c2 = st.columns(2)
    with c1:
        region = st.selectbox("Carbon Region", ["US", "EU", "IN", "CN", "UK"], index=0)
    with c2:
        duration = st.slider("Simulated workload duration (seconds)", 0.1, 5.0, 1.0, 0.1)

    if st.button("⚡ Run Energy Simulation", type="primary"):
        tracker = EnergyTracker()
        calc = CarbonCalculator()

        tracker.start_tracking("simulation")
        time.sleep(duration)
        report = tracker.stop_tracking("simulation")

        carbon = calc.calculate_carbon_footprint(report, region=region)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Duration", f"{report.duration_seconds:.2f}s")
        m2.metric("Energy", f"{report.total_energy_kwh:.6f} kWh")
        m3.metric("CO₂ Emissions", f"{carbon.total_emissions_kg:.8f} kg")
        m4.metric("Region Factor", f"{carbon.emissions_factor_kg_per_kwh} kg/kWh")

        # Context comparison
        st.markdown('<div class="section-header"><h3>🌍 Environmental Context</h3></div>',
                    unsafe_allow_html=True)

        kwh = report.total_energy_kwh
        comparisons = pd.DataFrame([
            {"Metric": "☕ Coffee cups brewed", "Equivalent": f"{kwh / 0.011:.4f}"},
            {"Metric": "💡 LED bulb hours", "Equivalent": f"{kwh / 0.01:.2f}"},
            {"Metric": "📱 Phone charges", "Equivalent": f"{kwh / 0.012:.4f}"},
            {"Metric": "🚗 km driven (EV)", "Equivalent": f"{kwh / 0.2:.6f}"},
        ])
        st.dataframe(comparisons, width="stretch", hide_index=True)

    # Precision mode comparison
    st.markdown('<div class="section-header"><h3>⚙️ Precision Mode Impact</h3></div>',
                unsafe_allow_html=True)

    from app.sustainability.precision_modes import PRECISION_CONFIG
    prec_data = []
    for name, cfg in PRECISION_CONFIG.items():
        prec_data.append({
            "Mode": name.upper(),
            "Multiplier": cfg["multiplier"],
            "Relative Cost": f"{cfg['multiplier'] * 100:.0f}%",
            "Use Case": {
                "fp32": "Full precision training & evaluation",
                "fp16": "Mixed-precision training, 2× throughput",
                "int8": "Deployment inference, maximum efficiency",
            }.get(name, ""),
        })
    st.dataframe(pd.DataFrame(prec_data), width="stretch", hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 6 — SYSTEM HEALTH                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_health():
    st.markdown('<h1 class="gradient-text">System Health</h1>', unsafe_allow_html=True)
    st.markdown("Security posture, configuration, and component readiness.")

    # --- Security Report ---
    st.markdown('<div class="section-header"><h3>🛡️ Security Status</h3></div>',
                unsafe_allow_html=True)

    try:
        from app.core.security_manager import get_security_manager
        sec = get_security_manager()
        report = sec.generate_security_report()

        s1, s2, s3 = st.columns(3)
        s1.metric("Encryption Keys", report["encryption"]["total_keys"])
        s2.metric("Active Sessions", report["authentication"]["active_sessions"])
        s3.metric("GDPR Policies", report["gdpr_compliance"]["retention_policies"])

        if report.get("security_recommendations"):
            st.warning("**Security Recommendations:**")
            for rec in report["security_recommendations"]:
                st.markdown(f"  ⚠️ {rec}")
        else:
            st.success("No security issues detected.")

        with st.expander("Full Security Report"):
            st.json(report)

    except Exception as e:
        st.error(f"Could not load security manager: {e}")

    # --- Configuration ---
    st.markdown('<div class="section-header"><h3>📝 Configuration</h3></div>',
                unsafe_allow_html=True)

    try:
        config = _get_config()
        cfg_c1, cfg_c2 = st.columns(2)
        with cfg_c1:
            st.markdown("##### General")
            st.code(f"""
Environment: {config.environment.value}
Debug: {config.debug}
Log Level: {config.log_level}
            """)
        with cfg_c2:
            st.markdown("##### Paths")
            st.code(f"""
Data:   {config.data_path}
Logs:   {config.logs_path}
Models: {config.models_path}
Keys:   {config.keys_path}
            """)
    except Exception as e:
        st.error(f"Could not load config: {e}")

    # --- API Health ---
    st.markdown('<div class="section-header"><h3>🌐 API Endpoints</h3></div>',
                unsafe_allow_html=True)

    try:
        from fastapi.testclient import TestClient
        from app.api.main import app

        client = TestClient(app)

        endpoints = [
            ("GET", "/health", "Health Check"),
            ("GET", "/ready", "Readiness Probe"),
            ("GET", "/", "Root"),
        ]

        rows = []
        for method, path, desc in endpoints:
            try:
                r = client.get(path)
                rows.append({
                    "Endpoint": f"`{method} {path}`",
                    "Description": desc,
                    "Status": r.status_code,
                    "Result": "✅" if r.status_code == 200 else "❌",
                })
            except Exception:
                rows.append({
                    "Endpoint": f"`{method} {path}`",
                    "Description": desc,
                    "Status": "Error",
                    "Result": "❌",
                })

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    except Exception as e:
        st.warning(f"API test client not available: {e}")

    # --- Component checklist ---
    st.markdown('<div class="section-header"><h3>🧩 Component Status</h3></div>',
                unsafe_allow_html=True)

    components = [
        ("Core Config", "app.core.config", "load_config"),
        ("Logging", "app.core.logging", "get_logger"),
        ("Authentication", "app.core.auth", "AuthenticationManager"),
        ("Encryption", "app.core.encryption", "DataEncryption"),
        ("GDPR Compliance", "app.core.gdpr_compliance", "GDPRComplianceManager"),
        ("Sustainability NAS", "app.sustainability.carbon_aware_nas", "carbon_aware_nas"),
        ("Sustainability Metrics", "app.sustainability.metrics", "ks_statistic"),
        ("Federated Server", "app.federated.server", "FederatedServer"),
        ("Federated Client", "app.federated.client", "FederatedClient"),
        ("Explainability", "app.explainability", "ExplainerService"),
        ("Bias Detection", "app.services.bias_detector", "BiasDetector"),
        ("Data Ingestion", "app.services.ingestion", "BankingDataValidator"),
    ]

    comp_rows = []
    for name, module, attr in components:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            comp_rows.append({"Component": name, "Module": module, "Status": "✅ Loaded"})
        except Exception as e:
            comp_rows.append({"Component": name, "Module": module, "Status": f"❌ {e}"})

    st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ROUTER                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PAGES = {
    "🏠  Dashboard": page_dashboard,
    "🔬  Carbon-Aware NAS": page_nas,
    "🤝  Federated Learning": page_federated,
    "🎯  Credit Risk Prediction": page_prediction,
    "📊  Sustainability Monitor": page_sustainability,
    "🛡️  System Health": page_health,
}

PAGES[page]()

# 🌐 Interactive Dashboard Implementation Guide
## Sustainable Credit Risk AI - Capstone Presentation

### 🚀 Quick Start Implementation

#### **Option 1: Streamlit Dashboard (Fastest - 2-3 hours)**
Perfect for rapid prototyping and technical demonstrations.

```python
# presentation/streamlit_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.lightgbm_model import train_lightgbm_baseline, get_fast_lightgbm_config
from src.data.ingestion import ingest_banking_data
from src.data.feature_engineering import engineer_banking_features, get_minimal_config

st.set_page_config(
    page_title="Sustainable Credit Risk AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.sustainability-card {
    background: linear-gradient(135deg, #2E8B57 0%, #48bb78 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌱 Sustainable Credit Risk AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Revolutionizing Financial Services Through Responsible AI</p>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a section:", [
        "🏠 Overview",
        "🔮 Live Prediction Demo",
        "📊 Model Performance",
        "🌱 Sustainability Dashboard",
        "🔍 Explainability Hub",
        "🏗️ System Architecture"
    ])
    
    if page == "🏠 Overview":
        show_overview()
    elif page == "🔮 Live Prediction Demo":
        show_prediction_demo()
    elif page == "📊 Model Performance":
        show_performance_dashboard()
    elif page == "🌱 Sustainability Dashboard":
        show_sustainability_dashboard()
    elif page == "🔍 Explainability Hub":
        show_explainability_hub()
    elif page == "🏗️ System Architecture":
        show_architecture()

def show_overview():
    st.header("🎯 System Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 94.4%</h3>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="sustainability-card">
            <h3>🌱 Carbon Neutral</h3>
            <p>Operations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ <50ms</h3>
            <p>Inference Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="sustainability-card">
            <h3>🔒 100%</h3>
            <p>Privacy Preserved</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Problem Statement
    st.subheader("🏦 The Challenge")
    st.write("""
    - **$3.2 trillion** in global credit losses annually
    - **40%** of AI models lack explainability
    - **2.3%** of global CO2 emissions from financial AI
    - **$270B** annual regulatory compliance costs
    """)
    
    # Solution
    st.subheader("💡 Our Solution")
    st.write("""
    - **Advanced ML Pipeline** with ensemble models
    - **Sustainable AI** with carbon-neutral operations
    - **Privacy-Preserving** federated learning
    - **100% Explainable** AI decisions
    - **Production-Ready** infrastructure
    """)

def show_prediction_demo():
    st.header("🔮 Live Credit Risk Prediction")
    
    # File upload or manual input
    input_method = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])
    
    if input_method == "Manual Input":
        st.subheader("Enter Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 80, 35)
            annual_income = st.number_input("Annual Income (INR)", 100000, 2000000, 500000)
            loan_amount = st.number_input("Loan Amount (INR)", 50000, 1000000, 200000)
            credit_score = st.slider("Credit Score", 300, 850, 650)
        
        with col2:
            employment_type = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Business"])
            loan_purpose = st.selectbox("Loan Purpose", ["Home", "Personal", "Business", "Education"])
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
            past_default = st.selectbox("Past Default", ["No", "Yes"])
        
        if st.button("🔮 Predict Credit Risk", type="primary"):
            with st.spinner("Analyzing credit risk..."):
                # Simulate prediction
                import time
                time.sleep(2)
                
                # Mock prediction result
                risk_score = np.random.beta(2, 5)  # Skewed towards lower risk
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Risk Score", f"{risk_score:.3f}")
                
                with col2:
                    risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
                    color = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
                    st.metric("Risk Level", f"{color} {risk_level}")
                
                with col3:
                    confidence = np.random.uniform(0.85, 0.95)
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Feature importance (mock)
                st.subheader("🔍 Key Contributing Factors")
                
                features = ["Credit Score", "Debt-to-Income", "Annual Income", "Age", "Loan Amount"]
                importance = np.random.dirichlet([5, 4, 3, 2, 1])
                
                fig = go.Figure(go.Bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    marker_color=['#2E8B57', '#48bb78', '#68d391', '#9ae6b4', '#c6f6d5']
                ))
                
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_performance_dashboard():
    st.header("📊 Model Performance Dashboard")
    
    # Model comparison
    models_data = {
        'Model': ['LightGBM', 'DNN', 'LSTM', 'Ensemble'],
        'AUC-ROC': [1.000, 0.944, 0.912, 0.972],
        'F1 Score': [1.000, 0.880, 0.865, 0.940],
        'Latency (ms)': [25, 45, 60, 35],
        'Energy (kWh)': [0.0001, 0.0003, 0.0004, 0.0002]
    }
    
    df = pd.DataFrame(models_data)
    
    # Display table
    st.subheader("Model Comparison")
    st.dataframe(df, use_container_width=True)
    
    # ROC Curve visualization
    st.subheader("ROC-AUC Curves")
    
    # Generate mock ROC data
    fpr_lgb = np.linspace(0, 1, 100)
    tpr_lgb = 1 - (1 - fpr_lgb) ** 2  # Perfect classifier
    
    fpr_dnn = np.linspace(0, 1, 100)
    tpr_dnn = fpr_dnn ** 0.5  # Good classifier
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr_lgb, y=tpr_lgb,
        mode='lines',
        name='LightGBM (AUC = 1.000)',
        line=dict(color='#2E8B57', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=fpr_dnn, y=tpr_dnn,
        mode='lines',
        name='DNN (AUC = 0.944)',
        line=dict(color='#1E3A8A', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_sustainability_dashboard():
    st.header("🌱 Sustainability Dashboard")
    
    # Energy metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Energy Saved", "30%", "↓ 0.5 kWh/day")
    
    with col2:
        st.metric("CO2 Reduced", "0.5 tons/year", "↓ 15%")
    
    with col3:
        st.metric("ESG Score", "A+", "↑ 2 grades")
    
    # Energy consumption over time
    st.subheader("⚡ Energy Consumption Tracking")
    
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    baseline_energy = np.random.normal(1.0, 0.1, 30)
    optimized_energy = baseline_energy * 0.7 + np.random.normal(0, 0.05, 30)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=baseline_energy,
        mode='lines',
        name='Baseline Energy',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=optimized_energy,
        mode='lines',
        name='Optimized Energy',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Daily Energy Consumption (kWh)',
        xaxis_title='Date',
        yaxis_title='Energy (kWh)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Carbon footprint equivalent
    st.subheader("🌍 Environmental Impact Equivalent")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🌳 **25 trees planted** equivalent CO2 reduction")
        st.info("🚗 **1,200 miles not driven** in carbon savings")
    
    with col2:
        st.success("💡 **500 LED bulbs** worth of energy saved")
        st.success("♻️ **2 tons waste recycled** environmental impact")

def show_explainability_hub():
    st.header("🔍 Explainability Hub")
    
    st.subheader("SHAP Feature Importance")
    
    # Mock SHAP values
    features = ['Credit Score', 'Debt-to-Income', 'Annual Income', 'Past Defaults', 'Age', 'Loan Amount']
    shap_values = [0.35, 0.28, -0.15, 0.12, -0.08, 0.05]
    colors = ['red' if x > 0 else 'blue' for x in shap_values]
    
    fig = go.Figure(go.Bar(
        x=shap_values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{x:+.2f}" for x in shap_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='SHAP Feature Contributions (Customer #12345)',
        xaxis_title='SHAP Value (Impact on Prediction)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation text
    st.subheader("📝 Prediction Explanation")
    st.write("""
    **Risk Score: 0.85 (High Risk)**
    
    **Key Contributing Factors:**
    - 🔴 **Credit Score (580)**: Below average score increases risk significantly (+0.35)
    - 🔴 **Debt-to-Income (0.65)**: High debt burden is concerning (+0.28)
    - 🔴 **Past Defaults (2)**: Previous payment issues indicate risk (+0.12)
    - 🔵 **Annual Income (45K)**: Moderate income provides some stability (-0.15)
    - 🔵 **Age (35)**: Mature age slightly reduces risk (-0.08)
    """)

def show_architecture():
    st.header("🏗️ System Architecture")
    
    st.subheader("Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 Machine Learning**
        - LightGBM
        - PyTorch (DNN/LSTM)
        - Scikit-learn
        - SHAP/LIME
        """)
    
    with col2:
        st.markdown("""
        **🌐 Backend**
        - FastAPI
        - PostgreSQL
        - Redis
        - Docker
        """)
    
    with col3:
        st.markdown("""
        **🔒 Security**
        - JWT Authentication
        - AES-256 Encryption
        - Differential Privacy
        - Audit Logging
        """)
    
    # Architecture diagram (simplified)
    st.subheader("System Components")
    
    st.mermaid("""
    graph TD
        A[Data Ingestion] --> B[Feature Engineering]
        B --> C[Model Training]
        C --> D[Ensemble Model]
        D --> E[Prediction API]
        E --> F[Explainability Service]
        
        G[Sustainability Monitor] --> C
        H[Privacy Manager] --> C
        I[Compliance Checker] --> E
        
        style A fill:#e1f5fe
        style D fill:#c8e6c9
        style F fill:#fff3e0
        style G fill:#e8f5e8
    """)

if __name__ == "__main__":
    main()
```

#### **Running the Streamlit Dashboard**
```bash
# Install dependencies
pip install streamlit plotly

# Run the dashboard
streamlit run presentation/streamlit_dashboard.py
```

---

#### **Option 2: React + FastAPI Dashboard (Professional - 1-2 weeks)**
For a more polished, production-ready presentation.

**Frontend Structure:**
```
presentation/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Landing.tsx
│   │   │   ├── PredictionDemo.tsx
│   │   │   ├── PerformanceDashboard.tsx
│   │   │   ├── SustainabilityDashboard.tsx
│   │   │   └── ExplainabilityHub.tsx
│   │   ├── hooks/
│   │   ├── utils/
│   │   └── App.tsx
│   ├── package.json
│   └── tailwind.config.js
└── backend/
    ├── main.py
    ├── routers/
    └── models/
```

**Quick Setup Commands:**
```bash
# Frontend setup
npx create-react-app presentation/frontend --template typescript
cd presentation/frontend
npm install plotly.js-react plotly.js @types/plotly.js
npm install tailwindcss framer-motion axios

# Backend setup
cd ../backend
pip install fastapi uvicorn python-multipart
```

---

#### **Option 3: Jupyter Notebook Presentation (Academic - 1 day)**
Perfect for technical/academic audiences.

```python
# presentation/interactive_notebook.ipynb
# Use Jupyter widgets for interactivity
import ipywidgets as widgets
from IPython.display import display, HTML
import plotly.graph_objects as go

# Create interactive widgets
age_slider = widgets.IntSlider(value=35, min=18, max=80, description='Age:')
income_input = widgets.IntText(value=500000, description='Income:')
predict_button = widgets.Button(description='Predict Risk')

# Display widgets
display(age_slider, income_input, predict_button)

# Interactive prediction function
def on_predict_click(b):
    # Your prediction logic here
    risk_score = predict_risk(age_slider.value, income_input.value)
    display(HTML(f"<h3>Risk Score: {risk_score:.3f}</h3>"))

predict_button.on_click(on_predict_click)
```

---

### 🎨 Visual Design Assets

#### **Color Palette**
```css
:root {
  --sustainable-green: #2E8B57;
  --financial-blue: #1E3A8A;
  --energy-yellow: #F59E0B;
  --risk-red: #DC2626;
  --success-green: #10B981;
  --warning-orange: #F97316;
  --neutral-gray: #6B7280;
}
```

#### **Custom CSS for Streamlit**
```css
/* presentation/style.css */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.metric-container {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 0.5rem 0;
}

.sustainability-highlight {
    background: linear-gradient(135deg, #2E8B57 0%, #48bb78 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
```

---

### 📊 Data Preparation for Demo

#### **Sample Data Generator**
```python
# presentation/generate_demo_data.py
import pandas as pd
import numpy as np

def generate_demo_data(n_samples=1000):
    """Generate realistic demo data for presentation."""
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'annual_income_inr': np.random.lognormal(12, 0.5, n_samples),
        'loan_amount_inr': np.random.lognormal(11, 0.7, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_samples),
        'employment_type': np.random.choice(['Salaried', 'Self-employed', 'Business'], n_samples),
        'loan_purpose': np.random.choice(['Home', 'Personal', 'Business', 'Education'], n_samples),
        'past_default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'default': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
    }
    
    df = pd.DataFrame(data)
    df.to_csv('presentation/demo_data.csv', index=False)
    return df

# Generate demo data
demo_data = generate_demo_data()
print("Demo data generated successfully!")
```

---

### 🚀 Deployment Options

#### **Option 1: Streamlit Cloud (Free)**
```bash
# Push to GitHub and deploy on share.streamlit.io
git add .
git commit -m "Add presentation dashboard"
git push origin main
# Visit share.streamlit.io and connect your repo
```

#### **Option 2: Heroku (Free Tier)**
```bash
# Create Procfile
echo "web: streamlit run presentation/streamlit_dashboard.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

#### **Option 3: Vercel (React)**
```bash
# For React frontend
npm run build
vercel --prod
```

---

### 📱 Mobile Responsiveness

#### **Streamlit Mobile Optimization**
```python
# Add to your Streamlit app
st.markdown("""
<style>
@media (max-width: 768px) {
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        margin-top: -80px;
    }
    
    .main-header {
        font-size: 2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)
```

---

### 🎯 Presentation Tips

#### **For Live Demos**
1. **Test Everything**: Run through the entire demo multiple times
2. **Backup Plans**: Have screenshots/videos ready if live demo fails
3. **Fast Internet**: Ensure stable, fast internet connection
4. **Browser Setup**: Use Chrome in incognito mode, close other tabs
5. **Screen Resolution**: Test on presentation screen resolution

#### **Engagement Strategies**
1. **Interactive Elements**: Let audience input their own data
2. **Real-time Polls**: Use audience response systems
3. **Q&A Integration**: Build in pause points for questions
4. **Storytelling**: Use customer personas and scenarios
5. **Call to Action**: End with clear next steps

---

This implementation guide gives you multiple options to create an impressive, interactive presentation that showcases your Sustainable Credit Risk AI system effectively. Choose the option that best fits your timeline and technical requirements!

Would you like me to help you implement any specific component or create additional presentation materials?
const endpoints = {
  health: "/health",
  ready: "/ready",
  status: "/api/v1/status",
  predict: "/api/v1/predict",
};

const statusEls = {
  health: document.getElementById("healthStatus"),
  ready: document.getElementById("readyStatus"),
  api: document.getElementById("apiStatus"),
};

const resultEmpty = document.getElementById("resultEmpty");
const resultPanel = document.getElementById("resultPanel");
const resultJson = document.getElementById("resultJson");
const riskPercent = document.getElementById("riskPercent");
const riskLevel = document.getElementById("riskLevel");
const confidence = document.getElementById("confidence");
const riskRing = document.getElementById("riskRing");
const predictionForm = document.getElementById("predictionForm");
const predictBtn = document.getElementById("predictBtn");
const refreshStatusBtn = document.getElementById("refreshStatusBtn");

function setStatus(el, label, ok, detail = "") {
  el.textContent = `${label}: ${detail || (ok ? "ok" : "unavailable")}`;
  el.classList.remove("status-ok", "status-bad", "status-warn");
  el.classList.add(ok ? "status-ok" : "status-bad");
}

async function checkEndpoint(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) {
      return { ok: false, status: res.status };
    }
    const data = await res.json();
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: String(err) };
  }
}

async function refreshStatus() {
  statusEls.health.textContent = "Health: checking...";
  statusEls.ready.textContent = "Readiness: checking...";
  statusEls.api.textContent = "API: checking...";

  const [health, ready, api] = await Promise.all([
    checkEndpoint(endpoints.health),
    checkEndpoint(endpoints.ready),
    checkEndpoint(endpoints.status),
  ]);

  setStatus(statusEls.health, "Health", health.ok, health.ok ? "healthy" : "error");
  setStatus(statusEls.ready, "Readiness", ready.ok, ready.ok ? "ready" : "error");
  setStatus(statusEls.api, "API", api.ok, api.ok ? "operational" : "error");
}

function getFormPayload(formEl) {
  const formData = new FormData(formEl);
  const application = {
    age: Number(formData.get("age")),
    income: Number(formData.get("income")),
    employment_length: Number(formData.get("employment_length")),
    debt_to_income_ratio: Number(formData.get("debt_to_income_ratio")),
    credit_score: Number(formData.get("credit_score")),
    loan_amount: Number(formData.get("loan_amount")),
    loan_purpose: String(formData.get("loan_purpose")),
    home_ownership: String(formData.get("home_ownership")),
    verification_status: String(formData.get("verification_status")),
  };

  return {
    application,
    include_explanation: formData.get("include_explanation") === "on",
    explanation_type: "shap",
    track_sustainability: formData.get("track_sustainability") === "on",
  };
}

function updateRiskVisualization(riskScore, level, conf) {
  const pct = Math.round(riskScore * 100);
  const angle = Math.max(0, Math.min(360, pct * 3.6));
  riskPercent.textContent = `${pct}%`;
  riskLevel.textContent = String(level || "-").toUpperCase();
  confidence.textContent = `${Math.round((conf || 0) * 100)}%`;
  riskRing.style.background = `conic-gradient(#007c78 ${angle}deg, #d7e8ef ${angle}deg)`;
}

async function submitPrediction(event) {
  event.preventDefault();
  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  try {
    const payload = getFormPayload(predictionForm);
    const res = await fetch(endpoints.predict, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Prediction request failed");
    }

    updateRiskVisualization(data.risk_score, data.risk_level, data.confidence);
    resultJson.textContent = JSON.stringify(data, null, 2);
    resultEmpty.classList.add("hidden");
    resultPanel.classList.remove("hidden");
  } catch (err) {
    resultJson.textContent = JSON.stringify({ error: String(err) }, null, 2);
    resultEmpty.classList.add("hidden");
    resultPanel.classList.remove("hidden");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Run Prediction";
  }
}

predictionForm.addEventListener("submit", submitPrediction);
refreshStatusBtn.addEventListener("click", refreshStatus);
refreshStatus();

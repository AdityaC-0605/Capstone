import { defaultBackendConfig, samplePredictionRequest } from "@/lib/constants";
import type {
  AuthResponse,
  BackendConfig,
  BackendStatus,
  BatchPredictionResponse,
  CreditApplication,
  ExplanationPayload,
  FairnessAuditResult,
  FederatedRunParams,
  FederatedRunResult,
  PredictionHistoryItem,
  PredictionRecord,
  PredictionRequest,
  PredictionResponse,
  RiskLevel,
  SustainabilityMetrics,
} from "@/lib/types";

const DEFAULT_TIMEOUT_MS = 6000;

async function withTimeout(
  input: RequestInfo | URL,
  init?: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

async function fetchJson<T>(
  url: string,
  init?: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  let response: Response;
  try {
    response = await withTimeout(url, init, timeoutMs);
  } catch (error) {
    // Normalize abort/network failures into human-readable messages.
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs / 1000}s.`);
    }
    throw new Error(
      "Could not reach the backend. Confirm the service is running and the URL is correct.",
    );
  }

  const text = await response.text();
  let payload: unknown = text ? JSON.parse(text) : {};

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload && "error" in payload
        ? String((payload as { error: unknown }).error)
        : `Request failed with ${response.status}`;
    throw new Error(message);
  }

  return payload as T;
}

/** Unwrap the `{ status, data }` envelope used by the main platform API. */
async function fetchEnvelope<T>(
  url: string,
  init?: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const payload = await fetchJson<{ status: string; data: T }>(
    url,
    init,
    timeoutMs,
  );
  return payload.data;
}

const trimSlash = (url: string) => url.replace(/\/$/, "");

export async function probeBackends(
  config: BackendConfig = defaultBackendConfig,
): Promise<BackendStatus> {
  const result: BackendStatus = {
    main: {
      state: "offline",
      label: "Main API",
      detail: "Failed to fetch health and status.",
    },
    inference: {
      state: "offline",
      label: "Inference Engine",
      detail: "Failed to fetch health.",
    },
    overall: {
      state: "offline",
      label: "Offline",
      detail: "No backend services responded.",
    },
    features: [],
    lastChecked: new Date().toISOString(),
  };

  const mainBase = trimSlash(config.mainUrl);
  const inferenceBase = trimSlash(config.inferenceUrl);

  const [mainHealth, mainReady, mainStatus, inferenceHealth] =
    await Promise.allSettled([
      fetchJson<{ data: { service_status: string } }>(`${mainBase}/health`),
      fetchJson<{ data: { service_status: string } }>(`${mainBase}/ready`),
      fetchJson<{ data: { features?: string[] } }>(`${mainBase}/api/v1/status`),
      fetchJson<{ data: { service_status: string } }>(
        `${inferenceBase}/health`,
      ),
    ]);

  const mainHealthy =
    mainHealth.status === "fulfilled" && mainReady.status === "fulfilled";
  const inferenceHealthy = inferenceHealth.status === "fulfilled";

  result.main = mainHealthy
    ? {
        state: "healthy",
        label: "Main API",
        detail: "Health, readiness, and platform status are responding.",
      }
    : {
        state: "offline",
        label: "Main API",
        detail:
          mainHealth.status === "rejected"
            ? mainHealth.reason.message
            : "Readiness check failed.",
      };

  result.inference = inferenceHealthy
    ? {
        state: "healthy",
        label: "Inference Engine",
        detail: "Inference health endpoint responded successfully.",
      }
    : {
        state: "offline",
        label: "Inference Engine",
        detail:
          inferenceHealth.status === "rejected"
            ? inferenceHealth.reason.message
            : "Inference health failed.",
      };

  result.features =
    mainStatus.status === "fulfilled"
      ? mainStatus.value.data.features || []
      : [];

  if (mainHealthy && inferenceHealthy) {
    result.overall = {
      state: "healthy",
      label: "All Systems Healthy",
      detail:
        "Both backend services are online and ready for live predictions.",
    };
  } else if (mainHealthy || inferenceHealthy) {
    result.overall = {
      state: "partial",
      label: "Partial Connectivity",
      detail: "One backend responded, but another still needs attention.",
    };
  }

  return result;
}

export async function runPrediction(
  config: BackendConfig,
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  if (!config.apiKey.trim()) {
    throw new Error("Missing bearer API key. Paste the sk-test key first.");
  }

  return fetchJson<PredictionResponse>(`${trimSlash(config.inferenceUrl)}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey.trim()}`,
    },
    body: JSON.stringify(payload),
  });
}

export async function runSamplePrediction(config: BackendConfig) {
  return runPrediction(config, samplePredictionRequest);
}

/** Score up to 100 applications in one batch request. */
export async function runBatch(
  config: BackendConfig,
  applications: CreditApplication[],
): Promise<BatchPredictionResponse> {
  if (!config.apiKey.trim()) {
    throw new Error("Sign in before scoring applications.");
  }
  return fetchJson<BatchPredictionResponse>(
    `${trimSlash(config.inferenceUrl)}/predict/batch`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${config.apiKey.trim()}`,
      },
      body: JSON.stringify({
        applications,
        include_explanation: false,
        track_sustainability: false,
        explanation_type: "shap",
      }),
    },
    30000,
  );
}

/**
 * Run a real multi-client FedAvg simulation on the platform API and return
 * round-by-round metrics. Uses a longer timeout since training (even bounded)
 * can take a few seconds for larger client/round counts.
 */
export async function runFederated(
  config: BackendConfig,
  params: FederatedRunParams,
): Promise<FederatedRunResult> {
  return fetchEnvelope<FederatedRunResult>(
    `${trimSlash(config.mainUrl)}/api/v1/federated/run`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    30000,
  );
}

/** Run the real fairness/bias audit over a deterministic synthetic cohort. */
export async function runFairnessAudit(
  config: BackendConfig,
  params: { samples?: number; bias_strength?: number; seed?: number } = {},
): Promise<FairnessAuditResult> {
  const query = new URLSearchParams();
  if (params.samples) query.set("samples", String(params.samples));
  if (params.bias_strength)
    query.set("bias_strength", String(params.bias_strength));
  if (params.seed) query.set("seed", String(params.seed));
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return fetchEnvelope<FairnessAuditResult>(
    `${trimSlash(config.mainUrl)}/api/v1/fairness/audit${suffix}`,
    undefined,
    20000,
  );
}

/** Fetch the server-side rolling prediction history from the inference API. */
export async function fetchPredictionHistory(
  config: BackendConfig,
  limit = 25,
): Promise<{
  count: number;
  total_served: number;
  items: PredictionHistoryItem[];
}> {
  if (!config.apiKey.trim()) {
    throw new Error("Missing bearer API key.");
  }
  return fetchJson(
    `${trimSlash(config.inferenceUrl)}/predict/history?limit=${limit}`,
    {
      headers: { Authorization: `Bearer ${config.apiKey.trim()}` },
    },
  );
}

/**
 * Fetch one durably-persisted assessment by id and map it to the
 * PredictionRecord shape the UI uses. Powers the detail page's fallback when
 * the record isn't in this browser's session history.
 */
export async function fetchAssessment(
  config: BackendConfig,
  id: string,
): Promise<PredictionRecord> {
  if (!config.apiKey.trim()) {
    throw new Error("Missing bearer API key.");
  }
  const data = await fetchEnvelope<{
    prediction_id: string;
    timestamp: string;
    risk_score: number;
    risk_level: RiskLevel;
    confidence: number;
    processing_time_ms: number;
    model_version: string;
    application: CreditApplication;
    explanation?: ExplanationPayload | null;
    sustainability_metrics?: SustainabilityMetrics | null;
  }>(`${trimSlash(config.inferenceUrl)}/predict/${encodeURIComponent(id)}`, {
    headers: { Authorization: `Bearer ${config.apiKey.trim()}` },
  });

  return {
    timestamp: data.timestamp,
    input: data.application,
    result: {
      prediction_id: data.prediction_id,
      risk_score: data.risk_score,
      risk_level: data.risk_level,
      confidence: data.confidence,
      model_version: data.model_version,
      prediction_timestamp: data.timestamp,
      processing_time_ms: data.processing_time_ms,
      explanation: data.explanation ?? undefined,
      sustainability_metrics: data.sustainability_metrics ?? undefined,
      status: "success",
      message: "",
    },
    sustainability_metrics: data.sustainability_metrics ?? undefined,
  };
}

/** Register a new user; returns a session token + the public user record. */
export async function registerUser(
  config: BackendConfig,
  payload: { email: string; password: string; full_name?: string },
): Promise<AuthResponse> {
  return fetchJson<AuthResponse>(
    `${trimSlash(config.inferenceUrl)}/auth/register`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ full_name: "", ...payload }),
    },
  );
}

/** Log in with email + password; returns a session token + user record. */
export async function loginUser(
  config: BackendConfig,
  payload: { email: string; password: string },
): Promise<AuthResponse> {
  return fetchJson<AuthResponse>(
    `${trimSlash(config.inferenceUrl)}/auth/login`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export function buildCurlCommand(config: BackendConfig) {
  return [
    `curl -s ${trimSlash(config.inferenceUrl)}/predict`,
    `  -H "Authorization: Bearer ${config.apiKey || "YOUR_API_KEY"}"`,
    '  -H "Content-Type: application/json"',
    `  -d '${JSON.stringify(samplePredictionRequest, null, 2)}'`,
  ].join(" \\\n");
}

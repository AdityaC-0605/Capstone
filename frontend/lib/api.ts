import { defaultBackendConfig, samplePredictionRequest } from "@/lib/constants";
import type {
  BackendConfig,
  BackendStatus,
  PredictionRequest,
  PredictionResponse,
} from "@/lib/types";

const timeoutMs = 6000;

async function withTimeout(input: RequestInfo | URL, init?: RequestInit) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await withTimeout(url, init);
  const text = await response.text();
  const payload = text ? (JSON.parse(text) as T) : ({} as T);

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload && "error" in payload
        ? String(payload.error)
        : `Request failed with ${response.status}`;
    throw new Error(message);
  }

  return payload;
}

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

  const mainBase = config.mainUrl.replace(/\/$/, "");
  const inferenceBase = config.inferenceUrl.replace(/\/$/, "");

  const [mainHealth, mainReady, mainStatus, inferenceHealth] =
    await Promise.allSettled([
      fetchJson<{ data: { service_status: string } }>(`${mainBase}/health`),
      fetchJson<{ data: { service_status: string } }>(`${mainBase}/ready`),
      fetchJson<{ data: { features?: string[] } }>(`${mainBase}/api/v1/status`),
      fetchJson<{ data: { service_status: string } }>(`${inferenceBase}/health`),
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
    mainStatus.status === "fulfilled" ? mainStatus.value.data.features || [] : [];

  if (mainHealthy && inferenceHealthy) {
    result.overall = {
      state: "healthy",
      label: "All Systems Healthy",
      detail: "Both backend services are online and ready for live predictions.",
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

  return fetchJson<PredictionResponse>(
    `${config.inferenceUrl.replace(/\/$/, "")}/predict`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${config.apiKey.trim()}`,
      },
      body: JSON.stringify(payload),
    },
  );
}

export async function runSamplePrediction(config: BackendConfig) {
  return runPrediction(config, samplePredictionRequest);
}

export function buildCurlCommand(config: BackendConfig) {
  return [
    `curl -s ${config.inferenceUrl.replace(/\/$/, "")}/predict`,
    `  -H "Authorization: Bearer ${config.apiKey || "YOUR_API_KEY"}"`,
    '  -H "Content-Type: application/json"',
    `  -d '${JSON.stringify(samplePredictionRequest, null, 2)}'`,
  ].join(" \\\n");
}

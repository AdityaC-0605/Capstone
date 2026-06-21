"use client";

import { useEffect, useMemo, useState } from "react";
import { X } from "lucide-react";

import { SyntaxCodeBlock } from "@/components/syntax-code-block";
import { buildCurlCommand, probeBackends, runSamplePrediction } from "@/lib/api";
import { defaultBackendConfig, samplePredictionRequest } from "@/lib/constants";
import { aggregateStatusColor } from "@/lib/format";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";
import { toast } from "@/store/use-toast-store";

interface SettingsPanelProps {
  standalone?: boolean;
}

const offlineStatus = (detail: string) => ({
  main: { state: "offline" as const, label: "Main API", detail },
  inference: {
    state: "offline" as const,
    label: "Inference Engine",
    detail: "Failed to fetch. Is the backend running?",
  },
  features: [],
  overall: { state: "offline" as const, label: "Offline", detail },
  lastChecked: new Date().toISOString(),
});

export function SettingsPanel({ standalone = false }: SettingsPanelProps) {
  const backendConfig = usePulseStore((state) => state.backendConfig);
  const backendStatus = usePulseStore((state) => state.backendStatus);
  const setBackendConfig = usePulseStore((state) => state.setBackendConfig);
  const setBackendStatus = usePulseStore((state) => state.setBackendStatus);
  const setSamplePredictionState = usePulseStore(
    (state) => state.setSamplePredictionState,
  );
  const addPredictionRecord = usePulseStore((state) => state.addPredictionRecord);
  const closeSettings = usePulseStore((state) => state.closeSettings);
  const ui = usePulseStore((state) => state.ui);

  const [formState, setFormState] = useState(backendConfig);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    setFormState(backendConfig);
  }, [backendConfig]);

  useEffect(() => {
    if (!standalone && !ui.settingsOpen) return;
    const refresh = async () => {
      try {
        setBackendStatus(await probeBackends(formState));
      } catch {
        setBackendStatus(offlineStatus("Failed to fetch. Is the backend running?"));
      }
    };
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setBackendStatus, standalone, ui.settingsOpen]);

  const curlCommand = useMemo(() => buildCurlCommand(formState), [formState]);

  const handleSave = async () => {
    const normalized = {
      mainUrl: formState.mainUrl.trim() || defaultBackendConfig.mainUrl,
      inferenceUrl:
        formState.inferenceUrl.trim() || defaultBackendConfig.inferenceUrl,
      apiKey: formState.apiKey.trim(),
    };
    setBackendConfig(normalized);
    setBackendStatus(await probeBackends(normalized));
    toast.success("Configuration saved", "Backend endpoints updated.");
    if (!standalone) closeSettings();
  };

  const handleSamplePrediction = async () => {
    setSamplePredictionState({
      samplePredictionLoading: true,
      samplePredictionError: "",
      samplePredictionResponse: "",
    });
    try {
      const response = await runSamplePrediction(formState);
      addPredictionRecord(samplePredictionRequest.application, response);
      setSamplePredictionState({
        samplePredictionLoading: false,
        samplePredictionResponse: JSON.stringify(response, null, 2),
      });
      toast.success("Sample prediction ran", `Risk ${(response.risk_score * 100).toFixed(0)}%.`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Sample prediction failed.";
      setSamplePredictionState({
        samplePredictionLoading: false,
        samplePredictionError: message,
      });
      toast.error("Sample prediction failed", message);
    }
  };

  const copyCurl = async () => {
    await navigator.clipboard.writeText(curlCommand);
    toast.success("Copied", "cURL command on your clipboard.");
  };

  const body = (
    <div className="space-y-7">
      {/* Endpoints */}
      <section className="space-y-4">
        <p className="section-kicker">Endpoints</p>
        <label className="block space-y-1.5">
          <span className="text-sm text-text-secondary">Main API URL</span>
          <input
            className="input-shell font-mono"
            value={formState.mainUrl}
            onChange={(event) =>
              setFormState((current) => ({ ...current, mainUrl: event.target.value }))
            }
          />
        </label>
        <label className="block space-y-1.5">
          <span className="text-sm text-text-secondary">Inference API URL</span>
          <input
            className="input-shell font-mono"
            value={formState.inferenceUrl}
            onChange={(event) =>
              setFormState((current) => ({
                ...current,
                inferenceUrl: event.target.value,
              }))
            }
          />
        </label>
        <label className="block space-y-1.5">
          <span className="text-sm text-text-secondary">Bearer API key</span>
          <div className="input-shell flex items-center gap-2 p-1 pl-3">
            <input
              className="w-full bg-transparent font-mono text-sm focus:outline-none"
              type={showKey ? "text" : "password"}
              value={formState.apiKey}
              placeholder="Paste the sk-test key from the inference log"
              onChange={(event) =>
                setFormState((current) => ({ ...current, apiKey: event.target.value }))
              }
            />
            <button
              type="button"
              onClick={() => setShowKey((current) => !current)}
              className="focus-ring rounded-[3px] border border-border px-2.5 py-1 font-mono text-[11px] uppercase tracking-wider text-text-muted hover:text-text-primary"
            >
              {showKey ? "Hide" : "Show"}
            </button>
          </div>
        </label>
      </section>

      {/* Status */}
      <section className="space-y-3">
        <p className="section-kicker">Connectivity</p>
        <div className="grid gap-3 sm:grid-cols-2">
          {[backendStatus.main, backendStatus.inference].map((item) => (
            <div key={item.label} className="inset p-3">
              <div className="flex items-center gap-2">
                <span className={cn("status-dot", aggregateStatusColor(item.state))} />
                <p className="text-sm font-medium text-text-primary">{item.label}</p>
                <span className="ml-auto font-mono text-[11px] uppercase tracking-wider text-text-muted">
                  {item.state}
                </span>
              </div>
              <p className="mt-2 text-xs leading-5 text-text-secondary">{item.detail}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Actions */}
      <section className="flex flex-wrap gap-3">
        <button type="button" className="button-primary" onClick={handleSave}>
          Save configuration
        </button>
        <button
          type="button"
          className="button-ghost"
          onClick={handleSamplePrediction}
          disabled={ui.samplePredictionLoading}
        >
          {ui.samplePredictionLoading ? "Running…" : "Run sample"}
        </button>
        <button type="button" className="button-ghost" onClick={copyCurl}>
          Copy cURL
        </button>
      </section>

      {/* Reference */}
      <section className="space-y-4">
        <div className="inset p-4">
          <p className="section-kicker">cURL template</p>
          <SyntaxCodeBlock
            code={curlCommand}
            className="mt-3 overflow-x-auto font-mono text-xs leading-relaxed"
          />
        </div>
        <div className="inset p-4">
          <p className="section-kicker">Sample response</p>
          <SyntaxCodeBlock
            code={
              ui.samplePredictionError
                ? `{"error":"${ui.samplePredictionError}"}`
                : ui.samplePredictionResponse || '{\n  "status": "idle"\n}'
            }
            className="mt-3 max-h-[300px] overflow-auto font-mono text-xs leading-relaxed"
          />
        </div>
      </section>
    </div>
  );

  if (standalone) {
    return (
      <div className="page-frame max-w-3xl py-2">
        <p className="section-kicker">Settings</p>
        <h1 className="mt-3 font-display text-3xl font-medium text-text-primary">
          Connect to your backend.
        </h1>
        <p className="mt-2 max-w-lg text-sm text-text-secondary">
          Point PulseLedger at your running services and paste the inference
          bearer key to enable live scoring.
        </p>
        <div className="mt-8 leaf p-6">{body}</div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col border-l border-border-strong bg-bg-surface shadow-[-8px_0_24px_rgba(26,23,20,0.08)]">
      <header className="flex items-center justify-between border-b border-border px-6 py-4">
        <div>
          <p className="section-kicker">Settings</p>
          <h2 className="mt-1 font-display text-xl font-medium text-text-primary">
            Backend connection
          </h2>
        </div>
        <button
          type="button"
          onClick={closeSettings}
          className="focus-ring flex h-8 w-8 items-center justify-center rounded-[3px] border border-border text-text-muted hover:text-text-primary"
          aria-label="Close settings"
        >
          <X className="h-4 w-4" />
        </button>
      </header>
      <div className="flex-1 overflow-y-auto px-6 py-6">{body}</div>
    </div>
  );
}

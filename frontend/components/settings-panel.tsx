"use client";

import { useEffect, useMemo, useState } from "react";
import { X } from "lucide-react";

import { buildCurlCommand, probeBackends, runSamplePrediction } from "@/lib/api";
import {
  defaultBackendConfig,
  samplePredictionRequest,
} from "@/lib/constants";
import { aggregateStatusColor } from "@/lib/format";
import { cn } from "@/lib/utils";
import { SyntaxCodeBlock } from "@/components/syntax-code-block";
import { usePulseStore } from "@/store/use-pulse-store";

interface SettingsPanelProps {
  standalone?: boolean;
}

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
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setFormState(backendConfig);
  }, [backendConfig]);

  useEffect(() => {
    if (!standalone && !ui.settingsOpen) return;
    const refresh = async () => {
      try {
        const status = await probeBackends(formState);
        setBackendStatus(status);
      } catch {
        setBackendStatus({
          main: {
            state: "offline",
            label: "Main API",
            detail: "Failed to fetch. Is the backend running?",
          },
          inference: {
            state: "offline",
            label: "Inference Engine",
            detail: "Failed to fetch. Is the backend running?",
          },
          features: [],
          overall: {
            state: "offline",
            label: "Offline",
            detail: "Failed to fetch. Is the backend running?",
          },
          lastChecked: new Date().toISOString(),
        });
      }
    };
    void refresh();
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
    const status = await probeBackends(normalized);
    setBackendStatus(status);
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2000);
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
    } catch (error) {
      setSamplePredictionState({
        samplePredictionLoading: false,
        samplePredictionError:
          error instanceof Error
            ? error.message
            : "Sample prediction failed unexpectedly.",
      });
    }
  };

  const copyCurl = async () => {
    await navigator.clipboard.writeText(curlCommand);
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div
      className={cn(
        standalone
          ? "min-h-[calc(100vh-6rem)] rounded-lg p-4"
          : "fixed inset-0 z-[60] flex items-center justify-center p-4",
      )}
    >
      <div className="glass-panel w-full max-w-4xl rounded-lg p-6 shadow-lg md:p-8">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div>
            <p className="section-kicker">Backend Configuration</p>
            <h2 className="mt-2 font-display text-2xl font-semibold text-text-primary">
              Connect PulseLedger to your backend services.
            </h2>
          </div>
          {!standalone ? (
            <button
              type="button"
              className="flex h-8 w-8 items-center justify-center rounded-md border border-border bg-bg-surface text-text-muted hover:text-text-primary hover:bg-bg-elevated transition"
              onClick={closeSettings}
            >
              <X className="h-4 w-4" />
            </button>
          ) : null}
        </div>

        <div className="mt-8 grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="space-y-5">
            <label className="block space-y-1.5">
              <span className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                Main API URL
              </span>
              <input
                className="input-shell font-mono"
                value={formState.mainUrl}
                onChange={(event) =>
                  setFormState((current) => ({
                    ...current,
                    mainUrl: event.target.value,
                  }))
                }
              />
            </label>
            <label className="block space-y-1.5">
              <span className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                Inference API URL
              </span>
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
              <span className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                Bearer API Key
              </span>
              <div className="input-shell flex items-center gap-2 p-1.5 pl-3">
                <input
                  className="w-full bg-transparent font-mono text-sm focus:outline-none"
                  type={showKey ? "text" : "password"}
                  value={formState.apiKey}
                  placeholder="Paste the sk-test key from the inference log"
                  onChange={(event) =>
                    setFormState((current) => ({
                      ...current,
                      apiKey: event.target.value,
                    }))
                  }
                />
                <button
                  type="button"
                  onClick={() => setShowKey((current) => !current)}
                  className="rounded-md border border-border bg-bg-surface px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-text-muted hover:text-text-primary transition"
                >
                  {showKey ? "Hide" : "Show"}
                </button>
              </div>
            </label>

            <div className="grid gap-3 pt-2 md:grid-cols-2">
              {[backendStatus.main, backendStatus.inference].map((item) => (
                <div
                  key={item.label}
                  className="rounded-md border border-border bg-bg-primary p-3"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "status-dot",
                        aggregateStatusColor(item.state),
                      )}
                    />
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                        {item.label}
                      </p>
                      <p className="mt-0.5 text-xs font-semibold text-text-primary">{item.state}</p>
                    </div>
                  </div>
                  <p className="mt-2 text-xs leading-5 text-text-secondary">{item.detail}</p>
                </div>
              ))}
            </div>

            <div className="flex flex-wrap gap-3 pt-4">
              <button type="button" className="button-ghost" onClick={copyCurl}>
                Copy cURL
              </button>
              <button
                type="button"
                className="button-ghost border-success/30 text-success hover:bg-success/10 hover:border-success/50"
                onClick={handleSamplePrediction}
                disabled={ui.samplePredictionLoading}
              >
                {ui.samplePredictionLoading ? "Running..." : "Sample Prediction"}
              </button>
              <button type="button" className="button-primary" onClick={handleSave}>
                Save Configuration
              </button>
            </div>

            {saved ? (
              <div className="rounded-md border border-success/30 bg-success/10 px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-success animate-enter">
                Saved successfully.
              </div>
            ) : null}
          </div>

          <div className="space-y-4">
            <div className="rounded-md border border-border bg-bg-primary p-4">
              <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                cURL Template
              </p>
              <SyntaxCodeBlock
                code={curlCommand}
                className="mt-3 overflow-x-auto font-mono text-xs leading-relaxed"
              />
            </div>
            <div className="rounded-md border border-border bg-bg-primary p-4">
              <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                Sample Response
              </p>
              <SyntaxCodeBlock
                code={
                  ui.samplePredictionError
                    ? `{"error":"${ui.samplePredictionError}"}`
                    : ui.samplePredictionResponse || '{\n  "status": "idle"\n}'
                }
                className="mt-3 max-h-[340px] overflow-auto font-mono text-xs leading-relaxed"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

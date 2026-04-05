"use client";

import { useMemo, useState } from "react";

import { runPrediction } from "@/lib/api";
import {
  defaultApplication,
  homeOwnershipOptions,
  loanPurposeOptions,
  riskBands,
  verificationOptions,
} from "@/lib/constants";
import {
  factorDirection,
  formatCarbon,
  formatCurrency,
  formatEnergy,
  formatFeatureLabel,
  formatPercent,
  formatRiskLabel,
  formatSeconds,
  narrativeForGroup,
  recommendationCopy,
  riskColorClasses,
  riskMarkerLeft,
  sentimentColorClasses,
} from "@/lib/format";
import type {
  CreditApplication,
  PredictionResponse,
  Recommendation,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { FactorFlipCard } from "@/components/factor-flip-card";
import { RangeField } from "@/components/range-field";
import { RiskGauge } from "@/components/risk-gauge";
import { SegmentedControl } from "@/components/segmented-control";
import { StateCard } from "@/components/state-card";
import { TerminalPanel } from "@/components/terminal-panel";
import { ToggleSwitch } from "@/components/toggle-switch";
import { usePulseStore } from "@/store/use-pulse-store";

const tabs = [
  "Explainability",
  "Counterfactuals",
  "Risk Groups",
  "Methodology",
] as const;

type Tab = (typeof tabs)[number];

export default function StudioPage() {
  const backendConfig = usePulseStore((state) => state.backendConfig);
  const predictionHistory = usePulseStore((state) => state.predictionHistory);
  const addPredictionRecord = usePulseStore((state) => state.addPredictionRecord);
  const openSettings = usePulseStore((state) => state.openSettings);

  const [application, setApplication] = useState<CreditApplication>(defaultApplication);
  const [includeExplanation, setIncludeExplanation] = useState(true);
  const [trackSustainability, setTrackSustainability] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("Explainability");
  const [footprintOpen, setFootprintOpen] = useState(true);

  const sustainabilityTrend = useMemo(
    () =>
      predictionHistory
        .slice(0, 5)
        .reverse()
        .map((entry, index) => ({
          run: index + 1,
          carbon: entry.sustainability_metrics?.carbon_emissions || 0,
        })),
    [predictionHistory],
  );

  const recommendationMap = useMemo(() => {
    const map = new Map<string, Recommendation>();
    (result?.explanation?.recommendations || []).forEach((recommendation) => {
      if (recommendation.feature) map.set(recommendation.feature, recommendation);
    });
    return map;
  }, [result]);

  const handleSubmit = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await runPrediction(backendConfig, {
        application,
        include_explanation: includeExplanation,
        track_sustainability: trackSustainability,
        explanation_type: "shap",
      });
      setResult(response);
      setFootprintOpen(true);
      addPredictionRecord(application, response);
    } catch (submissionError) {
      setError(
        submissionError instanceof Error
          ? submissionError.message
          : "Prediction failed unexpectedly.",
      );
    } finally {
      setLoading(false);
    }
  };

  const confidence = result?.explanation?.confidence;
  const confidenceWidth = `${Math.max(
    10,
    Math.round((confidence?.score || result?.confidence || 0) * 100),
  )}%`;
  const confidenceColor =
    confidence?.level === "high"
      ? "bg-success"
      : confidence?.level === "medium"
        ? "bg-warning"
        : "bg-destructive";
  const carbonBars =
    sustainabilityTrend.length > 0
      ? sustainabilityTrend
      : Array.from({ length: 5 }, (_, index) => ({
          run: index + 1,
          carbon: 0,
        }));
  const maxCarbon = Math.max(...carbonBars.map((item) => item.carbon), 0.0001);

  return (
    <div className="page-frame grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)_340px]">
      {/* ─── Left: Application Profile ─── */}
      <aside className="animate-enter-1 rounded-lg border border-border bg-bg-surface p-5 shadow-sm">
        <div className="mb-5">
          <p className="section-kicker">Application Profile</p>
          <h1 className="mt-2 font-display text-xl font-bold text-text-primary">
            Configure profile vectors.
          </h1>
        </div>

        <div className="space-y-4">
          <RangeField
            label="Age"
            min={18}
            max={80}
            value={application.age}
            valueLabel={`${application.age}`}
            onChange={(value) => setApplication((current) => ({ ...current, age: value }))}
          />
          <RangeField
            label="Annual Income"
            min={10000}
            max={250000}
            step={1000}
            value={application.income}
            valueLabel={formatCurrency(application.income)}
            onChange={(value) =>
              setApplication((current) => ({ ...current, income: value }))
            }
          />
          <RangeField
            label="Employment Length"
            min={0}
            max={30}
            value={application.employment_length}
            valueLabel={`${application.employment_length} years`}
            onChange={(value) =>
              setApplication((current) => ({
                ...current,
                employment_length: value,
              }))
            }
          />
          <RangeField
            label="Debt to Income Ratio"
            min={0}
            max={1}
            step={0.01}
            value={application.debt_to_income_ratio}
            valueLabel={formatPercent(application.debt_to_income_ratio)}
            warning={application.debt_to_income_ratio > 0.4}
            onChange={(value) =>
              setApplication((current) => ({
                ...current,
                debt_to_income_ratio: value,
              }))
            }
          />
          <RangeField
            label="Credit Score"
            min={300}
            max={850}
            value={application.credit_score}
            valueLabel={`${application.credit_score}`}
            scoreTrack
            onChange={(value) =>
              setApplication((current) => ({ ...current, credit_score: value }))
            }
          />
          <RangeField
            label="Loan Amount"
            min={1000}
            max={50000}
            step={500}
            value={application.loan_amount}
            valueLabel={formatCurrency(application.loan_amount)}
            onChange={(value) =>
              setApplication((current) => ({ ...current, loan_amount: value }))
            }
          />

          <div className="space-y-2">
            <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
              Loan Purpose
            </p>
            <div className="flex flex-wrap gap-1.5">
              {loanPurposeOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() =>
                    setApplication((current) => ({
                      ...current,
                      loan_purpose: option.value,
                    }))
                  }
                  className={cn(
                    "focus-ring rounded-md border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider transition-all duration-120 active:scale-[0.98]",
                    application.loan_purpose === option.value
                      ? "border-accent bg-accent text-white"
                      : "border-border bg-bg-surface text-text-muted hover:text-text-primary hover:bg-bg-elevated",
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <SegmentedControl
            label="Home Ownership"
            options={homeOwnershipOptions}
            value={application.home_ownership}
            onChange={(value) =>
              setApplication((current) => ({ ...current, home_ownership: value as CreditApplication["home_ownership"] }))
            }
          />
          <SegmentedControl
            label="Verification Status"
            options={verificationOptions}
            value={application.verification_status}
            onChange={(value) =>
              setApplication((current) => ({
                ...current,
                verification_status: value as CreditApplication["verification_status"],
              }))
            }
            compact
          />
          <ToggleSwitch
            label="Include Explanation"
            checked={includeExplanation}
            onChange={setIncludeExplanation}
          />
          <ToggleSwitch
            label="Track Sustainability"
            checked={trackSustainability}
            onChange={setTrackSustainability}
          />
          <div className="grid gap-2 pt-2 md:grid-cols-2 lg:grid-cols-1">
            <button
              type="button"
              className="button-primary w-full"
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="button-spinner mr-2" />
                  Running...
                </>
              ) : (
                "Run Inference"
              )}
            </button>
            <button
              type="button"
              className="button-ghost w-full"
              onClick={() => {
                setApplication(defaultApplication);
                setResult(null);
                setError("");
              }}
            >
              Reset
            </button>
          </div>
        </div>
      </aside>

      {/* ─── Center: Results ─── */}
      <section className="animate-enter-2 space-y-6">
        <RiskGauge
          score={result?.risk_score ?? 0}
          riskLevel={result?.risk_level ?? "idle"}
          idle={!result}
        />

        <div className="rounded-lg border border-border bg-bg-surface p-6 shadow-sm">
          {result ? (
            <>
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div>
                  <p className="section-kicker">Decision Output</p>
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <span
                      className={cn(
                        "rounded-md border px-3 py-1.5 text-xs font-bold uppercase tracking-wider",
                        riskColorClasses(result.risk_level),
                      )}
                    >
                      {formatRiskLabel(result.risk_level)}
                    </span>
                    <span className="rounded-md border border-border bg-bg-elevated px-3 py-1.5 font-mono text-xs font-semibold text-text-secondary">
                      {result.processing_time_ms.toFixed(0)} ms
                    </span>
                  </div>
                </div>
                <div className="font-mono text-[11px] text-text-muted">
                  <p>ID: {result.prediction_id}</p>
                  <p className="mt-0.5">Model: {result.model_version}</p>
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="mt-6">
                <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-text-muted">
                  <span>Confidence</span>
                  <span className="font-mono">
                    {result.explanation?.confidence?.level || "model"} ·{" "}
                    {Math.round((result.explanation?.confidence?.score || result.confidence) * 100)}%
                  </span>
                </div>
                <div className="mt-2 h-1.5 rounded-full bg-border">
                  <div
                    className={cn("h-full rounded-full transition-all duration-700", confidenceColor)}
                    style={{ width: confidenceWidth }}
                  />
                </div>
                <p className="mt-2 text-xs leading-5 text-text-secondary">
                  {confidence?.reason ||
                    "Confidence is derived from the model output and explanation service."}
                </p>
              </div>

              {/* Threshold Context */}
              <div className="mt-6">
                <div className="mb-2 flex items-center justify-between text-xs font-bold uppercase tracking-wider text-text-muted">
                  <span>Threshold</span>
                  <span className="max-w-[200px] truncate text-right">
                    {result.explanation?.risk_threshold_context || "Risk band bounds"}
                  </span>
                </div>
                <div className="relative h-1.5 rounded-full bg-border">
                  {riskBands.map((band, index) => (
                    <div
                      key={band.level}
                      className={cn(
                        "absolute top-0 h-full",
                        index === 0 && "rounded-l-full bg-success/30",
                        index === 1 && "bg-warning/30",
                        index === 2 && "bg-destructive/30",
                        index === 3 && "rounded-r-full bg-destructive/50",
                      )}
                      style={{
                        left: `${(index / riskBands.length) * 100}%`,
                        width: `${100 / riskBands.length}%`,
                      }}
                    />
                  ))}
                  <span
                    className="absolute top-1/2 h-3.5 w-3.5 -translate-y-1/2 rounded-full border-2 border-text-primary bg-bg-surface"
                    style={{ left: riskMarkerLeft(result.risk_score) }}
                  />
                </div>
              </div>
            </>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="section-kicker">Decision Output</p>
                  <h2 className="mt-2 font-display text-lg font-bold text-text-primary">
                    Awaiting inference cycle.
                  </h2>
                </div>
                <span className="rounded-md border border-accent/30 bg-accent/10 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-accent drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]">
                  Standby
                </span>
              </div>
              <p className="text-sm leading-relaxed text-text-secondary">
                Configure the profile vectors and execute the model via the panel to generate a live SHAP output.
              </p>
              <div className="mt-4">
                <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-text-muted">
                  <span>Confidence</span>
                  <span>Pending</span>
                </div>
                <div className="mt-2 h-1.5 rounded-full bg-border">
                  <div className="h-full w-[18%] rounded-full bg-accent/40 animate-pulse" />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ─── Sustainability Footprint ─── */}
        <div className="rounded-lg border border-border bg-bg-surface p-6 shadow-sm">
          <button
            type="button"
            onClick={() => setFootprintOpen((current) => !current)}
            className="focus-ring flex w-full items-center justify-between text-left rounded-md outline-none"
          >
            <div>
              <p className="section-kicker !text-success">Operational Footprint</p>
              <h2 className="mt-1.5 font-display text-lg font-bold text-text-primary">
                {result
                  ? "Sustainability telemetry captured."
                  : "Green telemetry ready."}
              </h2>
            </div>
            <span className="rounded-md border border-success/30 bg-success/10 px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-success transition hover:bg-success/20">
              {footprintOpen ? "Collapse" : "Expand"}
            </span>
          </button>

          {footprintOpen ? (
            <div className="mt-6 space-y-4">
              <div className="metric-grid gap-3">
                {[
                  { label: "Energy", value: formatEnergy(result?.sustainability_metrics?.energy_kwh || 0) },
                  { label: "Carbon", value: formatCarbon(result?.sustainability_metrics?.carbon_emissions || 0) },
                  { label: "Duration", value: formatSeconds(result?.sustainability_metrics?.duration_seconds || 0) },
                ].map((metric) => (
                  <div
                    key={metric.label}
                    className="rounded-md border border-success/20 bg-success/10 p-4"
                  >
                    <p className="text-[10px] font-bold uppercase tracking-wider text-success/80">
                      {metric.label}
                    </p>
                    <p className="mt-3 font-mono text-lg font-bold text-success drop-shadow-[0_0_8px_rgba(34,197,94,0.4)]">
                      {metric.value}
                    </p>
                  </div>
                ))}
              </div>

              <div className="group relative rounded-md border border-border bg-bg-elevated p-4">
                <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                  Session Carbon Trend
                </p>
                <div className="mt-4 flex h-20 items-end gap-2">
                  {carbonBars.map((item) => (
                    <div key={item.run} className="flex flex-1 flex-col items-center gap-1.5">
                      <div className="w-full rounded-t-md bg-border" style={{ height: "80px" }}>
                        <div
                          className={cn(
                            "w-full rounded-t-md transition-all duration-700",
                            item.carbon > 0 ? "bg-success" : "bg-bg-surface",
                          )}
                          style={{
                            height: `${Math.max(
                              10,
                              item.carbon > 0 ? (item.carbon / maxCarbon) * 100 : 15,
                            )}%`,
                            marginTop: "auto",
                          }}
                        />
                      </div>
                      <span className="font-mono text-[9px] font-bold text-text-muted">
                        R{item.run}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : null}
        </div>

        {loading ? (
          <StateCard
            title="Running"
            message="Evaluating vectors and computing deep SHAP allocations..."
            tone="loading"
          />
        ) : null}
        {error ? (
          <StateCard
            title="Prediction Error"
            message={error}
            actionLabel="Open Settings"
            onAction={openSettings}
            tone="error"
          />
        ) : null}
      </section>

      {/* ─── Right: Explainability Tabs ─── */}
      <aside className="animate-enter-3 rounded-lg border border-border bg-bg-surface p-5 shadow-sm">
        <div className="relative mb-5 flex gap-1 overflow-x-auto rounded-md border border-border bg-bg-elevated p-1 scrollbar-hide">
          {tabs.map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={cn(
                "focus-ring relative rounded-md px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider transition-all whitespace-nowrap",
                activeTab === tab
                  ? "bg-bg-surface shadow-[0_1px_2px_rgba(0,0,0,0.5)] text-text-primary"
                  : "text-text-muted hover:text-text-primary",
              )}
            >
              {tab}
            </button>
          ))}
        </div>

        {!result && activeTab === "Explainability" ? (
          <div className="space-y-4">
            <TerminalPanel
              label="Analyst Narrative"
              text="Awaiting live inference output. Executing analysis will stream the narrative directly into this pane."
            />
            {Array.from({ length: 2 }).map((_, index) => (
              <div
                key={index}
                className="rounded-md border border-border bg-bg-elevated p-4"
              >
                <div className="h-2.5 w-24 rounded-full bg-border" />
                <div className="mt-4 h-1.5 rounded-full bg-border">
                  <div
                    className="h-full rounded-full bg-bg-surface"
                    style={{ width: `${42 + index * 14}%` }}
                  />
                </div>
                <p className="mt-4 text-xs leading-relaxed text-text-muted">
                  Awaiting explanation payload.
                </p>
              </div>
            ))}
          </div>
        ) : null}
        {!result && activeTab !== "Explainability" ? (
          <StateCard
            title={activeTab}
            message="Run inference to generate dynamic SHAP components."
          />
        ) : null}

        {result && activeTab === "Explainability" ? (
          <div className="space-y-4">
            <TerminalPanel
              label="Analyst Narrative"
              text={
                result.explanation?.summary ||
                "Narrative summary not generated by methodology."
              }
            />
            <div className="flex items-center justify-between">
              <p className="section-kicker">Top Factors</p>
              <span
                className={cn(
                  "rounded-md border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider",
                  result.explanation?.confidence?.level === "high"
                    ? "border-success/30 bg-success/10 text-success"
                    : result.explanation?.confidence?.level === "medium"
                      ? "border-warning/30 bg-warning/10 text-warning"
                      : "border-destructive/30 bg-destructive/10 text-destructive",
                )}
              >
                {result.explanation?.confidence?.level || "model"}
              </span>
            </div>
            <div className="grid gap-3">
              {(result.explanation?.top_factors || []).map((factor) => (
                <FactorFlipCard
                  key={factor.feature}
                  factor={factor}
                  recommendation={recommendationMap.get(factor.feature)}
                />
              ))}
            </div>
          </div>
        ) : null}

        {result && activeTab === "Counterfactuals" ? (
          <div className="space-y-4">
            <div>
              <p className="section-kicker">What To Change</p>
              <h2 className="mt-1 font-display text-lg font-bold text-text-primary">
                Calculated shifts to clear thresholds.
              </h2>
            </div>
            {result.explanation?.counterfactual?.needed ? (
              Object.entries(result.explanation.counterfactual.changes || {}).map(
                ([feature, change]) => (
                  <div
                    key={feature}
                    className="rounded-md border border-border bg-bg-elevated p-4"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                        {formatFeatureLabel(feature)}
                      </p>
                      <p className="text-xs text-text-secondary">{change.action}</p>
                    </div>
                    <div className="mt-3 flex items-center gap-2">
                      <span className="rounded-md border border-border bg-bg-surface px-2.5 py-1.5 font-mono text-xs text-text-secondary">
                        {String(change.current_value)}
                      </span>
                      <span className="text-accent text-lg">→</span>
                      <span className="rounded-md border border-accent/30 bg-accent/10 px-2.5 py-1.5 font-mono text-xs font-bold text-accent">
                        {String(change.suggested_target)}
                      </span>
                    </div>
                  </div>
                ),
              )
            ) : (
              <StateCard
                title="Counterfactuals"
                message={
                  result.explanation?.counterfactual?.message ||
                  "No modifications required; profile meets threshold."
                }
              />
            )}
          </div>
        ) : null}

        {result && activeTab === "Risk Groups" ? (
          <div className="space-y-4">
            <div>
              <p className="section-kicker">Risk Groups</p>
              <h2 className="mt-1 font-display text-lg font-bold text-text-primary">
                Aggregated vector classifications.
              </h2>
            </div>
            {Object.entries(result.explanation?.risk_groups || {}).map(
              ([key, group]) => (
                <div
                  key={key}
                  className={cn(
                    "rounded-md border p-4 shadow-sm",
                    sentimentColorClasses(group.impact), // These were refactored in format.ts earlier? Wait, format.ts output needs to be checked. Assuming they map to semantic classes.
                  )}
                >
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-xs font-bold uppercase tracking-wider">
                      {group.label || formatFeatureLabel(key)}
                    </p>
                    <span className="font-mono text-xs">
                      {(group.total_contribution || 0).toFixed(3)}
                    </span>
                  </div>
                  <div className="mt-3 h-1.5 rounded-full bg-border">
                    <div
                      className={cn(
                        "h-full rounded-full transition-all duration-500",
                        group.impact === "risk_decrease"
                          ? "bg-success drop-shadow-[0_0_8px_rgba(34,197,94,0.4)]"
                          : group.impact === "risk_increase"
                            ? "bg-destructive drop-shadow-[0_0_8px_rgba(239,68,68,0.4)]"
                            : "bg-warning drop-shadow-[0_0_8px_rgba(245,158,11,0.4)]",
                      )}
                      style={{
                        width: `${Math.max(
                          10,
                          Math.min(100, Math.abs(group.total_contribution || 0) * 100),
                        )}%`,
                      }}
                    />
                  </div>
                  <p className="mt-3 text-sm leading-relaxed text-text-secondary">
                    {narrativeForGroup(key, group)}
                  </p>
                </div>
              ),
            )}
          </div>
        ) : null}

        {result && activeTab === "Methodology" ? (
          <div className="space-y-4">
            <div>
              <p className="section-kicker">Methodology</p>
              <h2 className="mt-1 font-display text-lg font-bold text-text-primary">
                Reference values for explainer.
              </h2>
            </div>
            <div className="inline-flex rounded-md border border-accent/20 bg-accent/10 px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-accent drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]">
              {result.explanation?.methodology?.method || "SHAP"}
            </div>
            <div className="rounded-md border border-border bg-bg-elevated p-4">
              <div className="grid grid-cols-2 gap-2 border-b border-border pb-3 text-[10px] font-bold uppercase tracking-wider text-text-muted">
                <span>Vector Layer</span>
                <span>Baseline</span>
              </div>
              <div className="mt-3 grid gap-2">
                {Object.entries(
                  result.explanation?.methodology?.baseline?.baseline_values || {},
                ).map(([feature, value]) => (
                  <div key={feature} className="grid grid-cols-2 gap-2 font-mono text-xs text-text-secondary">
                    <span>{formatFeatureLabel(feature)}</span>
                    <span className="font-bold text-text-primary">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
            <p className="text-xs leading-relaxed text-text-muted">
              Baseline profiles form the control state. SHAP attributions represent divergence from this exact profile.
            </p>
          </div>
        ) : null}
      </aside>
    </div>
  );
}

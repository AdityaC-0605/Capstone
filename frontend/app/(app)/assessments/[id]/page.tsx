"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft } from "lucide-react";

import { FactorFlipCard } from "@/components/factor-flip-card";
import { RiskGauge } from "@/components/risk-gauge";
import { ShapBarChart } from "@/components/shap-bar-chart";
import { StateCard } from "@/components/state-card";
import { TerminalPanel } from "@/components/terminal-panel";
import { fetchAssessment } from "@/lib/api";
import {
  formatCarbon,
  formatCurrency,
  formatEnergy,
  formatFeatureLabel,
  formatPercent,
  formatRiskLabel,
  formatSeconds,
  narrativeForGroup,
  relativeTime,
  riskColorClasses,
  sentimentColorClasses,
} from "@/lib/format";
import type { PredictionRecord, Recommendation } from "@/lib/types";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";

const tabs = ["Factors", "Counterfactuals", "Groups", "Method"] as const;
type Tab = (typeof tabs)[number];

export default function AssessmentDetailPage() {
  const params = useParams();
  const id = Array.isArray(params.id) ? params.id[0] : params.id;
  const history = usePulseStore((state) => state.predictionHistory);
  const backendConfig = usePulseStore((state) => state.backendConfig);
  const [activeTab, setActiveTab] = useState<Tab>("Factors");
  const [remote, setRemote] = useState<PredictionRecord | null>(null);
  const [lookup, setLookup] = useState<"idle" | "loading" | "missing">("idle");

  const localEntry = useMemo(
    () => history.find((item) => item.result.prediction_id === id),
    [history, id],
  );

  // Fall back to durable server storage when the assessment isn't in this
  // browser's session history (e.g. opened on another device or after a wipe).
  useEffect(() => {
    if (localEntry || !id) return;
    if (!backendConfig.apiKey.trim()) {
      setLookup("missing");
      return;
    }
    let active = true;
    setLookup("loading");
    fetchAssessment(backendConfig, id)
      .then((record) => {
        if (!active) return;
        setRemote(record);
        setLookup("idle");
      })
      .catch(() => {
        if (active) setLookup("missing");
      });
    return () => {
      active = false;
    };
  }, [localEntry, id, backendConfig]);

  const entry = localEntry ?? remote;

  const recommendationMap = useMemo(() => {
    const map = new Map<string, Recommendation>();
    (entry?.result.explanation?.recommendations || []).forEach((rec) => {
      if (rec.feature) map.set(rec.feature, rec);
    });
    return map;
  }, [entry]);

  if (!entry) {
    return (
      <div className="space-y-6">
        <Link
          href="/assessments"
          className="focus-ring inline-flex items-center gap-1.5 rounded-[3px] text-sm text-text-muted hover:text-text-primary"
        >
          <ArrowLeft className="h-4 w-4" />
          Assessments
        </Link>
        {lookup === "loading" ? (
          <StateCard
            tone="loading"
            title="Loading assessment"
            message="Fetching this assessment from durable storage…"
          />
        ) : (
          <StateCard
            title="Assessment not found"
            message="This assessment isn't in your history or on the server. It may have been cleared, or you need to set your API key in Settings."
            actionLabel="Run a new assessment"
            onAction={() => {
              window.location.href = "/assessments/new";
            }}
          />
        )}
      </div>
    );
  }

  const { result, input } = entry;
  const explanation = result.explanation;
  const confidence = explanation?.confidence;
  const confidenceScore = confidence?.score ?? result.confidence ?? 0;
  const footprint = result.sustainability_metrics;

  const applicantRows: Array<[string, string]> = [
    ["Age", `${input.age}`],
    ["Annual income", formatCurrency(input.income)],
    ["Employment", `${input.employment_length} yr`],
    ["Debt-to-income", formatPercent(input.debt_to_income_ratio)],
    ["Credit score", `${input.credit_score}`],
    ["Loan amount", formatCurrency(input.loan_amount)],
    ["Purpose", formatFeatureLabel(input.loan_purpose)],
    ["Home", formatFeatureLabel(input.home_ownership)],
    ["Verification", formatFeatureLabel(input.verification_status)],
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Link
          href="/assessments"
          className="focus-ring inline-flex items-center gap-1.5 rounded-[3px] text-sm text-text-muted hover:text-text-primary"
        >
          <ArrowLeft className="h-4 w-4" />
          Assessments
        </Link>
        <div className="flex items-center gap-3 font-mono text-xs text-text-muted">
          <span>{result.prediction_id}</span>
          <span className="text-border-strong">·</span>
          <span>{relativeTime(entry.timestamp)}</span>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)] lg:items-start">
        {/* Summary rail */}
        <div className="space-y-6">
          <RiskGauge score={result.risk_score} riskLevel={result.risk_level} />

          <div className="leaf p-6">
            <div className="flex items-center justify-between">
              <span
                className={cn(
                  "rounded-[3px] border px-2.5 py-1 font-mono text-xs uppercase tracking-wider",
                  riskColorClasses(result.risk_level),
                )}
              >
                {formatRiskLabel(result.risk_level)}
              </span>
              <span className="font-mono text-xs text-text-muted">
                {result.processing_time_ms.toFixed(0)} ms · {result.model_version}
              </span>
            </div>
            <div className="mt-5">
              <div className="flex items-baseline justify-between">
                <span className="section-kicker">Confidence</span>
                <span className="font-mono text-sm text-text-primary tabular">
                  {Math.round(confidenceScore * 100)}%
                  <span className="ml-1.5 text-text-muted">{confidence?.level || "model"}</span>
                </span>
              </div>
              <div className="mt-2 h-1 bg-bg-elevated">
                <div
                  className="h-full bg-accent"
                  style={{ width: `${Math.max(6, Math.round(confidenceScore * 100))}%` }}
                />
              </div>
              <p className="mt-2.5 text-[13px] leading-relaxed text-text-secondary">
                {confidence?.reason ||
                  explanation?.risk_threshold_context ||
                  "Confidence is derived from the model output and explanation service."}
              </p>
            </div>
          </div>

          <div className="leaf p-6">
            <p className="section-kicker">Applicant</p>
            <dl className="mt-3 divide-y divide-border border-t border-border">
              {applicantRows.map(([label, value]) => (
                <div key={label} className="flex items-center justify-between py-2">
                  <dt className="text-sm text-text-muted">{label}</dt>
                  <dd className="font-mono text-[13px] text-text-primary tabular">{value}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="leaf p-6">
            <p className="section-kicker">Operational Footprint</p>
            <dl className="mt-4 divide-y divide-border border-y border-border">
              {[
                ["Energy", formatEnergy(footprint?.energy_kwh || 0)],
                ["Carbon", formatCarbon(footprint?.carbon_emissions || 0)],
                ["Duration", formatSeconds(footprint?.duration_seconds || 0)],
              ].map(([label, value]) => (
                <div key={label} className="flex items-center justify-between py-2.5">
                  <dt className="text-sm text-text-muted">{label}</dt>
                  <dd className="font-mono text-sm text-text-primary tabular">{value}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>

        {/* Explanation */}
        <div className="leaf p-6">
          <div className="scrollbar-hide flex gap-1 overflow-x-auto border-b border-border pb-3">
            {tabs.map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => setActiveTab(tab)}
                className={cn(
                  "focus-ring shrink-0 rounded-[3px] px-3 py-1.5 text-sm font-medium transition-colors",
                  activeTab === tab
                    ? "bg-bg-elevated text-text-primary"
                    : "text-text-muted hover:text-text-primary",
                )}
              >
                {tab}
              </button>
            ))}
          </div>

          {!explanation ? (
            <div className="mt-5">
              <StateCard
                title="No explanation"
                message="This assessment was scored without an explanation. Re-run with SHAP enabled to see attributions."
              />
            </div>
          ) : null}

          {explanation && activeTab === "Factors" ? (
            <div className="mt-5 space-y-5">
              <TerminalPanel
                label="Analyst Narrative"
                text={explanation.summary || "No narrative summary was generated."}
              />
              {explanation.feature_importance &&
              Object.keys(explanation.feature_importance).length > 0 ? (
                <ShapBarChart importance={explanation.feature_importance} />
              ) : null}
              <p className="section-kicker">Top Factors</p>
              <div className="grid gap-3 sm:grid-cols-2">
                {(explanation.top_factors || []).map((factor) => (
                  <FactorFlipCard
                    key={factor.feature}
                    factor={factor}
                    recommendation={recommendationMap.get(factor.feature)}
                  />
                ))}
              </div>
            </div>
          ) : null}

          {explanation && activeTab === "Counterfactuals" ? (
            <div className="mt-5 space-y-3">
              <p className="text-sm text-text-secondary">
                The smallest changes that would move this application to a lower band.
              </p>
              {explanation.counterfactual?.needed ? (
                <div className="grid gap-3 sm:grid-cols-2">
                  {Object.entries(explanation.counterfactual.changes || {}).map(
                    ([feature, change]) => (
                      <div key={feature} className="inset p-4">
                        <div className="flex items-center justify-between gap-3">
                          <p className="text-sm font-medium text-text-primary">
                            {formatFeatureLabel(feature)}
                          </p>
                          <p className="text-xs text-text-muted">{change.action}</p>
                        </div>
                        <div className="mt-3 flex items-center gap-2 font-mono text-xs">
                          <span className="rounded-[3px] border border-border bg-bg-surface px-2.5 py-1.5 text-text-secondary">
                            {String(change.current_value)}
                          </span>
                          <span className="text-accent">→</span>
                          <span className="rounded-[3px] border border-accent/40 bg-accent/8 px-2.5 py-1.5 font-medium text-accent">
                            {String(change.suggested_target)}
                          </span>
                        </div>
                      </div>
                    ),
                  )}
                </div>
              ) : (
                <StateCard
                  title="No change required"
                  message={
                    explanation.counterfactual?.message ||
                    "This profile already clears its threshold."
                  }
                />
              )}
            </div>
          ) : null}

          {explanation && activeTab === "Groups" ? (
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              {Object.entries(explanation.risk_groups || {}).map(([key, group]) => (
                <div key={key} className={cn("rounded-md border p-4", sentimentColorClasses(group.impact))}>
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-medium">{group.label || formatFeatureLabel(key)}</p>
                    <span className="font-mono text-xs tabular">
                      {(group.total_contribution || 0).toFixed(3)}
                    </span>
                  </div>
                  <div className="mt-2.5 h-1 bg-bg-elevated">
                    <div
                      className={cn(
                        "h-full",
                        group.impact === "risk_decrease"
                          ? "bg-success"
                          : group.impact === "risk_increase"
                            ? "bg-destructive"
                            : "bg-warning",
                      )}
                      style={{
                        width: `${Math.max(8, Math.min(100, Math.abs(group.total_contribution || 0) * 100))}%`,
                      }}
                    />
                  </div>
                  <p className="mt-3 text-[13px] leading-relaxed text-text-secondary">
                    {narrativeForGroup(key, group)}
                  </p>
                </div>
              ))}
            </div>
          ) : null}

          {explanation && activeTab === "Method" ? (
            <div className="mt-5 space-y-4">
              <span className="inline-flex rounded-[3px] border border-accent/40 bg-accent/8 px-2.5 py-1 font-mono text-[11px] uppercase tracking-wider text-accent">
                {explanation.methodology?.method || "SHAP"}
              </span>
              <div className="inset p-4">
                <div className="grid grid-cols-2 gap-2 border-b border-border pb-2.5 font-mono text-[10px] uppercase tracking-wider text-text-muted">
                  <span>Vector</span>
                  <span>Baseline</span>
                </div>
                <div className="mt-3 grid gap-2">
                  {Object.entries(
                    explanation.methodology?.baseline?.baseline_values || {},
                  ).map(([feature, value]) => (
                    <div key={feature} className="grid grid-cols-2 gap-2 font-mono text-xs">
                      <span className="text-text-secondary">{formatFeatureLabel(feature)}</span>
                      <span className="font-medium text-text-primary tabular">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
              <p className="text-xs leading-relaxed text-text-muted">
                The baseline is the control profile. SHAP attributions measure how far
                this application diverges from it.
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

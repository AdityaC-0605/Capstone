"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AlertTriangle, ArrowLeft } from "lucide-react";

import { RangeField } from "@/components/range-field";
import { SegmentedControl } from "@/components/segmented-control";
import { ToggleSwitch } from "@/components/toggle-switch";
import { previewPrediction, runPrediction } from "@/lib/api";
import {
  defaultApplication,
  homeOwnershipOptions,
  loanPurposeOptions,
  verificationOptions,
} from "@/lib/constants";
import {
  formatCurrency,
  formatPercent,
  formatRiskLabel,
  gaugeColor,
  riskColorClasses,
} from "@/lib/format";
import type { CreditApplication, RiskLevel } from "@/lib/types";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";
import { toast } from "@/store/use-toast-store";

function FieldGroup({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="leaf p-6">
      <p className="section-kicker">{title}</p>
      <div className="mt-5 space-y-5">{children}</div>
    </section>
  );
}

export default function NewAssessmentPage() {
  const router = useRouter();
  const backendConfig = usePulseStore((state) => state.backendConfig);
  const addPredictionRecord = usePulseStore((state) => state.addPredictionRecord);

  const [application, setApplication] = useState<CreditApplication>(defaultApplication);
  const [includeExplanation, setIncludeExplanation] = useState(true);
  const [trackSustainability, setTrackSustainability] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [preview, setPreview] = useState<{
    risk_score: number;
    risk_level: RiskLevel;
    confidence: number;
  } | null>(null);
  const [previewing, setPreviewing] = useState(false);

  const update = (patch: Partial<CreditApplication>) =>
    setApplication((current) => ({ ...current, ...patch }));

  const hasKey = backendConfig.apiKey.trim().length > 0;

  // Debounced live estimate that updates as the analyst adjusts inputs.
  useEffect(() => {
    if (!hasKey) return;
    setPreviewing(true);
    const timer = window.setTimeout(async () => {
      try {
        setPreview(await previewPrediction(backendConfig, application));
      } catch {
        // best-effort preview; ignore transient errors
      } finally {
        setPreviewing(false);
      }
    }, 350);
    return () => window.clearTimeout(timer);
  }, [application, hasKey, backendConfig]);

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
      addPredictionRecord(application, response);
      toast.success(
        `Risk ${(response.risk_score * 100).toFixed(0)}% · ${response.risk_level.replace("_", " ")}`,
        `Scored in ${response.processing_time_ms.toFixed(0)} ms.`,
      );
      router.push(`/assessments/${response.prediction_id}`);
    } catch (submissionError) {
      const message =
        submissionError instanceof Error
          ? submissionError.message
          : "Prediction failed unexpectedly.";
      setError(message);
      toast.error("Inference failed", message);
      setLoading(false);
    }
  };

  const summaryRows: Array<[string, string]> = [
    ["Annual income", formatCurrency(application.income)],
    ["Loan amount", formatCurrency(application.loan_amount)],
    ["Debt-to-income", formatPercent(application.debt_to_income_ratio)],
    ["Credit score", `${application.credit_score}`],
  ];

  return (
    <div className="space-y-6">
      <Link
        href="/assessments"
        className="focus-ring inline-flex items-center gap-1.5 rounded-[3px] text-sm text-text-muted transition-colors hover:text-text-primary"
      >
        <ArrowLeft className="h-4 w-4" />
        Assessments
      </Link>

      {!hasKey ? (
        <div className="flex items-start gap-3 rounded-[3px] border border-warning/40 bg-warning/8 px-4 py-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
          <p className="text-sm text-text-secondary">
            No inference API key set. Add your bearer key in{" "}
            <Link href="/settings" className="text-accent hover:underline">
              Settings
            </Link>{" "}
            before scoring.
          </p>
        </div>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px] lg:items-start">
        <div className="space-y-6">
          <FieldGroup title="Applicant">
            <RangeField
              label="Age"
              min={18}
              max={80}
              value={application.age}
              valueLabel={`${application.age}`}
              onChange={(value) => update({ age: value })}
            />
            <RangeField
              label="Employment Length"
              min={0}
              max={30}
              value={application.employment_length}
              valueLabel={`${application.employment_length} yr`}
              onChange={(value) => update({ employment_length: value })}
            />
            <SegmentedControl
              label="Home Ownership"
              options={homeOwnershipOptions}
              value={application.home_ownership}
              onChange={(value) =>
                update({ home_ownership: value as CreditApplication["home_ownership"] })
              }
            />
            <SegmentedControl
              label="Income Verification"
              options={verificationOptions}
              value={application.verification_status}
              onChange={(value) =>
                update({
                  verification_status: value as CreditApplication["verification_status"],
                })
              }
              compact
            />
          </FieldGroup>

          <FieldGroup title="Financials">
            <RangeField
              label="Annual Income"
              min={10000}
              max={250000}
              step={1000}
              value={application.income}
              valueLabel={formatCurrency(application.income)}
              onChange={(value) => update({ income: value })}
            />
            <RangeField
              label="Debt-to-Income Ratio"
              min={0}
              max={1}
              step={0.01}
              value={application.debt_to_income_ratio}
              valueLabel={formatPercent(application.debt_to_income_ratio)}
              warning={application.debt_to_income_ratio > 0.4}
              onChange={(value) => update({ debt_to_income_ratio: value })}
            />
            <RangeField
              label="Credit Score"
              min={300}
              max={850}
              value={application.credit_score}
              valueLabel={`${application.credit_score}`}
              scoreTrack
              onChange={(value) => update({ credit_score: value })}
            />
          </FieldGroup>

          <FieldGroup title="Loan">
            <RangeField
              label="Loan Amount"
              min={1000}
              max={50000}
              step={500}
              value={application.loan_amount}
              valueLabel={formatCurrency(application.loan_amount)}
              onChange={(value) => update({ loan_amount: value })}
            />
            <div className="space-y-2">
              <p className="section-kicker">Loan Purpose</p>
              <div className="flex flex-wrap gap-1.5">
                {loanPurposeOptions.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => update({ loan_purpose: option.value })}
                    className={cn(
                      "focus-ring rounded-[3px] border px-2.5 py-1 text-[11px] font-medium transition-colors",
                      application.loan_purpose === option.value
                        ? "border-accent bg-accent text-bg-surface"
                        : "border-border text-text-muted hover:border-border-strong hover:text-text-primary",
                    )}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </FieldGroup>

          <FieldGroup title="Options">
            <ToggleSwitch
              label="Include SHAP Explanation"
              checked={includeExplanation}
              onChange={setIncludeExplanation}
            />
            <ToggleSwitch
              label="Track Sustainability"
              checked={trackSustainability}
              onChange={setTrackSustainability}
            />
          </FieldGroup>
        </div>

        {/* Summary rail */}
        <aside className="space-y-4 lg:sticky lg:top-[84px]">
          {/* Live estimate */}
          <div className="leaf p-6">
            <div className="flex items-center justify-between">
              <p className="section-kicker">Live Estimate</p>
              <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-text-muted">
                <span
                  className={cn(
                    "h-1.5 w-1.5 rounded-full",
                    previewing ? "animate-pulse bg-warning" : "bg-success",
                  )}
                />
                {previewing ? "updating" : "live"}
              </span>
            </div>
            {preview ? (
              <>
                <div className="mt-3 flex items-end gap-3">
                  <span
                    className="font-display text-5xl font-medium leading-none tabular transition-colors"
                    style={{ color: gaugeColor(preview.risk_score) }}
                  >
                    {preview.risk_score.toFixed(2)}
                  </span>
                  <span
                    className={cn(
                      "mb-1.5 rounded-[3px] border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                      riskColorClasses(preview.risk_level),
                    )}
                  >
                    {formatRiskLabel(preview.risk_level)}
                  </span>
                </div>
                <div className="relative mt-4">
                  <div className="flex h-1.5 overflow-hidden rounded-[2px]">
                    <span className="flex-1" style={{ background: "rgb(var(--color-success) / 0.32)" }} />
                    <span className="flex-1" style={{ background: "rgb(var(--color-warning) / 0.32)" }} />
                    <span className="flex-1" style={{ background: "rgb(var(--color-destructive) / 0.28)" }} />
                    <span className="flex-1" style={{ background: "rgb(var(--color-destructive) / 0.5)" }} />
                  </div>
                  <span
                    className="absolute -top-1 h-3.5 w-0.5 -translate-x-1/2 bg-text-primary transition-all duration-300"
                    style={{ left: `${Math.min(99, preview.risk_score * 100)}%` }}
                  />
                </div>
                <p className="mt-3 text-xs leading-relaxed text-text-muted">
                  {Math.round(preview.confidence * 100)}% confidence · updates as
                  you adjust inputs. Score to save the full explanation.
                </p>
              </>
            ) : (
              <p className="mt-3 text-sm text-text-muted">
                {hasKey
                  ? "Estimating…"
                  : "Sign in to see a live estimate."}
              </p>
            )}
          </div>

          {/* Review & submit */}
          <div className="leaf p-6">
            <p className="section-kicker">Review &amp; Submit</p>
            <dl className="mt-4 divide-y divide-border border-y border-border">
              {summaryRows.map(([label, value]) => (
                <div key={label} className="flex items-center justify-between py-2.5">
                  <dt className="text-sm text-text-muted">{label}</dt>
                  <dd className="font-mono text-sm text-text-primary tabular">{value}</dd>
                </div>
              ))}
            </dl>

            <button
              type="button"
              className="button-primary mt-5 w-full"
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="button-spinner" />
                  Scoring…
                </>
              ) : (
                "Score application"
              )}
            </button>
            <button
              type="button"
              className="button-ghost mt-2 w-full"
              onClick={() => {
                setApplication(defaultApplication);
                setError("");
              }}
            >
              Reset to defaults
            </button>

            {error ? (
              <p className="mt-4 rounded-[3px] border border-destructive/40 bg-destructive/8 px-3 py-2 text-xs leading-relaxed text-destructive">
                {error}
              </p>
            ) : null}
          </div>
        </aside>
      </div>
    </div>
  );
}

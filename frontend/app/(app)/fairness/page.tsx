"use client";

import { useState } from "react";

import { AnimatedNumber } from "@/components/animated-number";
import { RangeField } from "@/components/range-field";
import { StateCard } from "@/components/state-card";
import { fetchLiveFairnessAudit, runFairnessAudit } from "@/lib/api";
import { formatFeatureLabel } from "@/lib/format";
import type {
  FairnessAuditedAttribute,
  FairnessReport,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";
import { toast } from "@/store/use-toast-store";

const levelTone: Record<string, string> = {
  none: "text-success border-success/40 bg-success/8",
  low: "text-success border-success/40 bg-success/8",
  moderate: "text-warning border-warning/40 bg-warning/8",
  high: "text-destructive border-destructive/40 bg-destructive/8",
  severe: "text-destructive border-destructive/50 bg-destructive/12",
};

function tone(level: string) {
  return levelTone[level?.toLowerCase()] || "text-text-muted border-border bg-bg-elevated";
}

type Mode = "live" | "synthetic";

export default function FairnessPage() {
  const backendConfig = usePulseStore((state) => state.backendConfig);

  const [mode, setMode] = useState<Mode>("live");
  const [biasStrength, setBiasStrength] = useState(1.4);
  const [samples, setSamples] = useState(1000);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<FairnessReport | null>(null);
  // What was actually shown, and the live-audit metadata when applicable.
  const [shown, setShown] = useState<"live" | "synthetic" | null>(null);
  const [audited, setAudited] = useState<{
    n_decisions: number;
    approval_rule: string;
    attributes: FairnessAuditedAttribute[];
    label_dependent_metrics: string;
  } | null>(null);
  const [fallbackNote, setFallbackNote] = useState<string | null>(null);

  const runSynthetic = async () => {
    const result = await runFairnessAudit(backendConfig, {
      samples,
      bias_strength: biasStrength,
    });
    setReport(result.report);
    setAudited(null);
    setShown("synthetic");
    const violations = result.report.summary?.violations_detected ?? 0;
    toast.success(
      "Fairness audit complete",
      `${violations} violation${violations === 1 ? "" : "s"} across ${result.report.summary?.total_tests ?? 0} tests.`,
    );
  };

  const runAudit = async () => {
    setLoading(true);
    setError(null);
    setFallbackNote(null);
    try {
      if (mode === "synthetic") {
        await runSynthetic();
      } else {
        const live = await fetchLiveFairnessAudit(backendConfig);
        if (live.mode === "live" && live.report) {
          setReport(live.report);
          setAudited(live.audited ?? null);
          setShown("live");
          const v = live.report.summary?.violations_detected ?? 0;
          toast.success(
            "Live audit complete",
            `Audited ${live.audited?.n_decisions ?? 0} real decisions — ${v} violation${v === 1 ? "" : "s"}.`,
          );
        } else {
          // Not enough real history yet — fall back to the synthetic cohort.
          setFallbackNote(
            `${live.reason || "Not enough scored applications"} (${live.n_decisions ?? 0} so far). Showing a synthetic demo cohort instead.`,
          );
          await runSynthetic();
        }
      }
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Audit failed.";
      setError(message);
      toast.error("Fairness audit failed", message);
    } finally {
      setLoading(false);
    }
  };

  const summary = report?.summary;
  const rate = summary ? summary.violation_rate : 0;
  const verdict = !summary
    ? null
    : rate === 0
      ? { label: "Clean", tone: tone("low") }
      : rate < 0.34
        ? { label: "Review", tone: tone("moderate") }
        : { label: "Action needed", tone: tone("severe") };

  return (
    <div className="page-frame space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="max-w-2xl text-[15px] leading-relaxed text-text-secondary">
          Audit the model, not just the applicant. Demographic parity across
          protected groups — run it against the model&apos;s own decisions, or a
          synthetic cohort with injected bias to see the detector at work.
        </p>
        {shown ? (
          <span
            className={cn(
              "inline-flex shrink-0 items-center gap-1.5 rounded-[3px] border px-2 py-0.5 font-mono text-[11px] uppercase tracking-wider",
              shown === "live"
                ? "border-success/40 bg-success/8 text-success"
                : "border-border bg-bg-elevated text-text-muted",
            )}
          >
            <span
              className={cn(
                "status-dot",
                shown === "live" ? "status-online" : "",
              )}
            />
            {shown === "live" ? "Live model" : "Synthetic demo"}
          </span>
        ) : null}
      </div>

      <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
        {/* Controls */}
        <aside className="animate-enter-1 leaf h-fit p-5">
          <p className="section-kicker">Source</p>
          <div className="mt-3 flex rounded-[3px] border border-border p-0.5">
            {(["live", "synthetic"] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  "focus-ring flex-1 rounded-[2px] px-3 py-1.5 text-xs font-medium capitalize transition-colors",
                  mode === m
                    ? "bg-accent text-[rgb(var(--color-on-accent))]"
                    : "text-text-secondary hover:text-text-primary",
                )}
              >
                {m === "live" ? "Live model" : "Synthetic"}
              </button>
            ))}
          </div>

          {mode === "synthetic" ? (
            <div className="mt-5 space-y-5">
              <div>
                <RangeField
                  label="Injected Bias"
                  min={0.5}
                  max={2}
                  step={0.1}
                  value={biasStrength}
                  valueLabel={biasStrength.toFixed(1)}
                  warning={biasStrength > 1.5}
                  onChange={setBiasStrength}
                />
                <p className="mt-1.5 text-xs leading-relaxed text-text-muted">
                  1.0 is balanced; higher skews approvals toward one group so
                  you can watch the audit catch it.
                </p>
              </div>
              <RangeField
                label="Cohort Size"
                min={200}
                max={5000}
                step={100}
                value={samples}
                valueLabel={samples.toLocaleString()}
                onChange={setSamples}
              />
            </div>
          ) : (
            <p className="mt-4 text-xs leading-relaxed text-text-muted">
              Audits the model&apos;s actual approve/deny decisions from your
              persisted assessments, grouped by age band (and gender/race when
              submitted). Needs a handful of scored applications first.
            </p>
          )}

          <button
            type="button"
            className="button-primary mt-5 w-full"
            onClick={runAudit}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="button-spinner" />
                Auditing…
              </>
            ) : (
              "Run audit"
            )}
          </button>
          <p className="mt-3 text-xs leading-relaxed text-text-muted">
            {mode === "live" ? (
              <>
                Reads persisted decisions from{" "}
                <span className="font-mono text-text-secondary">
                  {backendConfig.inferenceUrl}
                </span>
                .
              </>
            ) : (
              <>
                Runs against{" "}
                <span className="font-mono text-text-secondary">
                  {backendConfig.mainUrl}
                </span>
                .
              </>
            )}
          </p>
        </aside>

        {/* Results */}
        <section className="animate-enter-2 space-y-6">
          {error ? (
            <StateCard tone="error" title="Could not run the audit" message={error} />
          ) : null}

          {fallbackNote ? (
            <StateCard tone="warning" title="Using synthetic cohort" message={fallbackNote} />
          ) : null}

          {!report && !error ? (
            <StateCard
              title="No audit yet"
              message="Choose a source and run the audit to measure demographic parity across protected groups."
            />
          ) : null}

          {/* Real-decision audit metadata */}
          {shown === "live" && audited ? (
            <div className="leaf p-6">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="section-kicker">Audited Decisions</p>
                <span className="font-mono text-[11px] uppercase tracking-wider text-text-muted">
                  {audited.n_decisions.toLocaleString()} real decisions
                </span>
              </div>
              <p className="mt-2 text-sm text-text-secondary">
                {audited.approval_rule}.
              </p>
              <div className="mt-4 grid gap-4 sm:grid-cols-2">
                {audited.attributes.map((attr) => (
                  <div
                    key={attr.attribute}
                    className="rounded-[3px] border border-border bg-bg-elevated/50 p-4"
                  >
                    <div className="flex items-center justify-between">
                      <p className="font-medium text-text-primary">
                        {formatFeatureLabel(attr.attribute)}
                      </p>
                      <span className="font-mono text-[11px] text-text-muted">
                        Δ {(attr.approval_disparity * 100).toFixed(0)}pp
                      </span>
                    </div>
                    <div className="mt-3 space-y-1.5">
                      {Object.entries(attr.groups).map(([g, stat]) => (
                        <div
                          key={g}
                          className="flex items-center justify-between text-xs"
                        >
                          <span className="text-text-secondary">
                            {g}{" "}
                            <span className="text-text-muted">
                              (n={stat.n})
                            </span>
                          </span>
                          <span className="font-mono text-text-primary tabular">
                            {(stat.approval_rate * 100).toFixed(0)}% approved
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-xs leading-relaxed text-text-muted">
                {audited.label_dependent_metrics}
              </p>
            </div>
          ) : null}

          {summary ? (
            <>
              {/* Summary */}
              <div className="leaf grid gap-px overflow-hidden bg-border sm:grid-cols-4">
                <div className="bg-bg-surface px-5 py-5">
                  <p className="section-kicker">Tests Run</p>
                  <p className="mt-2 font-display text-3xl font-medium text-text-primary tabular">
                    {summary.total_tests}
                  </p>
                </div>
                <div className="bg-bg-surface px-5 py-5">
                  <p className="section-kicker">Violations</p>
                  <p className="mt-2 font-display text-3xl font-medium text-text-primary tabular">
                    {summary.violations_detected}
                  </p>
                </div>
                <div className="bg-bg-surface px-5 py-5">
                  <p className="section-kicker">Violation Rate</p>
                  <AnimatedNumber
                    value={rate * 100}
                    formatter={(v) => `${v.toFixed(0)}%`}
                    className="mt-2 block font-display text-3xl font-medium text-text-primary tabular"
                  />
                </div>
                <div className="bg-bg-surface px-5 py-5">
                  <p className="section-kicker">Verdict</p>
                  {verdict ? (
                    <span
                      className={cn(
                        "mt-2 inline-flex rounded-[3px] border px-2.5 py-1 font-mono text-xs uppercase tracking-wider",
                        verdict.tone,
                      )}
                    >
                      {verdict.label}
                    </span>
                  ) : null}
                </div>
              </div>

              {/* Metric breakdown */}
              <div className="leaf p-6">
                <p className="section-kicker">By Fairness Metric</p>
                <div className="mt-4 overflow-x-auto">
                  <table className="ledger-table">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Tests</th>
                        <th>Violations</th>
                        <th>Avg Disparity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(report.by_fairness_metric || {}).map(
                        ([metric, stat]) => (
                          <tr key={metric}>
                            <td className="font-medium text-text-primary">
                              {formatFeatureLabel(metric)}
                            </td>
                            <td className="font-mono text-text-secondary tabular">
                              {stat.tests_conducted}
                            </td>
                            <td
                              className={cn(
                                "font-mono tabular",
                                stat.violations > 0 ? "text-destructive" : "text-success",
                              )}
                            >
                              {stat.violations}
                            </td>
                            <td className="font-mono text-text-primary tabular">
                              {stat.average_disparity.toFixed(3)}
                            </td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Protected attributes */}
              <div className="grid gap-4 sm:grid-cols-2">
                {Object.entries(report.by_protected_attribute || {}).map(
                  ([attr, stat]) => (
                    <div key={attr} className="leaf p-5">
                      <div className="flex items-center justify-between">
                        <p className="font-display text-lg font-medium text-text-primary">
                          {formatFeatureLabel(attr)}
                        </p>
                        <span
                          className={cn(
                            "rounded-[3px] border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                            tone(stat.worst_violation),
                          )}
                        >
                          {stat.worst_violation}
                        </span>
                      </div>
                      <p className="mt-3 text-sm text-text-secondary">
                        <span className="font-mono text-text-primary tabular">
                          {stat.violations}
                        </span>{" "}
                        of{" "}
                        <span className="font-mono text-text-primary tabular">
                          {stat.tests_conducted}
                        </span>{" "}
                        tests flagged ({(stat.violation_rate * 100).toFixed(0)}%).
                      </p>
                    </div>
                  ),
                )}
              </div>

              {/* Recommendations */}
              {report.recommendations && report.recommendations.length > 0 ? (
                <div className="leaf p-6">
                  <p className="section-kicker">Recommendations</p>
                  <ul className="mt-4 space-y-3">
                    {report.recommendations.map((rec, index) => (
                      <li key={index} className="flex gap-3 text-sm leading-relaxed text-text-secondary">
                        <span className="index-num pt-0.5">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </>
          ) : null}
        </section>
      </div>
    </div>
  );
}

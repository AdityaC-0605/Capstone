"use client";

import Link from "next/link";
import { useState } from "react";
import { ArrowLeft, Download, FileSpreadsheet, Upload } from "lucide-react";

import { StateCard } from "@/components/state-card";
import { runBatch } from "@/lib/api";
import { downloadCsv, parseCsv, toCsv } from "@/lib/csv";
import {
  homeOwnershipOptions,
  loanPurposeOptions,
  verificationOptions,
} from "@/lib/constants";
import {
  formatCurrency,
  formatRiskLabel,
  riskColorClasses,
} from "@/lib/format";
import type {
  BatchPredictionResponse,
  CreditApplication,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";
import { toast } from "@/store/use-toast-store";

const COLUMNS = [
  "age",
  "income",
  "employment_length",
  "debt_to_income_ratio",
  "credit_score",
  "loan_amount",
  "loan_purpose",
  "home_ownership",
  "verification_status",
] as const;

const NUMERIC: Record<string, [number, number, boolean]> = {
  age: [18, 100, true],
  income: [0, Infinity, false],
  employment_length: [0, 50, true],
  debt_to_income_ratio: [0, 1, false],
  credit_score: [300, 850, true],
  loan_amount: [1000, Infinity, false],
};

const PURPOSES = new Set(loanPurposeOptions.map((o) => o.value));
const HOMES = new Set(homeOwnershipOptions.map((o) => o.value));
const VERIFS = new Set(verificationOptions.map((o) => o.value));
const MAX_ROWS = 100;

interface RowError {
  row: number;
  message: string;
}

function buildApplications(text: string): {
  valid: CreditApplication[];
  errors: RowError[];
} {
  const rows = parseCsv(text);
  if (rows.length < 2) {
    return { valid: [], errors: [{ row: 0, message: "No data rows found." }] };
  }
  const header = rows[0].map((h) => h.trim().toLowerCase());
  const idx: Record<string, number> = {};
  COLUMNS.forEach((col) => {
    idx[col] = header.indexOf(col);
  });
  const missing = COLUMNS.filter((col) => idx[col] === -1);
  if (missing.length) {
    return {
      valid: [],
      errors: [{ row: 0, message: `Missing columns: ${missing.join(", ")}` }],
    };
  }

  const valid: CreditApplication[] = [];
  const errors: RowError[] = [];

  for (let r = 1; r < rows.length; r += 1) {
    const cells = rows[r];
    const get = (col: string) => (cells[idx[col]] ?? "").trim();
    const rowErrors: string[] = [];
    const app: Record<string, unknown> = {};

    Object.entries(NUMERIC).forEach(([field, [min, max, isInt]]) => {
      const raw = get(field);
      const num = Number(raw);
      if (raw === "" || Number.isNaN(num)) {
        rowErrors.push(`${field} is not a number`);
      } else if (num < min || num > max) {
        rowErrors.push(`${field} out of range`);
      } else if (isInt && !Number.isInteger(num)) {
        rowErrors.push(`${field} must be a whole number`);
      } else {
        app[field] = num;
      }
    });

    const purpose = get("loan_purpose").toLowerCase();
    if (!PURPOSES.has(purpose as never)) rowErrors.push("invalid loan_purpose");
    else app.loan_purpose = purpose;

    const home = get("home_ownership").toLowerCase();
    if (!HOMES.has(home as never)) rowErrors.push("invalid home_ownership");
    else app.home_ownership = home;

    const verification = get("verification_status").toLowerCase();
    if (!VERIFS.has(verification as never))
      rowErrors.push("invalid verification_status");
    else app.verification_status = verification;

    if (rowErrors.length) {
      errors.push({ row: r, message: rowErrors.join("; ") });
    } else {
      valid.push(app as unknown as CreditApplication);
    }
  }
  return { valid, errors };
}

const TEMPLATE = toCsv([
  [...COLUMNS],
  [35, 65000, 5, 0.3, 720, 25000, "debt_consolidation", "rent", "verified"],
  [44, 120000, 12, 0.18, 780, 18000, "home_improvement", "mortgage", "verified"],
  [29, 38000, 2, 0.46, 610, 9000, "medical", "rent", "not_verified"],
]);

export default function BatchAssessmentPage() {
  const backendConfig = usePulseStore((state) => state.backendConfig);

  const [fileName, setFileName] = useState("");
  const [valid, setValid] = useState<CreditApplication[]>([]);
  const [errors, setErrors] = useState<RowError[]>([]);
  const [parseError, setParseError] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BatchPredictionResponse | null>(null);

  const handleFile = (file: File) => {
    setResult(null);
    setParseError("");
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const { valid: rows, errors: rowErrors } = buildApplications(
          String(reader.result || ""),
        );
        setValid(rows);
        setErrors(rowErrors);
        if (!rows.length) {
          setParseError(
            rowErrors[0]?.message || "No valid applications found.",
          );
        }
      } catch {
        setParseError("Could not parse this file as CSV.");
        setValid([]);
        setErrors([]);
      }
    };
    reader.readAsText(file);
  };

  const score = async () => {
    const batch = valid.slice(0, MAX_ROWS);
    setLoading(true);
    try {
      const response = await runBatch(backendConfig, batch);
      setResult(response);
      toast.success(
        "Batch scored",
        `${response.batch_summary.successful_predictions}/${response.batch_summary.total_applications} applications scored.`,
      );
    } catch (caught) {
      const message =
        caught instanceof Error ? caught.message : "Batch scoring failed.";
      toast.error("Batch scoring failed", message);
    } finally {
      setLoading(false);
    }
  };

  const exportResults = () => {
    if (!result) return;
    const sent = valid.slice(0, MAX_ROWS);
    const header = [
      ...COLUMNS,
      "risk_score",
      "risk_level",
      "confidence",
      "status",
    ];
    const rows = result.predictions.map((prediction, i) => {
      const app = sent[i];
      return [
        ...COLUMNS.map((col) => (app ? (app[col] as string | number) : "")),
        prediction.risk_score.toFixed(4),
        prediction.risk_level,
        prediction.confidence.toFixed(4),
        prediction.status,
      ];
    });
    downloadCsv("pulseledger-batch-results.csv", toCsv([header, ...rows]));
  };

  const summary = result?.batch_summary;

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
        <button
          type="button"
          onClick={() => downloadCsv("pulseledger-template.csv", TEMPLATE)}
          className="focus-ring inline-flex items-center gap-1.5 rounded-[3px] text-sm text-accent hover:underline"
        >
          <Download className="h-4 w-4" />
          Download template
        </button>
      </div>

      <div>
        <p className="section-kicker">Bulk Scoring</p>
        <h1 className="mt-2 font-display text-2xl font-medium text-text-primary md:text-3xl">
          Score a portfolio from CSV.
        </h1>
        <p className="mt-2 max-w-xl text-sm leading-relaxed text-text-secondary">
          Upload up to {MAX_ROWS} applications with the template columns. Each
          row is validated locally, scored in one request, and downloadable with
          risk scores attached.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px] lg:items-start">
        <div className="space-y-6">
          {/* Upload */}
          <div className="leaf p-6">
            <label className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-[4px] border border-dashed border-border-strong bg-bg-elevated/40 px-6 py-10 text-center transition-colors hover:border-accent">
              <Upload className="h-6 w-6 text-text-muted" />
              <span className="text-sm font-medium text-text-primary">
                {fileName || "Choose a CSV file"}
              </span>
              <span className="text-xs text-text-muted">
                Columns: {COLUMNS.join(", ")}
              </span>
              <input
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) handleFile(file);
                }}
              />
            </label>

            {parseError ? (
              <p className="mt-4 rounded-[3px] border border-destructive/40 bg-destructive/8 px-3 py-2 text-xs text-destructive">
                {parseError}
              </p>
            ) : null}

            {valid.length > 0 || errors.length > 0 ? (
              <div className="mt-4 flex flex-wrap gap-2 text-xs">
                <span className="rounded-[3px] border border-success/40 bg-success/8 px-2.5 py-1 font-mono text-success">
                  {valid.length} valid
                </span>
                {errors.length > 0 ? (
                  <span className="rounded-[3px] border border-warning/40 bg-warning/8 px-2.5 py-1 font-mono text-warning">
                    {errors.length} skipped
                  </span>
                ) : null}
                {valid.length > MAX_ROWS ? (
                  <span className="rounded-[3px] border border-border px-2.5 py-1 font-mono text-text-muted">
                    first {MAX_ROWS} will be scored
                  </span>
                ) : null}
              </div>
            ) : null}

            {errors.length > 0 ? (
              <div className="mt-3 max-h-32 overflow-y-auto rounded-[3px] border border-border bg-bg-elevated/40 p-3 text-xs text-text-secondary">
                {errors.slice(0, 8).map((error) => (
                  <p key={error.row} className="font-mono">
                    {error.row === 0 ? "file" : `row ${error.row}`}:{" "}
                    {error.message}
                  </p>
                ))}
                {errors.length > 8 ? (
                  <p className="mt-1 text-text-muted">
                    …and {errors.length - 8} more
                  </p>
                ) : null}
              </div>
            ) : null}
          </div>

          {/* Results */}
          {result ? (
            <div className="leaf p-6">
              <div className="flex items-center justify-between">
                <p className="section-kicker">Results</p>
                <button
                  type="button"
                  onClick={exportResults}
                  className="focus-ring inline-flex items-center gap-1.5 rounded-[3px] text-sm text-accent hover:underline"
                >
                  <Download className="h-4 w-4" />
                  Download results
                </button>
              </div>
              <div className="mt-4 overflow-x-auto">
                <table className="ledger-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Risk</th>
                      <th>Band</th>
                      <th>Conf.</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.predictions.map((prediction, i) => (
                      <tr key={prediction.prediction_id || i}>
                        <td className="font-mono text-text-muted">{i + 1}</td>
                        <td className="font-mono font-medium text-text-primary tabular">
                          {prediction.risk_score.toFixed(3)}
                        </td>
                        <td>
                          <span
                            className={cn(
                              "rounded-[3px] border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                              riskColorClasses(prediction.risk_level),
                            )}
                          >
                            {formatRiskLabel(prediction.risk_level)}
                          </span>
                        </td>
                        <td className="font-mono text-text-secondary tabular">
                          {Math.round((prediction.confidence || 0) * 100)}%
                        </td>
                        <td className="font-mono text-[11px] uppercase tracking-wider text-text-muted">
                          {prediction.status}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <StateCard
              title="No results yet"
              message="Upload a CSV and run the batch to see scored applications here."
            />
          )}
        </div>

        {/* Run rail */}
        <aside className="lg:sticky lg:top-[84px]">
          <div className="leaf p-6">
            <p className="section-kicker">Run</p>
            <dl className="mt-4 divide-y divide-border border-y border-border">
              <div className="flex items-center justify-between py-2.5">
                <dt className="text-sm text-text-muted">Valid rows</dt>
                <dd className="font-mono text-sm text-text-primary tabular">
                  {Math.min(valid.length, MAX_ROWS)}
                </dd>
              </div>
              {summary ? (
                <>
                  <div className="flex items-center justify-between py-2.5">
                    <dt className="text-sm text-text-muted">Avg risk</dt>
                    <dd className="font-mono text-sm text-text-primary tabular">
                      {summary.average_risk_score.toFixed(3)}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between py-2.5">
                    <dt className="text-sm text-text-muted">Scored</dt>
                    <dd className="font-mono text-sm text-text-primary tabular">
                      {summary.successful_predictions}/
                      {summary.total_applications}
                    </dd>
                  </div>
                </>
              ) : null}
            </dl>

            {summary ? (
              <div className="mt-4 space-y-1.5">
                <p className="section-kicker">Distribution</p>
                {Object.entries(summary.risk_distribution).map(
                  ([level, count]) =>
                    count > 0 ? (
                      <div
                        key={level}
                        className="flex items-center justify-between text-xs"
                      >
                        <span className="text-text-secondary">
                          {formatRiskLabel(level)}
                        </span>
                        <span className="font-mono text-text-primary">
                          {count}
                        </span>
                      </div>
                    ) : null,
                )}
              </div>
            ) : null}

            <button
              type="button"
              className="button-primary mt-5 w-full"
              onClick={score}
              disabled={loading || valid.length === 0}
            >
              {loading ? (
                <>
                  <span className="button-spinner" />
                  Scoring…
                </>
              ) : (
                `Score ${Math.min(valid.length, MAX_ROWS) || ""} applications`
              )}
            </button>
            <p className="mt-3 flex items-center gap-1.5 text-xs text-text-muted">
              <FileSpreadsheet className="h-3.5 w-3.5" />
              Results are exportable; batch runs aren&apos;t stored in history.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}

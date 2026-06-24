"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ArrowRight, ArrowUpRight, Cpu, Network, Plus, Scale } from "lucide-react";

import { Sparkline } from "@/components/sparkline";
import { StateCard } from "@/components/state-card";
import { fetchModelInfo } from "@/lib/api";
import {
  formatCurrency,
  formatRiskLabel,
  relativeTime,
  riskColorClasses,
} from "@/lib/format";
import type { ModelInfo, RiskLevel } from "@/lib/types";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";

const quickActions = [
  {
    href: "/assessments/new",
    label: "New assessment",
    detail: "Score a credit application with a full SHAP explanation.",
    icon: Plus,
  },
  {
    href: "/fairness",
    label: "Run a fairness audit",
    detail: "Measure bias across protected groups before deployment.",
    icon: Scale,
  },
  {
    href: "/federated",
    label: "Train a federated round",
    detail: "Aggregate models across institutions without pooling data.",
    icon: Network,
  },
];

const BANDS: Array<{ key: RiskLevel; label: string; color: string }> = [
  { key: "low", label: "Low", color: "rgb(var(--color-success))" },
  { key: "medium", label: "Medium", color: "rgb(var(--color-warning))" },
  { key: "high", label: "High", color: "rgb(var(--color-destructive))" },
  { key: "very_high", label: "Severe", color: "rgb(var(--color-destructive))" },
];

function Kpi({
  label,
  value,
  hint,
  spark,
  color = "rgb(var(--color-accent))",
  accent,
}: {
  label: string;
  value: string;
  hint: string;
  spark?: number[];
  color?: string;
  accent?: boolean;
}) {
  return (
    <div className="leaf p-5">
      <p className="section-kicker">{label}</p>
      <p
        className={cn(
          "mt-2.5 font-display text-3xl font-medium tabular",
          accent ? "text-accent" : "text-text-primary",
        )}
      >
        {value}
      </p>
      {spark && spark.length > 1 ? (
        <Sparkline values={spark} color={color} />
      ) : (
        <p className="mt-1 text-xs text-text-muted">{hint}</p>
      )}
    </div>
  );
}

export default function DashboardPage() {
  const history = usePulseStore((state) => state.predictionHistory);
  const session = usePulseStore((state) => state.sessionSustainability);
  const backendStatus = usePulseStore((state) => state.backendStatus);
  const backendConfig = usePulseStore((state) => state.backendConfig);
  const federated = usePulseStore((state) => state.federatedState);
  const [model, setModel] = useState<ModelInfo | null>(null);

  useEffect(() => {
    if (!backendConfig.apiKey.trim()) return;
    let active = true;
    fetchModelInfo(backendConfig)
      .then((info) => active && setModel(info))
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [backendConfig]);

  const count = history.length;
  const chronological = [...history].reverse();
  const avgRisk = count
    ? history.reduce((acc, item) => acc + item.result.risk_score, 0) / count
    : 0;
  const avgConfidence = count
    ? history.reduce((acc, item) => acc + (item.result.confidence || 0), 0) / count
    : 0;

  const riskSpark = chronological.map((item) => item.result.risk_score);
  const confSpark = chronological.map((item) => item.result.confidence || 0);
  const carbonSpark = chronological.map(
    (item) => item.sustainability_metrics?.carbon_emissions || 0,
  );
  const trend = chronological.map((item, i) => ({
    n: i + 1,
    risk: Number(item.result.risk_score.toFixed(3)),
  }));

  const bandCounts = BANDS.map((band) => ({
    ...band,
    count: history.filter((item) => item.result.risk_level === band.key).length,
  }));
  const recent = history.slice(0, 6);

  return (
    <div className="space-y-8">
      <p className="max-w-2xl text-[15px] leading-relaxed text-text-secondary">
        Your credit-risk workspace. Score applications, inspect the reasoning, and
        keep the model honest on fairness and carbon.
      </p>

      {/* KPIs */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi label="Assessments" value={`${count}`} hint="this session" />
        <Kpi
          label="Average Risk"
          value={count ? avgRisk.toFixed(2) : "—"}
          hint="across session"
          spark={riskSpark}
          color="rgb(var(--color-warning))"
        />
        <Kpi
          label="Avg Confidence"
          value={count ? `${Math.round(avgConfidence * 100)}%` : "—"}
          hint="model certainty"
          spark={confSpark}
          color="rgb(var(--color-accent))"
        />
        <Kpi
          label="Session Carbon"
          value={
            session.totalCarbon > 0 ? session.totalCarbon.toFixed(4) : "0.0000"
          }
          hint="kg CO₂ tracked"
          spark={carbonSpark}
          color="rgb(var(--color-success))"
          accent
        />
      </section>

      {/* Trend + distribution */}
      <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        <section className="leaf p-6">
          <div className="flex items-baseline justify-between">
            <p className="section-kicker">Portfolio Risk Trend</p>
            <span className="font-mono text-xs text-text-muted">
              {count} scored
            </span>
          </div>
          {trend.length > 1 ? (
            <div className="mt-5 h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={trend}
                  margin={{ top: 6, right: 6, bottom: 0, left: -20 }}
                >
                  <defs>
                    <linearGradient id="riskFill" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="0%" stopColor="rgb(var(--color-accent))" stopOpacity={0.16} />
                      <stop offset="100%" stopColor="rgb(var(--color-accent))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--color-border))" vertical={false} />
                  <XAxis dataKey="n" tick={{ fill: "rgb(var(--color-text-muted))", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 1]} tick={{ fill: "rgb(var(--color-text-muted))", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: "rgb(var(--color-bg-surface))",
                      border: "1px solid rgb(var(--color-border-strong))",
                      borderRadius: "4px",
                      fontSize: "12px",
                    }}
                    formatter={(v: number) => [Number(v).toFixed(3), "Risk"]}
                    labelFormatter={(l) => `Assessment ${l}`}
                  />
                  <Area type="monotone" dataKey="risk" stroke="rgb(var(--color-accent))" fill="url(#riskFill)" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="mt-5">
              <StateCard
                title="Not enough data yet"
                message="Score a couple of assessments to plot the portfolio risk trend."
              />
            </div>
          )}
        </section>

        <section className="leaf p-6">
          <p className="section-kicker">Risk Band Distribution</p>
          {count > 0 ? (
            <>
              <div className="mt-5 flex h-2.5 overflow-hidden rounded-[2px]">
                {bandCounts.map((band) =>
                  band.count > 0 ? (
                    <span
                      key={band.key}
                      style={{
                        width: `${(band.count / count) * 100}%`,
                        background: band.color,
                        opacity: band.key === "very_high" ? 0.7 : 1,
                      }}
                    />
                  ) : null,
                )}
              </div>
              <dl className="mt-4 space-y-2.5">
                {bandCounts.map((band) => (
                  <div key={band.key} className="flex items-center justify-between text-sm">
                    <dt className="flex items-center gap-2 text-text-secondary">
                      <span className="band-dot" style={{ background: band.color, opacity: band.key === "very_high" ? 0.7 : 1 }} />
                      {band.label}
                    </dt>
                    <dd className="font-mono text-text-primary tabular">{band.count}</dd>
                  </div>
                ))}
              </dl>
            </>
          ) : (
            <div className="mt-5">
              <StateCard title="No assessments yet" message="Band mix appears once you score applications." />
            </div>
          )}
        </section>
      </div>

      {/* Recent + side rail */}
      <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        <section className="leaf overflow-hidden">
          <div className="flex items-center justify-between border-b border-border px-6 py-4">
            <p className="section-kicker">Recent Assessments</p>
            <Link href="/assessments" className="focus-ring rounded-[3px] text-xs text-accent hover:underline">
              View all
            </Link>
          </div>
          {recent.length > 0 ? (
            <table className="ledger-table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Amount</th>
                  <th>Risk</th>
                  <th>Band</th>
                  <th aria-hidden="true" />
                </tr>
              </thead>
              <tbody>
                {recent.map((item) => (
                  <tr key={item.result.prediction_id} className="group">
                    <td className="text-text-muted">{relativeTime(item.timestamp)}</td>
                    <td className="font-mono text-text-primary tabular">
                      {formatCurrency(item.input.loan_amount)}
                    </td>
                    <td className="font-mono font-medium text-text-primary tabular">
                      {item.result.risk_score.toFixed(2)}
                    </td>
                    <td>
                      <span
                        className={cn(
                          "rounded-[3px] border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                          riskColorClasses(item.result.risk_level),
                        )}
                      >
                        {formatRiskLabel(item.result.risk_level)}
                      </span>
                    </td>
                    <td className="text-right">
                      <Link
                        href={`/assessments/${item.result.prediction_id}`}
                        className="focus-ring inline-flex rounded-[3px] text-text-muted transition-colors group-hover:text-accent"
                        aria-label="Open assessment"
                      >
                        <ArrowUpRight className="h-4 w-4" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="p-6">
              <StateCard
                title="No assessments yet"
                message="Run your first credit assessment to populate the workspace."
                actionLabel="New assessment"
                onAction={() => {
                  window.location.href = "/assessments/new";
                }}
              />
            </div>
          )}
        </section>

        <section className="space-y-6">
          {/* Model card */}
          <div className="leaf p-6">
            <div className="flex items-center justify-between">
              <p className="section-kicker">Served Model</p>
              <span className="flex h-8 w-8 items-center justify-center rounded-[3px] border border-border bg-bg-elevated text-accent">
                <Cpu className="h-4 w-4" />
              </span>
            </div>
            {model ? (
              <dl className="mt-4 space-y-2.5 text-sm">
                <div className="flex items-center justify-between">
                  <dt className="text-text-secondary">Source</dt>
                  <dd>
                    <span
                      className={cn(
                        "rounded-[3px] border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                        model.model_source === "trained"
                          ? "border-success/40 bg-success/8 text-success"
                          : "border-border bg-bg-elevated text-text-muted",
                      )}
                    >
                      {model.model_source}
                    </span>
                  </dd>
                </div>
                <div className="flex items-center justify-between">
                  <dt className="text-text-secondary">Algorithm</dt>
                  <dd className="font-mono text-xs text-text-primary">
                    {model.algorithm || model.model_type}
                  </dd>
                </div>
                {typeof model.roc_auc === "number" ? (
                  <div className="flex items-center justify-between border-t border-border pt-2.5">
                    <dt className="text-text-secondary">Holdout ROC-AUC</dt>
                    <dd className="font-mono text-sm font-medium text-text-primary tabular">
                      {model.roc_auc.toFixed(3)}
                    </dd>
                  </div>
                ) : null}
              </dl>
            ) : (
              <p className="mt-4 text-sm text-text-muted">
                Sign in to load model details.
              </p>
            )}
          </div>

          {/* System health */}
          <div className="leaf p-6">
            <p className="section-kicker">System Health</p>
            <dl className="mt-4 space-y-3">
              {[
                { label: "Main API", s: backendStatus.main.state },
                { label: "Inference Engine", s: backendStatus.inference.state },
              ].map((row) => (
                <div key={row.label} className="flex items-center justify-between">
                  <dt className="flex items-center gap-2.5 text-sm text-text-secondary">
                    <span
                      className={cn(
                        "status-dot",
                        row.s === "healthy"
                          ? "status-online"
                          : row.s === "checking"
                            ? "status-checking"
                            : "status-offline",
                      )}
                    />
                    {row.label}
                  </dt>
                  <dd className="font-mono text-xs uppercase tracking-wider text-text-muted">
                    {row.s}
                  </dd>
                </div>
              ))}
              <div className="flex items-center justify-between border-t border-border pt-3">
                <dt className="text-sm text-text-secondary">Last federated loss</dt>
                <dd className="font-mono text-sm text-text-primary tabular">
                  {federated.bestValLoss !== null
                    ? federated.bestValLoss.toFixed(4)
                    : "—"}
                </dd>
              </div>
            </dl>
          </div>
        </section>
      </div>

      {/* Quick actions */}
      <section>
        <p className="section-kicker mb-3">Quick Actions</p>
        <div className="grid gap-4 md:grid-cols-3">
          {quickActions.map((action) => (
            <Link
              key={action.href}
              href={action.href}
              className="group leaf leaf-interactive flex flex-col gap-3 p-5"
            >
              <div className="flex items-center justify-between">
                <span className="flex h-9 w-9 items-center justify-center rounded-[3px] border border-border bg-bg-elevated text-accent">
                  <action.icon className="h-[18px] w-[18px]" />
                </span>
                <ArrowUpRight className="h-4 w-4 text-text-muted transition-colors group-hover:text-accent" />
              </div>
              <div>
                <p className="font-display text-base font-medium text-text-primary">
                  {action.label}
                </p>
                <p className="mt-1 text-sm leading-relaxed text-text-secondary">
                  {action.detail}
                </p>
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}

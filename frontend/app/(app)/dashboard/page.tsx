"use client";

import Link from "next/link";
import { ArrowRight, ArrowUpRight, Network, Plus, Scale } from "lucide-react";

import { StatCard } from "@/components/stat-card";
import { StateCard } from "@/components/state-card";
import {
  formatCarbon,
  formatCurrency,
  formatRiskLabel,
  relativeTime,
  riskColorClasses,
} from "@/lib/format";
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

export default function DashboardPage() {
  const history = usePulseStore((state) => state.predictionHistory);
  const session = usePulseStore((state) => state.sessionSustainability);
  const backendStatus = usePulseStore((state) => state.backendStatus);
  const federated = usePulseStore((state) => state.federatedState);

  const count = history.length;
  const avgRisk = count
    ? history.reduce((acc, item) => acc + item.result.risk_score, 0) / count
    : 0;
  const avgConfidence = count
    ? history.reduce((acc, item) => acc + (item.result.confidence || 0), 0) / count
    : 0;
  const recent = history.slice(0, 6);

  return (
    <div className="space-y-9">
      <p className="max-w-2xl text-[15px] leading-relaxed text-text-secondary">
        Your credit-risk workspace. Score applications, inspect the reasoning, and
        keep the model honest on fairness and carbon.
      </p>

      {/* KPIs */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Assessments" value={count} hint="this session" />
        <StatCard
          label="Average Risk"
          value={count ? avgRisk.toFixed(2) : "—"}
          hint="across session"
        />
        <StatCard
          label="Avg Confidence"
          value={count ? `${Math.round(avgConfidence * 100)}%` : "—"}
          hint="model certainty"
        />
        <StatCard
          label="Session Carbon"
          value={session.totalCarbon > 0 ? session.totalCarbon.toFixed(4) : "0.0000"}
          hint="kg CO₂ tracked"
          accent
        />
      </section>

      <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        {/* Recent assessments */}
        <section className="leaf overflow-hidden">
          <div className="flex items-center justify-between border-b border-border px-6 py-4">
            <p className="section-kicker">Recent Assessments</p>
            <Link
              href="/assessments"
              className="focus-ring rounded-[3px] text-xs text-accent hover:underline"
            >
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

        {/* System health */}
        <section className="space-y-4">
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
                <dt className="text-sm text-text-secondary">Last federated best loss</dt>
                <dd className="font-mono text-sm text-text-primary tabular">
                  {federated.bestValLoss !== null
                    ? federated.bestValLoss.toFixed(4)
                    : "—"}
                </dd>
              </div>
            </dl>
            <Link
              href="/settings"
              className="focus-ring mt-4 inline-flex items-center gap-1.5 rounded-[3px] text-xs text-accent hover:underline"
            >
              Configure backend
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
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
              className="group leaf flex flex-col gap-3 p-5 transition-colors hover:border-border-strong"
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

"use client";

import { useMemo } from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AnimatedNumber } from "@/components/animated-number";
import { Sparkline } from "@/components/sparkline";
import { StateCard } from "@/components/state-card";
import { formatCarbon, formatEnergy, formatSeconds, getContrastVerdict } from "@/lib/format";
import { clamp, cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";

function EcoGauge({ score }: { score: number }) {
  const normalized = clamp(score, 0, 100);
  const radius = 82;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (normalized / 100) * circumference;

  return (
    <div className="relative mx-auto h-[200px] w-[200px]">
      <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90">
        <circle
          cx="100"
          cy="100"
          r={radius}
          stroke="rgba(255,255,255,0.05)"
          strokeWidth="12"
          fill="none"
        />
        <circle
          cx="100"
          cy="100"
          r={radius}
          stroke="url(#ecoGradient)"
          strokeWidth="12"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-1000"
        />
        <defs>
          <linearGradient id="ecoGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgb(var(--color-success))" />
            <stop offset="100%" stopColor="#2DD4BF" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <p className="text-[10px] font-bold uppercase tracking-wider text-success/80">
          Eco Score
        </p>
        <AnimatedNumber
          value={normalized}
          formatter={(value) => value.toFixed(0)}
          className="mt-2 font-display text-4xl font-bold text-text-primary"
        />
      </div>
    </div>
  );
}

export default function SustainabilityPage() {
  const predictionHistory = usePulseStore((state) => state.predictionHistory);
  const session = usePulseStore((state) => state.sessionSustainability);
  const ui = usePulseStore((state) => state.ui);
  const setNasState = usePulseStore((state) => state.setNasState);

  const chartData = useMemo(
    () =>
      predictionHistory
        .filter((item) => item.sustainability_metrics)
        .map((item, index) => ({
          prediction: index + 1,
          carbon: item.sustainability_metrics?.carbon_emissions || 0,
          energy: item.sustainability_metrics?.energy_kwh || 0,
        }))
        .reverse(),
    [predictionHistory],
  );

  const ledSeconds = session.totalEnergy > 0 ? (session.totalEnergy / 0.005) * 3600 : 0;
  const carMeters = session.totalCarbon > 0 ? (session.totalCarbon / 0.21) * 1000 : 0;
  const treeMinutes =
    session.totalCarbon > 0
      ? session.totalCarbon / (21 / (365 * 24 * 60))
      : 0;
  const ecoScore = Math.min(100, Math.max(0, 100 - session.totalCarbon * 10000));

  const comparisons = [
    {
      icon: "🔦",
      label: `Equivalent to keeping an LED on for ${ledSeconds.toFixed(1)} seconds`,
      width: `${clamp(ledSeconds / 120, 0.08, 1) * 100}%`,
    },
    {
      icon: "🚗",
      label: `Equivalent to driving ${carMeters.toFixed(1)} meters in a car`,
      width: `${clamp(carMeters / 250, 0.08, 1) * 100}%`,
    },
    {
      icon: "🌳",
      label: `Carbon absorbed by a tree in ${treeMinutes.toFixed(1)} minutes`,
      width: `${clamp(treeMinutes / 240, 0.08, 1) * 100}%`,
    },
  ];

  const exportSessionReport = () => {
    const payload = {
      generatedAt: new Date().toISOString(),
      session,
      predictionHistory,
      nasExperiments: ui.nasExperiments,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "pulseledger-session-report.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  const triggerNasRun = async () => {
    setNasState({ nasRunning: true });
    await new Promise((resolve) => window.setTimeout(resolve, 2200));
    setNasState({
      nasRunning: false,
      nasExperiments: [
        {
          id: "nas-1",
          experiment: "run_nas",
          dataset: "Bank",
          architecture: "CarbonAwareMLP-48x24",
          valLoss: 0.318,
          carbonUsed: 0.0021,
          status: "simulated",
        },
        {
          id: "nas-2",
          experiment: "run_nas_german",
          dataset: "German",
          architecture: "PrecisionTunedMLP-64x32",
          valLoss: 0.287,
          carbonUsed: 0.0018,
          status: "simulated",
        },
      ],
    });
  };

  const metricCards = [
    {
      label: "Total Predictions",
      value: session.predictionCount,
      formatter: (value: number) => value.toFixed(0),
      color: "rgb(var(--color-accent))",
      border: "border-t-accent",
      glow: "drop-shadow-[0_0_12px_rgba(59,130,246,0.4)] text-accent",
      values: predictionHistory
        .slice(0, 5)
        .reverse()
        .map((_, index) => index + 1),
    },
    {
      label: "Total Energy",
      value: session.totalEnergy,
      formatter: formatEnergy,
      color: "rgb(var(--color-success))",
      border: "border-t-success",
      glow: "drop-shadow-[0_0_12px_rgba(34,197,94,0.4)] text-success",
      values: predictionHistory
        .slice(0, 5)
        .reverse()
        .map((item) => item.sustainability_metrics?.energy_kwh || 0),
    },
    {
      label: "Total Carbon",
      value: session.totalCarbon,
      formatter: formatCarbon,
      color: "rgb(var(--color-success))",
      border: "border-t-success",
      glow: "drop-shadow-[0_0_12px_rgba(34,197,94,0.4)] text-success",
      values: predictionHistory
        .slice(0, 5)
        .reverse()
        .map((item) => item.sustainability_metrics?.carbon_emissions || 0),
    },
    {
      label: "Avg Duration",
      value:
        session.predictionCount > 0
          ? session.totalDuration / session.predictionCount
          : 0,
      formatter: formatSeconds,
      color: "rgb(var(--color-warning))",
      border: "border-t-warning",
      glow: "drop-shadow-[0_0_12px_rgba(245,158,11,0.4)] text-warning",
      values: predictionHistory
        .slice(0, 5)
        .reverse()
        .map((item) => item.sustainability_metrics?.duration_seconds || 0),
    },
  ];

  return (
    <div className="page-frame relative space-y-6">
      {/* Background glow */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[360px] bg-[radial-gradient(ellipse_70%_50%_at_50%_0%,rgba(34,197,94,0.05)_0%,transparent_60%)]" />

      {/* ─── Hero ─── */}
      <section className="animate-enter-1 z-10 overflow-hidden rounded-lg border border-border bg-bg-surface p-6 shadow-sm md:p-8">
        <p className="section-kicker !text-success">Carbon-Aware AI Operations</p>
        <h1 className="mt-2 max-w-3xl font-display text-3xl font-bold text-text-primary md:text-4xl lg:text-5xl">
          Sustainability is a live operating signal across every session.
        </h1>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-text-secondary font-medium">
          Every tracked prediction updates the session dashboard, making energy,
          carbon, and latency visible alongside model outcomes.
        </p>
      </section>

      {/* ─── Metric Cards ─── */}
      <section className="animate-enter-2 relative z-10 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {metricCards.map((item) => (
          <div
            key={item.label}
            className={`rounded-lg border border-border bg-bg-surface border-t-2 p-5 shadow-sm ${item.border}`}
          >
            <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
              {item.label}
            </p>
            <AnimatedNumber
              value={item.value}
              formatter={item.formatter}
              className={`mt-4 block font-mono text-2xl font-bold text-text-primary ${item.glow}`}
            />
            <Sparkline values={item.values} color={item.color} />
          </div>
        ))}
      </section>

      {/* Terminal prompt if no predictions */}
      {session.predictionCount === 0 ? (
        <div className="animate-enter-2 terminal-panel relative z-10 rounded-lg p-5">
          <p className="font-mono text-sm leading-relaxed text-success">
            &gt; waiting for first tracked prediction...
            <br />
            &gt; run analysis from Studio with track_sustainability enabled
            <br />
            &gt; carbon telemetry will populate here
            <span className="cursor-blink">_</span>
          </p>
        </div>
      ) : null}

      {/* ─── Carbon Trend Chart ─── */}
      <section className="animate-enter-3 relative z-10 rounded-lg border border-border bg-bg-surface p-6 shadow-sm">
        <div className="mb-6">
          <p className="section-kicker !text-success">Carbon Trend</p>
          <h2 className="mt-1 font-display text-xl font-bold text-text-primary">
            Emissions per prediction across this session
          </h2>
        </div>
        {chartData.length > 0 ? (
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="carbonFill" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="rgb(var(--color-success))" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="rgb(var(--color-success))" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="prediction"
                  tick={{ fill: "#8c8a82", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "#8c8a82", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgb(var(--color-bg-elevated))",
                    border: "1px solid rgba(34,197,94,0.18)",
                    borderRadius: "8px",
                    color: "rgb(var(--color-text-primary))",
                    fontSize: "12px",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="carbon"
                  stroke="rgb(var(--color-success))"
                  fill="url(#carbonFill)"
                  strokeWidth={2.5}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="terminal-panel rounded-md p-5">
            <p className="font-mono text-sm leading-relaxed text-success">
              &gt; waiting for carbon telemetry...
              <br />
              &gt; this chart animates after the first tracked prediction
              <span className="cursor-blink">_</span>
            </p>
          </div>
        )}
      </section>

      {/* ─── Human Comparisons ─── */}
      <section className="animate-enter-3 relative z-10 rounded-lg border border-border bg-bg-surface p-6 shadow-sm">
        <div className="mb-6">
          <p className="section-kicker">What Your AI Consumed</p>
          <h2 className="mt-1 font-display text-xl font-bold text-text-primary">
            Human-scale comparisons for abstract carbon numbers.
          </h2>
        </div>
        <div className="space-y-4">
          {comparisons.map((item) => (
            <div
              key={item.label}
              className="rounded-md border border-border bg-bg-elevated p-4 shadow-inner"
            >
              <div className="flex items-center gap-3 text-sm font-medium text-text-primary">
                <span className="text-lg">{item.icon}</span>
                <span>{item.label}</span>
              </div>
              <div className="mt-4 h-1.5 rounded-full bg-border">
                <div
                  className="h-full rounded-full bg-success transition-all duration-1000 shadow-[0_0_8px_rgb(var(--color-success))]"
                  style={{ width: item.width }}
                />
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ─── NAS Experiments ─── */}
      <section className="animate-enter-3 relative z-10 rounded-lg border border-border bg-bg-surface p-6 shadow-sm">
        <div className="mb-6 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="section-kicker !text-success">Carbon-Aware NAS</p>
            <h2 className="mt-1 font-display text-xl font-bold text-text-primary">
              NAS experiment results
            </h2>
          </div>
          <button
            type="button"
            onClick={triggerNasRun}
            disabled={ui.nasRunning}
            className="button-ghost border-success/30 text-success hover:border-success/50 hover:bg-success/10 transition"
          >
            {ui.nasRunning ? (
              <span className="flex items-center gap-2">
                <span className="button-spinner" /> Triggering...
              </span>
            ) : (
              "Trigger NAS Run"
            )}
          </button>
        </div>

        {ui.nasExperiments.length === 0 ? (
          <div className="terminal-panel rounded-md p-5">
            <p className="font-mono text-sm text-success">
              No NAS runs recorded this session.
              <span className="cursor-blink ml-1 inline-block">_</span>
            </p>
          </div>
        ) : (
          <div className="overflow-hidden rounded-md border border-border">
            <table className="w-full border-collapse text-left text-sm">
              <thead className="bg-bg-elevated text-[10px] font-bold uppercase tracking-wider text-text-muted">
                <tr>
                  <th className="px-5 py-4">Experiment</th>
                  <th className="px-5 py-4">Dataset</th>
                  <th className="px-5 py-4">Architecture</th>
                  <th className="px-5 py-4">Val Loss</th>
                  <th className="px-5 py-4">Carbon</th>
                  <th className="px-5 py-4">Status</th>
                </tr>
              </thead>
              <tbody>
                {ui.nasExperiments.map((experiment, idx) => (
                  <tr
                    key={experiment.id}
                    className={cn(
                      "border-t border-border bg-bg-surface hover:bg-bg-elevated/40 transition-colors",
                    )}
                  >
                    <td className="px-5 py-4 font-mono font-medium text-text-primary">{experiment.experiment}</td>
                    <td className="px-5 py-4 text-text-secondary">{experiment.dataset}</td>
                    <td className="px-5 py-4 font-mono text-xs text-text-secondary">{experiment.architecture}</td>
                    <td className="px-5 py-4 font-mono font-bold text-text-primary">{experiment.valLoss.toFixed(3)}</td>
                    <td className="px-5 py-4 font-mono text-success drop-shadow-[0_0_8px_rgba(34,197,94,0.4)]">{formatCarbon(experiment.carbonUsed)}</td>
                    <td className="px-5 py-4 font-bold uppercase tracking-wider text-[10px] text-success">
                      {experiment.status}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* ─── Eco Gauge ─── */}
      <section className="animate-enter-3 relative z-10 rounded-lg border border-border bg-bg-surface p-6 shadow-sm md:p-8">
        <div className="grid gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:items-center">
          <EcoGauge score={ecoScore} />
          <div>
            <p className="section-kicker !text-success">Session Efficiency Verdict</p>
            <h2 className="mt-2 font-display text-3xl font-bold text-text-primary">
              {getContrastVerdict(ecoScore)}
            </h2>
            <p className="mt-4 max-w-xl text-sm leading-relaxed text-text-secondary font-medium">
              The eco score is computed from cumulative session carbon using the
              rule <code className="rounded-sm bg-border px-1.5 py-0.5 font-mono text-[11px] text-text-primary shadow-inner">100 - (carbon * 10000)</code> and
              capped between 0 and 100.
            </p>
            <button type="button" onClick={exportSessionReport} className="button-primary mt-6">
              Download Session Report
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

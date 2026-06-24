"use client";

import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AnimatedNumber } from "@/components/animated-number";
import { Sparkline } from "@/components/sparkline";
import { StateCard } from "@/components/state-card";
import {
  formatCarbon,
  formatEnergy,
  formatMethodLabel,
  formatSeconds,
  getContrastVerdict,
} from "@/lib/format";
import { clamp, cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";
import { toast } from "@/store/use-toast-store";

const sustainabilityTabs = ["Telemetry", "Experiments"] as const;
type Tab = (typeof sustainabilityTabs)[number];

function EcoGauge({ score }: { score: number }) {
  const normalized = clamp(score, 0, 100);
  const radius = 82;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (normalized / 100) * circumference;

  return (
    <div className="relative mx-auto h-[200px] w-[200px]">
      <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90">
        <circle cx="100" cy="100" r={radius} stroke="rgb(var(--color-border))" strokeWidth="10" fill="none" />
        <circle
          cx="100"
          cy="100"
          r={radius}
          stroke="rgb(var(--color-accent))"
          strokeWidth="10"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-1000"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <p className="section-kicker">Eco Score</p>
        <AnimatedNumber
          value={normalized}
          formatter={(value) => value.toFixed(0)}
          className="mt-1 font-display text-5xl font-medium text-text-primary tabular"
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
  const [tab, setTab] = useState<Tab>("Telemetry");

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
    session.totalCarbon > 0 ? session.totalCarbon / (21 / (365 * 24 * 60)) : 0;
  const ecoScore = Math.min(100, Math.max(0, 100 - session.totalCarbon * 10000));

  // The most recent tracked prediction tells us how figures were derived.
  const latestMetrics = predictionHistory.find(
    (item) => item.sustainability_metrics,
  )?.sustainability_metrics;
  const method = latestMetrics?.method;
  const region = latestMetrics?.region ?? "US";
  const gridFactor = latestMetrics?.emissions_factor_kg_per_kwh ?? 0.385;

  // Two significant figures so genuinely-small measured values stay legible
  // (e.g. "0.0021 meters") instead of collapsing to a misleading "0.0".
  const sig = (value: number) =>
    value === 0
      ? "0"
      : value >= 1
        ? value.toFixed(1)
        : value.toPrecision(2).replace(/\.?0+$/, "");

  const comparisons = [
    { label: `Keeping an LED lit for ${sig(ledSeconds)} seconds`, width: `${clamp(ledSeconds / 120, 0.08, 1) * 100}%` },
    { label: `Driving ${sig(carMeters)} meters in a car`, width: `${clamp(carMeters / 250, 0.08, 1) * 100}%` },
    { label: `A tree's carbon uptake over ${sig(treeMinutes)} minutes`, width: `${clamp(treeMinutes / 240, 0.08, 1) * 100}%` },
  ];

  const exportSessionReport = () => {
    const payload = {
      generatedAt: new Date().toISOString(),
      session,
      predictionHistory,
      nasExperiments: ui.nasExperiments,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "pulseledger-session-report.json";
    link.click();
    URL.revokeObjectURL(url);
    toast.success("Report downloaded", "pulseledger-session-report.json");
  };

  const triggerNasRun = async () => {
    setNasState({ nasRunning: true });
    await new Promise((resolve) => window.setTimeout(resolve, 2200));
    setNasState({
      nasRunning: false,
      nasExperiments: [
        { id: "nas-1", experiment: "run_nas", dataset: "Bank", architecture: "CarbonAwareMLP-48x24", valLoss: 0.318, carbonUsed: 0.0021, status: "simulated" },
        { id: "nas-2", experiment: "run_nas_german", dataset: "German", architecture: "PrecisionTunedMLP-64x32", valLoss: 0.287, carbonUsed: 0.0018, status: "simulated" },
      ],
    });
  };

  const metricCards = [
    { label: "Predictions", value: session.predictionCount, formatter: (v: number) => v.toFixed(0), color: "rgb(var(--color-accent))", values: predictionHistory.slice(0, 5).reverse().map((_, index) => index + 1) },
    { label: "Total Energy", value: session.totalEnergy, formatter: formatEnergy, color: "rgb(var(--color-success))", values: predictionHistory.slice(0, 5).reverse().map((item) => item.sustainability_metrics?.energy_kwh || 0) },
    { label: "Total Carbon", value: session.totalCarbon, formatter: formatCarbon, color: "rgb(var(--color-success))", values: predictionHistory.slice(0, 5).reverse().map((item) => item.sustainability_metrics?.carbon_emissions || 0) },
    { label: "Avg Duration", value: session.predictionCount > 0 ? session.totalDuration / session.predictionCount : 0, formatter: formatSeconds, color: "rgb(var(--color-warning))", values: predictionHistory.slice(0, 5).reverse().map((item) => item.sustainability_metrics?.duration_seconds || 0) },
  ];

  return (
    <div className="space-y-7">
      <p className="max-w-2xl text-[15px] leading-relaxed text-text-secondary">
        Every decision, booked against its footprint. Energy, carbon, and latency
        accrue across this session as you score applications with tracking on.
      </p>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-border">
        {sustainabilityTabs.map((item) => (
          <button
            key={item}
            type="button"
            onClick={() => setTab(item)}
            className={cn(
              "focus-ring -mb-px rounded-t-[3px] border-b-2 px-4 py-2.5 text-sm font-medium transition-colors",
              tab === item
                ? "border-accent text-text-primary"
                : "border-transparent text-text-muted hover:text-text-primary",
            )}
          >
            {item}
          </button>
        ))}
      </div>

      {tab === "Telemetry" ? (
        <div className="space-y-6">
          <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {metricCards.map((item) => (
              <div key={item.label} className="leaf p-5">
                <p className="section-kicker">{item.label}</p>
                <AnimatedNumber
                  value={item.value}
                  formatter={item.formatter}
                  className="mt-3 block font-mono text-2xl font-medium text-text-primary tabular"
                />
                <Sparkline values={item.values} color={item.color} />
              </div>
            ))}
          </section>

          {session.predictionCount === 0 ? (
            <StateCard
              title="No telemetry yet"
              message="Run a prediction in a new assessment with sustainability tracking enabled — energy and carbon will accrue here."
            />
          ) : null}

          <section className="leaf p-6">
            <p className="section-kicker">Carbon Trend</p>
            <h2 className="mt-1 font-display text-xl font-medium text-text-primary">
              Emissions per prediction, this session
            </h2>
            {chartData.length > 0 ? (
              <div className="mt-6 h-[280px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
                    <defs>
                      <linearGradient id="carbonFill" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stopColor="rgb(var(--color-success))" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="rgb(var(--color-success))" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--color-border))" vertical={false} />
                    <XAxis dataKey="prediction" tick={{ fill: "rgb(var(--color-text-muted))", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "rgb(var(--color-text-muted))", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={{ background: "rgb(var(--color-bg-surface))", border: "1px solid rgb(var(--color-border-strong))", borderRadius: "4px", color: "rgb(var(--color-text-primary))", fontSize: "12px" }}
                      formatter={(value: number) => [formatCarbon(value), "Carbon"]}
                      labelFormatter={(label) => `Prediction ${label}`}
                    />
                    <Area type="monotone" dataKey="carbon" stroke="rgb(var(--color-success))" fill="url(#carbonFill)" strokeWidth={2} isAnimationActive={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="mt-6">
                <StateCard title="Awaiting carbon telemetry" message="This chart draws in after the first tracked prediction." />
              </div>
            )}
          </section>

          <div className="grid gap-6 lg:grid-cols-2">
            <section className="leaf p-6">
              <p className="section-kicker">What Your AI Consumed</p>
              <h2 className="mt-1 font-display text-xl font-medium text-text-primary">
                Human-scale comparisons
              </h2>
              <div className="mt-6 space-y-4">
                {comparisons.map((item) => (
                  <div key={item.label}>
                    <p className="text-sm text-text-secondary">{item.label}</p>
                    <div className="mt-2 h-1 bg-bg-elevated">
                      <div className="h-full bg-success transition-all duration-1000" style={{ width: item.width }} />
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="leaf p-6">
              <div className="flex flex-col items-center gap-5 text-center">
                <EcoGauge score={ecoScore} />
                <div>
                  <p className="section-kicker">Efficiency Verdict</p>
                  <h2 className="mt-1 font-display text-2xl font-medium text-text-primary">
                    {getContrastVerdict(ecoScore)}
                  </h2>
                </div>
                <button type="button" onClick={exportSessionReport} className="button-ghost">
                  Download session report
                </button>
              </div>
            </section>
          </div>

          <section className="leaf p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="section-kicker">Methodology</p>
                <h2 className="mt-1 font-display text-xl font-medium text-text-primary">
                  How these figures are measured
                </h2>
              </div>
              <span
                className="rounded-[3px] border border-border bg-bg-elevated px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-text-secondary"
                title="Measurement method for the most recent tracked prediction"
              >
                {formatMethodLabel(method)}
              </span>
            </div>

            <div className="mt-5 grid gap-4 sm:grid-cols-3">
              <div className="rounded-[3px] border border-border bg-bg-elevated/50 p-4">
                <p className="section-kicker">Energy</p>
                <p className="mt-2 text-sm text-text-secondary">
                  Process CPU-time integrated against an effective per-core
                  power draw, plus a platform baseline — or hardware power via
                  CodeCarbon when enabled.
                </p>
              </div>
              <div className="rounded-[3px] border border-border bg-bg-elevated/50 p-4">
                <p className="section-kicker">Carbon</p>
                <p className="mt-2 text-sm text-text-secondary">
                  Energy × grid intensity for{" "}
                  <span className="font-mono text-text-primary">{region}</span>{" "}
                  (
                  <span className="font-mono text-text-primary">
                    {gridFactor.toFixed(3)}
                  </span>{" "}
                  kg CO₂/kWh).
                </p>
              </div>
              <div className="rounded-[3px] border border-border bg-bg-elevated/50 p-4">
                <p className="section-kicker">Scope</p>
                <p className="mt-2 text-sm text-text-secondary">
                  A single scoring call is near-negligible; the material
                  footprint of ML is in training — see the Experiments tab.
                </p>
              </div>
            </div>

            <p className="mt-4 text-xs leading-relaxed text-text-muted">
              Measurement falls back gracefully: CodeCarbon (opt-in, reads
              CPU/GPU/RAM power) → CPU-time estimate (default) → wall-clock
              estimate. The badge above shows which method produced the current
              figures. CPU-time is a defensible estimate, not a certified
              meter; calibrate via{" "}
              <span className="font-mono text-text-secondary">
                PULSELEDGER_CPU_CORE_WATTS
              </span>{" "}
              and{" "}
              <span className="font-mono text-text-secondary">
                PULSELEDGER_GRID_REGION
              </span>
              .
            </p>
          </section>
        </div>
      ) : null}

      {tab === "Experiments" ? (
        <section className="leaf p-6">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <p className="section-kicker">Carbon-Aware NAS</p>
                <span className="rounded-[3px] border border-border bg-bg-elevated px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-text-muted">
                  Preview
                </span>
              </div>
              <h2 className="mt-1 font-display text-xl font-medium text-text-primary">
                Architecture search results
              </h2>
            </div>
            <button type="button" onClick={triggerNasRun} disabled={ui.nasRunning} className="button-ghost">
              {ui.nasRunning ? (
                <>
                  <span className="button-spinner" /> Searching…
                </>
              ) : (
                "Trigger NAS run"
              )}
            </button>
          </div>

          {ui.nasExperiments.length === 0 ? (
            <div className="mt-5">
              <StateCard
                title="No NAS runs this session"
                message="Trigger a run to preview carbon-aware architecture results. (Illustrative — wire to app.sustainability.run_nas for live search.)"
              />
            </div>
          ) : (
            <div className="mt-5 overflow-x-auto">
              <table className="ledger-table">
                <thead>
                  <tr>
                    <th>Experiment</th>
                    <th>Dataset</th>
                    <th>Architecture</th>
                    <th>Val Loss</th>
                    <th>Carbon</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {ui.nasExperiments.map((experiment) => (
                    <tr key={experiment.id} className="fade-in-row">
                      <td className="font-mono text-text-primary">{experiment.experiment}</td>
                      <td className="text-text-secondary">{experiment.dataset}</td>
                      <td className="font-mono text-xs text-text-secondary">{experiment.architecture}</td>
                      <td className="font-mono font-medium text-text-primary tabular">{experiment.valLoss.toFixed(3)}</td>
                      <td className="font-mono text-success tabular">{formatCarbon(experiment.carbonUsed)}</td>
                      <td className="font-mono text-[10px] uppercase tracking-wider text-text-muted">{experiment.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      ) : null}
    </div>
  );
}

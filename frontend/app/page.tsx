"use client";

import Link from "next/link";
import { ArrowRight, Zap, Shield, Leaf } from "lucide-react";

import { AnimatedNumber } from "@/components/animated-number";
import { Reveal } from "@/components/reveal";
import { StatusPill } from "@/components/status-pill";
import { TiltCard } from "@/components/tilt-card";
import { pillarCards } from "@/lib/constants";
import { formatCarbon, formatEnergy } from "@/lib/format";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";

const pillarIcons: Record<string, typeof Zap> = {
  Explainability: Zap,
  "Federated Learning": Shield,
  "Carbon-Aware AI": Leaf,
};

const accentMap: Record<string, { border: string; text: string; bg: string; shadow: string }> = {
  amber: {
    border: "border-warning/30",
    text: "text-warning",
    bg: "bg-warning/10",
    shadow: "shadow-[0_0_24px_rgb(var(--color-warning)/0.15)]",
  },
  blue: {
    border: "border-accent/30",
    text: "text-accent",
    bg: "bg-accent/10",
    shadow: "shadow-[0_0_24px_rgb(var(--color-accent)/0.15)]",
  },
  green: {
    border: "border-success/30",
    text: "text-success",
    bg: "bg-success/10",
    shadow: "shadow-[0_0_24px_rgb(var(--color-success)/0.15)]",
  },
};

export default function LandingPage() {
  const backendStatus = usePulseStore((state) => state.backendStatus);

  return (
    <div className="page-frame space-y-8">
      {/* ─── Hero ─── */}
      <section className="animate-enter relative isolate min-h-[540px] overflow-hidden rounded-lg border border-border bg-bg-surface">
        <div className="absolute inset-0 bg-gradient-to-b from-bg-surface to-bg-primary" />

        <div className="relative z-10 flex min-h-[540px] flex-col justify-between gap-10 p-6 md:p-10">
          <div className="max-w-3xl space-y-6 pt-10 md:pt-14">
            <div className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accent/10 px-3.5 py-1.5 shadow-[0_0_12px_rgb(var(--color-accent)/0.2)]">
              <div className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse" />
              <span className="text-[11px] font-bold uppercase tracking-wider text-accent/90">
                Live Intelligence Platform
              </span>
            </div>

            <h1 className="font-display text-4xl font-bold leading-[1.08] tracking-tight text-text-primary md:text-6xl lg:text-7xl">
              Credit Risk Intelligence.{" "}
              <span className="bg-gradient-to-r from-accent via-accent-hover to-text-primary bg-clip-text text-transparent">
                Sustainably Explained.
              </span>
            </h1>

            <p className="max-w-xl text-base leading-relaxed text-text-muted font-medium">
              SHAP-powered inference, federated learning simulation, and
              carbon-aware AI telemetry — unified in a scalable, production-grade command center.
            </p>

            <div className="flex flex-col gap-3 sm:flex-row pt-2">
              <Link href="/studio" className="button-primary h-12 px-6 text-sm">
                Launch Studio
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/sustainability"
                className="button-ghost border-success/30 text-success hover:border-success/50 hover:bg-success/10 h-12 px-6 text-sm"
              >
                <Leaf className="h-4 w-4" />
                Sustainability Report
              </Link>
            </div>
          </div>

          {/* Status Pills */}
          <div className="grid gap-4 md:grid-cols-3">
            <StatusPill
              label="Main API"
              detail={backendStatus.main.detail}
              state={backendStatus.main.state}
            />
            <StatusPill
              label="Inference Engine"
              detail={backendStatus.inference.detail}
              state={backendStatus.inference.state}
            />
            <StatusPill
              label="Sustainability Monitor"
              detail={
                backendStatus.features.includes("sustainability-monitoring")
                  ? "Sustainability features active."
                  : "Awaiting status confirmation from the main API."
              }
              state={
                backendStatus.features.includes("sustainability-monitoring")
                  ? "healthy"
                  : backendStatus.overall.state
              }
            />
          </div>
        </div>
      </section>

      {/* ─── Three Pillars ─── */}
      <Reveal className="space-y-6">
        <div>
          <p className="section-kicker">Core Systems</p>
          <h2 className="mt-2 font-display text-2xl font-bold text-text-primary md:text-3xl">
            Built strictly on our three operational layers.
          </h2>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {pillarCards.map((card) => {
            const accent = accentMap[card.accent] || accentMap.amber;
            const Icon = pillarIcons[card.title] || Zap;

            return (
              <TiltCard
                key={card.title}
                className={cn(
                  "rounded-lg border p-6 bg-bg-surface transition-colors hover:bg-bg-elevated/40",
                  accent.border,
                  accent.shadow,
                )}
              >
                <div className={cn(
                  "mb-6 flex h-10 w-10 items-center justify-center rounded-md border border-border/50",
                  accent.bg,
                )}>
                  <Icon className={cn("h-5 w-5", accent.text)} />
                </div>
                <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                  {card.eyebrow}
                </p>
                <h3 className="mt-2 font-display text-lg font-bold text-text-primary">
                  {card.title}
                </h3>
                <p className="mt-3 text-sm leading-relaxed text-text-secondary">
                  {card.copy}
                </p>
              </TiltCard>
            );
          })}
        </div>
      </Reveal>

      {/* ─── Sustainability Strip ─── */}
      <Reveal>
        <section className="rounded-lg border border-border border-l-4 border-l-success bg-bg-surface p-6 shadow-sm">
          <div className="grid gap-6 md:grid-cols-[1.2fr_0.8fr] md:items-end">
            <div>
              <p className="section-kicker !text-success">Efficiency Built In</p>
              <h2 className="mt-2 font-display text-2xl font-bold text-text-primary md:text-3xl">
                Every payload ships with telemetry footprints.
              </h2>
              <p className="mt-3 max-w-lg text-sm leading-relaxed text-text-secondary">
                Carbon-aware tracking is integral. We log and report energy, emissions, and compute duration side-by-side with your predictive results.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-md border border-success/20 bg-success/10 p-4">
                <p className="text-[10px] font-bold uppercase tracking-wider text-success/80">
                  Energy
                </p>
                <AnimatedNumber
                  value={0.0004}
                  formatter={(value) => formatEnergy(value)}
                  className="mt-3 block text-xl font-bold text-success drop-shadow-[0_0_12px_rgba(34,197,94,0.4)]"
                  startOnView
                />
              </div>
              <div className="rounded-md border border-success/20 bg-success/10 p-4">
                <p className="text-[10px] font-bold uppercase tracking-wider text-success/80">
                  Carbon
                </p>
                <AnimatedNumber
                  value={0.0002}
                  formatter={(value) => formatCarbon(value)}
                  className="mt-3 block text-xl font-bold text-success drop-shadow-[0_0_12px_rgba(34,197,94,0.4)]"
                  startOnView
                />
              </div>
              <div className="rounded-md border border-success/20 bg-success/10 p-4">
                <p className="text-[10px] font-bold uppercase tracking-wider text-success/80">
                  Avg Inference
                </p>
                <AnimatedNumber
                  value={180}
                  formatter={(value) => `${value.toFixed(0)} ms`}
                  className="mt-3 block text-xl font-bold text-success drop-shadow-[0_0_12px_rgba(34,197,94,0.4)]"
                  startOnView
                />
              </div>
            </div>
          </div>
        </section>
      </Reveal>

      {/* ─── Studio Preview ─── */}
      <Reveal className="pb-8">
        <Link href="/studio" className="block focus-ring rounded-lg">
          <TiltCard className="group relative overflow-hidden rounded-lg border border-accent/30 bg-bg-surface p-6 shadow-[0_0_24px_rgb(var(--color-accent)/0.08)] hover:shadow-[0_0_32px_rgb(var(--color-accent)/0.15)] md:p-8 transition-shadow">
            <div className="max-w-lg">
              <p className="section-kicker !text-accent">Prediction Studio</p>
              <h2 className="mt-2 font-display text-2xl font-bold text-text-primary md:text-3xl">
                Real-time assessment arrays with fully actionable telemetry.
              </h2>
            </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
              <div className="rounded-md border border-border bg-bg-elevated p-5 shadow-sm">
                <div className="grid gap-4 md:grid-cols-[180px_minmax(0,1fr)]">
                  <div className="rounded-md border border-border bg-bg-primary p-4 shadow-inner">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                      Risk Score
                    </p>
                    <p className="mt-4 font-mono text-3xl font-bold text-accent drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]">
                      0.428
                    </p>
                    <div className="mt-3 h-1.5 rounded-full bg-bg-surface border border-border">
                      <div className="h-full w-[42%] rounded-full bg-accent shadow-[0_0_8px_rgb(var(--color-accent))]" />
                    </div>
                  </div>
                  <div className="rounded-md border border-border bg-bg-primary p-4 shadow-inner">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
                      Analyst Narrative
                    </p>
                    <p className="mt-3 text-sm leading-relaxed text-success/90 font-medium">
                      Moderate risk driven by debt burden and short employment
                      history, partially offset by verified income and stable
                      credit score trend.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex flex-col justify-between gap-4">
                <div className="rounded-md border border-success/30 bg-success/10 p-4">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-success/80">
                    Operational Footprint
                  </p>
                  <div className="mt-4 grid grid-cols-3 gap-2 text-sm font-mono font-bold text-success drop-shadow-[0_0_8px_rgba(34,197,94,0.4)]">
                    <span>0.0004 kWh</span>
                    <span>0.0002 kg CO₂</span>
                    <span>0.18 s</span>
                  </div>
                </div>
                <p className="text-sm leading-relaxed text-text-secondary mt-auto">
                  Run simulated predictions natively and track model confidence alongside dynamic environmental telemetry metrics.
                </p>
              </div>
            </div>
          </TiltCard>
        </Link>
      </Reveal>
    </div>
  );
}

"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { motion, type Variants } from "framer-motion";
import {
  ArrowRight,
  Cpu,
  Gauge,
  Leaf,
  Network,
  Scale,
  ShieldCheck,
} from "lucide-react";

import { AnimatedNumber } from "@/components/animated-number";
import { LandingHeroCanvas } from "@/components/landing-hero-canvas";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

const fadeUp: Variants = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] } },
};

const stagger: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.09 } },
};

function Reveal({
  children,
  className,
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  return (
    <motion.div
      variants={fadeUp}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-80px" }}
      transition={{ delay }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

const capabilities = [
  {
    icon: Gauge,
    eyebrow: "Explainability",
    title: "Every score, defended in plain language.",
    body: "SHAP attributions, counterfactuals, risk groups, and an analyst-grade narrative for each decision — so a number always comes with its reasons.",
    visual: "shap",
  },
  {
    icon: Network,
    eyebrow: "Federated Learning",
    title: "Train across institutions, never pool the data.",
    body: "A real FedAvg loop aggregates models from separate clients and reports genuine round-by-round validation loss and accuracy. Privacy stays intact.",
    visual: "federated",
  },
  {
    icon: Scale,
    eyebrow: "Fairness",
    title: "Audit the model, not just the applicant.",
    body: "Demographic parity, equalized odds, calibration and treatment equality measured across protected groups, with severity grading and remediation steps.",
    visual: "fairness",
  },
  {
    icon: Leaf,
    eyebrow: "Carbon-aware",
    title: "Book the footprint of every decision.",
    body: "Energy, emissions, and latency are tracked alongside predictions, so efficiency becomes an operating signal — not an afterthought.",
    visual: "carbon",
  },
];

const steps = [
  {
    n: "01",
    title: "Score",
    body: "Submit an application and get a calibrated risk score in well under a second.",
  },
  {
    n: "02",
    title: "Explain",
    body: "Read the SHAP attribution, the counterfactual to a lower band, and a written rationale.",
  },
  {
    n: "03",
    title: "Govern",
    body: "Audit the model for bias and track its carbon footprint before and after deployment.",
  },
];

const metrics = [
  { value: 0.18, suffix: "s", fixed: 2, label: "Median inference latency" },
  { value: 100, suffix: "%", fixed: 0, label: "Decisions shipped with reasons" },
  { value: 6, suffix: "", fixed: 0, label: "Fairness metrics audited" },
  { value: 4, suffix: "", fixed: 0, label: "Governance layers, one system" },
];

/* ── Small decorative visuals for the capability rows ── */
function ShapVisual() {
  const bars = [
    { w: 72, up: true },
    { w: 48, up: false },
    { w: 40, up: true },
    { w: 30, up: false },
    { w: 22, up: true },
  ];
  return (
    <div className="space-y-2.5">
      {bars.map((bar, i) => (
        <div key={i} className="flex items-center gap-3">
          <span className="h-2 w-2 rounded-[2px]" style={{ background: bar.up ? "rgb(var(--color-destructive))" : "rgb(var(--color-success))" }} />
          <div className="h-2 flex-1 overflow-hidden rounded-[2px] bg-bg-elevated">
            <motion.div
              initial={{ width: 0 }}
              whileInView={{ width: `${bar.w}%` }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] }}
              className="h-full"
              style={{ background: bar.up ? "rgb(var(--color-destructive))" : "rgb(var(--color-success))" }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function FederatedVisual() {
  const clients = [
    { x: 18, y: 22 },
    { x: 82, y: 20 },
    { x: 14, y: 74 },
    { x: 86, y: 76 },
  ];
  return (
    <svg viewBox="0 0 200 120" className="h-full w-full">
      {clients.map((c, i) => (
        <line key={i} x1="100" y1="60" x2={c.x * 2} y2={c.y} stroke="rgb(var(--color-border-strong))" strokeWidth="1" strokeDasharray="3 4" />
      ))}
      {clients.map((c, i) => (
        <g key={`n${i}`}>
          <circle cx={c.x * 2} cy={c.y} r="11" fill="rgb(var(--color-bg-elevated))" stroke="rgb(var(--color-border-strong))" />
          <text x={c.x * 2} y={c.y + 1} textAnchor="middle" dominantBaseline="middle" fontSize="7" fontFamily="ui-monospace, monospace" fill="rgb(var(--color-text-muted))">
            C{i + 1}
          </text>
        </g>
      ))}
      <circle cx="100" cy="60" r="16" fill="rgb(var(--color-accent))" />
      <text x="100" y="61" textAnchor="middle" dominantBaseline="middle" fontSize="8" fontFamily="ui-monospace, monospace" fill="rgb(var(--color-bg-surface))">
        AGG
      </text>
    </svg>
  );
}

function FairnessVisual() {
  const rows = [
    { label: "Parity", v: 28 },
    { label: "Eq. odds", v: 57 },
    { label: "Calibration", v: 18 },
  ];
  return (
    <div className="space-y-3">
      {rows.map((row, i) => (
        <div key={row.label}>
          <div className="flex justify-between font-mono text-[10px] uppercase tracking-wider text-text-muted">
            <span>{row.label}</span>
            <span>{(row.v / 100).toFixed(2)}</span>
          </div>
          <div className="mt-1 h-1.5 overflow-hidden rounded-[2px] bg-bg-elevated">
            <motion.div
              initial={{ width: 0 }}
              whileInView={{ width: `${row.v}%` }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: i * 0.1 }}
              className="h-full bg-destructive"
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function CarbonVisual() {
  return (
    <div className="flex items-center justify-center">
      <div className="text-center">
        <p className="section-kicker">Eco Score</p>
        <AnimatedNumber
          value={97}
          formatter={(v) => v.toFixed(0)}
          startOnView
          className="block font-display text-5xl font-medium text-accent tabular"
        />
        <p className="mt-1 font-mono text-[10px] uppercase tracking-wider text-success">Efficient</p>
      </div>
    </div>
  );
}

const visuals: Record<string, React.ReactNode> = {
  shap: <ShapVisual />,
  federated: <FederatedVisual />,
  fairness: <FairnessVisual />,
  carbon: <CarbonVisual />,
};

function Wordmark() {
  return (
    <span className="flex items-center gap-2.5">
      <span className="relative flex h-7 w-7 items-center justify-center rounded-[3px] bg-accent">
        <span className="absolute left-1.5 right-1.5 top-[8px] h-px bg-bg-surface/70" />
        <span className="absolute left-1.5 right-1.5 top-[13px] h-px bg-bg-surface/70" />
        <span className="absolute left-1.5 right-1.5 top-[18px] h-px bg-bg-surface/70" />
      </span>
      <span className="font-display text-lg font-semibold tracking-tight text-text-primary">
        PulseLedger
      </span>
    </span>
  );
}

export default function LandingPage() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="relative">
      {/* ── Nav ── */}
      <header
        className={cn(
          "fixed inset-x-0 top-0 z-50 transition-colors duration-300",
          scrolled
            ? "border-b border-border bg-bg-primary/85 backdrop-blur-md"
            : "border-b border-transparent",
        )}
      >
        <div className="mx-auto flex h-16 max-w-[1180px] items-center justify-between px-5 md:px-8">
          <Link href="/" className="focus-ring rounded-[3px]">
            <Wordmark />
          </Link>
          <nav className="hidden items-center gap-8 md:flex">
            {[
              ["Capabilities", "#capabilities"],
              ["Workflow", "#workflow"],
              ["Trust", "#trust"],
            ].map(([label, href]) => (
              <a
                key={href}
                href={href}
                className="focus-ring rounded-[3px] text-sm text-text-secondary transition-colors hover:text-text-primary"
              >
                {label}
              </a>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Link
              href="/login"
              className="focus-ring hidden rounded-[3px] px-3 py-1.5 text-sm text-text-secondary transition-colors hover:text-text-primary sm:block"
            >
              Sign in
            </Link>
            <Link href="/dashboard" className="button-primary">
              Launch app
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </header>

      {/* ── Hero ── */}
      <section className="relative overflow-hidden">
        <LandingHeroCanvas className="pointer-events-none absolute inset-0 h-full w-full opacity-70" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-bg-primary" />
        <div className="relative mx-auto grid max-w-[1180px] items-center gap-12 px-5 pb-20 pt-36 md:px-8 md:pt-44 lg:grid-cols-[1.05fr_0.95fr] lg:pb-28">
          <motion.div
            initial="hidden"
            animate="show"
            variants={stagger}
            className="max-w-xl"
          >
            <motion.div variants={fadeUp}>
              <span className="inline-flex items-center gap-2 rounded-full border border-border-strong bg-bg-surface px-3 py-1 font-mono text-[11px] uppercase tracking-[0.14em] text-text-muted">
                <span className="status-dot status-online" />
                Explainable credit-risk intelligence
              </span>
            </motion.div>
            <motion.h1
              variants={fadeUp}
              className="mt-6 font-display text-[2.9rem] font-medium leading-[1.03] tracking-tight text-text-primary md:text-[4.2rem]"
            >
              Credit risk,{" "}
              <span className="italic text-accent">fully accounted for.</span>
            </motion.h1>
            <motion.p
              variants={fadeUp}
              className="mt-6 max-w-md text-[16px] leading-relaxed text-text-secondary"
            >
              Score an application, read the reasoning behind every basis point,
              and keep the model honest on fairness and carbon — in one legible
              ledger, not a black box.
            </motion.p>
            <motion.div variants={fadeUp} className="mt-9 flex flex-wrap items-center gap-3">
              <Link href="/dashboard" className="button-primary h-11 px-5">
                Launch the workspace
                <ArrowRight className="h-4 w-4" />
              </Link>
              <a href="#workflow" className="button-ghost h-11 px-5">
                See how it works
              </a>
            </motion.div>
            <motion.div variants={fadeUp} className="mt-10 flex items-center gap-6 text-xs text-text-muted">
              <span className="flex items-center gap-1.5">
                <ShieldCheck className="h-4 w-4 text-success" /> SHAP-explained
              </span>
              <span className="flex items-center gap-1.5">
                <Network className="h-4 w-4 text-success" /> Federated
              </span>
              <span className="flex items-center gap-1.5">
                <Cpu className="h-4 w-4 text-success" /> Carbon-aware
              </span>
            </motion.div>
          </motion.div>

          {/* Live specimen */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="leaf overflow-hidden shadow-[0_24px_60px_-30px_rgba(26,23,20,0.4)]"
          >
            <div className="flex items-center justify-between border-b border-border px-5 py-3">
              <span className="section-kicker">Live Assessment</span>
              <span className="font-mono text-xs text-text-muted">APP-024601</span>
            </div>
            <div className="px-5 pt-6">
              <p className="section-kicker mb-1">Risk Score</p>
              <div className="flex items-end gap-4">
                <AnimatedNumber
                  value={0.42}
                  formatter={(v) => v.toFixed(2)}
                  startOnView
                  className="font-display text-6xl font-medium leading-none text-text-primary tabular"
                />
                <span className="mb-1.5 inline-flex items-center rounded-[3px] border border-warning/40 bg-warning/10 px-2.5 py-1 font-mono text-xs font-medium uppercase tracking-wider text-warning">
                  Medium band
                </span>
              </div>
            </div>
            <div className="px-5 pt-6">
              <div className="relative">
                <div className="flex h-1.5 overflow-hidden rounded-[2px]">
                  <span className="flex-1" style={{ background: "rgb(var(--color-success) / 0.32)" }} />
                  <span className="flex-1" style={{ background: "rgb(var(--color-warning) / 0.32)" }} />
                  <span className="flex-1" style={{ background: "rgb(var(--color-destructive) / 0.28)" }} />
                  <span className="flex-1" style={{ background: "rgb(var(--color-destructive) / 0.5)" }} />
                </div>
                <motion.span
                  initial={{ left: "0%" }}
                  animate={{ left: "42%" }}
                  transition={{ duration: 1, delay: 0.6, ease: [0.22, 1, 0.36, 1] }}
                  className="absolute -top-1 h-3.5 w-[3px] -translate-x-1/2 bg-text-primary"
                />
              </div>
              <div className="mt-2 flex justify-between font-mono text-[10px] uppercase tracking-wider text-text-muted">
                <span>Low</span>
                <span>Medium</span>
                <span>High</span>
                <span>Severe</span>
              </div>
            </div>
            <dl className="mt-6 divide-y divide-border border-t border-border">
              {[
                ["Top factor", "Debt-to-income · +0.18"],
                ["Counterfactual", "DTI 0.30 → 0.22 clears band"],
                ["Energy", "0.0004 kWh"],
                ["Latency", "180 ms"],
              ].map(([k, v]) => (
                <div key={k} className="flex items-center justify-between px-5 py-2.5">
                  <dt className="text-sm text-text-muted">{k}</dt>
                  <dd className="font-mono text-sm text-text-primary tabular">{v}</dd>
                </div>
              ))}
            </dl>
          </motion.div>
        </div>
      </section>

      {/* ── Trust strip ── */}
      <section id="trust" className="border-y border-border bg-bg-surface">
        <div className="mx-auto max-w-[1180px] px-5 py-8 md:px-8">
          <Reveal className="flex flex-col items-center gap-6 text-center">
            <p className="section-kicker">Built for risk, compliance &amp; ML teams</p>
            <div className="flex flex-wrap items-center justify-center gap-x-10 gap-y-4 font-display text-lg text-text-muted/70">
              <span>Lending</span>
              <span className="text-border-strong">·</span>
              <span>Neobanks</span>
              <span className="text-border-strong">·</span>
              <span>Credit Unions</span>
              <span className="text-border-strong">·</span>
              <span>Model Risk</span>
              <span className="text-border-strong">·</span>
              <span>Regulators</span>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ── Capabilities ── */}
      <section id="capabilities" className="mx-auto max-w-[1180px] scroll-mt-24 px-5 py-24 md:px-8">
        <Reveal className="max-w-2xl">
          <p className="section-kicker">Capabilities</p>
          <h2 className="mt-3 font-display text-3xl font-medium text-text-primary md:text-[2.6rem] md:leading-[1.1]">
            Four columns of trust, kept in one system.
          </h2>
        </Reveal>

        <div className="mt-16 space-y-20">
          {capabilities.map((cap, index) => (
            <Reveal key={cap.eyebrow}>
              <div
                className={cn(
                  "grid items-center gap-10 lg:grid-cols-2",
                  index % 2 === 1 && "lg:[&>*:first-child]:order-2",
                )}
              >
                <div>
                  <span className="flex h-11 w-11 items-center justify-center rounded-[4px] border border-border bg-bg-surface text-accent">
                    <cap.icon className="h-5 w-5" />
                  </span>
                  <p className="section-kicker mt-5">{cap.eyebrow}</p>
                  <h3 className="mt-2 max-w-md font-display text-2xl font-medium text-text-primary md:text-[1.9rem] md:leading-[1.15]">
                    {cap.title}
                  </h3>
                  <p className="mt-4 max-w-md text-[15px] leading-relaxed text-text-secondary">
                    {cap.body}
                  </p>
                </div>
                <div className="leaf flex min-h-[200px] items-center p-8">
                  <div className="w-full">{visuals[cap.visual]}</div>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ── Workflow ── */}
      <section id="workflow" className="scroll-mt-24 border-y border-border bg-bg-surface">
        <div className="mx-auto max-w-[1180px] px-5 py-24 md:px-8">
          <Reveal className="max-w-2xl">
            <p className="section-kicker">Workflow</p>
            <h2 className="mt-3 font-display text-3xl font-medium text-text-primary md:text-[2.6rem]">
              Score, explain, govern.
            </h2>
          </Reveal>
          <motion.div
            variants={stagger}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-80px" }}
            className="mt-14 grid gap-px overflow-hidden rounded-[4px] border border-border bg-border md:grid-cols-3"
          >
            {steps.map((step) => (
              <motion.div key={step.n} variants={fadeUp} className="bg-bg-primary p-8">
                <span className="index-num text-base">{step.n}</span>
                <h3 className="mt-4 font-display text-xl font-medium text-text-primary">
                  {step.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-text-secondary">
                  {step.body}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── Metrics ── */}
      <section className="mx-auto max-w-[1180px] px-5 py-24 md:px-8">
        <motion.div
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          className="grid gap-px overflow-hidden rounded-[4px] border border-border bg-border sm:grid-cols-2 lg:grid-cols-4"
        >
          {metrics.map((metric) => (
            <motion.div key={metric.label} variants={fadeUp} className="bg-bg-primary px-6 py-10 text-center">
              <AnimatedNumber
                value={metric.value}
                formatter={(v) => `${v.toFixed(metric.fixed)}${metric.suffix}`}
                startOnView
                className="block font-display text-4xl font-medium text-accent tabular md:text-5xl"
              />
              <p className="mt-3 text-sm text-text-secondary">{metric.label}</p>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ── Closing CTA ── */}
      <section className="mx-auto max-w-[1180px] px-5 pb-28 md:px-8">
        <Reveal>
          <div className="relative overflow-hidden rounded-[6px] bg-accent px-8 py-16 text-center md:py-20">
            <div className="relative">
              <h2 className="mx-auto max-w-2xl font-display text-3xl font-medium leading-[1.1] text-bg-surface md:text-[2.8rem]">
                Put a defensible number on every decision.
              </h2>
              <p className="mx-auto mt-4 max-w-md text-[15px] leading-relaxed text-bg-surface/75">
                Open the workspace and score your first application — explanation,
                fairness, and footprint included.
              </p>
              <Link
                href="/dashboard"
                className="focus-ring mt-8 inline-flex h-12 items-center gap-2 rounded-[3px] bg-bg-surface px-6 text-sm font-medium text-accent transition-transform hover:scale-[1.02] active:scale-100"
              >
                Launch the workspace
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </Reveal>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-border">
        <div className="mx-auto flex max-w-[1180px] flex-col gap-3 px-5 py-8 md:flex-row md:items-center md:justify-between md:px-8">
          <Wordmark />
          <p className="text-xs text-text-muted">
            Explainable credit risk · federated learning · fairness · carbon-aware
            operations.
          </p>
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-muted">
            SHAP · FedAvg · Fairness · Live
          </p>
        </div>
      </footer>
    </div>
  );
}

"use client";

import { gaugeColor } from "@/lib/format";

interface RiskGaugeProps {
  score: number;
  riskLevel: string;
  idle?: boolean;
}

const bands = [
  { label: "Low", weight: "0.32" },
  { label: "Medium", weight: "0.32", tone: "warning" },
  { label: "High", weight: "0.28", tone: "destructive" },
  { label: "Severe", weight: "0.5", tone: "destructive" },
];

/**
 * The risk instrument: a precise ink ruler rather than a glowing gauge.
 * A large serif score, a band chip, and a marker on a four-segment scale.
 */
export function RiskGauge({ score, riskLevel, idle }: RiskGaugeProps) {
  const normalized = idle ? 0 : Math.min(1, Math.max(0, score));
  const color = idle ? "rgb(var(--color-text-muted))" : gaugeColor(score);
  const displayScore = idle ? "—" : score.toFixed(2);
  const displayLevel = idle ? "Standby" : riskLevel.replaceAll("_", " ");

  return (
    <div className="leaf p-6">
      <div className="flex items-center justify-between">
        <span className="section-kicker">Risk Assessment</span>
        <span
          className="rounded-[3px] border px-2.5 py-1 font-mono text-[11px] font-medium uppercase tracking-wider"
          style={{
            color,
            borderColor: idle ? "rgb(var(--color-border))" : `${color}66`,
            background: idle ? "transparent" : `${color}14`,
          }}
        >
          {displayLevel}
        </span>
      </div>

      <div className="mt-5 flex items-end gap-4">
        <span
          className="font-display text-7xl font-medium leading-[0.9] tabular transition-colors duration-500"
          style={{ color: idle ? "rgb(var(--color-text-muted))" : "rgb(var(--color-text-primary))" }}
        >
          {displayScore}
        </span>
        <span className="pb-2 text-sm text-text-muted">
          on a 0–1 risk scale
        </span>
      </div>

      {/* Band ruler */}
      <div className="mt-7">
        <div className="relative">
          <div className="flex h-2 overflow-hidden rounded-[2px]">
            <span className="flex-1" style={{ background: "rgb(var(--color-success) / 0.32)" }} />
            <span className="flex-1" style={{ background: "rgb(var(--color-warning) / 0.32)" }} />
            <span className="flex-1" style={{ background: "rgb(var(--color-destructive) / 0.28)" }} />
            <span className="flex-1" style={{ background: "rgb(var(--color-destructive) / 0.5)" }} />
          </div>
          {!idle ? (
            <span
              className="absolute -top-1 h-4 w-[3px] -translate-x-1/2 rounded-[1px] transition-all duration-700 ease-out"
              style={{ left: `${normalized * 100}%`, background: "rgb(var(--color-text-primary))" }}
            />
          ) : null}
        </div>
        <div className="mt-2 flex justify-between font-mono text-[10px] uppercase tracking-wider text-text-muted">
          {bands.map((band) => (
            <span key={band.label}>{band.label}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

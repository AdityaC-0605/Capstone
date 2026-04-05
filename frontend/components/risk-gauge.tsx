"use client";

import { cn } from "@/lib/utils";
import { gaugeColor } from "@/lib/format";

interface RiskGaugeProps {
  score: number;
  riskLevel: string;
  idle?: boolean;
}

export function RiskGauge({ score, riskLevel, idle }: RiskGaugeProps) {
  const radius = 90;
  const circumference = Math.PI * radius; // semicircle
  const normalized = idle ? 0 : Math.min(1, Math.max(0, score));
  const offset = circumference - normalized * circumference;
  const color = idle ? "rgb(var(--color-border))" : gaugeColor(score);
  const displayScore = idle ? "—" : score.toFixed(3);
  const displayLevel = idle ? "idle" : riskLevel.replaceAll("_", " ");

  return (
    <div className="rounded-lg border border-border bg-bg-surface p-6 shadow-sm">
      <div className="flex flex-col items-center">
        <div className="relative h-[140px] w-[240px]">
          <svg viewBox="0 0 240 140" className="h-full w-full">
            {/* Background arc */}
            <path
              d="M 20 130 A 100 100 0 0 1 220 130"
              fill="none"
              stroke="rgb(var(--color-border))"
              strokeWidth="14"
              strokeLinecap="round"
            />
            {/* Filled arc */}
            <path
              d="M 20 130 A 100 100 0 0 1 220 130"
              fill="none"
              stroke={color}
              strokeWidth="14"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              className="transition-all duration-1000 ease-out"
              style={{
                filter: idle
                  ? "none"
                  : `drop-shadow(0 0 10px ${color}88)`,
              }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-end pb-1">
            <p
              className="font-mono text-3xl font-bold transition-colors duration-500"
              style={{ color: idle ? "rgb(var(--color-text-muted))" : color }}
            >
              {displayScore}
            </p>
            <p className="mt-1 text-[10px] font-bold uppercase tracking-wider text-text-muted">
              {displayLevel}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

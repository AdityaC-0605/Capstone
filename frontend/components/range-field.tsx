"use client";

import { cn } from "@/lib/utils";

interface RangeFieldProps {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  valueLabel: string;
  warning?: boolean;
  scoreTrack?: boolean;
  onChange: (value: number) => void;
}

export function RangeField({
  label,
  min,
  max,
  step = 1,
  value,
  valueLabel,
  warning,
  scoreTrack,
  onChange,
}: RangeFieldProps) {
  const percent = ((value - min) / (max - min)) * 100;

  const baseColor = warning
    ? "rgb(var(--color-destructive))"
    : scoreTrack
      ? "rgb(var(--color-warning))"
      : "rgb(var(--color-accent))";

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <p className="section-kicker">{label}</p>
        <p
          className={cn(
            "font-mono text-[13px] font-medium tabular",
            warning ? "text-destructive" : "text-text-primary",
          )}
        >
          {valueLabel}
        </p>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={cn(
          "range-track",
          scoreTrack && "score-track",
          warning && "warning-track",
        )}
        style={{
          background: `linear-gradient(90deg, ${baseColor} 0%, ${baseColor} ${percent}%, rgb(var(--color-bg-elevated)) ${percent}%, rgb(var(--color-bg-elevated)) 100%)`,
        }}
      />
    </div>
  );
}

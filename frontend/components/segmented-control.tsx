"use client";

import { cn } from "@/lib/utils";

interface SegmentedControlProps {
  label: string;
  options: Array<{ label: string; value: string }>;
  value: string;
  onChange: (value: string) => void;
  compact?: boolean;
}

export function SegmentedControl({
  label,
  options,
  value,
  onChange,
  compact,
}: SegmentedControlProps) {
  return (
    <div className="space-y-2">
      <p className="section-kicker">{label}</p>
      <div className="flex flex-wrap gap-1.5">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={cn(
              "focus-ring rounded-[3px] border px-3 py-1.5 text-xs font-medium transition-colors",
              compact && "px-2.5 py-1 text-[11px]",
              value === option.value
                ? "border-accent bg-accent text-bg-surface"
                : "border-border bg-bg-surface text-text-muted hover:border-border-strong hover:text-text-primary",
            )}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

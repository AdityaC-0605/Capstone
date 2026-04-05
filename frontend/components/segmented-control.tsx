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
    <div className="space-y-1.5">
      <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
        {label}
      </p>
      <div className="flex flex-wrap gap-1">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={cn(
              "focus-ring rounded-md border px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider transition-all duration-120 active:scale-[0.98]",
              compact && "text-[9px] px-2 py-1",
              value === option.value
                ? "border-accent bg-accent text-white shadow-sm"
                : "border-border bg-bg-surface text-text-muted hover:text-text-primary hover:bg-bg-elevated/80 hover:border-border-strong",
            )}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

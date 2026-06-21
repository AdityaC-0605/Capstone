"use client";

import { cn } from "@/lib/utils";

interface ToggleSwitchProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export function ToggleSwitch({ label, checked, onChange }: ToggleSwitchProps) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="focus-ring flex w-full items-center justify-between rounded-[3px] border border-border bg-bg-surface px-3 py-2.5 transition-colors hover:border-border-strong"
    >
      <span className="section-kicker">{label}</span>
      <div
        className={cn(
          "relative h-5 w-9 rounded-full transition-colors duration-200",
          checked ? "bg-accent" : "border border-border-strong bg-bg-elevated",
        )}
      >
        <span
          className={cn(
            "absolute top-0.5 h-4 w-4 rounded-full transition-transform duration-200",
            checked
              ? "translate-x-[18px] bg-bg-surface"
              : "translate-x-0.5 bg-border-strong",
          )}
        />
      </div>
    </button>
  );
}

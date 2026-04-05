"use client";

import { cn } from "@/lib/utils";
import { aggregateStatusColor } from "@/lib/format";

interface StatusPillProps {
  label: string;
  detail: string;
  state: string;
}

export function StatusPill({ label, detail, state }: StatusPillProps) {
  return (
    <div className="flex items-start gap-3 rounded-md border border-border bg-bg-surface px-4 py-3 shadow-sm transition-colors hover:bg-bg-elevated/50">
      <span className={cn("status-dot mt-1", aggregateStatusColor(state))} />
      <div className="min-w-0 flex-1">
        <p className="text-xs font-semibold text-text-primary">{label}</p>
        <p className="mt-0.5 truncate text-[11px] text-text-muted">{detail}</p>
      </div>
    </div>
  );
}

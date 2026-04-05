"use client";

import { cn } from "@/lib/utils";

interface StateCardProps {
  title: string;
  message: string;
  actionLabel?: string;
  onAction?: () => void;
  tone?: "default" | "loading" | "error";
}

export function StateCard({
  title,
  message,
  actionLabel,
  onAction,
  tone = "default",
}: StateCardProps) {
  return (
    <div
      className={cn(
        "rounded-md border p-4 shadow-sm",
        tone === "error"
          ? "border-destructive/30 bg-destructive/10"
          : tone === "loading"
            ? "border-accent/30 bg-accent/10"
            : "border-border bg-bg-surface",
      )}
    >
      <p
        className={cn(
          "text-xs font-bold uppercase tracking-wider",
          tone === "error"
            ? "text-destructive"
            : tone === "loading"
              ? "text-accent"
              : "text-text-muted",
        )}
      >
        {title}
      </p>
      <p className="mt-2 text-sm leading-6 text-text-secondary">{message}</p>
      {actionLabel && onAction ? (
        <button
          type="button"
          onClick={onAction}
          className="mt-3 focus-ring rounded-sm text-xs font-semibold text-accent hover:underline active:scale-[0.98] transition-transform"
        >
          {actionLabel}
        </button>
      ) : null}
    </div>
  );
}

"use client";

import { factorDirection, recommendationCopy } from "@/lib/format";
import { formatFeatureLabel } from "@/lib/format";
import type { PredictionTopFactor, Recommendation } from "@/lib/types";
import { cn } from "@/lib/utils";

interface FactorFlipCardProps {
  factor: PredictionTopFactor;
  recommendation?: Recommendation;
}

export function FactorFlipCard({ factor, recommendation }: FactorFlipCardProps) {
  const direction = factorDirection(factor);
  const isDecrease = direction === "decrease";
  const barWidth = `${Math.max(12, Math.min(100, Math.abs(factor.contribution || 0) * 100))}%`;

  return (
    <div className="factor-flip" tabIndex={0}>
      <div className="factor-flip-inner relative h-[140px]">
        {/* Front */}
        <div className="factor-face absolute inset-0 rounded-md border border-border bg-bg-surface p-4">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
              {formatFeatureLabel(factor.feature)}
            </p>
            <span
              className={cn(
                "rounded-md px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider",
                isDecrease
                  ? "bg-success/15 text-success"
                  : "bg-destructive/15 text-destructive",
              )}
            >
              {factor.magnitude || (isDecrease ? "decrease" : "increase")}
            </span>
          </div>
          <div className="mt-3 h-1.5 rounded-full bg-bg-elevated border border-border">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-700",
                isDecrease ? "bg-success" : "bg-destructive",
              )}
              style={{ width: barWidth }}
            />
          </div>
          <p className="mt-3 text-sm leading-5 text-text-secondary">
            {factor.description || factor.benchmark_context || "Hover to see recommendation."}
          </p>
        </div>

        {/* Back */}
        <div className="factor-face factor-face-back absolute inset-0 rounded-md border border-accent/30 bg-accent/10 p-4 shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]">
          <p className="text-[10px] font-bold uppercase tracking-wider text-accent">
            Recommendation
          </p>
          <p className="mt-2 text-sm leading-6 text-text-primary">
            {recommendationCopy(recommendation)}
          </p>
          {recommendation?.type && (
            <span
              className={cn(
                "mt-3 inline-block rounded-md px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider",
                recommendation.type === "preserve"
                  ? "bg-success/15 text-success"
                  : recommendation.type === "action_needed"
                    ? "bg-accent/15 text-accent"
                    : "bg-bg-elevated text-text-muted",
              )}
            >
              {recommendation.type.replaceAll("_", " ")}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

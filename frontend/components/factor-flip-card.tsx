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
        <div className="factor-face leaf absolute inset-0 p-4">
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-medium text-text-primary">
              {formatFeatureLabel(factor.feature)}
            </p>
            <span
              className={cn(
                "rounded-[3px] px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                isDecrease
                  ? "bg-success/12 text-success"
                  : "bg-destructive/12 text-destructive",
              )}
            >
              {factor.magnitude || (isDecrease ? "lowers" : "raises")}
            </span>
          </div>
          <div className="mt-3 h-1 bg-bg-elevated">
            <div
              className={cn(
                "h-full transition-all duration-700",
                isDecrease ? "bg-success" : "bg-destructive",
              )}
              style={{ width: barWidth }}
            />
          </div>
          <p className="mt-3 text-[13px] leading-5 text-text-secondary">
            {factor.description || factor.benchmark_context || "Hover for the recommended action."}
          </p>
        </div>

        {/* Back */}
        <div className="factor-face factor-face-back absolute inset-0 rounded-md border border-accent/40 bg-accent/8 p-4">
          <p className="section-kicker !text-accent">Recommendation</p>
          <p className="mt-2 text-[13px] leading-6 text-text-primary">
            {recommendationCopy(recommendation)}
          </p>
          {recommendation?.type && (
            <span
              className={cn(
                "mt-3 inline-block rounded-[3px] px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                recommendation.type === "preserve"
                  ? "bg-success/12 text-success"
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

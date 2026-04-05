import { riskBands } from "@/lib/constants";
import type {
  PredictionTopFactor,
  Recommendation,
  RiskGroup,
  RiskLevel,
  Sentiment,
} from "@/lib/types";

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const compactFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

export const formatCurrency = (value: number) => currencyFormatter.format(value);

export const formatCompact = (value: number) => compactFormatter.format(value);

export const formatPercent = (value: number, digits = 0) =>
  `${(value * 100).toFixed(digits)}%`;

export const formatRiskLabel = (value?: string) =>
  value ? value.replaceAll("_", " ").toUpperCase() : "UNKNOWN";

export const formatFeatureLabel = (value: string) =>
  value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (segment) => segment.toUpperCase());

export const formatSeconds = (value: number) =>
  value >= 1 ? `${value.toFixed(2)} s` : `${(value * 1000).toFixed(0)} ms`;

export const formatEnergy = (value: number) => `${value.toFixed(4)} kWh`;

export const formatCarbon = (value: number) => `${value.toFixed(4)} kg CO2`;

export const riskColorClasses = (risk?: RiskLevel) => {
  switch (risk) {
    case "low":
      return "bg-emerald/12 text-emerald border-emerald/35";
    case "medium":
      return "bg-gold/12 text-gold border-gold/35";
    case "high":
    case "very_high":
      return "bg-rose/12 text-rose border-rose/35";
    default:
      return "bg-white/4 text-copy border-white/8";
  }
};

export const gaugeColor = (score: number) => {
  if (score < 0.3) return "#34D399";
  if (score < 0.6) return "#F59E0B";
  return "#F43F5E";
};

export const aggregateStatusColor = (state: string) => {
  switch (state) {
    case "healthy":
      return "status-online";
    case "partial":
      return "status-partial";
    case "offline":
      return "status-offline";
    default:
      return "status-checking";
  }
};

export const factorDirection = (factor: PredictionTopFactor) => {
  if (factor.direction) {
    return factor.direction;
  }
  if (factor.impact?.includes("decrease")) {
    return "decrease";
  }
  if (factor.impact?.includes("increase")) {
    return "increase";
  }
  return (factor.contribution ?? 0) < 0 ? "decrease" : "increase";
};

export const factorBarColor = (factor: PredictionTopFactor) =>
  factorDirection(factor) === "decrease" ? "bg-emerald" : "bg-rose";

export const recommendationCopy = (recommendation?: Recommendation) => {
  if (!recommendation) {
    return "No linked recommendation was returned for this factor.";
  }
  return (
    recommendation.recommendation ||
    recommendation.advice ||
    recommendation.advisory ||
    "Review this factor with a credit analyst."
  );
};

export const sentimentColorClasses = (impact?: Sentiment) => {
  switch (impact) {
    case "risk_decrease":
      return "border-emerald/35 bg-emerald/8 text-emerald";
    case "risk_increase":
      return "border-rose/35 bg-rose/8 text-rose";
    default:
      return "border-gold/35 bg-gold/8 text-gold";
  }
};

export const narrativeForGroup = (key: string, group: RiskGroup) => {
  if (group.narrative) {
    return group.narrative;
  }

  const featureCount = group.features?.length ?? 0;
  const label = group.label || formatFeatureLabel(key);
  if (group.impact === "risk_decrease") {
    return `${label} is helping offset risk across ${featureCount} tracked signals.`;
  }
  if (group.impact === "risk_increase") {
    return `${label} is currently adding pressure to the decision profile.`;
  }
  return `${label} is presently balanced relative to the model baseline.`;
};

export const riskMarkerLeft = (score: number) =>
  `${Math.max(2, Math.min(98, score * 100))}%`;

export const riskBandLabel = (score: number) =>
  riskBands.find((band) => score <= band.max)?.label ?? "Very High";

export const getContrastVerdict = (ecoScore: number) => {
  if (ecoScore >= 85) return "EFFICIENT";
  if (ecoScore >= 60) return "MODERATE";
  return "REVIEW NEEDED";
};

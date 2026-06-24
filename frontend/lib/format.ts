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

export const formatCurrency = (value: number) => currencyFormatter.format(value);

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

// Measured per-inference energy/carbon are genuinely tiny, so scale the unit
// to keep small figures legible instead of rendering "0.0000".
export const formatEnergy = (value: number) => {
  if (value >= 1) return `${value.toFixed(3)} kWh`;
  if (value >= 1e-3) return `${(value * 1e3).toFixed(2)} Wh`;
  return `${(value * 1e6).toFixed(2)} mWh`;
};

export const formatCarbon = (value: number) => {
  if (value >= 1) return `${value.toFixed(3)} kg CO2`;
  if (value >= 1e-3) return `${(value * 1e3).toFixed(2)} g CO2`;
  return `${(value * 1e6).toFixed(2)} mg CO2`;
};

export const formatMethodLabel = (method?: string) => {
  switch (method) {
    case "codecarbon":
      return "Measured · CodeCarbon";
    case "cpu-time":
      return "Measured · CPU-time";
    case "wall-clock":
      return "Estimated · wall-clock";
    case "mock":
      return "Mock data";
    default:
      return "Unavailable";
  }
};

export const riskColorClasses = (risk?: RiskLevel) => {
  switch (risk) {
    case "low":
      return "bg-success/10 text-success border-success/35";
    case "medium":
      return "bg-warning/10 text-warning border-warning/35";
    case "high":
    case "very_high":
      return "bg-destructive/10 text-destructive border-destructive/35";
    default:
      return "bg-bg-elevated text-text-muted border-border";
  }
};

// Ledger palette: evergreen (low) → ochre (medium) → oxblood (high).
export const gaugeColor = (score: number) => {
  if (score < 0.3) return "#2E6B4F";
  if (score < 0.6) return "#B07A2C";
  return "#A3392A";
};

export const relativeTime = (iso: string) => {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
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

export const getContrastVerdict = (ecoScore: number) => {
  if (ecoScore >= 85) return "EFFICIENT";
  if (ecoScore >= 60) return "MODERATE";
  return "REVIEW NEEDED";
};

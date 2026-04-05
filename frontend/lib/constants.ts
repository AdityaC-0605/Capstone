import type {
  BackendConfig,
  BackendStatus,
  CreditApplication,
  HomeOwnership,
  LoanPurpose,
  RiskLevel,
  VerificationStatus,
} from "@/lib/types";

export const defaultBackendConfig: BackendConfig = {
  mainUrl: "http://localhost:8000",
  inferenceUrl: "http://localhost:8001",
  apiKey: "",
};

export const defaultApplication: CreditApplication = {
  age: 35,
  income: 65000,
  employment_length: 5,
  debt_to_income_ratio: 0.3,
  credit_score: 720,
  loan_amount: 25000,
  loan_purpose: "debt_consolidation",
  home_ownership: "rent",
  verification_status: "verified",
};

export const samplePredictionRequest = {
  application: defaultApplication,
  include_explanation: true,
  track_sustainability: true,
  explanation_type: "shap" as const,
};

export const loanPurposeOptions: Array<{
  label: string;
  value: LoanPurpose;
}> = [
  { label: "Debt Consolidation", value: "debt_consolidation" },
  { label: "Home Improvement", value: "home_improvement" },
  { label: "Major Purchase", value: "major_purchase" },
  { label: "Medical", value: "medical" },
  { label: "Vacation", value: "vacation" },
  { label: "Moving", value: "moving" },
  { label: "Wedding", value: "wedding" },
  { label: "Other", value: "other" },
];

export const homeOwnershipOptions: Array<{
  label: string;
  value: HomeOwnership;
}> = [
  { label: "Own", value: "own" },
  { label: "Rent", value: "rent" },
  { label: "Mortgage", value: "mortgage" },
  { label: "Other", value: "other" },
];

export const verificationOptions: Array<{
  label: string;
  value: VerificationStatus;
}> = [
  { label: "Verified", value: "verified" },
  { label: "Source Verified", value: "source_verified" },
  { label: "Not Verified", value: "not_verified" },
];

export const riskBands: Array<{
  level: RiskLevel;
  max: number;
  label: string;
}> = [
  { level: "low", max: 0.3, label: "Low" },
  { level: "medium", max: 0.6, label: "Medium" },
  { level: "high", max: 0.8, label: "High" },
  { level: "very_high", max: 1, label: "Very High" },
];

export const pillarCards = [
  {
    title: "Explainability",
    eyebrow: "SHAP Narrative",
    copy:
      "Turn a raw score into analyst-grade reasoning with top factors, recommendations, counterfactuals, and confidence context.",
    accent: "amber",
  },
  {
    title: "Federated Learning",
    eyebrow: "Distributed Training",
    copy:
      "Visualize client coordination, aggregation rounds, and convergence patterns without centralizing every sensitive record.",
    accent: "blue",
  },
  {
    title: "Carbon-Aware AI",
    eyebrow: "Identity Layer",
    copy:
      "Track energy, carbon, and experiment efficiency alongside credit decisions so sustainability becomes an operating signal.",
    accent: "green",
  },
];

export const defaultBackendStatus: BackendStatus = {
  main: {
    state: "checking",
    label: "Main API",
    detail: "Checking health and readiness.",
  },
  inference: {
    state: "checking",
    label: "Inference Engine",
    detail: "Checking health and auth path.",
  },
  overall: {
    state: "checking",
    label: "Connectivity",
    detail: "Probing backend services.",
  },
  features: [],
};

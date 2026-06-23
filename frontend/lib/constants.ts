import type {
  BackendConfig,
  BackendStatus,
  CreditApplication,
  HomeOwnership,
  LoanPurpose,
  VerificationStatus,
} from "@/lib/types";

export const defaultBackendConfig: BackendConfig = {
  // Configurable at build time for deployments; falls back to local dev.
  mainUrl: process.env.NEXT_PUBLIC_MAIN_URL || "http://localhost:8000",
  inferenceUrl:
    process.env.NEXT_PUBLIC_INFERENCE_URL || "http://localhost:8001",
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

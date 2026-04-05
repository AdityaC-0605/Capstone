export type RiskLevel = "low" | "medium" | "high" | "very_high";
export type ConfidenceLevel = "low" | "medium" | "high";
export type Sentiment = "risk_increase" | "risk_decrease" | "neutral";

export type LoanPurpose =
  | "debt_consolidation"
  | "home_improvement"
  | "major_purchase"
  | "medical"
  | "vacation"
  | "moving"
  | "wedding"
  | "other";

export type HomeOwnership = "own" | "rent" | "mortgage" | "other";
export type VerificationStatus =
  | "verified"
  | "source_verified"
  | "not_verified";

export interface CreditApplication {
  age: number;
  income: number;
  employment_length: number;
  debt_to_income_ratio: number;
  credit_score: number;
  loan_amount: number;
  loan_purpose: LoanPurpose;
  home_ownership: HomeOwnership;
  verification_status: VerificationStatus;
}

export interface PredictionRequest {
  application: CreditApplication;
  include_explanation: boolean;
  track_sustainability: boolean;
  explanation_type: "shap";
}

export interface PredictionTopFactor {
  feature: string;
  label?: string;
  value?: string | number;
  impact?: string;
  contribution?: number;
  magnitude?: "strong" | "moderate" | "minor";
  benchmark_context?: string;
  description?: string;
  direction?: "increase" | "decrease";
}

export interface Recommendation {
  feature?: string;
  type?: "action_needed" | "preserve" | "informational";
  recommendation?: string;
  advice?: string;
  advisory?: string;
}

export interface CounterfactualChange {
  current_value: string | number;
  suggested_target: string | number;
  action: string;
}

export interface RiskGroup {
  label?: string;
  impact?: Sentiment;
  total_contribution?: number;
  narrative?: string;
  features?: Array<string | { feature: string; contribution?: number }>;
}

export interface Methodology {
  method?: string;
  description?: string;
  interpretation?: string;
  baseline?: {
    description?: string;
    baseline_values?: Record<string, string | number>;
  };
}

export interface ExplanationPayload {
  prediction?: number;
  risk_level?: RiskLevel;
  risk_threshold_context?: string;
  risk_thresholds?: Array<{ level: RiskLevel; max_score: number }>;
  feature_importance?: Record<string, number>;
  top_factors?: PredictionTopFactor[];
  recommendations?: Recommendation[];
  counterfactual?: {
    needed?: boolean;
    message?: string;
    changes?: Record<string, CounterfactualChange>;
  };
  risk_groups?: Record<string, RiskGroup>;
  confidence?: {
    level?: ConfidenceLevel;
    score?: number;
    reason?: string;
  };
  methodology?: Methodology;
  summary?: string;
}

export interface SustainabilityMetrics {
  energy_kwh: number;
  carbon_emissions: number;
  duration_seconds: number;
}

export interface PredictionResponse {
  prediction_id: string;
  risk_score: number;
  risk_level: RiskLevel;
  confidence: number;
  model_version: string;
  prediction_timestamp: string;
  processing_time_ms: number;
  explanation?: ExplanationPayload;
  sustainability_metrics?: SustainabilityMetrics;
  status: string;
  message: string;
}

export interface BackendConfig {
  mainUrl: string;
  inferenceUrl: string;
  apiKey: string;
}

export interface ServiceStatus {
  state: "healthy" | "partial" | "offline" | "checking";
  label: string;
  detail: string;
}

export interface BackendStatus {
  main: ServiceStatus;
  inference: ServiceStatus;
  overall: ServiceStatus;
  features: string[];
  lastChecked?: string;
}

export interface PredictionRecord {
  timestamp: string;
  input: CreditApplication;
  result: PredictionResponse;
  sustainability_metrics?: SustainabilityMetrics;
}

export interface SessionSustainability {
  totalEnergy: number;
  totalCarbon: number;
  totalDuration: number;
  predictionCount: number;
}

export interface FederatedLossPoint {
  round: number;
  loss: number;
}

export interface FederatedState {
  running: boolean;
  rounds: number;
  lossHistory: FederatedLossPoint[];
  clients: number;
  localEpochs: number;
  completed: boolean;
  bestValLoss: number | null;
}

export interface NasExperiment {
  id: string;
  experiment: string;
  dataset: string;
  architecture: string;
  valLoss: number;
  carbonUsed: number;
  status: "queued" | "running" | "complete" | "simulated";
}

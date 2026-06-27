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
  method?: string;
  region?: string;
  emissions_factor_kg_per_kwh?: number;
  grid_source?: string;
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

export interface BatchSummary {
  total_applications: number;
  successful_predictions: number;
  failed_predictions: number;
  average_risk_score: number;
  risk_distribution: Record<string, number>;
}

export interface BatchPredictionResponse {
  batch_id: string;
  predictions: PredictionResponse[];
  batch_summary: BatchSummary;
  processing_time_ms: number;
  sustainability_metrics?: SustainabilityMetrics | null;
}

export interface BackendConfig {
  mainUrl: string;
  inferenceUrl: string;
  apiKey: string;
}

export interface ModelInfo {
  model_version: string;
  model_type: string;
  model_source: string;
  algorithm?: string | null;
  roc_auc?: number | null;
  trained_at?: string | null;
}

export interface AuthUser {
  id: number;
  email: string;
  full_name: string;
  role: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
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
  valAccuracy?: number;
  clientLoss?: number;
}

export interface FederatedState {
  running: boolean;
  rounds: number;
  lossHistory: FederatedLossPoint[];
  clients: number;
  localEpochs: number;
  completed: boolean;
  bestValLoss: number | null;
  bestRound?: number | null;
  wallTimeSeconds?: number | null;
  stoppedEarly?: boolean;
  source?: "live" | "simulated";
}

/** Raw round metric returned by the backend FedAvg simulation. */
export interface FederatedRoundMetric {
  round_number: number;
  participating_clients: number;
  average_client_loss: number;
  average_client_accuracy: number;
  average_val_loss: number;
  average_val_accuracy: number;
}

export interface FederatedRunResult {
  config: Record<string, number>;
  round_metrics: FederatedRoundMetric[];
  global_keys: string[];
  best_round: number;
  best_val_loss: number;
  stopped_early: boolean;
  best_model_path: string;
  wall_time_seconds: number;
  data_source?: string;
  dataset?: string;
}

export interface FederatedRunParams {
  number_of_clients: number;
  aggregation_rounds: number;
  local_epochs: number;
}

/** A single fairness metric outcome for one protected attribute. */
export interface FairnessFinding {
  protected_attribute?: string;
  fairness_metric?: string;
  metric_value?: number;
  threshold?: number;
  bias_level?: string;
  is_biased?: boolean;
}

export interface FairnessAttributeStat {
  tests_conducted: number;
  violations: number;
  violation_rate: number;
  worst_violation: string;
}

export interface FairnessMetricStat {
  tests_conducted: number;
  violations: number;
  violation_rate: number;
  average_disparity: number;
}

export interface FairnessReport {
  timestamp?: string;
  summary?: {
    total_tests: number;
    violations_detected: number;
    violation_rate: number;
    bias_level_distribution: Record<string, number>;
  };
  by_protected_attribute?: Record<string, FairnessAttributeStat>;
  by_fairness_metric?: Record<string, FairnessMetricStat>;
  recommendations?: string[];
}

export interface FairnessAuditResult {
  parameters: { samples: number; bias_strength: number; seed: number };
  report: FairnessReport;
}

export interface FairnessAuditedAttribute {
  attribute: string;
  groups: Record<string, { n: number; approval_rate: number }>;
  approval_disparity: number;
}

export interface LiveFairnessAudit {
  mode: "live" | "insufficient";
  report?: FairnessReport;
  audited?: {
    n_decisions: number;
    approval_rule: string;
    attributes: FairnessAuditedAttribute[];
    label_dependent_metrics: string;
  };
  reason?: string;
  n_decisions?: number;
  needed?: number;
}

export interface PredictionHistoryItem {
  prediction_id: string;
  timestamp: string;
  risk_score: number;
  risk_level: RiskLevel;
  confidence: number;
  processing_time_ms: number;
  loan_amount: number;
  loan_purpose: string;
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

export interface NasCandidate {
  architecture: string;
  hidden_scale: number;
  exit_level: number;
  precision: string;
  auc: number;
  ks: number;
  brier: number;
  val_loss: number;
  carbon_cost: number;
  passes: boolean;
}

export interface NasRunResult {
  status: "done" | "error";
  error?: string;
  dataset?: string;
  configs_tested?: number;
  passed_constraints?: number;
  fallback?: boolean;
  reference?: { auc: number; ks: number; brier: number };
  epochs?: number;
  train_samples?: number;
  candidates?: NasCandidate[];
  elapsed_seconds?: number;
}

export interface NasStatus {
  state: "idle" | "running" | "done" | "error";
  started_at: string | null;
  result: NasRunResult | null;
}

export interface SustainabilitySummary {
  count: number;
  total_energy_kwh: number;
  total_carbon_kg: number;
  total_duration_seconds: number;
  method: string | null;
  region: string | null;
  grid_source?: string | null;
}

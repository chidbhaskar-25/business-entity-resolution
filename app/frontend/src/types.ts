export interface HealthStatus {
  status: string;
  active_model: string;
  model_size_bytes: number;
  optimal_threshold: number;
  train_data_available: boolean;
  test_data_available: boolean;
  matching_results_exist: boolean;
  candidate_pairs_exist: boolean;
  timestamp: string;
}

export interface DatasetStats {
  train: {
    [key: string]: any;
    source1_sample_count?: number;
    source1_countries?: Record<string, number>;
    source1_missing_address_pct?: number;
  };
  test: {
    [key: string]: any;
  };
  ground_truth: {
    sampled_entities?: number;
    singleton_count?: number;
    singleton_percentage?: number;
    multi_match_count?: number;
    multi_match_percentage?: number;
  };
}

export interface ExperimentItem {
  experiment_id: string;
  model: string;
  features: string;
  blocking: string;
  threshold: number | null;
  precision: number | null;
  recall: number | null;
  f05: number | null;
  candidate_recall: number | null;
  runtime_seconds: number | null;
}

export interface MatchDetail {
  entity_id: string;
  business_name: string;
  business_address: string;
  country: string;
  source: string;
}

export interface CandidateDetail {
  entity_id: string;
  business_name: string;
  business_address: string;
  country: string;
  source: string;
  name_similarity: number;
  address_similarity: number;
  numeric_overlap: number;
  composite_score: number;
  country_match: boolean;
  model_probability: number;
  is_match: boolean;
}

export interface EntityMatchResult {
  source1_entity_id: string;
  business_name: string;
  business_address: string;
  country: string;
  is_singleton: boolean;
  matched_entity_ids: string[];
  candidate_entity_ids: string[];
  matches_details: MatchDetail[];
  candidate_details: CandidateDetail[];
}

export interface ValidationReport {
  is_valid: boolean;
  status: string;
  criteria_checks: Record<string, boolean>;
  errors: string[];
  warnings: string[];
  stats: Record<string, any>;
  report_summary: string;
}

export interface PredictResult {
  status: string;
  matching_file: string;
  candidate_file: string;
  sample_size: number | null;
  threshold_applied: number;
  validation_passed: boolean;
  validation_errors: string[];
  validation_stats: Record<string, any>;
}

export interface SubmissionInfo {
  package_exists: boolean;
  package_size_bytes: number;
  package_size_mb: number;
  matching_results_exists: boolean;
  candidate_pairs_exists: boolean;
  documentation_exists: boolean;
  total_test_entities: number;
  validation_status: string;
  download_url: string;
}

export interface EvaluationResult {
  model_path: string;
  model_type: string;
  threshold_applied: number;
  macro_f05: number;
  macro_precision: number;
  macro_recall: number;
  total_entities_evaluated: number;
  singleton_entities: number;
  singleton_accuracy: number;
  total_true_matches: number;
  total_predicted_matches: number;
  threshold_optimization?: any;
}

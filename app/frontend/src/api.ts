import type {
  DatasetStats,
  EntityMatchResult,
  EvaluationResult,
  ExperimentItem,
  HealthStatus,
  PredictResult,
  SubmissionInfo,
  ValidationReport,
} from './types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchDatasetStats(): Promise<DatasetStats> {
  const res = await fetch(`${API_BASE}/dataset/stats`);
  if (!res.ok) throw new Error(`Failed to load dataset stats: ${res.statusText}`);
  return res.json();
}

export async function fetchExperiments(): Promise<{ experiments: ExperimentItem[]; total: number }> {
  const res = await fetch(`${API_BASE}/experiments`);
  if (!res.ok) throw new Error(`Failed to load experiments: ${res.statusText}`);
  return res.json();
}

export async function fetchMetrics(): Promise<any> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error(`Failed to load metrics: ${res.statusText}`);
  return res.json();
}

export async function fetchEntityMatches(entityId: string): Promise<EntityMatchResult> {
  const res = await fetch(`${API_BASE}/matches/${encodeURIComponent(entityId)}`);
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Entity ${entityId} not found`);
  }
  return res.json();
}

export async function triggerValidation(params: {
  matchingPath?: string;
  candidatePath?: string;
  checkTargetIds?: boolean;
  s1SampleSize?: number | null;
}): Promise<ValidationReport> {
  const res = await fetch(`${API_BASE}/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      matching_path: params.matchingPath || 'output/matching_results.tsv',
      candidate_path: params.candidatePath || 'output/candidate_pairs.tsv',
      check_target_ids: params.checkTargetIds ?? false,
      s1_sample_size: params.s1SampleSize ?? null,
    }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || 'Validation execution failed');
  }
  return res.json();
}

export async function fetchSubmissionInfo(): Promise<SubmissionInfo> {
  const res = await fetch(`${API_BASE}/submission/info`);
  if (!res.ok) throw new Error(`Failed to load submission info: ${res.statusText}`);
  return res.json();
}

export async function triggerPrediction(params: {
  sampleSize?: number | null;
  targetSampleSize?: number | null;
  threshold?: number | null;
  chunkSize?: number;
}): Promise<PredictResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sample_size: params.sampleSize ?? 100,
      target_sample_size: params.targetSampleSize ?? 10000,
      threshold: params.threshold ?? null,
      chunk_size: params.chunkSize ?? 2000,
    }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || 'Inference execution failed');
  }
  return res.json();
}

export async function triggerEvaluation(params: {
  threshold?: number | null;
  optimizeThreshold?: boolean;
}): Promise<EvaluationResult> {
  const res = await fetch(`${API_BASE}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      threshold: params.threshold ?? null,
      optimize_threshold: params.optimizeThreshold ?? false,
    }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || 'Evaluation execution failed');
  }
  return res.json();
}

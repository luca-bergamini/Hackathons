const API_BASE = "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// --- Types ---

export interface ModelConfig {
  model_id: string;
  display_name?: string;
  provider?: string;
  cost_input_per_1m?: number;
  cost_output_per_1m?: number;
}

export interface AgentInfo {
  agent_id: string;
  task: string;
  num_records: number;
}

export interface DetectResult {
  agents: AgentInfo[];
  available_tasks: string[];
  total_records: number;
}

export interface JobSummary {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  dataset_key: string;
  selected_tasks: string[];
  selected_models: string[];
  num_records?: number;
  created_at: string;
  completed_at?: string;
  error?: string;
  synthetic_count?: number;
}

export interface PerRecord {
  test_id: string;
  task: string;
  model_id: string;
  score: number;
  correct: boolean;
  output_text?: string;
  judge_reasoning?: string;
}

export interface TaskModelMetric {
  task: string;
  model_id: string;
  avg_score: number;
  accuracy: number;
  num_records: number;
  num_correct: number;
}

export interface ModelMetric {
  overall_score: number;
  accuracy: number;
  num_records: number;
  num_correct: number;
}

export interface OperativeMetric {
  model: string;
  task: string;
  avg_latency_ms: number;
  total_requests: number;
  num_errors: number;
  num_retries: number;
  input_tokens: number;
  output_tokens: number;
  total_cost: number;
  avg_cost_per_request: number;
}

export interface Aggregated {
  per_record: PerRecord[];
  per_task_model: Record<string, TaskModelMetric>;
  per_model: Record<string, ModelMetric>;
  operative_metrics: Record<string, OperativeMetric>;
  best_models: {
    per_task: Record<string, { model_id: string; avg_score: number }>;
    overall: { model_id: string; overall_score: number };
  };
}

export interface InsightPattern {
  description: string;
  severity: "high" | "medium" | "low";
  test_ids: string[];
  recommendation?: string;
}

export interface InsightData {
  stats?: Record<string, unknown>;
  /** BN_INSIGHT_AGENT — error patterns returned at top level by insight_agent.run() */
  error_patterns?: InsightPattern[];
  /** BN_INSIGHT_AGENT — operational recommendations also at top level */
  operational_recommendations?: string[];
  insights?: {
    summary?: string;
    recommendation?: string;
    per_task_analysis?: Record<string, string>;
    anomalies_analysis?: string;
    operational_recommendations?: string[];
  };
}

export interface JobDetail extends JobSummary {
  has_report: boolean;
  has_json: boolean;
  aggregated: Aggregated | null;
  insights: InsightData | null;
}

export interface AblationEntry {
  change: string;
  delta_score: number;
  iteration: number;
}

export interface StatisticalTest {
  test_type: string; // "t-test" | "bootstrap"
  p_value: number;
  significant: boolean;
  confidence_level: number;
}

export interface PromptOptResult {
  best_prompt: string;
  baseline_score: number;
  best_score: number;
  iterations_run: number;
  beam_width: number;
  history: {
    iteration: number;
    score: number;
    num_correct: number;
    is_baseline?: boolean;
  }[];
  all_variants: {
    iteration: number;
    variant_index: number;
    score: number;
    prompt: string;
  }[];
  /** BN_PROMPT_OPTIMIZATION — statistical significance test result */
  statistical_test?: StatisticalTest;
  /** BN_PROMPT_OPTIMIZATION — ablation report: contribution of each change */
  ablation_report?: AblationEntry[];
}

// --- API calls ---

export const api = {
  health: () => request<{ status: string }>("/api/health"),

  getModels: () =>
    request<{ models: ModelConfig[]; judge: Record<string, string> }>("/api/models"),

  getDatasets: () =>
    request<{ datasets: string[]; bucket: string }>("/api/datasets"),

  detectTasks: (bucket: string, datasetKey: string) =>
    request<DetectResult>("/api/detect-tasks", {
      method: "POST",
      body: JSON.stringify({ bucket, dataset_key: datasetKey }),
    }),

  getDetected: () =>
    request<DetectResult & { detected: boolean; dataset_key: string }>("/api/detected"),

  createJob: (params: {
    dataset_key: string;
    selected_tasks: string[];
    selected_models: string[];
    enrich_synthetic: boolean;
    synth_records_per_task: number;
  }) =>
    request<{ job_id: string; status: string }>("/api/jobs", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  listJobs: () => request<{ jobs: JobSummary[] }>("/api/jobs"),

  getJob: (jobId: string) => request<JobDetail>(`/api/jobs/${jobId}`),

  getReportUrl: (jobId: string, fmt: "xlsx" | "json" = "xlsx") =>
    `${API_BASE}/api/jobs/${jobId}/report?fmt=${fmt}`,

  promptOptimize: (params: {
    agent_id: string;
    model_id: string;
    max_iterations: number;
    num_variants: number;
    beam_width: number;
    /** BN_PROMPT_OPTIMIZATION — multi-objective weights (must sum to ~1) */
    quality_weight?: number;
    cost_weight?: number;
    latency_weight?: number;
  }) =>
    request<PromptOptResult>("/api/prompt-optimize", {
      method: "POST",
      body: JSON.stringify(params),
    }),
};

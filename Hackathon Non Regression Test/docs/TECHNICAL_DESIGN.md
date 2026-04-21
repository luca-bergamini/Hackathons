# Technical Design — NRT Pipeline (Team 05)

## 1. Architecture Overview

The NRT (Non-Regression Testing) pipeline evaluates candidate LLM models against
a legacy baseline using an independent LLM judge. All inference runs on
**Amazon Bedrock** (`eu-west-1`) via the `converse` API.

```
┌──────────────────────────────────────────────────────────────┐
│  S3 Bucket (hackathon-genai-innovations-team-05-data)        │
│  └── dataset.jsonl                                           │
└────────┬─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│ Step 1 — Data        │  Load from S3, validate schema,
│ Processing           │  split by agent_id, identify task
│ (data_processing)    │  (heuristic + LLM fallback)
└────────┬─────────────┘
         │  {agents: {agent_id: {task, valid_records}}}
         ▼
┌──────────────────────┐
│ Step 2 — Model       │  Load configs/models.yaml
│ Selection            │  Validate no judge/candidate overlap
│ (model_selection)    │
└────────┬─────────────┘
         │  [model_configs], judge_config
         ▼
┌──────────────────────┐
│ Step 3 — Runner      │  Parallel inference via
│ (runner)             │  ThreadPoolExecutor (10 workers)
│                      │  MAX_RETRIES=3, TIMEOUT=30s
└────────┬─────────────┘
         │  {model_id: [inference_results]}
         ▼
┌──────────────────────┐
│ Step 4 — Evaluation  │  LLM-as-a-Judge (Claude Haiku 4.5)
│ (evaluation)         │  + deterministic metrics per task
│                      │  ThreadPoolExecutor (50 workers)
└────────┬─────────────┘
         │  {per_record, per_task_model, per_model}
         ▼
┌──────────────────────┐
│ Step 5 — Reporting   │  Excel (4 sheets) + JSON
│ (reporting)          │  Optional S3 upload
└────────┬─────────────┘
         │
    ┌────┴────────────────┐
    ▼                     ▼
┌──────────┐    ┌──────────────────┐
│ Insight  │    │ S3 Upload        │
│ Agent    │    │ (results/)       │
└──────────┘    └──────────────────┘
```

### Bonus Modules

| Module | Purpose |
|--------|---------|
| **Insight Agent** | Computes statistics and generates LLM-powered analysis of error patterns |
| **Prompt Optimizer** | Iterative prompt variant generation → Runner → Evaluator loop |
| **Synthetic Dataset** | LLM-generated test records following dataset schema |

## 2. Data Flow

### 2.1 Dataset Schema (Record)

Each record in `dataset.jsonl` contains:

| Field | Type | Description |
|-------|------|-------------|
| `test_id` | string | Unique identifier |
| `agent_id` | string | Agent that produced the record |
| `input_messages` | list[dict] | Conversation messages (role/content) |
| `ground_truth` | string/null | Expected correct output |
| `expected_output_type` | string | Output format (text, json, code) |
| `metadata` | dict | Additional context |
| `output` | string | Legacy model output |

### 2.2 Inter-Module Data Contracts

```
data_processing.run() -> {
    agents: { agent_id: { task: str, valid_records: [...] } }
}

runner.run() -> {
    model_id: [ { test_id, output_text, latency_ms, input_tokens, ... } ]
}

evaluation.run() -> {
    per_record: [ { test_id, model_id, task, score, ... } ],
    per_task_model: { task: { model_id: { avg_score, count } } },
    per_model: { model_id: { avg_score, tasks_evaluated } }
}

reporting.aggregate_scores() -> {
    per_record, per_task_model, per_model,
    operative_metrics, best_models
}
```

## 3. Evaluation Metrics

### 3.1 Scoring Architecture

Each task evaluator produces a **composite score** combining:
- **Deterministic metric** (0.0–1.0): task-specific, reproducible
- **LLM Judge score** (0.0–1.0): quality assessment via Claude Haiku 4.5

When ground truth is available: `score = 0.5 × deterministic + 0.5 × judge`.
When ground truth is absent: `score = judge_score`.

### 3.2 Per-Task Deterministic Metrics

| Task | Metric | Logic |
|------|--------|-------|
| Classification | Exact/contains match | Normalized string comparison |
| Context QA | Token overlap F1 | Precision × Recall over word tokens |
| Metadata Extraction | JSON validity + key/value accuracy | Parse JSON, compare keys and values |
| Code Generation | Syntax check + runtime execution | `ast.parse()` + subprocess with timeout |
| Translation / Summarization / Rephrasing | Length ratio + token overlap F1 | Word count ratio + token F1 |
| SQL Generation | Keyword presence + structural checks | SELECT/FROM/WHERE keyword matching |
| PII Redaction | Redaction ratio | Count of masked PII tokens |
| Intent Detection | Exact/contains match | Same as classification |

### 3.3 Advanced Evaluation (Code Generation)

Code generation uses a 4-layer evaluation:

1. **Syntax validation** via `ast.parse()` (safe, no code execution)
2. **Runtime execution** in isolated subprocess with 5s timeout
3. **Test detection** — bonus for code containing test functions/assertions
4. **LLM Judge** — holistic quality rating

Composite: `0.3 × syntax + 0.2 × runtime + 0.1 × tests + 0.4 × judge`

## 4. Parallelization Strategy

| Component | Executor | Workers | Rationale |
|-----------|----------|---------|-----------|
| Runner | `ThreadPoolExecutor` | 10 | I/O-bound Bedrock API calls, backpressure on quota |
| Evaluator | `ThreadPoolExecutor` | 50 (env: `EVAL_MAX_WORKERS`) | Judge calls are independent, high throughput needed |

Progress tracking uses `threading.Lock` with periodic logging every 25 completions.

Retry policy (Runner): exponential backoff with `MAX_RETRIES=3`,
`CALL_TIMEOUT_S=30`. Failed inferences are recorded with error details.

## 5. LLM Model Configuration

All models run on **Amazon Bedrock** via the `converse` API
(region `eu-west-1`).

### Candidates

| Model | Provider | Cost In/1M | Cost Out/1M |
|-------|----------|-----------|------------|
| gpt-oss-20b | OpenAI | $0.08 | $0.35 |
| Qwen3 Next 80B A3B | Qwen | $0.18 | $0.70 |
| Nova 2 Lite | Amazon | $0.34 | $2.87 |
| Devstral 2 123B | Mistral | $0.48 | $2.40 |

### Judge

| Model | Provider | Role |
|-------|----------|------|
| Claude Haiku 4.5 | Anthropic | LLM-as-a-Judge |

**Conflict-of-interest constraint**: the judge model must not appear in
the candidate list. Validated at startup by `model_selection`.

## 6. Prompt Optimization

The optimizer runs an iterative loop:

```
for iteration in 1..max_iterations:
    1. Generate N prompt variants (LLM-powered)
    2. For each variant:
       a. Replace system prompt in dataset records
       b. Run inference (runner.run_single_inference)
       c. Evaluate results (evaluation._evaluate_single)
    3. Select best variant by avg score
    4. If improvement < MIN_IMPROVEMENT (0.01): early stop
    5. Feed eval feedback to next iteration's generation prompt
```

- `DEFAULT_MAX_ITERATIONS = 3`
- `DEFAULT_NUM_VARIANTS = 3`
- Generates prompt variants using the same LLM that will be evaluated

## 7. Synthetic Dataset Generation

Generates task-specific synthetic records using LLM with seed examples:

1. Select up to 3 seed records from existing dataset per task
2. Prompt LLM with task-specific generation instructions
3. Parse JSON response into dataset-schema records
4. Assign unique `test_id` with `synth_` prefix
5. Merge with original dataset, deduplicating by `test_id`

Supported tasks: all 10 task types in `TASK_CLASSES`. Each has a
specialized generation prompt tuned for diversity and edge cases.

## 8. Payload Tracing

Every LLM call (inference, judge, insight, optimization) saves a
JSON payload to the `payloads/` directory:

```json
{
    "test_id": "...",
    "model_id": "...",
    "request": { ... },
    "response": { ... },
    "timestamp": 1719000000.123
}
```

This ensures full traceability and auditability of all LLM interactions.

## 9. S3 Persistence

Results are uploaded to S3 bucket `hackathon-genai-innovations-team-05-data`
under the `results/` prefix:

| File | Description |
|------|-------------|
| `results/report_YYYYMMDD_HHMMSS.xlsx` | Excel report (4 sheets) |
| `results/report_YYYYMMDD_HHMMSS.json` | Structured JSON output |
| `results/insights_YYYYMMDD_HHMMSS.json` | Insight agent analysis |

Upload is triggered by `reporting.run(upload_to_s3=True)` when `S3_BUCKET`
env var is set. The `upload_results_to_s3()` function uses `boto3.client("s3")`
with default credentials from the execution environment.

## 10. Security Considerations

- **Code execution sandboxing**: generated code runs in a subprocess with
  5-second timeout; temp files are cleaned in `finally` blocks
- **Path traversal prevention**: all payload file paths are sanitized via
  regex (`[^a-zA-Z0-9_-]` → `_`) and validated with `is_relative_to()`
- **No hardcoded credentials**: AWS auth via IAM role / environment
- **Input validation**: dataset records are validated for required fields
  before processing

"""Insight Agent — Step 5 Bonus.

Analizza i risultati di valutazione per identificare pattern di errore,
punti di forza/debolezza dei modelli e generare insight sintetici.
"""

import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PAYLOAD_DIR = Path("payloads")
_UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Deterministic statistics extraction
# ---------------------------------------------------------------------------

def _compute_per_task_stats(per_record: list[dict]) -> dict:
    """Compute per-task statistics from per_record."""
    task_stats: dict = {}
    for rec in per_record:
        task = rec.get("task", _UNKNOWN)
        ts = task_stats.setdefault(task, {"scores": [], "correct": 0, "total": 0,
                                          "errors": 0, "models": {}})
        ts["scores"].append(rec.get("score", 0.0))
        ts["total"] += 1
        ts["correct"] += int(bool(rec.get("correct")))
        ts["errors"] += int(bool(rec.get("error")))
        model = rec.get("model_id", _UNKNOWN)
        ts["models"].setdefault(model, []).append(rec.get("score", 0.0))

    per_task = {}
    for task, ts in task_stats.items():
        scores = ts["scores"]
        avg = round(sum(scores) / len(scores), 3) if scores else 0.0
        model_avgs = {m: round(sum(s) / len(s), 3) if s else 0.0 for m, s in ts["models"].items()}
        per_task[task] = {
            "avg_score": avg,
            "accuracy": round(ts["correct"] / ts["total"], 3) if ts["total"] else 0.0,
            "total_records": ts["total"],
            "num_errors": ts["errors"],
            "best_model": max(model_avgs, key=model_avgs.get) if model_avgs else "",
            "worst_model": min(model_avgs, key=model_avgs.get) if model_avgs else "",
            "model_scores": model_avgs,
        }
    return per_task


def _compute_per_model_stats(per_record: list[dict]) -> dict:
    """Compute per-model statistics from per_record."""
    model_stats: dict = {}
    for rec in per_record:
        model = rec.get("model_id", _UNKNOWN)
        ms = model_stats.setdefault(model, {"scores": [], "correct": 0, "total": 0,
                                            "errors": 0, "tasks": {}})
        ms["scores"].append(rec.get("score", 0.0))
        ms["total"] += 1
        ms["correct"] += int(bool(rec.get("correct")))
        ms["errors"] += int(bool(rec.get("error")))
        task = rec.get("task", _UNKNOWN)
        ms["tasks"].setdefault(task, []).append(rec.get("score", 0.0))

    per_model = {}
    for model, ms in model_stats.items():
        scores = ms["scores"]
        avg = round(sum(scores) / len(scores), 3) if scores else 0.0
        task_avgs = {t: round(sum(s) / len(s), 3) if s else 0.0 for t, s in ms["tasks"].items()}
        per_model[model] = {
            "overall_score": avg,
            "accuracy": round(ms["correct"] / ms["total"], 3) if ms["total"] else 0.0,
            "total_records": ms["total"],
            "num_errors": ms["errors"],
            "error_rate": round(ms["errors"] / ms["total"], 3) if ms["total"] else 0.0,
            "best_task": max(task_avgs, key=task_avgs.get) if task_avgs else "",
            "worst_task": min(task_avgs, key=task_avgs.get) if task_avgs else "",
            "task_scores": task_avgs,
        }
    return per_model


def _detect_anomalies(per_record: list[dict], per_model: dict) -> list[dict]:
    """Detect anomalies from per_record and per_model stats."""
    anomalies = [
        {
            "test_id": rec.get("test_id"),
            "model_id": rec.get("model_id"),
            "task": rec.get("task"),
            "reason": "Score 0 despite ground truth available",
        }
        for rec in per_record
        if rec.get("score", 0.0) == 0.0 and rec.get("ground_truth") is not None
    ]
    for model, info in per_model.items():
        if info["error_rate"] > 0.2:
            anomalies.append({
                "model_id": model,
                "reason": f"High error rate: {info['error_rate']:.1%}",
            })
    return anomalies[:20]


# Module-level lookup to avoid recreating the dict on every call (SonarQube S1854)
_TASK_OP_RECS: dict[str, str] = {
    "code_generation": (
        "Increase max_tokens for code_generation; add a chain-of-thought instruction "
        "in the system prompt asking the model to reason step-by-step before outputting code."
    ),
    "sql_generation": (
        "Include table schema definitions in the system prompt; "
        "instruct the model to output parameterized queries only."
    ),
    "pii_redaction": (
        "Add explicit examples of each PII type in the system prompt; "
        "use a strict instruction to replace ALL detected PII with [REDACTED]."
    ),
    "context_qa": (
        "Ensure context passages are included verbatim in input_messages; "
        "add a grounding instruction: Answer ONLY from the provided context."
    ),
    "metadata_extraction": (
        "Require strict JSON-only output in the system prompt; "
        "include the target JSON schema as an example to prevent hallucinated fields."
    ),
    "intent_detection": (
        "Supply an exhaustive list of valid intent labels in the system prompt; "
        "instruct the model to return a ranked JSON list for precision@k evaluation."
    ),
    "classification": (
        "List all valid labels with definitions in the system prompt; "
        "add examples of ambiguous cases to reduce misclassification."
    ),
    "translation": (
        "Specify both source and target language explicitly in the system prompt; "
        "add a fluency constraint: Use natural idiomatic expressions."
    ),
    "summarization": (
        "Set explicit length and coverage constraints in the system prompt; "
        "specify key information types that must be preserved."
    ),
    "rephrasing": (
        "Specify the target style (formal/informal, technical/simple) explicitly; "
        "provide 1-2 examples of the desired rephrasing style in the system prompt."
    ),
}


def _deterministic_recommendation(
    task: str, avg_score: float, severity: str, worst_model: str,
) -> str:
    """Generate a task-specific deterministic operational recommendation."""
    base = _TASK_OP_RECS.get(
        task, f"Review and improve the system prompt for task '{task}'."
    )
    model_note = (
        f" Worst performing model: {worst_model} — consider switching it for this task."
        if worst_model else ""
    )
    if severity == "high":
        urgency = " CRITICAL: avg score below 0.30 — immediate system-prompt revision required."
    elif severity == "medium":
        urgency = " MODERATE: avg score below 0.50 — prompt tuning recommended."
    else:
        urgency = ""
    return base + model_note + urgency
    return base + model_note + urgency


def _collect_low_score_index(
    per_record: list[dict],
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """Build per-task indices of failing record IDs and per-model failure counts."""
    task_low_scores: dict[str, list[str]] = {}
    task_model_fails: dict[str, dict[str, int]] = {}
    for rec in per_record:
        task = rec.get("task", _UNKNOWN)
        model = rec.get("model_id", _UNKNOWN)
        if rec.get("score", 1.0) < 0.5:
            tid = rec.get("test_id", "")
            if tid:
                task_low_scores.setdefault(task, []).append(tid)
            task_model_fails.setdefault(task, {})
            task_model_fails[task][model] = task_model_fails[task].get(model, 0) + 1
    return task_low_scores, task_model_fails


def _detect_error_patterns(per_task: dict, per_record: list[dict] | None = None) -> list[dict]:
    """BN_INSIGHT_AGENT: Return error patterns with test_id citations, severity, and recs.

    Severity:
    - high:   avg_score < 0.30 — critical failure, immediate intervention required
    - medium: 0.30 ≤ avg_score < 0.50 — below acceptable threshold, tuning needed
    - low:    0.50 ≤ avg_score < 0.70 — marginal performance, informational

    Each pattern includes:
    - cited_test_ids: up to 5 specific failing test IDs for traceability
    - model_failure_counts: per-model count of failing records for this task
    - worst_performing_model: model with most failures in this task
    - operational_recommendation: deterministic, task-specific action to take
    """
    per_record = per_record or []
    task_low_scores, task_model_fails = _collect_low_score_index(per_record)

    patterns = []
    for task, info in per_task.items():
        avg = info["avg_score"]
        if avg >= 0.70:
            continue
        if avg < 0.30:
            severity = "high"
        elif avg < 0.50:
            severity = "medium"
        else:
            severity = "low"
        cited_ids = task_low_scores.get(task, [])[:5]
        model_fails = task_model_fails.get(task, {})
        worst_model = max(model_fails, key=model_fails.get) if model_fails else ""
        op_rec = _deterministic_recommendation(task, avg, severity, worst_model)
        patterns.append({
            "task": task,
            "avg_score": avg,
            "severity": severity,
            "pattern": f"Task '{task}' has low avg score ({avg:.3f})",
            "cited_test_ids": cited_ids,
            "model_failure_counts": model_fails,
            "worst_performing_model": worst_model,
            "operational_recommendation": op_rec,
        })
    return patterns


def compute_stats(eval_results: dict) -> dict:
    """Estrae statistiche deterministiche dai risultati di valutazione.

    Args:
        eval_results: output di evaluation.run() con per_record, per_task_model, per_model.

    Returns:
        dict con per_task, per_model, anomalies, error_patterns.
    """
    per_record = eval_results.get("per_record", [])
    per_task = _compute_per_task_stats(per_record)
    per_model = _compute_per_model_stats(per_record)
    return {
        "per_task": per_task,
        "per_model": per_model,
        "anomalies": _detect_anomalies(per_record, per_model),
        "error_patterns": _detect_error_patterns(per_task, per_record),
        "total_records": len(per_record),
    }


# ---------------------------------------------------------------------------
# LLM-based insight generation
# ---------------------------------------------------------------------------

_INSIGHT_PROMPT = """You are an expert AI performance analyst. \
Analyze the following Non-Regression Testing results \
comparing candidate LLM models against a Legacy model.

## Statistics
{stats_json}

## Error Patterns (with specific failing test IDs)
{patterns_json}

## Instructions
Based on these results, provide:
1. **Executive Summary** (2-3 sentences): Overall assessment of model performance.
2. **Per-Task Analysis**: For each task, which model performs best/worst and why.
3. **Anomalies**: Flag any unexpected patterns (e.g., high error rates, score=0 with ground truth).
4. **Recommendation**: Which candidate model should be promoted to production, and for which tasks.
5. **Operational Recommendations**: Specific actionable recommendations referencing the failing \
test IDs, such as:
   - "Use model X for task Y because ..."
   - "Increase max_tokens for summarization because ..."
   - "Avoid model Z for SQL generation — test IDs abc, def show systematic failures"

Respond ONLY with valid JSON:
{{
  "summary": "<executive summary text>",
  "per_task_analysis": {{"<task>": "<analysis text>", ...}},
  "anomalies_analysis": "<anomalies discussion>",
  "recommendation": "<recommendation text>",
  "operational_recommendations": ["<rec1>", "<rec2>", ...]
}}"""


def _save_payload(call_id: str, model_id: str, request_body: dict, response_body: dict):
    """Salva request/response per payload tracing."""
    PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_call_id = re.sub(r'[^a-zA-Z0-9_-]', '_', call_id)
    safe_model = re.sub(r'[^a-zA-Z0-9_-]', '_', model_id)
    filepath = PAYLOAD_DIR / f"insight_agent_{safe_call_id}_{safe_model}.json"
    payload = {
        "call_id": call_id,
        "model_id": model_id,
        "request": request_body,
        "response": response_body,
        "timestamp": time.time(),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def generate_insights(stats: dict, llm_client, model_id: str) -> dict:
    """Genera insight qualitativi tramite chiamata LLM.

    Args:
        stats: output di compute_stats().
        llm_client: istanza BedrockClient.
        model_id: model_id del LLM da usare per l'analisi.

    Returns:
        dict con summary, per_task_analysis, anomalies_analysis, recommendation,
        operational_recommendations.
    """
    stats_json = json.dumps(stats, indent=2, default=str)
    if len(stats_json) > 5000:
        stats_json = stats_json[:5000] + "\n... (truncated)"

    # Include error patterns with test_id citations so LLM can give specific recs
    patterns = stats.get("error_patterns", [])
    patterns_json = json.dumps(patterns, indent=2, default=str)
    if len(patterns_json) > 2000:
        patterns_json = patterns_json[:2000] + "\n... (truncated)"

    prompt = _INSIGHT_PROMPT.format(stats_json=stats_json, patterns_json=patterns_json)
    call_id = str(int(time.time()))

    request_body = {
        "model_id": model_id,
        "prompt_preview": prompt[:200] + "...",
        "max_tokens": 2048,
    }

    try:
        response = llm_client.invoke(
            model_id=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.1,
        )
        raw = response.get("output_text", "").strip()

        _save_payload(call_id, model_id, request_body, response)

        # Parse JSON from response — try full match first, then partial repair
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed: dict = {}
        if match:
            json_str = match.group()
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                # Attempt partial field extraction via regex for truncated JSON
                logger.warning("JSON parsing failed, attempting partial field extraction")
                def _extract_str(key: str) -> str:
                    m = re.search(rf'"{key}"\s*:\s*"(.*?)(?<!\\)"', json_str, re.DOTALL)
                    return m.group(1).replace('\\"', '"') if m else ""
                def _extract_list(key: str) -> list:
                    m = re.search(rf'"{key}"\s*:\s*\[(.*?)(?:\]|$)', json_str, re.DOTALL)
                    if not m:
                        return []
                    items = re.findall(r'"(.*?)(?<!\\)"', m.group(1))
                    return items
                parsed = {
                    "summary": _extract_str("summary"),
                    "recommendation": _extract_str("recommendation"),
                    "anomalies_analysis": _extract_str("anomalies_analysis"),
                    "per_task_analysis": {},
                    "operational_recommendations": _extract_list("operational_recommendations"),
                }

        if parsed:
            return {
                "summary": parsed.get("summary", ""),
                "per_task_analysis": parsed.get("per_task_analysis", {}),
                "anomalies_analysis": parsed.get("anomalies_analysis", ""),
                "recommendation": parsed.get("recommendation", ""),
                "operational_recommendations": parsed.get("operational_recommendations", []),
            }

        # If JSON parsing fails entirely, treat raw text as summary
        return {
            "summary": raw[:1000],
            "per_task_analysis": {},
            "anomalies_analysis": "",
            "recommendation": "",
            "operational_recommendations": [],
        }

    except Exception as e:
        logger.warning("Insight generation LLM call failed: %s", e)
        _save_payload(call_id, model_id, request_body, {"error": str(e)})
        return {
            "summary": f"Insight generation failed: {e}",
            "per_task_analysis": {},
            "anomalies_analysis": "",
            "recommendation": "",
            "operational_recommendations": [],
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(eval_results: dict, llm_client, model_id: str) -> dict:
    """Esegue l'analisi completa dei risultati: statistiche + insight LLM.

    Args:
        eval_results: output di evaluation.run().
        llm_client: istanza BedrockClient.
        model_id: model_id per generazione insight (es. Judge model).

    Returns:
        dict con: stats, insights (summary, per_task_analysis, anomalies_analysis, recommendation).
    """
    # 1. Compute deterministic stats
    stats = compute_stats(eval_results)
    logger.info(
        "Stats computed: %d records, %d tasks, %d models, %d anomalies",
        stats["total_records"],
        len(stats["per_task"]),
        len(stats["per_model"]),
        len(stats["anomalies"]),
    )

    # 2. Generate LLM insights
    insights = generate_insights(stats, llm_client, model_id)
    logger.info("Insights generated successfully")

    # BN_INSIGHT_AGENT: expose error_patterns at top level for easy access in reports/UI
    return {
        "stats": stats,
        "insights": insights,
        "error_patterns": stats.get("error_patterns", []),
        "operational_recommendations": insights.get("operational_recommendations", []),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Insight Agent — Step 5 Bonus")
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    with open(args.eval_json, encoding="utf-8") as f:
        eval_data = json.load(f)

    from src.providers.bedrock import BedrockClient

    client = BedrockClient()
    result = run(eval_data, client, args.model)
    print(json.dumps(result, indent=2, default=str))

"""Evaluator Agent — Step 4."""

import ast
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import sqlparse
    _HAS_SQLPARSE = True
except ImportError:  # pragma: no cover
    _HAS_SQLPARSE = False

logger = logging.getLogger(__name__)

PAYLOAD_DIR = Path("payloads")
_UNKNOWN = "unknown"

# --- Duplicate string constants (SonarQube S1192) ---
_K_OUTPUT_TEXT = "output_text"
_K_GROUND_TRUTH = "ground_truth"
_K_SCORE = "score"
_K_CORRECT = "correct"
_K_JUDGE_SCORE = "judge_score"
_K_JUDGE_REASONING = "judge_reasoning"


def _save_payload(label, request_data, response_data):
    """Save judge call payload for traceability."""
    PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', str(label))[:30]
    path = PAYLOAD_DIR / f"eval_{safe_label}_{ts}.json"
    payload = {
        "timestamp": ts,
        "component": "evaluation",
        "label": label,
        "request": request_data,
        "response": response_data,
    }
    try:
        path.write_text(
            json.dumps(payload, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError):
        logger.debug("Payload save failed for %s", label, exc_info=True)


# IMPORTANTE: il modello Judge deve essere diverso dal modello valutato.

# I/O-bound (attesa API): si può andare ben oltre i core CPU.
# Limite reale = rate limit Bedrock + pool connessioni boto3.
# Configurabile via env var; default 50 (boto3 max_pool_connections viene alzato nel client).
EVAL_MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS", 50))

# ---------------------------------------------------------------------------
# Helper: LLM-as-a-Judge
# ---------------------------------------------------------------------------

def _llm_judge(
    judge_client,
    judge_model_id: str,
    task_description: str,
    output_text: str,
    ground_truth: str | None = None,
    criteria: str = "correctness, completeness, quality",
) -> dict:
    """Chiama il Judge LLM per valutare un output. Ritorna {score: 0-1, reasoning: str}."""
    gt_section = ""
    if ground_truth is not None:
        gt_section = f"\n\n## Expected Answer (Ground Truth)\n{ground_truth}"

    prompt = f"""You are an expert evaluator for the task: {task_description}.
Rate the following output on a scale from 0.0 to 1.0 based on: {criteria}.
{gt_section}

## Model Output
{output_text}

Respond ONLY with valid JSON: {{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    try:
        response = judge_client.invoke(
            model_id=judge_model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        _save_payload(
            task_description.replace(" ", "_")[:30],
            {"model": judge_model_id, "prompt": prompt[:500]},
            response,
        )
        text = response[_K_OUTPUT_TEXT].strip()
        # Extract JSON from response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            score = float(result.get(_K_SCORE, 0.0))
            return {_K_SCORE: max(0.0, min(1.0, score)), "reasoning": result.get("reasoning", "")}
    except Exception as e:
        logger.warning("LLM Judge error: %s", e)

    return {_K_SCORE: 0.0, "reasoning": "Judge evaluation failed"}


def _llm_judge_consensus(
    judge_client,
    judge_model_ids: list,
    task_description: str,
    output_text: str,
    ground_truth: str | None = None,
    criteria: str = "correctness, completeness, quality",
) -> dict:
    """Self-consistency cross-model judge: averages scores from ≥2 judge models.

    BN_ADVANCED_EVAL: aggregates signals from multiple models for higher reliability
    when no ground truth is available. At least 2 model IDs should be provided.
    """
    if not judge_model_ids:
        return {_K_SCORE: 0.0, "reasoning": "No judge models provided"}
    if len(judge_model_ids) == 1:
        return _llm_judge(judge_client, judge_model_ids[0], task_description,
                          output_text, ground_truth, criteria)
    results = [
        _llm_judge(judge_client, mid, task_description, output_text, ground_truth, criteria)
        for mid in judge_model_ids
    ]
    avg_score = round(sum(r[_K_SCORE] for r in results) / len(results), 3)
    reasoning = " | ".join(r["reasoning"] for r in results[:2])
    return {_K_SCORE: avg_score, "reasoning": reasoning, "consensus_models": len(results)}


def _normalize(text: str | None) -> str:
    """Normalize text for comparison."""
    if text is None:
        return ""
    return str(text).strip().lower()


def _token_overlap_f1(output: str, ground_truth: str) -> float:
    """Compute token-level F1 between output and ground truth."""
    out_tokens = set(_normalize(output).split())
    gt_tokens = set(_normalize(ground_truth).split())
    if not out_tokens or not gt_tokens:
        return 0.0
    common = out_tokens & gt_tokens
    precision = len(common) / len(out_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Must Have evaluators
# ---------------------------------------------------------------------------

def evaluate_classification(record: dict, judge_client, judge_model_id: str) -> dict:
    """Exact match + LLM-as-a-Judge per classification.

    BN_ADVANCED_EVAL composite weights rationale:
    - deterministic_score (0.50): exact/contains match — categorical labels allow strict comparison
    - judge_score (0.50): LLM judge covers paraphrase, label synonyms, and borderline cases
    When GT is unavailable, rely entirely on LLM judge (0% det / 100% judge).
    """
    output = _normalize(record.get(_K_OUTPUT_TEXT, ""))
    gt = _normalize(record.get(_K_GROUND_TRUTH))

    # Deterministic: exact match
    exact_match = 1.0 if gt and output == gt else 0.0

    # Deterministic: contains match (ground truth label appears in output)
    contains_match = 0.0
    if gt:
        contains_match = 1.0 if gt in output or output in gt else 0.0

    # LLM-as-a-Judge
    judge_result = _llm_judge(
        judge_client, judge_model_id,
        task_description="Text Classification",
        output_text=record.get(_K_OUTPUT_TEXT, ""),
        ground_truth=str(record.get(_K_GROUND_TRUTH, "")),
        criteria="correctness of the assigned label/category",
    )

    deterministic_score = max(exact_match, contains_match) if gt else 0.0
    if gt:
        combined = deterministic_score * 0.5 + judge_result[_K_SCORE] * 0.5
    else:
        combined = judge_result[_K_SCORE]

    return {
        _K_SCORE: round(combined, 3),
        "exact_match": exact_match,
        "contains_match": contains_match,
        _K_JUDGE_SCORE: judge_result[_K_SCORE],
        _K_JUDGE_REASONING: judge_result["reasoning"],
        _K_CORRECT: combined >= 0.5,
    }


def evaluate_context_qa(record: dict, judge_client, judge_model_id: str) -> dict:
    """Token overlap + LLM-as-a-Judge per context QA (RAG).

    BN_ADVANCED_EVAL composite weights rationale:
    - token_overlap_f1 (0.40): lexical overlap measures factual grounding in source context
    - judge_score (0.60): dominant weight because RAG quality (coherence, completeness)
      is not fully captured by token overlap alone
    When GT is unavailable, rely entirely on LLM judge (0% det / 100% judge).
    """
    output = record.get(_K_OUTPUT_TEXT, "")
    gt = record.get(_K_GROUND_TRUTH)

    # Deterministic: token overlap (simple F1-like)
    token_overlap = 0.0
    if gt:
        token_overlap = round(_token_overlap_f1(output, gt), 3)

    # LLM-as-a-Judge: relevance + faithfulness
    judge_result = _llm_judge(
        judge_client, judge_model_id,
        task_description="Context QA / RAG Answering",
        output_text=output,
        ground_truth=str(gt) if gt else None,
        criteria="relevance to the question, faithfulness to context, completeness",
    )

    combined = (token_overlap * 0.4 + judge_result[_K_SCORE] * 0.6) if gt else judge_result[_K_SCORE]

    return {
        _K_SCORE: round(combined, 3),
        "token_overlap_f1": token_overlap,
        _K_JUDGE_SCORE: judge_result[_K_SCORE],
        _K_JUDGE_REASONING: judge_result["reasoning"],
        _K_CORRECT: combined >= 0.5,
    }



def _parse_json_output(output: str) -> tuple[float, object]:
    """Try to parse JSON from text. Returns (json_valid_score, parsed_obj | None)."""
    try:
        return 1.0, json.loads(output)
    except (ValueError, TypeError):
        pass
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match:
        try:
            return 0.8, json.loads(match.group())
        except (ValueError, TypeError):
            pass
    return 0.0, None


def _compute_json_similarity(parsed_output: object, gt: object) -> tuple[float, float]:
    """Compute key_match and value_accuracy between two JSON-like objects."""
    import json as _json
    if not (parsed_output and gt):
        return 0.0, 0.0
    parsed_gt: dict = {}
    if isinstance(gt, dict):
        parsed_gt = gt
    elif isinstance(gt, str):
        try:
            parsed_gt = _json.loads(gt)
        except (ValueError, TypeError):
            pass
    if not (parsed_gt and isinstance(parsed_gt, dict) and isinstance(parsed_output, dict)):
        return 0.0, 0.0
    gt_keys = set(parsed_gt.keys())
    if not gt_keys:
        return 0.0, 0.0
    out_keys = set(parsed_output.keys())
    key_match = len(gt_keys & out_keys) / len(gt_keys)
    matching = sum(
        1 for k in gt_keys & out_keys
        if _normalize(str(parsed_gt[k])) == _normalize(str(parsed_output.get(k, "")))
    )
    return key_match, matching / len(gt_keys)

def evaluate_metadata_extraction(record: dict, judge_client, judge_model_id: str) -> dict:
    """JSON validity + key match + value accuracy.

    BN_ADVANCED_EVAL composite weights rationale:
    With GT: det_score(0.50) + judge(0.50) where det = json_valid(0.30) + key_match(0.35) + value_accuracy(0.35)
    - json_valid: structural requirement — malformed JSON cannot be consumed downstream
    - key_match: completeness — all required fields must be present
    - value_accuracy: correctness — the extracted values match the expected ones
    Without GT: json_valid(0.30) + judge(0.70) — structural check only; content quality via judge.
    """
    output = record.get(_K_OUTPUT_TEXT, "")
    gt = record.get(_K_GROUND_TRUTH)

    json_valid, parsed_output = _parse_json_output(output)
    key_match, value_accuracy = _compute_json_similarity(parsed_output, gt)

    judge_result = _llm_judge(
        judge_client, judge_model_id,
        task_description="Metadata Extraction to JSON",
        output_text=output,
        ground_truth=str(gt) if gt else None,
        criteria="JSON validity, key completeness, value accuracy",
    )

    det_score = (json_valid * 0.3 + key_match * 0.35 + value_accuracy * 0.35) if gt else json_valid
    combined = (det_score * 0.5 + judge_result[_K_SCORE] * 0.5) if gt else (
        json_valid * 0.3 + judge_result[_K_SCORE] * 0.7
    )

    return {
        _K_SCORE: round(combined, 3),
        "json_valid": json_valid,
        "key_match": round(key_match, 3),
        "value_accuracy": round(value_accuracy, 3),
        _K_JUDGE_SCORE: judge_result[_K_SCORE],
        _K_JUDGE_REASONING: judge_result["reasoning"],
        _K_CORRECT: combined >= 0.5,
    }


# ---------------------------------------------------------------------------
# Advanced Evaluation helpers (Bonus 3.7)
# ---------------------------------------------------------------------------


def _execute_code(code: str, timeout: int = 5) -> dict:
    """Execute Python code in a subprocess with timeout."""
    tmp_path = None
    try:
        fd_num, tmp_path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd_num, "w", encoding="utf-8") as fd:
            fd.write(code)
        proc = subprocess.run(  # noqa: S603
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "executed": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout[:500],
            "stderr": proc.stderr[:500],
            "runtime_valid": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "executed": True,
            "returncode": -1,
            "runtime_valid": False,
            "error": "timeout",
        }
    except (OSError, subprocess.SubprocessError) as e:
        return {
            "executed": False,
            "runtime_valid": False,
            "error": str(e),
        }
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _has_test_code(code: str) -> bool:
    """Check if code contains test functions or assertions."""
    return bool(
        re.search(r"def test_|unittest\.main|\bassert\b", code)
    )


def _generate_unit_tests(code: str, judge_client, judge_model_id: str) -> str:
    """BN_ADVANCED_EVAL: Auto-generate unit tests for code via LLM.

    Asks the LLM to generate pytest-compatible unit tests for the given code snippet.
    Returns the generated test code as a string (empty string if generation fails).
    """
    prompt = f"""You are a Python test engineer. Given the following code, write pytest unit tests
that verify its functional correctness. Cover normal cases and edge cases.

## Code to Test
```python
{code[:2000]}
```

Respond ONLY with valid Python code (no explanation). Start with `import` statements."""
    try:
        response = judge_client.invoke(
            model_id=judge_model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        test_code = response.get("output_text", "").strip()
        # Extract code block if wrapped in fences
        match = re.search(r"```(?:python)?\s*\n(.*?)```", test_code, re.DOTALL)
        return match.group(1) if match else test_code
    except Exception as e:
        logger.debug("Unit test generation failed: %s", e)
        return ""


def _collect_test_results(
    code: str,
    syntax_valid: float,
    exec_result: dict,
    judge_client,
    judge_model_id: str,
) -> tuple:
    """Return (has_tests, tests_passed, auto_test_passed) for code evaluation."""
    has_tests = bool(syntax_valid) and _has_test_code(code)
    tests_passed = None
    if has_tests and exec_result.get("executed"):
        tests_passed = exec_result.get("runtime_valid", False)

    auto_test_passed = None
    if syntax_valid == 1.0 and not has_tests:
        generated_tests = _generate_unit_tests(code, judge_client, judge_model_id)
        if generated_tests:
            combined_script = code + "\n\n" + generated_tests
            auto_test_result = _execute_code(combined_script)
            auto_test_passed = auto_test_result.get("runtime_valid", False)

    return has_tests, tests_passed, auto_test_passed


def _compute_code_score(
    syntax_valid: float,
    runtime_score: float,
    effective_tests_passed,
    exec_result: dict,
    judge_score: float,
) -> float:
    """Compute composite code generation score from component signals."""
    if effective_tests_passed is not None:
        return (
            syntax_valid * 0.10
            + runtime_score * 0.15
            + (1.0 if effective_tests_passed else 0.0) * 0.25
            + judge_score * 0.50
        )
    if exec_result.get("executed"):
        return syntax_valid * 0.15 + runtime_score * 0.15 + judge_score * 0.70
    return syntax_valid * 0.30 + judge_score * 0.70


def evaluate_code_generation(
    record: dict, judge_client, judge_model_id: str,
) -> dict:
    """Syntax + runtime execution + LLM-generated unit tests + LLM Judge.

    BN_ADVANCED_EVAL composite weights rationale:
    - syntax_valid (0.10): necessary but not sufficient — syntactically valid code may still be wrong
    - runtime_valid (0.15): running without errors is a stronger signal than syntax alone
    - auto_tests_passed (0.25): LLM-generated unit tests verify functional correctness
    - judge_score (0.50): LLM judge provides semantic quality assessment
    When no execution is possible: syntax(0.30) + judge(0.70) to rely on semantic evaluation.
    """
    output = record.get(_K_OUTPUT_TEXT, "")

    # Extract code from markdown fences if present
    code = output
    code_match = re.search(
        r"```(?:python)?\s*\n(.*?)```", output, re.DOTALL
    )
    if code_match:
        code = code_match.group(1)

    # Deterministic: syntax check via ast.parse (safe, no exec)
    syntax_valid = 0.0
    syntax_error = ""
    try:
        ast.parse(code)
        syntax_valid = 1.0
    except SyntaxError as e:
        syntax_error = str(e)

    # Advanced: runtime execution
    exec_result = {
        "runtime_valid": False, "executed": False,
    }
    if syntax_valid == 1.0:
        exec_result = _execute_code(code)
    runtime_score = (
        1.0 if exec_result.get("runtime_valid") else 0.0
    )

    has_tests, tests_passed, auto_test_passed = _collect_test_results(
        code, syntax_valid, exec_result, judge_client, judge_model_id,
    )
    effective_tests_passed = tests_passed if tests_passed is not None else auto_test_passed

    # LLM-as-a-Judge
    gt = record.get(_K_GROUND_TRUTH)
    judge_result = _llm_judge(
        judge_client, judge_model_id,
        task_description="Code Generation",
        output_text=output,
        ground_truth=str(gt) if gt else None,
        criteria=(
            "correctness, code quality,"
            " adherence to requirements"
        ),
    )

    combined = _compute_code_score(
        syntax_valid, runtime_score, effective_tests_passed,
        exec_result, judge_result[_K_SCORE],
    )

    stderr = exec_result.get("stderr", "")
    err = exec_result.get("error", "")
    return {
        _K_SCORE: round(combined, 3),
        "syntax_valid": syntax_valid,
        "syntax_error": syntax_error,
        "runtime_valid": exec_result.get("runtime_valid", False),
        "runtime_error": stderr or err,
        "has_tests": has_tests,
        "tests_passed": tests_passed,
        "auto_tests_passed": auto_test_passed,
        _K_JUDGE_SCORE: judge_result[_K_SCORE],
        _K_JUDGE_REASONING: judge_result["reasoning"],
        _K_CORRECT: combined >= 0.5,
    }


def evaluate_text_refinement(record: dict, judge_client, judge_model_id: str) -> dict:
    """Covers translation, summarization, rephrasing. Token overlap + LLM-as-a-Judge.

    BN_ADVANCED_EVAL composite weights rationale:
    - det_score (0.40): blend of token_overlap(0.50) + length_score(0.50)
      - token_overlap: measures lexical preservation/transformation quality
      - length_score: guards against empty outputs or wildly disproportionate lengths
    - judge_score (0.60): dominant weight because text quality is inherently subjective;
      fluency, meaning preservation, and style are best assessed by LLM judge
    When GT is unavailable, rely entirely on LLM judge (0% det / 100% judge).
    """
    output = record.get(_K_OUTPUT_TEXT, "")
    gt = record.get(_K_GROUND_TRUTH)
    task = record.get("task", "text_refinement")

    # Deterministic: length ratio (output shouldn't be empty or wildly different length)
    length_score = 0.0
    if output.strip():
        if gt:
            gt_len = max(len(str(gt).split()), 1)
            out_len = len(output.split())
            ratio = min(out_len, gt_len) / max(out_len, gt_len)
            length_score = ratio
        else:
            length_score = 1.0 if len(output.split()) > 3 else 0.5

    # Deterministic: token overlap with ground truth
    token_overlap = 0.0
    if gt:
        token_overlap = _token_overlap_f1(output, gt)

    # LLM-as-a-Judge with task-specific criteria
    criteria_map = {
        "translation": "translation accuracy, fluency, meaning preservation",
        "summarization": "conciseness, completeness of key information, faithfulness",
        "rephrasing": "meaning preservation, style change, grammatical correctness",
    }
    criteria = criteria_map.get(task, "quality, faithfulness to original, fluency")

    judge_result = _llm_judge(
        judge_client, judge_model_id,
        task_description=f"Text Refinement ({task})",
        output_text=output,
        ground_truth=str(gt) if gt else None,
        criteria=criteria,
    )

    det_score = (token_overlap * 0.5 + length_score * 0.5) if gt else length_score
    combined = (det_score * 0.4 + judge_result[_K_SCORE] * 0.6) if gt else judge_result[_K_SCORE]

    return {
        _K_SCORE: round(combined, 3),
        "token_overlap": round(token_overlap, 3),
        "length_score": round(length_score, 3),
        _K_JUDGE_SCORE: judge_result[_K_SCORE],
        _K_JUDGE_REASONING: judge_result["reasoning"],
        _K_CORRECT: combined >= 0.5,
    }


# ---------------------------------------------------------------------------
# Bonus evaluators — BN_EXTENDED_TASKS
# ---------------------------------------------------------------------------

def _sql_token_similarity(output_sql: str, gt_sql: str) -> float:
    """BN_EXTENDED_TASKS: Token-level SQL similarity using sqlparse normalisation.

    Normalises both SQL strings (uppercase keywords, strip comments/whitespace)
    and computes token-overlap F1. Returns 0.0 if sqlparse is unavailable.
    """
    if not _HAS_SQLPARSE or not output_sql.strip() or not gt_sql.strip():
        return 0.0
    fmt = {"strip_comments": True, "keyword_case": "upper", "strip_whitespace": True}
    out_norm = set(sqlparse.format(output_sql, **fmt).split())
    gt_norm = set(sqlparse.format(gt_sql, **fmt).split())
    if not out_norm or not gt_norm:
        return 0.0
    common = out_norm & gt_norm
    precision = len(common) / len(out_norm)
    recall = len(common) / len(gt_norm)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 3)


def _sql_syntax_score(sql_text: str) -> tuple[float, str]:
    """BN_EXTENDED_TASKS: Parse SQL with sqlparse for proper syntax validation.

    Returns (score 0.0-1.0, detected statement type).
    Falls back to keyword matching if sqlparse is unavailable.
    """
    _sql_kw = ["select", "insert", "update", "delete", "create", "alter", "drop", "with"]
    if not sql_text.strip():
        return 0.0, "empty"
    if _HAS_SQLPARSE:
        if not any(kw in sql_text.lower() for kw in _sql_kw):
            return 0.0, "none"
        statements = sqlparse.parse(sql_text)
        valid_stmts = [s for s in statements if s.get_type() is not None]
        if valid_stmts:
            stmt_type = valid_stmts[0].get_type() or "unknown"
            return 1.0, stmt_type.lower()
        # No recognized SQL type — keywords present but no parseable statement
        return 0.5, "keyword_only"
    # Fallback: keyword matching
    sql_keywords = ["select", "insert", "update", "delete", "create", "alter", "drop", "with"]
    if any(kw in sql_text.lower() for kw in sql_keywords):
        return 1.0, "keyword_match"
    return 0.0, "none"


def evaluate_sql_generation(record: dict, judge_client, judge_model_id: str) -> dict:
    """BN_EXTENDED_TASKS: SQL syntax via sqlparse + LLM-as-a-Judge.

    Composite weight rationale:
    - syntax_score (0.30): structural validity — well-formed SQL is necessary
    - judge_score (0.70): semantic correctness — does the query solve the requirement?
    """
    output = record.get(_K_OUTPUT_TEXT, "")
    gt = record.get(_K_GROUND_TRUTH)

    # Extract SQL from output (may be wrapped in fences)
    sql_match = re.search(r"```(?:sql)?\s*\n(.*?)```", output, re.DOTALL | re.IGNORECASE)
    sql_text = sql_match.group(1) if sql_match else output

    syntax_score, stmt_type = _sql_syntax_score(sql_text)

    # BN_EXTENDED_TASKS: token-level SQL similarity against ground truth
    sql_similarity = 0.0
    if gt:
        gt_sql_match = re.search(
            r"```(?:sql)?\s*\n(.*?)```", str(gt), re.DOTALL | re.IGNORECASE
        )
        gt_sql_text = gt_sql_match.group(1) if gt_sql_match else str(gt)
        sql_similarity = _sql_token_similarity(sql_text, gt_sql_text)

    judge_result = _llm_judge(
        judge_client, judge_model_id,
        task_description="SQL Generation",
        output_text=output,
        ground_truth=str(gt) if gt else None,
        criteria="SQL correctness, query logic, adherence to requirements",
    )

    # weight: 0.25 syntax + 0.05 sql_similarity (when gt available) + 0.70 judge
    # or: 0.30 syntax + 0.70 judge (no gt or no sqlparse)
    # see _sql_syntax_score and _sql_token_similarity docstrings for rationale
    if gt and sql_similarity > 0:
        combined = syntax_score * 0.25 + sql_similarity * 0.05 + judge_result[_K_SCORE] * 0.70
    else:
        combined = syntax_score * 0.30 + judge_result[_K_SCORE] * 0.70

    return {
        _K_SCORE: round(combined, 3),
        "syntax_score": syntax_score,
        "has_sql_syntax": syntax_score,  # backward-compatible alias
        "sql_type": stmt_type,
        "sql_token_similarity": sql_similarity,
        _K_JUDGE_SCORE: judge_result[_K_SCORE],
        _K_JUDGE_REASONING: judge_result["reasoning"],
        _K_CORRECT: combined >= 0.5,
    }


_PII_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # email
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # phone (US)
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b",  # credit card (Visa/MC)
]


def _extract_pii_entities(text: str) -> set:
    """Extract all PII entity values found in text using regex patterns."""
    entities: set = set()
    for pattern in _PII_PATTERNS:
        entities.update(re.findall(pattern, text))
    return entities


def _pii_f1_with_gt(
    input_entities: set, output_entities: set, gt_entities: set,
) -> dict:
    """BN_EXTENDED_TASKS: GT-based entity-level F1 for PII redaction.

    Uses the ground-truth redacted text to determine which entities should be removed.
    - should_redact = entities in input that are absent from GT (GT says remove them)
    - TP: correctly removed (in should_redact, absent from output)
    - FN: missed (in should_redact, still in output)
    - FP: over-redacted or new PII (in output but not expected by GT)
    """
    should_redact = input_entities - gt_entities
    if not should_redact:
        extra = output_entities - gt_entities
        prec = 1.0 if not extra else 0.0
        return {"precision": prec, "recall": 1.0, "f1": prec,
                "entities_found_in_input": len(input_entities), "entities_leaked": 0}
    tp = len(should_redact - output_entities)
    fn = len(should_redact & output_entities)
    fp = len(output_entities - gt_entities)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "entities_found_in_input": len(input_entities),
        "entities_leaked": fn,
    }


def _pii_entity_f1(input_text: str, output_text: str, gt_text: str | None) -> dict:
    """BN_EXTENDED_TASKS: Compute entity-level precision, recall, F1 for PII redaction.

    When gt_text is provided, uses GT-based evaluation (_pii_f1_with_gt) to compare
    output entities against the expected redacted text for true entity-level F1.
    When gt_text is None, falls back to proxy metric: how much PII was removed from input.
    - TP: input_entities correctly removed from output
    - FN: input_entities still present in output (missed redaction)
    - FP: unexpected PII in output not seen in input
    precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = harmonic mean
    """
    input_entities = _extract_pii_entities(input_text)
    output_entities = _extract_pii_entities(output_text)

    if gt_text is not None:
        gt_entities = _extract_pii_entities(gt_text)
        return _pii_f1_with_gt(input_entities, output_entities, gt_entities)

    if not input_entities:
        prec = 1.0 if not output_entities else 0.0
        return {"precision": prec, "recall": 1.0, "f1": prec,
                "entities_found_in_input": 0, "entities_leaked": 0}

    tp = len(input_entities - output_entities)  # correctly redacted
    fn = len(input_entities & output_entities)  # missed (leaked)
    fp = len(output_entities - input_entities)  # unexpected PII in output

    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "entities_found_in_input": len(input_entities),
        "entities_leaked": fn,
    }


def evaluate_pii_redaction(record: dict, judge_client, judge_model_id: str) -> dict:
    """BN_EXTENDED_TASKS: Entity-level precision, recall, F1 + LLM-as-a-Judge.

    Composite weight rationale:
    - entity_f1 (0.50): entity-level F1 measures actual redaction completeness per entity
    - judge_score (0.50): LLM judge evaluates content preservation quality
    """
    output = record.get(_K_OUTPUT_TEXT, "")
    gt = record.get(_K_GROUND_TRUTH)

    # Extract original user input to compute entity F1
    input_text = output  # fallback: use output as input estimate
    for msg in record.get("input_messages", []):
        if msg.get("role") == "user":
            input_text = str(msg.get("content", output))
            break

    entity_metrics = _pii_entity_f1(input_text, output, str(gt) if gt else None)
    entity_f1 = entity_metrics["f1"]

    judge_result = _llm_judge(
        judge_client, judge_model_id,
        task_description="PII Redaction",
        output_text=output,
        ground_truth=str(gt) if gt else None,
        criteria="completeness of PII removal, preservation of non-PII content",
    )

    # weight: 0.50 entity F1 + 0.50 judge (see docstring rationale)
    combined = entity_f1 * 0.50 + judge_result[_K_SCORE] * 0.50

    return {
        _K_SCORE: round(combined, 3),
        "entity_precision": entity_metrics["precision"],
        "entity_recall": entity_metrics["recall"],
        "entity_f1": entity_f1,
        "entities_found_in_input": entity_metrics["entities_found_in_input"],
        "entities_leaked": entity_metrics["entities_leaked"],
        # backward-compatible fields expected by existing tests
        "redaction_score": entity_f1,
        "pii_found": entity_metrics["entities_leaked"],
        _K_JUDGE_SCORE: judge_result[_K_SCORE],
        _K_JUDGE_REASONING: judge_result["reasoning"],
        _K_CORRECT: combined >= 0.5,
    }


def _extract_intent_ranking(output_text: str) -> list:
    """BN_EXTENDED_TASKS: Extract ranked intent list from model output.

    Tries to detect numbered lists like '1. intent_name' or 'intent_a, intent_b, intent_c'.
    Returns ordered list of intent strings (empty means single-value output).
    """
    numbered = re.findall(r"^\s*\d+[.)]\s*(.+?)$", output_text, re.MULTILINE)
    if len(numbered) >= 2:
        return [n.strip().lower() for n in numbered]
    comma = re.split(r",\s*|\s*;\s*", output_text.strip())
    if len(comma) >= 2:
        return [c.strip().lower() for c in comma if c.strip()]
    return []


def _precision_at_k(ranked: list, ground_truth: str, k: int) -> float:
    """Compute precision@k: 1.0 if ground_truth appears in top-k items, else 0.0."""
    return 1.0 if ground_truth in ranked[:k] else 0.0


def _compute_intent_ranking_score(
    ranked: list,
    gt: str,
    exact_match: float,
    contains_match: float,
    precision_at_1,
) -> float:
    """Return the deterministic ranking score for intent detection."""
    if ranked and gt and precision_at_1 is not None:
        return max(precision_at_1, contains_match)
    return max(exact_match, contains_match)


def evaluate_intent_detection(record: dict, judge_client, judge_model_id: str) -> dict:
    """BN_EXTENDED_TASKS: Precision@k ranking + exact/contains match + LLM-as-a-Judge.

    Composite weight rationale:
    - det_score (0.50): deterministic precision@k covers ranking quality
    - judge_score (0.50): LLM judge evaluates semantic intent correctness
    When GT is unavailable, rely entirely on LLM judge (0% / 100%).
    """
    raw_output = record.get(_K_OUTPUT_TEXT, "")
    output = _normalize(raw_output)
    gt = _normalize(record.get(_K_GROUND_TRUTH))

    exact_match = 1.0 if gt and output == gt else 0.0
    contains_match = 1.0 if gt and (gt in output or output in gt) else 0.0

    # Precision@k for ranked outputs
    ranked = _extract_intent_ranking(raw_output)
    precision_at_1 = _precision_at_k(ranked, gt, 1) if ranked and gt else None
    precision_at_3 = _precision_at_k(ranked, gt, 3) if ranked and gt else None
    precision_at_5 = _precision_at_k(ranked, gt, 5) if ranked and gt else None

    ranking_score = _compute_intent_ranking_score(
        ranked, gt, exact_match, contains_match, precision_at_1,
    )

    judge_result = _llm_judge(
        judge_client, judge_model_id,
        task_description="Intent Detection",
        output_text=raw_output,
        ground_truth=str(record.get(_K_GROUND_TRUTH, "")) if record.get(_K_GROUND_TRUTH) else None,
        criteria="correctness of detected intent, ranking quality",
    )

    det_score = ranking_score if gt else 0.0
    combined = (det_score * 0.50 + judge_result[_K_SCORE] * 0.50) if gt else judge_result[_K_SCORE]

    return {
        _K_SCORE: round(combined, 3),
        "exact_match": exact_match,
        "contains_match": contains_match,
        "precision_at_1": precision_at_1,
        "precision_at_3": precision_at_3,
        "precision_at_5": precision_at_5,
        "ranked_output": bool(ranked),
        _K_JUDGE_SCORE: judge_result[_K_SCORE],
        _K_JUDGE_REASONING: judge_result["reasoning"],
        _K_CORRECT: combined >= 0.5,
    }




TASK_EVALUATORS = {
    "classification": evaluate_classification,
    "context_qa": evaluate_context_qa,
    "metadata_extraction": evaluate_metadata_extraction,
    "code_generation": evaluate_code_generation,
    "translation": evaluate_text_refinement,
    "summarization": evaluate_text_refinement,
    "rephrasing": evaluate_text_refinement,
    "sql_generation": evaluate_sql_generation,
    "pii_redaction": evaluate_pii_redaction,
    "intent_detection": evaluate_intent_detection,
}


def _evaluate_single(
    result: dict, model_id: str, judge_client, judge_model_id: str,
    consensus_judge_model_ids: list[str] | None = None,
) -> dict | None:
    """Valuta un singolo record. Ritorna eval_result dict o None se skip."""
    if not result.get("success", False):
        return {
            "test_id": result.get("test_id"),
            "model_id": model_id,
            "task": result.get("task", _UNKNOWN),
            _K_SCORE: 0.0,
            _K_CORRECT: False,
            _K_OUTPUT_TEXT: "",
            "legacy_output": result.get("legacy_output"),
            _K_GROUND_TRUTH: result.get(_K_GROUND_TRUTH),
            "error": result.get("error"),
        }

    task = result.get("task", _UNKNOWN)
    evaluator = TASK_EVALUATORS.get(task)

    if evaluator is None:
        logger.warning("No evaluator for task=%s, skipping test_id=%s", task, result.get("test_id"))
        return None

    try:
        scores = evaluator(result, judge_client, judge_model_id)
    except Exception as e:
        logger.error("Evaluation error test_id=%s task=%s: %s", result.get("test_id"), task, e)
        scores = {_K_SCORE: 0.0, _K_CORRECT: False, "error": str(e)}

    # BN_ADVANCED_EVAL: self-consistency cross-model judge for no-GT records
    # When ground truth is absent, averaging ≥2 judge models reduces single-model bias.
    if (
        consensus_judge_model_ids
        and len(consensus_judge_model_ids) > 1
        and result.get(_K_GROUND_TRUTH) is None
        and _K_JUDGE_SCORE in scores
    ):
        consensus = _llm_judge_consensus(
            judge_client, consensus_judge_model_ids,
            task_description=task,
            output_text=result.get(_K_OUTPUT_TEXT, ""),
            ground_truth=None,
        )
        # Blend: average existing single-judge score with cross-model consensus
        scores[_K_JUDGE_SCORE] = round(
            (scores[_K_JUDGE_SCORE] + consensus[_K_SCORE]) / 2, 3,
        )
        scores["consensus_judge_score"] = consensus[_K_SCORE]
        scores["consensus_models_used"] = consensus.get("consensus_models", 1)

    return {
        "test_id": result.get("test_id"),
        "model_id": model_id,
        "task": task,
        _K_OUTPUT_TEXT: result.get(_K_OUTPUT_TEXT, ""),
        "legacy_output": result.get("legacy_output"),
        _K_GROUND_TRUTH: result.get(_K_GROUND_TRUTH),
        **scores,
    }



def _aggregate_eval_records(per_record: list[dict]) -> tuple[dict, dict]:
    """Aggregate per-record eval results into per_task_model and per_model dicts."""
    acc_tm: dict[tuple[str, str], list[float]] = {}
    acc_m: dict[str, list[float]] = {}
    for rec in per_record:
        score = rec.get(_K_SCORE, 0.0)
        key = (rec.get("task", _UNKNOWN), rec.get("model_id", _UNKNOWN))
        acc_tm.setdefault(key, []).append(score)
        acc_m.setdefault(rec.get("model_id", _UNKNOWN), []).append(score)
    per_task_model = {
        f"{t}|{m}": {
            "avg_score": round(sum(s) / len(s), 3) if s else 0.0,
            "num_records": len(s),
        }
        for (t, m), s in acc_tm.items()
    }
    per_model = {
        mid: {
            "overall_score": round(sum(s) / len(s), 3) if s else 0.0,
            "num_records": len(s),
        }
        for mid, s in acc_m.items()
    }
    return per_task_model, per_model

def run(
    inference_results: dict,
    judge_client,
    judge_model_id: str,
    consensus_judge_model_ids: list[str] | None = None,
) -> dict:
    """Esegue la valutazione in PARALLELO su tutti i risultati di inferenza.

    Args:
        inference_results: dict model_id -> lista di risultati dal Runner.
        judge_client: istanza BedrockClient per il Judge.
        judge_model_id: model_id del Judge (deve essere != dai candidate valutati).
        consensus_judge_model_ids: optional list of model IDs for cross-model consensus
            (BN_ADVANCED_EVAL). When ≥2 IDs are provided, records without ground truth
            are judged by multiple models and scores are averaged for higher reliability.

    Returns:
        dict con: per_record, per_task_model, per_model.
    """
    # Prepara lista piatta di (model_id, result) da valutare
    work_items: list[tuple[str, dict]] = []
    for model_id, results in inference_results.items():
        if model_id == judge_model_id:
            logger.error(
                "CONFLITTO DI INTERESSE: Judge %s == modello valutato. Skip.", model_id
            )
            continue
        for result in results:
            work_items.append((model_id, result))

    total = len(work_items)
    logger.info(
        "Eval parallela: %d record, %d workers",
        total, EVAL_MAX_WORKERS,
    )

    per_record = []
    progress_lock = threading.Lock()
    completed = [0]

    def _run_one(item: tuple[str, dict]) -> dict | None:
        model_id, result = item
        eval_result = _evaluate_single(
            result, model_id, judge_client, judge_model_id, consensus_judge_model_ids,
        )
        with progress_lock:
            completed[0] += 1
            if completed[0] % 25 == 0 or completed[0] == total:
                pct = completed[0] / total * 100
                logger.info(
                    "  Eval: %d/%d (%.0f%%)",
                    completed[0], total, pct,
                )
        return eval_result

    with ThreadPoolExecutor(max_workers=EVAL_MAX_WORKERS) as executor:
        futures = {executor.submit(_run_one, item): item for item in work_items}
        for future in as_completed(futures):
            try:
                eval_result = future.result()
                if eval_result is not None:
                    per_record.append(eval_result)
            except Exception as e:
                model_id, result = futures[future]
                logger.error(
                    "Eval error model=%s test_id=%s: %s",
                    model_id, result.get("test_id"), e,
                )

    # --- Aggregation ---
    per_task_model, per_model = _aggregate_eval_records(per_record)

    logger.info(
        "Evaluation completata: %d record valutati, %d coppie (task,model)",
        len(per_record), len(per_task_model),
    )

    return {
        "per_record": per_record,
        "per_task_model": per_task_model,
        "per_model": per_model,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluator — Step 4")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--results-json", required=True)
    parser.parse_args()

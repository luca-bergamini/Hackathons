"""Test per src.evaluation — copertura dei principali evaluator e della funzione run."""

import json
from unittest.mock import MagicMock

from src.evaluation.main import (
    TASK_EVALUATORS,
    _execute_code,
    _has_test_code,
    _normalize,
    evaluate_classification,
    evaluate_code_generation,
    evaluate_context_qa,
    evaluate_intent_detection,
    evaluate_metadata_extraction,
    evaluate_pii_redaction,
    evaluate_sql_generation,
    evaluate_text_refinement,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_judge_client(score: float = 0.8, reasoning: str = "ok"):
    """Create a mock judge client that returns a fixed score."""
    client = MagicMock()
    client.invoke.return_value = {
        "output_text": json.dumps({"score": score, "reasoning": reasoning}),
    }
    return client


JUDGE_MODEL = "judge-model-xyz"


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_basic(self):
        assert _normalize("  Hello World  ") == "hello world"

    def test_none(self):
        assert _normalize(None) == ""

    def test_number(self):
        assert _normalize(42) == "42"


# ---------------------------------------------------------------------------
# TASK_EVALUATORS completeness
# ---------------------------------------------------------------------------

class TestTaskEvaluatorsMap:
    """Verifica che tutti i task obbligatori e bonus siano registrati."""

    MANDATORY_TASKS = [
        "classification",
        "context_qa",
        "metadata_extraction",
        "code_generation",
        "translation",
        "summarization",
        "rephrasing",
    ]

    BONUS_TASKS = ["sql_generation", "pii_redaction", "intent_detection"]

    def test_mandatory_tasks_present(self):
        for task in self.MANDATORY_TASKS:
            assert task in TASK_EVALUATORS, f"Missing mandatory evaluator: {task}"

    def test_bonus_tasks_present(self):
        for task in self.BONUS_TASKS:
            assert task in TASK_EVALUATORS, f"Missing bonus evaluator: {task}"


# ---------------------------------------------------------------------------
# Classification evaluator
# ---------------------------------------------------------------------------

class TestEvaluateClassification:
    def test_exact_match(self):
        record = {"output_text": "positive", "ground_truth": "positive"}
        result = evaluate_classification(record, _make_judge_client(1.0), JUDGE_MODEL)
        assert result["exact_match"] == 1.0
        assert result["score"] > 0

    def test_no_match(self):
        record = {"output_text": "negative", "ground_truth": "positive"}
        result = evaluate_classification(record, _make_judge_client(0.0), JUDGE_MODEL)
        assert result["exact_match"] == 0.0

    def test_contains_match(self):
        record = {"output_text": "the label is positive", "ground_truth": "positive"}
        result = evaluate_classification(record, _make_judge_client(0.8), JUDGE_MODEL)
        assert result["contains_match"] == 1.0

    def test_no_ground_truth(self):
        record = {"output_text": "positive", "ground_truth": None}
        result = evaluate_classification(record, _make_judge_client(0.7), JUDGE_MODEL)
        assert result["score"] == 0.7  # only judge
        assert "judge_score" in result

    def test_correct_flag(self):
        record = {"output_text": "positive", "ground_truth": "positive"}
        result = evaluate_classification(record, _make_judge_client(1.0), JUDGE_MODEL)
        assert result["correct"] is True


# ---------------------------------------------------------------------------
# Context QA evaluator
# ---------------------------------------------------------------------------

class TestEvaluateContextQA:
    def test_with_overlap(self):
        record = {"output_text": "Rome is the capital", "ground_truth": "The capital is Rome"}
        result = evaluate_context_qa(record, _make_judge_client(0.9), JUDGE_MODEL)
        assert result["token_overlap_f1"] > 0
        assert result["score"] > 0

    def test_no_overlap(self):
        record = {"output_text": "xyz", "ground_truth": "The capital is Rome"}
        result = evaluate_context_qa(record, _make_judge_client(0.2), JUDGE_MODEL)
        assert result["token_overlap_f1"] == 0.0

    def test_no_ground_truth(self):
        record = {"output_text": "Rome", "ground_truth": None}
        result = evaluate_context_qa(record, _make_judge_client(0.8), JUDGE_MODEL)
        assert result["score"] == 0.8


# ---------------------------------------------------------------------------
# Metadata Extraction evaluator
# ---------------------------------------------------------------------------

class TestEvaluateMetadataExtraction:
    def test_valid_json_with_match(self):
        gt = {"name": "Alice", "age": "30"}
        record = {
            "output_text": json.dumps({"name": "Alice", "age": "30"}),
            "ground_truth": gt,
        }
        result = evaluate_metadata_extraction(record, _make_judge_client(0.9), JUDGE_MODEL)
        assert result["json_valid"] == 1.0
        assert result["key_match"] == 1.0
        assert result["value_accuracy"] == 1.0

    def test_invalid_json(self):
        record = {"output_text": "not valid json {{}}", "ground_truth": None}
        result = evaluate_metadata_extraction(record, _make_judge_client(0.5), JUDGE_MODEL)
        assert result["json_valid"] == 0.0

    def test_partial_json(self):
        record = {
            "output_text": 'Here is the result: {"name": "Bob"}',
            "ground_truth": {"name": "Bob", "age": "25"},
        }
        result = evaluate_metadata_extraction(record, _make_judge_client(0.7), JUDGE_MODEL)
        assert result["json_valid"] == 0.8  # extracted from text
        assert result["key_match"] == 0.5  # 1/2 keys

    def test_ground_truth_as_string(self):
        gt_str = json.dumps({"key": "value"})
        record = {
            "output_text": json.dumps({"key": "value"}),
            "ground_truth": gt_str,
        }
        result = evaluate_metadata_extraction(record, _make_judge_client(1.0), JUDGE_MODEL)
        assert result["key_match"] == 1.0


# ---------------------------------------------------------------------------
# Code Generation evaluator
# ---------------------------------------------------------------------------

class TestEvaluateCodeGeneration:
    def test_valid_code(self):
        record = {"output_text": "def hello():\n    return 'world'"}
        result = evaluate_code_generation(record, _make_judge_client(0.9), JUDGE_MODEL)
        assert result["syntax_valid"] == 1.0
        assert result["score"] > 0

    def test_invalid_code(self):
        record = {"output_text": "def hello(:\n    return"}
        result = evaluate_code_generation(record, _make_judge_client(0.3), JUDGE_MODEL)
        assert result["syntax_valid"] == 0.0
        assert result["syntax_error"] != ""

    def test_markdown_fence_extraction(self):
        record = {"output_text": "```python\ndef add(a, b):\n    return a + b\n```"}
        result = evaluate_code_generation(record, _make_judge_client(0.9), JUDGE_MODEL)
        assert result["syntax_valid"] == 1.0

    def test_runtime_valid_code(self):
        record = {"output_text": "print('hello')"}
        result = evaluate_code_generation(
            record, _make_judge_client(0.9), JUDGE_MODEL
        )
        assert result["runtime_valid"] is True

    def test_runtime_error_detected(self):
        record = {"output_text": "raise ValueError('boom')"}
        result = evaluate_code_generation(
            record, _make_judge_client(0.5), JUDGE_MODEL
        )
        assert result["syntax_valid"] == 1.0
        assert result["runtime_valid"] is False

    def test_tests_detected_and_passed(self):
        code = "x = 1\nassert x == 1\nprint('ok')"
        record = {"output_text": code}
        result = evaluate_code_generation(
            record, _make_judge_client(0.9), JUDGE_MODEL
        )
        assert result["has_tests"] is True
        assert result["tests_passed"] is True


# ---------------------------------------------------------------------------
# Execute Code and Test Detection (Advanced 3.7)
# ---------------------------------------------------------------------------


class TestExecuteCode:
    def test_valid_code_runs(self):
        result = _execute_code("print('hello')")
        assert result["executed"] is True
        assert result["runtime_valid"] is True
        assert "hello" in result["stdout"]

    def test_runtime_error(self):
        result = _execute_code(
            "raise ValueError('boom')"
        )
        assert result["executed"] is True
        assert result["runtime_valid"] is False


class TestHasTestCode:
    def test_with_assert(self):
        assert _has_test_code("assert 1 == 1") is True

    def test_with_test_function(self):
        assert _has_test_code("def test_foo(): pass") is True

    def test_no_tests(self):
        assert _has_test_code("x = 1\nprint(x)") is False


# ---------------------------------------------------------------------------
# Text Refinement evaluator
# ---------------------------------------------------------------------------

class TestEvaluateTextRefinement:
    def test_translation_with_gt(self):
        record = {
            "output_text": "The cat is on the table",
            "ground_truth": "The cat is on the table",
            "task": "translation",
        }
        result = evaluate_text_refinement(record, _make_judge_client(1.0), JUDGE_MODEL)
        assert result["token_overlap"] == 1.0
        assert result["score"] > 0

    def test_summarization_no_gt(self):
        record = {
            "output_text": "This is a summary of the text.",
            "ground_truth": None,
            "task": "summarization",
        }
        result = evaluate_text_refinement(record, _make_judge_client(0.8), JUDGE_MODEL)
        assert result["score"] == 0.8

    def test_empty_output(self):
        record = {"output_text": "", "ground_truth": "expected", "task": "rephrasing"}
        result = evaluate_text_refinement(record, _make_judge_client(0.0), JUDGE_MODEL)
        assert result["length_score"] == 0.0

    def test_criteria_map_used(self):
        for task_name in ["translation", "summarization", "rephrasing"]:
            record = {"output_text": "text", "ground_truth": None, "task": task_name}
            client = _make_judge_client(0.5)
            evaluate_text_refinement(record, client, JUDGE_MODEL)
            assert client.invoke.called


# ---------------------------------------------------------------------------
# Bonus: SQL Generation evaluator
# ---------------------------------------------------------------------------

class TestEvaluateSQLGeneration:
    def test_valid_sql(self):
        record = {"output_text": "SELECT * FROM users WHERE age > 18", "ground_truth": None}
        result = evaluate_sql_generation(record, _make_judge_client(0.9), JUDGE_MODEL)
        assert result["has_sql_syntax"] == 1.0

    def test_no_sql(self):
        record = {"output_text": "This is plain text", "ground_truth": None}
        result = evaluate_sql_generation(record, _make_judge_client(0.2), JUDGE_MODEL)
        assert result["has_sql_syntax"] == 0.0


# ---------------------------------------------------------------------------
# Bonus: PII Redaction evaluator
# ---------------------------------------------------------------------------

class TestEvaluatePIIRedaction:
    def test_clean_output(self):
        record = {"output_text": "The user said [REDACTED].", "ground_truth": None}
        result = evaluate_pii_redaction(record, _make_judge_client(0.9), JUDGE_MODEL)
        assert result["redaction_score"] == 1.0
        assert result["pii_found"] == 0

    def test_pii_found(self):
        record = {"output_text": "Contact john@example.com or 123-456-7890", "ground_truth": None}
        result = evaluate_pii_redaction(record, _make_judge_client(0.3), JUDGE_MODEL)
        assert result["pii_found"] >= 1
        assert result["redaction_score"] < 1.0


# ---------------------------------------------------------------------------
# Bonus: Intent Detection evaluator
# ---------------------------------------------------------------------------

class TestEvaluateIntentDetection:
    def test_exact_match(self):
        record = {"output_text": "booking", "ground_truth": "booking"}
        result = evaluate_intent_detection(record, _make_judge_client(1.0), JUDGE_MODEL)
        assert result["exact_match"] == 1.0

    def test_no_gt(self):
        record = {"output_text": "question", "ground_truth": None}
        result = evaluate_intent_detection(record, _make_judge_client(0.7), JUDGE_MODEL)
        assert result["score"] == 0.7


# ---------------------------------------------------------------------------
# run() — orchestratore evaluation
# ---------------------------------------------------------------------------

class TestRun:
    def _mock_inference_results(self):
        return {
            "model-a": [
                {
                    "test_id": "t1",
                    "model_id": "model-a",
                    "output_text": "positive",
                    "task": "classification",
                    "ground_truth": "positive",
                    "legacy_output": "positive",
                    "success": True,
                },
                {
                    "test_id": "t2",
                    "model_id": "model-a",
                    "output_text": "",
                    "task": "classification",
                    "ground_truth": "negative",
                    "legacy_output": "negative",
                    "success": False,
                    "error": "timeout",
                },
            ],
        }

    def test_basic_run(self):
        results = self._mock_inference_results()
        judge = _make_judge_client(0.8)
        output = run(results, judge, JUDGE_MODEL)
        assert "per_record" in output
        assert "per_task_model" in output
        assert "per_model" in output
        assert len(output["per_record"]) == 2

    def test_failed_record_gets_zero(self):
        results = self._mock_inference_results()
        judge = _make_judge_client(0.8)
        output = run(results, judge, JUDGE_MODEL)
        failed = [r for r in output["per_record"] if r["test_id"] == "t2"][0]
        assert failed["score"] == 0.0
        assert failed["correct"] is False

    def test_conflict_of_interest_skip(self):
        results = {"judge-model-xyz": [
            {
                "test_id": "t1",
                "model_id": "judge-model-xyz",
                "output_text": "positive",
                "task": "classification",
                "ground_truth": "positive",
                "legacy_output": "positive",
                "success": True,
            },
        ]}
        judge = _make_judge_client(1.0)
        output = run(results, judge, JUDGE_MODEL)
        assert len(output["per_record"]) == 0

    def test_unknown_task_skipped(self):
        results = {"model-a": [
            {
                "test_id": "t1",
                "model_id": "model-a",
                "output_text": "text",
                "task": "unknown_task_xyz",
                "ground_truth": None,
                "legacy_output": None,
                "success": True,
            },
        ]}
        judge = _make_judge_client(0.5)
        output = run(results, judge, JUDGE_MODEL)
        assert len(output["per_record"]) == 0

    def test_aggregation_computed(self):
        results = {
            "model-a": [
                {
                    "test_id": f"t{i}",
                    "model_id": "model-a",
                    "output_text": "positive",
                    "task": "classification",
                    "ground_truth": "positive",
                    "legacy_output": "positive",
                    "success": True,
                }
                for i in range(3)
            ],
        }
        judge = _make_judge_client(0.8)
        output = run(results, judge, JUDGE_MODEL)
        assert "classification|model-a" in output["per_task_model"]
        assert output["per_task_model"]["classification|model-a"]["num_records"] == 3
        assert "model-a" in output["per_model"]
        assert output["per_model"]["model-a"]["num_records"] == 3

    def test_multiple_models(self):
        results = {
            "model-a": [
                {
                    "test_id": "t1", "model_id": "model-a",
                    "output_text": "pos", "task": "classification",
                    "ground_truth": "pos", "legacy_output": "pos", "success": True,
                },
            ],
            "model-b": [
                {
                    "test_id": "t1", "model_id": "model-b",
                    "output_text": "neg", "task": "classification",
                    "ground_truth": "pos", "legacy_output": "pos", "success": True,
                },
            ],
        }
        judge = _make_judge_client(0.6)
        output = run(results, judge, JUDGE_MODEL)
        assert "model-a" in output["per_model"]
        assert "model-b" in output["per_model"]


# ---------------------------------------------------------------------------
# Edge cases: Judge error handling
# ---------------------------------------------------------------------------

class TestLLMJudgeErrorHandling:
    def test_judge_returns_invalid_json(self):
        client = MagicMock()
        client.invoke.return_value = {"output_text": "not json at all"}
        record = {"output_text": "positive", "ground_truth": "positive"}
        result = evaluate_classification(record, client, JUDGE_MODEL)
        assert result["judge_score"] == 0.0

    def test_judge_raises_exception(self):
        client = MagicMock()
        client.invoke.side_effect = RuntimeError("API error")
        record = {"output_text": "positive", "ground_truth": "positive"}
        result = evaluate_classification(record, client, JUDGE_MODEL)
        assert result["judge_score"] == 0.0
        assert "failed" in result["judge_reasoning"]

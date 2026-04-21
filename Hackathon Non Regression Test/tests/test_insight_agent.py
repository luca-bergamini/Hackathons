"""Test per src.insight_agent — coverage del modulo Insight Agent."""

import json
from unittest.mock import MagicMock

from src.insight_agent.main import (
    compute_stats,
    generate_insights,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_eval_results():
    return {
        "per_record": [
            {
                "test_id": "t1", "model_id": "m-a", "task": "classification",
                "score": 0.9, "correct": True, "ground_truth": "pos",
            },
            {
                "test_id": "t2", "model_id": "m-a", "task": "classification",
                "score": 0.0, "correct": False, "ground_truth": "neg",
                "error": "timeout",
            },
            {
                "test_id": "t1", "model_id": "m-b", "task": "classification",
                "score": 0.8, "correct": True, "ground_truth": "pos",
            },
            {
                "test_id": "t3", "model_id": "m-a", "task": "context_qa",
                "score": 0.7, "correct": True, "ground_truth": "Rome",
            },
        ],
        "per_task_model": {},
        "per_model": {},
    }


def _make_judge_client(insight_text: str | None = None):
    if insight_text is None:
        insight_text = json.dumps({
            "summary": "Model A performs well overall.",
            "per_task_analysis": {"classification": "Good accuracy."},
            "anomalies_analysis": "Score 0 on t2 is notable.",
            "recommendation": "Promote Model A.",
        })
    client = MagicMock()
    client.invoke.return_value = {
        "output_text": insight_text,
        "input_tokens": 500,
        "output_tokens": 200,
    }
    return client


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_basic(self):
        stats = compute_stats(_mock_eval_results())
        assert stats["total_records"] == 4
        assert "per_task" in stats
        assert "per_model" in stats

    def test_per_task(self):
        stats = compute_stats(_mock_eval_results())
        assert "classification" in stats["per_task"]
        cls_stats = stats["per_task"]["classification"]
        assert cls_stats["total_records"] == 3
        assert cls_stats["best_model"] != ""

    def test_per_model(self):
        stats = compute_stats(_mock_eval_results())
        assert "m-a" in stats["per_model"]
        assert stats["per_model"]["m-a"]["total_records"] == 3

    def test_anomalies_detected(self):
        stats = compute_stats(_mock_eval_results())
        assert len(stats["anomalies"]) > 0

    def test_empty_input(self):
        stats = compute_stats({"per_record": []})
        assert stats["total_records"] == 0
        assert stats["per_task"] == {}

    def test_error_patterns_for_low_scores(self):
        eval_data = {
            "per_record": [
                {
                    "test_id": f"t{i}", "model_id": "m", "task": "hard_task",
                    "score": 0.1, "correct": False, "ground_truth": "x",
                }
                for i in range(5)
            ],
        }
        stats = compute_stats(eval_data)
        assert len(stats["error_patterns"]) > 0
        assert stats["error_patterns"][0]["task"] == "hard_task"

    def test_high_error_rate_anomaly(self):
        eval_data = {
            "per_record": [
                {
                    "test_id": f"t{i}", "model_id": "bad-model",
                    "task": "classification", "score": 0.0,
                    "correct": False, "ground_truth": "x",
                    "error": "fail",
                }
                for i in range(5)
            ],
        }
        stats = compute_stats(eval_data)
        model_info = stats["per_model"]["bad-model"]
        assert model_info["error_rate"] == 1.0
        anomalies = [a for a in stats["anomalies"] if "error rate" in a.get("reason", "").lower()]
        assert len(anomalies) > 0


# ---------------------------------------------------------------------------
# generate_insights
# ---------------------------------------------------------------------------

class TestGenerateInsights:
    def test_basic(self):
        stats = compute_stats(_mock_eval_results())
        client = _make_judge_client()
        insights = generate_insights(stats, client, "judge-model")
        assert "summary" in insights
        assert insights["summary"] != ""
        assert client.invoke.called

    def test_invalid_json_response(self):
        stats = compute_stats(_mock_eval_results())
        client = _make_judge_client("Not valid json at all")
        insights = generate_insights(stats, client, "judge-model")
        assert "summary" in insights

    def test_llm_error_handled(self):
        stats = compute_stats(_mock_eval_results())
        client = MagicMock()
        client.invoke.side_effect = RuntimeError("API error")
        insights = generate_insights(stats, client, "judge-model")
        assert "failed" in insights["summary"].lower()


# ---------------------------------------------------------------------------
# run (orchestrator)
# ---------------------------------------------------------------------------

class TestRun:
    def test_basic(self):
        client = _make_judge_client()
        result = run(_mock_eval_results(), client, "judge-model")
        assert "stats" in result
        assert "insights" in result
        assert result["stats"]["total_records"] == 4
        assert result["insights"]["summary"] != ""

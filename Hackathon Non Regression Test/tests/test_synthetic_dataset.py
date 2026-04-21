"""Test per src.synthetic_dataset — coverage del modulo Synthetic Dataset Generation."""

import json
import os
import tempfile
from unittest.mock import MagicMock

from src.synthetic_dataset.main import (
    _build_records,
    _format_seeds,
    _infer_output_type,
    enrich_dataset,
    generate_records_for_task,
    run,
    save_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_client(examples: list[dict] | None = None):
    """Mock BedrockClient che ritorna record sintetici."""
    if examples is None:
        examples = [
            {
                "user_content": "Sample text for testing",
                "ground_truth": "label_a",
                "expected_output_type": "label",
            },
            {
                "user_content": "Another test sample",
                "ground_truth": "label_b",
                "expected_output_type": "label",
            },
        ]
    response_json = json.dumps({"examples": examples})
    client = MagicMock()
    client.invoke.return_value = {
        "output_text": response_json,
        "input_tokens": 200,
        "output_tokens": 300,
    }
    return client


def _sample_seed_records(task: str = "classification") -> list[dict]:
    return [
        {
            "test_id": "seed_1",
            "input_messages": [
                {"role": "system", "content": "Classify the text into Sport or Politics."},
                {"role": "user", "content": "The team won the championship."},
            ],
            "ground_truth": "Sport",
            "expected_output_type": "label",
            "metadata": {},
            "agent_id": "classifier_v1",
            "output": "Sport",
        },
    ]


# ---------------------------------------------------------------------------
# Test _infer_output_type
# ---------------------------------------------------------------------------

class TestInferOutputType:
    def test_known_tasks(self):
        assert _infer_output_type("classification") == "label"
        assert _infer_output_type("context_qa") == "text"
        assert _infer_output_type("metadata_extraction") == "json"
        assert _infer_output_type("code_generation") == "code"
        assert _infer_output_type("sql_generation") == "sql"

    def test_unknown_task_defaults_text(self):
        assert _infer_output_type("unknown_task") == "text"


# ---------------------------------------------------------------------------
# Test _format_seeds
# ---------------------------------------------------------------------------

class TestFormatSeeds:
    def test_no_seeds(self):
        assert "No seed" in _format_seeds(None)
        assert "No seed" in _format_seeds([])

    def test_with_seeds(self):
        seeds = _sample_seed_records()
        result = _format_seeds(seeds)
        assert "Example 1" in result
        assert "championship" in result

    def test_truncates_long_content(self):
        record = {
            "input_messages": [
                {"role": "user", "content": "x" * 1000},
            ],
            "ground_truth": "something",
        }
        result = _format_seeds([record])
        assert "..." in result


# ---------------------------------------------------------------------------
# Test _build_records
# ---------------------------------------------------------------------------

class TestBuildRecords:
    def test_builds_valid_records(self):
        raw = [
            {"user_content": "Hello world", "ground_truth": "greeting"},
            {"user_content": "What is 2+2?", "ground_truth": "4"},
        ]
        records = _build_records(raw, "classification", "agent_1", "System prompt", "label")
        assert len(records) == 2
        for r in records:
            assert r["test_id"].startswith("synth_")
            assert len(r["input_messages"]) == 2
            assert r["input_messages"][0]["role"] == "system"
            assert r["input_messages"][1]["role"] == "user"
            assert r["agent_id"] == "agent_1"
            assert r["output"] is None  # sintetico, no legacy output

    def test_skips_empty_user_content(self):
        raw = [
            {"user_content": "", "ground_truth": "x"},
            {"user_content": "valid", "ground_truth": "y"},
        ]
        records = _build_records(raw, "classification", "a", "s", "label")
        assert len(records) == 1

    def test_skips_non_dict(self):
        raw = ["not a dict", {"user_content": "ok", "ground_truth": "y"}]
        records = _build_records(raw, "classification", "a", "s", "label")
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Test generate_records_for_task
# ---------------------------------------------------------------------------

class TestGenerateRecordsForTask:
    def test_generates_records(self):
        client = _make_mock_client()
        records = generate_records_for_task("classification", 2, client, "model_x")
        assert len(records) == 2
        assert all(r["test_id"].startswith("synth_") for r in records)

    def test_with_seed_records(self):
        client = _make_mock_client()
        seeds = _sample_seed_records()
        records = generate_records_for_task(
            "classification", 2, client, "model_x",
            seed_records=seeds,
        )
        assert len(records) == 2
        # Should use agent_id from seeds
        assert records[0]["agent_id"] == "classifier_v1"

    def test_returns_empty_on_error(self):
        client = MagicMock()
        client.invoke.side_effect = Exception("API error")
        records = generate_records_for_task("classification", 5, client, "model_x")
        assert records == []


# ---------------------------------------------------------------------------
# Test save_dataset
# ---------------------------------------------------------------------------

class TestSaveDataset:
    def test_saves_jsonl(self):
        records = [
            {"test_id": "t1", "data": "hello"},
            {"test_id": "t2", "data": "world"},
        ]
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)

        try:
            save_dataset(records, path)
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["test_id"] == "t1"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test enrich_dataset
# ---------------------------------------------------------------------------

class TestEnrichDataset:
    def test_merges_without_duplicates(self):
        original = [{"test_id": "a"}, {"test_id": "b"}]
        synthetic = [{"test_id": "b"}, {"test_id": "c"}]
        result = enrich_dataset(original, synthetic)
        assert len(result) == 3
        ids = {r["test_id"] for r in result}
        assert ids == {"a", "b", "c"}

    def test_empty_synthetic(self):
        original = [{"test_id": "a"}]
        result = enrich_dataset(original, [])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test run (full flow, mocked LLM)
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_generates_for_multiple_tasks(self):
        client = _make_mock_client()
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)

        try:
            records = run(
                tasks=["classification", "context_qa"],
                llm_client=client,
                model_id="model_x",
                records_per_task=2,
                output_path=path,
            )
            assert len(records) == 4  # 2 per task
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 4
        finally:
            os.unlink(path)

    def test_run_with_seeds(self):
        client = _make_mock_client()
        seeds = {"classification": _sample_seed_records()}
        records = run(
            tasks=["classification"],
            llm_client=client,
            model_id="model_x",
            records_per_task=2,
            output_path="",
            seed_records_by_task=seeds,
        )
        assert len(records) == 2

"""Test per src.runner."""

import json
import os
import shutil
from unittest.mock import MagicMock

from src.runner.main import (
    MAX_RETRIES,
    run_single_inference,
    run_task_model_pair,
)


def _make_record(**overrides) -> dict:
    base = {
        "test_id": "t001",
        "input_messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello."},
        ],
        "ground_truth": "hello",
        "expected_output_type": "text",
        "metadata": {},
        "agent_id": "agent_01",
        "output": "hello",
        "task": "classification",
    }
    base.update(overrides)
    return base


def _mock_client(output_text="hello", input_tokens=10, output_tokens=5):
    client = MagicMock()
    client.invoke.return_value = {
        "output_text": output_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stop_reason": "end_turn",
    }
    return client


class TestRunSingleInference:
    def setup_method(self):
        if os.path.exists("payloads"):
            shutil.rmtree("payloads")

    def teardown_method(self):
        if os.path.exists("payloads"):
            shutil.rmtree("payloads")

    def test_successful_inference(self):
        client = _mock_client()
        result = run_single_inference(_make_record(), "test-model", client)
        assert result["success"] is True
        assert result["output_text"] == "hello"
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5
        assert result["retries"] == 0
        assert result["error"] is None

    def test_retries_on_failure(self):
        client = MagicMock()
        client.invoke.side_effect = Exception("API error")
        result = run_single_inference(_make_record(), "test-model", client)
        assert result["success"] is False
        assert result["retries"] == MAX_RETRIES
        assert "API error" in result["error"]

    def test_payload_saved(self):
        client = _mock_client()
        run_single_inference(_make_record(), "test-model", client)
        assert os.path.exists("payloads")
        files = os.listdir("payloads")
        assert len(files) == 1
        with open(os.path.join("payloads", files[0]), encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["test_id"] == "t001"
        assert payload["model_id"] == "test-model"


class TestRunTaskModelPair:
    def setup_method(self):
        if os.path.exists("payloads"):
            shutil.rmtree("payloads")

    def teardown_method(self):
        if os.path.exists("payloads"):
            shutil.rmtree("payloads")

    def test_processes_all_records(self):
        client = _mock_client()
        records = [_make_record(test_id=f"t{i}") for i in range(5)]
        results = run_task_model_pair(records, "test-model", client)
        assert len(results) == 5
        assert all(r["success"] for r in results)


class TestRun:
    """Test the parallel orchestrator run()."""

    def setup_method(self):
        if os.path.exists("payloads"):
            shutil.rmtree("payloads")

    def teardown_method(self):
        if os.path.exists("payloads"):
            shutil.rmtree("payloads")

    def test_run_with_single_model(self):
        from src.runner.main import run
        client = _mock_client()
        records = [
            _make_record(test_id="t1", task="classification"),
            _make_record(test_id="t2", task="classification"),
        ]
        model_configs = [{"model_id": "model-a", "max_tokens": 512}]
        results = run(records, model_configs, client)
        assert "model-a" in results
        assert len(results["model-a"]) == 2
        assert all(r["success"] for r in results["model-a"])

    def test_run_with_multiple_models(self):
        from src.runner.main import run
        client = _mock_client()
        records = [_make_record(test_id="t1", task="classification")]
        model_configs = [
            {"model_id": "model-a"},
            {"model_id": "model-b"},
        ]
        results = run(records, model_configs, client)
        assert "model-a" in results
        assert "model-b" in results
        assert len(results["model-a"]) == 1
        assert len(results["model-b"]) == 1

    def test_run_groups_by_task(self):
        from src.runner.main import run
        client = _mock_client()
        records = [
            _make_record(test_id="t1", task="classification"),
            _make_record(test_id="t2", task="context_qa"),
        ]
        model_configs = [{"model_id": "model-a"}]
        results = run(records, model_configs, client)
        assert len(results["model-a"]) == 2
        tasks = {r["task"] for r in results["model-a"]}
        assert tasks == {"classification", "context_qa"}

    def test_run_handles_future_exception(self):
        from src.runner.main import run
        client = MagicMock()
        client.invoke.side_effect = Exception("API error")
        records = [_make_record(test_id="t1", task="classification")]
        model_configs = [{"model_id": "model-a"}]
        results = run(records, model_configs, client)
        assert "model-a" in results
        # Records attempted but all failed
        assert len(results["model-a"]) == 1
        assert results["model-a"][0]["success"] is False

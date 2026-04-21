"""Test per src.data_processing."""

import csv
import json
import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from src.data_processing.main import (
    REQUIRED_FIELDS,
    _classify_task_with_llm,
    _extract_messages,
    _save_payload,
    _score_task,
    generate_profiling_csv,
    identify_task,
    load_dataset_from_s3,
    run,
    split_by_agent,
    validate_record,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _make_record(**overrides) -> dict:
    """Crea un record valido con valori di default."""
    base = {
        "test_id": "t001",
        "input_messages": [
            {"role": "system", "content": "You are a classifier."},
            {"role": "user", "content": "Classify this text."},
        ],
        "ground_truth": "positive",
        "expected_output_type": "label",
        "metadata": {"model": "legacy"},
        "agent_id": "agent_01",
        "output": "positive",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# validate_record
# ---------------------------------------------------------------------------
class TestValidateRecord:
    def test_valid_record(self):
        assert validate_record(_make_record()) is True

    def test_missing_field(self):
        rec = _make_record()
        del rec["test_id"]
        assert validate_record(rec) is False

    def test_missing_system_message(self):
        rec = _make_record(input_messages=[{"role": "user", "content": "hi"}])
        assert validate_record(rec) is False

    def test_missing_user_message(self):
        rec = _make_record(input_messages=[{"role": "system", "content": "hi"}])
        assert validate_record(rec) is False

    def test_input_messages_not_list(self):
        rec = _make_record(input_messages="not a list")
        assert validate_record(rec) is False

    def test_empty_input_messages(self):
        rec = _make_record(input_messages=[])
        assert validate_record(rec) is False

    def test_ground_truth_null_still_valid(self):
        rec = _make_record(ground_truth=None)
        assert validate_record(rec) is True

    def test_all_required_fields_present(self):
        rec = _make_record()
        assert REQUIRED_FIELDS.issubset(rec.keys())


# ---------------------------------------------------------------------------
# split_by_agent
# ---------------------------------------------------------------------------
class TestSplitByAgent:
    def test_groups_correctly(self):
        records = [
            _make_record(agent_id="a1"),
            _make_record(agent_id="a2"),
            _make_record(agent_id="a1"),
        ]
        groups = split_by_agent(records)
        assert len(groups) == 2
        assert len(groups["a1"]) == 2
        assert len(groups["a2"]) == 1

    def test_missing_agent_id(self):
        rec = _make_record()
        del rec["agent_id"]
        groups = split_by_agent([rec])
        assert "unknown" in groups

    def test_empty_list(self):
        assert split_by_agent([]) == {}


# ---------------------------------------------------------------------------
# _extract_messages
# ---------------------------------------------------------------------------
class TestExtractMessages:
    def test_extracts_system_and_user(self):
        records = [_make_record()]
        sys_text, usr_text = _extract_messages(records)
        assert "classifier" in sys_text.lower()
        assert "classify" in usr_text.lower()

    def test_max_user_samples(self):
        records = [
            _make_record(input_messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"msg{i}"},
            ])
            for i in range(10)
        ]
        _, usr_text = _extract_messages(records)
        assert usr_text.count("msg") == 5

    def test_handles_non_list_messages(self):
        rec = _make_record(input_messages="broken")
        sys_text, usr_text = _extract_messages([rec])
        assert sys_text == ""
        assert usr_text == ""


# ---------------------------------------------------------------------------
# _score_task
# ---------------------------------------------------------------------------
class TestScoreTask:
    def test_classification_scores_on_label_output(self):
        score = _score_task("classification", "classify the text", "", "label")
        assert score > 0

    def test_code_generation_scores_on_code_output(self):
        score = _score_task("code_generation", "generate python code", "", "code")
        assert score > 0

    def test_no_match_returns_zero(self):
        score = _score_task("sql_generation", "hello world", "hello", "text")
        assert score == 0.0

    def test_json_output_boosts_metadata_extraction(self):
        score = _score_task("metadata_extraction", "extract data", "", "json")
        assert score >= 5.0


# ---------------------------------------------------------------------------
# identify_task (heuristic fallback — LLM mocked)
# ---------------------------------------------------------------------------
class TestIdentifyTask:
    @patch("src.data_processing.main._classify_task_with_llm", return_value=None)
    def test_classification_task(self, _mock):
        records = [_make_record(
            input_messages=[
                {"role": "system", "content": "Classify the sentiment using the label space."},
                {"role": "user", "content": "Label space: positive|negative\nText: great!"},
            ],
            expected_output_type="label",
        )]
        assert identify_task(records) == "classification"

    @patch("src.data_processing.main._classify_task_with_llm", return_value=None)
    def test_code_generation_task(self, _mock):
        records = [_make_record(
            input_messages=[
                {"role": "system", "content": "Generate working Python 3.11 code."},
                {"role": "user", "content": "Implement def factorial(n):"},
            ],
            expected_output_type="code",
        )]
        assert identify_task(records) == "code_generation"

    @patch("src.data_processing.main._classify_task_with_llm", return_value=None)
    def test_sql_generation_task(self, _mock):
        records = [_make_record(
            input_messages=[
                {"role": "system", "content": "You are given a SQLite database. Generate SQL."},
                {"role": "user", "content": "SELECT all customers from table."},
            ],
            expected_output_type="sql",
        )]
        assert identify_task(records) == "sql_generation"

    @patch("src.data_processing.main._classify_task_with_llm", return_value=None)
    def test_summarization_task(self, _mock):
        records = [_make_record(
            input_messages=[
                {"role": "system", "content": "Summarize the text in 2-3 concise sentences."},
                {"role": "user", "content": "Meeting notes: the team discussed..."},
            ],
            expected_output_type="text",
        )]
        assert identify_task(records) == "summarization"

    @patch("src.data_processing.main._classify_task_with_llm", return_value=None)
    def test_pii_redaction_task(self, _mock):
        records = [_make_record(
            input_messages=[
                {"role": "system", "content": "Redact all PII from the text. Use [REDACTED]."},
                {"role": "user", "content": "John Doe, john@email.com"},
            ],
            expected_output_type="text",
        )]
        assert identify_task(records) == "pii_redaction"

    @patch("src.data_processing.main._classify_task_with_llm", return_value=None)
    def test_empty_records(self, _mock):
        assert identify_task([]) == "classification"

    def test_llm_result_used_when_available(self):
        with patch(
            "src.data_processing.main._classify_task_with_llm",
            return_value="translation",
        ):
            records = [_make_record()]
            assert identify_task(records) == "translation"


# ---------------------------------------------------------------------------
# generate_profiling_csv
# ---------------------------------------------------------------------------
class TestGenerateProfilingCsv:
    def test_creates_correct_csv(self):
        profiles = {
            "agent_01": {
                "task": "classification",
                "records": [
                    _make_record(),
                    _make_record(ground_truth=None),
                    _make_record(),
                ],
            },
        }
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)

        try:
            generate_profiling_csv(profiles, path)
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            row = rows[0]
            assert row["agent_id"] == "agent_01"
            assert row["task"] == "classification"
            assert row["numero_record"] == "3"
            assert row["numero_record_malformati"] == "0"
            assert float(row["percentuale_ground_truth"]) == 66.7
        finally:
            os.unlink(path)

    def test_malformed_records_counted(self):
        bad_rec = {"test_id": "t1"}
        profiles = {
            "agent_x": {
                "task": "translation",
                "records": [_make_record(), bad_rec],
            },
        }
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)

        try:
            generate_profiling_csv(profiles, path)
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["numero_record_malformati"] == "1"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _save_payload
# ---------------------------------------------------------------------------
class TestSavePayload:
    def test_saves_json_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.data_processing.main.PAYLOAD_DIR", tmp_path / "payloads",
        )
        _save_payload("call1", "test-model", {"q": 1}, {"a": 2})
        files = list((tmp_path / "payloads").glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["call_id"] == "call1"
        assert data["model_id"] == "test-model"


# ---------------------------------------------------------------------------
# load_dataset_from_s3
# ---------------------------------------------------------------------------
class TestLoadDatasetFromS3:
    def test_loads_jsonl(self):
        records = [
            {"test_id": "t1", "agent_id": "a1"},
            {"test_id": "t2", "agent_id": "a2"},
        ]
        body = "\n".join(json.dumps(r) for r in records)
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(body.encode("utf-8")),
        }
        with patch("src.data_processing.main.boto3.client", return_value=mock_s3):
            result = load_dataset_from_s3("bucket", "key.jsonl")
        assert len(result) == 2
        assert result[0]["test_id"] == "t1"

    def test_skips_malformed_lines(self):
        body = '{"ok": true}\nNOT_JSON\n{"ok": false}'
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(body.encode("utf-8")),
        }
        with patch("src.data_processing.main.boto3.client", return_value=mock_s3):
            result = load_dataset_from_s3("bucket", "key.jsonl")
        assert len(result) == 2

    def test_empty_lines_skipped(self):
        body = '{"a":1}\n\n\n{"b":2}\n'
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(body.encode("utf-8")),
        }
        with patch("src.data_processing.main.boto3.client", return_value=mock_s3):
            result = load_dataset_from_s3("bucket", "key.jsonl")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _classify_task_with_llm
# ---------------------------------------------------------------------------
class TestClassifyTaskWithLLM:
    def test_returns_valid_task(self):
        with patch("src.data_processing.main._get_bedrock_client") as mock_get:
            mock_client = MagicMock()
            mock_client.invoke.return_value = {
                "output_text": "classification",
                "input_tokens": 10,
                "output_tokens": 5,
            }
            mock_get.return_value = mock_client
            with patch("src.data_processing.main._save_payload"):
                result = _classify_task_with_llm(
                    "classify text", "some text", "label", "agent_01",
                )
        assert result == "classification"

    def test_returns_none_on_error(self):
        with patch("src.data_processing.main._get_bedrock_client") as mock_get:
            mock_get.return_value.invoke.side_effect = RuntimeError("fail")
            with patch("src.data_processing.main._save_payload"):
                result = _classify_task_with_llm(
                    "sys", "usr", "label", "agent_01",
                )
        assert result is None

    def test_returns_none_on_unexpected_task(self):
        with patch("src.data_processing.main._get_bedrock_client") as mock_get:
            mock_client = MagicMock()
            mock_client.invoke.return_value = {
                "output_text": "unknown_xyz_task",
                "input_tokens": 10,
                "output_tokens": 5,
            }
            mock_get.return_value = mock_client
            with patch("src.data_processing.main._save_payload"):
                result = _classify_task_with_llm(
                    "sys", "usr", "label", "a1",
                )
        assert result is None

    def test_fuzzy_match_works(self):
        with patch("src.data_processing.main._get_bedrock_client") as mock_get:
            mock_client = MagicMock()
            mock_client.invoke.return_value = {
                "output_text": "the task is context_qa based on...",
                "input_tokens": 10,
                "output_tokens": 5,
            }
            mock_get.return_value = mock_client
            with patch("src.data_processing.main._save_payload"):
                result = _classify_task_with_llm(
                    "sys", "usr", "text", "a1",
                )
        assert result == "context_qa"


# ---------------------------------------------------------------------------
# run() orchestrator
# ---------------------------------------------------------------------------
class TestRun:
    def test_run_e2e(self, tmp_path, monkeypatch):
        records = [
            _make_record(test_id=f"t{i}", agent_id="agent_01")
            for i in range(3)
        ]
        body = "\n".join(json.dumps(r) for r in records)
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": BytesIO(body.encode("utf-8")),
        }
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        csv_path = str(tmp_path / "output.csv")

        with patch("src.data_processing.main.boto3.client", return_value=mock_s3), \
             patch(
                 "src.data_processing.main._classify_task_with_llm",
                 return_value="classification",
             ):
            result = run(
                s3_bucket="test-bucket",
                dataset_key="dataset/dataset.jsonl",
                output_csv=csv_path,
            )

        assert "agents" in result
        assert "profiles" in result
        assert "agent_01" in result["agents"]
        assert os.path.exists(csv_path)

    def test_run_raises_without_bucket(self):
        with pytest.raises(ValueError, match="S3_BUCKET"):
            run(s3_bucket="")

"""Test per src.prompt_optimizer — coverage del modulo Prompt Optimization."""

from unittest.mock import MagicMock, patch

from src.prompt_optimizer.main import (
    DEFAULT_BEAM_WIDTH,
    _format_eval_feedback,
    _replace_system_prompt,
    evaluate,
    generate,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_client(output_text: str = '{"variants": ["v1", "v2", "v3"]}'):
    """Crea un mock BedrockClient che ritorna output_text fisso."""
    client = MagicMock()
    client.invoke.return_value = {
        "output_text": output_text,
        "input_tokens": 100,
        "output_tokens": 50,
    }
    return client


def _sample_records(task: str = "classification", n: int = 2) -> list[dict]:
    return [
        {
            "test_id": f"test_{i}",
            "input_messages": [
                {"role": "system", "content": "Classify the text."},
                {"role": "user", "content": f"Sample text {i}"},
            ],
            "ground_truth": "label_a",
            "expected_output_type": "label",
            "metadata": {},
            "agent_id": "agent_1",
            "output": "label_a",
            "task": task,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test _replace_system_prompt
# ---------------------------------------------------------------------------

class TestReplaceSystemPrompt:
    def test_replaces_existing_system_message(self):
        records = _sample_records()
        modified = _replace_system_prompt(records, "NEW PROMPT")
        for r in modified:
            system_msg = [m for m in r["input_messages"] if m["role"] == "system"]
            assert len(system_msg) == 1
            assert system_msg[0]["content"] == "NEW PROMPT"

    def test_adds_system_if_missing(self):
        records = [
            {
                "test_id": "t1",
                "input_messages": [{"role": "user", "content": "hello"}],
                "task": "classification",
            }
        ]
        modified = _replace_system_prompt(records, "ADDED PROMPT")
        assert modified[0]["input_messages"][0]["role"] == "system"
        assert modified[0]["input_messages"][0]["content"] == "ADDED PROMPT"

    def test_does_not_mutate_originals(self):
        records = _sample_records()
        original_content = records[0]["input_messages"][0]["content"]
        _replace_system_prompt(records, "CHANGED")
        assert records[0]["input_messages"][0]["content"] == original_content


# ---------------------------------------------------------------------------
# Test _format_eval_feedback
# ---------------------------------------------------------------------------

class TestFormatEvalFeedback:
    def test_empty_results(self):
        assert _format_eval_feedback({}) == "No previous evaluation available."
        assert _format_eval_feedback(None) == "No previous evaluation available."

    def test_with_score(self):
        feedback = _format_eval_feedback({"avg_score": 0.75, "num_records": 10, "num_correct": 7})
        assert "0.750" in feedback
        assert "10" in feedback
        assert "7" in feedback

    def test_with_common_errors(self):
        feedback = _format_eval_feedback({
            "avg_score": 0.5,
            "num_records": 5,
            "num_correct": 2,
            "common_errors": ["wrong label", "empty output", "timeout"],
        })
        assert "Common errors" in feedback
        assert "wrong label" in feedback

    def test_score_below_target(self):
        feedback = _format_eval_feedback({"avg_score": None})
        assert feedback == "Score below target."


# ---------------------------------------------------------------------------
# Test generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_returns_variants_from_llm(self):
        client = _make_mock_client('{"variants": ["prompt_a", "prompt_b", "prompt_c"]}')
        result = generate("agent_1", "model_x", "base prompt", {}, client, num_variants=3)
        assert len(result) == 3
        assert result[0] == "prompt_a"

    def test_fallback_on_llm_error(self):
        client = MagicMock()
        client.invoke.side_effect = Exception("API error")
        result = generate("agent_1", "model_x", "base prompt", {}, client, num_variants=3)
        assert len(result) == 3
        assert all(isinstance(v, str) and len(v) > 0 for v in result)

    def test_fallback_on_invalid_json(self):
        client = _make_mock_client("not valid json at all")
        result = generate("agent_1", "model_x", "base prompt", {}, client, num_variants=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Test evaluate (mocked Runner + Evaluator)
# ---------------------------------------------------------------------------

class TestEvaluate:
    @patch("src.prompt_optimizer.main.evaluation")
    @patch("src.prompt_optimizer.main.runner")
    def test_evaluate_returns_scores(self, mock_runner, mock_eval):
        mock_runner.run.return_value = {
            "model_x": [
                {
                    "test_id": "t1",
                    "success": True,
                    "output_text": "a",
                    "task": "classification",
                }
            ]
        }
        mock_eval.run.return_value = {
            "per_record": [{"test_id": "t1", "score": 0.8, "correct": True}],
        }
        client = _make_mock_client()
        result = evaluate(
            "agent_1", "model_x", "prompt",
            _sample_records(), client, client, "judge_model",
        )
        assert result["avg_score"] == 0.8
        assert result["num_correct"] == 1
        assert result["num_records"] == 1


# ---------------------------------------------------------------------------
# Test run (full loop, mocked internals)
# ---------------------------------------------------------------------------

class TestRun:
    @patch("src.prompt_optimizer.main.evaluate")
    @patch("src.prompt_optimizer.main.generate")
    def test_run_improves_prompt(self, mock_generate, mock_evaluate):
        """Simula un loop dove la variante migliora il punteggio."""
        mock_evaluate.side_effect = [
            {"avg_score": 0.5, "num_correct": 3, "num_records": 5, "per_record": []},  # baseline
            # variant 1 iter 1
            {"avg_score": 0.7, "num_correct": 4, "num_records": 5, "per_record": []},
            # variant 2 iter 1
            {"avg_score": 0.6, "num_correct": 3, "num_records": 5, "per_record": []},
            # iter 2 — no improvement
            {"avg_score": 0.65, "num_correct": 3, "num_records": 5, "per_record": []},
            {"avg_score": 0.68, "num_correct": 3, "num_records": 5, "per_record": []},
        ]
        mock_generate.return_value = ["improved_v1", "improved_v2"]

        result = run(
            "agent_1", "model_x", "base", _sample_records(),
            _make_mock_client(), max_iterations=2, num_variants=2,
            beam_width=1,
        )

        assert result["baseline_score"] == 0.5
        assert result["best_score"] == 0.7
        assert result["improvement"] > 0
        assert result["best_prompt"] == "improved_v1"
        assert result["beam_width"] == 1

    @patch("src.prompt_optimizer.main.evaluate")
    @patch("src.prompt_optimizer.main.generate")
    def test_run_stops_early_no_improvement(self, mock_generate, mock_evaluate):
        """Se nessuna variante migliora, si ferma prima."""
        mock_evaluate.side_effect = [
            # baseline alta
            {"avg_score": 0.9, "num_correct": 9, "num_records": 10, "per_record": []},
            {"avg_score": 0.85, "num_correct": 8, "num_records": 10, "per_record": []},
        ]
        mock_generate.return_value = ["worse_v1"]

        result = run(
            "agent_1", "model_x", "already good", _sample_records(),
            _make_mock_client(), max_iterations=5, num_variants=1,
            beam_width=1,
        )

        assert result["best_prompt"] == "already good"
        assert result["improvement"] == 0.0
        assert result["iterations_run"] <= 2

    @patch("src.prompt_optimizer.main.evaluate")
    @patch("src.prompt_optimizer.main.generate")
    def test_run_handles_variant_evaluation_error(
        self, mock_generate, mock_evaluate,
    ):
        mock_evaluate.side_effect = [
            {"avg_score": 0.5, "num_correct": 3, "num_records": 5, "per_record": []},
            Exception("Variant eval crashed"),
        ]
        mock_generate.return_value = ["bad_variant"]
        result = run(
            "agent_1", "model_x", "base", _sample_records(),
            _make_mock_client(), max_iterations=1, num_variants=1,
            beam_width=1,
        )
        assert result["best_prompt"] == "base"
        assert result["best_score"] == 0.5

    @patch("src.prompt_optimizer.main.evaluate")
    @patch("src.prompt_optimizer.main.generate")
    def test_run_beam_search_keeps_top_k(
        self, mock_generate, mock_evaluate,
    ):
        """Beam search con width=2 mantiene i 2 migliori candidati."""
        mock_evaluate.side_effect = [
            # baseline
            {"avg_score": 0.4, "num_correct": 2,
             "num_records": 5, "per_record": []},
            # iter 1: 1 beam entry × 2 variants
            {"avg_score": 0.7, "num_correct": 4,
             "num_records": 5, "per_record": []},
            {"avg_score": 0.6, "num_correct": 3,
             "num_records": 5, "per_record": []},
            # iter 2: 2 beam entries × 2 variants = 4 evals
            {"avg_score": 0.75, "num_correct": 4,
             "num_records": 5, "per_record": []},
            {"avg_score": 0.65, "num_correct": 3,
             "num_records": 5, "per_record": []},
            {"avg_score": 0.72, "num_correct": 4,
             "num_records": 5, "per_record": []},
            {"avg_score": 0.68, "num_correct": 3,
             "num_records": 5, "per_record": []},
        ]
        mock_generate.return_value = ["beam_v1", "beam_v2"]

        result = run(
            "agent_1", "model_x", "base", _sample_records(),
            _make_mock_client(), max_iterations=2, num_variants=2,
            beam_width=2,
        )

        assert result["beam_width"] == 2
        assert result["best_score"] == 0.75
        assert result["baseline_score"] == 0.4
        assert result["improvement"] > 0
        # generate called once per beam entry per iteration
        # iter 1: 1 beam entry → 1 call; iter 2: 2 beam entries → 2 calls
        assert mock_generate.call_count == 3

    def test_default_beam_width_constant(self):
        """Verifica che la costante DEFAULT_BEAM_WIDTH sia definita."""
        assert DEFAULT_BEAM_WIDTH >= 1

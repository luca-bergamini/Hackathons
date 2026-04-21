"""Test per src.frontend.app — smoke test di import e helper logic."""

from src.reporting.main import (
    _build_cost_map,
    _compute_operative_metrics,
    aggregate_scores,
)


class TestFrontendImports:
    """Verifica che il modulo app.py sia importabile senza errori."""

    def test_can_import_helpers(self):
        """I moduli backend usati dal frontend sono importabili."""
        from src.data_processing import main as data_processing
        from src.evaluation import main as evaluation
        from src.insight_agent import main as insight_agent
        from src.model_selection import main as model_selection
        from src.prompt_optimizer import main as prompt_optimizer
        from src.providers.bedrock import BedrockClient
        from src.reporting import main as reporting
        from src.runner import main as runner_mod

        assert callable(data_processing.load_dataset_from_s3)
        assert callable(data_processing.split_by_agent)
        assert callable(data_processing.validate_record)
        assert callable(data_processing.identify_task)
        assert callable(model_selection.load_model_configs)
        assert callable(model_selection.load_judge_config)
        assert callable(runner_mod.run)
        assert callable(evaluation.run)
        assert callable(reporting.run)
        assert callable(reporting.aggregate_scores)
        assert callable(insight_agent.run)
        assert callable(prompt_optimizer.run)
        assert callable(BedrockClient)


class TestPipelineDataContracts:
    """Verifica che le strutture dati tra moduli siano compatibili."""

    def test_aggregate_scores_has_expected_keys(self):
        eval_results = {
            "per_record": [
                {
                    "test_id": "t1", "model_id": "m-a",
                    "task": "classification", "score": 0.9,
                    "correct": True, "output_text": "pos",
                    "legacy_output": "pos", "ground_truth": "pos",
                },
            ],
            "per_task_model": {},
            "per_model": {},
        }
        agg = aggregate_scores(eval_results)

        # Keys the frontend expects
        assert "per_record" in agg
        assert "per_task_model" in agg
        assert "per_model" in agg
        assert "best_models" in agg

    def test_per_task_model_has_task_and_model_id(self):
        """Il frontend itera per_task_model.values() e legge 'task', 'model_id'."""
        eval_results = {
            "per_record": [
                {
                    "test_id": "t1", "model_id": "m-a",
                    "task": "classification", "score": 0.9,
                    "correct": True, "output_text": "x",
                    "legacy_output": "x", "ground_truth": "x",
                },
            ],
            "per_task_model": {},
            "per_model": {},
        }
        agg = aggregate_scores(eval_results)
        for info in agg["per_task_model"].values():
            assert "task" in info
            assert "model_id" in info
            assert "avg_score" in info
            assert "accuracy" in info
            assert "num_records" in info

    def test_per_model_has_overall_score(self):
        """Il frontend legge per_model[mid]['overall_score']."""
        eval_results = {
            "per_record": [
                {
                    "test_id": "t1", "model_id": "m-a",
                    "task": "classification", "score": 0.8,
                    "correct": True, "output_text": "x",
                    "legacy_output": "x", "ground_truth": "x",
                },
            ],
            "per_task_model": {},
            "per_model": {},
        }
        agg = aggregate_scores(eval_results)
        for info in agg["per_model"].values():
            assert "overall_score" in info
            assert "num_records" in info

    def test_operative_metrics_column_names(self):
        """Il frontend accede a 'model', 'avg_latency_ms', 'total_cost'."""
        runner_results = {
            "m-a": [
                {
                    "test_id": "t1", "task": "classification",
                    "success": True, "latency_ms": 150,
                    "retries": 0, "input_tokens": 100,
                    "output_tokens": 10,
                },
            ],
        }
        configs = [
            {"model_id": "m-a", "cost_input_1m": 0.08, "cost_output_1m": 0.35},
        ]
        cost_map = _build_cost_map(configs)
        op = _compute_operative_metrics(runner_results, cost_map)

        for info in op.values():
            assert "model" in info, "frontend expects 'model' column"
            assert "avg_latency_ms" in info
            assert "total_cost" in info
            assert "input_tokens" in info
            assert "output_tokens" in info

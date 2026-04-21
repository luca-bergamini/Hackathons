"""Test per src.model_selection."""

import yaml
import pytest

from src.model_selection.main import load_model_configs, load_judge_config, run


def _write_yaml(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


class TestLoadModelConfigs:
    def test_loads_valid_config(self, tmp_path):
        cfg = {"models": [
            {"model_id": "m1", "display_name": "M1", "provider": "amazon"},
            {"model_id": "m2", "display_name": "M2", "provider": "amazon"},
            {"model_id": "m3", "display_name": "M3", "provider": "amazon"},
        ]}
        path = tmp_path / "models.yaml"
        _write_yaml(cfg, str(path))
        models = load_model_configs(path)
        assert len(models) == 3
        assert models[0]["model_id"] == "m1"

    def test_empty_models_raises(self, tmp_path):
        path = tmp_path / "models.yaml"
        _write_yaml({"models": []}, str(path))
        with pytest.raises(ValueError, match="Nessun modello"):
            load_model_configs(path)

    def test_missing_model_id_raises(self, tmp_path):
        cfg = {"models": [{"display_name": "NoId"}]}
        path = tmp_path / "models.yaml"
        _write_yaml(cfg, str(path))
        with pytest.raises(ValueError, match="model_id mancante"):
            load_model_configs(path)


class TestLoadJudgeConfig:
    def test_loads_judge(self, tmp_path):
        cfg = {"judge": {"model_id": "j1", "display_name": "Judge"}, "models": []}
        path = tmp_path / "models.yaml"
        _write_yaml(cfg, str(path))
        judge = load_judge_config(path)
        assert judge["model_id"] == "j1"

    def test_missing_judge_raises(self, tmp_path):
        cfg = {"models": [{"model_id": "m1"}]}
        path = tmp_path / "models.yaml"
        _write_yaml(cfg, str(path))
        with pytest.raises(ValueError, match="judge"):
            load_judge_config(path)


class TestDefaultConfig:
    def test_real_config_is_valid(self):
        """Verifica che configs/models.yaml è parsabile e ha almeno 3 modelli."""
        models = load_model_configs("configs/models.yaml")
        assert len(models) >= 3
        for m in models:
            assert "model_id" in m


class TestRun:
    def test_run_successful(self, tmp_path):
        cfg = {
            "judge": {"model_id": "judge-1", "display_name": "Judge"},
            "models": [
                {"model_id": "m1", "display_name": "M1", "provider": "a"},
                {"model_id": "m2", "display_name": "M2", "provider": "a"},
                {"model_id": "m3", "display_name": "M3", "provider": "a"},
            ],
        }
        path = tmp_path / "models.yaml"
        _write_yaml(cfg, str(path))
        models = run(config_path=path)
        assert len(models) == 3

    def test_run_conflict_of_interest_raises(self, tmp_path):
        cfg = {
            "judge": {"model_id": "m1", "display_name": "Judge"},
            "models": [
                {"model_id": "m1", "display_name": "M1", "provider": "a"},
                {"model_id": "m2", "display_name": "M2", "provider": "a"},
                {"model_id": "m3", "display_name": "M3", "provider": "a"},
            ],
        }
        path = tmp_path / "models.yaml"
        _write_yaml(cfg, str(path))
        with pytest.raises(ValueError, match="conflitto di interesse"):
            run(config_path=path)

    def test_run_with_real_config(self):
        models = run(config_path="configs/models.yaml")
        assert len(models) >= 3

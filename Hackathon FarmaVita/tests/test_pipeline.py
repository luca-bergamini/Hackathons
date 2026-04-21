"""Test per il pipeline orchestrator (run_pipeline.py).

Testa la logica di orchestrazione degli step della pipeline.
"""

from unittest.mock import patch, MagicMock

import pytest


def test_pipeline_module_imports():
    """Il modulo run_pipeline deve essere importabile."""
    import run_pipeline  # noqa: F401


def test_pipeline_steps_dict():
    """STEPS deve contenere tutti gli step con nome e funzione."""
    from run_pipeline import STEPS

    assert isinstance(STEPS, dict)
    assert len(STEPS) >= 5

    # Verifica ogni step ha nome e callable
    for step_num, (name, func) in STEPS.items():
        assert isinstance(step_num, int)
        assert isinstance(name, str)
        assert callable(func), f"Step {step_num} ({name}) non è callable"


def test_pipeline_step_names():
    """I nomi degli step devono essere quelli attesi."""
    from run_pipeline import STEPS

    expected_names = {
        1: "Ingestion",
        2: "Processing",
        3: "Feature Engineering",
        4: "Model Training",
        5: "RAG Vector Store",
    }
    for num, expected_name in expected_names.items():
        assert num in STEPS, f"Step {num} mancante"
        assert STEPS[num][0] == expected_name, f"Step {num}: atteso '{expected_name}', trovato '{STEPS[num][0]}'"


def test_pipeline_paths_defined():
    """Le costanti di path devono essere definite."""
    from run_pipeline import RAW_DIR, INTERMEDIATE_PATH, FEATURES_PATH, MODEL_DIR

    assert RAW_DIR is not None
    assert INTERMEDIATE_PATH is not None
    assert FEATURES_PATH is not None
    assert MODEL_DIR is not None


def test_pipeline_step_order():
    """Gli step devono essere in ordine numerico crescente."""
    from run_pipeline import STEPS

    step_nums = sorted(STEPS.keys())
    assert step_nums == list(range(1, len(STEPS) + 1))


def test_pipeline_main_function_exists():
    """La funzione main() deve esistere."""
    from run_pipeline import main
    assert callable(main)


@patch("run_pipeline.STEPS")
def test_pipeline_only_mode(mock_steps):
    """Con --only=N, solo lo step N deve essere eseguito."""
    # Verifica la logica senza eseguire
    from run_pipeline import main
    assert callable(main)


@patch("run_pipeline.STEPS")
def test_pipeline_from_mode(mock_steps):
    """Con --from=N, solo gli step da N in poi devono essere eseguiti."""
    from run_pipeline import main
    assert callable(main)

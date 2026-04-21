"""Test aggiuntivi per models — copertura run_optuna_search e run_training_pipeline."""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_run_optuna_search(sample_features_df):
    """run_optuna_search deve restituire uno study con best_value."""
    from src.features.main import FEATURE_COLS_V2
    from src.models.main import prepare_train_val_split, run_optuna_search

    df = sample_features_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    X_tr, y_tr, X_val, y_val_log, y_val_orig = prepare_train_val_split(df, val_weeks=1)

    # Se validation è vuoto, skip
    if len(X_val) == 0:
        pytest.skip("Validation set vuoto con questi dati")

    study = run_optuna_search(X_tr, y_tr, X_val, y_val_log, y_val_orig, n_trials=2)

    assert study is not None
    assert study.best_value > 0
    assert isinstance(study.best_params, dict)
    assert "max_depth" in study.best_params


def test_run_training_pipeline(sample_features_df, tmp_path):
    """run_training_pipeline deve eseguire tutta la pipeline (con pochi trial)."""
    from src.models.main import run_training_pipeline

    df = sample_features_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Salva il df come parquet per l'input
    input_path = str(tmp_path / "features.parquet")
    df.to_parquet(input_path, index=False)

    model_dir = str(tmp_path / "models_out")

    model, study, val_rmspe = run_training_pipeline(
        input_path=input_path,
        n_trials=2,
        model_dir=model_dir,
    )

    assert model is not None
    assert hasattr(model, "predict")
    assert val_rmspe > 0
    assert os.path.exists(os.path.join(model_dir, "xgboost_model.pkl"))
    assert os.path.exists(os.path.join(model_dir, "feature_importance.txt"))


def test_run_training_pipeline_pkl_content(sample_features_df, tmp_path):
    """Il pkl salvato deve contenere tutti i campi richiesti."""
    from src.models.main import run_training_pipeline

    df = sample_features_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    input_path = str(tmp_path / "features.parquet")
    df.to_parquet(input_path, index=False)
    model_dir = str(tmp_path / "models_out")

    run_training_pipeline(input_path=input_path, n_trials=2, model_dir=model_dir)

    with open(os.path.join(model_dir, "xgboost_model.pkl"), "rb") as f:
        saved = pickle.load(f)

    assert "model" in saved
    assert "feature_cols" in saved
    assert "best_params" in saved
    assert "val_rmspe" in saved
    assert "log_target" in saved
    assert saved["log_target"] is True

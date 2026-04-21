"""Test per il modulo models (src/models/main.py).

Testa le funzioni di training, valutazione e salvataggio del modello XGBoost.
"""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_models_module_imports():
    """Verify the models module can be imported."""
    from src.models import main  # noqa: F401


def test_rmspe_function():
    """RMSPE deve calcolare correttamente l'errore percentuale."""
    from src.models.main import rmspe

    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 380, 520])

    result = rmspe(y_true, y_pred)
    assert result > 0
    assert result < 1  # errori piccoli → RMSPE < 1


def test_rmspe_perfect_prediction():
    """RMSPE con previsione perfetta deve dare 0."""
    from src.models.main import rmspe

    y = np.array([100.0, 200.0, 300.0])
    assert rmspe(y, y) == pytest.approx(0.0)


def test_rmspe_ignores_zeros():
    """RMSPE deve ignorare le righe con y_true = 0."""
    from src.models.main import rmspe

    y_true = np.array([0, 100, 200])
    y_pred = np.array([50, 110, 190])

    # Solo 2 righe usate (100, 200)
    result = rmspe(y_true, y_pred)
    assert result > 0


def test_rmspe_scorer():
    """Lo scorer deve restituire il negativo di RMSPE (per sklearn)."""
    from src.models.main import rmspe_scorer

    y = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    score = rmspe_scorer(y, y_pred)
    assert score <= 0  # sklearn convention: negate for maximize


def test_prepare_train_val_split(sample_features_df):
    """Lo split temporale deve separare train e validation."""
    from src.features.main import FEATURE_COLS_V2
    from src.models.main import prepare_train_val_split

    df = sample_features_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    X_tr, y_tr, X_val, y_val_log, y_val_orig = prepare_train_val_split(df, val_weeks=1)

    assert len(X_tr) > 0, "Training set vuoto"
    assert len(y_tr) == len(X_tr)
    # Con solo 20 giorni di dati, il validation potrebbe essere piccolo
    assert len(y_val_log) == len(y_val_orig)


def test_prepare_train_val_split_log_target(sample_features_df):
    """Il target di training deve essere in log scale."""
    from src.models.main import prepare_train_val_split

    df = sample_features_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    X_tr, y_tr, X_val, y_val_log, y_val_orig = prepare_train_val_split(df, val_weeks=1)

    # log1p(sales) deve essere < sales originali per sales > 1
    if len(y_val_orig) > 0:
        for orig, log_val in zip(y_val_orig, y_val_log):
            if orig > 1:
                assert log_val < orig


def test_prepare_train_val_split_only_open(sample_features_df):
    """Lo split deve includere solo negozi aperti."""
    from src.models.main import prepare_train_val_split

    df = sample_features_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    X_tr, y_tr, X_val, y_val_log, y_val_orig = prepare_train_val_split(df)

    # Verifica che is_open=0 non sia nel training
    if "is_open" in X_tr.columns:
        assert (X_tr["is_open"] == 0).sum() == 0


def test_train_final_model(sample_features_df):
    """Il modello finale deve allenare senza errori."""
    from src.models.main import train_final_model

    df = sample_features_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    best_params = {
        "max_depth": 3,
        "n_estimators": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "gamma": 0.0,
    }

    model = train_final_model(df, best_params)
    assert model is not None
    assert hasattr(model, "predict")


def test_save_model(sample_features_df):
    """save_model deve creare il file .pkl e feature_importance.txt."""
    from src.models.main import train_final_model, save_model

    df = sample_features_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    params = {
        "max_depth": 3, "n_estimators": 10, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
        "reg_alpha": 0.01, "reg_lambda": 0.01, "gamma": 0.0,
    }
    model = train_final_model(df, params)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_model(model, params, val_rmspe=0.15, output_dir=tmpdir)

        pkl_path = os.path.join(tmpdir, "xgboost_model.pkl")
        txt_path = os.path.join(tmpdir, "feature_importance.txt")

        assert os.path.exists(pkl_path), "File pkl non creato"
        assert os.path.exists(txt_path), "File txt non creato"

        # Verifica contenuto pkl
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        assert "model" in saved
        assert "feature_cols" in saved
        assert "best_params" in saved
        assert "val_rmspe" in saved
        assert saved["log_target"] is True


def test_save_model_importance_content(sample_features_df):
    """Il file feature_importance.txt deve contenere tutte le feature."""
    from src.models.main import train_final_model, save_model

    df = sample_features_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    params = {
        "max_depth": 3, "n_estimators": 10, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
        "reg_alpha": 0.01, "reg_lambda": 0.01, "gamma": 0.0,
    }
    model = train_final_model(df, params)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_model(model, params, val_rmspe=0.15, output_dir=tmpdir)

        with open(os.path.join(tmpdir, "feature_importance.txt")) as f:
            content = f.read()

        assert "Feature Importance" in content
        assert "RMSPE Validation" in content
        assert "store_id" in content

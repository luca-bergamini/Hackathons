"""Test aggiuntivi per features — copertura add_promo_features e run_feature_engineering."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_add_promo_features_with_mock_engine(sample_raw_df):
    """add_promo_features deve aggiungere promo e promo2_active con engine mockato."""
    from src.features.main import (
        impute_missing_values, add_temporal_features,
        add_categorical_encoding, add_promo_features,
    )

    df = impute_missing_values(sample_raw_df)
    df = add_temporal_features(df)
    df = add_categorical_encoding(df)

    # Rimuovi colonne promo pre-esistenti (add_promo_features le crea dal DB)
    df = df.drop(columns=["promo", "has_promo2", "promo2_active"], errors="ignore")

    # Mock dell'engine e dei risultati SQL
    mock_engine = MagicMock()

    promo_daily = pd.DataFrame({
        "store_id": [1, 1, 2, 2],
        "date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-01", "2015-01-03"]),
        "promo": [1, 1, 1, 1],
    })

    promo_cont = pd.DataFrame({
        "store_id": [1, 2, 3],
        "since_week": [10.0, 20.0, None],
        "since_year": [2013.0, 2014.0, None],
        "active_months": ["Jan,Feb,Mar", "Jun,Jul", None],
    })

    with patch("src.features.main.pd.read_sql") as mock_read_sql:
        mock_read_sql.side_effect = [promo_daily, promo_cont]
        result = add_promo_features(df, engine=mock_engine)

    assert "promo" in result.columns
    assert "has_promo2" in result.columns
    assert "promo2_active" in result.columns
    assert result["promo"].isna().sum() == 0


def test_add_promo_features_creates_engine_if_none(sample_raw_df):
    """Se engine=None, add_promo_features deve creare la connessione."""
    from src.features.main import (
        impute_missing_values, add_temporal_features,
        add_categorical_encoding, add_promo_features,
    )

    df = impute_missing_values(sample_raw_df)
    df = add_temporal_features(df)
    df = add_categorical_encoding(df)

    # Rimuovi colonne promo pre-esistenti
    df = df.drop(columns=["promo", "has_promo2", "promo2_active"], errors="ignore")

    promo_daily = pd.DataFrame({
        "store_id": [1], "date": pd.to_datetime(["2015-01-01"]), "promo": [1],
    })
    promo_cont = pd.DataFrame({
        "store_id": [1], "since_week": [10.0], "since_year": [2013.0],
        "active_months": ["Jan,Feb"],
    })

    with patch("src.features.main.create_engine") as mock_create, \
         patch("src.features.main.pd.read_sql") as mock_read_sql:
        mock_create.return_value = MagicMock()
        mock_read_sql.side_effect = [promo_daily, promo_cont]
        result = add_promo_features(df, engine=None)

    mock_create.assert_called_once()
    assert "promo" in result.columns


@patch("src.features.main.pd.read_parquet")
@patch("src.features.main.add_promo_features")
def test_run_feature_engineering(mock_promo, mock_read_parquet, sample_raw_df, tmp_path):
    """run_feature_engineering deve eseguire tutta la pipeline FE."""
    from src.features.main import run_feature_engineering

    # Prepara un df con le colonne minime necessarie
    df = sample_raw_df.copy()
    mock_read_parquet.return_value = df

    # Mock promo — restituisce il df con le colonne promo aggiunte
    def add_promo_cols(df_in, engine=None):
        df_out = df_in.copy()
        df_out["promo"] = 0
        df_out["has_promo2"] = 0
        df_out["promo2_active"] = 0
        return df_out

    mock_promo.side_effect = add_promo_cols

    output_path = str(tmp_path / "output.parquet")
    result = run_feature_engineering("fake_input.parquet", output_path)

    assert result is not None
    assert len(result) > 0
    assert (tmp_path / "output.parquet").exists()


@patch("src.features.main.pd.read_parquet")
@patch("src.features.main.add_promo_features")
def test_run_feature_engineering_removes_zero_sales(mock_promo, mock_read_parquet, sample_raw_df, tmp_path):
    """run_feature_engineering deve rimuovere righe con sales=0 e is_open=1 nel train."""
    from src.features.main import run_feature_engineering

    df = sample_raw_df.copy()
    # Aggiungi righe con sales=0 e is_open=1 nel train
    df.loc[0:3, "sales"] = 0
    df.loc[0:3, "is_open"] = 1
    df.loc[0:3, "is_train"] = True

    mock_read_parquet.return_value = df

    def add_promo_cols(df_in, engine=None):
        df_out = df_in.copy()
        for col in ["promo", "has_promo2", "promo2_active"]:
            if col not in df_out.columns:
                df_out[col] = 0
        return df_out

    mock_promo.side_effect = add_promo_cols

    output_path = str(tmp_path / "output.parquet")
    result = run_feature_engineering("fake.parquet", output_path)

    # Le righe con sales=0 e is_open=1 nel train devono essere state rimosse
    train_open_zero = result[(result["is_train"] == True) & (result["is_open"] == 1) & (result["sales"] == 0)]
    assert len(train_open_zero) == 0

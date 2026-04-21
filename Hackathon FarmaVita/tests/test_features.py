"""Test per il modulo di feature engineering (src/features/main.py).

Testa tutte le funzioni di feature engineering:
- Imputation, temporal, categorical, target encoding, interactions, promo.
"""

import numpy as np
import pandas as pd
import pytest


def test_features_module_imports():
    """Verify the features module can be imported."""
    from src.features import main  # noqa: F401


def test_feature_cols_v2_defined():
    """FEATURE_COLS_V2 deve essere una lista non vuota di stringhe."""
    from src.features.main import FEATURE_COLS_V2
    assert isinstance(FEATURE_COLS_V2, list)
    assert len(FEATURE_COLS_V2) > 30  # almeno 39 feature
    assert all(isinstance(c, str) for c in FEATURE_COLS_V2)


def test_feature_cols_v2_no_duplicates():
    """Non ci devono essere feature duplicate nella lista."""
    from src.features.main import FEATURE_COLS_V2
    assert len(FEATURE_COLS_V2) == len(set(FEATURE_COLS_V2))


# --- impute_missing_values ---

def test_impute_missing_values(sample_raw_df):
    """Dopo l'imputation, le feature principali non devono avere NaN."""
    from src.features.main import impute_missing_values

    df = sample_raw_df.copy()
    # Inietta NaN artificiali
    df.loc[0:5, "temperature_avg"] = np.nan
    df.loc[10:15, "unemployment_rate"] = np.nan
    df.loc[20:25, "trend_index"] = np.nan
    df.loc[30:35, "distance_meters"] = np.nan
    df.loc[40:45, "has_local_event"] = np.nan

    result = impute_missing_values(df)

    for col in ["temperature_avg", "unemployment_rate", "trend_index",
                "distance_meters", "has_local_event", "is_open"]:
        assert result[col].isna().sum() == 0, f"NaN rimasti in {col}"


def test_impute_fixes_negative_unemployment(sample_raw_df):
    """I valori negativi di unemployment_rate devono diventare positivi (abs)."""
    from src.features.main import impute_missing_values

    df = sample_raw_df.copy()
    df.loc[0:5, "unemployment_rate"] = -4.89

    result = impute_missing_values(df)
    assert (result["unemployment_rate"] < 0).sum() == 0


def test_impute_fills_weather_event(sample_raw_df):
    """weather_event NaN deve essere riempito con 'None'."""
    from src.features.main import impute_missing_values

    df = sample_raw_df.copy()
    result = impute_missing_values(df)
    assert result["weather_event"].isna().sum() == 0


def test_impute_removes_temp_columns(sample_raw_df):
    """Le colonne temporanee _month e _year devono essere rimosse."""
    from src.features.main import impute_missing_values

    result = impute_missing_values(sample_raw_df)
    assert "_month" not in result.columns
    assert "_year" not in result.columns


# --- add_temporal_features ---

def test_add_temporal_features(sample_raw_df):
    """Deve aggiungere tutte le feature temporali."""
    from src.features.main import add_temporal_features

    result = add_temporal_features(sample_raw_df)

    expected_cols = [
        "day_of_week", "month", "year", "day_of_month",
        "week_of_year", "is_weekend", "day_of_year",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        "woy_sin", "woy_cos", "competition_open_months",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Manca colonna {col}"


def test_temporal_day_of_week_range(sample_raw_df):
    """day_of_week deve essere 0-6."""
    from src.features.main import add_temporal_features
    result = add_temporal_features(sample_raw_df)
    assert result["day_of_week"].min() >= 0
    assert result["day_of_week"].max() <= 6


def test_temporal_is_weekend(sample_raw_df):
    """is_weekend=1 solo quando day_of_week >= 5."""
    from src.features.main import add_temporal_features
    result = add_temporal_features(sample_raw_df)

    for _, row in result.iterrows():
        if row["day_of_week"] >= 5:
            assert row["is_weekend"] == 1
        else:
            assert row["is_weekend"] == 0


def test_temporal_sin_cos_range(sample_raw_df):
    """Sin/cos features devono essere tra -1 e 1."""
    from src.features.main import add_temporal_features
    result = add_temporal_features(sample_raw_df)

    for col in ["dow_sin", "dow_cos", "month_sin", "month_cos", "woy_sin", "woy_cos"]:
        assert result[col].min() >= -1.0 - 1e-10
        assert result[col].max() <= 1.0 + 1e-10


def test_temporal_competition_open_months_non_negative(sample_raw_df):
    """competition_open_months non deve essere negativo."""
    from src.features.main import add_temporal_features
    result = add_temporal_features(sample_raw_df)
    assert (result["competition_open_months"] >= 0).all()


# --- add_categorical_encoding ---

def test_add_categorical_encoding(sample_raw_df):
    """Deve creare colonne _enc per le categoriche."""
    from src.features.main import impute_missing_values, add_categorical_encoding

    df = impute_missing_values(sample_raw_df)
    result = add_categorical_encoding(df)

    for col in ["type_code_enc", "level_code_enc", "state_enc", "weather_event_enc"]:
        assert col in result.columns, f"Manca {col}"
        assert result[col].dtype in [np.int32, np.int64, int]


def test_categorical_encoding_no_nan(sample_raw_df):
    """Le colonne _enc non devono avere NaN."""
    from src.features.main import impute_missing_values, add_categorical_encoding

    df = impute_missing_values(sample_raw_df)
    result = add_categorical_encoding(df)

    for col in ["type_code_enc", "level_code_enc", "state_enc", "weather_event_enc"]:
        assert result[col].isna().sum() == 0


# --- add_target_encoding ---

def test_add_target_encoding(sample_raw_df):
    """Deve aggiungere le medie storiche per store/dow/state."""
    from src.features.main import (
        impute_missing_values, add_temporal_features,
        add_categorical_encoding, add_target_encoding,
    )

    df = impute_missing_values(sample_raw_df)
    df = add_temporal_features(df)
    df = add_categorical_encoding(df)
    result = add_target_encoding(df)

    for col in ["store_avg_sales", "store_med_sales", "store_dow_avg", "state_month_avg"]:
        assert col in result.columns, f"Manca {col}"
        assert result[col].isna().sum() == 0, f"NaN in {col}"


def test_target_encoding_uses_train_only(sample_raw_df):
    """Le medie devono essere calcolate SOLO dal train set."""
    from src.features.main import (
        impute_missing_values, add_temporal_features,
        add_categorical_encoding, add_target_encoding,
    )

    df = impute_missing_values(sample_raw_df)
    df = add_temporal_features(df)
    df = add_categorical_encoding(df)

    # Imposta sales=0 per test set — non dovrebbe influire sulle medie
    df.loc[df["is_train"] == False, "sales"] = 0
    result = add_target_encoding(df)

    # store_avg_sales per store 1 deve essere basato solo su train
    train_avg = df[df["is_train"] == True].groupby("store_id")["sales"].mean()
    assert result.loc[result["store_id"] == 1, "store_avg_sales"].iloc[0] == pytest.approx(
        train_avg.get(1, 0), rel=0.01
    )


# --- add_interaction_features ---

def test_add_interaction_features(sample_features_df):
    """Deve creare distance_x_type e store_type_assort."""
    from src.features.main import add_interaction_features

    df = sample_features_df.copy()
    result = add_interaction_features(df)

    assert "distance_x_type" in result.columns
    assert "store_type_assort" in result.columns


def test_interaction_distance_x_type_formula(sample_features_df):
    """distance_x_type = distance_meters * type_code_enc."""
    from src.features.main import add_interaction_features

    result = add_interaction_features(sample_features_df)
    expected = result["distance_meters"] * result["type_code_enc"]
    pd.testing.assert_series_equal(result["distance_x_type"], expected, check_names=False)


# --- Overall ---

def test_all_feature_engineering_functions_exist():
    """Tutte le funzioni di FE devono essere importabili."""
    from src.features.main import (
        impute_missing_values,
        add_temporal_features,
        add_categorical_encoding,
        add_target_encoding,
        add_interaction_features,
        add_promo_features,
        run_feature_engineering,
    )
    assert all(callable(f) for f in [
        impute_missing_values, add_temporal_features,
        add_categorical_encoding, add_target_encoding,
        add_interaction_features, add_promo_features,
        run_feature_engineering,
    ])

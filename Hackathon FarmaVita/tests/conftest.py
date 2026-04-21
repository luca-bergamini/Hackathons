"""Shared fixtures for all tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df():
    """Minimal DataFrame that mimics df_final_features schema."""
    np.random.seed(42)
    n = 60  # 3 stores × 20 days
    stores = [1, 2, 3] * 20
    dates = pd.date_range("2015-01-01", periods=20).tolist() * 3
    df = pd.DataFrame({
        "store_id": sorted(stores),
        "date": dates[:n],
        "sales": np.random.randint(3000, 12000, n).astype(float),
        "customers": np.random.randint(300, 1200, n),
        "is_open": [1] * 57 + [0] * 3,
        "is_train": [True] * 50 + [False] * 10,
        "state": ["HE"] * 20 + ["NW"] * 20 + ["BY"] * 20,
        "type_code": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
        "level_code": ["a"] * 30 + ["b"] * 30,
        "weather_event": ["None"] * 55 + [None] * 5,
        "distance_meters": np.random.uniform(100, 5000, n),
        "open_since_month": [6.0] * n,
        "open_since_year": [2010.0] * n,
        "temperature_avg": np.random.uniform(-5, 35, n),
        "temperature_min": np.random.uniform(-10, 25, n),
        "temperature_max": np.random.uniform(0, 40, n),
        "precipitation_mm": np.random.uniform(0, 30, n),
        "gdp_index": np.random.uniform(90, 110, n),
        "unemployment_rate": np.random.uniform(3, 12, n),
        "consumer_confidence_index": np.random.uniform(-10, 20, n),
        "trend_index": np.random.uniform(20, 80, n),
        "has_local_event": np.random.choice([0, 1], n),
        "promo": np.random.choice([0, 1], n),
        "has_promo2": np.random.choice([0, 1], n),
        "promo2_active": np.random.choice([0, 1], n),
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def sample_features_df(sample_raw_df):
    """DataFrame with all feature columns the model expects."""
    from src.features.main import FEATURE_COLS_V2

    df = sample_raw_df.copy()

    # Add temporal features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear

    # Cyclic
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Encoding
    from sklearn.preprocessing import LabelEncoder
    for col in ["type_code", "level_code", "state", "weather_event"]:
        df[col] = df[col].fillna("None")
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # Target encoding placeholders
    df["store_avg_sales"] = df.groupby("store_id")["sales"].transform("mean")
    df["store_med_sales"] = df.groupby("store_id")["sales"].transform("median")
    df["store_dow_avg"] = df.groupby(["store_id", "day_of_week"])["sales"].transform("mean")
    df["state_month_avg"] = df.groupby(["state", "month"])["sales"].transform("mean")

    # Interactions
    df["distance_x_type"] = df["distance_meters"] * df["type_code_enc"]
    df["store_type_assort"] = df["type_code_enc"] * 10 + df["level_code_enc"]
    df["competition_open_months"] = ((df["year"] - df["open_since_year"]) * 12 + (df["month"] - df["open_since_month"])).clip(lower=0)

    # Ensure all expected columns exist
    for col in FEATURE_COLS_V2:
        if col not in df.columns:
            df[col] = 0

    return df

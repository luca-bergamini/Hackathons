"""Test per il modulo serving (src/serving/main.py).

Testa gli endpoint FastAPI, il SalesPredictor e la logica di predizione.
Usa TestClient di FastAPI per test HTTP e mock per il modello.
"""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_serving_module_imports():
    """Verify the serving module can be imported."""
    from src.serving import main  # noqa: F401


def test_fastapi_app_exists():
    """L'app FastAPI deve esistere."""
    from src.serving.main import app
    assert app is not None
    assert app.title == "FarmaVita Sales Predictor API"


def test_pydantic_schemas():
    """Gli schemi Pydantic devono validare correttamente."""
    from src.serving.main import (
        PredictRequest, PredictResponse,
        BatchItem, BatchRequest, BatchResponse,
        ExplainRequest, ExplainResponse,
        HealthResponse,
    )

    # PredictRequest
    req = PredictRequest(store_id=42, date="2015-09-15", is_open=1)
    assert req.store_id == 42
    assert req.date == "2015-09-15"
    assert req.is_open == 1

    # PredictResponse
    resp = PredictResponse(store_id=42, date="2015-09-15", predicted_sales=5432.0)
    assert resp.predicted_sales == 5432.0

    # BatchItem
    item = BatchItem(id=1, store_id=42, date="2015-09-15", is_open=1)
    assert item.id == 1

    # ExplainRequest
    explain = ExplainRequest(store_id=42, question="Test?")
    assert explain.question == "Test?"

    # HealthResponse
    health = HealthResponse(status="healthy")
    assert health.status == "healthy"


def test_pydantic_batch_request():
    """BatchRequest deve accettare una lista di BatchItem."""
    from src.serving.main import BatchItem, BatchRequest

    items = [
        BatchItem(id=1, store_id=42, date="2015-09-15", is_open=1),
        BatchItem(id=2, store_id=43, date="2015-09-16", is_open=0),
    ]
    req = BatchRequest(requests=items)
    assert len(req.requests) == 2


class TestSalesPredictor:
    """Test del SalesPredictor con modello fittizio."""

    @pytest.fixture
    def predictor_setup(self, sample_features_df, tmp_path):
        """Crea un modello fittizio e il parquet feature per il predictor."""
        from xgboost import XGBRegressor
        from src.features.main import FEATURE_COLS_V2

        df = sample_features_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Train a tiny model
        train = df[df["is_train"] == True]
        X = train[FEATURE_COLS_V2].fillna(0)
        y = np.log1p(train["sales"])

        model = XGBRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)

        # Save model pkl
        model_path = str(tmp_path / "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_cols": FEATURE_COLS_V2,
                "best_params": {},
                "val_rmspe": 0.15,
                "log_target": True,
            }, f)

        # Save features parquet
        features_path = str(tmp_path / "features.parquet")
        df.to_parquet(features_path, index=False)

        return model_path, features_path

    def test_predictor_initializes(self, predictor_setup):
        """SalesPredictor deve inizializzarsi senza errori."""
        from src.serving.main import SalesPredictor
        model_path, features_path = predictor_setup
        p = SalesPredictor(model_path, features_path)
        assert p.model is not None
        assert len(p.feature_cols) > 0

    def test_predictor_predict_single(self, predictor_setup):
        """predict_single deve restituire un float >= 0."""
        from src.serving.main import SalesPredictor
        model_path, features_path = predictor_setup
        p = SalesPredictor(model_path, features_path)

        result = p.predict_single(1, "2015-01-10", 1)
        assert isinstance(result, float)
        assert result >= 0

    def test_predictor_predict_single_closed(self, predictor_setup):
        """Negozio chiuso → vendite = 0."""
        from src.serving.main import SalesPredictor
        model_path, features_path = predictor_setup
        p = SalesPredictor(model_path, features_path)

        result = p.predict_single(1, "2015-01-10", 0)
        assert result == 0.0

    def test_predictor_predict_batch(self, predictor_setup):
        """predict_batch deve restituire una lista di float."""
        from src.serving.main import SalesPredictor
        model_path, features_path = predictor_setup
        p = SalesPredictor(model_path, features_path)

        items = [
            {"store_id": 1, "date": "2015-01-10", "is_open": 1},
            {"store_id": 2, "date": "2015-01-11", "is_open": 1},
            {"store_id": 1, "date": "2015-01-12", "is_open": 0},
        ]
        results = p.predict_batch(items)
        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)
        assert results[2] == 0.0  # chiuso

    def test_predictor_build_features(self, predictor_setup):
        """build_features deve restituire un dict con tutti i feature."""
        from src.serving.main import SalesPredictor
        model_path, features_path = predictor_setup
        p = SalesPredictor(model_path, features_path)

        features = p.build_features(1, "2015-01-10", 1)
        assert isinstance(features, dict)
        assert "store_id" in features
        assert "day_of_week" in features
        assert "month" in features
        assert features["store_id"] == 1
        assert features["is_open"] == 1

    def test_predictor_build_features_cyclics(self, predictor_setup):
        """Le feature cicliche devono essere tra -1 e 1."""
        from src.serving.main import SalesPredictor
        model_path, features_path = predictor_setup
        p = SalesPredictor(model_path, features_path)

        features = p.build_features(1, "2015-06-15", 1)
        for key in ["dow_sin", "dow_cos", "month_sin", "month_cos", "woy_sin", "woy_cos"]:
            assert -1.0 <= features[key] <= 1.0, f"{key} fuori range"

    def test_predictor_unknown_store(self, predictor_setup):
        """Un store sconosciuto deve comunque predire (con defaults)."""
        from src.serving.main import SalesPredictor
        model_path, features_path = predictor_setup
        p = SalesPredictor(model_path, features_path)

        # Store 9999 non esiste nel dataset
        result = p.predict_single(9999, "2015-01-10", 1)
        assert isinstance(result, float)
        assert result >= 0


class TestEndpoints:
    """Test degli endpoint HTTP con TestClient."""

    @pytest.fixture
    def client(self, sample_features_df, tmp_path):
        """Crea un TestClient con predictor mockato."""
        from fastapi.testclient import TestClient
        from src.serving.main import app, SalesPredictor
        import src.serving.main as serving_module

        from xgboost import XGBRegressor
        from src.features.main import FEATURE_COLS_V2

        df = sample_features_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        train = df[df["is_train"] == True]
        X = train[FEATURE_COLS_V2].fillna(0)
        y = np.log1p(train["sales"])
        model = XGBRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)

        model_path = str(tmp_path / "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model, "feature_cols": FEATURE_COLS_V2,
                "best_params": {}, "val_rmspe": 0.15, "log_target": True,
            }, f)

        features_path = str(tmp_path / "features.parquet")
        df.to_parquet(features_path, index=False)

        # Set predictor globale
        serving_module.predictor = SalesPredictor(model_path, features_path)

        return TestClient(app)

    def test_health_endpoint(self, client):
        """/health deve restituire 200 e status healthy."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_predict_endpoint(self, client):
        """/predict deve restituire la predizione."""
        resp = client.post("/predict", json={
            "store_id": 1, "date": "2015-01-10", "is_open": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_sales" in data
        assert data["store_id"] == 1
        assert data["predicted_sales"] >= 0

    def test_predict_closed_store(self, client):
        """Negozio chiuso → sales = 0."""
        resp = client.post("/predict", json={
            "store_id": 1, "date": "2015-01-10", "is_open": 0,
        })
        assert resp.status_code == 200
        assert resp.json()["predicted_sales"] == 0.0

    def test_predict_batch_endpoint(self, client):
        """/predict/batch deve restituire le predizioni per tutti gli item."""
        resp = client.post("/predict/batch", json={
            "requests": [
                {"id": 1, "store_id": 1, "date": "2015-01-10", "is_open": 1},
                {"id": 2, "store_id": 2, "date": "2015-01-11", "is_open": 1},
                {"id": 3, "store_id": 1, "date": "2015-01-12", "is_open": 0},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 3
        assert data["predictions"][2]["predicted_sales"] == 0.0

    def test_predict_batch_ids_match(self, client):
        """I prediction id devono corrispondere ai request id."""
        resp = client.post("/predict/batch", json={
            "requests": [
                {"id": 100, "store_id": 1, "date": "2015-01-10", "is_open": 1},
                {"id": 200, "store_id": 2, "date": "2015-01-11", "is_open": 1},
            ]
        })
        data = resp.json()
        ids = [p["id"] for p in data["predictions"]]
        assert ids == [100, 200]

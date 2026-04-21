"""Test aggiuntivi per serving — copertura startup_event, explain endpoint, edge cases."""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestStartupEvent:
    """Test dello startup event."""

    @pytest.fixture
    def setup_predictor(self, sample_features_df, tmp_path):
        """Setup modello e path per il predictor."""
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

        return model_path, features_path

    def test_explain_endpoint_with_mock_rag(self, setup_predictor):
        """L'endpoint /explain deve chiamare il RAG e restituire la risposta."""
        from fastapi.testclient import TestClient
        from src.serving.main import app, SalesPredictor
        import src.serving.main as serving_module

        model_path, features_path = setup_predictor
        serving_module.predictor = SalesPredictor(model_path, features_path)

        mock_result = {
            "answer": "Lo store 42 ha vendite medie di 5000 euro.",
            "sources": ["FarmaVita Store ID 42. Average daily sales: 5000."],
        }

        with patch("src.serving.main.query_sales_advisor", return_value=mock_result):
            client = TestClient(app)
            resp = client.post("/explain", json={
                "store_id": 42,
                "question": "Come vanno le vendite?",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["store_id"] == 42
        assert "5000" in data["answer"]
        assert len(data["sources"]) > 0

    def test_explain_endpoint_error_handling(self, setup_predictor):
        """/explain deve restituire 500 se il RAG fallisce."""
        from fastapi.testclient import TestClient
        from src.serving.main import app, SalesPredictor
        import src.serving.main as serving_module

        model_path, features_path = setup_predictor
        serving_module.predictor = SalesPredictor(model_path, features_path)

        with patch("src.serving.main.query_sales_advisor", side_effect=Exception("Bedrock error")):
            client = TestClient(app)
            resp = client.post("/explain", json={
                "store_id": 42,
                "question": "Test?",
            })

        assert resp.status_code == 500
        assert "Errore Sales Advisor" in resp.json()["detail"]

    def test_predict_unknown_store_returns_prediction(self, setup_predictor):
        """Uno store sconosciuto deve restituire una predizione con defaults."""
        from fastapi.testclient import TestClient
        from src.serving.main import app, SalesPredictor
        import src.serving.main as serving_module

        model_path, features_path = setup_predictor
        serving_module.predictor = SalesPredictor(model_path, features_path)

        client = TestClient(app)
        resp = client.post("/predict", json={
            "store_id": 9999, "date": "2015-06-15", "is_open": 1,
        })
        assert resp.status_code == 200
        assert resp.json()["predicted_sales"] >= 0

    def test_predict_batch_empty_list(self, setup_predictor):
        """Batch vuoto deve restituire lista vuota."""
        from fastapi.testclient import TestClient
        from src.serving.main import app, SalesPredictor
        import src.serving.main as serving_module

        model_path, features_path = setup_predictor
        serving_module.predictor = SalesPredictor(model_path, features_path)

        client = TestClient(app)
        resp = client.post("/predict/batch", json={"requests": []})
        assert resp.status_code == 200
        assert resp.json()["predictions"] == []

    def test_health_returns_unhealthy_without_model(self):
        """/health deve restituire unhealthy se il modello non è caricato."""
        from fastapi.testclient import TestClient
        from src.serving.main import app
        import src.serving.main as serving_module

        # Forza predictor a None
        old = serving_module.predictor
        serving_module.predictor = None

        with patch("src.serving.main.SalesPredictor", side_effect=Exception("No model")):
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert "unhealthy" in resp.json()["status"]

        serving_module.predictor = old

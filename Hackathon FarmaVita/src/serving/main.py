"""API serving module — FastAPI per predizione vendite FarmaVita.

Endpoint:
    POST /predict       — predizione singola (store_id, date, is_open)
    POST /predict/batch — predizione batch (53.520 righe test set)
    POST /explain       — RAG Sales Advisor (Bedrock + FAISS)
    GET  /health        — health check

Avvio locale:
    uvicorn src.serving.main:app --reload --port 8000
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.main import (
    query_sales_advisor,
    prepare_store_documents,
    build_vector_store,
    get_vector_store,
    VECTOR_STORE_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Pydantic schemas (contratti API)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    store_id: int
    date: str
    is_open: int


class PredictResponse(BaseModel):
    store_id: int
    date: str
    predicted_sales: float


class BatchItem(BaseModel):
    id: int
    store_id: int
    date: str
    is_open: int


class BatchRequest(BaseModel):
    requests: List[BatchItem]


class BatchPrediction(BaseModel):
    id: int
    predicted_sales: float


class BatchResponse(BaseModel):
    predictions: List[BatchPrediction]


class ExplainRequest(BaseModel):
    store_id: int
    question: str


class ExplainResponse(BaseModel):
    store_id: int
    question: str
    answer: str
    sources: List[str]


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
#  Model & Feature Loader
# ---------------------------------------------------------------------------

class SalesPredictor:
    """Carica il modello e i dati necessari per costruire le feature a runtime."""

    def __init__(self, model_path: str, features_path: str):
        # 1. Carica modello
        logger.info(f"Caricamento modello da {model_path}...")
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)

        self.model = saved['model']
        self.feature_cols = saved['feature_cols']
        self.log_target = saved.get('log_target', True)
        self.val_rmspe = saved.get('val_rmspe', None)
        logger.info(f"Modello caricato (RMSPE val: {self.val_rmspe})")

        # 2. Carica features dataset per costruire lookup tables
        logger.info(f"Caricamento dati feature da {features_path}...")
        df = pd.read_parquet(features_path)
        df['date'] = pd.to_datetime(df['date'], format='mixed')

        # 3. Lookup: profilo store (features costanti per store)
        store_cols = [
            'store_id', 'distance_meters', 'open_since_month', 'open_since_year',
            'type_code_enc', 'level_code_enc', 'state_enc',
            'distance_x_type', 'store_type_assort',
            'store_avg_sales', 'store_med_sales',
            'has_promo2',
        ]
        available_store_cols = [c for c in store_cols if c in df.columns]
        self.store_profiles = (
            df[df['is_train'] == True]
            .groupby('store_id')[available_store_cols]
            .first()
            .to_dict('index')
        )

        # 4. Lookup: store_dow_avg (media vendite per store × giorno settimana)
        train = df[df['is_train'] == True]
        if 'day_of_week' in df.columns:
            self.store_dow_avg = (
                train.groupby(['store_id', 'day_of_week'])['store_dow_avg']
                .first()
                .to_dict()
            )
        else:
            self.store_dow_avg = {}

        # 5. Lookup: state_month_avg
        if 'state_enc' in df.columns and 'month' in df.columns:
            # Usiamo state_enc come chiave perché è quello che il modello vede
            self.state_month_avg = (
                train.groupby(['state_enc', 'month'])['state_month_avg']
                .first()
                .to_dict()
            )
        else:
            self.state_month_avg = {}

        # 6. Lookup: weather/macro per (state_enc, month) — mediane storiche
        weather_macro_cols = [
            'temperature_avg', 'temperature_min', 'temperature_max',
            'precipitation_mm', 'weather_event_enc',
            'gdp_index', 'unemployment_rate', 'consumer_confidence_index',
            'trend_index', 'has_local_event',
        ]
        available_wm = [c for c in weather_macro_cols if c in df.columns]
        if available_wm and 'state_enc' in df.columns and 'month' in df.columns:
            self.weather_macro_lookup = (
                train.groupby(['state_enc', 'month'])[available_wm]
                .median()
                .to_dict('index')
            )
        else:
            self.weather_macro_lookup = {}

        # 7. Lookup: promo per (store_id, date)
        if 'promo' in df.columns:
            promo_entries = df[df['promo'] == 1][['store_id', 'date']].copy()
            self.promo_set = set(zip(promo_entries['store_id'], promo_entries['date']))
        else:
            self.promo_set = set()

        # 8. Lookup: promo2_active per (store_id, month)
        if 'promo2_active' in df.columns:
            p2_entries = df[df['promo2_active'] == 1][['store_id', 'date']].copy()
            p2_entries['month'] = p2_entries['date'].dt.month
            self.promo2_active_set = set(zip(p2_entries['store_id'], p2_entries['month']))
        else:
            self.promo2_active_set = set()

        # Defaults per feature mancanti
        self.global_medians = train[available_wm].median().to_dict() if available_wm else {}

        logger.info(f"Predictor pronto: {len(self.store_profiles)} store profiles caricati")

    def build_features(self, store_id: int, date_str: str, is_open: int) -> dict:
        """Costruisce il vettore feature per una singola predizione."""
        dt = pd.to_datetime(date_str)
        dow = dt.dayofweek
        month = dt.month
        year = dt.year

        # Store profile
        profile = self.store_profiles.get(store_id, {})

        # Temporal
        features = {
            'store_id': store_id,
            'is_open': is_open,
            'day_of_week': dow,
            'month': month,
            'year': year,
            'day_of_month': dt.day,
            'week_of_year': dt.isocalendar().week,
            'is_weekend': 1 if dow >= 5 else 0,
            'day_of_year': dt.dayofyear,
        }

        # Ciclici
        features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        woy = features['week_of_year']
        features['woy_sin'] = np.sin(2 * np.pi * woy / 52)
        features['woy_cos'] = np.cos(2 * np.pi * woy / 52)

        # Store profile
        features['distance_meters'] = profile.get('distance_meters', 0)
        features['open_since_month'] = profile.get('open_since_month', 1)
        features['open_since_year'] = profile.get('open_since_year', 2000)
        features['type_code_enc'] = profile.get('type_code_enc', 0)
        features['level_code_enc'] = profile.get('level_code_enc', 0)
        features['state_enc'] = profile.get('state_enc', 0)
        features['distance_x_type'] = profile.get('distance_x_type', 0)
        features['store_type_assort'] = profile.get('store_type_assort', 0)
        features['store_avg_sales'] = profile.get('store_avg_sales', 0)
        features['store_med_sales'] = profile.get('store_med_sales', 0)
        features['has_promo2'] = profile.get('has_promo2', 0)

        # Competition open months
        osi_month = features['open_since_month']
        osi_year = features['open_since_year']
        features['competition_open_months'] = max(0, (year - osi_year) * 12 + (month - osi_month))

        # Target encoding lookups
        features['store_dow_avg'] = self.store_dow_avg.get((store_id, dow), features['store_avg_sales'])
        state_enc = features['state_enc']
        features['state_month_avg'] = self.state_month_avg.get((state_enc, month), 0)

        # Weather/macro (lookup storico per state+month, fallback a mediana globale)
        wm = self.weather_macro_lookup.get((state_enc, month), self.global_medians)
        features['temperature_avg'] = wm.get('temperature_avg', 0)
        features['temperature_min'] = wm.get('temperature_min', 0)
        features['temperature_max'] = wm.get('temperature_max', 0)
        features['precipitation_mm'] = wm.get('precipitation_mm', 0)
        features['weather_event_enc'] = wm.get('weather_event_enc', 0)
        features['gdp_index'] = wm.get('gdp_index', 0)
        features['unemployment_rate'] = wm.get('unemployment_rate', 0)
        features['consumer_confidence_index'] = wm.get('consumer_confidence_index', 0)
        features['trend_index'] = wm.get('trend_index', 0)
        features['has_local_event'] = wm.get('has_local_event', 0)

        # Promo
        features['promo'] = 1 if (store_id, dt) in self.promo_set else 0
        features['promo2_active'] = 1 if (store_id, month) in self.promo2_active_set else 0

        return features

    def predict_single(self, store_id: int, date_str: str, is_open: int) -> float:
        """Predice le vendite per un singolo store/data."""
        if is_open == 0:
            return 0.0

        features = self.build_features(store_id, date_str, is_open)
        row = pd.DataFrame([features])[self.feature_cols].fillna(0)

        pred = self.model.predict(row)[0]
        if self.log_target:
            pred = np.expm1(pred)
        return max(0.0, float(pred))

    def predict_batch(self, items: List[dict]) -> List[float]:
        """Predice le vendite per un batch di richieste."""
        results = []
        # Separa aperti/chiusi
        open_items = [(i, item) for i, item in enumerate(items) if item['is_open'] == 1]
        closed_items = [(i, item) for i, item in enumerate(items) if item['is_open'] == 0]

        results = [0.0] * len(items)

        # Chiusi → 0
        for i, _ in closed_items:
            results[i] = 0.0

        # Aperti → batch predict
        if open_items:
            features_list = [
                self.build_features(item['store_id'], item['date'], item['is_open'])
                for _, item in open_items
            ]
            batch_df = pd.DataFrame(features_list)[self.feature_cols].fillna(0)
            preds = self.model.predict(batch_df)
            if self.log_target:
                preds = np.expm1(preds)
            preds = np.clip(preds, 0, None)

            for (i, _), pred in zip(open_items, preds):
                results[i] = float(pred)

        return results


# ---------------------------------------------------------------------------
#  FastAPI Application
# ---------------------------------------------------------------------------

# Paths configurabili via env
MODEL_PATH = os.environ.get("MODEL_PATH", "src/models/xgboost_model.pkl")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "data/featured/df_final_features.parquet")

app = FastAPI(
    title="FarmaVita Sales Predictor API",
    description="API per la predizione delle vendite giornaliere dei negozi FarmaVita",
    version="1.0.0",
)

# Caricamento lazy del predictor (al primo avvio)
predictor: Optional[SalesPredictor] = None


def get_predictor() -> SalesPredictor:
    """Carica il predictor al primo utilizzo (singleton)."""
    global predictor
    if predictor is None:
        predictor = SalesPredictor(
            model_path=MODEL_PATH,
            features_path=FEATURES_PATH,
        )
    return predictor


@app.on_event("startup")
async def startup_event():
    """Pre-carica modello, lookup tables e vector store RAG all'avvio."""
    logger.info("Avvio API — caricamento modello...")
    try:
        get_predictor()
        logger.info("Modello caricato con successo!")
    except Exception as e:
        logger.error(f"Errore caricamento modello: {e}")
        logger.warning("Il modello verrà caricato alla prima richiesta")

    # Inizializza il vector store RAG se non esiste già
    try:
        vs = get_vector_store()
        if vs is None:
            logger.info("Vector store non trovato — inizializzazione RAG...")
            df = pd.read_parquet(FEATURES_PATH)
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            docs = prepare_store_documents(df)
            if docs:
                build_vector_store(docs)
                logger.info("Vector store RAG creato con successo!")
            else:
                logger.warning("Nessun documento creato per il RAG")
        else:
            logger.info("Vector store RAG già presente — caricato.")
    except Exception as e:
        logger.error(f"Errore inizializzazione RAG: {e}")
        logger.warning("Il Sales Advisor potrebbe non funzionare")


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — verifica che il servizio sia attivo e il modello caricato."""
    try:
        p = get_predictor()
        return HealthResponse(status="healthy")
    except Exception:
        return HealthResponse(status="unhealthy — model not loaded")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predizione vendite per un singolo negozio/data.

    Request:  {"store_id": 42, "date": "2015-09-15", "is_open": 1}
    Response: {"store_id": 42, "date": "2015-09-15", "predicted_sales": 5432.0}
    """
    try:
        p = get_predictor()
        sales = p.predict_single(request.store_id, request.date, request.is_open)
        return PredictResponse(
            store_id=request.store_id,
            date=request.date,
            predicted_sales=round(sales, 2),
        )
    except Exception as e:
        logger.error(f"Errore predizione: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Predizione batch — usato dalla valutazione per generare le 53.520 predizioni.

    Request:  {"requests": [{"id": 1, "store_id": 42, "date": "2015-09-15", "is_open": 1}, ...]}
    Response: {"predictions": [{"id": 1, "predicted_sales": 5432.0}, ...]}
    """
    try:
        p = get_predictor()
        items = [
            {"store_id": r.store_id, "date": r.date, "is_open": r.is_open}
            for r in request.requests
        ]
        predictions = p.predict_batch(items)

        return BatchResponse(
            predictions=[
                BatchPrediction(id=r.id, predicted_sales=round(pred, 2))
                for r, pred in zip(request.requests, predictions)
            ]
        )
    except Exception as e:
        logger.error(f"Errore predizione batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """RAG Sales Advisor — spiega i pattern di vendita di un negozio.

    Request:  {"store_id": 42, "question": "Perche il negozio 42 ha vendite basse il lunedi?"}
    Response: {"store_id": 42, "question": "...", "answer": "...", "sources": [...]}
    """
    try:
        # Arricchisci la domanda con il contesto dello store_id
        enriched_question = f"Riguardo allo store {request.store_id}: {request.question}"
        result = query_sales_advisor(enriched_question)
        return ExplainResponse(
            store_id=request.store_id,
            question=request.question,
            answer=result.get("answer", "Nessuna risposta disponibile."),
            sources=result.get("sources", []),
        )
    except Exception as e:
        logger.error(f"Errore RAG explain: {e}")
        raise HTTPException(status_code=500, detail=f"Errore Sales Advisor: {str(e)}")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    uvicorn.run(
        "src.serving.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

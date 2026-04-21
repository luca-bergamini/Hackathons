"""
Test della propria API — genera submission.csv chiamando /predict/batch.

Questo script vi permette di testare la vostra API prima della valutazione finale.
Chiama il vostro endpoint /predict/batch con tutte le 53.520 prediction_requests
e salva le predizioni in submission.csv.

NOTA: la valutazione ufficiale viene fatta dagli organizzatori chiamando
direttamente la vostra API. Questo script serve solo per il vostro self-test.

Uso:
    # 1. Avviare la vostra API
    # 2. Lanciare lo script
    python scripts/generate_submission.py
    python scripts/generate_submission.py --api-url http://localhost:8000
    python scripts/generate_submission.py --api-url https://xxx.execute-api.eu-west-1.amazonaws.com/prod

Output:
    submission.csv nella root del progetto
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import psycopg2
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_SIZE = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_prediction_requests():
    """Carica le prediction_requests dal database."""
    conn = psycopg2.connect(
        host=os.environ.get(
            "DB_HOST",
            "hackathon-farmavita-db.cjcn7vyqigdy.eu-west-1.rds.amazonaws.com",
        ),
        port=int(os.environ.get("DB_PORT", "5432")),
        dbname=os.environ.get("DB_NAME", "farmavita"),
        user=os.environ.get("DB_USER", "hackathon_reader"),
        password=os.environ.get("DB_PASSWORD", "ReadOnly_FarmaVita2026"),
    )

    df = pd.read_sql("SELECT id, store_id, date, is_open FROM raw.prediction_requests ORDER BY id", conn)
    conn.close()
    df['date'] = df['date'].astype(str)
    logger.info(f"Caricate {len(df):,} prediction_requests dal database")
    return df


def call_predict_batch(api_url, batch):
    """Chiama POST /predict/batch con un batch di richieste.

    Request body:
        {"requests": [{"id": 1, "store_id": 42, "date": "2015-09-15", "is_open": 1}, ...]}

    Expected response:
        {"predictions": [{"id": 1, "predicted_sales": 5432.0}, ...]}
    """
    payload = {
        "requests": batch.to_dict(orient="records"),
    }

    response = requests.post(
        f"{api_url.rstrip('/')}/predict/batch",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    return data["predictions"]


def main():
    parser = argparse.ArgumentParser(description="Self-test: genera submission.csv chiamando la propria API")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("API_URL", "http://localhost:8000"),
        help="URL base dell'API (default: http://localhost:8000 o env API_URL)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Dimensione batch per /predict/batch (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    api_url = args.api_url
    logger.info(f"API URL: {api_url}")

    # 1. Verifica che l'API sia raggiungibile
    try:
        health = requests.get(f"{api_url.rstrip('/')}/health", timeout=10)
        health.raise_for_status()
        logger.info("API health check: OK")
    except requests.exceptions.ConnectionError:
        logger.error(
            f"Impossibile connettersi all'API su {api_url}\n"
            "Assicurarsi che l'API sia in esecuzione.\n"
            "Esempio: uvicorn src.serving.main:app --host 0.0.0.0 --port 8000"
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        logger.warning(f"Health check ha risposto con errore: {e} — provo comunque...")

    # 2. Carica prediction_requests dal DB
    pred_requests = load_prediction_requests()

    # 3. Chiama l'API in batch
    all_predictions = []
    n_batches = (len(pred_requests) + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    for i in range(0, len(pred_requests), args.batch_size):
        batch = pred_requests.iloc[i : i + args.batch_size]
        batch_num = (i // args.batch_size) + 1

        try:
            preds = call_predict_batch(api_url, batch)
            all_predictions.extend(preds)
            logger.info(f"  Batch {batch_num}/{n_batches}: {len(preds)} predizioni ricevute")
        except requests.exceptions.HTTPError as e:
            logger.error(f"  Batch {batch_num}/{n_batches}: ERRORE API — {e}")
            logger.error(f"  Response: {e.response.text[:500] if e.response else 'N/A'}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"  Batch {batch_num}/{n_batches}: ERRORE — {e}")
            sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"Tutte le predizioni ricevute in {elapsed:.1f}s")

    # 4. Costruisci il DataFrame delle predizioni
    predictions_df = pd.DataFrame(all_predictions)

    if "id" not in predictions_df.columns or "predicted_sales" not in predictions_df.columns:
        logger.error(
            "La risposta dell'API deve contenere 'id' e 'predicted_sales'.\n"
            f"Colonne ricevute: {list(predictions_df.columns)}"
        )
        sys.exit(1)

    # 5. Validazione
    submission = predictions_df[["id", "predicted_sales"]].copy()
    submission["predicted_sales"] = pd.to_numeric(submission["predicted_sales"], errors="coerce")

    nan_count = submission["predicted_sales"].isna().sum()
    if nan_count > 0:
        logger.warning(f"  {nan_count:,} predizioni NaN trovate — sostituite con 0")
        submission["predicted_sales"] = submission["predicted_sales"].fillna(0)

    # Clip negative a 0
    neg_count = (submission["predicted_sales"] < 0).sum()
    if neg_count > 0:
        logger.warning(f"  {neg_count:,} predizioni negative — clippate a 0")
        submission["predicted_sales"] = np.clip(submission["predicted_sales"], 0, None)

    # Verifica completezza
    expected_ids = set(pred_requests["id"].values)
    received_ids = set(submission["id"].values)
    missing = expected_ids - received_ids
    if missing:
        logger.error(f"ERRORE: {len(missing):,} predizioni mancanti (id mancanti: {sorted(missing)[:10]}...)")
        sys.exit(1)

    # Ordina per id
    submission = submission.sort_values("id").reset_index(drop=True)

    # 6. Salva
    output_path = os.path.join(BASE_DIR, "submission.csv")
    submission.to_csv(output_path, index=False)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"  Submission salvata: {output_path}")
    logger.info(f"  Righe:             {len(submission):,}")
    logger.info(f"  Sales media:       {submission['predicted_sales'].mean():.0f}")
    logger.info(f"  Sales min/max:     {submission['predicted_sales'].min():.0f} / {submission['predicted_sales'].max():.0f}")
    logger.info(f"{'=' * 50}")

    assert len(submission) == len(pred_requests), (
        f"ERRORE: {len(submission):,} predizioni ma ne servono {len(pred_requests):,}!"
    )
    logger.info("Validazione OK — la vostra API funziona correttamente!")


if __name__ == "__main__":
    main()

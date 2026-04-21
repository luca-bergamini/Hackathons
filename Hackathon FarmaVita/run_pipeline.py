"""Pipeline orchestrator — FarmaVita Store Sales Predictor.

Esegue l'intera pipeline in sequenza:
  1. Ingestion:  DB PostgreSQL → data/raw/*.parquet
  2. Processing: data/raw/ → data/processed/intermediate_processed.parquet
  3. Features:   intermediate_processed → data/processed/df_final_features.parquet
  4. Models:     df_final_features → models/xgboost_model.pkl

Uso:
    python run_pipeline.py              # pipeline completa
    python run_pipeline.py --from 3     # da step 3 (features) in poi
    python run_pipeline.py --only 2     # solo step 2 (processing)
"""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("pipeline")

# Carica .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv non installato, uso variabili d'ambiente di sistema")

# ---- Path di default (condivisi tra tutti i moduli) ----
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
INTERMEDIATE_PATH = f"{PROCESSED_DIR}/intermediate_processed.parquet"
FEATURES_PATH = f"{PROCESSED_DIR}/df_final_features.parquet"
MODEL_DIR = "models"


def step_1_ingestion():
    """Step 1: Scarica tutte le tabelle dal DB PostgreSQL → data/raw/*.parquet"""
    logger.info("=" * 60)
    logger.info("  STEP 1 — INGESTION (DB → Parquet)")
    logger.info("=" * 60)
    from src.ingestion.main import main as ingestion_main
    ingestion_main()


def step_2_processing():
    """Step 2: JOIN tabelle normalizzate → dataset unificato"""
    logger.info("=" * 60)
    logger.info("  STEP 2 — PROCESSING (JOIN + Augmentation)")
    logger.info("=" * 60)
    from src.processing.main import main as processing_main
    processing_main()


def step_3_features():
    """Step 3: Feature engineering → dataset pronto per il modello"""
    logger.info("=" * 60)
    logger.info("  STEP 3 — FEATURE ENGINEERING")
    logger.info("=" * 60)
    from src.features.main import run_feature_engineering
    run_feature_engineering(
        input_path=INTERMEDIATE_PATH,
        output_path=FEATURES_PATH,
    )


def step_4_training(n_trials: int = 100):
    """Step 4: Training XGBoost con Optuna → modello salvato"""
    logger.info("=" * 60)
    logger.info(f"  STEP 4 — MODEL TRAINING (Optuna, {n_trials} trials)")
    logger.info("=" * 60)
    from src.models.main import run_training_pipeline
    run_training_pipeline(
        input_path=FEATURES_PATH,
        n_trials=n_trials,
        model_dir=MODEL_DIR,
    )


def step_5_rag():
    """Step 5: Costruisce il vector store FAISS per il RAG Sales Advisor."""
    logger.info("=" * 60)
    logger.info("  STEP 5 — RAG (Vector Store Construction)")
    logger.info("=" * 60)
    import pandas as pd
    from src.rag.main import prepare_store_documents, build_vector_store
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    docs = prepare_store_documents(df)
    if docs:
        build_vector_store(docs)
        logger.info(f"Vector store creato con {len(docs)} documenti")
    else:
        logger.warning("Nessun documento generato per il RAG")


STEPS = {
    1: ("Ingestion", step_1_ingestion),
    2: ("Processing", step_2_processing),
    3: ("Feature Engineering", step_3_features),
    4: ("Model Training", step_4_training),
    5: ("RAG Vector Store", step_5_rag),
}


def main():
    parser = argparse.ArgumentParser(description="FarmaVita Pipeline Orchestrator")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Step da cui partire (1-5, default: 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="Esegui solo questo step (1-5)")
    parser.add_argument("--trials", type=int, default=100,
                        help="Numero di trial Optuna per step 4 (default: 100)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  FARMAVITA PIPELINE ORCHESTRATOR")
    logger.info("=" * 60)

    # Determina quali step eseguire
    if args.only:
        steps_to_run = [args.only]
    else:
        steps_to_run = [s for s in STEPS if s >= args.from_step]

    logger.info(f"Step da eseguire: {[f'{s}. {STEPS[s][0]}' for s in steps_to_run]}")

    total_start = time.time()

    for step_num in steps_to_run:
        name, func = STEPS[step_num]
        step_start = time.time()
        try:
            if step_num == 4:
                func(n_trials=args.trials)
            else:
                func()
            elapsed = time.time() - step_start
            logger.info(f"✓ Step {step_num} ({name}) completato in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"✗ Step {step_num} ({name}) fallito: {e}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"  PIPELINE COMPLETATA in {total_elapsed:.1f}s")
    logger.info("=" * 60)
    logger.info(f"  Output:")
    logger.info(f"    Raw data:     {RAW_DIR}/")
    logger.info(f"    Processed:    {INTERMEDIATE_PATH}")
    logger.info(f"    Features:     {FEATURES_PATH}")
    logger.info(f"    Modello:      {MODEL_DIR}/xgboost_model.pkl")
    logger.info(f"    Importances:  {MODEL_DIR}/feature_importance.txt")
    logger.info(f"    FAISS index:  docs/faiss_index/")


if __name__ == "__main__":
    main()

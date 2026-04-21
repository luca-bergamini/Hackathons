"""FarmaVita Store Sales ingestion module.

Il database PostgreSQL e' una sorgente dati esterna di SOLA LETTURA con tabelle NORMALIZZATE.
Dovete implementare la connessione al database e il caricamento dei dati.

Credenziali di connessione: vedi file .env.example
Documentazione schema completa: vedi docs/DATABASE_SCHEMA.md
"""

import logging
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# Configura il logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credenziali database (sorgente esterna READ-ONLY)
# ---------------------------------------------------------------------------
DB_HOST = os.environ.get("DB_HOST", "hackathon-farmavita-db.cjcn7vyqigdy.eu-west-1.rds.amazonaws.com")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "farmavita")
DB_USER = os.environ.get("DB_USER", "hackathon_reader")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "ReadOnly_FarmaVita2026")

def main():
    # Costruzione stringa di connessione
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    logger.info("Connessione al database PostgreSQL...")
    engine = create_engine(db_url)
    
    # Crea la directory di destinazione se non esiste
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    schemas_and_tables = {
        "raw": [
            "daily_sales",
            "stores",
            "store_types",
            "assortment_levels",
            "competitions",
            "promo_daily",
            "promo_continuous",
            "state_holidays",
            "school_holidays",
            "prediction_requests"
        ],
        "augmentation": [
            "store_states",
            "weather",
            "google_trends",
            "macroeconomic",
            "local_events"
        ]
    }
    
    for schema, tables in schemas_and_tables.items():
        logger.info(f"--- Estrazione schema: {schema} ---")
        for table in tables:
            out_file = out_dir / f"{schema}_{table}.parquet"
            if out_file.exists():
                logger.info(f"Tabella {schema}.{table} gia' estratta in {out_file}, skipping...")
                continue
            
            logger.info(f"Lettura tabella {schema}.{table} dal DB...")
            try:
                # read_sql_table was throwing NotSupportedError sometimes if table not found, fallback to read_sql if needed
                df = pd.read_sql_table(table, engine, schema=schema)
                logger.info(f"Estratte {len(df)} righe. Salvataggio in {out_file}...")
                df.to_parquet(out_file, engine='pyarrow', index=False)
            except Exception as e:
                logger.error(f"Errore nell'estrazione della tabella {schema}.{table}: {e}")

    logger.info("Ingestion completata!")

if __name__ == "__main__":
    main()

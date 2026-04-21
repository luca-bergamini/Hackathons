import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_data(raw_dir: Path, out_dir: Path):
    logger.info("Avvio data processing e join tabelle normalizzate...")
    
    # 1. Caricamento main dataset
    train = pd.read_parquet(raw_dir / "raw_daily_sales.parquet")
    test = pd.read_parquet(raw_dir / "raw_prediction_requests.parquet")
    
    train["is_train"] = 1
    test["is_train"] = 0
    
    df = pd.concat([train, test], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["is_open"] = df["is_open"].fillna(1)
    
    # 2. Stores e anagrafica
    logger.info("Join anagrafica stores...")
    stores = pd.read_parquet(raw_dir / "raw_stores.parquet")
    store_types = pd.read_parquet(raw_dir / "raw_store_types.parquet")
    assortments = pd.read_parquet(raw_dir / "raw_assortment_levels.parquet")
    states = pd.read_parquet(raw_dir / "augmentation_store_states.parquet")
    
    stores = stores.merge(store_types, left_on="store_type_id", right_on="id", how="left")
    stores = stores.merge(assortments, left_on="assortment_id", right_on="id", how="left", suffixes=('_type', '_assortment'))
    stores = stores.drop(columns=["id_type", "id_assortment", "store_type_id", "assortment_id"], errors='ignore')
    
    # Join con stati
    stores = stores.merge(states, left_on="store_id", right_on="store", how="left")
    if "store" in stores.columns:
         stores = stores.drop(columns=["store"])
    
    # Competitors
    logger.info("Join competitors e imputazione missing values...")
    competitors = pd.read_parquet(raw_dir / "raw_competitions.parquet")
    max_dist = competitors["distance_meters"].max()
    competitors["distance_meters"] = competitors["distance_meters"].fillna(max_dist * 2)
    competitors["open_since_month"] = competitors["open_since_month"].fillna(1)
    competitors["open_since_year"] = competitors["open_since_year"].fillna(1900)
    
    stores = stores.merge(competitors, on="store_id", how="left")
    
    # 3. Join main con stores
    df = df.merge(stores, on="store_id", how="left")
    
    # 4. Join Augmentation
    logger.info("Costruzione dataset augmentation isolato a partire da weather (per stato)...")
    
    try:
        # Base: Weather (giornaliero per stato)
        aug_df = pd.read_parquet(raw_dir / "augmentation_weather.parquet")
        aug_df["date"] = pd.to_datetime(aug_df["date"], format='mixed', errors='coerce')
        aug_df = aug_df.dropna(subset=["date"]).drop_duplicates(["state", "date"])
        
        # Macroeconomic
        macro = pd.read_parquet(raw_dir / "augmentation_macroeconomic.parquet")
        macro["year_month"] = macro["year_month"].astype(str)
        macro = macro.drop_duplicates(["state", "year_month"])
        aug_df["year_month"] = aug_df["date"].dt.strftime("%Y-%m")
        aug_df = aug_df.merge(macro, on=["state", "year_month"], how="left")
        aug_df = aug_df.drop(columns=["year_month", "year_month_doc"], errors="ignore")
        
        # Google Trends
        trends = pd.read_parquet(raw_dir / "augmentation_google_trends.parquet")
        trends["week_start"] = pd.to_datetime(trends["week_start"], format='mixed', errors='coerce')
        trends = trends.dropna(subset=["week_start"]).groupby(["state", "week_start"], as_index=False).mean(numeric_only=True)
        aug_df["trend_week_start"] = aug_df["date"] - pd.to_timedelta(aug_df["date"].dt.dayofweek, unit='d')
        aug_df = aug_df.merge(trends, left_on=["state", "trend_week_start"], right_on=["state", "week_start"], how="left")
        aug_df = aug_df.drop(columns=["trend_week_start", "week_start"], errors="ignore")
        
        # Local events
        events = pd.read_parquet(raw_dir / "augmentation_local_events.parquet")
        events["date_start"] = pd.to_datetime(events["date_start"], format='mixed', errors='coerce')
        events = events[events["event_type"].notna()].dropna(subset=["date_start"])[["state", "date_start", "event_type"]].drop_duplicates(["state", "date_start"])
        aug_df = aug_df.merge(events, left_on=["state", "date"], right_on=["state", "date_start"], how="left")
        aug_df["has_local_event"] = aug_df["event_type"].notna().astype(int)
        aug_df = aug_df.drop(columns=["date_start", "event_type"], errors="ignore")
        
        # Store States (mappiamo gli stati ai negozi)
        logger.info("Espansione augmentation dataset per singola farmacia (Store)...")
        store_states = pd.read_parquet(raw_dir / "augmentation_store_states.parquet")
        aug_df = aug_df.merge(store_states, on="state", how="inner")
        
        # Merge finale sul df principale
        logger.info("Join completo augmentation su dataset principale...")
        logger.info(f"aug_df columns: {aug_df.columns.tolist()}")
        logger.info(f"aug_df shape: {aug_df.shape}")
        
        df = df.merge(aug_df, left_on=["store_id", "date"], right_on=["store", "date"], how="left", suffixes=("", "_aug"))
        df = df.drop(columns=["store", "state_aug"], errors="ignore")
        
    except Exception as e:
        logger.exception(f"Errore durante augmentation join: {e}")
        raise

    # Salvataggio intermedio per feature engineering
    out_file = out_dir / "intermediate_processed.parquet"
    logger.info(f"Dimensione dataset unificato: {df.shape}")
    logger.info(f"Salvataggio dataset di base in {out_file}...")
    df.to_parquet(out_file, engine='pyarrow', index=False)
    logger.info("Processing completato!")

def main():
    raw_dir = Path("data/raw")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    process_data(raw_dir, out_dir)

if __name__ == "__main__":
    main()

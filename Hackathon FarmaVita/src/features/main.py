"""Feature engineering module - creazione di feature derivate per la predizione vendite FarmaVita.

Questo modulo contiene tutte le funzioni di feature engineering:
- Imputation dei valori mancanti (strategie per feature)
- Feature temporali (day_of_week, month, sin/cos encoding, ecc.)
- Target encoding (medie storiche per store, state, ecc.)
- Feature di interazione e promo
- Encoding categoriche

Documentazione completa: docs/FEATURE_ENGINEERING.txt
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

# Lista feature usate dal modello
FEATURE_COLS_V2 = [
    # Originali
    'store_id', 'is_open', 'day_of_week', 'month', 'year', 'day_of_month',
    'week_of_year', 'is_weekend', 'distance_meters', 'competition_open_months',
    'temperature_avg', 'temperature_min', 'temperature_max', 'precipitation_mm',
    'gdp_index', 'unemployment_rate', 'consumer_confidence_index',
    'trend_index', 'has_local_event',
    'type_code_enc', 'level_code_enc', 'state_enc', 'weather_event_enc',
    # Cicliche
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'woy_sin', 'woy_cos',
    # Target encoding
    'store_avg_sales', 'store_med_sales', 'store_dow_avg', 'state_month_avg',
    # Interazioni
    'distance_x_type', 'store_type_assort', 'day_of_year',
    # Promo
    'promo', 'has_promo2', 'promo2_active',
]


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa i valori mancanti con strategie specifiche per ogni feature.

    Strategie:
    - Temperature/precipitazioni: mediana per (state, month)
    - GDP/consumer confidence: forward fill temporale
    - Unemployment rate: mediana per (state, year) + fix valori negativi
    - Trend index: mediana per (state, month)
    - Competition: mediana globale
    - has_local_event: 0 (nessun evento)
    - weather_event: 'None'
    """
    logger.info("Imputation valori mancanti...")
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df['_month'] = df['date'].dt.month
    df['_year'] = df['date'].dt.year

    # 1. Temperature: mediana per (state, month)
    for col in ['temperature_avg', 'temperature_min', 'temperature_max', 'precipitation_mm']:
        df[col] = df.groupby(['state', '_month'])[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df.groupby('_month')[col].transform(
            lambda x: x.fillna(x.median())
        )

    # 2. Macro-economici: forward fill temporale
    df = df.sort_values('date')
    for col in ['gdp_index', 'consumer_confidence_index']:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    # 3. Unemployment rate: mediana per (state, anno) + fix negativi
    df['unemployment_rate'] = df['unemployment_rate'].abs()
    df['unemployment_rate'] = df.groupby(['state', '_year'])['unemployment_rate'].transform(
        lambda x: x.fillna(x.median())
    )
    df['unemployment_rate'] = df['unemployment_rate'].fillna(df['unemployment_rate'].median())

    # 4. Trend index: mediana per (state, month)
    df['trend_index'] = df.groupby(['state', '_month'])['trend_index'].transform(
        lambda x: x.fillna(x.median())
    )
    df['trend_index'] = df['trend_index'].fillna(df['trend_index'].median())

    # 5. Competition: mediana globale
    df['distance_meters'] = df['distance_meters'].fillna(df['distance_meters'].median())
    df['open_since_month'] = df['open_since_month'].fillna(df['open_since_month'].median())
    df['open_since_year'] = df['open_since_year'].fillna(df['open_since_year'].median())

    # 6. has_local_event: 0, weather_event: 'None', is_open: 1
    df['has_local_event'] = df['has_local_event'].fillna(0)
    df['weather_event'] = df['weather_event'].fillna('None')
    df['is_open'] = df['is_open'].fillna(1)

    # Pulizia colonne temporanee
    df.drop(columns=['_month', '_year'], inplace=True, errors='ignore')

    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if len(remaining) > 0:
        logger.warning(f"NaN rimasti dopo imputation: {remaining.to_dict()}")
    else:
        logger.info("Nessun NaN rimasto nelle feature!")

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge feature temporali e cicliche dalla colonna date."""
    logger.info("Aggiunta feature temporali...")
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'], format='mixed')

    # Feature temporali base
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear

    # Anzianità competitor
    df['competition_open_months'] = (
        (df['year'] - df['open_since_year']) * 12 +
        (df['month'] - df['open_since_month'])
    ).clip(lower=0)

    # Encoding ciclico (sin/cos)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['woy_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['woy_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    return df


def add_categorical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encode delle variabili categoriche con LabelEncoder."""
    logger.info("Encoding categoriche...")
    df = df.copy()

    for col in ['type_code', 'level_code', 'state', 'weather_event']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    return df


def add_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge medie storiche dal train set (target encoding).

    NOTA: le medie sono calcolate SOLO dal train set per evitare data leakage.
    """
    logger.info("Aggiunta target encoding...")
    df = df.copy()

    train_only = df[df['is_train'] == True]

    # Media e mediana vendite per store
    store_avg = train_only.groupby('store_id')['sales'].mean().rename('store_avg_sales')
    store_med = train_only.groupby('store_id')['sales'].median().rename('store_med_sales')
    df = df.merge(store_avg, on='store_id', how='left')
    df = df.merge(store_med, on='store_id', how='left')

    # Media vendite per store × day_of_week
    store_dow = train_only.groupby(['store_id', 'day_of_week'])['sales'].mean().rename('store_dow_avg')
    df = df.merge(store_dow, on=['store_id', 'day_of_week'], how='left')

    # Media vendite per state × month
    state_month = train_only.groupby(['state', 'month'])['sales'].mean().rename('state_month_avg')
    df = df.merge(state_month, on=['state', 'month'], how='left')

    # Fill NaN nelle nuove feature
    for col in ['store_avg_sales', 'store_med_sales', 'store_dow_avg', 'state_month_avg']:
        df[col] = df[col].fillna(df[col].median())

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge feature di interazione tra feature top."""
    logger.info("Aggiunta feature di interazione...")
    df = df.copy()

    df['distance_x_type'] = df['distance_meters'] * df['type_code_enc']
    df['store_type_assort'] = df['type_code_enc'] * 10 + df['level_code_enc']

    return df


def add_promo_features(df: pd.DataFrame, engine=None) -> pd.DataFrame:
    """Aggiunge feature promo dal database (promo_daily e promo_continuous).

    Args:
        df: DataFrame con le feature
        engine: SQLAlchemy engine per connettersi al DB. Se None, crea la connessione.
    """
    logger.info("Aggiunta feature promo dal DB...")
    df = df.copy()

    if engine is None:
        DB_HOST = os.environ.get("DB_HOST", "hackathon-farmavita-db.cjcn7vyqigdy.eu-west-1.rds.amazonaws.com")
        DB_PORT = os.environ.get("DB_PORT", "5432")
        DB_NAME = os.environ.get("DB_NAME", "farmavita")
        DB_USER = os.environ.get("DB_USER", "hackathon_reader")
        DB_PASSWORD = os.environ.get("DB_PASSWORD", "ReadOnly_FarmaVita2026")
        engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    # 1. Promo giornaliera
    promo_daily = pd.read_sql("SELECT store_id, date, 1 as promo FROM raw.promo_daily", engine)
    promo_daily['date'] = pd.to_datetime(promo_daily['date'], format='mixed')
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.merge(promo_daily, on=['store_id', 'date'], how='left')
    df['promo'] = df['promo'].fillna(0).astype(int)

    # 2. Promo2 continua
    promo_cont = pd.read_sql("SELECT * FROM raw.promo_continuous", engine)

    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

    df = df.merge(promo_cont[['store_id', 'since_week', 'since_year', 'active_months']],
                  on='store_id', how='left')
    df['has_promo2'] = df['since_year'].notna().astype(int)

    # Calcolo vettoriale se promo2 è attiva
    df['promo2_start'] = pd.to_datetime(
        df['since_year'].fillna(2000).astype(int).astype(str) + '-01-01'
    ) + pd.to_timedelta(df['since_week'].fillna(0) * 7, unit='D')

    df['promo2_active'] = 0
    for month_name, month_num in month_map.items():
        mask = (df['active_months'].str.contains(month_name, na=False) &
                (df['date'].dt.month == month_num) &
                (df['date'] >= df['promo2_start']))
        df.loc[mask, 'promo2_active'] = 1

    # Pulizia colonne temporanee
    df.drop(columns=['since_week', 'since_year', 'active_months', 'promo2_start'],
            inplace=True, errors='ignore')

    logger.info(f"Promo attiva: {df['promo'].sum():,} giorni, Promo2 attiva: {df['promo2_active'].sum():,} giorni")
    return df


def run_feature_engineering(input_path: str, output_path: str, engine=None) -> pd.DataFrame:
    """Pipeline completa di feature engineering.

    Args:
        input_path: percorso del file parquet di input (intermediate_processed.parquet)
        output_path: percorso dove salvare il parquet con le feature
        engine: SQLAlchemy engine (opzionale, per feature promo)

    Returns:
        DataFrame con tutte le feature
    """
    logger.info(f"Caricamento dati da {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"Shape iniziale: {df.shape}")

    # Pipeline feature engineering
    df = impute_missing_values(df)
    df = add_temporal_features(df)
    df = add_categorical_encoding(df)
    df = add_target_encoding(df)
    df = add_interaction_features(df)
    df = add_promo_features(df, engine=engine)

    # Rimuovi sales=0 con negozio aperto dal train
    before = len(df)
    df = df[~((df['is_open'] == 1) & (df['sales'] == 0) & (df['is_train'] == True))]
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Rimossi {removed} record con sales=0 e negozio aperto (solo train)")

    # Salvataggio
    df.to_parquet(output_path, index=False)
    logger.info(f"Dataset salvato in {output_path} — Shape: {df.shape}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    run_feature_engineering(
        input_path="data/processed/intermediate_processed.parquet",
        output_path="data/processed/df_final_features.parquet",
    )

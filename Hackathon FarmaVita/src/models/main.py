"""Model training module - training e valutazione XGBoost per predizione vendite FarmaVita.

TASK: Predire le vendite giornaliere (Sales) per ogni riga di raw.prediction_requests.
Metrica: RMSPE (Root Mean Square Percentage Error).
Target: log1p(sales) — le predizioni vengono riconvertite con expm1().
Ottimizzazione iperparametri: Optuna (Bayesian Optimization).
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor

from src.features.main import FEATURE_COLS_V2

logger = logging.getLogger(__name__)

# Sopprime i log verbosi di Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Percentage Error (metrica ufficiale di valutazione).

    RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2))
    Nota: le righe con y_true = 0 vengono escluse dal calcolo.
    """
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


def rmspe_scorer(y_true, y_pred):
    """Scorer per sklearn (negativo perché sklearn massimizza)."""
    return -rmspe(y_true, y_pred)


def prepare_train_val_split(df: pd.DataFrame, val_weeks: int = 6):
    """Crea train/validation split temporale.

    Le ultime `val_weeks` settimane del training diventano validation set.
    Solo negozi aperti nel training (sales=0 per chiusi per definizione).

    Returns:
        X_tr, y_tr, X_val, y_val_log, y_val_orig, val DataFrame
    """
    train_data = df[df['is_train'] == True].copy()
    train_data = train_data[train_data['is_open'] == 1]

    cutoff_date = train_data['date'].max() - pd.Timedelta(weeks=val_weeks)
    tr = train_data[train_data['date'] <= cutoff_date]
    val = train_data[train_data['date'] > cutoff_date]

    X_tr = tr[FEATURE_COLS_V2].fillna(0)
    y_tr = np.log1p(tr['sales'])
    X_val = val[FEATURE_COLS_V2].fillna(0)
    y_val_log = np.log1p(val['sales'])
    y_val_orig = val['sales'].values

    logger.info(f"Train:      {len(tr):,} righe  (fino a {cutoff_date.date()})")
    logger.info(f"Validation: {len(val):,} righe  (da {val['date'].min().date()} a {val['date'].max().date()})")

    return X_tr, y_tr, X_val, y_val_log, y_val_orig


def run_optuna_search(X_tr, y_tr, X_val, y_val_log, y_val_orig, n_trials: int = 100):
    """Cerca i migliori iperparametri con Optuna (Bayesian Optimization).

    Args:
        X_tr, y_tr: training set (target in log scale)
        X_val, y_val_log: validation set (target in log scale per eval_set)
        y_val_orig: validation target in scala originale (per RMSPE)
        n_trials: numero di trial Optuna

    Returns:
        study: oggetto Optuna con i risultati
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 6, 14),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
        }

        model = XGBRegressor(
            **params,
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val_log)], verbose=False)

        pred_orig = np.expm1(model.predict(X_val)).clip(min=0)
        return rmspe(y_val_orig, pred_orig)

    study = optuna.create_study(direction='minimize', study_name='xgboost_log_sales')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best RMSPE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study


def train_final_model(df: pd.DataFrame, best_params: dict):
    """Allena il modello finale su tutto il training set con i best params.

    Args:
        df: DataFrame completo con feature
        best_params: migliori iperparametri da Optuna

    Returns:
        model: modello allenato
        test_final: DataFrame test con predizioni
    """
    train_final = df[(df['is_train'] == True) & (df['is_open'] == 1)].copy()
    test_final = df[df['is_train'] == False].copy()

    X_train_full = train_final[FEATURE_COLS_V2].fillna(0)
    y_train_full = np.log1p(train_final['sales'])
    X_test_full = test_final[FEATURE_COLS_V2].fillna(0)

    final_model = XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    final_model.fit(
        X_train_full, y_train_full,
        eval_set=[(X_train_full, y_train_full)],
        verbose=False,
    )

    logger.info(f"Modello finale allenato su {len(train_final):,} righe")

    return final_model


def save_model(model, best_params: dict, val_rmspe: float, output_dir: str = "models"):
    """Salva modello (pkl) e feature importance (txt).

    Args:
        model: modello XGBoost allenato
        best_params: migliori iperparametri
        val_rmspe: RMSPE sul validation set
        output_dir: directory di output
    """
    os.makedirs(output_dir, exist_ok=True)

    # Modello → PKL
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': FEATURE_COLS_V2,
            'best_params': best_params,
            'val_rmspe': val_rmspe,
            'log_target': True,
        }, f)
    logger.info(f"Modello salvato in {model_path}")

    # Feature Importance → TXT
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS_V2)
    importances = importances.sort_values(ascending=False)
    importance_path = os.path.join(output_dir, "feature_importance.txt")
    with open(importance_path, 'w') as f:
        f.write('Feature Importance (XGBoost - log-trained)\n')
        f.write('=' * 50 + '\n\n')
        for feat, imp in importances.items():
            f.write(f'{feat:40s} {imp:.6f}\n')
        f.write(f'\nTotale feature: {len(importances)}\n')
        f.write(f'RMSPE Validation: {val_rmspe:.4f}\n')
        f.write(f'Best params: {best_params}\n')
    logger.info(f"Feature importance salvata in {importance_path}")



def run_training_pipeline(input_path: str, n_trials: int = 100, model_dir: str = "models"):
    """Pipeline completa di training: caricamento → split → Optuna → train finale → salvataggio.

    Args:
        input_path: percorso del parquet con le feature (output di feature engineering)
        n_trials: numero di trial Optuna
        model_dir: directory dove salvare modello e feature importance
    """
    logger.info(f"Caricamento dati da {input_path}...")
    df = pd.read_parquet(input_path)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    logger.info(f"Shape: {df.shape}")

    # 1. Train/Validation split
    logger.info("=== STEP 1: Validation split temporale ===")
    X_tr, y_tr, X_val, y_val_log, y_val_orig = prepare_train_val_split(df)

    # 2. Optuna search
    logger.info(f"=== STEP 2: Optuna search ({n_trials} trials) ===")
    study = run_optuna_search(X_tr, y_tr, X_val, y_val_log, y_val_orig, n_trials=n_trials)

    # 3. Valutazione validation
    logger.info("=== STEP 3: Valutazione validation ===")
    best_v2 = XGBRegressor(
        **study.best_params,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    best_v2.fit(X_tr, y_tr, eval_set=[(X_val, y_val_log)], verbose=False)
    val_pred = np.expm1(best_v2.predict(X_val)).clip(min=0)
    val_rmspe = rmspe(y_val_orig, val_pred)
    logger.info(f"RMSPE Validation: {val_rmspe:.4f}")
    logger.info(f"MAE:  {np.mean(np.abs(y_val_orig - val_pred)):,.0f}")

    # 4. Train finale su tutto il training
    logger.info("=== STEP 4: Training finale ===")
    final_model = train_final_model(df, study.best_params)

    # 5. Salvataggio
    logger.info("=== STEP 5: Salvataggio ===")
    save_model(final_model, study.best_params, val_rmspe, output_dir=model_dir)

    logger.info("Pipeline completata!")
    return final_model, study, val_rmspe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    run_training_pipeline(
        input_path="data/processed/df_final_features.parquet",
        n_trials=100,
        model_dir="models",
    )

"""Model Selection Agent — Step 2."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/models.yaml")


def load_model_configs(config_path: str | Path = DEFAULT_CONFIG_PATH) -> list[dict]:
    """Carica la lista di modelli candidate dal file YAML.

    Returns una lista di dict, ciascuno con almeno: model_id, display_name, provider.
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])
    if not models:
        raise ValueError(f"Nessun modello trovato in {config_path}")

    for m in models:
        if "model_id" not in m:
            raise ValueError(f"model_id mancante in config modello: {m}")

    logger.info("Caricati %d modelli candidate da %s", len(models), config_path)
    return models


def load_judge_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Carica la configurazione del modello Judge dal file YAML."""
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    judge = config.get("judge")
    if not judge or "model_id" not in judge:
        raise ValueError(f"Configurazione judge mancante o incompleta in {config_path}")

    logger.info("Judge model: %s", judge["model_id"])
    return judge


def run(config_path: str | Path = DEFAULT_CONFIG_PATH) -> list[dict]:
    """Esegue il model selection: carica e valida i modelli, poi li restituisce."""
    models = load_model_configs(config_path)
    judge = load_judge_config(config_path)

    # Verifica che il judge non sia tra i candidate (conflitto di interesse)
    candidate_ids = {m["model_id"] for m in models}
    if judge["model_id"] in candidate_ids:
        raise ValueError(
            f"Judge model {judge['model_id']} non può essere un candidate — conflitto di interesse"
        )

    logger.info(
        "Model selection completato: %d candidate, judge=%s",
        len(models),
        judge["display_name"],
    )
    return models


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model Selection — Step 2")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    run(config_path=args.config)

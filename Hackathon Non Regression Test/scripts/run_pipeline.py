"""Pipeline end-to-end: Data Processing → Runner → Evaluator → Reporting.

Usage:
    python scripts/run_pipeline.py [--bucket BUCKET] [--key KEY]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.data_processing.main import run as data_processing_run
from src.evaluation.main import run as evaluation_run
from src.insight_agent.main import run as insight_run
from src.model_selection.main import load_judge_config, load_model_configs
from src.providers.bedrock import BedrockClient
from src.reporting.main import run as reporting_run
from src.runner.main import run as runner_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NRT Pipeline end-to-end")
    parser.add_argument("--bucket", default=None, help="S3 bucket name")
    parser.add_argument("--key", default="dataset/dataset.jsonl", help="S3 key")
    parser.add_argument("--output-excel", default="deliverable_intermedio_2.xlsx")
    parser.add_argument("--output-json", default="report_results.json")
    parser.add_argument("--output-csv", default="deliverable_intermedio_1.csv")
    args = parser.parse_args()

    # --- Step 1: Data Processing ---
    logger.info("=" * 60)
    logger.info("STEP 1 — Data Processing Agent")
    logger.info("=" * 60)
    dp_result = data_processing_run(
        s3_bucket=args.bucket,
        dataset_key=args.key,
        output_csv=args.output_csv,
    )
    logger.info("Data Processing completato: %d agents trovati", len(dp_result["agents"]))

    # Raccogli tutti i valid records con task assegnato
    all_records = []
    for agent_id, info in dp_result["agents"].items():
        task = info["task"]
        for rec in info["valid_records"]:
            rec["task"] = task
            all_records.append(rec)
    logger.info("Totale record validi per il Runner: %d", len(all_records))

    # --- Step 2: Model Selection ---
    logger.info("=" * 60)
    logger.info("STEP 2 — Model Selection")
    logger.info("=" * 60)
    model_configs = load_model_configs()
    judge_config = load_judge_config()
    logger.info(
        "Modelli candidate: %d, Judge: %s",
        len(model_configs),
        judge_config["display_name"],
    )

    # --- Step 3: Runner ---
    logger.info("=" * 60)
    logger.info("STEP 3 — Runner (Inferenza Parallela)")
    logger.info("=" * 60)
    region = os.environ.get("AWS_REGION", "eu-west-1")
    bedrock = BedrockClient(region=region)
    runner_results = runner_run(all_records, model_configs, bedrock)
    total_inferences = sum(len(v) for v in runner_results.values())
    logger.info("Runner completato: %d inferenze totali", total_inferences)

    # --- Step 4: Evaluator ---
    logger.info("=" * 60)
    logger.info("STEP 4 — Evaluator Agent")
    logger.info("=" * 60)
    judge_model_id = judge_config["model_id"]
    eval_results = evaluation_run(runner_results, bedrock, judge_model_id)
    logger.info(
        "Evaluation completata: %d record valutati",
        len(eval_results.get("per_record", [])),
    )

    # --- Step 5: Insight Agent ---
    logger.info("=" * 60)
    logger.info("STEP 5 — Insight Agent")
    logger.info("=" * 60)
    insight_results = insight_run(eval_results, bedrock, judge_model_id)
    logger.info("Insight Agent completato")

    # --- Step 6: Reporting ---
    logger.info("=" * 60)
    logger.info("STEP 6 — Report Aggregation + S3 Upload")
    logger.info("=" * 60)
    reporting_run(
        eval_results=eval_results,
        runner_results=runner_results,
        model_configs=model_configs,
        output_path=args.output_excel,
        json_output_path=args.output_json,
        insight_results=insight_results,
    )
    logger.info("Pipeline completata!")
    logger.info("  CSV:   %s", args.output_csv)
    logger.info("  Excel: %s", args.output_excel)
    logger.info("  JSON:  %s", args.output_json)


if __name__ == "__main__":
    main()

"""Script di debug per scaricare il dataset da S3 e testare identify_task.

Eseguire dalla root del progetto:
    python scripts/debug_identify_task.py
"""

import json
import logging
import os
import sys

# Assicura che src sia nel path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from src.data_processing.main import (
    identify_task,
    load_dataset_from_s3,
    split_by_agent,
    validate_record,
    _extract_messages,
    _score_task,
    _classify_task_with_llm,
    TASK_CLASSES,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BUCKET = os.getenv("S3_BUCKET", "hackathon-genai-innovations-team-05-data")
DATASET_KEY = "dataset/dataset.jsonl"
LOCAL_CACHE = "dataset_cache.jsonl"


def download_or_load() -> list[dict]:
    """Scarica dal S3 oppure carica dalla cache locale se già scaricato."""
    if os.path.exists(LOCAL_CACHE):
        logger.info("Loading from local cache: %s", LOCAL_CACHE)
        records = []
        with open(LOCAL_CACHE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    logger.info("Downloading from S3: s3://%s/%s", BUCKET, DATASET_KEY)
    records = load_dataset_from_s3(BUCKET, DATASET_KEY)

    # Salva in cache locale per non riscaricare ogni volta
    with open(LOCAL_CACHE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), LOCAL_CACHE)
    return records


def print_separator():
    print("=" * 90)


def _get_heuristic_best(system_text, user_text, output_type):
    """Return (best_task, best_score) from heuristic scoring only."""
    best_task = "classification"
    best_score = 0.0
    for t in TASK_CLASSES:
        s = _score_task(t, system_text, user_text, output_type)
        if s > best_score:
            best_score = s
            best_task = t
    return best_task, best_score


def _get_first_user_sample(recs):
    """Extract first 200 chars of the first user message across recs."""
    for r in recs:
        for msg in r.get("input_messages", []):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))[:200]
    return ""


def _print_agent_detail(agent_id, recs):
    """Print full diagnostic block for a single agent."""
    total = len(recs)
    valid = [r for r in recs if validate_record(r)]
    malformed = total - len(valid)

    task = identify_task(recs)
    system_text, user_text = _extract_messages(recs)

    output_type = None
    for r in recs:
        ot = r.get("expected_output_type")
        if ot:
            output_type = str(ot)
            break

    heuristic_best, heuristic_best_score = _get_heuristic_best(system_text, user_text, output_type)
    llm_task = _classify_task_with_llm(system_text, user_text, output_type)

    print(f"\n{'─' * 90}")
    print(f"AGENT: {agent_id}")
    print(f"  Records: {total} | Valid: {len(valid)} | Malformed: {malformed}")
    print(f"  expected_output_type: {output_type}")
    print(f"  TASK FINALE (ibrido):    >>> {task} <<<")
    print(f"  Task da LLM:             >>> {llm_task or 'FAILED'} <<<")
    print(f"  Task da euristiche:      >>> {heuristic_best} (score={heuristic_best_score:.1f}) <<<")
    if llm_task and llm_task != heuristic_best:
        print(f"  ⚠️  MISMATCH: LLM={llm_task} vs Heuristic={heuristic_best}")
    print()

    print(f"  SYSTEM PROMPT (primi 200 char):")
    print(f"    {system_text[:200]}...")
    print()

    first_user = _get_first_user_sample(recs)
    print(f"  USER MSG SAMPLE (primi 200 char):")
    print(f"    {first_user}...")
    print()

    print(f"  SCORES PER TASK:")
    scores = {t: _score_task(t, system_text, user_text, output_type) for t in TASK_CLASSES}
    for t, s in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " <<<" if t == task else ""
        bar = "█" * int(s)
        print(f"    {t:25s} score={s:6.1f} {bar}{marker}")

    valid_with_gt = sum(1 for r in valid if r.get("ground_truth") is not None)
    pct_gt = round((valid_with_gt / len(valid)) * 100, 1) if valid else 0.0
    print(f"\n  Ground truth: {valid_with_gt}/{len(valid)} = {pct_gt}%")


def _print_summary_table(agent_groups):
    """Print the RIEPILOGO ASSIGNMENT table."""
    print_separator()
    print("\nRIEPILOGO ASSIGNMENT:")
    print(f"{'agent_id':20s} {'task':25s} {'records':>8s} {'GT%':>8s} {'malformed':>10s}")
    print("─" * 75)
    for agent_id in sorted(agent_groups.keys()):
        recs = agent_groups[agent_id]
        task = identify_task(recs)
        valid = [r for r in recs if validate_record(r)]
        malformed = len(recs) - len(valid)
        valid_with_gt = sum(1 for r in valid if r.get("ground_truth") is not None)
        pct_gt = round((valid_with_gt / len(valid)) * 100, 1) if valid else 0.0
        print(f"{agent_id:20s} {task:25s} {len(recs):8d} {pct_gt:7.1f}% {malformed:10d}")
    print_separator()


def main():
    records = download_or_load()
    print_separator()
    print(f"TOTALE RECORD: {len(records)}")
    print_separator()

    agent_groups = split_by_agent(records)
    print(f"AGENT_IDs TROVATI: {len(agent_groups)}")
    print_separator()

    for agent_id in sorted(agent_groups.keys()):
        _print_agent_detail(agent_id, agent_groups[agent_id])

    _print_summary_table(agent_groups)


if __name__ == "__main__":
    main()

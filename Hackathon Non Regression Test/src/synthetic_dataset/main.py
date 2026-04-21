"""Synthetic Dataset Generator — Bonus.

Genera record sintetici a partire da seed reali per ampliare la copertura del dataset NRT.
I record prodotti rispettano lo schema della Sezione 2.1 e includono ground truth coerente.
"""

import json
import logging
import re
import uuid

logger = logging.getLogger(__name__)

DEFAULT_RECORDS_PER_TASK = 10


# ---------------------------------------------------------------------------
# Prompt di generazione per task
# ---------------------------------------------------------------------------

_GENERATION_PROMPTS: dict[str, str] = {
    "classification": (
        "Generate {n} diverse text classification examples. "
        "Each must have a user text and a correct label from the same label set as the seeds. "
        "Include edge cases (ambiguous, short, multi-topic texts)."
    ),
    "context_qa": (
        "Generate {n} diverse question-answering examples with context passages. "
        "Each must have a context, a question, and a correct answer extracted from the context. "
        "Include tricky questions, negation, and multi-hop reasoning."
    ),
    "metadata_extraction": (
        "Generate {n} diverse metadata extraction examples. "
        "Each must have an unstructured text and the correct JSON with extracted fields. "
        "Vary the text style, field count, and include edge cases "
        "(missing fields, unusual formats)."
    ),
    "code_generation": (
        "Generate {n} diverse code generation examples. "
        "Each must have a specification/requirement and the correct Python code. "
        "Include varying difficulty: simple functions, algorithms, "
        "class design, edge case handling."
    ),
    "translation": (
        "Generate {n} diverse translation examples. "
        "Each must have text in one language and the correct translation. "
        "Include idiomatic expressions, technical terms, and varying text lengths."
    ),
    "summarization": (
        "Generate {n} diverse summarization examples. "
        "Each must have a long text and a correct concise summary. "
        "Vary topics, text length, and summary detail level."
    ),
    "rephrasing": (
        "Generate {n} diverse rephrasing examples. "
        "Each must have an original text and a correctly rephrased version. "
        "Vary styles: formal↔informal, technical↔simple, concise↔expanded."
    ),
    "sql_generation": (
        "Generate {n} diverse SQL generation examples. "
        "Each must have a natural language query and the correct SQL. "
        "Include JOINs, aggregations, subqueries, and edge cases."
    ),
    "pii_redaction": (
        "Generate {n} diverse PII redaction examples. "
        "Each must have text containing PII "
        "(names, emails, phones, SSN) and the correctly redacted version. "
        "Vary PII types and positions."
    ),
    "intent_detection": (
        "Generate {n} diverse intent detection examples. "
        "Each must have a user utterance and the correct intent label. "
        "Include ambiguous, multi-intent, and short queries."
    ),
}


def generate_records_for_task(
    task: str,
    n: int,
    llm_client,
    model_id: str,
    seed_records: list[dict] | None = None,
) -> list[dict]:
    """Genera n record sintetici per un task specifico.

    Args:
        task: tipo di task (classification, context_qa, ...).
        n: numero di record da generare.
        llm_client: istanza BedrockClient.
        model_id: modello da usare per la generazione.
        seed_records: record reali come esempi (opzionale).

    Returns:
        Lista di dict nel formato dataset standard (Sezione 2.1).
    """
    # Prepara seed examples per il prompt
    seed_text = _format_seeds(seed_records, max_seeds=3)
    default_instr = f"Generate {n} diverse examples for the task: {task}."
    task_instruction = _GENERATION_PROMPTS.get(task, default_instr)
    task_instruction = task_instruction.format(n=n)

    # Determina agent_id e system prompt dai seed
    agent_id = "synthetic_agent"
    system_prompt = f"You are an assistant performing {task}."
    expected_output_type = _infer_output_type(task)

    if seed_records:
        agent_id = seed_records[0].get("agent_id", agent_id)
        for msg in seed_records[0].get("input_messages", []):
            if msg.get("role") == "system":
                system_prompt = msg.get("content", system_prompt)
                break

    generation_prompt = f"""You are a synthetic data generator for AI testing.

## Task Type
{task}

## Seed Examples
{seed_text}

## Instructions
{task_instruction}

Generate exactly {n} examples. Each example must be a JSON object with these fields:
- "user_content": the user message text
- "ground_truth": the correct expected output
- "expected_output_type": "{expected_output_type}"

Respond ONLY with valid JSON: {{"examples": [...]}}"""

    try:
        response = llm_client.invoke(
            model_id=model_id,
            messages=[{"role": "user", "content": generation_prompt}],
            max_tokens=4096,
            temperature=0.8,
        )
        text = response["output_text"].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            raw_examples = parsed.get("examples", [])
            return _build_records(raw_examples, task, agent_id, system_prompt, expected_output_type)
    except Exception as e:
        logger.error("Errore generazione sintetica per task=%s: %s", task, e)

    return []


def _format_seeds(seed_records: list[dict] | None, max_seeds: int = 3) -> str:
    """Formatta alcuni record seed come esempi nel prompt."""
    if not seed_records:
        return "No seed examples provided."

    lines = []
    for i, record in enumerate(seed_records[:max_seeds]):
        user_content = ""
        for msg in record.get("input_messages", []):
            if msg.get("role") == "user":
                # Truncate long user content
                content = str(msg.get("content", ""))
                user_content = content[:500] + ("..." if len(content) > 500 else "")
                break

        gt = record.get("ground_truth")
        gt_str = str(gt)[:300] if gt is not None else "null"
        lines.append(f"Example {i + 1}:\n  Input: {user_content}\n  Expected output: {gt_str}")

    return "\n\n".join(lines)


def _infer_output_type(task: str) -> str:
    """Deduce il expected_output_type dal task."""
    mapping = {
        "classification": "label",
        "context_qa": "text",
        "metadata_extraction": "json",
        "code_generation": "code",
        "translation": "text",
        "summarization": "text",
        "rephrasing": "text",
        "sql_generation": "sql",
        "pii_redaction": "text",
        "intent_detection": "label",
    }
    return mapping.get(task, "text")


def _build_records(
    raw_examples: list[dict],
    task: str,
    agent_id: str,
    system_prompt: str,
    expected_output_type: str,
) -> list[dict]:
    """Converte esempi grezzi dal LLM in record conformi allo schema Sezione 2.1."""
    records = []
    for example in raw_examples:
        if not isinstance(example, dict):
            continue

        user_content = example.get("user_content", "")
        if not user_content:
            continue

        ground_truth = example.get("ground_truth")
        test_id = f"synth_{uuid.uuid4().hex[:12]}"

        record = {
            "test_id": test_id,
            "input_messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_content)},
            ],
            "ground_truth": ground_truth,
            "expected_output_type": example.get("expected_output_type", expected_output_type),
            "metadata": {"synthetic": True, "source_task": task},
            "agent_id": agent_id,
            "output": None,  # sintetico — nessun output Legacy
        }
        records.append(record)

    return records


def save_dataset(records: list[dict], output_path: str) -> None:
    """Salva record in formato JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Salvati %d record sintetici in %s", len(records), output_path)


def enrich_dataset(
    original_records: list[dict],
    synthetic_records: list[dict],
) -> list[dict]:
    """Unisce record originali e sintetici, evitando duplicati per test_id."""
    existing_ids = {r.get("test_id") for r in original_records}
    enriched = list(original_records)
    added = 0
    for r in synthetic_records:
        if r.get("test_id") not in existing_ids:
            enriched.append(r)
            existing_ids.add(r.get("test_id"))
            added += 1
    logger.info(
        "Dataset arricchito: %d originali + %d sintetici = %d totali",
        len(original_records), added, len(enriched),
    )
    return enriched


def _generate_task_records(
    task: str,
    records_per_task: int,
    llm_client,
    model_id: str,
    seed_records_by_task: dict,
    edge_seeds_by_task: dict,
) -> list[dict]:
    """Generate base + edge-case records for a single task."""
    seeds = seed_records_by_task.get(task)
    records = generate_records_for_task(
        task=task,
        n=records_per_task,
        llm_client=llm_client,
        model_id=model_id,
        seed_records=seeds,
    )

    edge_seeds = edge_seeds_by_task.get(task, [])
    if edge_seeds:
        n_edge = max(2, records_per_task // 2)
        logger.info(
            "  Generating %d edge-case records for task=%s based on %d failures",
            n_edge, task, len(edge_seeds),
        )
        edge_records = generate_records_for_task(
            task=task,
            n=n_edge,
            llm_client=llm_client,
            model_id=model_id,
            seed_records=edge_seeds[:5],
        )
        for r in edge_records:
            r.setdefault("metadata", {})["edge_case"] = True
        records.extend(edge_records)

    return records


def _apply_validation(
    records: list[dict],
    evaluator_fn,
    eval_threshold: float,
    task: str,
) -> list[dict]:
    """Validate records with evaluator_fn; discard those below eval_threshold."""
    validated = []
    discarded = 0
    for r in records:
        try:
            score = evaluator_fn(r)
            if score >= eval_threshold:
                r.setdefault("metadata", {})["validation_score"] = round(score, 3)
                validated.append(r)
            else:
                discarded += 1
        except Exception as e:
            logger.debug("Validation failed for record %s: %s", r.get("test_id"), e)
            validated.append(r)  # keep if validator errors
    logger.info(
        "  Validation: %d kept, %d discarded (threshold=%.2f) for task=%s",
        len(validated), discarded, eval_threshold, task,
    )
    return validated


def run(
    tasks: list[str],
    llm_client,
    model_id: str,
    records_per_task: int = DEFAULT_RECORDS_PER_TASK,
    output_path: str = "synthetic_dataset.jsonl",
    seed_records_by_task: dict[str, list[dict]] | None = None,
    evaluator_fn=None,
    eval_threshold: float = 0.3,
    low_score_records: list[dict] | None = None,
) -> list[dict]:
    """Genera dataset sintetico per i task richiesti.

    BN_SYNTHETIC_DATASET:
    - Auto-validation: if evaluator_fn is provided, each generated record is
      validated. Records scoring below eval_threshold are discarded.
    - Edge-case targeting: if low_score_records is provided, generates additional
      records similar to failing test cases to cover critical areas.

    Args:
        tasks: lista di task da generare.
        llm_client: istanza BedrockClient.
        model_id: modello da usare per la generazione.
        records_per_task: numero di record per task.
        output_path: path dove salvare il dataset JSONL.
        seed_records_by_task: dict task -> lista record seed (opzionale).
        evaluator_fn: optional callable(record) -> float score (for auto-validation).
        eval_threshold: minimum score for a synthetic record to be kept (default 0.3).
        low_score_records: records from prior pipeline runs with low scores,
                           used as seeds for targeted edge-case generation.

    Returns:
        Lista di tutti i record sintetici generati e validati.
    """
    if seed_records_by_task is None:
        seed_records_by_task = {}

    # Build edge-case seeds from low-score pipeline records
    edge_seeds_by_task: dict[str, list[dict]] = {}
    if low_score_records:
        for rec in low_score_records:
            task = rec.get("task", "")
            if task:
                edge_seeds_by_task.setdefault(task, []).append(rec)
        logger.info(
            "Edge-case targeting: found low-score records for %d tasks",
            len(edge_seeds_by_task),
        )

    all_records: list[dict] = []

    for task in tasks:
        logger.info("Generazione sintetica per task=%s (%d record)...", task, records_per_task)
        records = _generate_task_records(
            task, records_per_task, llm_client, model_id,
            seed_records_by_task, edge_seeds_by_task,
        )

        if evaluator_fn is not None:
            records = _apply_validation(records, evaluator_fn, eval_threshold, task)

        logger.info("  Generati %d record per task=%s", len(records), task)
        all_records.extend(records)

    if all_records and output_path:
        save_dataset(all_records, output_path)

    logger.info(
        "Generazione sintetica completata: %d record per %d task",
        len(all_records), len(tasks),
    )
    return all_records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator — Bonus")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_RECORDS_PER_TASK)
    parser.add_argument("--output", default="synthetic_dataset.jsonl")
    parser.add_argument("--seeds-json", default=None, help="JSON file con seed_records_by_task")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    seeds = None
    if args.seeds_json:
        with open(args.seeds_json, encoding="utf-8") as f:
            seeds = json.load(f)

    from src.providers.bedrock import BedrockClient

    client = BedrockClient()
    result = run(
        tasks=args.tasks,
        llm_client=client,
        model_id=args.model,
        records_per_task=args.n,
        output_path=args.output,
        seed_records_by_task=seeds,
    )
    print(f"Generated {len(result)} synthetic records.")

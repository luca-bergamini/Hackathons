"""Runner — Step 3."""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_WORKERS = 10
CALL_TIMEOUT_S = 30
MAX_RETRIES = 3
PAYLOAD_DIR = Path("payloads")
_UNKNOWN = "unknown"


def _sanitize_path_component(value: str) -> str:
    """Sanitize a value for safe use as a filename component."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', value)


def _save_payload(
    record: dict, model_id: str,
    request_body: dict, response_body: dict,
):
    """Salva request/response JSON per payload tracing."""
    PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    test_id = _sanitize_path_component(
        record.get("test_id", _UNKNOWN),
    )
    safe_model = _sanitize_path_component(model_id)
    filepath = PAYLOAD_DIR / f"{test_id}_{safe_model}.json"
    if not filepath.resolve().is_relative_to(PAYLOAD_DIR.resolve()):
        logger.warning("Path traversal detected: %s", test_id)
        return
    payload = {
        "test_id": test_id,
        "model_id": model_id,
        "request": request_body,
        "response": response_body,
        "timestamp": time.time(),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_single_inference(
    record: dict, model_id: str, providers_client, max_tokens: int = 1024
) -> dict:
    """Esegue una singola inferenza con retry (max 3), timeout (30s), tracking latenza+token.

    Returns dict con: test_id, model_id, output_text, input_tokens, output_tokens,
                       latency_ms, retries, error, success, task, ground_truth, legacy_output
    """
    test_id = record.get("test_id", _UNKNOWN)
    messages = record.get("input_messages", [])

    result = {
        "test_id": test_id,
        "model_id": model_id,
        "output_text": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
        "retries": 0,
        "error": None,
        "success": False,
        "task": record.get("task", _UNKNOWN),
        "ground_truth": record.get("ground_truth"),
        "legacy_output": record.get("output"),
        "expected_output_type": record.get("expected_output_type", ""),
    }

    request_body = {"model_id": model_id, "messages": messages}
    response_body = {}
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            start = time.time()
            response = providers_client.invoke(
                model_id=model_id,
                messages=messages,
                max_tokens=max_tokens,
            )
            elapsed_ms = (time.time() - start) * 1000

            if elapsed_ms > CALL_TIMEOUT_S * 1000:
                raise TimeoutError(
                    f"Durata {elapsed_ms:.0f}ms > timeout {CALL_TIMEOUT_S}s"
                )

            result["output_text"] = response["output_text"]
            result["input_tokens"] = response["input_tokens"]
            result["output_tokens"] = response["output_tokens"]
            result["latency_ms"] = round(elapsed_ms, 1)
            result["retries"] = attempt
            result["success"] = True

            response_body = response
            break

        except Exception as e:
            last_error = str(e)
            result["retries"] = attempt + 1
            logger.warning(
                "Retry %d/%d per test_id=%s model=%s: %s",
                attempt + 1, MAX_RETRIES, test_id, model_id, e,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(1 * (attempt + 1))  # backoff

    if not result["success"]:
        result["error"] = last_error
        logger.error(
            "Fallita test_id=%s model=%s dopo %d retry",
            test_id, model_id, MAX_RETRIES,
        )

    # Payload tracing — salva sempre, anche in caso di errore
    _save_payload(record, model_id, request_body, response_body or {"error": last_error})

    return result


def run_task_model_pair(
    records: list[dict], model_id: str, providers_client, max_tokens: int = 1024
) -> list[dict]:
    """Esegue inferenza su un batch di record per una coppia (task, model)."""
    results = []
    for record in records:
        res = run_single_inference(record, model_id, providers_client, max_tokens=max_tokens)
        results.append(res)
    return results


def run(records: list[dict], model_configs: list[dict], providers_client) -> dict:
    """Esegue le inferenze in parallelo per ogni coppia (task, modello).

    Args:
        records: lista di record dal data processing, con chiave 'task' già assegnata.
        model_configs: lista di config modello con almeno 'model_id'.
        providers_client: istanza di BedrockClient.

    Returns:
        dict con chiave per model_id -> lista di risultati inferenza.
    """
    # Raggruppa record per task
    by_task: dict[str, list[dict]] = {}
    for r in records:
        task = r.get("task", _UNKNOWN)
        by_task.setdefault(task, []).append(r)

    all_results: dict[str, list[dict]] = {}

    # Parallelismo per coppia (task, modello)
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for model_cfg in model_configs:
            model_id = model_cfg["model_id"]
            all_results[model_id] = []
            max_tokens = model_cfg.get("max_tokens", 1024)
            for task, task_records in by_task.items():
                future = executor.submit(
                    run_task_model_pair, task_records, model_id, providers_client, max_tokens
                )
                futures[future] = (model_id, task)

        for future in as_completed(futures):
            model_id, task = futures[future]
            try:
                results = future.result()
                all_results[model_id].extend(results)
                logger.info(
                    "Completata coppia task=%s model=%s: %d record",
                    task, model_id, len(results),
                )
            except Exception as e:
                logger.error("Errore coppia task=%s model=%s: %s", task, model_id, e)

    total = sum(len(v) for v in all_results.values())
    logger.info("Runner completato: %d risultati totali per %d modelli", total, len(model_configs))

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Runner — Step 3")
    parser.parse_args()

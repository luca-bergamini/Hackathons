"""Data Processing Agent — Step 1."""

import csv
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import boto3
from dotenv import load_dotenv

from src.providers.bedrock import BedrockClient

load_dotenv()

logger = logging.getLogger(__name__)

PAYLOAD_DIR = Path("payloads")
_UNKNOWN = "unknown"
_AWS_ACCOUNT_ID: str = os.environ.get("AWS_ACCOUNT_ID", "")

# Regex constants for heuristic patterns (SonarQube S1192)
_RE_TEXT = r"\btext\b"
_RE_JSON = r"\bjson\b"
_RE_SQL = r"\bsql\b"
_RE_REDACT = r"\bredact"

REQUIRED_FIELDS = {
    "test_id", "input_messages", "ground_truth",
    "expected_output_type", "metadata", "agent_id", "output",
}

TASK_CLASSES = [
    "classification", "context_qa", "metadata_extraction", "code_generation",
    "translation", "summarization", "rephrasing",
    "sql_generation", "pii_redaction", "intent_detection",
]

# ---------------------------------------------------------------------------
# Euristiche per task identification
# Ogni task ha:
#   - keywords nel system prompt
#   - keywords nel user message
#   - keywords nel expected_output_type
#   - peso (priority): più alto = match più specifico
# ---------------------------------------------------------------------------
_TASK_HEURISTICS: dict[str, dict] = {
    "classification": {
        "system_kw": [
            r"\bclassif", r"\bcategor", r"\blabel", r"\bassign.*class",
            r"\bsentiment", r"\btopic\b", r"\btag\b", r"\bclasse\b",
            r"\bcategoria\b", r"\betichetta\b", r"\bclassifica\b",
            r"\bspam\b", r"\bham\b", r"\blabel.?space\b",
            r"\boutput.*(?:label|the label)\b",
        ],
        "user_kw": [
            r"\bclassif", r"\bcategor", r"\blabel", r"\bwhich category\b",
            r"\bsentiment\b", r"\bclasse\b", r"\bcategoria\b",
            r"\blabel.?space\b",
        ],
        "output_type_kw": [r"\blabel\b"],
        "priority": 5,
    },
    "context_qa": {
        "system_kw": [
            r"\bcontext\b", r"\bdocument", r"\bpassage\b", r"\brag\b",
            r"\bbased on.*(?:text|context|document|passage)",
            r"\banswer.*(?:question|query).*(?:using|based|from|given)",
            r"\bquestion.?answer", r"\bqa\b", r"\bcontesto\b",
            r"\brispondi.*(?:domand|basandoti)", r"\bdocumento\b",
            r"\bverbatim\b", r"\bexact\s*span\b", r"\bunanswerable\b",
            r"\bchunks?\b", r"\bcite\b",
        ],
        "user_kw": [
            r"\bcontext\s*:", r"\bdocument\s*:", r"\bpassage\s*:",
            r"\bbased on the (?:above|following|given|provided)",
            r"\bgiven the (?:context|document|passage|text)",
            r"\bcontesto\s*:", r"\bdocumento\s*:",
            r"\bquestion\s*:", r"\b\[\d+\]",
        ],
        "output_type_kw": [_RE_TEXT],
        "priority": 6,
    },
    "metadata_extraction": {
        "system_kw": [
            r"\bextract", _RE_JSON, r"\bmetadata\b", r"\bstructur",
            r"\bkey[- ]?value", r"\bfield", r"\bschema\b", r"\bentit",
            r"\bestrai", r"\bestrazione\b", r"\bcampo\b", r"\bcampi\b",
            r"\bformato json\b",
        ],
        "user_kw": [
            r"\bextract.*(?:from|into|as)\b", _RE_JSON,
            r"\breturn.*json\b", r"\bkey\s*:", r"\bfield",
            r"\bestrai\b",
        ],
        "output_type_kw": [_RE_JSON],
        "priority": 7,
    },
    "code_generation": {
        "system_kw": [
            r"\bcode\b", r"\bprogram", r"\bfunction\b", r"\bimplement",
            r"\bunit\s*test", r"\bpython\b", r"\bjavascript\b",
            r"\balgorithm\b", r"\bclass\b.*(?:method|inherit)",
            r"\bwrite.*(?:code|function|program|script)",
            r"\bgenera.*codice\b", r"\bfunzione\b", r"\bscript\b",
            r"\bcodice\b",
        ],
        "user_kw": [
            r"\bwrite.*(?:code|function|class|program|script)\b",
            r"\bimplement\b", r"\bdef\s+\w+", r"\bgenerate.*code\b",
            r"\bunit\s*test", r"\bcodice\b", r"\bfunzione\b",
        ],
        "output_type_kw": [r"\bcode\b"],
        "priority": 8,
    },
    "translation": {
        "system_kw": [
            r"\btranslat", r"\btraduc", r"\btradurre\b", r"\btraduzione\b",
            r"\bfrom\s+\w+\s+to\s+\w+", r"\btarget\s*language\b",
            r"\bsource\s*language\b", r"\blingua\b",
        ],
        "user_kw": [
            r"\btranslat", r"\btraduc",
            r"\binto\s+(?:english|italian|french|german|spanish|chinese|japanese)",
            r"\bin\s+(?:inglese|italiano|francese|tedesco|spagnolo)",
        ],
        "output_type_kw": [_RE_TEXT],
        "priority": 9,
    },
    "summarization": {
        "system_kw": [
            r"\bsummar", r"\briassun", r"\bsynopsis\b", r"\bcondense\b",
            r"\bbrief\b.*(?:version|summary)", r"\bshorten\b",
            r"\bsintesi\b", r"\briassumi\b", r"\briassunto\b",
            r"\bconcise\b", r"\bkey\s*points?\b", r"\bhighlight",
            r"\bretain.*(?:important|key|main)",
            r"\b(?:2|3|two|three).*sentences?\b",
        ],
        "user_kw": [
            r"\bsummar", r"\briassun", r"\bin\s+(?:brief|short|few words)\b",
            r"\bcondense\b", r"\btl;?dr\b", r"\bsintesi\b",
            r"\bmeeting\s*notes?\b", r"\bhighlight\b",
        ],
        "output_type_kw": [_RE_TEXT],
        "priority": 10,
    },
    "rephrasing": {
        "system_kw": [
            r"\brephras", r"\brewrite\b", r"\breformulat", r"\bparaphras",
            r"\briformula", r"\briscriv", r"\bstile\b.*(?:cambia|diverso)",
            r"\btone\b", r"\btono\b",
            r"\bregister\b", r"\bformality\b", r"\bverbosity\b",
            r"\bvocabulary\b", r"\badjust.*(?:tone|style|formality)\b",
        ],
        "user_kw": [
            r"\brephras", r"\brewrite\b", r"\breformulat", r"\bparaphras",
            r"\briformula", r"\briscriv",
            r"\boriginal\s*message\b", r"\btarget.*(?:register|style)\b",
        ],
        "output_type_kw": [_RE_TEXT],
        "priority": 11,
    },
    "sql_generation": {
        "system_kw": [
            _RE_SQL, r"\bquery\b", r"\bselect\b", r"\bdatabase\b",
            r"\btable\b", r"\bschema\b.*(?:table|column|sql)",
            r"\bgenera.*query\b",
        ],
        "user_kw": [
            _RE_SQL, r"\bquery\b", r"\bselect\b.*\bfrom\b",
            r"\btable\b", r"\bwhere\b.*\band\b",
        ],
        "output_type_kw": [_RE_SQL],
        "priority": 12,
    },
    "pii_redaction": {
        "system_kw": [
            r"\bpii\b", _RE_REDACT, r"\bpersonal.*(?:data|info)",
            r"\banonymiz", r"\bsensitive\b.*(?:data|info)",
            r"\bprivacy\b", r"\bmask\b", r"\bemail\b.*(?:phone|name)",
            r"\bdati\s*personal", r"\boscura", r"\banonimizz",
            r"\bredacted", r"\b\[redacted", r"\bidentif.*information\b",
            r"\bname.*email.*phone\b",
        ],
        "user_kw": [
            _RE_REDACT, r"\bremove.*(?:pii|personal|sensitive)\b",
            r"\bmask\b", r"\banonymiz", r"\boscura\b",
        ],
        "output_type_kw": [_RE_TEXT, _RE_REDACT],
        "priority": 13,
    },
    "intent_detection": {
        "system_kw": [
            r"\bintent\b", r"\bintention\b", r"\buser.*intent\b",
            r"\bdetect.*intent\b", r"\bclassif.*intent\b",
            r"\bintento\b", r"\bscopo\b",
            r"\bprimary.*intent\b", r"\bmultiple\s*intents?\b",
            r"\bmessaggio\s*utente\b",
        ],
        "user_kw": [
            r"\bintent\b", r"\bwhat.*user.*want\b",
            r"\bintento\b",
            r"\bmessaggio\s*utente\b",
        ],
        "output_type_kw": [r"\blabel\b"],
        "priority": 14,
    },
}


def _extract_from_record(
    msgs: list,
    system_text: str,
    user_texts: list[str],
    max_user_samples: int,
) -> tuple[str, list[str]]:
    """Process one record's messages; return updated (system_text, user_texts)."""
    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        text = content if isinstance(content, str) else str(content)
        if role == "system" and not system_text:
            system_text = text
        if role == "user" and len(user_texts) < max_user_samples:
            user_texts.append(text)
    return system_text, user_texts


def _extract_messages(records: list[dict]) -> tuple[str, str]:
    """Estrae il system prompt e un campione di user messages dai record di un agent_id.

    Ritorna (system_text, combined_user_text).
    Il system prompt è uguale per tutti i record dello stesso agent_id,
    quindi ne basta uno. Per gli user messages ne campiona fino a 5.
    """
    system_text = ""
    user_texts: list[str] = []
    max_user_samples = 5

    for rec in records:
        msgs = rec.get("input_messages", [])
        if not isinstance(msgs, list):
            continue
        system_text, user_texts = _extract_from_record(
            msgs, system_text, user_texts, max_user_samples,
        )

    return system_text, "\n".join(user_texts)


# ---------------------------------------------------------------------------
# LLM-based task identification (most accurate)
# ---------------------------------------------------------------------------
_BEDROCK_CLIENT: BedrockClient | None = None
_LLM_CLASSIFIER_MODEL = "eu.amazon.nova-lite-v1:0"

_CLASSIFIER_PROMPT = """You are a task classifier. Given a system prompt and a sample user message from an LLM application, you must determine which SINGLE task category best describes this application.

The ONLY valid categories are:
- classification: Assigning predefined categories/labels to text (sentiment, topic, spam, etc.)
- context_qa: Answering questions using provided context/documents/passages (RAG-style)
- metadata_extraction: Extracting structured information from text into JSON format
- code_generation: Generating source code from specifications, including unit tests
- translation: Translating text from one language to another
- summarization: Generating concise summaries of text
- rephrasing: Reformulating text while preserving meaning (changing style/tone/structure)
- sql_generation: Generating SQL queries from natural language requests
- pii_redaction: Detecting and masking/redacting personally identifiable information
- intent_detection: Identifying user intent from a textual request

RULES:
- Output ONLY the category name, nothing else.
- If the system prompt mentions "classify" with a "label space" and asks to output a single label, it is CLASSIFICATION (even if the prompt starts with words like "rephrase" or other misleading verbs).
- If the system prompt asks to detect user INTENT from a set of intents, it is INTENT_DETECTION.
- Pay attention to the GOAL of the prompt, not superficial keywords.
- If the expected_output_type is "label", strongly prefer classification or intent_detection.
- If the expected_output_type is "json", strongly prefer metadata_extraction.
- If the expected_output_type is "code", it is code_generation.
- If the expected_output_type is "sql", it is sql_generation.
"""  # noqa: E501


def _get_bedrock_client() -> BedrockClient:
    global _BEDROCK_CLIENT  # noqa: PLW0603
    if _BEDROCK_CLIENT is None:
        region = os.environ.get("AWS_REGION", "eu-west-1")
        _BEDROCK_CLIENT = BedrockClient(region=region)
    return _BEDROCK_CLIENT


def _save_payload(
    call_id: str, model_id: str, request_body: dict, response_body: dict,
) -> None:
    """Salva request/response JSON per payload tracing (OBBLIGATORIO per validità)."""
    PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_call_id = re.sub(r'[^a-zA-Z0-9_-]', '_', call_id)
    safe_model = re.sub(r'[^a-zA-Z0-9_-]', '_', model_id)
    filepath = PAYLOAD_DIR / f"task_classify_{safe_call_id}_{safe_model}.json"
    payload = {
        "call_id": call_id,
        "model_id": model_id,
        "request": request_body,
        "response": response_body,
        "timestamp": time.time(),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _classify_task_with_llm(
    system_text: str,
    user_text_sample: str,
    output_type: str | None,
    agent_id: str = _UNKNOWN,
) -> str | None:
    """Classifica il task tramite una chiamata LLM a Bedrock (Converse API).

    Ritorna il nome del task o None se la chiamata fallisce.
    """
    user_content = (
        f"SYSTEM PROMPT:\n{system_text[:2000]}\n\n"
        f"SAMPLE USER MESSAGE:\n{user_text_sample[:1000]}\n\n"
    )
    if output_type:
        user_content += f"EXPECTED OUTPUT TYPE: {output_type}\n\n"
    user_content += "What is the task category? Reply with ONLY the category name."

    request_body = {
        "model_id": _LLM_CLASSIFIER_MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "system": _CLASSIFIER_PROMPT[:200] + "...",
        "max_tokens": 30,
        "temperature": 0.0,
    }

    try:
        client = _get_bedrock_client()
        result = client.invoke(
            model_id=_LLM_CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": user_content}],
            system=_CLASSIFIER_PROMPT,
            max_tokens=30,
            temperature=0.0,
        )
        raw = result["output_text"].strip().lower()

        # Payload tracing — salva request/response
        _save_payload(agent_id, _LLM_CLASSIFIER_MODEL, request_body, result)

        # Pulisci la risposta: rimuovi punteggiatura e spazi extra
        raw = raw.strip().strip(".").strip()
        # Mappa risposte con trattino/spazio al formato corretto
        raw = raw.replace(" ", "_").replace("-", "_")

        if raw in TASK_CLASSES:
            return raw

        # Fuzzy match: controlla se qualche task class è contenuta nella risposta
        for tc in TASK_CLASSES:
            if tc in raw:
                return tc

        logger.warning("LLM returned unexpected task: '%s'", raw)
        return None

    except Exception as e:
        # Payload tracing anche in caso di errore
        _save_payload(
            agent_id, _LLM_CLASSIFIER_MODEL, request_body, {"error": str(e)},
        )
        logger.warning("LLM classification failed: %s", e)
        return None


def _score_task(task: str, system_text: str, user_text: str,
                output_type: str | None) -> float:
    """Calcola uno score per un task basandosi sulle keyword matches.

    Pesi:
    - system prompt match:        3.0 per keyword
    - user message match:         2.0 per keyword
    - expected_output_type match: 5.0 per keyword (molto specifico)
    """
    rules = _TASK_HEURISTICS[task]
    score = 0.0

    text_lower_sys = system_text.lower()
    text_lower_usr = user_text.lower()

    for pattern in rules["system_kw"]:
        if re.search(pattern, text_lower_sys):
            score += 3.0

    for pattern in rules["user_kw"]:
        if re.search(pattern, text_lower_usr):
            score += 2.0

    if output_type:
        ot_lower = output_type.lower()
        for pattern in rules["output_type_kw"]:
            if re.search(pattern, ot_lower):
                score += 5.0

    return score


def identify_task(records: list[dict]) -> str:
    """Identifica il task di un gruppo di record (stesso agent_id).

    Strategia:
    1. PRIMARIA — LLM-based: invia system prompt + sample user message a un LLM
       su Bedrock, che classifica con precisione ~100%.
    2. FALLBACK — Euristiche keyword-based: in caso di errore LLM o risposta
       non valida, usa le regex per determinare il task.

    Args:
        records: lista di record appartenenti allo stesso agent_id

    Returns:
        stringa del task identificato (es. 'classification', 'context_qa', ...)
    """
    if not records:
        return "classification"

    system_text, user_text = _extract_messages(records)

    # Prendi agent_id dal primo record
    agent_id = records[0].get("agent_id", _UNKNOWN) if records else _UNKNOWN

    # Prendi l'expected_output_type dal primo record valido
    output_type = None
    for rec in records:
        ot = rec.get("expected_output_type")
        if ot:
            output_type = str(ot)
            break

    # --- 1. Tentativo LLM (più preciso) ---
    llm_task = _classify_task_with_llm(system_text, user_text, output_type, agent_id)
    if llm_task:
        logger.info(
            "Task identified (LLM): %s | system_preview='%s'",
            llm_task, system_text[:80],
        )
        return llm_task

    # --- 2. Fallback euristiche ---
    best_task = "classification"
    best_score = 0.0

    for task in TASK_CLASSES:
        score = _score_task(task, system_text, user_text, output_type)
        if score > best_score:
            best_score = score
            best_task = task

    logger.info(
        "Task identified (heuristic fallback): %s (score=%.1f) | system_preview='%s'",
        best_task, best_score, system_text[:80],
    )
    return best_task


def load_dataset_from_s3(bucket: str, key: str) -> list[dict]:
    """Scarica il dataset JSONL da S3 e ritorna una lista di dict."""
    region = os.environ.get("AWS_REGION", "eu-west-1")
    s3 = boto3.client("s3", region_name=region)
    logger.info("Downloading s3://%s/%s ...", bucket, key)
    _owner_kw = {"ExpectedBucketOwner": _AWS_ACCOUNT_ID} if _AWS_ACCOUNT_ID else {}
    response = s3.get_object(Bucket=bucket, Key=key, **_owner_kw)
    body = response["Body"].read().decode("utf-8")

    records: list[dict] = []
    for line_num, line in enumerate(body.strip().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON at line %d", line_num)
    logger.info("Loaded %d records from S3", len(records))
    return records


def validate_record(record: dict) -> bool:
    """Verifica che un record sia valido.

    Un record è valido se:
    1. Contiene tutte le chiavi di REQUIRED_FIELDS
    2. input_messages contiene almeno un messaggio con role='system'
       e almeno un messaggio con role='user'
    """
    if not REQUIRED_FIELDS.issubset(record.keys()):
        return False

    messages = record.get("input_messages")
    if not isinstance(messages, list):
        return False

    roles = {msg.get("role") for msg in messages if isinstance(msg, dict)}
    return "system" in roles and "user" in roles


def split_by_agent(records: list[dict]) -> dict:
    """Raggruppa i record per agent_id. Ritorna {agent_id: [records]}."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        agent_id = rec.get("agent_id", _UNKNOWN)
        groups[agent_id].append(rec)
    return dict(groups)


def generate_profiling_csv(profiles: dict, output_path: str) -> None:
    """Genera il CSV di profiling.

    profiles: {agent_id: {"task": str, "records": [dict]}}
    Output CSV: agent_id,task,numero_record,percentuale_ground_truth,numero_record_malformati
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent_id", "task", "numero_record",
            "percentuale_ground_truth", "numero_record_malformati",
        ])
        for agent_id, info in sorted(profiles.items()):
            all_records = info["records"]
            task = info["task"]
            total = len(all_records)

            valid_records = [r for r in all_records if validate_record(r)]
            malformed = total - len(valid_records)

            valid_with_gt = sum(
                1 for r in valid_records if r.get("ground_truth") is not None
            )
            valid_count = len(valid_records)
            pct_gt = round((valid_with_gt / valid_count) * 100, 1) if valid_count > 0 else 0.0

            writer.writerow([agent_id, task, total, pct_gt, malformed])

    logger.info("Profiling CSV written to %s", output_path)


def run(
    s3_bucket: str | None = None,
    dataset_key: str = "dataset/dataset.jsonl",
    output_csv: str = "deliverable_intermedio_1.csv",
) -> dict:
    """Esegue l'intero Step 1: load → validate → split → identify → profile CSV.

    Returns:
        dict with:
          - "agents": {agent_id: {"task": str, "valid_records": [...], "malformed_records": [...]}}
          - "profiles": [list of profiling row dicts]
    """
    if s3_bucket is None:
        s3_bucket = os.environ.get("S3_BUCKET", "")
    if not s3_bucket:
        raise ValueError("S3_BUCKET not set. Pass --bucket or set S3_BUCKET env var.")

    # 1. Carica dataset da S3
    all_records = load_dataset_from_s3(s3_bucket, dataset_key)

    # 2. Suddividi per agent_id
    agent_groups = split_by_agent(all_records)
    logger.info("Found %d distinct agent_ids.", len(agent_groups))

    # 3. Per ogni agent_id: validate, identify task, compute profiling
    profiles_list: list[dict] = []
    agents_output: dict[str, dict] = {}
    profiles_for_csv: dict[str, dict] = {}

    for agent_id, recs in sorted(agent_groups.items()):
        total = len(recs)
        valid = [r for r in recs if validate_record(r)]
        malformed = [r for r in recs if not validate_record(r)]
        num_valid = len(valid)
        num_malformed = len(malformed)

        # Identify task (usa valid records, fallback a tutti se nessuno valido)
        task = identify_task(valid if valid else recs)

        # Ground truth percentage (solo su record validi)
        if num_valid > 0:
            with_gt = sum(1 for r in valid if r.get("ground_truth") is not None)
            pct_gt = round((with_gt / num_valid) * 100, 1)
        else:
            pct_gt = 0.0

        profiles_list.append({
            "agent_id": agent_id,
            "task": task,
            "numero_record": total,
            "percentuale_ground_truth": pct_gt,
            "numero_record_malformati": num_malformed,
        })

        agents_output[agent_id] = {
            "task": task,
            "valid_records": valid,
            "malformed_records": malformed,
        }

        # Per generate_profiling_csv (backward compat)
        profiles_for_csv[agent_id] = {"task": task, "records": recs}

        logger.info(
            "  %s -> task=%s  total=%d  valid=%d  malformed=%d  gt=%.1f%%",
            agent_id, task, total, num_valid, num_malformed, pct_gt,
        )

    # 4. Genera CSV
    generate_profiling_csv(profiles_for_csv, output_csv)

    return {"agents": agents_output, "profiles": profiles_list}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Data Processing — Step 1")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--key", default="dataset/dataset.jsonl")
    parser.add_argument("--output", default="deliverable_intermedio_1.csv")
    args = parser.parse_args()
    result = run(s3_bucket=args.bucket, dataset_key=args.key, output_csv=args.output)
    print(json.dumps(result["profiles"], indent=2))

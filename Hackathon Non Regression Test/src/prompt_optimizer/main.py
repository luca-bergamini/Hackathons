"""Prompt Optimizer — Bonus.

Loop iterativo: genera varianti del system prompt → Runner → Evaluator → seleziona migliore.
Il modello Candidate resta invariato; cambiano solo i system prompt.
"""

import copy
import json
import logging
import re

from src.evaluation import main as evaluation
from src.runner import main as runner

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 3
DEFAULT_NUM_VARIANTS = 3
DEFAULT_BEAM_WIDTH = 2  # beam search: top-K candidati tra iterazioni
MIN_IMPROVEMENT = 0.01  # soglia minima di miglioramento tra iterazioni

# Default multi-objective weights (sum must equal 1.0)
# quality: primary metric — how correct are the outputs
# cost: fraction of baseline cost saved (higher is better)
# latency: fraction of baseline latency saved (higher is better)
DEFAULT_WEIGHTS = {"quality": 0.70, "cost": 0.15, "latency": 0.15}

# --- Duplicate string constants (SonarQube S1192) ---
_K_AVG_SCORE = "avg_score"
_K_NUM_CORRECT = "num_correct"
_K_NUM_RECORDS = "num_records"
_K_SCORE = "score"


# ---------------------------------------------------------------------------
# Genera varianti di system prompt via LLM
# ---------------------------------------------------------------------------

def generate(
    agent_id: str,
    model_id: str,
    current_prompt: str,
    eval_results: dict,
    llm_client,
    num_variants: int = DEFAULT_NUM_VARIANTS,
) -> list[str]:
    """Genera N varianti del system prompt chiedendo a un LLM.

    Args:
        agent_id: identificatore dell'agente applicativo.
        model_id: modello usato per generare varianti (tipicamente il Judge).
        current_prompt: system prompt corrente da migliorare.
        eval_results: risultati della valutazione corrente (per dare contesto).
        llm_client: istanza BedrockClient.
        num_variants: numero di varianti da generare.

    Returns:
        Lista di stringhe — ciascuna è una variante del system prompt.
    """
    # Prepara feedback sintetico dalle eval
    feedback = _format_eval_feedback(eval_results)

    generation_prompt = f"""You are an expert prompt engineer. Your task is to improve a system prompt for an AI agent.

## Current System Prompt
{current_prompt}

## Current Performance Feedback
{feedback}

## Instructions
Generate exactly {num_variants} improved variants of the system prompt above.
Each variant should try a DIFFERENT optimization strategy:
1. Make instructions more explicit and structured (add format constraints, step-by-step)
2. Add clearer output format requirements and examples
3. Simplify and focus on core task with strong constraints

Rules:
- Keep the same task/objective as the original prompt
- Each variant must be a complete, standalone system prompt
- Do NOT include explanations — only the prompt text

Respond with valid JSON: {{"variants": ["<variant_1>", "<variant_2>", "<variant_3>"]}}"""

    try:
        response = llm_client.invoke(
            model_id=model_id,
            messages=[{"role": "user", "content": generation_prompt}],
            max_tokens=2048,
            temperature=0.7,
        )
        text = response["output_text"].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            variants = parsed.get("variants", [])
            if isinstance(variants, list) and variants:
                return [str(v) for v in variants[:num_variants]]
    except Exception as e:
        logger.error("Errore generazione varianti prompt per agent_id=%s: %s", agent_id, e)

    # Fallback: restituisce leggere modifiche meccaniche
    return [
        current_prompt + "\n\nBe precise and concise in your response.",
        "You are an expert. " + current_prompt
        + "\n\nRespond ONLY with the requested output.",
        current_prompt + "\n\nIMPORTANT: Follow the output format exactly as specified.",
    ][:num_variants]


def _format_eval_feedback(eval_results: dict | None) -> str:
    """Converte risultati eval in feedback testuale per il prompt generator."""
    if not eval_results:
        return "No previous evaluation available."

    lines = []
    score = eval_results.get(_K_AVG_SCORE)
    if score is not None:
        lines.append(f"Average score: {score:.3f}")

    num = eval_results.get(_K_NUM_RECORDS, 0)
    if num:
        lines.append(f"Records evaluated: {num}")

    correct = eval_results.get(_K_NUM_CORRECT, 0)
    if num:
        lines.append(f"Correct: {correct}/{num} ({correct / num * 100:.1f}%)")

    errors = eval_results.get("common_errors", [])
    if errors:
        lines.append("Common errors: " + "; ".join(errors[:3]))

    return "\n".join(lines) if lines else "Score below target."


# ---------------------------------------------------------------------------
# Valuta un singolo prompt candidato
# ---------------------------------------------------------------------------

def evaluate(
    _agent_id: str,
    model_id: str,
    prompt: str,
    records: list[dict],
    llm_client,
    judge_client=None,
    judge_model_id: str | None = None,
) -> dict:
    """Esegue Runner + Evaluator su un set di record con un dato system prompt.

    Sostituisce il system message nei record con `prompt`, esegue inferenza
    col modello Candidate, poi valuta con l'Evaluator.

    Returns:
        dict con avg_score, num_records, num_correct, per_record scores.
    """
    if judge_client is None:
        judge_client = llm_client
    if judge_model_id is None:
        judge_model_id = model_id

    # Sostituisci il system prompt in ogni record
    modified_records = _replace_system_prompt(records, prompt)

    # Esegui inferenza
    model_config = [{"model_id": model_id, "max_tokens": 1024}]
    inference_results = runner.run(modified_records, model_config, llm_client)

    # Valuta
    eval_results = evaluation.run(inference_results, judge_client, judge_model_id)

    # Aggrega
    per_record = eval_results.get("per_record", [])
    scores = [r.get(_K_SCORE, 0.0) for r in per_record]
    num_correct = sum(1 for r in per_record if r.get("correct", False))

    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    return {
        _K_AVG_SCORE: avg_score,
        _K_NUM_RECORDS: len(scores),
        _K_NUM_CORRECT: num_correct,
        "per_record": per_record,
    }


def _replace_system_prompt(records: list[dict], new_prompt: str) -> list[dict]:
    """Crea copie dei record con il system prompt sostituito."""
    modified = []
    for record in records:
        r = copy.deepcopy(record)
        messages = r.get("input_messages", [])
        found = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = new_prompt
                found = True
                break
        if not found:
            messages.insert(0, {"role": "system", "content": new_prompt})
        r["input_messages"] = messages
        modified.append(r)
    return modified


# ---------------------------------------------------------------------------
# Multi-objective scoring helpers (BN_PROMPT_OPTIMIZATION)
# ---------------------------------------------------------------------------

def _compute_composite_score(
    quality: float,
    cost: float | None,
    latency: float | None,
    baseline_cost: float | None,
    baseline_latency: float | None,
    weights: dict,
) -> float:
    """BN_PROMPT_OPTIMIZATION: multi-objective composite score.

    Combines quality, cost efficiency, and latency efficiency using
    configurable weights. Cost and latency are normalized relative to baseline
    so that improvements reduce the composite penalty.

    Args:
        quality: quality score in [0, 1] (higher is better).
        cost: current variant average cost per request (lower is better).
        latency: current variant average latency ms (lower is better).
        baseline_cost: baseline cost per request for normalization.
        baseline_latency: baseline latency ms for normalization.
        weights: dict with keys 'quality', 'cost', 'latency' summing to 1.0.
    """
    w_q = weights.get("quality", DEFAULT_WEIGHTS["quality"])
    w_c = weights.get("cost", DEFAULT_WEIGHTS["cost"])
    w_l = weights.get("latency", DEFAULT_WEIGHTS["latency"])

    # Cost efficiency: 1.0 if same cost, > 1.0 if cheaper, < 1.0 if more expensive
    cost_score = 1.0
    if cost is not None and baseline_cost and baseline_cost > 0:
        cost_score = min(1.0, baseline_cost / max(cost, 1e-9))

    # Latency efficiency: same logic
    latency_score = 1.0
    if latency is not None and baseline_latency and baseline_latency > 0:
        latency_score = min(1.0, baseline_latency / max(latency, 1e-9))

    return round(quality * w_q + cost_score * w_c + latency_score * w_l, 4)


def _statistical_significance(
    baseline_scores: list[float],
    best_scores: list[float],
) -> dict:
    """BN_PROMPT_OPTIMIZATION: test whether best_prompt is statistically better than baseline.

    Uses a paired t-test (scipy.stats.ttest_rel) when records are paired,
    or a bootstrap confidence interval as fallback.
    Returns p_value, is_significant (p < 0.05), and method used.
    """
    n = min(len(baseline_scores), len(best_scores))
    if n < 2:
        return {"method": "none", "p_value": None, "is_significant": False,
                "message": "Not enough samples for significance test"}

    base = baseline_scores[:n]
    best = best_scores[:n]
    mean_diff = sum(b - a for a, b in zip(base, best)) / n

    try:
        from scipy import stats as scipy_stats  # noqa: PLC0415
        stat, p_value = scipy_stats.ttest_rel(best, base, alternative="greater")
        return {
            "method": "paired_ttest",
            "t_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "mean_difference": round(mean_diff, 4),
            "is_significant": float(p_value) < 0.05,
        }
    except ImportError:
        pass

    # Bootstrap fallback: resample differences and check if 95% CI > 0
    import random as _random
    diffs = [b - a for a, b in zip(base, best)]
    bootstrap_means = []
    for _ in range(1000):
        sample = [_random.choice(diffs) for _ in range(len(diffs))]  # noqa: S311
        bootstrap_means.append(sum(sample) / len(sample))
    bootstrap_means.sort()
    ci_low = bootstrap_means[25]   # 2.5th percentile
    ci_high = bootstrap_means[975]  # 97.5th percentile
    is_significant = ci_low > 0
    return {
        "method": "bootstrap_95ci",
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "mean_difference": round(mean_diff, 4),
        "is_significant": is_significant,
    }


def _compute_ablation_report(history: list[dict]) -> list[dict]:
    """BN_PROMPT_OPTIMIZATION: ablation report showing contribution of each iteration.

    For each consecutive pair in optimization history, computes the score delta
    and a diff summary of prompt changes (word-level additions/removals).
    """
    ablation = []
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        delta = round(curr[_K_SCORE] - prev[_K_SCORE], 4)

        # Word-level diff to identify what changed
        prev_words = set(str(prev.get("prompt", "")).split())
        curr_words = set(str(curr.get("prompt", "")).split())
        added = list(curr_words - prev_words)[:10]
        removed = list(prev_words - curr_words)[:10]

        ablation.append({
            "from_iteration": prev["iteration"],
            "to_iteration": curr["iteration"],
            "score_delta": delta,
            "score_before": prev[_K_SCORE],
            "score_after": curr[_K_SCORE],
            "words_added": added,
            "words_removed": removed,
            "contribution": "positive" if delta > 0 else ("neutral" if delta == 0 else "negative"),
        })
    return ablation


# ---------------------------------------------------------------------------
# Loop di ottimizzazione principale
# ---------------------------------------------------------------------------

def _avg_latency_from_per_record(per_record: list[dict]) -> float | None:
    """Extract average latency_ms from per_record list; return None if unavailable."""
    latencies = [r.get("latency_ms") for r in per_record if r.get("latency_ms")]
    return sum(latencies) / len(latencies) if latencies else None


def _build_optimization_report(
    agent_id: str,
    model_id: str,
    base_prompt: str,
    baseline_score: float,
    best: dict,
    beam: list,
    beam_width: int,
    history: list,
    all_variants: list,
    baseline_eval: dict,
    effective_weights: dict,
) -> dict:
    """Assemble the final optimization report with significance and ablation data."""
    best_prompt = best["prompt"]
    best_score = best[_K_SCORE]

    baseline_per_record = baseline_eval.get("per_record", [])
    best_per_record = best.get("eval", {}).get("per_record", [])
    baseline_scores_list = [r.get(_K_SCORE, 0.0) for r in baseline_per_record]
    best_scores_list = [r.get(_K_SCORE, 0.0) for r in best_per_record]
    significance = _statistical_significance(baseline_scores_list, best_scores_list)
    ablation = _compute_ablation_report(history)

    # BN_PROMPT_OPTIMIZATION: use real latency from per_record for multi-objective composite
    baseline_latency = _avg_latency_from_per_record(baseline_per_record)
    best_latency = _avg_latency_from_per_record(best_per_record)

    composite = _compute_composite_score(
        quality=best_score,
        cost=None, latency=best_latency,
        baseline_cost=None, baseline_latency=baseline_latency,
        weights=effective_weights,
    )

    return {
        "agent_id": agent_id,
        "model_id": model_id,
        "baseline_prompt": base_prompt,
        "baseline_score": baseline_score,
        "best_prompt": best_prompt,
        "best_score": best_score,
        "composite_score": composite,
        "weights": effective_weights,
        "improvement": round(best_score - baseline_score, 3),
        "iterations_run": len(history) - 1,
        "beam_width": beam_width,
        "history": history,
        "all_variants": all_variants,
        "significance_test": significance,
        "ablation_report": ablation,
    }


def _run_single_beam_iteration(
    beam: list[dict],
    agent_id: str,
    model_id: str,
    records: list[dict],
    llm_client,
    judge_client,
    judge_model_id: str | None,
    num_variants: int,
    iteration: int,
) -> list[dict]:
    """Generate and evaluate all variants for one beam search iteration."""
    candidates: list[dict] = []
    for bi, beam_entry in enumerate(beam):
        variants = generate(
            agent_id, model_id,
            beam_entry["prompt"],
            beam_entry["eval"],
            llm_client, num_variants,
        )
        for vi, variant in enumerate(variants):
            label = f"beam[{bi}]\u2192var[{vi}]"
            logger.info("  Valutando %s (iter %d)...", label, iteration)
            try:
                v_eval = evaluate(
                    agent_id, model_id, variant,
                    records, llm_client,
                    judge_client, judge_model_id,
                )
                v_score = v_eval[_K_AVG_SCORE]
            except Exception as e:
                logger.error("Errore valutazione %s: %s", label, e)
                v_score = 0.0
                v_eval = {_K_AVG_SCORE: 0.0, _K_NUM_CORRECT: 0, _K_NUM_RECORDS: 0}
            candidates.append({
                "iteration": iteration,
                "variant_index": vi,
                "beam_index": bi,
                "prompt": variant,
                _K_SCORE: v_score,
                _K_NUM_CORRECT: v_eval.get(_K_NUM_CORRECT, 0),
                "eval": v_eval,
            })
    return candidates


def run(
    agent_id: str,
    model_id: str,
    base_prompt: str,
    records: list[dict],
    llm_client,
    judge_client=None,
    judge_model_id: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    num_variants: int = DEFAULT_NUM_VARIANTS,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    weights: dict | None = None,
) -> dict:
    """Esegue beam search iterativa per Prompt Optimization.

    BN_PROMPT_OPTIMIZATION:
    - Multi-objective: accepts configurable weights for quality, cost, latency.
    - Statistical significance: runs paired t-test (or bootstrap) comparing best vs baseline.
    - Ablation report: shows score contribution of each iteration's prompt changes.

    Args:
        agent_id: ID dell'agente applicativo.
        model_id: modello Candidate (resta invariato).
        base_prompt: system prompt iniziale.
        records: record del dataset per questo agent_id.
        llm_client: BedrockClient per inferenza.
        judge_client: BedrockClient per il Judge.
        judge_model_id: model_id del Judge.
        max_iterations: numero massimo di iterazioni.
        num_variants: varianti per ciascun candidato nel beam.
        beam_width: numero di candidati mantenuti tra iterazioni.
        weights: dict with keys 'quality', 'cost', 'latency' (default: {quality:0.70,...}).

    Returns:
        dict con: best_prompt, baseline_score, best_score, history, all_variants,
                  iterations_run, beam_width, significance_test, ablation_report.
    """
    if judge_client is None:
        judge_client = llm_client
    beam_width = max(1, beam_width)
    effective_weights = weights if weights else DEFAULT_WEIGHTS

    logger.info(
        "Avvio Prompt Optimization (beam search, width=%d) "
        "per agent_id=%s, model=%s, max_iter=%d, weights=%s",
        beam_width, agent_id, model_id, max_iterations, effective_weights,
    )

    # Baseline: valuta il prompt originale
    baseline_eval = evaluate(
        agent_id, model_id, base_prompt, records, llm_client,
        judge_client, judge_model_id,
    )
    baseline_score = baseline_eval[_K_AVG_SCORE]

    history = [
        {
            "iteration": 0,
            "prompt": base_prompt,
            _K_SCORE: baseline_score,
            _K_NUM_CORRECT: baseline_eval[_K_NUM_CORRECT],
            "is_baseline": True,
        }
    ]
    all_variants: list[dict] = []

    # Inizializza il beam con il prompt di base
    beam: list[dict] = [
        {
            "prompt": base_prompt,
            _K_SCORE: baseline_score,
            "eval": baseline_eval,
        }
    ]

    for iteration in range(1, max_iterations + 1):
        beam_best = max(b[_K_SCORE] for b in beam)
        logger.info(
            "Iterazione %d/%d — beam size=%d, best=%.3f",
            iteration, max_iterations, len(beam), beam_best,
        )

        candidates = _run_single_beam_iteration(
            beam, agent_id, model_id, records, llm_client,
            judge_client, judge_model_id, num_variants, iteration,
        )
        all_variants.extend(candidates)

        if not candidates:
            break

        # Unisci beam corrente + nuovi candidati,
        # seleziona top-K (beam_width) per prossima iterazione
        pool = [
            {"prompt": b["prompt"], _K_SCORE: b[_K_SCORE], "eval": b["eval"]}
            for b in beam
        ] + [
            {"prompt": c["prompt"], _K_SCORE: c[_K_SCORE], "eval": c.get("eval", {})}
            for c in candidates
        ]
        pool.sort(key=lambda x: x[_K_SCORE], reverse=True)
        new_beam = pool[:beam_width]

        # Check se c'è miglioramento rispetto a passata
        new_best = new_beam[0][_K_SCORE]
        old_best = beam[0][_K_SCORE]
        if new_best <= old_best + MIN_IMPROVEMENT:
            logger.info(
                "  Nessun miglioramento (%.3f → %.3f). "
                "Stop.",
                old_best, new_best,
            )
            history.append({
                "iteration": iteration,
                "prompt": new_beam[0]["prompt"],
                _K_SCORE: new_beam[0][_K_SCORE],
                _K_NUM_CORRECT: new_beam[0]["eval"].get(
                    _K_NUM_CORRECT, 0,
                ),
                "is_baseline": False,
                "stopped_early": True,
            })
            beam = new_beam
            break

        beam = new_beam
        history.append({
            "iteration": iteration,
            "prompt": beam[0]["prompt"],
            _K_SCORE: beam[0][_K_SCORE],
            _K_NUM_CORRECT: beam[0]["eval"].get(
                _K_NUM_CORRECT, 0,
            ),
            "is_baseline": False,
        })

    # Il miglior prompt è il top del beam finale
    best = max(beam, key=lambda x: x[_K_SCORE])

    # BN_PROMPT_OPTIMIZATION: statistical significance test, ablation, composite score
    report = _build_optimization_report(
        agent_id=agent_id,
        model_id=model_id,
        base_prompt=base_prompt,
        baseline_score=baseline_score,
        best=best,
        beam=beam,
        beam_width=beam_width,
        history=history,
        all_variants=all_variants,
        baseline_eval=baseline_eval,
        effective_weights=effective_weights,
    )

    logger.info(
        "Prompt Optimization completata (beam=%d): "
        "baseline=%.3f → best=%.3f (Δ=%.3f) in %d iter",
        beam_width, baseline_score, report["best_score"],
        report["improvement"], report["iterations_run"],
    )

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prompt Optimizer — Bonus")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-prompt", required=True)
    parser.add_argument("--records-json", required=True)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--num-variants", type=int, default=DEFAULT_NUM_VARIANTS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.records_json, encoding="utf-8") as f:
        records = json.load(f)

    from src.providers.bedrock import BedrockClient

    client = BedrockClient()
    result = run(
        agent_id=args.agent_id,
        model_id=args.model,
        base_prompt=args.base_prompt,
        records=records,
        llm_client=client,
        max_iterations=args.max_iterations,
        num_variants=args.num_variants,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

"""Dashboard Streamlit per la pipeline NRT.

Sezioni:
1. Config Job — configurazione e avvio pipeline (task 3.1)
2. Monitoraggio — lista job con stato (task 3.1)
3. Risultati — metriche aggregate + download report (task 3.1)
4. Insight & Confronto — verdict per record, grafici confronto modelli (task 3.10)
5. Prompt Optimization — ottimizzazione prompt per agent_id (task 3.13)
"""

import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so "from src.xxx" imports work
# when Streamlit runs this file directly.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import boto3  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger(__name__)

# --- Duplicate string constants (SonarQube S1192) ---
_K_STATUS = "status"
_K_COMPLETED = "completed"
_K_SELECTED_TASKS = "selected_tasks"
_K_SELECTED_MODELS = "selected_models"
_K_DATASET_KEY = "dataset_key"
_K_MODEL_ID = "model_id"
_K_OVERALL_SCORE = "overall_score"
_K_AVG_SCORE = "avg_score"
_MODELS_CONFIG_FILENAME = "models.yaml"
_PAGE_RISULTATI = "\U0001f4c8 Risultati"
_PAGE_PROMPT_OPT = "\U0001f6e0\ufe0f Prompt Optimization"
_LBL_SCORE_MEDIO = "Score Medio"

st.set_page_config(page_title="NRT Dashboard", page_icon="🧪", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, object] = {
    "jobs": {},            # job_id -> job dict
    "detected_data": None, # result of "Detect Tasks"
    "prompt_opt_results": None,
    "synthetic_records": None,  # generated synthetic records
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ---------------------------------------------------------------------------
# Helpers — S3
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Caricamento dataset da S3…")
def _list_s3_datasets(bucket: str, prefix: str = "dataset/") -> list[str]:
    """Elenca i file .jsonl disponibili nel bucket S3."""
    region = os.environ.get("AWS_REGION", "eu-west-1")
    s3 = boto3.client("s3", region_name=region)
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".jsonl"):
                keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# Helpers — pipeline execution
# ---------------------------------------------------------------------------


def _setup_pipeline_clients(job: dict, region: str, model_selection, bedrock_client_cls):
    """Initialise AWS clients, load model configs, build consensus judge list."""
    config_path = Path(_PROJECT_ROOT) / "configs" / _MODELS_CONFIG_FILENAME
    llm_client = bedrock_client_cls(region=region)
    models = model_selection.load_model_configs(config_path)
    judge_cfg = model_selection.load_judge_config(config_path)
    judge_model_id = judge_cfg[_K_MODEL_ID]
    judge_client = bedrock_client_cls(region=region)
    selected_ids = set(job[_K_SELECTED_MODELS])
    model_configs = [m for m in models if m[_K_MODEL_ID] in selected_ids] or models
    consensus_judge_ids = [judge_model_id]
    extra_judges = [m[_K_MODEL_ID] for m in model_configs if m[_K_MODEL_ID] != judge_model_id]
    if extra_judges:
        consensus_judge_ids.append(extra_judges[0])
    return llm_client, judge_client, judge_model_id, model_configs, consensus_judge_ids


def _build_records_to_run(job: dict, detected: dict) -> list[dict]:
    """Filter agent records by selected tasks; attach task label to each record."""
    agent_groups = detected["agent_groups"]
    selected_tasks = set(job[_K_SELECTED_TASKS])
    records: list[dict] = []
    for agent_id, recs in agent_groups.items():
        task = detected["agent_tasks"].get(agent_id, "unknown")
        if task in selected_tasks:
            for r in recs:
                r["task"] = task
                records.append(r)
    return records


def _collect_previous_low_scores(all_jobs: dict) -> list[dict]:
    """Return low-score records from the last completed job for edge-case targeting."""
    for prev_job in all_jobs.values():
        if prev_job.get(_K_STATUS) == _K_COMPLETED and prev_job.get("eval_results"):
            return [
                r for r in prev_job["eval_results"].get("per_record", [])
                if r.get("score", 1.0) < 0.3
            ][:50]
    return []


def _enrich_with_synthetic(
    job: dict,
    job_id: str,
    records_to_run: list[dict],
    evaluation,
    synthetic_ds,
    llm_client,
    model_configs: list[dict],
    judge_client,
    judge_model_id: str,
    all_jobs: dict,
) -> list[dict]:
    """Run synthetic dataset enrichment; return enriched record list."""
    n_per_task = job.get("synth_records_per_task", 5)
    seed_by_task: dict[str, list[dict]] = {}
    for r in records_to_run:
        seed_by_task.setdefault(r.get("task", "unknown"), []).append(r)
    previous_low_scores = _collect_previous_low_scores(all_jobs)

    def _evaluator_fn(record: dict) -> float:
        task_name = record.get("task", "")
        ev = evaluation.TASK_EVALUATORS.get(task_name)
        if ev is None:
            return 0.5
        try:
            res = ev(record, judge_client, judge_model_id)
            return float(res.get("score", 0.5))
        except Exception:
            return 0.5

    synth_records = synthetic_ds.run(
        tasks=list(seed_by_task.keys()),
        llm_client=llm_client,
        model_id=model_configs[0][_K_MODEL_ID],
        records_per_task=n_per_task,
        output_path="",
        seed_records_by_task=seed_by_task,
        evaluator_fn=_evaluator_fn,
        low_score_records=previous_low_scores or None,
    )
    job["synthetic_count"] = len(synth_records)
    return synthetic_ds.enrich_dataset(records_to_run, synth_records)


def _run_reporting_step(
    job: dict,
    eval_results: dict,
    runner_results: dict,
    model_configs: list[dict],
    reporting,
) -> None:
    """Generate Excel and JSON reports; update job dict with paths and aggregated data."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = f"report_{ts}.xlsx"
    json_path = f"report_{ts}.json"
    reporting.run(
        eval_results,
        runner_results=runner_results,
        model_configs=model_configs,
        output_path=report_path,
        json_output_path=json_path,
    )
    job["report_path"] = report_path
    job["json_path"] = json_path
    job["aggregated"] = reporting.aggregate_scores(eval_results, runner_results, model_configs)


def _run_pipeline_job(job_id: str, _status=None) -> None:
    """Esegue la pipeline completa per un job e aggiorna st.session_state.

    Args:
        job_id: ID del job da eseguire.
        _status: optional Streamlit st.status() container for live progress updates.
    """
    from src.evaluation import main as evaluation
    from src.insight_agent import main as insight_agent
    from src.model_selection import main as model_selection
    from src.providers.bedrock import BedrockClient
    from src.reporting import main as reporting
    from src.runner import main as runner_mod
    from src.synthetic_dataset import main as synthetic_ds

    def _step(msg: str) -> None:
        logger.info(msg)
        if _status is not None:
            _status.write(f"⏳ {msg}")

    job = st.session_state.jobs[job_id]
    region = os.environ.get("AWS_REGION", "eu-west-1")

    try:
        job[_K_STATUS] = "running"
        _step("Caricamento modelli e configurazione client…")
        (llm_client, judge_client, judge_model_id,
         model_configs, consensus_judge_ids) = _setup_pipeline_clients(
            job, region, model_selection, BedrockClient,
        )
        _step("Preparazione record da valutare…")
        detected = st.session_state.detected_data
        if detected is None:
            raise ValueError("Nessun dataset rilevato. Esegui prima 'Detect Tasks'.")
        records_to_run = _build_records_to_run(job, detected)
        if not records_to_run:
            raise ValueError(
                "Nessun record trovato per i task "
                f"selezionati: {set(job[_K_SELECTED_TASKS])}"
            )
        if job.get("enrich_synthetic"):
            _step("Generazione dataset sintetico (BN_SYNTHETIC_DATASET)…")
            records_to_run = _enrich_with_synthetic(
                job, job_id, records_to_run, evaluation, synthetic_ds,
                llm_client, model_configs, judge_client, judge_model_id,
                st.session_state.jobs,
            )
        job["num_records"] = len(records_to_run)
        _step(f"Esecuzione inferenza su {len(records_to_run)} record…")
        runner_results = runner_mod.run(records_to_run, model_configs, llm_client)
        job["runner_results"] = runner_results
        _step("Valutazione risultati (con consensus judge per record senza GT)…")
        eval_results = evaluation.run(
            runner_results, judge_client, judge_model_id,
            consensus_judge_model_ids=consensus_judge_ids,
        )
        job["eval_results"] = eval_results
        _step("Generazione report Excel e JSON…")
        _run_reporting_step(job, eval_results, runner_results, model_configs, reporting)
        _step("Generazione insight con Insight Agent…")
        try:
            job["insights"] = insight_agent.run(eval_results, judge_client, judge_model_id)
        except Exception as e:
            logger.warning("Insight agent failed: %s", e)
            job["insights"] = None
        job[_K_STATUS] = _K_COMPLETED
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        if _status is not None:
            _status.update(label="✅ Pipeline completata!", state="complete")

    except Exception as e:
        job[_K_STATUS] = "failed"
        job["error"] = str(e)
        if _status is not None:
            _status.update(label=f"❌ Pipeline fallita: {e}", state="error")
        logger.exception("Pipeline job %s failed", job_id)


# ===================================================================
# PAGE RENDERERS
# ===================================================================

# ---------------------------------------------------------------------------
# 1. Config Job helpers
# ---------------------------------------------------------------------------


def _do_detect_tasks(bucket: str, dataset_key: str) -> None:
    """Run detect-tasks logic and store results in session state."""
    from src.data_processing import main as data_processing

    with st.spinner("Caricamento dataset e rilevamento task\u2026"):
        try:
            all_records = data_processing.load_dataset_from_s3(bucket, dataset_key)
            agent_groups = data_processing.split_by_agent(all_records)
            agent_tasks: dict[str, str] = {}
            for aid, recs in agent_groups.items():
                valid = [r for r in recs if data_processing.validate_record(r)]
                agent_tasks[aid] = data_processing.identify_task(valid if valid else recs)

            st.session_state.detected_data = {
                "all_records": all_records,
                "agent_groups": agent_groups,
                "agent_tasks": agent_tasks,
                _K_DATASET_KEY: dataset_key,
            }
            n_agents = len(agent_groups)
            n_tasks = len(set(agent_tasks.values()))
            st.success(f"Rilevati {n_agents} agent e {n_tasks} task distinti.")
        except Exception as e:
            st.error(f"Errore durante il rilevamento: {e}")


def _load_candidate_models(config_path) -> dict[str, str]:
    """Load model configs and return {display_name: model_id} mapping."""
    try:
        from src.model_selection import main as model_selection

        models = model_selection.load_model_configs(config_path)
        return {m.get("display_name", m[_K_MODEL_ID]): m[_K_MODEL_ID] for m in models}
    except Exception as e:
        st.warning(f"Impossibile caricare modelli da `{config_path}`: {e}")
        return {}


def _render_synth_enrichment_ui(selected_tasks: list[str]) -> tuple[bool, int]:
    """Render synthetic enrichment UI; return (enrich_enabled, records_per_task)."""
    st.subheader("4 \xb7 Arricchimento Dataset Sintetico")
    st.caption(
        "Genera record sintetici aggiuntivi per ampliare "
        "la copertura del dataset prima della pipeline."
    )
    enrich = st.checkbox(
        "\U0001f9ec Abilita generazione sintetica pre-pipeline",
        value=False,
        key="enrich_synth",
    )
    n = 5
    if enrich:
        n = st.slider("Record sintetici per task", min_value=1, max_value=30, value=5,
                      key="synth_n")
        st.info(
            f"Verranno generati ~{n} record per {len(selected_tasks)} task selezionati "
            f"(~{n * len(selected_tasks)} totali) e uniti al dataset originale."
        )
    return enrich, n


def _render_dataset_selector(bucket: str) -> str:
    """Render the dataset selection UI; return the chosen S3 key."""
    st.subheader("1 \xb7 Seleziona Dataset")
    datasets: list[str] = []
    if bucket:
        try:
            datasets = _list_s3_datasets(bucket)
        except Exception as e:
            st.warning(f"Impossibile elencare dataset S3: {e}")
    else:
        st.warning("Variabile `S3_BUCKET` non impostata. Inserisci manualmente la chiave S3.")

    if datasets:
        return st.selectbox("Dataset S3", datasets)
    return st.text_input("Chiave S3 del dataset", value="dataset/dataset.jsonl")


# ---------------------------------------------------------------------------
# 1. Config Job
# ---------------------------------------------------------------------------

def render_config_job() -> None:
    """Sezione di configurazione e avvio pipeline."""
    st.header("⚙️ Configurazione Job")

    bucket = os.environ.get("S3_BUCKET", "")

    # --- Dataset selection ---
    dataset_key = _render_dataset_selector(bucket)

    # --- Detect Tasks ---
    st.subheader("2 \xb7 Rileva Task")
    if st.button("\U0001f50d Detect Tasks", type="primary"):
        _do_detect_tasks(bucket, dataset_key)

    detected = st.session_state.detected_data
    if detected:
        df_agents = pd.DataFrame([
            {
                "Agent ID": aid,
                "Task": detected["agent_tasks"].get(aid, "?"),
                "Records": len(recs),
            }
            for aid, recs in detected["agent_groups"].items()
        ])
        st.dataframe(df_agents, use_container_width=True, hide_index=True)
        available_tasks = sorted({t for t in detected["agent_tasks"].values()})
    else:
        available_tasks = []

    # --- Task selection ---
    st.subheader("3 \xb7 Seleziona Task e Modelli")

    selected_tasks = st.multiselect(
        "Task da valutare",
        available_tasks,
        default=available_tasks if available_tasks else None,
        key="sel_tasks",
    )

    # --- Model selection ---
    config_path = Path(_PROJECT_ROOT) / "configs" / _MODELS_CONFIG_FILENAME
    model_options = _load_candidate_models(config_path)
    selected_display = st.multiselect(
        "Modelli Candidate",
        list(model_options.keys()),
        default=list(model_options.keys()) if model_options else None,
        key="sel_models",
    )
    selected_model_ids = [model_options[d] for d in selected_display]

    # --- Synthetic Dataset Enrichment ---
    enrich_synthetic, synth_records_per_task = _render_synth_enrichment_ui(selected_tasks)

    # --- Launch ---
    st.subheader("5 \xb7 Avvia Pipeline")
    can_launch = bool(detected and selected_tasks and selected_model_ids)

    if st.button("🚀 Avvia Pipeline", type="primary", disabled=not can_launch):
        job_id = str(uuid.uuid4())[:8]
        st.session_state.jobs[job_id] = {
            _K_STATUS: "running",
            _K_DATASET_KEY: dataset_key,
            _K_SELECTED_TASKS: selected_tasks,
            _K_SELECTED_MODELS: selected_model_ids,
            "enrich_synthetic": enrich_synthetic,
            "synth_records_per_task": synth_records_per_task,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runner_results": None,
            "eval_results": None,
            "aggregated": None,
            "report_path": None,
            "json_path": None,
            "insights": None,
            "error": None,
        }
        # BN_ADVANCED_FRONTEND: st.status() provides live expanding progress updates
        with st.status("🚀 Pipeline in esecuzione…", expanded=True) as _pipeline_status:
            _run_pipeline_job(job_id, _pipeline_status)

        job = st.session_state.jobs[job_id]
        if job[_K_STATUS] == _K_COMPLETED:
            st.success(f"✅ Job **{job_id}** completato!")
        else:
            st.error(f"❌ Job **{job_id}** fallito: {job.get('error', 'errore sconosciuto')}")

    if not can_launch and not detected:
        st.info("Esegui prima **Detect Tasks** per rilevare i task disponibili.")


# ---------------------------------------------------------------------------
# 2. Monitoraggio
# ---------------------------------------------------------------------------

def render_monitoraggio() -> None:
    """Mostra la lista di tutti i job con stato."""
    st.header("\U0001f4ca Monitoraggio Job")

    jobs = st.session_state.jobs
    if not jobs:
        st.info("Nessun job avviato. Vai alla sezione **Config Job** per iniziare.")
        return

    # BN_ADVANCED_FRONTEND: auto-polling ≤5 s when there are running/pending jobs.
    # st.rerun() triggers a page re-run; time.sleep ensures at most 5 s between polls.
    _running_statuses = {"pending", "running"}
    has_active = any(j.get(_K_STATUS) in _running_statuses for j in jobs.values())

    col_refresh, col_status = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Aggiorna ora", key="manual_refresh"):
            st.rerun()
    with col_status:
        if has_active:
            st.info("⏳ Job attivi rilevati — aggiornamento automatico ogni 5 secondi.")

    if has_active:
        import time as _time
        _time.sleep(5)
        st.rerun()

    status_icons = {
        "pending": "\U0001f7e1",
        "running": "\U0001f535",
        _K_COMPLETED: "\U0001f7e2",
        "failed": "\U0001f534",
    }

    rows = []
    _default_icon = "\u26aa"
    for jid, j in sorted(jobs.items(), key=lambda x: x[1].get("created_at", ""), reverse=True):
        icon = status_icons.get(j["status"], _default_icon)
        rows.append({
            "Job ID": jid,
            "Dataset": j.get(_K_DATASET_KEY, ""),
            "Tasks": ", ".join(j.get(_K_SELECTED_TASKS, [])),
            "Models": len(j.get(_K_SELECTED_MODELS, [])),
            "Status": f"{icon} {j['status'].upper()}",
            "Creato": j.get("created_at", "")[:19],
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Dettagli espandibili
    for jid, j in jobs.items():
        with st.expander(f"Dettagli job {jid}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Stato", j[_K_STATUS].upper())
            col2.metric("Task", len(j.get(_K_SELECTED_TASKS, [])))
            col3.metric("Modelli", len(j.get(_K_SELECTED_MODELS, [])))

            if j.get("error"):
                st.error(f"Errore: {j['error']}")
            if j.get("num_records"):
                st.write(f"Record processati: **{j['num_records']}**")
            if j.get("completed_at"):
                st.write(f"Completato: {j['completed_at'][:19]}")


# ---------------------------------------------------------------------------
# 3. Risultati
# ---------------------------------------------------------------------------

def _render_per_model_table(per_model: dict) -> None:
    """Render the per-model metrics table."""
    import pandas as pd
    if not per_model:
        return
    df = pd.DataFrame([
        {
            "Modello": mid,
            _LBL_SCORE_MEDIO: info.get(_K_OVERALL_SCORE, 0),
            "Accuracy": info.get("accuracy", 0),
            "Record": info.get("num_records", 0),
            "Corretti": info.get("num_correct", 0),
        }
        for mid, info in per_model.items()
    ]).sort_values(_LBL_SCORE_MEDIO, ascending=False)
    import streamlit as _st
    _st.dataframe(df, use_container_width=True, hide_index=True)


def _render_best_models_section(best: dict) -> None:
    """Render best models (overall + per-task)."""
    import streamlit as _st
    import pandas as pd
    if not best:
        return
    overall = best.get("overall", {})
    if overall.get(_K_MODEL_ID):
        mid = overall[_K_MODEL_ID]
        sc = overall.get(_K_OVERALL_SCORE, "?")
        _st.success(f"🏆 **Miglior modello complessivo**: {mid} (score: {sc})")
    per_task_best = best.get("per_task", {})
    if per_task_best:
        _st.subheader("Miglior Modello per Task")
        df_best = pd.DataFrame([
            {"Task": t, "Modello": info[_K_MODEL_ID], "Score": info[_K_AVG_SCORE]}
            for t, info in sorted(per_task_best.items())
        ])
        _st.dataframe(df_best, use_container_width=True, hide_index=True)


def _render_task_model_breakdown(per_task_model: dict) -> None:
    """Render per task x model breakdown in expanders."""
    import streamlit as _st
    import pandas as pd
    if not per_task_model:
        return
    _st.subheader("Dettaglio per Task × Modello")
    tasks = sorted({v["task"] for v in per_task_model.values()})
    for task in tasks:
        with _st.expander(f"📋 {task}"):
            task_rows = [
                {
                    "Modello": v[_K_MODEL_ID],
                    _LBL_SCORE_MEDIO: v[_K_AVG_SCORE],
                    "Accuracy": v["accuracy"],
                    "Record": v["num_records"],
                }
                for v in per_task_model.values() if v["task"] == task
            ]
            _st.dataframe(
                pd.DataFrame(task_rows).sort_values(_LBL_SCORE_MEDIO, ascending=False),
                use_container_width=True, hide_index=True,
            )


def _render_insight_summary(insights: dict | None) -> None:
    """Render insight summary and recommendation."""
    import streamlit as _st
    if not insights:
        return
    insight_data = insights.get("insights", {})
    if insight_data.get("summary"):
        _st.info(insight_data["summary"])
    if insight_data.get("recommendation"):
        _st.success(f"**Raccomandazione**: {insight_data['recommendation']}")


def render_risultati() -> None:
    """Mostra risultati aggregati e permette download report."""
    st.header(_PAGE_RISULTATI)

    completed = {jid: j for jid, j in st.session_state.jobs.items() if j[_K_STATUS] == _K_COMPLETED}
    if not completed:
        st.info("Nessun job completato. Avvia una pipeline dalla sezione **Config Job**.")
        return

    job_id = st.selectbox(
        "Seleziona job",
        list(completed.keys()),
        format_func=lambda x: f"{x} — {completed[x].get('dataset_key', '')}",
    )
    job = completed[job_id]
    agg = job.get("aggregated")

    if not agg:
        st.warning("Dati aggregati non disponibili per questo job.")
        return

    st.subheader("Metriche per Modello")
    _render_per_model_table(agg.get("per_model", {}))
    _render_best_models_section(agg.get("best_models", {}))
    _render_task_model_breakdown(agg.get("per_task_model", {}))

    insights = job.get("insights")
    if insights:
        st.subheader("💡 Insight")
        _render_insight_summary(insights)

    # --- Download ---
    st.subheader("📥 Download")
    col1, col2 = st.columns(2)

    report_path = job.get("report_path")
    if report_path and Path(report_path).exists():
        with open(report_path, "rb") as f:
            col1.download_button(
                "⬇️ Scarica Report Excel",
                data=f.read(),
                file_name=Path(report_path).name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    json_path = job.get("json_path")
    if json_path and Path(json_path).exists():
        with open(json_path, "rb") as f:
            col2.download_button(
                "⬇️ Scarica JSON",
                data=f.read(),
                file_name=Path(json_path).name,
                mime="application/json",
            )


# ---------------------------------------------------------------------------
# 4. Insight & Confronto (Advanced — 3.10)
# ---------------------------------------------------------------------------

def _render_verdict_tab(agg: dict) -> None:
    """Render Verdict tab content."""
    import streamlit as _st
    import pandas as pd
    per_record = agg.get("per_record", [])
    if not per_record:
        _st.info("Nessun record di valutazione disponibile.")
        return
    df_verdict = pd.DataFrame([
        {
            "test_id": r.get("test_id", ""),
            "task": r.get("task", ""),
            "model": r.get(_K_MODEL_ID, ""),
            "score": r.get("score", 0.0),
            "verdict": "✅ CORRECT" if r.get("correct") else "❌ INCORRECT",
            "output": str(r.get("output_text", ""))[:200],
        }
        for r in per_record
    ])
    col1, col2 = _st.columns(2)
    filter_task = col1.multiselect("Filtra per task", sorted(df_verdict["task"].unique()), key="v_task")
    filter_model = col2.multiselect("Filtra per modello", sorted(df_verdict["model"].unique()), key="v_model")
    if filter_task:
        df_verdict = df_verdict[df_verdict["task"].isin(filter_task)]
    if filter_model:
        df_verdict = df_verdict[df_verdict["model"].isin(filter_model)]
    _st.dataframe(df_verdict, use_container_width=True, hide_index=True, height=500)
    _st.caption(f"{len(df_verdict)} record visualizzati")


def _render_compare_tab(agg: dict) -> None:
    """Render Confronto Modelli tab content."""
    import streamlit as _st
    import pandas as pd
    per_task_model = agg.get("per_task_model", {})
    if not per_task_model:
        _st.info("Dati di confronto non disponibili.")
        return
    rows = [{"Modello": v[_K_MODEL_ID], "Task": v["task"], "Score": v[_K_AVG_SCORE]}
            for v in per_task_model.values()]
    df_pivot = pd.DataFrame(rows).pivot_table(
        index="Modello", columns="Task", values="Score", aggfunc="first",
    )
    _st.subheader("Score per Modello × Task")
    _st.dataframe(df_pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                  use_container_width=True)
    per_model = agg.get("per_model", {})
    if per_model:
        _st.subheader("Score Complessivo per Modello")
        df_bar = pd.DataFrame([
            {"Modello": mid, "Score": info["overall_score"]}
            for mid, info in per_model.items()
        ]).set_index("Modello")
        _st.bar_chart(df_bar)
    if not df_pivot.empty:
        _st.subheader("Score per Task (confronto modelli)")
        _st.bar_chart(df_pivot.T)


def _render_perf_tab(agg: dict) -> None:
    """Render Performance tab content."""
    import streamlit as _st
    operative = agg.get("operative_metrics", {})
    if not operative:
        _st.info(
            "Metriche operative non disponibili. "
            "Assicurati che il runner abbia tracciato latenza e token."
        )
        return
    import pandas as pd
    df_op = pd.DataFrame(list(operative.values()))
    _st.subheader("⏱️ Latenza Media per Modello")
    if "model" in df_op.columns and "avg_latency_ms" in df_op.columns:
        df_lat = df_op.groupby("model")["avg_latency_ms"].mean().sort_values(ascending=False)
        _st.bar_chart(df_lat)
    _st.subheader("💰 Costo Totale per Modello")
    if "total_cost" in df_op.columns:
        _st.bar_chart(df_op.groupby("model")["total_cost"].sum().sort_values(ascending=False))
    _st.subheader("🔤 Token Usage per Modello")
    if "input_tokens" in df_op.columns:
        _st.bar_chart(df_op.groupby("model")[["input_tokens", "output_tokens"]].sum())
    _st.subheader("📋 Tabella Metriche Operative")
    _st.dataframe(df_op, use_container_width=True, hide_index=True)


def render_advanced() -> None:
    """Verdict per record, confronto modelli, grafici performance/latenza/costo."""
    st.header("🔍 Insight & Confronto Modelli")

    completed = {jid: j for jid, j in st.session_state.jobs.items() if j[_K_STATUS] == _K_COMPLETED}
    if not completed:
        st.info("Nessun job completato disponibile.")
        return

    job_id = st.selectbox("Job", list(completed.keys()), key="adv_job",
                          format_func=lambda x: f"{x} — {completed[x].get('dataset_key', '')}")
    job = completed[job_id]
    agg = job.get("aggregated")
    if not agg:
        st.warning("Dati non disponibili.")
        return

    tab_verdict, tab_compare, tab_perf = st.tabs([
        "📝 Verdict per Record",
        "📊 Confronto Modelli",
        "⚡ Performance",
    ])

    with tab_verdict:
        _render_verdict_tab(agg)

    with tab_compare:
        _render_compare_tab(agg)

    with tab_perf:
        _render_perf_tab(agg)


# ---------------------------------------------------------------------------
# 5. Prompt Optimization (3.13)
# ---------------------------------------------------------------------------

def _load_opt_models(config_path) -> tuple[list, dict]:
    """Load models and judge config for prompt optimization."""
    import streamlit as _st
    try:
        from src.model_selection import main as model_selection
        models = model_selection.load_model_configs(config_path)
        judge_cfg = model_selection.load_judge_config(config_path)
        return models, judge_cfg
    except Exception as e:
        _st.warning(f"Impossibile caricare i modelli: {e}")
        return [], {}


def _render_opt_results_section(result: dict) -> None:
    """Render prompt optimization results."""
    import streamlit as _st
    import pandas as pd
    _st.subheader("Risultati Ottimizzazione")
    col1, col2, col3, col4 = _st.columns(4)
    col1.metric("Score Baseline", f"{result.get('baseline_score', 0):.3f}")
    col2.metric("Score Migliore", f"{result.get('best_score', 0):.3f}")
    delta = result.get("best_score", 0) - result.get("baseline_score", 0)
    col3.metric("Miglioramento", f"{delta:+.3f}")
    col4.metric("Score Composito", f"{result.get('composite_score', 0):.4f}")

    weights = result.get("weights", {})
    if weights:
        _st.caption(
            f"Pesi multi-obiettivo: qualità={weights.get('quality', 0):.0%} · "
            f"costo={weights.get('cost', 0):.0%} · latenza={weights.get('latency', 0):.0%}"
        )

    # BN_PROMPT_OPTIMIZATION: statistical significance
    sig = result.get("significance_test", {})
    if sig:
        method = sig.get("method", "none")
        is_sig = sig.get("is_significant", False)
        sig_label = "✅ Significativo" if is_sig else "⚠️ Non significativo"
        p_val = sig.get("p_value")
        mean_diff = sig.get("mean_difference", 0)
        _st.info(
            f"**Test statistico** ({method}): {sig_label} — "
            f"Δ medio={mean_diff:+.4f}"
            + (f" · p={p_val:.4f}" if p_val is not None else "")
        )

    _st.write(f"Iterazioni eseguite: **{result.get('iterations_run', '?')}**")

    # BN_PROMPT_OPTIMIZATION: ablation report
    ablation = result.get("ablation_report", [])
    if ablation:
        with _st.expander("📊 Ablation Report (contributo per iterazione)"):
            df_ab = pd.DataFrame([
                {
                    "Da iter": a["from_iteration"],
                    "A iter": a["to_iteration"],
                    "Score prima": a["score_before"],
                    "Score dopo": a["score_after"],
                    "Delta": f"{a['score_delta']:+.4f}",
                    "Contributo": a["contribution"],
                }
                for a in ablation
            ])
            _st.dataframe(df_ab, use_container_width=True, hide_index=True)

    history = result.get("history", [])
    if history:
        _st.subheader("Storico Iterazioni")
        df_hist = pd.DataFrame([
            {
                "Iterazione": h.get("iteration", 0),
                "Score": h.get("score", 0),
                "Corretti": h.get("num_correct", 0),
                "Baseline": "✅" if h.get("is_baseline") else "",
            }
            for h in history
        ])
        _st.dataframe(df_hist, use_container_width=True, hide_index=True)
        _st.line_chart(df_hist.set_index("Iterazione")["Score"])
    best_prompt = result.get("best_prompt", "")
    if best_prompt:
        _st.subheader("Miglior Prompt Trovato")
        _st.code(best_prompt[:3000], language="text")
    all_variants = result.get("all_variants", [])
    if all_variants:
        with _st.expander(f"Tutte le varianti testate ({len(all_variants)})"):
            for v in all_variants:
                it, vi, sc = v.get("iteration"), v.get("variant_index"), v.get("score", 0)
                _st.write(f"**Iter {it}, Var {vi}** — Score: {sc:.3f}")
                _st.code(v.get("prompt", "")[:500], language="text")
                _st.divider()


def render_prompt_optimization() -> None:
    """Sezione dedicata all'ottimizzazione prompt per agent_id."""
    st.header(_PAGE_PROMPT_OPT)

    detected = st.session_state.detected_data
    if not detected:
        st.info(
            "Esegui prima **Detect Tasks** nella sezione "
            "Config Job per rilevare gli agent disponibili."
        )
        return

    agent_ids = sorted(detected["agent_groups"].keys())
    agent_tasks = detected["agent_tasks"]

    # --- Selezione agent ---
    selected_agent = st.selectbox(
        "Agent ID",
        agent_ids,
        format_func=lambda x: f"{x} ({agent_tasks.get(x, '?')})",
    )

    # --- Selezione modello candidate ---
    config_path = Path(_PROJECT_ROOT) / "configs" / _MODELS_CONFIG_FILENAME
    models, judge_cfg = _load_opt_models(config_path)

    if not models:
        st.warning("Impossibile caricare i modelli. Verifica `configs/models.yaml`.")
        return

    model_options = {m.get("display_name", m[_K_MODEL_ID]): m[_K_MODEL_ID] for m in models}
    selected_model_name = st.selectbox("Modello Candidate", list(model_options.keys()))
    selected_model_id = model_options[selected_model_name]

    # --- Parametri ---
    col1, col2, col3 = st.columns(3)
    max_iterations = col1.slider("Max Iterazioni", 1, 5, 3)
    num_variants = col2.slider("Varianti per Iterazione", 1, 5, 3)
    beam_width = col3.slider(
        "Beam Width", 1, 5, 2,
        help="Numero di candidati mantenuti in parallelo (beam search).",
    )

    # --- Mostra system prompt corrente ---
    agent_records = detected["agent_groups"].get(selected_agent, [])
    current_prompt = ""
    for rec in agent_records:
        for msg in rec.get("input_messages", []):
            if msg.get("role") == "system":
                current_prompt = msg.get("content", "")
                break
        if current_prompt:
            break

    if current_prompt:
        st.subheader("System Prompt Corrente")
        st.code(current_prompt[:2000], language="text")

    # --- Avvio ---
    if st.button("🚀 Avvia Ottimizzazione", type="primary"):
        from src.prompt_optimizer import main as prompt_optimizer
        from src.providers.bedrock import BedrockClient

        region = os.environ.get("AWS_REGION", "eu-west-1")

        with st.spinner("Ottimizzazione in corso… potrebbe richiedere diversi minuti."):
            try:
                llm_client = BedrockClient(region=region)
                judge_client = BedrockClient(region=region)
                judge_model_id = judge_cfg.get(_K_MODEL_ID, "")

                result = prompt_optimizer.run(
                    agent_id=selected_agent,
                    model_id=selected_model_id,
                    base_prompt=current_prompt,
                    records=agent_records,
                    llm_client=llm_client,
                    judge_client=judge_client,
                    judge_model_id=judge_model_id,
                    max_iterations=max_iterations,
                    num_variants=num_variants,
                    beam_width=beam_width,
                )
                st.session_state.prompt_opt_results = result
                st.success("Ottimizzazione completata!")
            except Exception as e:
                st.error(f"Errore durante l'ottimizzazione: {e}")

    # --- Risultati ---
    result = st.session_state.prompt_opt_results
    if result:
        _render_opt_results_section(result)


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    st.sidebar.title("🧪 NRT Pipeline")
    st.sidebar.caption("Non-Regression Testing Dashboard")

    page = st.sidebar.radio(
        "Navigazione",
        [
            "⚙️ Config Job",
            "📊 Monitoraggio",
            _PAGE_RISULTATI,
            "🔍 Insight & Confronto",
            _PAGE_PROMPT_OPT,
        ],
    )

    if page == "⚙️ Config Job":
        render_config_job()
    elif page == "📊 Monitoraggio":
        render_monitoraggio()
    elif page == _PAGE_RISULTATI:
        render_risultati()
    elif page == "🔍 Insight & Confronto":
        render_advanced()
    elif page == _PAGE_PROMPT_OPT:
        render_prompt_optimization()


if __name__ == "__main__":
    main()

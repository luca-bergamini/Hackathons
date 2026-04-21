"""FastAPI backend for NRT Pipeline React frontend.

Exposes the full NRT pipeline as REST endpoints.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

logger = logging.getLogger(__name__)

app = FastAPI(title="NRT Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_state: dict = {
    "jobs": {},
    "detected_data": None,
    "prompt_opt_results": {},
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DetectTasksRequest(BaseModel):
    bucket: str = Field(default="")
    dataset_key: str = Field(default="dataset/dataset.jsonl")


class LaunchJobRequest(BaseModel):
    dataset_key: str
    selected_tasks: list[str]
    selected_models: list[str]
    enrich_synthetic: bool = False
    synth_records_per_task: int = 5


class PromptOptRequest(BaseModel):
    agent_id: str
    model_id: str
    max_iterations: int = 3
    num_variants: int = 3
    beam_width: int = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_region() -> str:
    return os.environ.get("AWS_REGION", "eu-west-1")


def _run_pipeline_job(job_id: str) -> None:
    """Run the full pipeline for a job (in background thread)."""
    from src.evaluation import main as evaluation
    from src.insight_agent import main as insight_agent
    from src.model_selection import main as model_selection
    from src.providers.bedrock import BedrockClient
    from src.reporting import main as reporting
    from src.runner import main as runner_mod
    from src.synthetic_dataset import main as synthetic_ds

    job = _state["jobs"][job_id]
    region = _get_region()

    try:
        job["status"] = "running"

        config_path = Path(_PROJECT_ROOT) / "configs" / "models.yaml"
        llm_client = BedrockClient(region=region)
        models = model_selection.load_model_configs(config_path)
        judge_cfg = model_selection.load_judge_config(config_path)
        judge_model_id = judge_cfg["model_id"]
        judge_client = BedrockClient(region=region)

        selected_ids = set(job["selected_models"])
        model_configs = [m for m in models if m["model_id"] in selected_ids]
        if not model_configs:
            model_configs = models

        detected = _state["detected_data"]
        if detected is None:
            raise ValueError("No dataset detected. Run detect-tasks first.")

        agent_groups = detected["agent_groups"]
        selected_tasks = set(job["selected_tasks"])
        records_to_run: list[dict] = []
        for agent_id, recs in agent_groups.items():
            task = detected["agent_tasks"].get(agent_id, "unknown")
            if task in selected_tasks:
                for r in recs:
                    r["task"] = task
                    records_to_run.append(r)

        if not records_to_run:
            raise ValueError(f"No records for selected tasks: {selected_tasks}")

        # Synthetic enrichment
        if job.get("enrich_synthetic"):
            n_per_task = job.get("synth_records_per_task", 5)
            seed_by_task: dict[str, list[dict]] = {}
            for r in records_to_run:
                t = r.get("task", "unknown")
                seed_by_task.setdefault(t, []).append(r)
            synth_records = synthetic_ds.run(
                tasks=list(seed_by_task.keys()),
                llm_client=llm_client,
                model_id=model_configs[0]["model_id"],
                records_per_task=n_per_task,
                output_path="",
                seed_records_by_task=seed_by_task,
            )
            records_to_run = synthetic_ds.enrich_dataset(records_to_run, synth_records)
            job["synthetic_count"] = len(synth_records)

        job["num_records"] = len(records_to_run)

        # Runner
        runner_results = runner_mod.run(records_to_run, model_configs, llm_client)
        job["runner_results"] = runner_results

        # Evaluation
        eval_results = evaluation.run(runner_results, judge_client, judge_model_id)
        job["eval_results"] = eval_results

        # Reporting
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

        # Insight Agent
        try:
            insights = insight_agent.run(eval_results, judge_client, judge_model_id)
            job["insights"] = insights
        except Exception as e:
            logger.warning("Insight agent failed: %s", e)
            job["insights"] = None

        job["status"] = "completed"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.exception("Pipeline job %s failed", job_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def get_models():
    """List available candidate models and judge config."""
    from src.model_selection import main as model_selection

    config_path = Path(_PROJECT_ROOT) / "configs" / "models.yaml"
    try:
        models = model_selection.load_model_configs(config_path)
        judge_cfg = model_selection.load_judge_config(config_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"models": models, "judge": judge_cfg}


@app.get("/api/datasets")
def list_datasets():
    """List .jsonl datasets in S3 bucket."""
    import boto3

    bucket = os.environ.get("S3_BUCKET", "")
    if not bucket:
        return {"datasets": [], "bucket": ""}

    region = _get_region()
    s3 = boto3.client("s3", region_name=region)
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="dataset/"):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".jsonl"):
                keys.append(k)

    return {"datasets": keys, "bucket": bucket}


@app.post("/api/detect-tasks")
def detect_tasks(req: DetectTasksRequest):
    """Load dataset from S3 and detect tasks per agent."""
    from src.data_processing import main as data_processing

    bucket = req.bucket or os.environ.get("S3_BUCKET", "")
    try:
        all_records = data_processing.load_dataset_from_s3(bucket, req.dataset_key)
        agent_groups = data_processing.split_by_agent(all_records)
        agent_tasks: dict[str, str] = {}
        for aid, recs in agent_groups.items():
            valid = [r for r in recs if data_processing.validate_record(r)]
            agent_tasks[aid] = data_processing.identify_task(valid if valid else recs)

        _state["detected_data"] = {
            "all_records": all_records,
            "agent_groups": agent_groups,
            "agent_tasks": agent_tasks,
            "dataset_key": req.dataset_key,
        }

        agents_info = [
            {
                "agent_id": aid,
                "task": agent_tasks.get(aid, "?"),
                "num_records": len(recs),
            }
            for aid, recs in agent_groups.items()
        ]
        available_tasks = sorted({t for t in agent_tasks.values()})

        return {
            "agents": agents_info,
            "available_tasks": available_tasks,
            "total_records": len(all_records),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detected")
def get_detected():
    """Return currently detected data if available."""
    detected = _state["detected_data"]
    if not detected:
        return {"detected": False}

    agents_info = [
        {
            "agent_id": aid,
            "task": detected["agent_tasks"].get(aid, "?"),
            "num_records": len(recs),
        }
        for aid, recs in detected["agent_groups"].items()
    ]
    available_tasks = sorted({t for t in detected["agent_tasks"].values()})

    return {
        "detected": True,
        "agents": agents_info,
        "available_tasks": available_tasks,
        "dataset_key": detected.get("dataset_key", ""),
        "total_records": len(detected.get("all_records", [])),
    }


@app.post("/api/jobs")
def create_job(req: LaunchJobRequest):
    """Create and start a pipeline job (async)."""
    if not _state["detected_data"]:
        raise HTTPException(status_code=400, detail="Run detect-tasks first.")

    job_id = str(uuid.uuid4())[:8]
    _state["jobs"][job_id] = {
        "status": "pending",
        "dataset_key": req.dataset_key,
        "selected_tasks": req.selected_tasks,
        "selected_models": req.selected_models,
        "enrich_synthetic": req.enrich_synthetic,
        "synth_records_per_task": req.synth_records_per_task,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner_results": None,
        "eval_results": None,
        "aggregated": None,
        "report_path": None,
        "json_path": None,
        "insights": None,
        "error": None,
    }

    thread = Thread(target=_run_pipeline_job, args=(job_id,), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "pending"}


@app.get("/api/jobs")
def list_jobs():
    """List all jobs with status."""
    jobs_list = []
    for jid, j in sorted(
        _state["jobs"].items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True,
    ):
        jobs_list.append({
            "job_id": jid,
            "status": j["status"],
            "dataset_key": j.get("dataset_key", ""),
            "selected_tasks": j.get("selected_tasks", []),
            "selected_models": j.get("selected_models", []),
            "num_records": j.get("num_records"),
            "created_at": j.get("created_at", ""),
            "completed_at": j.get("completed_at"),
            "error": j.get("error"),
            "synthetic_count": j.get("synthetic_count"),
        })
    return {"jobs": jobs_list}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """Get full job details including results."""
    if job_id not in _state["jobs"]:
        raise HTTPException(status_code=404, detail="Job not found")

    j = _state["jobs"][job_id]
    agg = j.get("aggregated")
    insights = j.get("insights")

    result = {
        "job_id": job_id,
        "status": j["status"],
        "dataset_key": j.get("dataset_key", ""),
        "selected_tasks": j.get("selected_tasks", []),
        "selected_models": j.get("selected_models", []),
        "num_records": j.get("num_records"),
        "created_at": j.get("created_at", ""),
        "completed_at": j.get("completed_at"),
        "error": j.get("error"),
        "synthetic_count": j.get("synthetic_count"),
        "has_report": bool(j.get("report_path") and Path(j["report_path"]).exists()),
        "has_json": bool(j.get("json_path") and Path(j["json_path"]).exists()),
        "aggregated": _sanitize_aggregated(agg) if agg else None,
        "insights": insights,
    }
    return result


@app.get("/api/jobs/{job_id}/report")
def download_report(job_id: str, fmt: str = "xlsx"):
    """Download report file (xlsx or json)."""
    if job_id not in _state["jobs"]:
        raise HTTPException(status_code=404, detail="Job not found")

    j = _state["jobs"][job_id]

    if fmt == "json":
        path = j.get("json_path")
        media = "application/json"
    else:
        path = j.get("report_path")
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(path, media_type=media, filename=Path(path).name)


@app.post("/api/prompt-optimize")
def prompt_optimize(req: PromptOptRequest):
    """Run prompt optimization for an agent."""
    from src.model_selection import main as model_selection
    from src.prompt_optimizer import main as prompt_optimizer
    from src.providers.bedrock import BedrockClient

    detected = _state["detected_data"]
    if not detected:
        raise HTTPException(status_code=400, detail="Run detect-tasks first.")

    agent_records = detected["agent_groups"].get(req.agent_id, [])
    if not agent_records:
        raise HTTPException(status_code=404, detail=f"Agent {req.agent_id} not found")

    # Extract current system prompt
    current_prompt = ""
    for rec in agent_records:
        for msg in rec.get("input_messages", []):
            if msg.get("role") == "system":
                current_prompt = msg.get("content", "")
                break
        if current_prompt:
            break

    region = _get_region()
    config_path = Path(_PROJECT_ROOT) / "configs" / "models.yaml"

    try:
        judge_cfg = model_selection.load_judge_config(config_path)
        llm_client = BedrockClient(region=region)
        judge_client = BedrockClient(region=region)

        result = prompt_optimizer.run(
            agent_id=req.agent_id,
            model_id=req.model_id,
            base_prompt=current_prompt,
            records=agent_records,
            llm_client=llm_client,
            judge_client=judge_client,
            judge_model_id=judge_cfg.get("model_id", ""),
            max_iterations=req.max_iterations,
            num_variants=req.num_variants,
            beam_width=req.beam_width,
        )
        _state["prompt_opt_results"][req.agent_id] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompt-optimize/{agent_id}")
def get_prompt_opt_result(agent_id: str):
    """Get cached prompt optimization result."""
    result = _state["prompt_opt_results"].get(agent_id)
    if not result:
        raise HTTPException(status_code=404, detail="No optimization result found")
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_aggregated(agg: dict) -> dict:
    """Ensure aggregated data is JSON-serializable."""
    import json

    try:
        json.dumps(agg)
        return agg
    except (TypeError, ValueError):
        # Fallback: convert via string
        return json.loads(json.dumps(agg, default=str))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

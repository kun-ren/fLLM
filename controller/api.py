"""
FastAPI backend — REST API + embedded Gradio web UI.
Run with: python -m controller.api
Or: python -m controller.app  (launches GUI directly)
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.config_manager import get_config_manager, ConfigManager
from controller.experiment_store import get_experiment_store
from controller.schema import GROUPS, SCHEMA, HyperParam, ParamMode
from controller.train_worker import get_train_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Pydantic models ───────────────────────────────────────────────────────────

class HyperParamUpdate(BaseModel):
    name: str
    value: Optional[float | int | bool | str] = None
    mode: Optional[str] = None  # "single" or "range"
    min_val: Optional[float] = None
    max_val: Optional[float] = None

class RunRequest(BaseModel):
    run_type: str  # "train" | "test" | "backtest"
    experiment_id: Optional[str] = None

class ExperimentCreate(BaseModel):
    name: str
    config_name: Optional[str] = None
    notes: Optional[str] = ""

class ConfigSave(BaseModel):
    name: str

class ConfigLoad(BaseModel):
    name: str

# ── FastAPI app ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="fLLM Control Panel",
        description="Control panel for the financial ML training pipeline",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # ── Config endpoints ─────────────────────────────────────────────────────

    @app.get("/api/config/schema")
    def get_schema():
        """Return the hyperparameter schema."""
        by_group = {g: [] for g in GROUPS}
        for name, p in get_config_manager()._params.items():
            by_group[p.group].append(p.to_dict())
        return by_group

    @app.get("/api/config/active")
    def get_active_config():
        """Return current resolved values."""
        cm = get_config_manager()
        resolved = cm.resolve_all()
        return {
            name: {
                "value": resolved[name],
                "mode": cm.get(name).mode.value,
                "min_val": cm.get(name).min_val,
                "max_val": cm.get(name).max_val,
            }
            for name in cm._params
        }

    @app.post("/api/config/update")
    def update_param(update: HyperParamUpdate):
        cm = get_config_manager()
        mode = ParamMode(update.mode) if update.mode else ParamMode.SINGLE
        cm.set(
            update.name,
            value=update.value,
            mode=mode,
            min_val=update.min_val,
            max_val=update.max_val,
        )
        return {"ok": True, "name": update.name}

    @app.post("/api/config/save")
    def save_config(cfg: ConfigSave):
        path = get_config_manager().save_config(cfg.name)
        return {"ok": True, "path": str(path)}

    @app.post("/api/config/load")
    def load_config(cfg: ConfigLoad):
        path = get_config_manager().load_config(cfg.name)
        return {"ok": True, "path": str(path)}

    @app.get("/api/config/list")
    def list_configs():
        return get_config_manager().list_configs()

    @app.get("/api/config/summary")
    def config_summary():
        return get_config_manager().summary()

    # ── Experiment endpoints ──────────────────────────────────────────────────

    @app.post("/api/experiments")
    def create_experiment(req: ExperimentCreate):
        store = get_experiment_store()
        exp_id = store.create_experiment(req.name, req.config_name, req.notes or "")
        return {"id": exp_id}

    @app.get("/api/experiments")
    def list_experiments():
        return get_experiment_store().list_experiments()

    @app.get("/api/experiments/{exp_id}")
    def get_experiment(exp_id: str):
        exp = get_experiment_store().get_experiment(exp_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")
        return exp

    @app.get("/api/experiments/{exp_id}/runs")
    def get_experiment_runs(exp_id: str):
        return get_experiment_store().list_runs(experiment_id=exp_id)

    # ── Run endpoints ────────────────────────────────────────────────────────

    @app.post("/api/runs/start")
    def start_run(req: RunRequest):
        worker = get_train_worker()
        if worker.is_running():
            raise HTTPException(409, "A job is already running — stop it first")
        worker.start(req.run_type, req.experiment_id)
        return {"run_id": worker.run_id, "status": "started"}

    @app.post("/api/runs/stop")
    def stop_run():
        worker = get_train_worker()
        worker.stop()
        return {"status": "stopped"}

    @app.post("/api/runs/pause")
    def pause_run():
        worker = get_train_worker()
        worker.pause()
        return {"status": worker.status}

    @app.post("/api/runs/resume")
    def resume_run():
        worker = get_train_worker()
        worker.resume()
        return {"status": worker.status}

    @app.get("/api/runs/status")
    def run_status():
        worker = get_train_worker()
        return {
            "status": worker.status,
            "run_id": worker.run_id,
            "run_type": worker.run_type,
            "progress": worker.progress,
            "epoch": worker.epoch,
            "total_epochs": worker.total_epochs,
            "current_loss": worker.current_loss,
            "logs": worker.logs[-100:],  # last 100 lines
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        run = get_experiment_store().get_run(run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        return run

    @app.get("/api/runs/{run_id}/logs")
    def get_run_logs(run_id: str):
        store = get_experiment_store()
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        return {"logs": run.get("log_text", "").splitlines()}

    @app.get("/api/runs/{run_id}/train_logs")
    def get_train_epoch_logs(run_id: str):
        return get_experiment_store().get_train_logs(run_id)

    @app.get("/api/runs")
    def list_runs(run_type: Optional[str] = None, limit: int = 50):
        return get_experiment_store().list_runs(run_type=run_type, limit=limit)

    @app.get("/api/runs/latest/{run_type}")
    def get_latest_run(run_type: str):
        run = get_experiment_store().get_latest_run(run_type)
        if not run:
            raise HTTPException(404, f"No {run_type} run found")
        return run

    @app.get("/api/summary")
    def summary():
        return get_experiment_store().get_summary()

    # ── GCS endpoints ────────────────────────────────────────────────────────

    class GCSDownloadRequest(BaseModel):
        bucket_name: Optional[str] = None
        blob_prefix: Optional[str] = None
        destination_dir: Optional[str] = None

    @app.post("/api/gcs/download")
    def download_gcs_data(req: GCSDownloadRequest, background_tasks: BackgroundTasks):
        cm = get_config_manager()
        bucket = req.bucket_name or cm.get("gcs_bucket_name").value
        prefix = req.blob_prefix or cm.get("gcs_blob_prefix").value
        dest = req.destination_dir or cm.get("gcs_destination_dir").value

        from config.googleCloud import get_latest_blob_path, download_blob
        blob_path = get_latest_blob_path(bucket, prefix)
        if not blob_path:
            raise HTTPException(404, f"No blobs found under {prefix}")

        filename = blob_path.split("/")[-1]
        dest_path = str(Path(dest) / filename)
        Path(dest).mkdir(parents=True, exist_ok=True)
        background_tasks.add_task(download_blob, bucket, blob_path, dest_path)
        return {"status": "downloading", "blob": blob_path, "destination": dest_path}

    # ── Backtest detail endpoints ────────────────────────────────────────────

    @app.get("/api/runs/{run_id}/trades")
    def get_backtest_trades(run_id: str):
        return get_experiment_store().get_backtest_trades(run_id)

    @app.get("/api/runs/{run_id}/threshold_sweep")
    def get_threshold_sweep_data(run_id: str):
        return get_experiment_store().get_threshold_sweep(run_id)

    return app


def run_server(host: str = "0.0.0.0", port: int = 7860, **kwargs):
    """Run the FastAPI + Gradio server."""
    app = create_app()

    import gradio as gr
    from controller.app import build_ui

    # Build the Gradio UI and mount it under /ui
    ui = build_ui()
    app = gr.mount_gradio_app(app, ui, path="/ui")

    uvicorn.run(app, host=host, port=port, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)

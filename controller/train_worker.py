"""
Background training worker.
Runs train / test / backtest in a background thread with start / stop / pause.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from controller.config_manager import get_config_manager
from controller.experiment_store import get_experiment_store
from data_processing.dataset import OHLCDataset, preprocess_dataframe
from models.crossformer import CrossformerEncoderTimeStep, EmbeddingHead, train_crossformer_rl
from models.pooling import POOLING_REGISTRY
from models.simple_reversal_loss import SimpleReversalLoss

logger = logging.getLogger(__name__)


class TrainWorker:
    """
    Manages a background training/test/backtest job.
    Thread-safe start/stop/pause interface.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused by default
        self._lock = threading.Lock()
        self._status = "idle"  # idle | running | paused | stopped | finished | error
        self._run_id: Optional[str] = None
        self._run_type: Optional[str] = None
        self._progress_callbacks: list[Callable] = []
        self._result: Optional[dict] = None
        self._progress: float = 0.0
        self._epoch: int = 0
        self._total_epochs: int = 0
        self._current_loss: float = 0.0
        self._log_lines: list[str] = []
        self._start_time: Optional[float] = None

    # -- Status ----------------------------------------------------------------

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def current_loss(self) -> float:
        return self._current_loss

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def total_epochs(self) -> int:
        return self._total_epochs

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def run_type(self) -> Optional[str]:
        return self._run_type

    @property
    def logs(self) -> list[str]:
        return list(self._log_lines)

    @property
    def result(self) -> Optional[dict]:
        return self._result

    def is_running(self) -> bool:
        return self.status == "running"

    # -- Callbacks -------------------------------------------------------------

    def add_progress_callback(self, cb: Callable):
        self._progress_callbacks.append(cb)

    def emit_progress(self):
        for cb in self._progress_callbacks:
            try:
                cb(self)
            except Exception:
                pass

    # -- Control ---------------------------------------------------------------

    def start(self, run_type: str, experiment_id: Optional[str] = None):
        if self._thread and self._thread.is_alive():
            logger.warning("Worker already running -- call stop first")
            return
        with self._lock:
            self._stop_event.clear()
            self._pause_event.set()
            self._status = "running"
            self._progress = 0.0
            self._epoch = 0
            self._current_loss = 0.0
            self._log_lines.clear()
            self._result = None
            self._run_type = run_type
        self._thread = threading.Thread(
            target=self._run_job,
            args=(run_type, experiment_id),
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Started {run_type} job")

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()
        logger.info("Stop requested")

    def pause(self):
        self._pause_event.clear()
        with self._lock:
            self._status = "paused"
        logger.info("Paused")

    def resume(self):
        self._pause_event.set()
        with self._lock:
            self._status = "running"
        logger.info("Resumed")

    def wait_while_paused(self):
        self._pause_event.wait()

    # -- Core job --------------------------------------------------------------

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        logger.info(msg)
        self.emit_progress()

    def _emit_progress(self):
        """Internal method to emit progress updates."""
        self.emit_progress()

    def _run_job(self, run_type: str, experiment_id: Optional[str]):
        self._start_time = time.time()
        store = get_experiment_store()
        cm = get_config_manager()
        self._run_id = store.create_run(run_type, experiment_id, cm.resolve_all())
        try:
            if run_type == "train":
                self._job_train()
            elif run_type == "test":
                self._job_test(cm, store)
            elif run_type == "backtest":
                self._job_backtest(cm, store)
            else:
                raise ValueError(f"Unknown run_type: {run_type}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log(f"ERROR: {e}\n{tb}")
            with self._lock:
                self._status = "error"
            store.update_run(self._run_id, status="error", log_text="\n".join(self._log_lines))
        else:
            duration = time.time() - self._start_time
            with self._lock:
                self._status = "finished"
            store.update_run(
                self._run_id,
                status="finished",
                finished_at=datetime.now().isoformat(),
                duration_sec=duration,
                log_text="\n".join(self._log_lines),
            )
            self._log(f"Finished in {duration:.1f}s")
            self._result = store.get_run(self._run_id)

    # -- Train job -------------------------------------------------------------

    def _job_train(self):
        self._log("Loading config...")
        from models.crossformer import train_crossformer_rl

        store = get_experiment_store()

        for progress_update in train_crossformer_rl(run_id=self._run_id):
            if self._stop_event.is_set():
                self._log("Training stopped by user")
                break

            self.wait_while_paused()

            # Extract progress info
            status = progress_update.get("status", "training")
            epoch = progress_update.get("epoch", 0)
            loss = progress_update.get("loss", 0.0)
            progress = progress_update.get("progress", 0.0)
            checkpoint_path = progress_update.get("checkpoint_path")

            # Update worker state
            if progress_update.get("total_epochs"):
                self._total_epochs = progress_update["total_epochs"]

            self._epoch = epoch
            self._current_loss = loss
            self._progress = progress

            # Log progress
            if progress_update.get("log"):
                self._log(progress_update["log"])

            # Update store with current progress
            if status == "training" and checkpoint_path:
                store.update_run(
                    self._run_id,
                    train_loss=loss,
                    best_loss=progress_update.get("best_loss", loss),
                    final_epoch=epoch,
                    checkpoint_path=checkpoint_path,
                )

            # Handle completion
            if status == "completed":
                best_loss = progress_update.get("best_loss", loss)
                best_epoch = progress_update.get("best_epoch", epoch)
                total_epochs = progress_update.get("total_epochs", epoch)

                store.update_run(
                    self._run_id,
                    train_loss=loss,
                    best_loss=best_loss,
                    epochs_trained=best_epoch,
                    checkpoint_path=checkpoint_path,
                )

                self._log(f"Training completed: {best_epoch}/{total_epochs} epochs, best loss: {best_loss:.6f}")

            self._emit_progress()


    # -- Test job --------------------------------------------------------------

    def _job_test(self, cm, store):
        self._log("Test job: loading config...")
        resolved = cm.resolve_all()
        device = resolved["device"] or "cuda"

        from strategies.delta_reverse import backtesting, threshold_sweep

        test_path = resolved.get("test_dataset_path", resolved.get("train_dataset_path"))
        self._log(f"Loading test data from {test_path}...")

        # Temporarily set train_dataset_path so preprocess reads the test file
        original_path = cm.get("train_dataset_path").value
        cm.set("train_dataset_path", value=test_path)
        data, close_col = preprocess_dataframe(device=device)
        cm.set("train_dataset_path", value=original_path)

        dataset = OHLCDataset(data, close_col, device=device)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        confidences, references = self._run_inference(store, resolved, dataset, loader, device)

        self._log(f"Test on {len(confidences)} samples")
        commission_rate = float(resolved.get("commission_rate", 0.0))

        profits = backtesting(confidences, references, commission_rate=commission_rate)
        sweep_results = threshold_sweep(confidences, references, commission_rate=commission_rate)

        self._save_backtest_results(store, profits, sweep_results)

    # -- Backtest job ----------------------------------------------------------

    def _job_backtest(self, cm, store):
        self._log("Backtest job: loading config...")
        resolved = cm.resolve_all()
        device = resolved["device"] or "cuda"

        from strategies.delta_reverse import backtesting, threshold_sweep

        # Use test_dataset_path for backtesting
        bt_path = resolved.get("test_dataset_path", resolved.get("train_dataset_path"))
        self._log(f"Loading backtest data from {bt_path}...")

        original_path = cm.get("train_dataset_path").value
        cm.set("train_dataset_path", value=bt_path)
        data, close_col = preprocess_dataframe(device=device)
        cm.set("train_dataset_path", value=original_path)

        dataset = OHLCDataset(data, close_col, device=device)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        confidences, references = self._run_inference(store, resolved, dataset, loader, device)

        self._log(f"Backtest on {len(confidences)} samples")
        commission_rate = float(resolved.get("commission_rate", 0.0))

        profits = backtesting(confidences, references, commission_rate=commission_rate)
        sweep_results = threshold_sweep(confidences, references, commission_rate=commission_rate)

        self._save_backtest_results(store, profits, sweep_results)

    # -- Shared helpers --------------------------------------------------------

    def _run_inference(self, store, resolved, dataset, loader, device):
        """Load checkpoint and run inference, returning (confidences, references) as numpy."""
        latest = store.get_latest_run("train")
        if latest and latest.get("checkpoint_path"):
            ckpt_path = latest["checkpoint_path"]
            self._log(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            hp = ckpt.get("hyperparams", {})
            C = dataset[0][0].shape[-1]
            encoder = CrossformerEncoderTimeStep(
                input_dim=C,
                d_model=int(hp.get("d_model", 64)),
                n_heads=int(hp.get("n_heads", 4)),
                n_layers=int(hp.get("n_layers", 3)),
                dim_feedforward=int(hp.get("dim_feedforward", 128)),
            ).to(device)
            head = EmbeddingHead(
                d_model=int(hp.get("d_model", 64)),
                hidden_dim=int(hp.get("hidden_dim", 128)),
                pooling=hp.get("pooling", "AttentionPooling"),
            ).to(device)
            encoder.load_state_dict(ckpt["encoder"])
            head.load_state_dict(ckpt["head"])
            encoder.eval()
            head.eval()

            all_conf, all_ref = [], []
            with torch.no_grad():
                for batch_data, reference_k in loader:
                    batch_data = batch_data.to(device)
                    emb = encoder(batch_data)
                    out = head(emb)
                    all_conf.append(out.cpu())
                    all_ref.append(reference_k)

            confidences = torch.cat(all_conf, dim=0).numpy().flatten()
            references = torch.cat(all_ref, dim=0).numpy()
            # Use last look-ahead slice
            if references.ndim == 3:
                references = references[:, -1, :]
        else:
            self._log("No checkpoint found -- using random confidence for demo")
            confidences = (torch.rand(len(dataset)) * 2 - 1).numpy()
            refs = []
            for i in range(len(dataset)):
                refs.append(dataset[i][1])
            references = torch.stack(refs).numpy()
            if references.ndim == 3:
                references = references[:, -1, :]

        return confidences, references

    def _save_backtest_results(self, store, profits, sweep_results):
        """Save trades, sweep, and update run metrics."""
        if profits:
            store.save_backtest_trades(self._run_id, profits)
            avg_profit = sum(p[3] for p in profits) / len(profits)
            num_trades = len(profits)
            winning = sum(1 for p in profits if p[3] > 0)
            win_rate = winning / num_trades
            self._log(f"Results: avg_net_profit={avg_profit:.2f}, "
                      f"trades={num_trades}, win_rate={win_rate:.2%}")
            store.update_run(
                self._run_id,
                avg_profit=avg_profit,
                num_trades=num_trades,
                win_rate=win_rate,
            )
        else:
            self._log("No trades executed")
            store.update_run(self._run_id, num_trades=0)

        if sweep_results:
            store.save_threshold_sweep(self._run_id, sweep_results)
            self._log(f"Threshold sweep: {len(sweep_results)} thresholds evaluated")

    @progress.setter
    def progress(self, value):
        self._progress = value


# -- Global singleton ----------------------------------------------------------
_worker: Optional[TrainWorker] = None


def get_train_worker() -> TrainWorker:
    global _worker
    if _worker is None:
        _worker = TrainWorker()
    return _worker

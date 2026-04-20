"""
Experiment & run history store backed by SQLite.
Records train / test / backtest runs with their hyperparameters and metrics.
"""
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from controller.schema import GROUPS

logger = logging.getLogger(__name__)

DB_PATH = Path("controller/history.db")


def _row_to_dict(row: tuple, cols: list) -> dict:
    return dict(zip(cols, row))


class ExperimentStore:
    """SQLite-backed store for experiment and run history."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    config_name TEXT,
                    created_at  TEXT NOT NULL,
                    status      TEXT DEFAULT 'created',
                    notes       TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id             TEXT PRIMARY KEY,
                    experiment_id  TEXT,
                    run_type       TEXT NOT NULL,
                    created_at     TEXT NOT NULL,
                    finished_at    TEXT,
                    status         TEXT DEFAULT 'running',
                    duration_sec   REAL,

                    -- Hyperparameters (JSON)
                    hyperparams    TEXT,

                    -- Training metrics
                    train_loss     REAL,
                    val_loss       REAL,
                    best_loss      REAL,
                    epochs_trained INTEGER,
                    final_epoch    INTEGER,
                    learning_rate  REAL,
                    batch_size     INTEGER,

                    -- Backtest results
                    avg_profit     REAL,
                    num_trades     INTEGER,
                    win_rate       REAL,
                    profit_factor  REAL,
                    max_drawdown   REAL,

                    -- Model
                    checkpoint_path TEXT,
                    log_text       TEXT,

                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS train_logs (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id    TEXT NOT NULL,
                    epoch     INTEGER,
                    step      INTEGER,
                    loss      REAL,
                    timestamp TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_experiment
                    ON runs(experiment_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_type
                    ON runs(run_type)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT NOT NULL,
                    trade_index  INTEGER,
                    exit_step    INTEGER,
                    gross_profit REAL,
                    net_profit   REAL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threshold_sweeps (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id             TEXT NOT NULL,
                    threshold          REAL,
                    total_gross_profit REAL,
                    total_net_profit   REAL,
                    win_rate           REAL,
                    num_trades         INTEGER,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)
            conn.commit()

    # ── Experiments ───────────────────────────────────────────────────────────

    def create_experiment(self, name: str, config_name: Optional[str] = None,
                          notes: str = "") -> str:
        exp_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO experiments (id, name, config_name, created_at, notes) "
                "VALUES (?, ?, ?, ?, ?)",
                (exp_id, name, config_name, now, notes),
            )
            conn.commit()
        logger.info(f"Created experiment {exp_id}: {name}")
        return exp_id

    def update_experiment_status(self, exp_id: str, status: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE experiments SET status = ? WHERE id = ?",
                (status, exp_id),
            )
            conn.commit()

    def list_experiments(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_experiment(self, exp_id: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (exp_id,)
            ).fetchone()
        return dict(row) if row else None

    # ── Runs ─────────────────────────────────────────────────────────────────

    def create_run(
        self,
        run_type: str,          # "train" | "test" | "backtest"
        experiment_id: Optional[str] = None,
        hyperparams: Optional[dict] = None,
    ) -> str:
        run_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        hp_json = json.dumps(hyperparams, default=str) if hyperparams else "{}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO runs "
                "(id, experiment_id, run_type, created_at, hyperparams, status) "
                "VALUES (?, ?, ?, ?, ?, 'running')",
                (run_id, experiment_id, run_type, now, hp_json),
            )
            conn.commit()
        logger.info(f"Created {run_type} run {run_id}")
        return run_id

    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        finished_at: Optional[str] = None,
        duration_sec: Optional[float] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        best_loss: Optional[float] = None,
        epochs_trained: Optional[int] = None,
        final_epoch: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        avg_profit: Optional[float] = None,
        num_trades: Optional[int] = None,
        win_rate: Optional[float] = None,
        profit_factor: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
        log_text: Optional[str] = None,
    ):
        fields = []
        values = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if finished_at is not None:
            fields.append("finished_at = ?")
            values.append(finished_at)
        if duration_sec is not None:
            fields.append("duration_sec = ?")
            values.append(duration_sec)
        if train_loss is not None:
            fields.append("train_loss = ?")
            values.append(train_loss)
        if val_loss is not None:
            fields.append("val_loss = ?")
            values.append(val_loss)
        if best_loss is not None:
            fields.append("best_loss = ?")
            values.append(best_loss)
        if epochs_trained is not None:
            fields.append("epochs_trained = ?")
            values.append(epochs_trained)
        if final_epoch is not None:
            fields.append("final_epoch = ?")
            values.append(final_epoch)
        if learning_rate is not None:
            fields.append("learning_rate = ?")
            values.append(learning_rate)
        if batch_size is not None:
            fields.append("batch_size = ?")
            values.append(batch_size)
        if avg_profit is not None:
            fields.append("avg_profit = ?")
            values.append(avg_profit)
        if num_trades is not None:
            fields.append("num_trades = ?")
            values.append(num_trades)
        if win_rate is not None:
            fields.append("win_rate = ?")
            values.append(win_rate)
        if profit_factor is not None:
            fields.append("profit_factor = ?")
            values.append(profit_factor)
        if max_drawdown is not None:
            fields.append("max_drawdown = ?")
            values.append(max_drawdown)
        if checkpoint_path is not None:
            fields.append("checkpoint_path = ?")
            values.append(checkpoint_path)
        if log_text is not None:
            fields.append("log_text = ?")
            values.append(log_text)
        if not fields:
            return
        values.append(run_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE runs SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            conn.commit()

    def append_train_log(self, run_id: str, epoch: int, step: int, loss: float):
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO train_logs (run_id, epoch, step, loss, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, epoch, step, loss, now),
            )
            conn.commit()

    def get_train_logs(self, run_id: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM train_logs WHERE run_id = ? ORDER BY timestamp",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_runs(self, run_type: Optional[str] = None,
                  experiment_id: Optional[str] = None,
                  limit: int = 50) -> list[dict]:
        conditions = []
        params = []
        if run_type:
            conditions.append("run_type = ?")
            params.append(run_type)
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)
        where = " AND ".join(conditions) if conditions else "1=1"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM runs WHERE {where} ORDER BY created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_latest_run(self, run_type: Optional[str] = None) -> Optional[dict]:
        where = f"WHERE run_type = '{run_type}'" if run_type else ""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def delete_run(self, run_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.execute("DELETE FROM train_logs WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM backtest_trades WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM threshold_sweeps WHERE run_id = ?", (run_id,))
            conn.commit()
        return cur.rowcount > 0

    # ── Backtest trades ──────────────────────────────────────────────────────

    def save_backtest_trades(self, run_id: str, trades: list):
        """Save individual trade results. trades: list of (index, exit_step, gross, net)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO backtest_trades "
                "(run_id, trade_index, exit_step, gross_profit, net_profit) "
                "VALUES (?, ?, ?, ?, ?)",
                [(run_id, t[0], t[1], t[2], t[3]) for t in trades],
            )
            conn.commit()

    def get_backtest_trades(self, run_id: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY trade_index",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Threshold sweeps ─────────────────────────────────────────────────────

    def save_threshold_sweep(self, run_id: str, results: list[dict]):
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO threshold_sweeps "
                "(run_id, threshold, total_gross_profit, total_net_profit, win_rate, num_trades) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(run_id, r["threshold"], r["total_gross_profit"],
                  r["total_net_profit"], r["win_rate"], r["num_trades"])
                 for r in results],
            )
            conn.commit()

    def get_threshold_sweep(self, run_id: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM threshold_sweeps WHERE run_id = ? ORDER BY threshold",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Summary stats ────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            by_type = conn.execute(
                "SELECT run_type, COUNT(*) FROM runs GROUP BY run_type"
            ).fetchall()
            finished = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status = 'finished'"
            ).fetchone()[0]
            best_run = conn.execute(
                "SELECT * FROM runs WHERE status='finished' "
                "ORDER BY avg_profit DESC LIMIT 1"
            ).fetchone()
        return {
            "total_runs": total,
            "by_type": {r[0]: r[1] for r in by_type},
            "finished_runs": finished,
            "best_run": dict(best_run) if best_run else None,
        }


# ── Global singleton ──────────────────────────────────────────────────────────
_store: Optional[ExperimentStore] = None


def get_experiment_store() -> ExperimentStore:
    global _store
    if _store is None:
        _store = ExperimentStore()
    return _store

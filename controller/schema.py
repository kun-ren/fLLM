"""
Hyperparameter schema definitions.
Each hyperparameter can be specified as a single value or a range [min, max].
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


class ParamMode(Enum):
    SINGLE = "single"
    RANGE = "range"


@dataclass
class HyperParam:
    """A hyperparameter that can be a single value or a range."""
    name: str
    value: Any                     # current / default single value
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    mode: ParamMode = ParamMode.SINGLE
    description: str = ""
    group: str = "General"
    slider: bool = False           # render as slider in GUI

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "step": self.step,
            "mode": self.mode.value,
            "description": self.description,
            "group": self.group,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HyperParam":
        return cls(
            name=d["name"],
            value=d["value"],
            min_val=d.get("min_val"),
            max_val=d.get("max_val"),
            step=d.get("step"),
            mode=ParamMode(d.get("mode", "single")),
            description=d.get("description", ""),
            group=d.get("group", "General"),
        )

    def get_active_value(self) -> Any:
        return self.value


# ──────────────────────────────────────────────────────────────────────────────
# Schema definition
# ──────────────────────────────────────────────────────────────────────────────

GROUPS = ["Data", "GCS", "Model", "Optimizer", "Training", "Loss", "Backtest"]

SCHEMA: dict[str, HyperParam] = {

    # ── Data ────────────────────────────────────────────────────────────────

    "dataset_filetype": HyperParam(
        name="dataset_filetype",
        value="csv",
        description="Dataset file type (csv / parquet / feather)",
        group="Data",
    ),
    "train_dataset_path": HyperParam(
        name="train_dataset_path",
        value="data/Binance_BTC_USDT_USDT_3m.csv",
        description="Path to OHLC CSV file",
        group="Data",
    ),
    "test_dataset_path": HyperParam(
        name="test_dataset_path",
        value="data/Binance_BTC_USDT_USDT_3m.csv",
        description="Path to test OHLC CSV file",
        group="Data",
    ),

    "use_last_n": HyperParam(
        name="use_last_n",
        value=False,
        description="Use last N rows of CSV ",
        group="Data",
    ),

    "use_last_n_num": HyperParam(
        name="use_last_n_num",
        value=None,
        min_val=1000,
        max_val=1000000,
        step=1000,
        description="Use last N rows of CSV to train ",
        group="Data",
        slider=True,
    ),
    "normalize": HyperParam(
        name="normalize",
        value=True,
        description="Z-score normalize volume features",
        group="Data",
    ),

    "seq_len": HyperParam(
        name="seq_len",
        value=64,
        min_val=8,
        max_val=512,
        step=8,
        description="Input sequence length",
        group="Data",
        slider=True,
    ),
    "sliding_window": HyperParam(
        name="sliding_window",
        value=False,
        description="Use sliding window sampling",
        group="Data",
    ),

    "sliding_step": HyperParam(
        name="sliding_step",
        value=1,
        min_val=1,
        max_val=64,
        step=1,
        description="Step size for sliding window",
        group="Data",
        slider=True,
    ),


    "num_look_ahead": HyperParam(
        name="num_look_ahead",
        value=10,
        min_val=1,
        max_val=50,
        step=1,
        description="Number of future steps for profit reference",
        group="Data",
        slider=True,
    ),

    # ── Model ───────────────────────────────────────────────────────────────

    # ── GCS ─────────────────────────────────────────────────────────────────
    "gcs_bucket_name": HyperParam(
        name="gcs_bucket_name",
        value="binance-histrial-files",
        description="GCS bucket name for data download",
        group="GCS",
    ),
    "gcs_blob_prefix": HyperParam(
        name="gcs_blob_prefix",
        value="aggTrades/BTCUSDT",
        description="Blob prefix path within bucket",
        group="GCS",
    ),
    "gcs_destination_dir": HyperParam(
        name="gcs_destination_dir",
        value="data",
        description="Local directory to save downloaded files",
        group="GCS",
    ),

    # ── Model ───────────────────────────────────────────────────────────────
    "aggregation_level": HyperParam(
        name="aggregation_level",
        value=4,
        min_val=2,
        max_val=10,
        step=1,
        description="Crossformer hierarchical attention aggregation_level(num of time series samples)",
        group="Model",
        slider=True,
    ),
    "num_tsa_layer": HyperParam(
        name="num_tsa_layer",
        value=1,
        min_val=1,
        max_val=10,
        step=1,
        description="num_tsa_layer",
        group="Model",
        slider=True,
    ),
    "router": HyperParam(
        name="router",
        value=False,
        description="router",
        group="Model",
        slider=True,
    ),
    "factor": HyperParam(
        name="factor",
        value=1,
        min_val=4,
        max_val=10,
        step=1,
        description="num_routers",
        group="Model",
        slider=True,
    ),

    "d_model": HyperParam(
        name="d_model",
        value=64,
        min_val=32,
        max_val=256,
        step=32,
        description="Model embedding dimension",
        group="Model",
        slider=True,
    ),
    "n_heads": HyperParam(
        name="n_heads",
        value=8,
        min_val=1,
        max_val=16,
        step=1,
        description="Number of attention heads",
        group="Model",
        slider=True,
    ),
    "n_layers": HyperParam(
        name="n_layers",
        value=4,
        min_val=1,
        max_val=12,
        step=1,
        description="Number of transformer layers",
        group="Model",
        slider=True,
    ),
    "dim_feedforward": HyperParam(
        name="dim_feedforward",
        value=128,
        min_val=64,
        max_val=512,
        step=64,
        description="Feedforward hidden dimension",
        group="Model",
        slider=True,
    ),

    "hidden_dim": HyperParam(
        name="hidden_dim",
        value=128,
        min_val=32,
        max_val=512,
        step=32,
        description="Embedding head hidden dimension",
        group="Model",
        slider=True,
    ),

    "dropout": HyperParam(
        name="dropout",
        value=0.1,
        min_val=0.0,
        max_val=0.5,
        step=0.05,
        description="Dropout rate",
        group="Model",
        slider=True,
    ),

    "pooling": HyperParam(
        name="pooling",
        value="AttentionPooling",
        description="Pooling strategy for sequence embeddings",
        group="Model",
        slider=True,
    ),

    # ── Optimizer ───────────────────────────────────────────────────────────
    "lr": HyperParam(
        name="lr",
        value=1e-3,
        min_val=1e-5,
        max_val=1e-2,
        step=None,
        description="Learning rate (log scale)",
        group="Optimizer",
        slider=True,
    ),
    "weight_decay": HyperParam(
        name="weight_decay",
        value=0.05,
        min_val=0.0,
        max_val=0.5,
        step=0.01,
        description="L2 weight decay",
        group="Optimizer",
        slider=True,
    ),
    "betas": HyperParam(
        name="betas",
        value="0.9, 0.999",
        description="AdamW beta coefficients (comma-separated)",
        group="Optimizer",
    ),
    "eps": HyperParam(
        name="eps",
        value=1e-8,
        min_val=1e-10,
        max_val=1e-6,
        step=None,
        description="AdamW epsilon for numerical stability",
        group="Optimizer",
        slider=True,
    ),

    # ── Training ─────────────────────────────────────────────────────────────
    "batch_size": HyperParam(
        name="batch_size",
        value=64,
        min_val=4,
        max_val=256,
        step=4,
        description="Training batch size",
        group="Training",
        slider=True,
    ),
    "epochs": HyperParam(
        name="epochs",
        value=10,
        min_val=1,
        max_val=100,
        step=1,
        description="Number of training epochs",
        group="Training",
        slider=True,
    ),
    "device": HyperParam(
        name="device",
        value="cuda",
        description="Device (cuda / cpu)",
        group="Training",
    ),
    "save_dir": HyperParam(
        name="save_dir",
        value="checkpoints",
        description="Directory to save model checkpoints",
        group="Training",
    ),

    # ── Loss ─────────────────────────────────────────────────────────────────
    "threshold_bps": HyperParam(
        name="threshold_bps",
        value=2.0,
        min_val=0.1,
        max_val=20.0,
        step=0.1,
        description="Profit normalization threshold (bps)",
        group="Loss",
        slider=True,
    ),
    "stop_loss_bps": HyperParam(
        name="stop_loss_bps",
        value=10.0,
        min_val=1.0,
        max_val=50.0,
        step=1.0,
        description="Stop-loss threshold in bps",
        group="Loss",
        slider=True,
    ),
    "loss_take_profit_bps": HyperParam(
        name="loss_take_profit_bps",
        value=5.0,
        min_val=0.5,
        max_val=50.0,
        step=0.5,
        description="Take-profit cap in loss function (bps)",
        group="Loss",
        slider=True,
    ),
    "loss_stop_loss_bps": HyperParam(
        name="loss_stop_loss_bps",
        value=10.0,
        min_val=1.0,
        max_val=50.0,
        step=1.0,
        description="Stop-loss cap in loss function (bps)",
        group="Loss",
        slider=True,
    ),

    # ── Backtest ─────────────────────────────────────────────────────────────
    "confidence_threshold": HyperParam(
        name="confidence_threshold",
        value=0.6,
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        description="Min confidence (reversal strength) to execute trade - filters tradable signals",
        group="Backtest",
        slider=True,
    ),
    "take_profit": HyperParam(
        name="take_profit",
        value=4.0,
        min_val=0.5,
        max_val=50.0,
        step=0.5,
        description="Take-profit threshold (bps)",
        group="Backtest",
        slider=True,
    ),
    "take_loss": HyperParam(
        name="take_loss",
        value=-50.0,
        min_val=-200.0,
        max_val=-1.0,
        step=1.0,
        description="Stop-loss threshold (bps, negative)",
        group="Backtest",
        slider=True,
    ),
    "margin": HyperParam(
        name="margin",
        value=15.0,
        min_val=1.0,
        max_val=100.0,
        step=1.0,
        description="Margin parameter",
        group="Backtest",
        slider=True,
    ),
    "commission_rate": HyperParam(
        name="commission_rate",
        value=0.0004,
        min_val=0.0,
        max_val=0.005,
        step=0.0001,
        description="Commission rate per trade (e.g. 0.04% = 0.0004)",
        group="Backtest",
        slider=True,
    ),
    "threshold_sweep_values": HyperParam(
        name="threshold_sweep_values",
        value="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        description="Comma-separated confidence thresholds for sweep analysis",
        group="Backtest",
    ),
}

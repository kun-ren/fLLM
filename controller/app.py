"""
Gradio web UI for the fLLM control panel.
Can be run standalone or embedded via FastAPI.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

from controller.config_manager import get_config_manager
from controller.experiment_store import get_experiment_store
from controller.schema import GROUPS, SCHEMA, HyperParam, ParamMode
from controller.train_worker import get_train_worker
from models.pooling import POOLING_REGISTRY
from web.static.css import page_css

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

FILETYPE_CHOICES = ["csv", "parquet", "feather"]
DEVICE_CHOICES = ["cuda", "cpu"]
DROPDOWN_PARAMS = {
    "dataset_filetype": FILETYPE_CHOICES,
    "device": DEVICE_CHOICES,
    "pooling": list(POOLING_REGISTRY.keys()),
}
TEXT_PARAMS = {
    "train_dataset_path", "test_dataset_path", "save_dir", "betas",
    "gcs_bucket_name", "gcs_blob_prefix", "gcs_destination_dir",
    "threshold_sweep_values",
}


def _do_load(cm, name):
    try:
        cm.load_config(name)
        return f"Loaded: {name}"
    except FileNotFoundError:
        return f"Config '{name}' not found"


# ──────────────────────────────────────────────────────────────────────────────
# Config Tab
# ──────────────────────────────────────────────────────────────────────────────

def build_config_tab(cm, store):
    """Build the hyperparameter configuration tab with GCS download controls."""

    group_blocks = {}
    for group in GROUPS:
        group_params = {n: p for n, p in cm._params.items() if p.group == group}
        if not group_params:
            continue

        with gr.Group():
            gr.Markdown(f"### {group}")
            editors = {}
            for name, param in group_params.items():
                if name in DROPDOWN_PARAMS:
                    comp = gr.Dropdown(
                        choices=DROPDOWN_PARAMS[name],
                        value=str(param.value or DROPDOWN_PARAMS[name][0]),
                        label=name,
                        info=param.description,
                        interactive=True,
                    )
                elif name in TEXT_PARAMS:
                    comp = gr.Textbox(
                        value=str(param.value or ""),
                        label=name,
                        info=param.description,
                        interactive=True,
                    )
                elif isinstance(param.value, bool):
                    comp = gr.Checkbox(
                        value=param.value,
                        label=name,
                        info=param.description,
                    )
                elif param.min_val is not None and param.max_val is not None:
                    comp = gr.Slider(
                        minimum=float(param.min_val),
                        maximum=float(param.max_val),
                        value=float(param.value) if param.value is not None else float(param.min_val),
                        step=float(param.step) if param.step else 1.0,
                        label=name,
                        info=param.description,
                        interactive=True,
                    )
                else:
                    comp = gr.Number(
                        value=param.value,
                        label=name,
                        info=param.description,
                        interactive=True,
                    )
                editors[name] = comp

                # Persist changes immediately to ConfigManager
                comp.change(
                    fn=lambda val, n=name: cm.set(n, value=val),
                    inputs=[comp],
                    outputs=[],
                )

            group_blocks[group] = editors

            # GCS download button after GCS group
            if group == "GCS":
                gcs_download_btn = gr.Button("Download Latest from GCS", variant="secondary")
                gcs_status = gr.Textbox(label="GCS Status", interactive=False)

                def do_gcs_download():
                    try:
                        from config.googleCloud import get_latest_blob_path, download_blob
                        bucket = cm.get("gcs_bucket_name").value
                        prefix = cm.get("gcs_blob_prefix").value
                        dest_dir = cm.get("gcs_destination_dir").value
                        Path(dest_dir).mkdir(parents=True, exist_ok=True)
                        blob_path = get_latest_blob_path(bucket, prefix)
                        if not blob_path:
                            return "No blobs found under prefix"
                        filename = blob_path.split("/")[-1]
                        dest_path = f"{dest_dir}/{filename}"
                        download_blob(bucket, blob_path, dest_path)
                        return f"Downloaded: {dest_path}"
                    except Exception as e:
                        return f"Error: {e}"

                gcs_download_btn.click(fn=do_gcs_download, inputs=[], outputs=[gcs_status])

    # Save/Load/Delete config
    with gr.Row():
        config_name_in = gr.Textbox(label="Config name to save/load", value="default")
        save_btn = gr.Button("Save Config", variant="primary")
        load_btn = gr.Button("Load Config")
        delete_btn = gr.Button("Delete Config", variant="stop")
        config_msg = gr.Textbox(label="Status", interactive=False)

    save_btn.click(
        fn=lambda name: (cm.save_config(name), f"Saved: {name}")[1],
        inputs=config_name_in,
        outputs=config_msg,
    )
    load_btn.click(
        fn=lambda name: _do_load(cm, name),
        inputs=config_name_in,
        outputs=config_msg,
    )
    delete_btn.click(
        fn=lambda name: (cm.delete_config(name), f"Deleted: {name}")[1],
        inputs=config_name_in,
        outputs=config_msg,
    )

    summary_box = gr.JSON(label="Active resolved config", value=cm.resolve_all())
    apply_btn = gr.Button("Apply to .env", variant="secondary")
    apply_btn.click(
        fn=lambda: cm.resolve_all(),
        inputs=[],
        outputs=summary_box,
    )

    return group_blocks


# ──────────────────────────────────────────────────────────────────────────────
# Train Tab
# ──────────────────────────────────────────────────────────────────────────────

def build_train_tab():
    """Build the training/test/backtest control tab."""
    worker = get_train_worker()
    store = get_experiment_store()

    gr.Markdown("### Training / Testing / Backtesting Controls")

    with gr.Row():
        run_type_select = gr.Dropdown(
            choices=["train", "test", "backtest"],
            value="train",
            label="Run Type",
        )
        start_btn = gr.Button("Start", variant="primary")
        stop_btn = gr.Button("Stop", variant="stop", interactive=False)

    # Reversal strength filter (confidence threshold) - prominent display
    with gr.Group():
        gr.Markdown("#### Reversal Strength Filter (反转强度过滤)")
        confidence_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=get_config_manager().get("confidence_threshold").value,
            step=0.05,
            label="Confidence Threshold (reversal strength to filter tradable signals)",
            info="Only signals with |confidence| >= this threshold will be traded",
            interactive=True,
        )
        confidence_slider.change(
            fn=lambda val: get_config_manager().set("confidence_threshold", value=val),
            inputs=[confidence_slider],
            outputs=[],
        )

    status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
    progress_bar = gr.Slider(minimum=0, maximum=1, value=0, label="Progress", interactive=False)
    loss_text = gr.Number(label="Current Loss", value=0, interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            log_output = gr.Textbox(label="Live Logs", lines=20, interactive=False)
        with gr.Column(scale=1):
            live_loss_chart = gr.LinePlot(
                label="Training Loss (Live)",
                x="step", y="loss",
                height=400,
            )

    loss_history = gr.State({"step": [], "loss": []})

    def start_job(run_type):
        """Start background job and poll for updates."""
        worker.start(run_type)
        # Initial yield
        yield (
            f"Starting {run_type}...",
            0.0,
            0.0,
            "",
            {"step": [], "loss": []},
            gr.Button(interactive=False),
            gr.Button(interactive=True),
        )

        while worker.is_running() or worker.status == "paused":
            time.sleep(1)
            logs = worker.logs
            log_text = "\n".join(logs[-50:])
            losses = {"step": [], "loss": []}

            # Fetch train logs from DB for chart
            if worker.run_id:
                train_logs = store.get_train_logs(worker.run_id)
                if train_logs:
                    losses = {
                        "step": [f"E{l['epoch']}-S{l['step']}" for l in train_logs[-200:]],
                        "loss": [l["loss"] for l in train_logs[-200:]],
                    }

            yield (
                f"{worker.status} | Epoch {worker.epoch}/{worker.total_epochs}",
                worker.progress,
                worker.current_loss,
                log_text,
                losses,
                gr.Button(interactive=False),
                gr.Button(interactive=True),
            )

        # Final state
        logs = worker.logs
        log_text = "\n".join(logs[-50:])
        losses = {"step": [], "loss": []}
        if worker.run_id:
            train_logs = store.get_train_logs(worker.run_id)
            if train_logs:
                losses = {
                    "step": [f"E{l['epoch']}-S{l['step']}" for l in train_logs[-200:]],
                    "loss": [l["loss"] for l in train_logs[-200:]],
                }

        yield (
            f"Completed ({worker.status})",
            1.0,
            worker.current_loss,
            log_text,
            losses,
            gr.Button(interactive=True),
            gr.Button(interactive=False),
        )

    def stop_job():
        worker.stop()
        return "Stopping..."

    start_btn.click(
        fn=start_job,
        inputs=[run_type_select],
        outputs=[status_text, progress_bar, loss_text, log_output,
                 live_loss_chart, start_btn, stop_btn],
    )
    stop_btn.click(fn=stop_job, inputs=[], outputs=[status_text])

    return status_text, log_output


# ──────────────────────────────────────────────────────────────────────────────
# History Tab
# ──────────────────────────────────────────────────────────────────────────────

def build_history_tab():
    """Build run history tab with clickable detail charts."""
    store = get_experiment_store()

    with gr.Row():
        filter_type = gr.Dropdown(
            choices=["all", "train", "test", "backtest"],
            value="all",
            label="Filter by type",
        )
        refresh_btn = gr.Button("Refresh")

    history_table = gr.DataFrame(
        label="Run History",
        headers=["id", "run_type", "status", "created_at", "train_loss",
                 "avg_profit", "num_trades", "win_rate"],
    )

    detail_box = gr.JSON(label="Run Detail")

    # Detail charts (shown on row click)
    gr.Markdown("### Detail Charts (click a row above to view)")
    gr.Markdown("View training loss, per-trade profits, equity curve, and reversal strength threshold analysis")

    with gr.Row():
        with gr.Column():
            detail_loss_chart = gr.LinePlot(
                label="Training Loss Curve (with TP/SL)",
                x="step", y="loss", height=300,
                tooltip=["step", "loss"],
            )
        with gr.Column():
            detail_profit_chart = gr.BarPlot(
                label="Per-Trade Net Profit (after commission fees 手续费)",
                x="trade_num", y="net_profit", height=300,
                tooltip=["trade_num", "net_profit"],
            )

    with gr.Row():
        with gr.Column():
            equity_chart = gr.LinePlot(
                label="Equity Curve (Cumulative Net Profit after 手续费)",
                x="trade_num", y="cumulative", height=300,
                tooltip=["trade_num", "cumulative"],
            )
        with gr.Column():
            sweep_chart = gr.LinePlot(
                label="Threshold Sweep: Net Profit (after 手续费)",
                x="threshold", y="total_net_profit", height=300,
                tooltip=["threshold", "total_net_profit"],
            )

    with gr.Row():
        with gr.Column():
            sweep_winrate_chart = gr.LinePlot(
                label="Reversal Strength Threshold Sweep: Win Rate",
                x="threshold", y="win_rate", height=300,
                tooltip=["threshold", "win_rate"],
            )
        with gr.Column():
            sweep_trades_chart = gr.BarPlot(
                label="Reversal Strength Threshold Sweep: Number of Trades",
                x="threshold", y="num_trades", height=300,
                tooltip=["threshold", "num_trades"],
            )

    def load_history(ft):
        t = None if ft == "all" else ft
        runs = store.list_runs(run_type=t, limit=100)
        if not runs:
            return [], {}
        cols = ["id", "run_type", "status", "created_at", "train_loss",
                "avg_profit", "num_trades", "win_rate"]
        rows = [[r.get(c) for c in cols] for r in runs]
        return rows, {}

    def on_row_select(evt: gr.SelectData, table_data):
        """When a row is clicked, load all detail charts for that run."""
        empty = ({}, {}, {}, {}, {}, {}, {})
        if table_data is None or len(table_data) == 0:
            return empty

        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if isinstance(table_data, pd.DataFrame):
            run_id = str(table_data.iloc[row_idx, 0])
        else:
            run_id = str(table_data[row_idx][0])

        run = store.get_run(run_id)
        if not run:
            return empty

        # Run detail JSON
        detail = run

        # Loss chart
        logs = store.get_train_logs(run_id)
        loss_data = {}
        if logs:
            loss_data = {
                "step": [f"E{l['epoch']}-S{l['step']}" for l in logs],
                "loss": [l["loss"] for l in logs],
            }

        # Trade profit chart + equity curve
        trades = store.get_backtest_trades(run_id)
        trade_data = {}
        equity_data = {}
        if trades:
            trade_data = {
                "trade_num": list(range(1, len(trades) + 1)),
                "net_profit": [t["net_profit"] for t in trades],
            }
            cum = 0.0
            eq = []
            for t in trades:
                cum += t["net_profit"]
                eq.append(cum)
            equity_data = {
                "trade_num": list(range(1, len(trades) + 1)),
                "cumulative": eq,
            }

        # Threshold sweep charts
        sweep = store.get_threshold_sweep(run_id)
        sweep_profit_data = {}
        sweep_wr_data = {}
        sweep_nt_data = {}
        if sweep:
            sweep_profit_data = {
                "threshold": [s["threshold"] for s in sweep],
                "total_net_profit": [s["total_net_profit"] for s in sweep],
            }
            sweep_wr_data = {
                "threshold": [s["threshold"] for s in sweep],
                "win_rate": [s["win_rate"] for s in sweep],
            }
            sweep_nt_data = {
                "threshold": [s["threshold"] for s in sweep],
                "num_trades": [s["num_trades"] for s in sweep],
            }

        return (detail, loss_data, trade_data, equity_data,
                sweep_profit_data, sweep_wr_data, sweep_nt_data)

    refresh_btn.click(
        fn=load_history,
        inputs=[filter_type],
        outputs=[history_table, detail_box],
    )
    history_table.select(
        fn=on_row_select,
        inputs=[history_table],
        outputs=[detail_box, detail_loss_chart, detail_profit_chart,
                 equity_chart, sweep_chart, sweep_winrate_chart, sweep_trades_chart],
    )

    return history_table, detail_box


# ──────────────────────────────────────────────────────────────────────────────
# Charts Tab
# ──────────────────────────────────────────────────────────────────────────────

def build_charts_tab():
    """Build the charts / analytics tab."""
    store = get_experiment_store()

    gr.Markdown("### Training Loss Curve")
    loss_chart = gr.LinePlot(
        label="Training Loss (with TP/SL in loss function)", x="step", y="loss",
    )

    gr.Markdown("### Threshold Sweep Analysis (Reversal Strength Filter)")
    gr.Markdown("Analyze how different reversal strength thresholds affect trading performance")
    with gr.Row():
        sweep_net_chart = gr.LinePlot(
            label="Net Profit vs Reversal Strength Threshold (after 手续费)",
            x="threshold", y="total_net_profit", height=300,
            tooltip=["threshold", "total_net_profit"],
        )
        sweep_gross_chart = gr.LinePlot(
            label="Gross Profit vs Reversal Strength Threshold (before 手续费)",
            x="threshold", y="total_gross_profit", height=300,
            tooltip=["threshold", "total_gross_profit"],
        )

    with gr.Row():
        sweep_wr_chart = gr.LinePlot(
            label="Win Rate vs Reversal Strength Threshold",
            x="threshold", y="win_rate", height=300,
            tooltip=["threshold", "win_rate"],
        )
        sweep_trades_chart = gr.BarPlot(
            label="Number of Trades vs Reversal Strength Threshold",
            x="threshold", y="num_trades", height=300,
            tooltip=["threshold", "num_trades"],
        )

    run_dropdown = gr.Dropdown(choices=[], label="Select Run")
    refresh_btn = gr.Button("Refresh Charts")

    def update_run_choices():
        runs = store.list_runs(limit=50)
        choices = [f"{r['id']} ({r['run_type']})" for r in runs]
        return gr.Dropdown(choices=choices)

    def load_charts(run_label):
        if not run_label:
            return {}, {}, {}, {}, {}
        run_id = run_label.split(" ")[0]

        # Loss chart
        logs = store.get_train_logs(run_id)
        loss_data = {}
        if logs:
            loss_data = {
                "step": [f"E{l['epoch']}-S{l['step']}" for l in logs],
                "loss": [l["loss"] for l in logs],
            }

        # Sweep charts
        sweep = store.get_threshold_sweep(run_id)
        net_data, gross_data, wr_data, trades_data = {}, {}, {}, {}
        if sweep:
            thresholds = [s["threshold"] for s in sweep]
            net_data = {
                "threshold": thresholds,
                "total_net_profit": [s["total_net_profit"] for s in sweep],
            }
            gross_data = {
                "threshold": thresholds,
                "total_gross_profit": [s["total_gross_profit"] for s in sweep],
            }
            wr_data = {
                "threshold": thresholds,
                "win_rate": [s["win_rate"] for s in sweep],
            }
            trades_data = {
                "threshold": thresholds,
                "num_trades": [s["num_trades"] for s in sweep],
            }

        return loss_data, net_data, gross_data, wr_data, trades_data

    refresh_btn.click(fn=update_run_choices, inputs=[], outputs=[run_dropdown])
    run_dropdown.change(
        fn=load_charts,
        inputs=[run_dropdown],
        outputs=[loss_chart, sweep_net_chart, sweep_gross_chart,
                 sweep_wr_chart, sweep_trades_chart],
    )

    return loss_chart


# ──────────────────────────────────────────────────────────────────────────────
# Main UI builder
# ──────────────────────────────────────────────────────────────────────────────

def build_ui():
    cm = get_config_manager()
    store = get_experiment_store()

    with gr.Blocks(css=page_css, theme=gr.themes.Soft()) as ui:
        state = gr.State("config")

        with gr.Row(elem_id="home-page"):
            # Sidebar
            with gr.Column(elem_id="sidebar-container"):
                with gr.Column(elem_id="sidebar"):
                    gr.Markdown("### Menu")
                    btn_config = gr.Button("Config")
                    btn_model = gr.Button("Model")
                    btn_train = gr.Button("Train/Test")
                    btn_history = gr.Button("History")
                    btn_charts = gr.Button("Charts")

            # Content
            with gr.Column(elem_id="content"):
                gr.Markdown("# fLLM Control Panel")

                tab_config = gr.Column(visible=True)
                with tab_config:
                    dropdown = gr.Dropdown(
                        label="Choose Config",
                        choices=cm.list_configs(),
                        value=cm.list_configs()[0] if cm.list_configs() else "default",
                    )

                    def onconfigchange(config_name):
                        cm.load_config(config_name)

                    dropdown.change(fn=onconfigchange, inputs=dropdown)
                    build_config_tab(cm, store)

                tab_model = gr.Column(visible=False)
                with tab_model:
                    gr.Markdown("### Model Architecture")
                    gr.Markdown("Model inspection — coming soon.")

                tab_train = gr.Column(visible=False)
                with tab_train:
                    build_train_tab()

                tab_history = gr.Column(visible=False)
                with tab_history:
                    build_history_tab()

                tab_charts = gr.Column(visible=False)
                with tab_charts:
                    build_charts_tab()

            def switch_tab(target_name):
                return [
                    gr.update(visible=(target_name == "config")),
                    gr.update(visible=(target_name == "model")),
                    gr.update(visible=(target_name == "train")),
                    gr.update(visible=(target_name == "history")),
                    gr.update(visible=(target_name == "charts")),
                    target_name,
                ]

            tab_list = [tab_config, tab_model, tab_train, tab_history, tab_charts, state]

            btn_config.click(fn=lambda: switch_tab("config"), outputs=tab_list)
            btn_model.click(fn=lambda: switch_tab("model"), outputs=tab_list)
            btn_train.click(fn=lambda: switch_tab("train"), outputs=tab_list)
            btn_history.click(fn=lambda: switch_tab("history"), outputs=tab_list)
            btn_charts.click(fn=lambda: switch_tab("charts"), outputs=tab_list)

    return ui


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7868)

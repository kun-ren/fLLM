"""
CLI for the fLLM control panel using Typer.
Usage:
  python -m controller.cli --help
  python -m controller.cli config list
  python -m controller.cli config set d_model 128
  python -m controller.cli run start train
  python -m controller.cli run status
  python -m controller.cli history
  python -m controller.cli gui
"""
from __future__ import annotations

import logging
import time
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.config_manager import get_config_manager
from controller.experiment_store import get_experiment_store
from controller.schema import GROUPS
from controller.train_worker import get_train_worker

logging.basicConfig(level=logging.INFO)
console = Console()
app = typer.Typer(help="fLLM Control Panel CLI")

# ──────────────────────────────────────────────────────────────────────────────
# Config subcommands
# ──────────────────────────────────────────────────────────────────────────────

config_app = typer.Typer(help="Manage hyperparameters")
app.add_typer(config_app, name="config")


@config_app.command("list")
def config_list():
    """List all hyperparameters and their current values."""
    cm = get_config_manager()
    table = Table(title="Hyperparameters")
    table.add_column("Name", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Mode", style="yellow")
    table.add_column("Range", style="blue")
    table.add_column("Group", style="magenta")

    for name, p in cm._params.items():
        mode_str = p.mode.value.upper()
        if p.mode.value == "range" and p.min_val is not None:
            range_str = f"[{p.min_val}, {p.max_val}]"
        else:
            range_str = ""
        table.add_row(name, str(p.value), mode_str, range_str, p.group)

    console.print(table)


@config_app.command("set")
def config_set(
    name: str = typer.Argument(..., help="Parameter name"),
    value: str = typer.Argument(..., help="Value (or min,max for range mode)"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Mode: single or range"),
):
    """Set a hyperparameter value."""
    cm = get_config_manager()
    if name not in cm._params:
        console.print(f"[red]Unknown parameter: {name}[/red]")
        raise typer.Exit(1)

    p = cm._params[name]
    if mode:
        mode = mode.lower()
        if mode not in ("single", "range"):
            console.print("[red]Mode must be 'single' or 'range'[/red]")
            raise typer.Exit(1)
    else:
        mode = p.mode.value

    if mode == "range" and "," in value:
        min_str, max_str = value.split(",")
        min_val = float(min_str.strip())
        max_val = float(max_str.strip())
        # Use midpoint as current value
        mid_val = (min_val + max_val) / 2
        if isinstance(p.value, int):
            mid_val = int(mid_val)
        from controller.schema import ParamMode as PM
        cm.set(name, value=mid_val, mode=PM.RANGE, min_val=min_val, max_val=max_val)
    else:
        # Parse value
        if isinstance(p.value, bool):
            val = value.lower() in ("true", "1", "yes")
        elif isinstance(p.value, int):
            val = int(value)
        elif isinstance(p.value, float):
            val = float(value)
        else:
            val = value
        cm.set(name, value=val)

    console.print(f"[green]Set {name} = {cm.get(name).value}[/green]")


@config_app.command("save")
def config_save(name: str = typer.Argument(..., help="Config file name")):
    """Save current config to a file."""
    cm = get_config_manager()
    path = cm.save_config(name)
    console.print(f"[green]Saved to {path}[/green]")


@config_app.command("load")
def config_load(name: str = typer.Argument(..., help="Config file name")):
    """Load config from a file."""
    cm = get_config_manager()
    try:
        path = cm.load_config(name)
        console.print(f"[green]Loaded from {path}[/green]")
    except FileNotFoundError:
        console.print(f"[red]Config '{name}' not found[/red]")
        raise typer.Exit(1)


@config_app.command("saved")
def config_saved():
    """List saved config files."""
    cm = get_config_manager()
    names = cm.list_configs()
    if not names:
        console.print("[yellow]No saved configs found[/yellow]")
    else:
        for n in names:
            console.print(f"  - {n}")


@config_app.command("apply")
def config_apply():
    """Apply current config values to .env and reload env_loader."""
    cm = get_config_manager()
    cm.apply_to_env()
    console.print("[green]Config applied to .env[/green]")
    console.print(cm.summary())


@config_app.command("summary")
def config_summary():
    """Print current resolved config."""
    cm = get_config_manager()
    console.print(cm.summary())


# ──────────────────────────────────────────────────────────────────────────────
# Run subcommands
# ──────────────────────────────────────────────────────────────────────────────

run_app = typer.Typer(help="Start / stop / control training runs")
app.add_typer(run_app, name="run")


@run_app.command("start")
def run_start(
    run_type: str = typer.Argument(..., help="Type: train, test, or backtest"),
    experiment: Optional[str] = typer.Option(None, "--experiment", "-e", help="Experiment name"),
):
    """Start a training / test / backtest run."""
    if run_type not in ("train", "test", "backtest"):
        console.print(f"[red]Invalid run_type: {run_type}[/red]")
        raise typer.Exit(1)

    worker = get_train_worker()
    if worker.is_running():
        console.print("[yellow]A job is already running — stop it first[/yellow]")
        raise typer.Exit(1)

    store = get_experiment_store()
    exp_id = None
    if experiment:
        exp_id = store.create_experiment(experiment)
        console.print(f"Created experiment: {exp_id}")

    console.print(f"[green]Starting {run_type} run...[/green]")
    worker.start(run_type, exp_id)
    time.sleep(0.5)
    _print_status(worker)


@run_app.command("stop")
def run_stop():
    """Stop the current running job."""
    worker = get_train_worker()
    if not worker.is_running():
        console.print("[yellow]No job is currently running[/yellow]")
        return
    worker.stop()
    console.print("[yellow]Stop signal sent[/yellow]")
    time.sleep(1)
    _print_status(worker)


@run_app.command("pause")
def run_pause():
    """Pause the current job."""
    worker = get_train_worker()
    if not worker.is_running():
        console.print("[yellow]No running job to pause[/yellow]")
        return
    worker.pause()
    console.print(f"[yellow]Paused — status: {worker.status}[/yellow]")


@run_app.command("resume")
def run_resume():
    """Resume a paused job."""
    worker = get_train_worker()
    if worker.status != "paused":
        console.print("[yellow]No paused job to resume[/yellow]")
        return
    worker.resume()
    console.print("[green]Resumed[/green]")


@run_app.command("status")
def run_status():
    """Show current job status."""
    worker = get_train_worker()
    _print_status(worker)


@run_app.command("logs")
def run_logs(lines: int = typer.Option(50, "--lines", "-n", help="Number of log lines")):
    """Print recent log lines."""
    worker = get_train_worker()
    logs = worker.logs[-lines:]
    if not logs:
        console.print("[yellow]No logs yet[/yellow]")
    else:
        for line in logs:
            console.print(line)


def _print_status(worker):
    console.print("\n[bold]Worker Status[/bold]")
    console.print(f"  Status:      {worker.status}")
    console.print(f"  Run ID:      {worker.run_id}")
    console.print(f"  Run Type:    {worker.run_type}")
    console.print(f"  Progress:    {worker.progress * 100:.1f}%")
    console.print(f"  Epoch:       {worker.epoch} / {worker.total_epochs}")
    console.print(f"  Current Loss:{worker.current_loss:.6f}")


# ──────────────────────────────────────────────────────────────────────────────
# History subcommands
# ──────────────────────────────────────────────────────────────────────────────

history_app = typer.Typer(help="View experiment and run history")
app.add_typer(history_app, name="history")


@history_app.command("runs")
def history_runs(
    run_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type"),
    limit: int = typer.Option(50, "--limit", "-n"),
):
    """List recent runs."""
    store = get_experiment_store()
    runs = store.list_runs(run_type=run_type, limit=limit)
    if not runs:
        console.print("[yellow]No runs found[/yellow]")
        return

    table = Table(title="Run History")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Created", style="blue")
    table.add_column("Train Loss", style="green")
    table.add_column("Avg Profit", style="green")
    table.add_column("Trades", style="yellow")

    for r in runs:
        table.add_row(
            r.get("id", ""),
            r.get("run_type", ""),
            r.get("status", ""),
            r.get("created_at", "")[:19],
            f"{r.get('train_loss', ''):.6f}" if r.get("train_loss") else "-",
            f"{r.get('avg_profit', ''):.2f}" if r.get("avg_profit") else "-",
            str(r.get("num_trades", "-")),
        )
    console.print(table)


@history_app.command("experiments")
def history_experiments():
    """List all experiments."""
    store = get_experiment_store()
    exps = store.list_experiments()
    if not exps:
        console.print("[yellow]No experiments found[/yellow]")
        return

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Created", style="blue")
    table.add_column("Config", style="magenta")

    for e in exps:
        table.add_row(
            e.get("id", ""),
            e.get("name", ""),
            e.get("status", ""),
            e.get("created_at", "")[:19],
            e.get("config_name", ""),
        )
    console.print(table)


@history_app.command("show")
def history_show(run_id: str = typer.Argument(..., help="Run ID")):
    """Show detail for a specific run."""
    store = get_experiment_store()
    run = store.get_run(run_id)
    if not run:
        console.print(f"[red]Run {run_id} not found[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Run {run_id}[/bold]")
    console.print(f"  Type:       {run.get('run_type')}")
    console.print(f"  Status:     {run.get('status')}")
    console.print(f"  Created:    {run.get('created_at')}")
    console.print(f"  Duration:   {run.get('duration_sec', 0):.1f}s")
    console.print(f"  Train Loss: {run.get('train_loss')}")
    console.print(f"  Best Loss:  {run.get('best_loss')}")
    console.print(f"  Avg Profit: {run.get('avg_profit')}")
    console.print(f"  Num Trades: {run.get('num_trades')}")
    console.print(f"  Win Rate:   {run.get('win_rate')}")
    console.print(f"  Checkpoint: {run.get('checkpoint_path')}")


@history_app.command("summary")
def history_summary():
    """Show summary statistics."""
    store = get_experiment_store()
    s = store.get_summary()
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Total runs:     {s['total_runs']}")
    console.print(f"  Finished runs:  {s['finished_runs']}")
    for t, c in s["by_type"].items():
        console.print(f"  {t}: {c}")
    if s["best_run"]:
        br = s["best_run"]
        console.print(f"\n[bold]Best Run (by avg_profit)[/bold]")
        console.print(f"  ID:         {br.get('id')}")
        console.print(f"  Type:       {br.get('run_type')}")
        console.print(f"  Avg Profit: {br.get('avg_profit')}")
        console.print(f"  Win Rate:   {br.get('win_rate')}")


@history_app.command("logs")
def history_logs(run_id: str = typer.Argument(..., help="Run ID")):
    """Print logs for a specific run."""
    store = get_experiment_store()
    run = store.get_run(run_id)
    if not run:
        console.print(f"[red]Run {run_id} not found[/red]")
        raise typer.Exit(1)
    log_text = run.get("log_text", "")
    if not log_text:
        console.print("[yellow]No logs for this run[/yellow]")
    else:
        for line in log_text.splitlines():
            console.print(line)


# ──────────────────────────────────────────────────────────────────────────────
# GUI launch
# ──────────────────────────────────────────────────────────────────────────────

@app.command("gui")
def gui(
    port: int = typer.Option(7860, "--port", "-p", help="Server port"),
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
):
    """Launch the Gradio web UI."""
    from controller.app import build_ui
    import gradio as gr

    console.print(f"[green]Launching GUI at http://{host}:{port}[/green]")
    ui = build_ui()
    ui.launch(server_name=host, server_port=port, share=False)


@app.command("serve")
def serve(
    port: int = typer.Option(7860, "--port", "-p", help="Server port"),
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
):
    """Run the FastAPI server with embedded Gradio UI."""
    from controller.api import run_server
    console.print(f"[green]Starting server at http://{host}:{port}[/green]")
    run_server(host=host, port=port)


if __name__ == "__main__":
    app()

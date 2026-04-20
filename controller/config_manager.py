"""
Configuration manager — loads, saves, and resolves hyperparameters.
Each param can be a single value or a [min, max] range.
"""
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from controller.schema import SCHEMA, HyperParam, ParamMode

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("controller/configs")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Manages active hyperparameters with range resolution and .env persistence."""

    def __init__(self):
        self._params: dict[str, HyperParam] = {}
        self._active_config_path: Optional[Path] = None
        self._load_defaults()
        self.save_config("default")

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load_defaults(self):
        for name, param in SCHEMA.items():
            self._params[name] = HyperParam(
                name=param.name,
                value=param.value,
                min_val=param.min_val,
                max_val=param.max_val,
                step=param.step,
                mode=param.mode,
                description=param.description,
                group=param.group,
            )

    def save_config(self, name: str) -> Path:
        """Save current params to a JSON config file."""
        path = CONFIG_DIR / f"{name}.json"
        data = {
            name_: p.to_dict() for name_, p in self._params.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._active_config_path = path
        logger.info(f"Config saved to {path}")
        return path

    def load_config(self, name: str) -> Path:
        """Load params from a JSON config file."""
        path = CONFIG_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Config '{name}' not found at {path}")
        with open(path) as f:
            data = json.load(f)
        for name_, d in data.items():
            if name_ in self._params:
                self._params[name_] = HyperParam.from_dict(d)
        self._active_config_path = path
        logger.info(f"Config loaded from {path}")
        return path

    def list_configs(self) -> list[str]:
        """List all saved config files."""
        return [p.stem for p in CONFIG_DIR.glob("*.json")]

    def delete_config(self, name: str) -> bool:
        """Delete a saved config."""
        path = CONFIG_DIR / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    # ── Param access ─────────────────────────────────────────────────────────

    def get(self, name: str) -> HyperParam:
        return self._params[name]

    def set(self, name: str, value=None, mode: ParamMode = ParamMode.SINGLE,
            min_val=None, max_val=None):
        """Update a hyperparameter."""
        p = self._params[name]
        p.value = value if value is not None else p.value
        p.mode = mode
        p.min_val = min_val if min_val is not None else p.min_val
        p.max_val = max_val if max_val is not None else p.max_val

    def get_active_config_path(self) -> Optional[Path]:
        return self._active_config_path

    # ── Range resolution ─────────────────────────────────────────────────────

    def resolve_value(self, name: str) -> float | int | bool | str | None:
        """
        Return the single value for a param.
        If mode is RANGE, sample uniformly (or pick midpoint for int).
        """
        p = self._params[name]
        if p.mode == ParamMode.RANGE and p.min_val is not None and p.max_val is not None:
            if isinstance(p.value, bool):
                return p.value
            if p.step and p.step < 1:  # float range
                return random.uniform(p.min_val, p.max_val)
            # int range — pick midpoint then quantize
            raw = (p.min_val + p.max_val) / 2
            if p.step:
                raw = round(raw / p.step) * p.step
            return int(raw) if isinstance(p.value, int) else float(raw)
        return p.value

    def resolve_all(self) -> dict[str, float | int | bool | str | None]:
        """Resolve all params to their active values."""
        return {name: self.resolve_value(name) for name in self._params}

    # ── Apply to environment / modules ──────────────────────────────────────

    def apply_to_env(self):
        """Write resolved values back to the .env file and reload env_loader."""
        resolved = self.resolve_all()
        env_path = Path(".env")
        if not env_path.exists():
            logger.warning(".env not found — skipping env write")
            return

        # Build env-key → value mapping
        env_updates: dict[str, str] = {}
        for schema_key, val in resolved.items():
            if val is None:
                env_updates[schema_key] = "None"
            elif isinstance(val, bool):
                env_updates[schema_key] = str(val)
            else:
                env_updates[schema_key] = str(val)

        lines = env_path.read_text().splitlines()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue
            if "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in env_updates:
                    new_lines.append(f"{key}={env_updates[key]}")
                    env_updates.pop(key)  # mark as written
                    continue
            new_lines.append(line)

        # Append any new keys that weren't in .env
        for key, val_str in env_updates.items():
            new_lines.append(f"{key}={val_str}")

        env_path.write_text("\n".join(new_lines) + "\n")
        from config import env_loader
        env_loader.reload_config()
        logger.info("Config applied to .env and env_loader reloaded")

    def apply_to_model(self):
        """Apply resolved values directly to model/training code."""
        resolved = self.resolve_all()
        # This is called during training to inject the current resolved config
        return resolved

    # ── Status ────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = []
        for group in ["Data", "Model", "Optimizer", "Training", "Loss", "Backtest"]:
            lines.append(f"\n## {group}")
            for name, p in self._params.items():
                if p.group == group:
                    if p.mode == ParamMode.RANGE and p.min_val is not None:
                        lines.append(f"  {name}: [{p.min_val}, {p.max_val}] (range)")
                    else:
                        lines.append(f"  {name}: {p.value}")
        return "\n".join(lines)


# ── Global singleton ──────────────────────────────────────────────────────────
_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    global _manager
    if _manager is None:
        _manager = ConfigManager()
    return _manager

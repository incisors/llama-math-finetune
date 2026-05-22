"""Shared utilities: seeding, config loading, W&B init."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set RNG seed across python / numpy / torch / cuda for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def validate_effective_batch(cfg: dict[str, Any], num_gpus: int = 1) -> None:
    """Check that per_device * grad_accum * num_gpus == declared effective_batch_size."""
    t = cfg["training"]
    declared = t["effective_batch_size"]
    actual = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] * num_gpus
    if declared != actual:
        raise ValueError(
            f"Effective batch size mismatch: declared={declared}, "
            f"actual={actual} (per_device={t['per_device_train_batch_size']} "
            f"× grad_accum={t['gradient_accumulation_steps']} × num_gpus={num_gpus})"
        )


def init_wandb(cfg: dict[str, Any], extra_config: dict[str, Any] | None = None):
    """Initialize a W&B run. Returns the run handle."""
    import wandb

    out = cfg["output"]
    config_to_log = {**cfg}
    if extra_config:
        config_to_log.update(extra_config)
    return wandb.init(
        project=out["wandb_project"],
        name=out["wandb_run_name"],
        config=config_to_log,
        save_code=True,
    )

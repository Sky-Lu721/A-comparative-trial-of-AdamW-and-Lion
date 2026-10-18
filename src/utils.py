"""Configuration, reproducibility, and atomic artifact utilities."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs and select deterministic PyTorch algorithms."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python in each DataLoader worker."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read YAML; interpret relative output paths relative to project root."""
    with open(path or ROOT / "configs/experiment_config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def project_path(path: str) -> Path:
    """Resolve an absolute or project-relative path."""
    return ROOT / path


def save_json(path: Path, data: Any) -> None:
    """Atomically save JSON; represent missing numeric values as null."""
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(data), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_csv(path: Path, frame: Any) -> None:
    """Atomically save a pandas DataFrame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def select_device(name: str) -> torch.device:
    """Use CUDA when available, otherwise fall back explicitly to CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable: falling back to CPU.")
        name = "cpu"
    return torch.device(name)

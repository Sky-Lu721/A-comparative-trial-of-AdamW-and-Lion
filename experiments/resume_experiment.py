"""Reuse completed runs; restart interrupted runs from paired initial weights."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from _common import config_from_cli
from src import tuning
from src.utils import project_path, save_json


def reusable_summary(run_dir: Path, config: dict, name: str, seed: int,
                     lr: float, wd: float, epochs: int) -> dict | None:
    """Validate cached settings before reusing a complete observation."""
    if not run_dir.exists():
        return None
    config_path = run_dir / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError(f"Configuration mismatch: {config_path}")
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = dict(optimizer=name, seed=seed, best_learning_rate=lr,
                    best_weight_decay=wd, epochs=epochs)
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Cached run mismatch ({key}): {summary_path}")
    if not config_path.exists() or not (run_dir / "checkpoint.pt").exists():
        raise ValueError(f"Incomplete completed-run artifacts: {run_dir}")
    return summary


def main() -> None:
    """Run the full pipeline without rerunning completed candidates or seeds."""
    config = config_from_cli()
    if str(config["training"]["device"]).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Resuming this GPU experiment requires working CUDA")
    base = project_path(config["paths"]["results"])
    tuning.ensure_training_output(base)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive = base / "interrupted" / stamp
    original = tuning.run_training

    def resume_training(config: dict, name: str, seed: int, lr: float,
                        wd: float, epochs: int, run_dir: Path,
                        initial: dict, **kwargs: Any) -> dict:
        cached = reusable_summary(run_dir, config, name, seed, lr, wd, epochs)
        if cached is not None:
            print(f"Reusing completed run: {run_dir}", flush=True)
            return cached
        if run_dir.exists() and any(run_dir.iterdir()):
            destination = archive / run_dir.relative_to(base)
            destination.parent.mkdir(parents=True, exist_ok=True)
            run_dir.rename(destination)
            print(f"Archived interrupted run: {destination}", flush=True)
        return original(config, name, seed, lr, wd, epochs, run_dir,
                        initial, **kwargs)

    # Only orchestration is changed. Training and measurement code stay identical.
    tuning.run_training = resume_training
    stage = "sanity"
    try:
        save_json(base / "pipeline_status.json", {"status": "running", "stage": stage})
        tuning.run_final(config, sanity=True)
        for stage in ("lr", "weight_decay"):
            config_path = base / f"tuning_{stage}_config.json"
            if config_path.exists():
                if json.loads(config_path.read_text(encoding="utf-8")) != config:
                    raise ValueError(f"Configuration mismatch: {config_path}")
            csv_path = base / f"tuning_{stage}.csv"
            if csv_path.exists():
                destination = archive / csv_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                csv_path.rename(destination)
            save_json(base / "pipeline_status.json",
                      {"status": "running", "stage": f"{stage}_search"})
            tuning.run_search(config, stage)
        stage = "final_training_and_report"
        save_json(base / "pipeline_status.json", {"status": "running", "stage": stage})
        tuning.run_final(config)
        save_json(base / "pipeline_status.json", {"status": "complete", "stage": "all"})
    except BaseException as error:
        save_json(base / "pipeline_status.json",
                  {"status": "interrupted" if isinstance(error, KeyboardInterrupt)
                   else "failed", "stage": stage, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()

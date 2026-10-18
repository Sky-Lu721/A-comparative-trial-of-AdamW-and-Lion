"""Run sanity, equal-budget tuning, final training, and report generation."""
from __future__ import annotations

import traceback

import torch

from _common import config_from_cli
from src.tuning import ensure_training_output, run_final, run_search
from src.utils import project_path, save_json


def main() -> None:
    """Execute sanity, LR search, WD search, final training, and report generation."""
    config = config_from_cli()
    base = project_path(config["paths"]["results"])
    ensure_training_output(base)
    if str(config["training"]["device"]).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Configured GPU experiment requires working CUDA")
    stages = [
        ("sanity", lambda: run_final(config, sanity=True)),
        ("lr_search", lambda: run_search(config, "lr")),
        ("weight_decay_search", lambda: run_search(config, "weight_decay")),
        ("final_training_and_report", lambda: run_final(config)),
    ]
    try:
        for stage, execute in stages:
            print(f"Starting stage: {stage}", flush=True)
            save_json(base / "pipeline_status.json", {"status": "running", "stage": stage})
            execute()
            print(f"Completed stage: {stage}", flush=True)
        save_json(base / "pipeline_status.json", {"status": "complete", "stage": "all"})
    except Exception:
        save_json(base / "pipeline_status.json",
                  {"status": "failed", "stage": stage, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()

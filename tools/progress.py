"""Read actual pipeline progress without importing training libraries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def progress(base: Path) -> dict:
    """Summarize completed observations and the latest unfinished run."""
    state_path = base / "pipeline_status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    completed = list((base / "tuning").glob("*/*/*/summary.json"))
    finals = list((base / "final").glob("*/seed_*/summary.json"))
    incomplete = []
    for path in base.rglob("config.json"):
        ignored = {"sanity", "interrupted"}
        if (ignored.intersection(path.relative_to(base).parts)
                or (path.parent / "summary.json").exists()):
            continue
        incomplete.append(path.parent)
    current = max(incomplete, key=lambda p: p.stat().st_mtime) if incomplete else None
    rows = []
    if current and (current / "history.csv").exists():
        with (current / "history.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    state.update(completed_tuning_runs=len(completed), total_tuning_runs=18,
                 completed_final_runs=len(finals), total_final_runs=6,
                 current_run=str(current.relative_to(base)) if current else None,
                 completed_epochs_current_run=len(rows))
    if rows:
        state["latest_epoch"] = {key: float(rows[-1][key]) for key in
                                 ("epoch", "train_loss", "validation_accuracy",
                                  "epoch_time", "steps_per_second")}
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(progress(args.results), indent=2))

"""Validate published measurements without datasets, checkpoints, or training."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a saved table without importing the training stack."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    """Fail explicitly even when Python runs with optimizations enabled."""
    if not condition:
        raise ValueError(message)


def audit(base: Path, plots: Path) -> dict:
    """Check completeness, paired initialization, timing, and sample statistics."""
    status = json.loads((base / "pipeline_status.json").read_text(encoding="utf-8"))
    require(status["status"] == "complete", "Experiment is not marked complete")
    config = json.loads((base / "tuning_weight_decay_config.json").read_text())
    seeds = config["training"]["seeds"]
    names = ("adamw", "lion")
    rows = read_csv(base / "final_results.csv")
    expected = {(name, seed) for name in names for seed in seeds}
    observed = {(row["optimizer"], int(row["seed"])) for row in rows}
    require(len(rows) == len(expected) and observed == expected,
            "Missing or duplicated final optimizer/seed combinations")
    for seed in seeds:
        paired = [r for r in rows if int(r["seed"]) == seed]
        require(len({r["initial_weights_sha256"] for r in paired}) == 1,
                f"Initial weights do not match for seed {seed}")
    for row in rows:
        directory = base / "final" / row["optimizer"] / f"seed_{row['seed']}"
        history = read_csv(directory / "history.csv")
        epochs = int(row["epochs"])
        require(epochs == config["training"]["epochs"], "Epoch budget mismatch")
        require([int(r["epoch"]) for r in history] == list(range(1, epochs + 1)),
                f"Incomplete history: {directory}")
        require(int(row["train_samples"]) == 45000
                and int(row["validation_samples"]) == 5000, "Split sizes differ")
        total = sum(float(r["epoch_time"]) for r in history)
        require(math.isclose(total, float(row["training_time_sec"]), rel_tol=1e-9),
                "Training time differs from the sum of epoch times")
        require(math.isclose(float(row["steps_per_sec"]),
                             float(row["training_steps"]) / total, rel_tol=1e-9),
                "Throughput is inconsistent with steps and training time")
        require(math.isclose(max(float(r["validation_accuracy"]) for r in history),
                             float(row["best_val_accuracy"])), "Best validation mismatch")
        target = float(row["target_accuracy"])
        crossing = next((r for r in history if float(r["validation_accuracy"]) >= target), None)
        if crossing:
            require(math.isclose(float(row["time_to_target_accuracy"]),
                                 float(crossing["cumulative_training_time"])),
                    "Time-to-target differs from the first threshold crossing")
        else:
            require(not row["time_to_target_accuracy"], "Unreached target has a value")
    tuning = read_csv(base / "tuning_results.csv")
    require(len(tuning) == 18, "Expected 18 equal-budget tuning observations")
    for stage, key in (("lr", "learning_rates"), ("weight_decay", "weight_decays")):
        count = len(config["tuning"][key])
        for name in names:
            candidates = [r for r in tuning if r["stage"] == stage and r["optimizer"] == name]
            require(len(candidates) == count, f"Incomplete {stage} budget for {name}")
            for index in range(count):
                history = read_csv(base / "tuning" / stage / name / str(index) / "history.csv")
                require(len(history) == config["tuning"]["epochs"], "Incomplete tuning history")
    for item in read_csv(base / "summary_statistics.csv"):
        values = [float(r[item["metric"]]) for r in rows
                  if r["optimizer"] == item["optimizer"] and r[item["metric"]]]
        require(int(item["available_seeds"]) == len(values), "Seed count mismatch")
        if values:
            require(math.isclose(float(item["mean"]), statistics.mean(values), rel_tol=1e-9),
                    f"Mean mismatch for {item['metric']}")
        if len(values) > 1:
            require(math.isclose(float(item["std"]), statistics.stdev(values),
                                 rel_tol=1e-8, abs_tol=1e-12), "Sample SD mismatch")
    figures = list(plots.glob("figure*.png"))
    require(len(figures) == 7, "Expected seven published figures")
    for path in figures:
        require(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), f"Invalid PNG: {path}")
    return dict(status="passed", final_runs=len(rows), tuning_runs=len(tuning),
                final_epochs=sum(int(r["epochs"]) for r in rows), figures=len(figures),
                paired_initial_weights_verified=True, sample_statistics_verified=True,
                checkpoints_required=False)


def main() -> None:
    """Audit the included result snapshot or another complete result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/gpu_full")
    parser.add_argument("--plots", type=Path, default=ROOT / "plots/gpu_full")
    args = parser.parse_args()
    print(json.dumps(audit(args.results, args.plots), indent=2))


if __name__ == "__main__":
    main()

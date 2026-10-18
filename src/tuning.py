"""Equal-budget sequential LR and weight-decay search, using validation only."""
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .train import initial_weights, run_training
from .utils import project_path, save_csv, save_json

OPTIMIZERS = ("adamw", "lion")


def ensure_training_output(base: Path) -> None:
    """Protect the distributed measurements from accidental training writes."""
    if (base / "PUBLISHED_RESULTS.md").exists():
        raise ValueError(
            "This directory contains published results. Use "
            "configs/reproduce_gpu.yaml or choose new results/plots paths. "
            "Report regeneration through src.evaluate remains available.")


def best_candidate(frame: pd.DataFrame) -> pd.Series:
    """Maximize best validation accuracy; break ties by lower LR then WD."""
    return frame.sort_values(
        ["best_val_accuracy", "learning_rate", "weight_decay"],
        ascending=[False, True, True], kind="stable").iloc[0]


def load_selection(base: Path, stage: str) -> dict:
    """Require successful tuning before dependent experiment stages."""
    path = base / f"best_{stage}.json"
    if not path.exists():
        raise FileNotFoundError(f"Run {stage} tuning first: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_search(config: dict[str, Any], stage: str) -> None:
    """Evaluate all candidates; persist progress and selected hyperparameters."""
    if stage not in ("lr", "weight_decay"):
        raise ValueError(stage)
    base = project_path(config["paths"]["results"])
    ensure_training_output(base)
    settings = config["tuning"]
    seed = settings["seed"]
    initial = initial_weights(config, seed, base)
    lr_selection = load_selection(base, "lr") if stage == "weight_decay" else None
    if stage == "weight_decay":
        previous = json.loads((base / "tuning_lr_config.json").read_text())
        for key in ("dataset", "training", "optimizers", "tuning"):
            if previous[key] != config[key]:
                raise ValueError(f"Configuration changed since LR tuning: {key}")
    stage_path = base / f"tuning_{stage}.csv"
    if stage_path.exists():
        raise FileExistsError(f"Tuning already exists: {stage_path}")
    save_json(base / f"tuning_{stage}_config.json", config)
    rows, selected = [], {}
    candidates = settings["learning_rates" if stage == "lr" else "weight_decays"]
    for name in OPTIMIZERS:
        for index, candidate in enumerate(candidates):
            lr = candidate if stage == "lr" else lr_selection[name]["learning_rate"]
            wd = settings["lr_search_weight_decay"] if stage == "lr" else candidate
            result = run_training(config, name, seed, lr, wd, settings["epochs"],
                                  base / "tuning" / stage / name / str(index), initial)
            rows.append(dict(stage=stage, optimizer=name, seed=seed, learning_rate=lr,
                             weight_decay=wd, best_val_accuracy=result["best_val_accuracy"],
                             final_val_accuracy=result["final_val_accuracy"],
                             final_val_loss=result["final_val_loss"],
                             training_time=result["training_time_sec"]))
            save_csv(stage_path, pd.DataFrame(rows))
        best = best_candidate(pd.DataFrame([r for r in rows if r["optimizer"] == name]))
        selected[name] = dict(learning_rate=float(best["learning_rate"]),
                              weight_decay=float(best["weight_decay"]))
    save_json(base / f"best_{stage}.json", selected)
    frames = [pd.read_csv(path) for path in
              (base / "tuning_lr.csv", base / "tuning_weight_decay.csv") if path.exists()]
    save_csv(base / "tuning_results.csv", pd.concat(frames, ignore_index=True))


def run_final(config: dict[str, Any], sanity: bool = False) -> None:
    """Run paired seeds; isolate sanity artifacts from official experiment data."""
    base = project_path(config["paths"]["results"])
    ensure_training_output(base)
    selection = load_selection(base, "weight_decay") if not sanity else None
    if not sanity:
        tuning_config = json.loads((base / "tuning_weight_decay_config.json").read_text())
        for key in ("dataset", "training", "optimizers", "tuning"):
            if config[key] != tuning_config[key]:
                raise ValueError(f"Configuration changed since tuning: {key}")
    output = base / "sanity" if sanity else base / "final"
    settings = config["sanity"] if sanity else config["training"]
    seeds = [settings["seed"]] if sanity else settings["seeds"]
    rows = []
    for seed in seeds:
        initial = initial_weights(config, seed, output)
        for name in OPTIMIZERS:
            hp = settings if sanity else selection[name]
            result = run_training(
                config, name, seed, hp["learning_rate"], hp["weight_decay"],
                settings["epochs"], output / name / f"seed_{seed}", initial,
                sanity=sanity, test=not sanity)
            rows.append(result)
            save_csv(output / "final_results.csv", pd.DataFrame(rows))
    if not sanity:
        save_csv(base / "final_results.csv", pd.DataFrame(rows))
    from .evaluate import generate_report
    generate_report(config, sanity=sanity)

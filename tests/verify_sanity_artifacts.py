"""Audit completed real-data sanity outputs; never train or fabricate scores."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, project_path, save_json


def verify() -> dict:
    """Raise on inconsistent artifacts and save the observed audit outcome."""
    config = load_config()
    results = project_path(config["paths"]["results"])
    base = results / "sanity"
    plots = project_path(config["paths"]["plots"]) / "sanity"
    frame = pd.read_csv(base / "final_results.csv")
    expected = {("adamw", config["sanity"]["seed"]),
                ("lion", config["sanity"]["seed"])}
    assert len(frame) == 2 and set(zip(frame.optimizer, frame.seed)) == expected
    assert frame["initial_weights_sha256"].nunique() == 1
    checked = []
    for row in frame.itertuples():
        run = base / row.optimizer / f"seed_{row.seed}"
        history = pd.read_csv(run / "history.csv")
        summary = json.loads((run / "summary.json").read_text())
        assert len(history) == config["sanity"]["epochs"] == 2
        np.testing.assert_array_equal(history.epoch, [1, 2])
        required = ["learning_rate", "train_loss", "train_accuracy", "validation_loss",
                    "validation_accuracy", "epoch_time", "steps_per_second"]
        assert np.isfinite(history[required].to_numpy()).all()
        assert (history.epoch_time > 0).all()
        for metric in ("train_accuracy", "validation_accuracy"):
            assert history[metric].between(0, 100).all()
        assert "final_test_accuracy" not in summary
        assert "final_test_loss" not in summary
        assert summary["train_samples"] == config["sanity"]["train_limit"]
        assert summary["validation_samples"] == config["sanity"]["validation_limit"]
        assert math.isclose(summary["training_time_sec"], history.epoch_time.sum())
        assert math.isclose(summary["steps_per_sec"],
                            summary["training_steps"] / summary["training_time_sec"])
        assert math.isclose(summary["best_val_accuracy"], history.validation_accuracy.max())
        expected_steps = math.ceil(summary["train_samples"]
                                   / config["training"]["batch_size"]) * 2
        assert summary["training_steps"] == expected_steps
        np.testing.assert_allclose(history.learning_rate,
                                   [config["sanity"]["learning_rate"],
                                    (config["sanity"]["learning_rate"]
                                     + config["training"]["scheduler_min_lr"]) / 2])
        if summary["device"] == "cpu":
            assert summary["peak_gpu_memory"] == "unavailable"
            assert summary["peak_gpu_memory_mb"] is None
            assert pd.isna(row.peak_gpu_memory_mb)
        if history.validation_accuracy.max() < config["training"]["target_accuracy"]:
            assert summary["time_to_target_accuracy"] is None
        checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 2
        tensors = checkpoint["optimizer"]["state"]
        assert tensors
        expected_keys = ({"exp_avg"} if row.optimizer == "lion"
                         else {"exp_avg", "exp_avg_sq", "step"})
        assert all(set(state) == expected_keys for state in tensors.values())
        stored_bytes = sum(value.numel() * value.element_size()
                           for state in tensors.values() for value in state.values()
                           if isinstance(value, torch.Tensor))
        assert stored_bytes == summary["optimizer_state_bytes"]
        initial = torch.load(base / "initial_weights" / f"seed_{row.seed}.pt",
                             map_location="cpu", weights_only=True)
        digest = hashlib.sha256()
        for key, value in initial.items():
            digest.update(key.encode())
            digest.update(value.contiguous().numpy().tobytes())
        assert digest.hexdigest() == summary["initial_weights_sha256"]
        checked.append(dict(optimizer=row.optimizer,
                            loss_first_epoch=float(history.iloc[0].train_loss),
                            loss_last_epoch=float(history.iloc[-1].train_loss),
                            loss_decreased=bool(history.iloc[-1].train_loss
                                                < history.iloc[0].train_loss),
                            validation_accuracy=float(history.iloc[-1].validation_accuracy),
                            optimizer_state_mb=summary["optimizer_state_mb"],
                            training_time_sec=summary["training_time_sec"],
                            steps_per_sec=summary["steps_per_sec"],
                            device=summary["device"]))
    images = sorted(plots.glob("figure*.png"))
    assert len(images) == 7
    for image in images:
        with Image.open(image) as rendered:
            assert rendered.width > 0 and rendered.height > 0
            rendered.verify()
    stats = pd.read_csv(base / "summary_statistics.csv")
    assert stats["std"].isna().all()  # One sanity seed cannot define sample std.
    assert not (results / "final_results.csv").exists()
    outcome = dict(status="passed", real_dataset="CIFAR-10", epochs_per_optimizer=2,
                   measured_runs=checked, plots_verified=[p.name for p in images],
                   official_final_experiment_run=False,
                   gpu_test_available=bool(torch.cuda.is_available()))
    save_json(results / "validation" / "sanity_audit.json", outcome)
    return outcome


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))

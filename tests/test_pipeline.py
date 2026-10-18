"""Pipeline regression tests with isolated, explicitly synthetic test fixtures.

No fixtures or temporary results are published as experiment observations.
The separate sanity CLI is responsible for testing the real CIFAR-10 dataset.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.data import make_loaders
from src.evaluate import generate_report
from src.models import create_model
from src.optimizers import create_optimizer
from src.train import evaluate_model, run_training
from src.tuning import best_candidate, run_final, run_search
from src.utils import load_config, save_csv, save_json, select_device, set_seed


class FixtureCIFAR:
    """Small deterministic image fixture with CIFAR's dataset interface."""

    def __init__(self, root, train=True, download=False, transform=None):
        self.train = train
        self.transform = transform

    def __len__(self):
        return 100 if self.train else 20

    def __getitem__(self, index):
        image = np.random.default_rng(index).integers(
            0, 256, size=(32, 32, 3), dtype=np.uint8)
        return self.transform(Image.fromarray(image)), index % 10


def tiny_model() -> torch.nn.Module:
    """Use a small network to test training orchestration on CPU."""
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 10))


def fixture_loaders(config, seed, sanity=False, include_test=False):
    """Return deterministic loaders; never download a dataset in unit tests."""
    generator = torch.Generator().manual_seed(314)
    dataset = TensorDataset(torch.randn(17, 3, 2, 2, generator=generator),
                            torch.arange(17) % 10)
    loaders = {
        "train": DataLoader(dataset, batch_size=8, shuffle=True,
                            generator=torch.Generator().manual_seed(seed)),
        "validation": DataLoader(dataset, batch_size=8),
    }
    if include_test:
        loaders["test"] = DataLoader(dataset, batch_size=8)
    return loaders


class PipelineTests(unittest.TestCase):
    """Check reproducibility, metric semantics, orchestration, and plots."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".test_tmp_", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config()
        self.config["dataset"]["root"] = str(self.root / "data")
        self.config["paths"] = {"results": str(self.root / "results"),
                                "plots": str(self.root / "plots")}
        self.config["training"]["device"] = "cpu"
        self.config["training"]["num_workers"] = 0
        self.config["training"]["cpu_threads"] = 2
        self.config["training"]["batch_size"] = 8
        self.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.original_threads)

    def test_model_architecture_and_output(self) -> None:
        model = create_model().eval()
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertIsInstance(model.maxpool, torch.nn.Identity)
        with torch.inference_mode():
            self.assertEqual(model(torch.zeros(2, 3, 32, 32)).shape, (2, 10))

    def test_all_rngs_are_reproducible(self) -> None:
        def draw():
            return random.random(), np.random.rand(), torch.rand(4)
        set_seed(42)
        first = draw()
        set_seed(42)
        second = draw()
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    @patch("src.data.datasets.CIFAR10", FixtureCIFAR)
    def test_shared_split_no_overlap_and_no_test_by_default(self) -> None:
        first = make_loaders(self.config, 42)
        second = make_loaders(self.config, 123)
        self.assertEqual(set(first), {"train", "validation"})
        np.testing.assert_array_equal(first["train"].dataset.indices,
                                      second["train"].dataset.indices)
        train = set(first["train"].dataset.indices)
        val = set(first["validation"].dataset.indices)
        self.assertEqual((len(train), len(val)), (90, 10))
        self.assertFalse(train & val)
        self.assertEqual(train | val, set(range(100)))
        sample = first["validation"].dataset[0][0]
        torch.testing.assert_close(sample, first["validation"].dataset[0][0])
        transforms = first["validation"].dataset.dataset.transform.transforms
        self.assertEqual([type(t).__name__ for t in transforms],
                         ["ToTensor", "Normalize"])

    @patch("src.data.datasets.CIFAR10", FixtureCIFAR)
    def test_paired_shuffle_and_augmentation(self) -> None:
        batches = []
        for _ in range(2):
            set_seed(2026)
            loaders = make_loaders(self.config, 2026)
            batches.append(next(iter(loaders["train"])))
        for left, right in zip(*batches):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    @patch("src.data.datasets.CIFAR10", FixtureCIFAR)
    def test_corrupted_split_is_rejected(self) -> None:
        make_loaders(self.config, 42)
        path = next((self.root / "data").glob("split_*.npz"))
        np.savez(path, train=np.array([0]), validation=np.array([1]))
        with self.assertRaisesRegex(ValueError, "Persisted dataset split"):
            make_loaders(self.config, 42)

    def test_adamw_factory_uses_official_implementation(self) -> None:
        model = tiny_model()
        optimizer = create_optimizer("adamw", model, 1e-3, 1e-4, self.config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_evaluation_is_sample_weighted(self) -> None:
        model = torch.nn.Identity()
        logits = torch.tensor([[4., 0.], [0., 2.], [2., 0.]])
        labels = torch.tensor([0, 1, 1])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        loss, accuracy = evaluate_model(model, loader, torch.device("cpu"))
        expected = torch.nn.functional.cross_entropy(logits, labels).item()
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertAlmostEqual(accuracy, 200 / 3)

    def test_empty_evaluation_is_rejected(self) -> None:
        dataset = TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_model(torch.nn.Identity(), DataLoader(dataset), torch.device("cpu"))

    @patch("src.utils.torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_falls_back_to_cpu(self, available) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(select_device("cuda:0").type, "cpu")

    def test_best_candidate_uses_validation_and_fixed_tie_break(self) -> None:
        frame = pd.DataFrame([
            dict(best_val_accuracy=80., learning_rate=1e-3, weight_decay=1e-4),
            dict(best_val_accuracy=81., learning_rate=1e-4, weight_decay=1e-3),
            dict(best_val_accuracy=81., learning_rate=1e-4, weight_decay=1e-4),
        ])
        best = best_candidate(frame)
        self.assertEqual(best.learning_rate, 1e-4)
        self.assertEqual(best.weight_decay, 1e-4)

    @patch("src.train.make_loaders", side_effect=fixture_loaders)
    @patch("src.train.create_model", side_effect=tiny_model)
    def test_cpu_training_replay_checkpoint_and_report(self, model_factory, loaders) -> None:
        set_seed(42)
        initial = copy.deepcopy(tiny_model().state_dict())
        summaries = []
        histories = []
        base = self.root / "results" / "sanity"
        with contextlib.redirect_stdout(io.StringIO()):
            for name in ("adamw", "lion"):
                run = base / name / "seed_42"
                summary = run_training(self.config, name, 42, 1e-3, 1e-4,
                                       2, run, initial, sanity=True)
                summaries.append(summary)
                history = pd.read_csv(run / "history.csv")
                histories.append(history)
                self.assertEqual(len(history), 2)
                self.assertTrue(math.isnan(summary["peak_gpu_memory_mb"]))
                self.assertEqual(summary["peak_gpu_memory"], "unavailable")
                self.assertNotIn("final_test_accuracy", summary)
                self.assertEqual(summary["training_steps"], 6)
                self.assertAlmostEqual(summary["steps_per_sec"],
                                       6 / summary["training_time_sec"])
                self.assertAlmostEqual(history.iloc[1].learning_rate, 0.0005)
                stored = json.loads((run / "summary.json").read_text())
                self.assertIsNone(stored["peak_gpu_memory_mb"])
                checkpoint = torch.load(run / "checkpoint.pt", weights_only=True)
                self.assertEqual(checkpoint["epoch"], 2)
                loaded = tiny_model()
                loaded.load_state_dict(checkpoint["model"])
                replay = run_training(self.config, name, 42, 1e-3, 1e-4,
                                      2, self.root / "replay" / name, initial, sanity=True)
                replay_history = pd.read_csv(self.root / "replay" / name / "history.csv")
                fields = ["train_loss", "train_accuracy", "validation_loss",
                          "validation_accuracy", "learning_rate"]
                pd.testing.assert_frame_equal(history[fields], replay_history[fields])
                self.assertEqual(replay["initial_weights_sha256"],
                                 summary["initial_weights_sha256"])
                with self.assertRaises(FileExistsError):
                    run_training(self.config, name, 42, 1e-3, 1e-4, 2,
                                 run, initial, sanity=True)
            self.assertEqual(summaries[0]["initial_weights_sha256"],
                             summaries[1]["initial_weights_sha256"])
            self.assertGreater(summaries[0]["optimizer_state_bytes"],
                               summaries[1]["optimizer_state_bytes"])
            save_csv(base / "final_results.csv", pd.DataFrame(summaries))
            generate_report(self.config, sanity=True)
        images = list((self.root / "plots" / "sanity").glob("*.png"))
        self.assertEqual(len(images), 7)
        for image in images:
            with Image.open(image) as rendered:
                rendered.verify()
        statistics = pd.read_csv(base / "summary_statistics.csv")
        self.assertTrue(statistics["std"].isna().all())
        self.assertFalse((self.root / "results" / "final_results.csv").exists())

    @patch("src.tuning.initial_weights", return_value={})
    def test_full_search_budget_and_validation_selection(self, initial) -> None:
        # Mock scores are test fixtures only, stored in a temporary directory.
        def measured_fixture(config, name, seed, lr, wd, epochs, run_dir, weights):
            return dict(best_val_accuracy=60 + np.log10(lr) - np.log10(wd),
                        final_val_accuracy=50., final_val_loss=1.5,
                        training_time_sec=0.01)
        with patch("src.tuning.run_training", side_effect=measured_fixture) as runner:
            run_search(self.config, "lr")
            self.assertEqual(runner.call_count, 10)
            for name in ("adamw", "lion"):
                calls = [c for c in runner.call_args_list if c.args[1] == name]
                self.assertEqual([c.args[3] for c in calls],
                                 self.config["tuning"]["learning_rates"])
                self.assertTrue(all(not c.kwargs.get("test", False) for c in calls))
            runner.reset_mock()
            run_search(self.config, "weight_decay")
            self.assertEqual(runner.call_count, 8)
        frame = pd.read_csv(self.root / "results" / "tuning_results.csv")
        self.assertEqual(len(frame), 18)
        best = json.loads((self.root / "results" / "best_weight_decay.json").read_text())
        for name in ("adamw", "lion"):
            self.assertEqual(best[name], {"learning_rate": 0.001, "weight_decay": 1e-5})
        changed = copy.deepcopy(self.config)
        changed["training"]["batch_size"] += 1
        with self.assertRaisesRegex(ValueError, "Configuration changed"):
            run_search(changed, "weight_decay")

    def test_incomplete_final_report_is_rejected(self) -> None:
        save_csv(self.root / "results" / "final" / "final_results.csv",
                 pd.DataFrame([dict(optimizer="lion", seed=42)]))
        with self.assertRaisesRegex(ValueError, "every configured"):
            generate_report(self.config)

    @patch("src.train.make_loaders", side_effect=fixture_loaders)
    @patch("src.train.create_model", side_effect=tiny_model)
    @patch("src.tuning.initial_weights")
    def test_six_run_final_pipeline_on_tiny_test_fixture(
            self, weights, model_factory, loaders) -> None:
        # This is a functional test, not a CIFAR-10 comparison experiment.
        self.config["training"]["epochs"] = 1
        self.config["training"]["target_accuracy"] = 0.
        base = self.root / "results"
        selection = {name: {"learning_rate": 1e-3, "weight_decay": 1e-4}
                     for name in ("adamw", "lion")}
        save_json(base / "best_weight_decay.json", selection)
        save_json(base / "tuning_weight_decay_config.json", self.config)

        def paired_weights(config, seed, output):
            set_seed(seed)
            return copy.deepcopy(tiny_model().state_dict())

        weights.side_effect = paired_weights
        with contextlib.redirect_stdout(io.StringIO()):
            run_final(self.config)
        results = pd.read_csv(base / "final_results.csv")
        self.assertEqual(len(results), 6)
        self.assertTrue(results["final_test_accuracy"].notna().all())
        self.assertTrue(results["final_test_loss"].notna().all())
        self.assertTrue(results["time_to_target_accuracy"].notna().all())
        self.assertTrue(all(call.kwargs["include_test"] for call in loaders.call_args_list))
        for seed, group in results.groupby("seed"):
            self.assertEqual(group["initial_weights_sha256"].nunique(), 1)
        statistics = pd.read_csv(base / "summary_statistics.csv")
        for name in ("adamw", "lion"):
            values = results.loc[results.optimizer == name, "final_test_accuracy"]
            row = statistics[(statistics.optimizer == name)
                             & (statistics.metric == "final_test_accuracy")].iloc[0]
            self.assertAlmostEqual(row["mean"], values.mean())
            self.assertAlmostEqual(row["std"], values.std(ddof=1))
            self.assertEqual(row["available_seeds"], 3)
        self.assertEqual(len(list((self.root / "plots").glob("*.png"))), 7)


if __name__ == "__main__":
    unittest.main()

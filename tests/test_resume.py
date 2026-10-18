"""Check that restart orchestration only reuses complete, matching runs."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from src.tuning import ensure_training_output

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
SPEC = importlib.util.spec_from_file_location(
    "resume_experiment", ROOT / "experiments/resume_experiment.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResumeTests(unittest.TestCase):
    """Keep cached experiment observations tied to their original settings."""

    def test_published_results_reject_training_writes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            ensure_training_output(base)
            (base / "PUBLISHED_RESULTS.md").write_text("Published measurements")
            with self.assertRaisesRegex(ValueError, "published results"):
                ensure_training_output(base)

    def test_matching_and_interrupted_runs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            run = Path(directory)
            config = {"training": {"epochs": 10}}
            (run / "config.json").write_text(json.dumps(config))
            args = (run, config, "lion", 42, 0.001, 0.0001, 10)
            self.assertIsNone(MODULE.reusable_summary(*args))
            summary = dict(optimizer="lion", seed=42, best_learning_rate=0.001,
                           best_weight_decay=0.0001, epochs=10)
            (run / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                MODULE.reusable_summary(*args)
            (run / "checkpoint.pt").touch()
            self.assertEqual(MODULE.reusable_summary(*args), summary)
            with self.assertRaises(ValueError):
                MODULE.reusable_summary(run, config, "lion", 123, 0.001, 0.0001, 10)
            with self.assertRaises(ValueError):
                MODULE.reusable_summary(run, {}, "lion", 42, 0.001, 0.0001, 10)


if __name__ == "__main__":
    unittest.main()

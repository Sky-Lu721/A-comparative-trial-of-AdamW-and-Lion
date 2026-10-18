"""Aggregate saved observations and generate seven comparison figures."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .utils import load_config, project_path, save_csv

COLORS = {"adamw": "#3366aa", "lion": "#dd8844"}
LABELS = {"adamw": "AdamW", "lion": "Lion"}


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a readable figure and release its resources."""
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def curve_plot(history: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    """Plot seed means and sample standard deviations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, subset in history.groupby("optimizer"):
        grouped = subset.groupby("epoch")[metric].agg(["mean", "std"])
        x = grouped.index.to_numpy()
        mean = grouped["mean"].to_numpy()
        ax.plot(x, mean, label=LABELS[name], color=COLORS[name])
        if grouped["std"].notna().any():
            std = grouped["std"].to_numpy()
            ax.fill_between(x, mean - std, mean + std, color=COLORS[name], alpha=0.2)
    ax.set(xlabel="Epoch", ylabel=ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    save_figure(fig, path)


def bar_plot(results: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    """Show measured mean ± sample std; missing values never become zeros."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for index, name in enumerate(COLORS):
        values = pd.to_numeric(results.loc[results.optimizer == name, metric],
                               errors="coerce").dropna()
        total = (results.optimizer == name).sum()
        if len(values):
            error = values.std(ddof=1) if len(values) > 1 else None
            ax.bar(index, values.mean(), yerr=error, capsize=5, color=COLORS[name])
            if len(values) < total:
                ax.text(index, values.mean(), f"available {len(values)}/{total}",
                        ha="center", va="bottom")
        else:
            ax.text(index, 0.5, "unavailable", transform=ax.get_xaxis_transform(),
                    ha="center", va="center")
    ax.set_xticks([0, 1], ["AdamW", "Lion"])
    ax.set(xlim=(-0.6, 1.6), ylabel=ylabel)
    if pd.to_numeric(results[metric], errors="coerce").notna().sum() == 0:
        ax.set_yticks([])
    save_figure(fig, path)


def sensitivity_plot(base: Path, path: Path) -> None:
    """Use only stage-one validation results, with logarithmic LR scale."""
    fig, ax = plt.subplots(figsize=(7, 4))
    source = base / "tuning_lr.csv"
    if source.exists():
        frame = pd.read_csv(source)
        for name, group in frame.groupby("optimizer"):
            group = group.sort_values("learning_rate")
            ax.plot(group.learning_rate, group.best_val_accuracy, "o-",
                    label=LABELS[name], color=COLORS[name])
        ax.set_xscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "unavailable: LR search has not been run",
                ha="center", transform=ax.transAxes)
    ax.set(xlabel="Learning rate", ylabel="Best validation accuracy (%)")
    save_figure(fig, path)


def generate_report(config: dict[str, Any], sanity: bool = False) -> None:
    """Read completed artifacts; report means, sample std, and available counts."""
    base = project_path(config["paths"]["results"])
    run_base = base / "sanity" if sanity else base / "final"
    results = pd.read_csv(run_base / "final_results.csv")
    if not sanity:
        expected = {(name, seed) for name in COLORS for seed in config["training"]["seeds"]}
        observed = set(zip(results.optimizer, results.seed))
        if observed != expected or results.duplicated(["optimizer", "seed"]).any():
            raise ValueError("Final report requires every configured optimizer/seed exactly once")
    histories = []
    for row in results.itertuples():
        history = pd.read_csv(run_base / row.optimizer / f"seed_{row.seed}" / "history.csv")
        if len(history) != row.epochs:
            raise ValueError("Incomplete epoch history")
        history["optimizer"], history["seed"] = row.optimizer, row.seed
        histories.append(history)
    history = pd.concat(histories, ignore_index=True)
    plots = project_path(config["paths"]["plots"])
    plots = plots / "sanity" if sanity else plots
    plots.mkdir(parents=True, exist_ok=True)
    curve_plot(history, "train_loss", "Training loss", plots / "figure1_training_loss.png")
    curve_plot(history, "validation_accuracy", "Validation accuracy (%)",
               plots / "figure2_validation_accuracy.png")
    # Sanity never evaluates the official test set.
    if "final_test_accuracy" not in results:
        results["final_test_accuracy"] = np.nan
    metrics = {
        "final_test_accuracy": ("Test accuracy (%)", "figure3_test_accuracy.png"),
        "peak_gpu_memory_mb": ("Peak GPU allocation (MiB)", "figure4_gpu_memory.png"),
        "steps_per_sec": ("Training steps / second", "figure5_throughput.png"),
        "time_to_target_accuracy": (
            f"Training seconds to {config['training']['target_accuracy']:g}% validation",
            "figure7_time_to_target.png"),
    }
    for metric, (ylabel, filename) in metrics.items():
        bar_plot(results, metric, ylabel, plots / filename)
    sensitivity_plot(base if not sanity else run_base, plots / "figure6_lr_sensitivity.png")
    numerical = list(metrics) + ["training_time_sec", "best_val_accuracy",
                                "optimizer_state_mb"]
    if "final_test_loss" in results:
        numerical.append("final_test_loss")
    summary = []
    for name, group in results.groupby("optimizer"):
        for metric in numerical:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            summary.append(dict(optimizer=name, metric=metric,
                                mean=values.mean(), std=values.std(ddof=1),
                                available_seeds=len(values), total_seeds=len(group)))
    output = run_base if sanity else base
    save_csv(output / "summary_statistics.csv", pd.DataFrame(summary))
    lines = ["Measured results (sample standard deviation, ddof=1).",
             "Missing observations are unavailable; inspect available_seeds."]
    for row in summary:
        value = (f"{row['mean']:.4f} ± {row['std']:.4f}"
                 if row["available_seeds"] > 1 else
                 f"{row['mean']:.4f} (std unavailable)" if row["available_seeds"] else
                 "unavailable")
        lines.append(f"{row['optimizer']}: {row['metric']} = {value} "
                     f"({row['available_seeds']}/{row['total_seeds']} seeds)")
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    """Regenerate statistics and figures without training or test re-selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    generate_report(load_config(args.config), sanity=args.sanity)


if __name__ == "__main__":
    main()

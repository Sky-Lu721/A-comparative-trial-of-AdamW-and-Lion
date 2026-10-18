# Reproducibility and provenance

The published snapshot is `results/gpu_full/` with figures in `plots/gpu_full/`.
It contains actual CIFAR-10 measurements: 10 LR candidates, 8 WD candidates,
and 6 final runs. Each candidate has 10 epochs and each final run 50 epochs.
Sanity runs and interrupted attempts are not part of the published snapshot.

## Configuration files

- `configs/experiment_gpu.yaml` is the original experiment configuration.
- `results/gpu_full/tuning_lr_config.json`,
  `tuning_weight_decay_config.json`, and each run's `config.json` record
  the configuration as it existed during execution. These records remain unchanged.
- `configs/reproduce_gpu.yaml` keeps the original numerical settings while
  enabling initial dataset download and redirecting outputs to `runs/`.
- `configs/experiment_config.yaml` is a portable starting point with automatic
  CPU fallback and zero data-loader workers. It is not an exact match for
  the recorded GPU timing setup.

The original configuration's `cpu_threads: 4` applies only to CPU execution.
Recorded CUDA runs report 24 PyTorch CPU threads in their summaries. GPU data
loading uses two workers and persistent workers. Training uses FP32 without AMP.

## Recorded environment

| Component | Recorded value |
|---|---|
| Operating system | Windows |
| Python | 3.14.3 |
| PyTorch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Physical GPU memory | Approximately 8 GiB |
| NumPy | 2.4.4 |
| pandas | 3.0.2 |
| matplotlib | 3.10.8 |

See `environment-gpu.lock.txt` for the complete package record. Its CUDA wheel
versions require an appropriate official PyTorch index and compatible platform.
The repository's broad `requirements.txt` supports installation on other systems;
those systems are not promised identical numerical results or training speed.

## Artifact inventory

| Artifact | Purpose |
|---|---|
| `tuning_lr.csv` | All ten first-stage LR observations |
| `tuning_weight_decay.csv` | All eight second-stage WD observations |
| `tuning_results.csv` | Combined tuning table |
| `best_lr.json` | LR selection by validation accuracy |
| `best_weight_decay.json` | Final LR/WD selections |
| `final_results.csv` | Six final test and efficiency observations |
| `summary_statistics.csv` | Mean, sample SD, and available seed counts |
| `report.txt` | Human-readable statistics |
| `final/<optimizer>/seed_<seed>/history.csv` | Complete final epoch histories |
| `tuning/<stage>/<optimizer>/<candidate>/history.csv` | Tuning epoch histories |
| Run `summary.json` | Metrics, versions, and initialization hash |
| Run `config.json` | Frozen run configuration |
| `pipeline_status.json` | Successful completion marker |
| `final_audit.json` | Historical audit from the original local experiment |

The historical audit was run before checkpoint archival. The current
`python tools/audit_final.py` validates distributed tables, histories, paired
initialization hashes, timing arithmetic, sample statistics, and PNG signatures.
It does not require checkpoints and does not prove identical weights or
predictions beyond the measurements and hashes recorded by the original runs.

## Restart history and timing

The experiment paused after LR tuning for laptop relocation. An incomplete WD
attempt was archived and restarted. A later interrupted AdamW seed-123 attempt
also restarted from initial weights. Completed candidate/seed runs were reused.
The interrupted attempts do not contribute to published means or curves.

Wall-clock conditions changed between runs. Fixed run ordering, worker startup,
system load, power and thermal conditions, and the absence of dedicated timing
warmup limit interpretation of throughput differences. No telemetry isolates
each factor, so no single cause of the variation is established.

## What the repository can reproduce

Report generation needs only the published CSV files and plotting dependencies:

```bash
python -m src.evaluate --config configs/experiment_gpu.yaml
```

Retraining needs the downloadable dataset and a compatible PyTorch environment:

```bash
python experiments/run_all.py --config configs/reproduce_gpu.yaml
```

Weights, datasets and initial-state `.pt` files are intentionally excluded.
Consequently the published snapshot is not a training-resume checkpoint.
Local new runs save their own checkpoints and support recovery at complete-run
boundaries, not from the middle of an epoch or interrupted run.

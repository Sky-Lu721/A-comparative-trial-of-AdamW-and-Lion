# Published experiment snapshot

These are measured results from the original CIFAR-10 / ResNet-18 experiment.
This marker prevents training and recovery commands from writing to this
directory. Use `configs/reproduce_gpu.yaml` for new training outputs.

CSV/JSON histories, summaries and configurations support inspection and report
regeneration. Dataset archives, trained checkpoints, and initial-weight tensors
are intentionally excluded. The initialization hashes remain in run summaries.

Regenerate the summary report and figures without retraining:

```bash
python -m src.evaluate --config configs/experiment_gpu.yaml
```

See the repository README and `docs/REPRODUCIBILITY.md` for measurement definitions,
environment details, limitations, and restart history.

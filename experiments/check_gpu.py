"""Validate real CUDA arithmetic and both optimizers on CIFAR-sized ResNet-18."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import torchvision

import _common
from src.models import create_model
from src.optimizers import create_optimizer
from src.utils import load_config, save_json, set_seed


def check_gpu() -> dict:
    """Execute CUDA kernels and actual forward/backward/optimizer steps."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter")
    config = load_config()
    set_seed(42, deterministic=True)
    device = torch.device("cuda:0")
    inputs = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    expected = inputs @ inputs.T
    measured = (inputs.to(device) @ inputs.T.to(device)).cpu()
    torch.testing.assert_close(measured, expected)
    initial = copy.deepcopy(create_model().state_dict())
    images = torch.randn(4, 3, 32, 32, device=device)
    labels = torch.tensor([0, 1, 2, 3], device=device)
    observations = []
    for name in ("adamw", "lion"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = create_model().to(device)
        model.load_state_dict(initial)
        optimizer = create_optimizer(name, model, 1e-3, 1e-4, config)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite CUDA smoke-test loss")
        loss.backward()
        before = next(model.parameters()).detach().clone()
        optimizer.step()
        if torch.equal(before, next(model.parameters())):
            raise RuntimeError("CUDA optimizer step did not update the model")
        torch.cuda.synchronize(device)
        if name == "lion":
            if not all(set(state) == {"exp_avg"} for state in optimizer.state.values()):
                raise RuntimeError("Lion stores unexpected optimizer state")
        observations.append(dict(optimizer=name, status="passed",
                                 forward_backward_step=True,
                                 peak_allocated_mb=torch.cuda.max_memory_allocated(device)
                                 / 1024**2))
        del optimizer, model, loss, before
    result = dict(status="passed", executable=sys.executable,
                  torch=torch.__version__, torchvision=torchvision.__version__,
                  cuda_build=torch.version.cuda, cuda_available=True,
                  device_name=torch.cuda.get_device_name(device),
                  capability=list(torch.cuda.get_device_capability(device)),
                  matrix_arithmetic="passed", optimizer_checks=observations,
                  purpose="installation smoke test; not comparison measurements")
    save_json(Path(__file__).resolve().parents[1]
              / "runs/diagnostics/gpu_check.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(check_gpu(), indent=2))

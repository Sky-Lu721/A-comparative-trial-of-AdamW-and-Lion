"""Official AdamW and a one-momentum-tensor Lion implementation."""
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class Lion(Optimizer):
    """Sign update with decoupled weight decay and separate momentum betas."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-4,
                 betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0) -> None:
        if lr < 0 or weight_decay < 0:
            raise ValueError("lr and weight_decay must be nonnegative")
        if len(betas) != 2 or not all(0 <= b < 1 for b in betas):
            raise ValueError("betas must lie in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Apply sign direction before updating the persistent momentum."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse or parameter.is_complex():
                    raise RuntimeError("Lion supports dense, real tensors only")
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                momentum = state["exp_avg"]
                direction = momentum.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(direction, alpha=-group["lr"])
                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


def create_optimizer(name: str, model: torch.nn.Module, lr: float,
                     weight_decay: float, config: dict[str, Any]) -> Optimizer:
    """Use the same all-parameter decay policy for both optimizers."""
    options = config["optimizers"][name]
    betas = tuple(options["betas"])
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas,
                                 eps=options["eps"], weight_decay=weight_decay,
                                 foreach=False, fused=False)
    if name == "lion":
        return Lion(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

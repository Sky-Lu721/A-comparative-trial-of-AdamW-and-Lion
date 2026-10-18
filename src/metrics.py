"""Measured optimizer storage and convergence metrics."""
import math

import pandas as pd
import torch


def time_to_target_accuracy(history: pd.DataFrame, threshold: float) -> float:
    """Return cumulative fit time at the first qualifying validation epoch."""
    reached = history[history["validation_accuracy"] >= threshold]
    return float(reached.iloc[0]["cumulative_training_time"]) if len(reached) else math.nan


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Count actual persistent tensor storage, including AdamW scalar steps."""
    return sum(value.numel() * value.element_size()
               for state in optimizer.state.values() for value in state.values()
               if isinstance(value, torch.Tensor))

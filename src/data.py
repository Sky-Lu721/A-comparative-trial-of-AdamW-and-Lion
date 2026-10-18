"""Shared persisted splits with train-only random augmentation."""
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .utils import project_path, seed_worker


def make_loaders(config: dict[str, Any], seed: int,
                 sanity: bool = False, include_test: bool = False
                 ) -> dict[str, DataLoader]:
    """Rebuild identically seeded loaders for each paired run."""
    data_config = config["dataset"]
    root = project_path(data_config["root"])
    normalization = transforms.Normalize(data_config["mean"], data_config["std"])
    basic = transforms.Compose([transforms.ToTensor(), normalization])
    augmented = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalization,
    ])
    train_data = datasets.CIFAR10(root, train=True, download=data_config["download"],
                                 transform=augmented)
    val_data = datasets.CIFAR10(root, train=True, download=False, transform=basic)
    fraction = data_config["validation_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must lie in (0, 1)")
    permutation = np.random.default_rng(data_config["split_seed"]).permutation(len(train_data))
    count = int(len(train_data) * fraction)
    val_indices, train_indices = permutation[:count], permutation[count:]
    root.mkdir(parents=True, exist_ok=True)
    split_path = root / f"split_{data_config['split_seed']}_{count}.npz"
    if split_path.exists():
        with np.load(split_path) as stored:
            if not (np.array_equal(stored["train"], train_indices)
                    and np.array_equal(stored["validation"], val_indices)):
                raise ValueError("Persisted dataset split does not match configuration")
    else:
        np.savez(split_path, train=train_indices, validation=val_indices)
    if sanity:
        train_indices = train_indices[:config["sanity"]["train_limit"]]
        val_indices = val_indices[:config["sanity"]["validation_limit"]]
    subsets = {"train": Subset(train_data, train_indices),
               "validation": Subset(val_data, val_indices)}
    if include_test:
        subsets["test"] = datasets.CIFAR10(root, train=False,
                                          download=data_config["download"], transform=basic)
    return {
        name: DataLoader(
            dataset, batch_size=config["training"]["batch_size"],
            shuffle=name == "train", num_workers=config["training"]["num_workers"],
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed + offset),
            pin_memory=torch.cuda.is_available(), drop_last=False,
            persistent_workers=(config["training"]["num_workers"] > 0
                                and config["training"].get("persistent_workers", False)),
        ) for offset, (name, dataset) in enumerate(subsets.items())
    }

#!/usr/bin/env python3
"""Utility script to compare consecutive dataloader initializations.

The script mimics the data pipeline initialization from ``scripts/train.py`` and
collects a fixed number of batches from two freshly constructed dataloaders. It
then compares the collected batches to provide a quick signal about whether the
dataloader order appears random.
"""

from __future__ import annotations

import dataclasses
import logging

import jax
import numpy as np
import tyro

import openpi_cot.datasets.cot_data_loader as cot_data_loader
import openpi_cot.training.config as train_config


@dataclasses.dataclass
class Args:
    """Command line arguments for the randomness test."""

    config_name: str
    device: str | None = None
    split: str = "train"
    num_examples: int = 10
    shuffle: bool = True
    first_seed: int | None = None
    second_seed: int | None = None
    exp_name: str = "dataloader_randomness_test"


def _normalize_batch(batch):
    """Bring all leaves to host numpy arrays or python scalars for comparison."""

    def _to_numpy(value):
        if isinstance(value, jax.Array):
            return np.asarray(value)
        if isinstance(value, np.ndarray):
            return value
        return value

    return jax.tree_util.tree_map(_to_numpy, jax.device_get(batch))


def _trees_equal(left, right) -> bool:
    """Return True if both pytrees share the same structure and values."""
    try:
        comparison = jax.tree_util.tree_map(
            lambda x, y: np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y,
            left,
            right,
        )
    except ValueError:
        return False

    flat, _ = jax.tree_util.tree_flatten(comparison)
    return bool(all(flat))


def _collect_examples(loader, count: int):
    iterator = iter(loader)
    batches = []
    for _ in range(count):
        batches.append(_normalize_batch(next(iterator)))
    return batches


def _build_config(config_name: str, device: str | None, exp_name: str) -> train_config.TrainConfig:
    cfg = train_config.get_config(config_name, device=device)
    if getattr(cfg, "exp_name", None) in (None, "", tyro.MISSING):
        cfg = dataclasses.replace(cfg, exp_name=exp_name)
    return cfg


def main(args: Args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    config = _build_config(args.config_name, args.device, args.exp_name)

    first_seed = args.first_seed if args.first_seed is not None else config.seed
    second_seed = args.second_seed if args.second_seed is not None else first_seed

    logging.info(
        "Collecting %d batches from dataloader (split=%s, shuffle=%s, seed=%d)",
        args.num_examples,
        args.split,
        args.shuffle,
        first_seed,
    )
    first_loader = cot_data_loader.create_data_loader(
        config,
        shuffle=args.shuffle,
        split=args.split,
        seed=first_seed,
    )
    first_examples = _collect_examples(first_loader, args.num_examples)

    logging.info(
        "Collecting %d batches from second dataloader initialization (split=%s, shuffle=%s, seed=%d)",
        args.num_examples,
        args.split,
        args.shuffle,
        second_seed,
    )
    second_loader = cot_data_loader.create_data_loader(
        config,
        shuffle=args.shuffle,
        split=args.split,
        seed=second_seed,
    )
    second_examples = _collect_examples(second_loader, args.num_examples)

    comparisons = [_trees_equal(a, b) for a, b in zip(first_examples, second_examples, strict=True)]
    identical = sum(comparisons)
    logging.info(
        "Compared %d example batches: %d identical, %d different",
        args.num_examples,
        identical,
        args.num_examples - identical,
    )

    for idx, same in enumerate(comparisons):
        status = "IDENTICAL" if same else "DIFFERENT"
        logging.info("Example %d: %s", idx, status)

    if identical == args.num_examples:
        logging.warning(
            "All sampled batches are identical. If you expected randomness, "
            "ensure that shuffling is enabled and that the second seed differs."
        )
    else:
        logging.info("At least one batch differed between the two runs.")


if __name__ == "__main__":
    main(tyro.cli(Args))

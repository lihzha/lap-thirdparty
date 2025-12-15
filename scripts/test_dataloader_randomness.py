#!/usr/bin/env python3
"""Utility script to compare consecutive dataloader initializations.

The script mimics the data pipeline initialization from ``scripts/train.py`` and
collects a fixed number of batches from two freshly constructed dataloaders. It
then compares the collected batches to provide a quick signal about whether the
dataloader order appears random.
"""

from __future__ import annotations

import logging

import jax
import numpy as np

import openpi_cot.datasets.cot_data_loader as cot_data_loader
import openpi_cot.training.config as _config


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


def main(config: _config.TrainConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    first_seed = 123
    second_seed = 456
    num_examples = 20
    split = "train"
    shuffle = True

    logging.info(
        "Collecting %d batches from dataloader (split=%s, shuffle=%s, seed=%d)",
        num_examples,
        split,
        shuffle,
        first_seed,
    )
    first_loader = cot_data_loader.create_data_loader(
        config,
        shuffle=shuffle,
        split=split,
        seed=first_seed,
    )
    first_examples = _collect_examples(first_loader, num_examples)

    logging.info(
        "Collecting %d batches from second dataloader initialization (split=%s, shuffle=%s, seed=%d)",
        num_examples,
        split,
        shuffle,
        second_seed,
    )
    second_loader = cot_data_loader.create_data_loader(
        config,
        shuffle=shuffle,
        split=split,
        seed=second_seed,
    )
    second_examples = _collect_examples(second_loader, num_examples)

    comparisons = [_trees_equal(a, b) for a, b in zip(first_examples, second_examples, strict=True)]
    identical = sum(comparisons)
    logging.info(
        "Compared %d example batches: %d identical, %d different",
        num_examples,
        identical,
        num_examples - identical,
    )

    for idx, same in enumerate(comparisons):
        status = "IDENTICAL" if same else "DIFFERENT"
        logging.info("Example %d: %s", idx, status)

    if identical == num_examples:
        logging.warning(
            "All sampled batches are identical. If you expected randomness, "
            "ensure that shuffling is enabled and that the second seed differs."
        )
    else:
        logging.info("At least one batch differed between the two runs.")


if __name__ == "__main__":
    main(_config.cli())

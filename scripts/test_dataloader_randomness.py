#!/usr/bin/env python3
"""Compare two dataloader initializations to check for randomness."""

from __future__ import annotations

import logging
import os

import jax
import numpy as np

import openpi_cot.datasets.cot_data_loader as cot_data_loader
from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer
import openpi_cot.training.config as _config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import platform

import etils.epath as epath
from rail_tpu_utils import prevent_cross_region

import openpi_cot.training.mh_sharding as sharding


def _normalize_batch(batch):
    """Convert leaves to host numpy arrays for stable comparisons."""

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


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_tpu(config: _config.TrainConfig):
    def _is_tpu_runtime() -> bool:
        try:
            return any(d.platform == "tpu" for d in jax.devices())
        except Exception:
            return False

    if (
        ("v6" in config.name and config.fsdp_devices > 8)
        or ("v4" in config.name and config.fsdp_devices > 4)
        or ("v5" in config.name and config.fsdp_devices > 8)
    ):
        jax.distributed.initialize()
    if "local" in config.name:
        os.environ["CURL_CA_BUNDLE"] = (
            "/etc/pki/tls/certs/ca-bundle.crt"  # Ensure the CA bundle is set for SSL verification
        )

    data_dir = save_dir = config.data.rlds_data_dir
    cache_dir = os.environ.get("OPENPI_DATA_HOME", None)
    if _is_tpu_runtime() and (str(data_dir).startswith("gs://") or str(save_dir).startswith("gs://")):
        prevent_cross_region(data_dir, save_dir)
        if cache_dir is not None:
            prevent_cross_region(cache_dir, save_dir)
    # Determine effective FSDP devices for single-process GPU/CPU runs.
    process_count = getattr(jax, "process_count", lambda: 1)()
    local_devices = getattr(jax, "local_device_count", lambda: 1)()
    global_devices = getattr(jax, "device_count", lambda: local_devices)()
    logging.info(f"Local devices: {local_devices}, Global devices: {global_devices}, Process count: {process_count}")
    if process_count == 1:
        # Choose the largest divisor of available devices not exceeding configured fsdp_devices
        target = min(config.fsdp_devices, local_devices)
        effective_fsdp_devices = 1
        for d in range(target, 0, -1):
            if global_devices % d == 0:
                effective_fsdp_devices = d
                break
        if effective_fsdp_devices != config.fsdp_devices:
            logging.info(
                "Using fsdp_devices=%d for single-process run (available devices=%d)",
                effective_fsdp_devices,
                global_devices,
            )
    else:
        effective_fsdp_devices = config.fsdp_devices

    logging.info(f"Running on: {platform.node()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    return effective_fsdp_devices


def main(config: _config.TrainConfig):
    init_logging()
    effective_fsdp_devices = init_tpu(config)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # Human-readable mesh overview
    sharding.log_mesh_and_sharding_header(mesh, title="Device mesh")

    first_loader = cot_data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        split="train",
        seed=123,
        persistent_iterator=True,
    )
    first_examples = _collect_examples(first_loader, 10)

    second_loader = cot_data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        split="train",
        seed=123,
        persistent_iterator=True,
    )
    second_examples = _collect_examples(second_loader, 10)

    tok = PaligemmaCoTTokenizer(max_len=300)

    first_dataset_batches = [batch[0].tokenized_dataset_name for batch in first_examples]
    second_dataset_batches = [batch[0].tokenized_dataset_name for batch in second_examples]
    for i, (first_batch, second_batch) in enumerate(zip(first_dataset_batches, second_dataset_batches)):
        first_decoded = []
        for ids in first_batch:
            for _id in ids:
                first_decoded.append(tok.decode(_id))
        second_decoded = []
        for ids in second_batch:
            for _id in ids:
                second_decoded.append(tok.decode(_id))
        for j, (first_str, second_str) in enumerate(zip(first_decoded, second_decoded)):
            if first_str != second_str:
                logging.info("Difference found in batch %d, example %d:", i, j)
                logging.info("First:  %s", first_str)
                logging.info("Second: %s", second_str)

    comparisons = [_trees_equal(a, b) for a, b in zip(first_examples, second_examples, strict=True)]
    identical = sum(comparisons)
    logging.info(
        "Compared %d example batches: %d identical, %d different",
        50,
        identical,
        50 - identical,
    )

    for idx, same in enumerate(comparisons):
        status = "IDENTICAL" if same else "DIFFERENT"
        logging.info("Example %d: %s", idx, status)

    if identical == 50:
        logging.warning(
            "All sampled batches are identical. If you expected randomness, "
            "ensure that shuffling is enabled and that the second seed differs."
        )
    else:
        logging.info("At least one batch differed between the two runs.")


if __name__ == "__main__":
    main(_config.cli())

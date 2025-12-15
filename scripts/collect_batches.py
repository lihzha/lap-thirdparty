#!/usr/bin/env python3
"""Compare two dataloader initializations to check for randomness."""

from __future__ import annotations

from collections import Counter
import dataclasses
import datetime
import logging
import os

import jax
from jax.experimental import multihost_utils
import numpy as np

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional for offline runs
    wandb = None  # type: ignore[assignment]

import openpi_cot.datasets.cot_data_loader as cot_data_loader
from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer
import openpi_cot.training.config as _config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import platform

import etils.epath as epath
from rail_tpu_utils import prevent_cross_region

import openpi_cot.training.mh_sharding as sharding


def _decode_tokenized_names(tokenized_names, tokenizer: PaligemmaCoTTokenizer) -> list[str]:
    """Convert padded token IDs into dataset name strings."""
    if tokenized_names is None:
        return []

    tokenized_names = np.asarray(tokenized_names)
    if tokenized_names.ndim == 1:
        tokenized_names = tokenized_names[None, :]

    decoded = []
    for sample_tokens in tokenized_names:
        decoded.append(tokenizer.decode(sample_tokens))
    return decoded


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
    rewind_to_step: int | None = None,
):
    if not enabled:
        logging.info("wandb disabled via config; skipping remote logging.")
        return False

    if wandb is None:
        logging.warning("wandb requested but the package is not installed; skipping remote logging.")
        return False

    # Only initialize wandb in the main process
    if jax.process_index() != 0:
        wandb.init(mode="disabled")
        return False

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        logging.warning("Checkpoint directory %s does not exist; skipping wandb logging.", ckpt_dir)
        return False

    if resuming:
        run_id_path = ckpt_dir / "wandb_id.txt"
        if not run_id_path.exists():
            logging.warning("wandb resume requested but %s not found; starting a fresh run.", run_id_path)
            resuming = False
        else:
            run_id = run_id_path.read_text().strip()
            if rewind_to_step is not None:
                # Use wandb's rewind feature to resume from a specific step
                wandb.init(
                    resume_from=f"{run_id}?_step={rewind_to_step}",
                    project=config.project_name,
                )
            else:
                wandb.init(id=run_id, resume="must", project=config.project_name)
    if not resuming:
        wandb_mode = "online" if os.environ.get("WANDB_DISABLED", "false").lower() not in {"1", "true"} else "offline"
        run_name = f"collect-batches-{config.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            name=run_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            reinit=True,
            mode=wandb_mode,
        )
        if rewind_to_step is not None:
            logging.info("Requested rewind_to_step=%s but starting a new run; ignoring rewind.", rewind_to_step)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

    return True


def _visualize_dataset_distribution(
    dataset_batches: list[np.ndarray | None],
    tokenizer: PaligemmaCoTTokenizer,
    *,
    log_to_wandb: bool = False,
) -> None:
    """Log dataset distribution percentages per batch and overall."""
    if not dataset_batches:
        logging.warning("No dataset batches provided; skipping visualization.")
        return

    valid_batches = []
    batch_indices = []
    for idx, tokenized_names in enumerate(dataset_batches):
        if tokenized_names is None:
            logging.warning("Batch %d missing tokenized dataset names; skipping.", idx)
            continue
        decoded_names = _decode_tokenized_names(tokenized_names, tokenizer)

        if not decoded_names:
            continue

        valid_batches.append(decoded_names)
        batch_indices.append(idx)

    if not valid_batches:
        logging.warning("No valid dataset names found; visualization skipped.")
        return

    batch_counters = [Counter(batch_names) for batch_names in valid_batches]
    total_counts = Counter()
    for counter in batch_counters:
        total_counts.update(counter)

    unique_datasets = sorted(total_counts, key=total_counts.get, reverse=True)
    logging.info("Found %d unique datasets across %d batches.", len(unique_datasets), len(valid_batches))
    logging.info("Top datasets: %s", total_counts.most_common(10))

    def _percentage_map(counter: Counter[str]) -> tuple[float, dict[str, float]]:
        total = sum(counter.values())
        if total == 0:
            return 0.0, {}
        percentages = {name: (counter.get(name, 0) / total) * 100.0 for name in unique_datasets}
        return float(total), percentages

    wandb_table = None
    wandb_table_has_rows = False
    if log_to_wandb and wandb is not None and unique_datasets:
        table_columns = ["batch_index", *unique_datasets]
        wandb_table = wandb.Table(columns=table_columns)

    for idx, counter in zip(batch_indices, batch_counters):
        total, percentages = _percentage_map(counter)
        if total == 0:
            logging.info("Batch %d has no dataset samples.", idx)
            continue
        formatted = ", ".join(f"{name}: {pct:.2f}%%" for name, pct in percentages.items())
        logging.info("Batch %d dataset percentages: %s", idx, formatted)
        if wandb_table is not None:
            wandb_table.add_data(idx, *(percentages.get(name, 0.0) for name in unique_datasets))
            wandb_table_has_rows = True

    overall_total = sum(total_counts.values())
    if overall_total == 0:
        logging.warning("Total dataset count is zero; percentage logging skipped.")
        return

    overall_percentages = {name: (total_counts[name] / overall_total) * 100.0 for name in unique_datasets}
    formatted_overall = ", ".join(f"{name}: {pct:.2f}%%" for name, pct in overall_percentages.items())
    logging.info("Overall dataset percentages: %s", formatted_overall)

    if log_to_wandb and wandb is not None:
        wandb_data: dict[str, object] = {
            "dataset_distribution/overall_percentages": overall_percentages,
        }
        if wandb_table is not None and wandb_table_has_rows:
            wandb_data["dataset_distribution/batch_percentages"] = wandb_table
        wandb.log(wandb_data)


def _materialize_tokenized_dataset_name(
    tokenized_dataset_name: jax.Array | np.ndarray | None,
) -> np.ndarray | None:
    """Return a host numpy array for dataset names, gathering across hosts if needed."""
    if tokenized_dataset_name is None:
        return None
    if isinstance(tokenized_dataset_name, np.ndarray):
        return tokenized_dataset_name
    if not isinstance(tokenized_dataset_name, jax.Array):
        return np.asarray(tokenized_dataset_name)

    try:
        return np.asarray(tokenized_dataset_name)
    except RuntimeError as err:
        if "non-addressable" not in str(err).lower():
            raise
        logging.debug("Encountered non-addressable dataset-name array; gathering shards across hosts.")
        try:
            dtype = np.dtype(tokenized_dataset_name.dtype)
        except TypeError:
            first_shard = tokenized_dataset_name.addressable_shards[0] if tokenized_dataset_name.addressable_shards else None
            dtype = np.asarray(first_shard.data).dtype if first_shard is not None else np.float32
        host_buf = np.zeros(tokenized_dataset_name.shape, dtype=dtype)
        for shard in tokenized_dataset_name.addressable_shards:
            host_buf[shard.index] = np.asarray(shard.data)
        if jax.process_count() == 1:
            return host_buf
        reduced = multihost_utils.process_allreduce(jax.device_put(host_buf))
        return np.asarray(reduced)


def _collect_dataset_name_batches(loader, count: int) -> list[np.ndarray | None]:
    iterator = iter(loader)
    batches: list[np.ndarray | None] = []
    for _ in range(count):
        observation, _ = next(iterator)
        batches.append(_materialize_tokenized_dataset_name(observation.tokenized_dataset_name))
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
    init_wandb(
        config,
        resuming=False,
        enabled=config.wandb_enabled,
        rewind_to_step=getattr(config, "rewind_to_step", None),
    )

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # Human-readable mesh overview
    sharding.log_mesh_and_sharding_header(mesh, title="Device mesh")

    first_loader = cot_data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        split="train",
        seed=42,
        persistent_iterator=True,
    )
    dataset_batches = _collect_dataset_name_batches(first_loader, 1000)
    tok = PaligemmaCoTTokenizer(max_len=300)
    _visualize_dataset_distribution(dataset_batches, tok, log_to_wandb=True)


if __name__ == "__main__":
    main(_config.cli())

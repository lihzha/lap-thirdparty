#!/usr/bin/env python3
"""Compare two dataloader initializations to check for randomness."""

from __future__ import annotations

from collections import Counter
import dataclasses
import datetime
import logging
import os

import jax
import matplotlib
import numpy as np
import wandb

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

import openpi_cot.datasets.cot_data_loader as cot_data_loader
from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer
import openpi_cot.training.config as _config
import openpi_cot.training.utils as training_utils

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
    # Only initialize wandb in the main process
    if jax.process_index() != 0:
        wandb.init(mode="disabled")
        return False

    if resuming:
        resuming = False
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

    logging.info("wandb initialized (run_id=%s, run_name=%s)", wandb.run.id, wandb.run.name)
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
    per_batch_percentages: list[dict[str, float]] = []
    plot_batch_indices: list[int] = []
    if log_to_wandb and wandb is not None and unique_datasets and jax.process_index() == 0:
        table_columns = ["batch_index", *unique_datasets]
        wandb_table = wandb.Table(columns=table_columns)

    for idx, counter in zip(batch_indices, batch_counters):
        total, percentages = _percentage_map(counter)
        if total == 0:
            logging.info("Batch %d has no dataset samples.", idx)
            continue
        formatted = ", ".join(f"{name}: {pct:.2f}%%" for name, pct in percentages.items())
        logging.info("Batch %d dataset percentages: %s", idx, formatted)
        if wandb_table is not None and jax.process_index() == 0:
            wandb_table.add_data(idx, *(percentages.get(name, 0.0) for name in unique_datasets))
            wandb_table_has_rows = True
        per_batch_percentages.append(percentages)
        plot_batch_indices.append(idx)

    overall_total = sum(total_counts.values())
    if overall_total == 0:
        logging.warning("Total dataset count is zero; percentage logging skipped.")
        return

    overall_percentages = {name: (total_counts[name] / overall_total) * 100.0 for name in unique_datasets}
    formatted_overall = ", ".join(f"{name}: {pct:.2f}%%" for name, pct in overall_percentages.items())
    logging.info("Overall dataset percentages: %s", formatted_overall)

    def _create_line_plot() -> matplotlib.figure.Figure | None:
        if not plot_batch_indices or not per_batch_percentages:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        for dataset_name in unique_datasets:
            y_values = [batch_pct.get(dataset_name, 0.0) for batch_pct in per_batch_percentages]
            ax.plot(plot_batch_indices, y_values, marker="o", linewidth=1.5, label=dataset_name)
        ax.set_xlabel("Batch index")
        ax.set_ylabel("Dataset percentage")
        ax.set_ylim(0, 100)
        ax.set_title("Dataset percentage per batch")
        ax.grid(True, linestyle="--", alpha=0.3)
        if unique_datasets:
            ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
        fig.tight_layout()
        return fig

    if log_to_wandb and wandb is not None and jax.process_index() == 0:
        wandb_data: dict[str, object] = {
            "dataset_distribution/overall_percentages": overall_percentages,
        }
        if wandb_table is not None and wandb_table_has_rows:
            wandb_data["dataset_distribution/batch_percentages"] = wandb_table
        overall_table = wandb.Table(columns=["dataset", "percentage"])
        for dataset_name in unique_datasets:
            overall_table.add_data(dataset_name, overall_percentages.get(dataset_name, 0.0))
        wandb_data["dataset_distribution/overall_percentage_table"] = overall_table
        line_plot_figure = _create_line_plot()
        if line_plot_figure is not None:
            wandb_data["dataset_distribution/batch_percentage_plot"] = wandb.Image(line_plot_figure)
            plt.close(line_plot_figure)
        wandb.log(wandb_data)


def _normalize_batch(batch):
    """Convert leaves to host numpy arrays for stable comparisons."""

    return jax.tree_util.tree_map(training_utils.to_local_array, batch)


def _collect_dataset_name_batches(loader, count: int) -> list[np.ndarray | None]:
    iterator = iter(loader)
    batches: list[np.ndarray | None] = []
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
        or ("v5" in config.name and config.fsdp_devices > 4)
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
    wandb_enabled = init_wandb(
        config,
        resuming=False,
        enabled=True,
        rewind_to_step=getattr(config, "rewind_to_step", None),
    )
    logging.info("Effective FSDP devices: %d", effective_fsdp_devices)
    logging.info("wandb enabled: %s", wandb_enabled)

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
    batches = _collect_dataset_name_batches(first_loader, 100)
    dataset_batches = [batch[0].tokenized_dataset_name for batch in batches]
    tok = PaligemmaCoTTokenizer(max_len=300)
    _visualize_dataset_distribution(dataset_batches, tok, log_to_wandb=True)


if __name__ == "__main__":
    main(_config.cli())

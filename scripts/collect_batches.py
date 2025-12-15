#!/usr/bin/env python3
"""Compare two dataloader initializations to check for randomness."""

from __future__ import annotations

from collections import Counter
import dataclasses
import datetime
import logging
import os
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _log_plot_to_wandb(image_path: Path, key: str) -> None:
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    try:
        wandb.log({key: wandb.Image(str(image_path))})
    except Exception as err:  # pragma: no cover - best effort logging
        logging.warning("Failed to log %s to wandb: %s", key, err)


def _visualize_dataset_distribution(
    dataset_batches: list[tuple[np.ndarray | None, np.ndarray | None]],
    tokenizer: PaligemmaCoTTokenizer,
    *,
    log_to_wandb: bool = False,
) -> None:
    """Visualize dataset distributions per batch and across all batches."""
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

    per_batch_counts = np.asarray([[counter.get(name, 0) for name in unique_datasets] for counter in batch_counters])
    x = np.arange(len(valid_batches))
    fig_width = max(12, len(valid_batches) * 0.2)
    fig_height = max(6, len(unique_datasets) * 0.3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bottom = np.zeros(len(valid_batches))
    for dataset_name_idx, dataset_name in enumerate(unique_datasets):
        counts = per_batch_counts[:, dataset_name_idx]
        ax.bar(x, counts, bottom=bottom, label=dataset_name)
        bottom += counts

    ax.set_xlabel("Batch index")
    ax.set_ylabel("Samples per dataset")
    ax.set_title("Dataset distribution per batch")
    ax.set_xticks(x)
    ax.set_xticklabels([str(idx) for idx in batch_indices], rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    per_batch_path = Path(f"dataset_visualization_{timestamp}_per_batch.png")
    fig.savefig(per_batch_path, dpi=200)
    plt.close(fig)
    logging.info("Saved per-batch dataset distribution plot to %s", per_batch_path)
    if log_to_wandb:
        _log_plot_to_wandb(per_batch_path, "dataset_distribution/per_batch")

    fig2, ax2 = plt.subplots(figsize=(max(8, len(unique_datasets) * 0.5), 6))
    ax2.bar(unique_datasets, [total_counts[name] for name in unique_datasets])
    ax2.set_ylabel("Total samples")
    ax2.set_xlabel("Dataset")
    ax2.set_title("Overall dataset distribution across batches")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    fig2.tight_layout()
    overall_path = Path(f"dataset_visualization_{timestamp}_overall.png")
    fig2.savefig(overall_path, dpi=200)
    plt.close(fig2)
    logging.info("Saved overall dataset distribution plot to %s", overall_path)
    if log_to_wandb:
        _log_plot_to_wandb(overall_path, "dataset_distribution/overall")


def _normalize_batch(batch):
    """Convert leaves to host numpy arrays for stable comparisons."""

    def _to_numpy(value):
        if isinstance(value, jax.Array):
            return np.asarray(value)
        if isinstance(value, np.ndarray):
            return value
        return value

    return jax.tree_util.tree_map(_to_numpy, jax.device_get(batch))


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
    batches = _collect_examples(first_loader, 100)
    tok = PaligemmaCoTTokenizer(max_len=300)

    dataset_batches = [batch[0].tokenized_dataset_name for batch in batches]
    _visualize_dataset_distribution(dataset_batches, tok)


if __name__ == "__main__":
    main(_config.cli())

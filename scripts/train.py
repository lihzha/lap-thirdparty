import dataclasses
import datetime
import logging
import math
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for remote environments
import matplotlib.pyplot as plt
import numpy as np
from openpi.models import model as _model
from openpi.models.model import Observation
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import optax
import psutil
from rail_tpu_utils import prevent_cross_region
import tqdm_loggable.auto as tqdm
import wandb

import openpi_cot.dataloader.cot_data_loader as _data_loader
from openpi_cot.models.adapters.model_adapter import CoTObservation
from openpi_cot.models.adapters.tokenizer_adapter import PaligemmaCoTTokenizer
import openpi_cot.training.checkpoints as _checkpoints
import openpi_cot.training.config as _config
import openpi_cot.training.mh_sharding as sharding
import openpi_cot.training.utils as training_utils
import openpi_cot.training.vis_tools as vis_tools
import openpi_cot.training.weight_loaders as _weight_loaders


def vis_batch(batch, tok=None, step=None):
    """Visualize a training batch for debugging purposes.

    Args:
        batch: Tuple of (observation, actions)
        tok: Tokenizer for decoding tokenized prompts (optional)
        step: Training step number for wandb logging (optional)
    """
    obs = batch[0]
    actions = batch[1]

    logging.info("=" * 80)
    logging.info("BATCH VISUALIZATION")
    logging.info("=" * 80)

    # 1. Visualize images: print shape and log to wandb
    logging.info("\n--- IMAGES ---")
    wandb_images = {}
    for key, img in obs.images.items():
        logging.info(f"{key}: shape={img.shape}, dtype={img.dtype}, min={img.min():.3f}, max={img.max():.3f}")

        num_samples = img.shape[0]
        sample_images = []
        for t in range(min(num_samples, 4)):  # Log up to 4 samples
            sample_img = img[t, 0]  # [H, W, C]

            # Convert from [-1, 1] to [0, 255]
            sample_img_uint8 = ((sample_img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

            # Convert to numpy if it's a JAX array
            sample_img_uint8 = np.asarray(sample_img_uint8)

            # Add to wandb images list
            sample_images.append(wandb.Image(sample_img_uint8, caption=f"{key}_t{t}"))
            logging.info(
                f"  Prepared image [{key}] timestep {t} for wandb "
                f"(range: [{sample_img_uint8.min()}, {sample_img_uint8.max()}])"
            )

        if sample_images:
            wandb_images[f"batch_vis/{key}"] = sample_images

    # 2. Visualize image_masks: print shape
    logging.info("\n--- IMAGE MASKS ---")
    for key, mask in obs.image_masks.items():
        logging.info(f"{key}: shape={mask.shape}, dtype={mask.dtype}, true_count={mask.sum()}/{mask.size}")

    # 3. Visualize state: print shape and min/max for each dimension
    logging.info("\n--- STATE ---")
    state = obs.state
    logging.info(f"state: shape={state.shape}, dtype={state.dtype}")
    if len(state.shape) >= 2:
        for dim_idx in range(state.shape[-1]):
            dim_data = state[..., dim_idx]
            logging.info(
                f"  dim {dim_idx}: min={dim_data.min():.4f}, max={dim_data.max():.4f}, mean={dim_data.mean():.4f}"
            )

    # 4. Visualize tokenized_prompt with tokenizer
    logging.info("\n--- TOKENIZED PROMPTS ---")
    tokenized_prompt = obs.tokenized_prompt
    tokenized_prompt_mask = obs.tokenized_prompt_mask
    token_ar_mask = obs.token_ar_mask
    token_loss_mask = obs.token_loss_mask
    print(token_ar_mask, token_loss_mask)

    logging.info(f"tokenized_prompt: shape={tokenized_prompt.shape}, dtype={tokenized_prompt.dtype}")
    logging.info(f"tokenized_prompt_mask: shape={tokenized_prompt_mask.shape}, dtype={tokenized_prompt_mask.dtype}")
    # logging.info(f"token_ar_mask: shape={token_ar_mask.shape}, dtype={token_ar_mask.dtype}")
    # logging.info(f"token_loss_mask: shape={token_loss_mask.shape}, dtype={token_loss_mask.dtype}")

    if tok is not None:
        # Decode first sample in batch
        sample_idx = 0
        if tokenized_prompt.shape[0] > 0:
            # Full tokenized prompt
            tokens_full = tokenized_prompt[sample_idx]
            decoded_full = tok.decode(tokens_full)
            logging.info(f"\n[Sample {sample_idx}] Full tokenized_prompt:")
            logging.info(f"  Decoded: {decoded_full[:500]}...")  # First 500 chars

            # Tokenized prompt with prompt mask applied
            tokens_masked = tokenized_prompt[sample_idx] * tokenized_prompt_mask[sample_idx]
            decoded_masked = tok.decode(tokens_masked)
            logging.info(f"\n[Sample {sample_idx}] tokenized_prompt * tokenized_prompt_mask:")
            logging.info(f"  Decoded: {decoded_masked[:500]}...")

            # # Tokenized prompt with AR mask applied
            # tokens_ar = tokenized_prompt[sample_idx] * token_ar_mask[sample_idx]
            # decoded_ar = tok.decode(tokens_ar)
            # logging.info(f"\n[Sample {sample_idx}] tokenized_prompt * token_ar_mask:")
            # logging.info(f"  Decoded: {decoded_ar[:500]}...")
    else:
        logging.info("  (Tokenizer not provided - skipping decode)")

    # 5. Print token_loss_mask statistics
    # logging.info(f"\ntoken_loss_mask: sum={token_loss_mask.sum()}, mean={token_loss_mask.mean():.4f}")

    # 6. Visualize actions
    logging.info("\n--- ACTIONS ---")
    logging.info(f"actions: shape={actions.shape}, dtype={actions.dtype}")
    if len(actions.shape) >= 2:
        for dim_idx in range(actions.shape[-1]):
            dim_data = actions[..., dim_idx]
            logging.info(
                f"  dim {dim_idx}: min={dim_data.min():.4f}, max={dim_data.max():.4f}, mean={dim_data.mean():.4f}"
            )

    logging.info("=" * 80)

    # Log images to wandb
    if wandb_images and jax.process_index() == 0 and step is not None:
        wandb.log(wandb_images, step=step)
        logging.info(f"Logged {len(wandb_images)} image groups to wandb")


def log_mem(msg: str):
    """
    Returns:
        ram (float): Host RAM usage in GB (RSS).
        tpu_mem (float): TPU HBM memory usage in GB (sum across devices).
    """
    # --- Host RAM (Resident Set Size) ---
    proc = psutil.Process(os.getpid())
    ram_gb = proc.memory_info().rss / (1024**3)  # RSS in GB
    logging.info(f"{msg}: RAM: {ram_gb:.2f}GB")


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


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
    rewind_to_step: int | None = None,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    # Only initialize wandb in the main process
    if jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        if rewind_to_step is not None:
            # Use wandb's rewind feature to resume from a specific step
            wandb.init(
                resume_from=f"{run_id}?_step={rewind_to_step}",
                project=config.project_name,
            )
        else:
            wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


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


@dataclasses.dataclass
class HostBatchCache:
    host_batch: tuple[CoTObservation, _model.Actions] | None = None
    local_batch_size: int = 0
    step: int | None = None

    def ensure(
        self,
        *,
        step: int,
        batch: tuple[CoTObservation, _model.Actions],
    ) -> tuple[tuple[CoTObservation, _model.Actions] | None, int]:
        if self.step != step:
            self.host_batch = jax.tree.map(training_utils.to_local_array, batch)
            obs_local = self.host_batch[0] if self.host_batch else None
            self.local_batch_size = vis_tools.infer_local_batch_size(obs_local)
            self.step = step
        return self.host_batch, self.local_batch_size


class DatasetStatsTracker:
    """Tracks per-dataset loss, critical/number/direction token accuracy (micro-averaged), and example counts across training."""

    def __init__(self):
        # Use micro-averaging: track total correct tokens and total tokens per dataset
        self.dataset_stats = {}  # {dataset_name: {"total_loss": float, "count": int,
        #  "critical_correct": int, "critical_total": int,
        #  "number_correct": int, "number_total": int,
        #  "direction_correct": int, "direction_total": int}}
        # Track cumulative stats (never reset) - for since-start averages
        self.cumulative_stats = {}  # {dataset_name: {"total_loss": float, "count": int,
        #  "critical_correct": int, "critical_total": int,
        #  "number_correct": int, "number_total": int,
        #  "direction_correct": int, "direction_total": int}}

    def update(
        self,
        dataset_names: list[str],
        losses: np.ndarray,
        critical_token_data: tuple[np.ndarray, np.ndarray] | None = None,
        number_token_data: tuple[np.ndarray, np.ndarray] | None = None,
        direction_token_data: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        """Update statistics with new batch data using micro-averaging.

        Args:
            dataset_names: List of dataset names for each sample
            losses: Array of per-sample losses
            critical_token_data: Tuple of (correct_counts, total_counts) for critical tokens (optional)
            number_token_data: Tuple of (correct_counts, total_counts) for number tokens (optional)
            direction_token_data: Tuple of (correct_counts, total_counts) for direction tokens (optional)
        """
        for idx, name in enumerate(dataset_names):
            # Update current interval stats
            if name not in self.dataset_stats:
                self.dataset_stats[name] = {
                    "total_loss": 0.0,
                    "count": 0,
                    "critical_correct": 0,
                    "critical_total": 0,
                    "number_correct": 0,
                    "number_total": 0,
                    "direction_correct": 0,
                    "direction_total": 0,
                }
            self.dataset_stats[name]["total_loss"] += float(losses[idx])
            self.dataset_stats[name]["count"] += 1

            # Update cumulative stats (never reset) - for since-start averages
            if name not in self.cumulative_stats:
                self.cumulative_stats[name] = {
                    "total_loss": 0.0,
                    "count": 0,
                    "critical_correct": 0,
                    "critical_total": 0,
                    "number_correct": 0,
                    "number_total": 0,
                    "direction_correct": 0,
                    "direction_total": 0,
                }
            self.cumulative_stats[name]["total_loss"] += float(losses[idx])
            self.cumulative_stats[name]["count"] += 1

            # Micro-averaging: accumulate token-level counts (both interval and cumulative)
            if critical_token_data is not None:
                correct_counts, total_counts = critical_token_data
                self.dataset_stats[name]["critical_correct"] += int(correct_counts[idx])
                self.dataset_stats[name]["critical_total"] += int(total_counts[idx])
                self.cumulative_stats[name]["critical_correct"] += int(correct_counts[idx])
                self.cumulative_stats[name]["critical_total"] += int(total_counts[idx])
            if number_token_data is not None:
                correct_counts, total_counts = number_token_data
                self.dataset_stats[name]["number_correct"] += int(correct_counts[idx])
                self.dataset_stats[name]["number_total"] += int(total_counts[idx])
                self.cumulative_stats[name]["number_correct"] += int(correct_counts[idx])
                self.cumulative_stats[name]["number_total"] += int(total_counts[idx])
            if direction_token_data is not None:
                correct_counts, total_counts = direction_token_data
                self.dataset_stats[name]["direction_correct"] += int(correct_counts[idx])
                self.dataset_stats[name]["direction_total"] += int(total_counts[idx])
                self.cumulative_stats[name]["direction_correct"] += int(correct_counts[idx])
                self.cumulative_stats[name]["direction_total"] += int(total_counts[idx])

    def get_metrics(self) -> dict[str, float]:
        """Get current average losses and micro-averaged token accuracies for all datasets."""
        metrics = {}

        # Current interval metrics
        for dataset_name, stats in self.dataset_stats.items():
            if stats["count"] > 0:
                # Loss (macro-averaged per sample)
                avg_loss = stats["total_loss"] / stats["count"]
                metrics[f"dataset/{dataset_name}/avg_loss"] = avg_loss
                metrics[f"dataset/{dataset_name}/count"] = stats["count"]

                # Micro-averaged critical token accuracy
                if stats["critical_total"] > 0:
                    avg_critical_acc = stats["critical_correct"] / stats["critical_total"]
                    metrics[f"dataset/{dataset_name}/avg_critical_token_acc"] = avg_critical_acc
                    metrics[f"dataset/{dataset_name}/critical_token_count"] = stats["critical_total"]

                # Micro-averaged number token accuracy
                if stats["number_total"] > 0:
                    avg_number_acc = stats["number_correct"] / stats["number_total"]
                    metrics[f"dataset/{dataset_name}/avg_number_token_acc"] = avg_number_acc
                    metrics[f"dataset/{dataset_name}/number_token_count"] = stats["number_total"]

                # Micro-averaged direction token accuracy
                if stats["direction_total"] > 0:
                    avg_direction_acc = stats["direction_correct"] / stats["direction_total"]
                    metrics[f"dataset/{dataset_name}/avg_direction_token_acc"] = avg_direction_acc
                    metrics[f"dataset/{dataset_name}/direction_token_count"] = stats["direction_total"]

        # Cumulative (since-start) metrics
        for dataset_name, stats in self.cumulative_stats.items():
            if stats["count"] > 0:
                # Cumulative loss (macro-averaged per sample)
                cumulative_avg_loss = stats["total_loss"] / stats["count"]
                metrics[f"dataset/{dataset_name}/cumulative_avg_loss"] = cumulative_avg_loss
                metrics[f"dataset/{dataset_name}/cumulative_count"] = stats["count"]

                # Cumulative micro-averaged critical token accuracy
                if stats["critical_total"] > 0:
                    cumulative_avg_critical_acc = stats["critical_correct"] / stats["critical_total"]
                    metrics[f"dataset/{dataset_name}/cumulative_avg_critical_token_acc"] = cumulative_avg_critical_acc

                # Cumulative micro-averaged number token accuracy
                if stats["number_total"] > 0:
                    cumulative_avg_number_acc = stats["number_correct"] / stats["number_total"]
                    metrics[f"dataset/{dataset_name}/cumulative_avg_number_token_acc"] = cumulative_avg_number_acc

                # Cumulative micro-averaged direction token accuracy
                if stats["direction_total"] > 0:
                    cumulative_avg_direction_acc = stats["direction_correct"] / stats["direction_total"]
                    metrics[f"dataset/{dataset_name}/cumulative_avg_direction_token_acc"] = cumulative_avg_direction_acc

        return metrics

    def reset(self):
        """Reset accumulated statistics for the current log interval. Cumulative counts are preserved."""
        self.dataset_stats = {}


class LocalDatasetInfoBuffer:
    """Buffers local dataset info per step (without multihost gathering). Only gathers at log_interval."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Accumulate local tokenized names, losses, and token-level counts across steps
        self.tokenized_names_buffer = []  # List of arrays, one per step
        self.losses_buffer = []  # List of arrays, one per step
        # For micro-averaging: buffer (correct_count, total_count) tuples
        self.critical_correct_buffer = []  # List of arrays, one per step (may be empty)
        self.critical_total_buffer = []  # List of arrays, one per step (may be empty)
        self.number_correct_buffer = []  # List of arrays, one per step (may be empty)
        self.number_total_buffer = []  # List of arrays, one per step (may be empty)
        self.direction_correct_buffer = []  # List of arrays, one per step (may be empty)
        self.direction_total_buffer = []  # List of arrays, one per step (may be empty)

    def add_local_batch(
        self,
        tokenized_dataset_names: at.Array,
        per_sample_losses: at.Array,
        per_sample_critical_token_data: tuple[at.Array, at.Array] | None = None,
        per_sample_number_token_data: tuple[at.Array, at.Array] | None = None,
        per_sample_direction_token_data: tuple[at.Array, at.Array] | None = None,
    ):
        """Add local batch data (no multihost gathering here).

        Args:
            tokenized_dataset_names: Local tokenized dataset names
            per_sample_losses: Local per-sample losses
            per_sample_critical_token_data: Tuple of (correct_counts, total_counts) for critical tokens (optional)
            per_sample_number_token_data: Tuple of (correct_counts, total_counts) for number tokens (optional)
            per_sample_direction_token_data: Tuple of (correct_counts, total_counts) for direction tokens (optional)
        """
        # Convert to numpy and store
        self.tokenized_names_buffer.append(np.asarray(training_utils.to_local_array(tokenized_dataset_names)))
        self.losses_buffer.append(np.asarray(training_utils.to_local_array(per_sample_losses)))

        # Buffer token-level counts for micro-averaging
        if per_sample_critical_token_data is not None:
            correct_counts, total_counts = per_sample_critical_token_data
            self.critical_correct_buffer.append(np.asarray(training_utils.to_local_array(correct_counts)))
            self.critical_total_buffer.append(np.asarray(training_utils.to_local_array(total_counts)))
        if per_sample_number_token_data is not None:
            correct_counts, total_counts = per_sample_number_token_data
            self.number_correct_buffer.append(np.asarray(training_utils.to_local_array(correct_counts)))
            self.number_total_buffer.append(np.asarray(training_utils.to_local_array(total_counts)))
        if per_sample_direction_token_data is not None:
            correct_counts, total_counts = per_sample_direction_token_data
            self.direction_correct_buffer.append(np.asarray(training_utils.to_local_array(correct_counts)))
            self.direction_total_buffer.append(np.asarray(training_utils.to_local_array(total_counts)))

    def gather_and_update_stats(self, dataset_stats_tracker: DatasetStatsTracker) -> None:
        """Gather buffered data from all hosts and update dataset statistics.

        This should be called at log_interval to batch multihost communication.
        """
        if not self.tokenized_names_buffer:
            return

        # Concatenate all buffered local data
        all_local_names = np.concatenate(self.tokenized_names_buffer, axis=0)
        all_local_losses = np.concatenate(self.losses_buffer, axis=0)

        # Multihost gather
        process_count = jax.process_count()
        if process_count > 1:
            all_names = jax.experimental.multihost_utils.process_allgather(all_local_names)
            all_losses = jax.experimental.multihost_utils.process_allgather(all_local_losses)
            # Flatten: [num_processes, batch_per_process, seq_len] -> [total_batch, seq_len]
            all_names = np.asarray(all_names).reshape(-1, all_local_names.shape[-1])
            all_losses = np.asarray(all_losses).flatten()
        else:
            all_names = all_local_names
            all_losses = all_local_losses

        # Gather critical token counts if available
        all_critical_data = None
        if self.critical_correct_buffer and self.critical_total_buffer:
            all_local_critical_correct = np.concatenate(self.critical_correct_buffer, axis=0)
            all_local_critical_total = np.concatenate(self.critical_total_buffer, axis=0)
            if process_count > 1:
                all_critical_correct = jax.experimental.multihost_utils.process_allgather(all_local_critical_correct)
                all_critical_total = jax.experimental.multihost_utils.process_allgather(all_local_critical_total)
                all_critical_correct = np.asarray(all_critical_correct).flatten()
                all_critical_total = np.asarray(all_critical_total).flatten()
            else:
                all_critical_correct = all_local_critical_correct
                all_critical_total = all_local_critical_total
            all_critical_data = (all_critical_correct, all_critical_total)

        # Gather number token counts if available
        all_number_data = None
        if self.number_correct_buffer and self.number_total_buffer:
            all_local_number_correct = np.concatenate(self.number_correct_buffer, axis=0)
            all_local_number_total = np.concatenate(self.number_total_buffer, axis=0)
            if process_count > 1:
                all_number_correct = jax.experimental.multihost_utils.process_allgather(all_local_number_correct)
                all_number_total = jax.experimental.multihost_utils.process_allgather(all_local_number_total)
                all_number_correct = np.asarray(all_number_correct).flatten()
                all_number_total = np.asarray(all_number_total).flatten()
            else:
                all_number_correct = all_local_number_correct
                all_number_total = all_local_number_total
            all_number_data = (all_number_correct, all_number_total)

        # Gather direction token counts if available
        all_direction_data = None
        if self.direction_correct_buffer and self.direction_total_buffer:
            all_local_direction_correct = np.concatenate(self.direction_correct_buffer, axis=0)
            all_local_direction_total = np.concatenate(self.direction_total_buffer, axis=0)
            if process_count > 1:
                all_direction_correct = jax.experimental.multihost_utils.process_allgather(all_local_direction_correct)
                all_direction_total = jax.experimental.multihost_utils.process_allgather(all_local_direction_total)
                all_direction_correct = np.asarray(all_direction_correct).flatten()
                all_direction_total = np.asarray(all_direction_total).flatten()
            else:
                all_direction_correct = all_local_direction_correct
                all_direction_total = all_local_direction_total
            all_direction_data = (all_direction_correct, all_direction_total)

        # Decode gathered tokenized names
        decoded_names = []
        for i in range(all_names.shape[0]):
            try:
                name = self.tokenizer.decode(all_names[i])
                name = name.strip()
                decoded_names.append(name)
            except Exception as e:
                logging.warning(f"Failed to decode dataset name for sample {i}: {e}")
                decoded_names.append("unknown")

        # Update stats with token-level data
        dataset_stats_tracker.update(decoded_names, all_losses, all_critical_data, all_number_data, all_direction_data)

        # Clear buffers
        self.tokenized_names_buffer = []
        self.losses_buffer = []
        self.critical_correct_buffer = []
        self.critical_total_buffer = []
        self.number_correct_buffer = []
        self.number_total_buffer = []
        self.direction_correct_buffer = []
        self.direction_total_buffer = []


def gather_dataset_info_multihost(
    tokenized_dataset_names: at.Array,
    per_sample_losses: at.Array,
    tokenizer,
) -> tuple[list[str], np.ndarray]:
    """Gather dataset names and losses from all hosts.

    Args:
        tokenized_dataset_names: Tokenized dataset names [batch_size, seq_len]
        per_sample_losses: Per-sample losses [batch_size]
        tokenizer: Tokenizer to decode dataset names

    Returns:
        Tuple of (dataset_names, losses) where both are gathered from all hosts
    """
    # Convert to local arrays (process-specific)
    local_dataset_names_tokens = np.asarray(training_utils.to_local_array(tokenized_dataset_names))
    local_losses = np.asarray(training_utils.to_local_array(per_sample_losses))

    # For multi-host: gather tokenized names and losses (both numeric)
    process_count = jax.process_count()
    if process_count > 1:
        # Gather numeric arrays using JAX (works fine with int/float dtypes)
        all_dataset_names_tokens = jax.experimental.multihost_utils.process_allgather(local_dataset_names_tokens)
        all_losses = jax.experimental.multihost_utils.process_allgather(local_losses)

        # Flatten: shape goes from [num_processes, batch_per_process, seq_len] to [total_batch, seq_len]
        # For losses: [num_processes, batch_per_process] to [total_batch]
        all_dataset_names_tokens = np.asarray(all_dataset_names_tokens).reshape(
            -1, local_dataset_names_tokens.shape[-1]
        )
        all_losses = np.asarray(all_losses).flatten()
    else:
        all_dataset_names_tokens = local_dataset_names_tokens
        all_losses = local_losses

    # Now decode all the gathered tokenized names (on all hosts, but with same data)
    all_decoded_names = []
    for i in range(all_dataset_names_tokens.shape[0]):
        try:
            name = tokenizer.decode(all_dataset_names_tokens[i])
            # Clean up any special tokens or padding
            name = name.strip()
            all_decoded_names.append(name)
        except Exception as e:
            logging.warning(f"Failed to decode dataset name for sample {i}: {e}")
            all_decoded_names.append("unknown")

    return all_decoded_names, all_losses


def create_dataset_stats_plots(
    dataset_stats: dict[str, dict[str, float]],
    cumulative_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, plt.Figure]:
    """Create bar plots for dataset statistics.

    Args:
        dataset_stats: Dictionary mapping dataset names to {
            "total_loss": float, "count": int,
            "critical_correct": int, "critical_total": int,
            "number_correct": int, "number_total": int,
            "direction_correct": int, "direction_total": int
        }
        cumulative_stats: Optional dictionary with same structure for cumulative (since-start) metrics

    Returns:
        Dictionary with plots for interval and cumulative metrics
    """
    if not dataset_stats:
        return {}

    # Extract dataset names and statistics
    dataset_names = list(dataset_stats.keys())
    counts = [dataset_stats[name]["count"] for name in dataset_names]
    avg_losses = [
        dataset_stats[name]["total_loss"] / dataset_stats[name]["count"] if dataset_stats[name]["count"] > 0 else 0.0
        for name in dataset_names
    ]
    # Micro-averaged accuracies: correct_tokens / total_tokens
    avg_critical_accs = [
        dataset_stats[name]["critical_correct"] / dataset_stats[name]["critical_total"]
        if dataset_stats[name]["critical_total"] > 0
        else 0.0
        for name in dataset_names
    ]
    avg_number_accs = [
        dataset_stats[name]["number_correct"] / dataset_stats[name]["number_total"]
        if dataset_stats[name]["number_total"] > 0
        else 0.0
        for name in dataset_names
    ]
    avg_direction_accs = [
        dataset_stats[name]["direction_correct"] / dataset_stats[name]["direction_total"]
        if dataset_stats[name]["direction_total"] > 0
        else 0.0
        for name in dataset_names
    ]

    # Sort by count (descending) for better visualization
    sorted_indices = np.argsort(counts)[::-1]
    dataset_names_sorted = [dataset_names[i] for i in sorted_indices]
    counts_sorted = [counts[i] for i in sorted_indices]
    avg_losses_sorted = [avg_losses[i] for i in sorted_indices]
    avg_critical_accs_sorted = [avg_critical_accs[i] for i in sorted_indices]
    avg_number_accs_sorted = [avg_number_accs[i] for i in sorted_indices]
    avg_direction_accs_sorted = [avg_direction_accs[i] for i in sorted_indices]

    plots = {}

    # Create counts bar plot
    fig_counts, ax_counts = plt.subplots(figsize=(12, 6))
    bars_counts = ax_counts.bar(range(len(dataset_names_sorted)), counts_sorted, color="steelblue")
    ax_counts.set_xlabel("Dataset", fontsize=12)
    ax_counts.set_ylabel("Number of Examples", fontsize=12)
    ax_counts.set_title("Dataset Example Counts", fontsize=14, fontweight="bold")
    ax_counts.set_xticks(range(len(dataset_names_sorted)))
    ax_counts.set_xticklabels(dataset_names_sorted, rotation=45, ha="right")
    ax_counts.grid(axis="y", alpha=0.3)

    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars_counts, counts_sorted)):
        height = bar.get_height()
        ax_counts.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plots["counts"] = fig_counts

    # Create cumulative (since-start) plots if cumulative_stats is provided
    if cumulative_stats:
        # Extract cumulative statistics (sorted by same order as interval plots)
        cumulative_counts_list = [cumulative_stats.get(name, {}).get("count", 0) for name in dataset_names_sorted]
        cumulative_avg_losses = [
            cumulative_stats[name]["total_loss"] / cumulative_stats[name]["count"]
            if name in cumulative_stats and cumulative_stats[name]["count"] > 0
            else 0.0
            for name in dataset_names_sorted
        ]
        cumulative_avg_critical_accs = [
            cumulative_stats[name]["critical_correct"] / cumulative_stats[name]["critical_total"]
            if name in cumulative_stats and cumulative_stats[name]["critical_total"] > 0
            else 0.0
            for name in dataset_names_sorted
        ]
        cumulative_avg_number_accs = [
            cumulative_stats[name]["number_correct"] / cumulative_stats[name]["number_total"]
            if name in cumulative_stats and cumulative_stats[name]["number_total"] > 0
            else 0.0
            for name in dataset_names_sorted
        ]
        cumulative_avg_direction_accs = [
            cumulative_stats[name]["direction_correct"] / cumulative_stats[name]["direction_total"]
            if name in cumulative_stats and cumulative_stats[name]["direction_total"] > 0
            else 0.0
            for name in dataset_names_sorted
        ]

        # Cumulative counts plot
        fig_cumulative, ax_cumulative = plt.subplots(figsize=(12, 6))
        bars_cumulative = ax_cumulative.bar(
            range(len(dataset_names_sorted)), cumulative_counts_list, color="mediumseagreen"
        )
        ax_cumulative.set_xlabel("Dataset", fontsize=12)
        ax_cumulative.set_ylabel("Cumulative Number of Examples (Since Start)", fontsize=12)
        ax_cumulative.set_title("Dataset Cumulative Example Counts (Since Start)", fontsize=14, fontweight="bold")
        ax_cumulative.set_xticks(range(len(dataset_names_sorted)))
        ax_cumulative.set_xticklabels(dataset_names_sorted, rotation=45, ha="right")
        ax_cumulative.grid(axis="y", alpha=0.3)

        for i, (bar, count) in enumerate(zip(bars_cumulative, cumulative_counts_list)):
            height = bar.get_height()
            ax_cumulative.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        plots["cumulative_counts"] = fig_cumulative

        # Cumulative average loss plot
        fig_cumulative_loss, ax_cumulative_loss = plt.subplots(figsize=(12, 6))
        bars_cumulative_loss = ax_cumulative_loss.bar(
            range(len(dataset_names_sorted)), cumulative_avg_losses, color="coral"
        )
        ax_cumulative_loss.set_xlabel("Dataset", fontsize=12)
        ax_cumulative_loss.set_ylabel("Cumulative Average Loss (Since Start)", fontsize=12)
        ax_cumulative_loss.set_title("Dataset Cumulative Average Loss (Since Start)", fontsize=14, fontweight="bold")
        ax_cumulative_loss.set_xticks(range(len(dataset_names_sorted)))
        ax_cumulative_loss.set_xticklabels(dataset_names_sorted, rotation=45, ha="right")
        ax_cumulative_loss.grid(axis="y", alpha=0.3)

        for i, (bar, loss) in enumerate(zip(bars_cumulative_loss, cumulative_avg_losses)):
            height = bar.get_height()
            ax_cumulative_loss.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{loss:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        plots["cumulative_avg_loss"] = fig_cumulative_loss

        # Cumulative critical token accuracy plot
        if any(acc > 0 for acc in cumulative_avg_critical_accs):
            fig_cumulative_crit, ax_cumulative_crit = plt.subplots(figsize=(12, 6))
            bars_cumulative_crit = ax_cumulative_crit.bar(
                range(len(dataset_names_sorted)), cumulative_avg_critical_accs, color="mediumseagreen"
            )
            ax_cumulative_crit.set_xlabel("Dataset", fontsize=12)
            ax_cumulative_crit.set_ylabel("Cumulative Avg Critical Token Accuracy (Since Start)", fontsize=12)
            ax_cumulative_crit.set_title(
                "Dataset Cumulative Average Critical Token Accuracy (Since Start)", fontsize=14, fontweight="bold"
            )
            ax_cumulative_crit.set_xticks(range(len(dataset_names_sorted)))
            ax_cumulative_crit.set_xticklabels(dataset_names_sorted, rotation=45, ha="right")
            ax_cumulative_crit.grid(axis="y", alpha=0.3)
            ax_cumulative_crit.set_ylim([0, 1])

            for i, (bar, acc) in enumerate(zip(bars_cumulative_crit, cumulative_avg_critical_accs)):
                height = bar.get_height()
                ax_cumulative_crit.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{acc:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            plt.tight_layout()
            plots["cumulative_avg_critical_token_acc"] = fig_cumulative_crit

        # Cumulative number token accuracy plot
        if any(acc > 0 for acc in cumulative_avg_number_accs):
            fig_cumulative_num, ax_cumulative_num = plt.subplots(figsize=(12, 6))
            bars_cumulative_num = ax_cumulative_num.bar(
                range(len(dataset_names_sorted)), cumulative_avg_number_accs, color="skyblue"
            )
            ax_cumulative_num.set_xlabel("Dataset", fontsize=12)
            ax_cumulative_num.set_ylabel("Cumulative Avg Number Token Accuracy (Since Start)", fontsize=12)
            ax_cumulative_num.set_title(
                "Dataset Cumulative Average Number Token Accuracy (Since Start)", fontsize=14, fontweight="bold"
            )
            ax_cumulative_num.set_xticks(range(len(dataset_names_sorted)))
            ax_cumulative_num.set_xticklabels(dataset_names_sorted, rotation=45, ha="right")
            ax_cumulative_num.grid(axis="y", alpha=0.3)
            ax_cumulative_num.set_ylim([0, 1])

            for i, (bar, acc) in enumerate(zip(bars_cumulative_num, cumulative_avg_number_accs)):
                height = bar.get_height()
                ax_cumulative_num.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{acc:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            plt.tight_layout()
            plots["cumulative_avg_number_token_acc"] = fig_cumulative_num

        # Cumulative direction token accuracy plot
        if any(acc > 0 for acc in cumulative_avg_direction_accs):
            fig_cumulative_dir, ax_cumulative_dir = plt.subplots(figsize=(12, 6))
            bars_cumulative_dir = ax_cumulative_dir.bar(
                range(len(dataset_names_sorted)), cumulative_avg_direction_accs, color="lightcoral"
            )
            ax_cumulative_dir.set_xlabel("Dataset", fontsize=12)
            ax_cumulative_dir.set_ylabel("Cumulative Avg Direction Token Accuracy (Since Start)", fontsize=12)
            ax_cumulative_dir.set_title(
                "Dataset Cumulative Average Direction Token Accuracy (Since Start)", fontsize=14, fontweight="bold"
            )
            ax_cumulative_dir.set_xticks(range(len(dataset_names_sorted)))
            ax_cumulative_dir.set_xticklabels(dataset_names_sorted, rotation=45, ha="right")
            ax_cumulative_dir.grid(axis="y", alpha=0.3)
            ax_cumulative_dir.set_ylim([0, 1])

            for i, (bar, acc) in enumerate(zip(bars_cumulative_dir, cumulative_avg_direction_accs)):
                height = bar.get_height()
                ax_cumulative_dir.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{acc:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            plt.tight_layout()
            plots["cumulative_avg_direction_token_acc"] = fig_cumulative_dir

    # Create average loss bar plot
    fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
    bars_loss = ax_loss.bar(range(len(dataset_names_sorted)), avg_losses_sorted, color="coral")
    ax_loss.set_xlabel("Dataset", fontsize=12)
    ax_loss.set_ylabel("Average Loss", fontsize=12)
    ax_loss.set_title("Dataset Average Loss", fontsize=14, fontweight="bold")
    ax_loss.set_xticks(range(len(dataset_names_sorted)))
    ax_loss.set_xticklabels(dataset_names_sorted, rotation=45, ha="right")
    ax_loss.grid(axis="y", alpha=0.3)

    # Add value labels on top of bars
    for i, (bar, loss) in enumerate(zip(bars_loss, avg_losses_sorted)):
        height = bar.get_height()
        ax_loss.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{loss:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plots["avg_loss"] = fig_loss

    # Create average critical token accuracy bar plot (if data is available)
    if any(acc > 0 for acc in avg_critical_accs_sorted):
        fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
        bars_acc = ax_acc.bar(range(len(dataset_names_sorted)), avg_critical_accs_sorted, color="mediumseagreen")
        ax_acc.set_xlabel("Dataset", fontsize=12)
        ax_acc.set_ylabel("Average Critical Token Accuracy", fontsize=12)
        ax_acc.set_title("Dataset Average Critical Token Accuracy", fontsize=14, fontweight="bold")
        ax_acc.set_xticks(range(len(dataset_names_sorted)))
        ax_acc.set_xticklabels(dataset_names_sorted, rotation=45, ha="right")
        ax_acc.grid(axis="y", alpha=0.3)
        ax_acc.set_ylim([0, 1])  # Accuracy should be between 0 and 1

        # Add value labels on top of bars
        for i, (bar, acc) in enumerate(zip(bars_acc, avg_critical_accs_sorted)):
            height = bar.get_height()
            ax_acc.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{acc:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plots["avg_critical_token_acc"] = fig_acc

    # Create average number token accuracy bar plot (if data is available)
    if any(acc > 0 for acc in avg_number_accs_sorted):
        fig_num_acc, ax_num_acc = plt.subplots(figsize=(12, 6))
        bars_num_acc = ax_num_acc.bar(range(len(dataset_names_sorted)), avg_number_accs_sorted, color="skyblue")
        ax_num_acc.set_xlabel("Dataset", fontsize=12)
        ax_num_acc.set_ylabel("Average Number Token Accuracy", fontsize=12)
        ax_num_acc.set_title("Dataset Average Number Token Accuracy", fontsize=14, fontweight="bold")
        ax_num_acc.set_xticks(range(len(dataset_names_sorted)))
        ax_num_acc.set_xticklabels(dataset_names_sorted, rotation=45, ha="right")
        ax_num_acc.grid(axis="y", alpha=0.3)
        ax_num_acc.set_ylim([0, 1])  # Accuracy should be between 0 and 1

        # Add value labels on top of bars
        for i, (bar, acc) in enumerate(zip(bars_num_acc, avg_number_accs_sorted)):
            height = bar.get_height()
            ax_num_acc.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{acc:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plots["avg_number_token_acc"] = fig_num_acc

    # Create average direction token accuracy bar plot (if data is available)
    if any(acc > 0 for acc in avg_direction_accs_sorted):
        fig_dir_acc, ax_dir_acc = plt.subplots(figsize=(12, 6))
        bars_dir_acc = ax_dir_acc.bar(range(len(dataset_names_sorted)), avg_direction_accs_sorted, color="lightcoral")
        ax_dir_acc.set_xlabel("Dataset", fontsize=12)
        ax_dir_acc.set_ylabel("Average Direction Token Accuracy", fontsize=12)
        ax_dir_acc.set_title("Dataset Average Direction Token Accuracy", fontsize=14, fontweight="bold")
        ax_dir_acc.set_xticks(range(len(dataset_names_sorted)))
        ax_dir_acc.set_xticklabels(dataset_names_sorted, rotation=45, ha="right")
        ax_dir_acc.grid(axis="y", alpha=0.3)
        ax_dir_acc.set_ylim([0, 1])  # Accuracy should be between 0 and 1

        # Add value labels on top of bars
        for i, (bar, acc) in enumerate(zip(bars_dir_acc, avg_direction_accs_sorted)):
            height = bar.get_height()
            ax_dir_acc.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{acc:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plots["avg_direction_token_acc"] = fig_dir_acc

    return plots


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    del partial_params
    import gc

    gc.collect()

    return train_state, state_sharding


class TrainingStepRunner:
    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[CoTObservation | Observation, _model.Actions],
    ) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
        model = nnx.merge(state.model_def, state.params)
        model.train()

        # Determine stage configuration based on training schedule
        stage_config = None
        if self.config.training_schedule is not None:
            # Use JAX-compatible method that works with traced values
            stage_config = self.config.training_schedule.get_stage_config_for_step(state.step)

        @at.typecheck
        def loss_fn(
            model: _model.BaseModel,
            rng: at.KeyArrayLike,
            observation: CoTObservation | Observation,
            actions: _model.Actions,
        ):
            (
                per_sample_loss,
                token_accuracy,
                critical_token_accuracy,
                number_token_accuracy,
                direction_token_accuracy,
                metrics,
            ) = model.compute_loss(rng, observation, actions, train=True, stage_config=stage_config)
            return jnp.mean(per_sample_loss), (
                per_sample_loss,
                token_accuracy,
                critical_token_accuracy,
                number_token_accuracy,
                direction_token_accuracy,
                metrics,
            )

        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        diff_state = nnx.DiffState(0, self.config.trainable_filter)
        (
            (
                loss,
                (
                    per_sample_loss,
                    token_accuracy,
                    critical_token_accuracy,
                    number_token_accuracy,
                    direction_token_accuracy,
                    loss_metrics,
                ),
            ),
            grads,
        ) = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, observation, actions)

        params = state.params.filter(self.config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        new_params = nnx.state(model)

        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
        if state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_params,
                ),
            )

        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )

        info = {
            "loss": loss,
            "per_sample_loss": per_sample_loss,
            "grad_norm": optax.global_norm(grads),
            "param_norm": optax.global_norm(kernel_params),
            "token_accuracy": token_accuracy,
            "critical_token_accuracy": critical_token_accuracy,
            "number_token_accuracy": number_token_accuracy,
            "direction_token_accuracy": direction_token_accuracy,
        }

        # Add individual loss components from metrics
        for key, value in loss_metrics.items():
            if key.endswith("_loss"):
                # For loss components, compute mean
                info[key] = jnp.mean(value)
            else:
                # For accuracies, use as-is
                info[key] = value

        return new_state, info


class ValidationStepRunner:
    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[CoTObservation | Observation, _model.Actions],
    ) -> dict[str, at.Array]:
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        @at.typecheck
        def loss_fn(
            model: _model.BaseModel,
            rng: at.KeyArrayLike,
            observation: CoTObservation | Observation,
            actions: _model.Actions,
        ):
            (
                per_sample_loss,
                token_accuracy,
                critical_token_accuracy,
                number_token_accuracy,
                direction_token_accuracy,
                metrics,
            ) = model.compute_loss(rng, observation, actions, train=False)
            return (
                jnp.mean(per_sample_loss),
                token_accuracy,
                critical_token_accuracy,
                number_token_accuracy,
                direction_token_accuracy,
                metrics,
            )

        eval_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        (
            loss,
            token_accuracy,
            critical_token_accuracy,
            number_token_accuracy,
            direction_token_accuracy,
            val_metrics,
        ) = loss_fn(model, eval_rng, observation, actions)

        result = {
            "val_loss": loss,
            "val_token_accuracy": token_accuracy,
            "val_critical_token_accuracy": critical_token_accuracy,
            "val_number_token_accuracy": number_token_accuracy,
            "val_direction_token_accuracy": direction_token_accuracy,
        }

        # Add individual validation loss components from metrics
        for key, value in val_metrics.items():
            if key.endswith("_loss"):
                # For loss components, compute mean and prefix with val_
                result[f"val_{key}"] = jnp.mean(value)
            else:
                # For accuracies, prefix with val_
                result[f"val_{key}"] = value

        return result


def main(config: _config.TrainConfig):
    init_logging()
    effective_fsdp_devices = init_tpu(config)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # Human-readable mesh overview
    sharding.log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", sharding.format_sharding(data_sharding))
    logging.info("Replicated sharding spec: %s", sharding.format_sharding(replicated_sharding))

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        async_timeout_secs=config.checkpoint_async_timeout_secs,
        async_enable=config.checkpoint_async_enable,
    )
    init_wandb(
        config,
        resuming=resuming,
        enabled=config.wandb_enabled,
        rewind_to_step=getattr(config, "rewind_to_step", None),
    )

    # Log training start timestamp and preemption tracking info
    training_start_timestamp = datetime.datetime.now().isoformat()
    logging.info(f"Training started at: {training_start_timestamp}")
    logging.info(f"Resuming from checkpoint: {resuming}")

    # Log training schedule information if configured
    if config.training_schedule is not None:
        logging.info("=" * 80)
        logging.info("TRAINING SCHEDULE CONFIGURED")
        logging.info("=" * 80)
        for i, stage in enumerate(config.training_schedule.stages):
            end_str = f"step {stage.end_step}" if stage.end_step is not None else "end of training"
            logging.info(f"Stage {i}: steps {stage.start_step} -> {end_str}")
            logging.info(
                f"  Loss enables: langact={stage.enable_langact_training}, "
                f"action={stage.enable_action_training}, prediction={stage.enable_prediction_training}"
            )
            logging.info(
                f"  Loss weights: lang={stage.language_loss_weight:.2f}, "
                f"action={stage.action_loss_weight:.2f}, pred={stage.prediction_loss_weight:.2f}"
            )
            logging.info(
                f"  Loss probs: lang={stage.langact_prob:.2f}, "
                f"action={stage.action_prob:.2f}, pred={stage.prediction_prob:.2f}"
            )
        logging.info("=" * 80)
        # Validate schedule
        config.training_schedule.validate_for_training(config.num_train_steps)
    else:
        logging.info("No training schedule configured - using static model configuration")

    log_mem("Before init")
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
        persistent_iterator=True,
    )

    try:
        tok = data_loader.tokenizer
    except:
        tok = PaligemmaCoTTokenizer(max_len=200)

    # Initialize dataset log tracker for uniform sample logging across datasets
    dataset_log_tracker = vis_tools.DatasetLogTracker(tokenizer=tok)

    # # Initialize hard example tracker for logging difficult samples
    # hard_example_tracker = vis_tools.HardExampleTracker(
    #     tokenizer=tok,
    #     max_hard_examples=50,
    #     buffer_ratio=0.1,  # Maintain buffer of top candidates
    #     resize_hw=(128, 128),
    # )

    log_mem("Before init train state")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)

    log_mem("After init train state")

    logging.info(f"Initialized train state (param shapes):\n{training_utils.array_tree_to_info(train_state.params)}")
    sharding.log_param_sharding_planned(train_state_sharding)
    sharding.log_param_sharding_actual(train_state.params)

    # Restore checkpoint BEFORE creating iterator to ensure dataloader state is restored correctly
    dataloader_restored = False
    if resuming:
        try:
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
            dataloader_restored = True
            logging.info("Successfully restored checkpoint and dataloader state")
        except Exception as e:
            logging.error(f"Failed to restore dataloader state: {e}")
            dataloader_restored = False

    # Create iterator and get first batch AFTER restoring checkpoint to ensure iterator state is restored
    try:
        data_iter = iter(data_loader)
        log_mem("Before getting batch")
        batch = next(data_iter)
        vis_batch(batch, tok=tok)
        dataloader_initialized = True
        logging.info("Successfully initialized dataloader and retrieved first batch")
    except Exception as e:
        logging.error(f"Failed to initialize dataloader: {e}")
        dataloader_initialized = False
        raise  # Re-raise the exception as this is critical

    log_mem("After getting batch")
    logging.info(f"Initialized data loader (shapes):\n{training_utils.array_tree_to_info(batch)}")
    sharding.log_batch_sharding(batch)

    train_runner = TrainingStepRunner(config)
    ptrain_step = jax.jit(
        train_runner,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    if config.do_val:
        dataset = getattr(data_loader, "dataset", None)
        hash_tables = None
        if dataset:
            hash_tables = dataset.hash_tables
        val_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            split="val",
            max_samples=getattr(config.data, "val_max_samples", None),
            hash_tables=hash_tables,
            persistent_iterator=False,
        )
        # Try to obtain the tokenizer from the transform pipeline for decoding
        # tok = data_loader.tokenizer
        val_runner = ValidationStepRunner(config)
        pval_step = jax.jit(
            val_runner,
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        )

        # Jitted reasoning sampler returning only (id_buf, t)
        # def _sample_reasoning_ids_t(state: training_utils.TrainState, observation: CoTObservation):
        #     model_local = nnx.merge(state.model_def, state.params)
        #     id_buf, t, *_ = model_local.sample_reasoning(observation)
        #     return id_buf, t

        # psample_reasoning = jax.jit(
        #     _sample_reasoning_ids_t,
        #     # Expect observation replicated; return replicated outputs for consistent host access
        #     in_shardings=(train_state_sharding, replicated_sharding),
        #     out_shardings=(replicated_sharding, replicated_sharding),
        # )
        # Determine how many validation batches to evaluate each time.
        # If a fixed validation subset size is configured, compute batches from it;
        # otherwise fall back to a heuristic constant divided by global batch size.
        if getattr(config.data, "val_max_samples", None):
            # local batch size per host mirrors RLDS dataset batching
            process_count = getattr(jax, "process_count", lambda: 1)()
            local_bs = max(1, config.batch_size // process_count)
            num_val_batches = math.ceil(config.data.val_max_samples / local_bs)
        else:
            num_val_batches = int(60000 / config.batch_size)  # adjust if needed
    start_step = int(train_state.step)

    # Log preemption tracking information to wandb
    if jax.process_index() == 0:
        preemption_info = {
            "preemption/start_timestamp": training_start_timestamp,
            "preemption/is_resuming": float(resuming),
            "preemption/start_step": start_step,
            "preemption/dataloader_restored": float(dataloader_restored),
            "preemption/dataloader_initialized": float(dataloader_initialized),
        }
        wandb.log(preemption_info, step=start_step)
        logging.info(f"Logged preemption tracking info: {preemption_info}")

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        # disable=(jax.process_index() != 0),
    )

    infos = []
    host_batch_cache = HostBatchCache()
    dataset_stats_tracker = DatasetStatsTracker()
    dataset_info_buffer = LocalDatasetInfoBuffer(tok)

    # Track current training stage for transition detection
    current_stage_idx = None
    if config.training_schedule is not None:
        # Find the initial stage
        for i, stage in enumerate(config.training_schedule.stages):
            if stage.start_step <= start_step and (stage.end_step is None or start_step < stage.end_step):
                current_stage_idx = i
                logging.info(f"Starting training at step {start_step} in stage {i}")
                break

    for step in pbar:
        # Detect and log stage transitions
        if config.training_schedule is not None:
            for i, stage in enumerate(config.training_schedule.stages):
                if stage.start_step == step and i != current_stage_idx:
                    current_stage_idx = i
                    end_str = f"step {stage.end_step}" if stage.end_step is not None else "end of training"
                    logging.info("=" * 80)
                    logging.info(f"STAGE TRANSITION at step {step}: Entering Stage {i}")
                    logging.info(f"  Duration: steps {stage.start_step} -> {end_str}")
                    logging.info(
                        f"  Loss enables: langact={stage.enable_langact_training}, "
                        f"action={stage.enable_action_training}, prediction={stage.enable_prediction_training}"
                    )
                    logging.info(
                        f"  Loss weights: lang={stage.language_loss_weight:.2f}, "
                        f"action={stage.action_loss_weight:.2f}, pred={stage.prediction_loss_weight:.2f}"
                    )
                    logging.info(
                        f"  Loss probs: lang={stage.langact_prob:.2f}, "
                        f"action={stage.action_prob:.2f}, pred={stage.prediction_prob:.2f}"
                    )
                    logging.info("=" * 80)
                    if jax.process_index() == 0:
                        wandb.log(
                            {
                                "stage/current_stage": i,
                                "stage/langact_enabled": float(stage.enable_langact_training),
                                "stage/action_enabled": float(stage.enable_action_training),
                                "stage/prediction_enabled": float(stage.enable_prediction_training),
                                "stage/language_loss_weight": stage.language_loss_weight,
                                "stage/action_loss_weight": stage.action_loss_weight,
                                "stage/prediction_loss_weight": stage.prediction_loss_weight,
                            },
                            step=step,
                        )
                    break

        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        per_sample_loss = info.get("per_sample_loss")
        if per_sample_loss is None:
            raise ValueError("Training step info missing per_sample_loss")

        # Buffer local dataset info (no multihost gathering at every step)
        if hasattr(batch[0], "tokenized_dataset_name"):
            try:
                # Extract per-sample token counts for micro-averaging
                critical_data = None
                if "per_sample_critical_correct" in info and "per_sample_critical_total" in info:
                    critical_data = (info["per_sample_critical_correct"], info["per_sample_critical_total"])

                number_data = None
                if "per_sample_number_correct" in info and "per_sample_number_total" in info:
                    number_data = (info["per_sample_number_correct"], info["per_sample_number_total"])

                direction_data = None
                if "per_sample_direction_correct" in info and "per_sample_direction_total" in info:
                    direction_data = (info["per_sample_direction_correct"], info["per_sample_direction_total"])

                dataset_info_buffer.add_local_batch(
                    batch[0].tokenized_dataset_name,
                    per_sample_loss,
                    critical_data,
                    number_data,
                    direction_data,
                )
            except Exception as e:
                logging.warning(f"Failed to buffer dataset info at step {step}: {e}")

        # Update hard example tracker with per-sample losses from current batch
        # per_sample_loss_local = training_utils.to_local_array(per_sample_loss)
        # hard_example_tracker.update(per_sample_loss_local)

        # Add hard examples from current batch to buffer (if they qualify)
        # This needs to happen for every step, not just at log_interval
        # host_batch_local_curr, local_size_curr = host_batch_cache.ensure(step=step, batch=batch)
        # if per_sample_loss_local is not None and local_size_curr > 0:
        #     process_idx = jax.process_index()
        #     # Compute global index base for this process
        #     global_batch_idx = step * config.batch_size
        #     local_batch_offset = process_idx * local_size_curr

        # hard_example_tracker.add_local_examples(
        #     step_idx=step,
        #     host_batch_local=host_batch_local_curr,
        #     local_losses=per_sample_loss_local,
        #     global_idx_base=global_batch_idx + local_batch_offset,
        #     process_idx=process_idx,
        # )

        if step % config.log_interval == 0:
            # Gather and update dataset stats from buffered local data (batched at log_interval)
            dataset_info_buffer.gather_and_update_stats(dataset_stats_tracker)

            # infos appended above
            stacked_infos = common_utils.stack_forest(infos)
            reduce_overrides = {
                "grad_norm": jnp.mean,
                "loss": jnp.mean,
                "param_norm": jnp.mean,
            }
            reduced_info = {}
            per_sample_losses_chunk: list[np.ndarray] = []
            for key, value in stacked_infos.items():
                if key == "per_sample_loss":
                    per_sample_losses_chunk.append(np.asarray(training_utils.to_local_array(value)).reshape(-1))
                    reduced_info["max_per_sample_loss"] = jnp.max(value)
                elif key.startswith("per_sample_"):
                    # Skip per_sample_* metrics - they're only used for dataset-level statistics
                    continue
                else:
                    reduced_info[key] = reduce_overrides.get(key, jnp.mean)(value)
            reduced_info = jax.device_get(reduced_info)

            # Add dataset statistics to logging
            dataset_metrics = dataset_stats_tracker.get_metrics()
            reduced_info.update(dataset_metrics)

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if jax.process_index() == 0:
                wandb.log(reduced_info, step=step)

                # Create and log dataset statistics bar plots
                if dataset_stats_tracker.dataset_stats:
                    plots = create_dataset_stats_plots(
                        dataset_stats_tracker.dataset_stats, dataset_stats_tracker.cumulative_stats
                    )
                    if plots:
                        log_dict = {}
                        # Interval plots
                        if "counts" in plots:
                            log_dict["dataset_stats/interval_counts_plot"] = wandb.Image(plots["counts"])
                        if "avg_loss" in plots:
                            log_dict["dataset_stats/interval_avg_loss_plot"] = wandb.Image(plots["avg_loss"])
                        if "avg_critical_token_acc" in plots:
                            log_dict["dataset_stats/interval_avg_critical_token_acc_plot"] = wandb.Image(
                                plots["avg_critical_token_acc"]
                            )
                        if "avg_number_token_acc" in plots:
                            log_dict["dataset_stats/interval_avg_number_token_acc_plot"] = wandb.Image(
                                plots["avg_number_token_acc"]
                            )
                        if "avg_direction_token_acc" in plots:
                            log_dict["dataset_stats/interval_avg_direction_token_acc_plot"] = wandb.Image(
                                plots["avg_direction_token_acc"]
                            )

                        # Cumulative (since-start) plots
                        if "cumulative_counts" in plots:
                            log_dict["dataset_stats/cumulative_counts_plot"] = wandb.Image(plots["cumulative_counts"])
                        if "cumulative_avg_loss" in plots:
                            log_dict["dataset_stats/cumulative_avg_loss_plot"] = wandb.Image(plots["cumulative_avg_loss"])
                        if "cumulative_avg_critical_token_acc" in plots:
                            log_dict["dataset_stats/cumulative_avg_critical_token_acc_plot"] = wandb.Image(
                                plots["cumulative_avg_critical_token_acc"]
                            )
                        if "cumulative_avg_number_token_acc" in plots:
                            log_dict["dataset_stats/cumulative_avg_number_token_acc_plot"] = wandb.Image(
                                plots["cumulative_avg_number_token_acc"]
                            )
                        if "cumulative_avg_direction_token_acc" in plots:
                            log_dict["dataset_stats/cumulative_avg_direction_token_acc_plot"] = wandb.Image(
                                plots["cumulative_avg_direction_token_acc"]
                            )

                        wandb.log(log_dict, step=step)

                        # Close all figures to prevent memory leaks
                        for plot_fig in plots.values():
                            plt.close(plot_fig)

                if config.model.enable_langact_training:
                    host_batch_local, local_size = host_batch_cache.ensure(step=step, batch=batch)
                    vis_tools.log_random_examples(
                        step,
                        host_batch_local,
                        tok,
                        local_batch_size=local_size,
                        dataset_log_tracker=dataset_log_tracker,
                    )

                    # Log dataset logging statistics periodically
                    if step % (config.log_interval * 10) == 0:
                        log_stats = dataset_log_tracker.get_stats()
                        if log_stats:
                            logging.info(f"Dataset logging counts: {log_stats}")
                            # Log to wandb as well
                            wandb_log_stats = {f"dataset_log_count/{name}": count for name, count in log_stats.items()}
                            wandb.log(wandb_log_stats, step=step)

            # Log hard examples from the interval (multi-host gathering happens inside)
            # payload = hard_example_tracker.log_if_ready(step)
            # if payload is not None:
            #     vis_tools.log_hard_examples_payload(payload)

            infos = []
            # Reset dataset stats tracker to only track the next log_interval window
            dataset_stats_tracker.reset()
        # Periodic validation
        if config.do_val and step % getattr(config, "val_interval", 500) == 0:
            # use a pbar to track the validation progress
            val_pbar = tqdm.tqdm(
                range(num_val_batches),
                initial=0,
                total=num_val_batches,
                dynamic_ncols=True,
                disable=(jax.process_index() != 0),
            )
            with sharding.set_mesh(mesh):
                val_infos = []
                # Recreate a fresh iterator to ensure the same fixed validation subset each time.
                val_iter = iter(val_loader)
                for _ in val_pbar:
                    val_batch = next(val_iter)
                    val_info = pval_step(train_rng, train_state, val_batch)
                    # Convert to local array to avoid multi-host array issues during stacking
                    val_info_local = jax.device_get(val_info)
                    val_infos.append(val_info_local)

                stacked_val = common_utils.stack_forest(val_infos)
                reduced_val = jax.tree.map(jnp.mean, stacked_val)
                val_info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_val.items())
                val_pbar.write(f"Step {step} (val): {val_info_str}")
                if jax.process_index() == 0:
                    wandb.log(reduced_val, step=step)

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps:
            checkpoint_manager = _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                max_retries=config.checkpoint_max_retries,
                retry_delay_secs=config.checkpoint_retry_delay_secs,
                retry_backoff=config.checkpoint_retry_backoff,
                fallback_to_sync=config.checkpoint_fallback_to_sync,
                async_timeout_secs=config.checkpoint_async_timeout_secs,
                keep_period=config.keep_period,
            )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())

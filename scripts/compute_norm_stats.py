"""Compute normalization statistics for openpi-cot configs.

This script computes normalization statistics (mean, std, min, max, quantiles) for
state and action data in the dataset. These stats are used during training to normalize
inputs/outputs for better training stability.

Usage:
    # For Tiger demos (LeRobot dataset)
    python scripts/compute_norm_stats.py --config-name pi05_tiger_finetune_local

    # For DROID (RLDS dataset)
    python scripts/compute_norm_stats.py --config-name pi_droid_cot_local --max-frames 10_000_000

The computed statistics will be saved to:
    <assets_dir>/<asset_id>/norm_stats.json

For example, for Tiger demos with asset_id="tiger":
    /n/fs/robot-data/pi0-cot/assets/tiger/norm_stats.json
"""

import logging

import numpy as np
from tqdm import tqdm
import tyro

import openpi.models.model as _model
import openpi.training.data_loader as up_data_loader
import openpi.transforms as up_tf
from openpi_cot.shared.adapters import normalize_adapter
import openpi_cot.training.config as _config


def create_lerobot_dataloader(
    data_config: _config.CoTDataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[up_data_loader.Dataset, int]:
    """Create a LeRobot data loader for computing normalization stats."""
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # Create the base LeRobot dataset
    dataset = up_data_loader.create_torch_dataset(data_config, action_horizon, model_config)

    # Apply only the data transforms (not model transforms, as we only need raw state/actions)
    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        # Remove strings since they are not supported by JAX
        _RemoveStrings(),
    ]

    dataset = up_data_loader.TransformedDataset(dataset, transforms)

    # Determine number of batches
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    data_loader = up_data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    return data_loader, num_batches


class _RemoveStrings(up_tf.DataTransformFn):
    """Remove string fields from the dataset (not needed for norm stats)."""

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class RunningStats:
    """Accumulate statistics for normalization in a numerically stable way."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None  # Sum of squared differences from mean
        self.min = None
        self.max = None
        self.values_for_quantiles = []  # Store samples for quantile computation

    def update(self, data: np.ndarray):
        """Update statistics with a new batch of data.

        Uses Welford's online algorithm for numerical stability.
        """
        data = np.asarray(data, dtype=np.float64)  # Use float64 for precision

        # Flatten to (N, D) where D is the feature dimension
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            # Flatten all but last dimension
            data = data.reshape(-1, data.shape[-1])

        batch_size = data.shape[0]

        if self.mean is None:
            # First batch - initialize
            self.mean = np.zeros(data.shape[1], dtype=np.float64)
            self.m2 = np.zeros(data.shape[1], dtype=np.float64)
            self.min = np.full(data.shape[1], np.inf, dtype=np.float64)
            self.max = np.full(data.shape[1], -np.inf, dtype=np.float64)

        # Update count
        old_n = self.n
        self.n += batch_size

        # Update min/max
        self.min = np.minimum(self.min, data.min(axis=0))
        self.max = np.maximum(self.max, data.max(axis=0))

        # Update mean and M2 using Welford's algorithm
        for sample in data:
            delta = sample - self.mean
            self.mean += delta / self.n
            delta2 = sample - self.mean
            self.m2 += delta * delta2

        # Store some samples for quantile estimation (subsample to avoid OOM)
        if len(self.values_for_quantiles) < 100000:
            # Subsample if batch is large
            step = max(1, batch_size // 1000)
            self.values_for_quantiles.append(data[::step])

    def get_statistics(self) -> normalize_adapter.ExtendedNormStats:
        """Compute final statistics."""
        if self.n == 0:
            raise ValueError("No data has been added to RunningStats")

        mean = self.mean.astype(np.float32)
        variance = self.m2 / self.n if self.n > 1 else np.zeros_like(self.mean)
        std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)

        # Compute quantiles from stored samples
        all_values = np.concatenate(self.values_for_quantiles, axis=0)
        q01 = np.percentile(all_values, 1, axis=0).astype(np.float32)
        q99 = np.percentile(all_values, 99, axis=0).astype(np.float32)

        return normalize_adapter.ExtendedNormStats(
            mean=mean,
            std=std,
            q01=q01,
            q99=q99,
            min=self.min.astype(np.float32),
            max=self.max.astype(np.float32),
            num_transitions=int(self.n),
            num_trajectories=None,  # Could be computed if needed
        )


def main(config_name: str, max_frames: int | None = None, num_workers: int = 4):
    """Compute and save normalization statistics for a config.

    Args:
        config_name: Name of the training config (e.g., "pi05_tiger_finetune_local")
        max_frames: Maximum number of frames to use (default: use all data)
        num_workers: Number of workers for data loading (default: 4)
    """
    logging.basicConfig(level=logging.INFO)

    # Load config
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    logging.info(f"Computing normalization stats for config: {config_name}")
    logging.info(f"  repo_id: {data_config.repo_id}")
    logging.info(f"  asset_id: {data_config.asset_id}")

    # Check if using RLDS or LeRobot
    if data_config.rlds_data_dir is not None:
        raise NotImplementedError(
            "RLDS datasets handle normalization at the dataset level. "
            "Normalization stats are not needed for RLDS configs. "
            "If you need to compute stats for an RLDS dataset, please use the "
            "third_party/openpi/scripts/compute_norm_stats.py script instead."
        )

    # Create data loader (LeRobot)
    logging.info("Creating data loader...")
    data_loader, num_batches = create_lerobot_dataloader(
        data_config,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        num_workers,
        max_frames,
    )

    # Accumulate statistics
    keys = ["state", "actions"]
    stats = {key: RunningStats() for key in keys}

    logging.info(f"Computing statistics over {num_batches} batches...")
    for batch in tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            if key in batch:
                stats[key].update(np.asarray(batch[key]))

    # Get final statistics
    norm_stats = {key: stat.get_statistics() for key, stat in stats.items()}

    # Log statistics
    logging.info("\n" + "=" * 80)
    logging.info("COMPUTED NORMALIZATION STATISTICS")
    logging.info("=" * 80)

    for key in keys:
        if key not in norm_stats:
            continue
        s = norm_stats[key]
        logging.info(f"\n{key.upper()}:")
        logging.info(f"  Shape: {s.mean.shape}")
        logging.info(f"  Num transitions: {s.num_transitions}")
        logging.info(f"  Mean: {s.mean}")
        logging.info(f"  Std:  {s.std}")
        logging.info(f"  Min:  {s.min}")
        logging.info(f"  Max:  {s.max}")
        logging.info(f"  Q01:  {s.q01}")
        logging.info(f"  Q99:  {s.q99}")

        # Check for zero std dimensions
        zero_dims = np.where(s.std == 0)[0]
        if len(zero_dims) > 0:
            logging.warning(f"  ⚠️  Warning: {len(zero_dims)} dimensions have std=0: {zero_dims.tolist()}")

    # Save statistics
    output_path = config.assets_dirs / data_config.asset_id
    logging.info(f"\n{'=' * 80}")
    logging.info(f"Saving statistics to: {output_path}")
    logging.info(f"{'=' * 80}\n")

    normalize_adapter.save(str(output_path), norm_stats)

    logging.info("✓ Done! Normalization statistics have been computed and saved.")
    logging.info(f"  You can now run training with: python scripts/train.py {config_name}")


if __name__ == "__main__":
    tyro.cli(main)

# openpi_cot/training/cot_data_loader.py
import dataclasses
import logging
import os
from typing import Literal

import jax
import numpy as np
import openpi.models.model as _model
import openpi.training.data_loader as up  # upstream module
import openpi.transforms as up_tf
import tensorflow as tf

from openpi_cot.dataloader.cot_rlds_dataset import DroidCoTDataset
from openpi_cot.dataloader.cot_rlds_dataset import OXECoTDatasets
from openpi_cot.dataloader.helpers import NormalizationType
from openpi_cot.models.adapters.model_adapter import CoTObservation
from openpi_cot.models.adapters.tokenizer_adapter import PaligemmaCoTTokenizer
import openpi_cot.training.config as _config


def _create_rlds_dataset(
    data_cfg: _config.CoTDataConfig,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    enable_prediction_training: bool = False,
    *,
    shuffle: bool,
    seed: int,
    max_samples: int | None,
    split: str,
    hash_tables: dict | None = None,
) -> up.Dataset:
    # Per-host batching; avoids redundant slicing work in multi-process setups
    local_bsz = max(1, batch_size // jax.process_count())

    # Some configs may expose an optional `dataset_type` attribute.
    dataset_type = getattr(data_cfg, "dataset_type", "droid")
    rlds_data_dir = getattr(data_cfg, "rlds_data_dir", None)
    assert rlds_data_dir is not None, "rlds_data_dir is required"
    droid_required = hasattr(data_cfg, "language_action_dir") and data_cfg.language_action_dir is not None
    oxe_required = hasattr(data_cfg, "data_mix") and data_cfg.data_mix is not None
    dataset_cls = None

    if dataset_type == "droid":
        assert droid_required, "language_action_dir is required"
        dataset_cls = DroidCoTDataset
    if dataset_type == "oxe" or dataset_type == "combined":
        assert oxe_required, "data_mix is required"
        dataset_cls = OXECoTDatasets

    if dataset_cls is None:
        return up.create_rlds_dataset(data_cfg, action_horizon, local_bsz, shuffle=shuffle)

    # Build kwargs dynamically
    kwargs = dict(
        data_dir=rlds_data_dir,
        batch_size=local_bsz,
        shuffle=shuffle,
        max_samples=max_samples,
        seed=seed,
        config=data_cfg,
        split=split,
        action_horizon=action_horizon,
        action_dim=action_dim,
        hash_tables=hash_tables,
        standalone=True,
        action_proprio_normalization_type=NormalizationType.NORMAL,
        enable_prediction_training=enable_prediction_training,
    )

    return dataset_cls(**kwargs)


def _make_iterable_transforms(
    data_cfg: _config.CoTDataConfig,
    *,
    skip_norm_stats: bool,
    split: str | None,
) -> list[up_tf.DataTransformFn]:
    norm_stats = {}
    if data_cfg.repo_id != "fake" and not skip_norm_stats:
        if data_cfg.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_cfg.norm_stats

    if norm_stats is None:
        logging.info("Not using normalization stats in the cot_data_loader.")

    tx = [
        *data_cfg.repack_transforms.inputs,
        *data_cfg.data_transforms.inputs,
        up_tf.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
        *data_cfg.model_transforms.inputs,
    ]

    if split is not None and split != "train":
        new_tx = []
        for t in tx:
            if hasattr(t, "wrist_image_dropout_prob"):
                new_tx.append(dataclasses.replace(t, wrist_image_dropout_prob=0.0))
            else:
                new_tx.append(t)
        tx = new_tx

    return tx


class IterableTransformedDataset(up.IterableTransformedDataset):
    def __init__(self, batch_size, *args, persistent_iterator=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.persistent_iterator = persistent_iterator
        self._persistent_tf_iterator = None

    def get_or_create_tf_iterator(self):
        """Get or create the underlying TensorFlow iterator.

        This is used for checkpointing - we need to maintain a persistent TensorFlow
        iterator that can be saved/restored.
        """
        if self._persistent_tf_iterator is None:
            # Get the underlying dataset and create a TensorFlow iterator
            underlying_dataset = self._dataset
            if hasattr(underlying_dataset, "create_checkpointable_iterator"):
                self._persistent_tf_iterator = underlying_dataset.create_checkpointable_iterator()
            else:
                # Fallback: create iterator directly from dataset
                self._persistent_tf_iterator = iter(underlying_dataset.dataset)
        return self._persistent_tf_iterator

    def __iter__(self):
        if self.persistent_iterator:
            # Use persistent iterator for checkpointing support
            dataset_iter = self.get_or_create_tf_iterator()

            # TensorFlow iterator yields TF tensors, need to convert to numpy
            def to_numpy(x):
                if isinstance(x, tf.Tensor):
                    return x.numpy()
                return np.asarray(x) if hasattr(x, "__array__") else x
        else:
            # Regular behavior: create new iterator each time
            # This already yields numpy arrays via as_numpy_iterator()
            dataset_iter = iter(self._dataset)
            to_numpy = lambda x: x

        for sample in dataset_iter:
            # Convert sample from TF tensors to numpy if needed
            sample = jax.tree.map(to_numpy, sample)

            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(self.batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)


# ---------- Public entry: create_data_loader ----------
def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    seed: int = 0,
    max_samples: int | None = None,
    split: str = "train",
    framework: Literal["jax", "pytorch"] = "jax",
    hash_tables: dict | None = None,
    persistent_iterator: bool = False,
) -> up.DataLoader[tuple[CoTObservation, _model.Actions]]:
    # Avoid import-time side effects:
    # Only clear LEROBOT_HOME if we are about to construct a LeRobot dataset.
    if config.data.repo_id not in (None, "fake") and config.data.rlds_data_dir is None:
        os.environ.pop("LEROBOT_HOME", None)

    data_cfg: _config.CoTDataConfig = config.data.create(config.assets_dirs, config.model)
    logging.info("data_config: %s", data_cfg)

    # If RLDS, follow the RLDS path with our two hooks; else, fall back to upstream torch loader
    if data_cfg.rlds_data_dir is not None:
        if framework == "pytorch":
            raise NotImplementedError("PyTorch RLDS data loader is not supported yet")

        # 1) dataset
        ds = _create_rlds_dataset(
            data_cfg=data_cfg,
            batch_size=config.batch_size,
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
            enable_prediction_training=config.model.enable_prediction_training,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples if max_samples is not None else getattr(data_cfg, "max_samples", None),
            split=split,
            hash_tables=hash_tables,
        )

        # 2) transforms (split-aware)
        tx = _make_iterable_transforms(data_cfg, skip_norm_stats=data_cfg.norm_stats is None, split=split)
        iterable = IterableTransformedDataset(
            max(1, config.batch_size // jax.process_count()),
            ds,
            tx,
            is_batched=True,
            persistent_iterator=persistent_iterator,
        )

        return CoTRLDSDataLoader(
            iterable,
            sharding=sharding,
            num_batches=num_batches,
            data_cfg=data_cfg,
            persistent_iterator=persistent_iterator,
        )

    # Non-RLDS: delegate entirely to upstream (this will require torch if used)
    return up.create_torch_data_loader(
        data_cfg,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
        skip_norm_stats=data_cfg.norm_stats is None,
        framework=framework,
    )


class CoTRLDSDataLoader:
    """Iterates an IterableTransformedDataset and returns sharded jax.Arrays.

    If you run on multiple JAX processes (e.g. multi-host TPU), each process
    automatically receives its 1/`process_count` share of every batch.
    """

    def __init__(
        self,
        dataset: up.IterableTransformedDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        data_cfg: _config.CoTDataConfig,
        persistent_iterator: bool = False,
    ):
        self._dataset = dataset
        self._num_batches = num_batches
        self._data_cfg = data_cfg
        self._n_proc = jax.process_count()
        self._proc_idx = jax.process_index()
        self._persistent_iterator = persistent_iterator
        self._iterator = None
        self._checkpoint = None
        self._seen_batches = 0

        if sharding is None:
            sharding = jax.sharding.PositionalSharding(jax.local_devices())
        self._sharding = sharding

    def _to_device(self, batch):
        def put(x):
            if not (hasattr(x, "shape") and x.shape):
                return x
            # Skip strings/bytes/object arrays – JAX can't put them on device
            if hasattr(x, "dtype") and (x.dtype == np.object_ or getattr(x.dtype, "kind", None) in ("U", "S")):
                return x
            if isinstance(self._sharding, jax.sharding.NamedSharding):
                return jax.make_array_from_process_local_data(self._sharding, x)
            return jax.device_put(x, self._sharding)

        return jax.tree_util.tree_map(put, batch)

    def _assert_divisible(self, batch):
        sizes = [x.shape[0] for x in jax.tree_util.tree_leaves(batch) if hasattr(x, "shape") and x.shape]
        if not sizes:
            return
        b = max(sizes)  # this is per-host if dataset was shard(...)’ed

        if isinstance(self._sharding, jax.sharding.NamedSharding):
            mesh = self._sharding.mesh
            # DATA axis size across the whole mesh:
            data_axis_size = mesh.shape.get("data", None)  # or use your DATA_AXIS constant
            if data_axis_size is None:
                return  # no data axis; nothing to check

            # Special case: for cross-host FSDP when data_axis_size == 1,
            # we don't need data parallelism across hosts - each host gets the same data
            # and works together on FSDP sharding
            if data_axis_size == 1 and self._n_proc > 1:
                # Cross-host FSDP: validate against local device count instead
                ldc = jax.local_device_count()
                if b % ldc != 0:
                    raise ValueError(
                        f"Per-host batch {b} must be divisible by local_device_count {ldc} for cross-host FSDP"
                    )
                return

            # Standard data parallelism validation
            dp_per_host = data_axis_size // self._n_proc
            if dp_per_host == 0 or data_axis_size % self._n_proc != 0:
                raise ValueError("Mesh/data axis inconsistent with process_count.")
            if b % dp_per_host != 0:
                raise ValueError(f"Per-host batch {b} must be divisible by dp_per_host {dp_per_host}")
        else:
            # PositionalSharding fallback shards leading axis across local devices
            ldc = jax.local_device_count()
            if b % ldc != 0:
                raise ValueError(f"Per-host batch {b} must be divisible by local_device_count {ldc}")

    # ──────────────────────────────────────────────────────────────────────────
    def __iter__(self):
        seen = 0
        data_iter = iter(self._dataset)
        while True:
            if self._num_batches is not None and seen >= self._num_batches:
                return

            # Pull next preprocessed batch (may block on upstream I/O/TF)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self._dataset)
                continue

            self._assert_divisible(batch)
            batch = self._to_device(batch)
            seen += 1
            self._seen_batches += 1  # Track total batches seen for checkpointing
            yield CoTObservation.from_dict(batch), batch["actions"]

    def data_config(self) -> _config.CoTDataConfig:
        return self._data_cfg

    @property
    def dataset(self) -> up.Dataset:
        return self._dataset._dataset

    @property
    def tokenizer(self) -> PaligemmaCoTTokenizer:
        return self._dataset._transform.transforms[-2].tokenizer

    def get_norm_stats_for_checkpoint(self) -> tuple[dict | None, str]:
        """Get normalization statistics to save with checkpoint.

        Returns:
            tuple: (norm_stats dict, description string)
            - For OXE with global normalization: (global_statistics, "global")
            - For OXE without global or DROID: (dataset_statistics, "per-dataset")
            - For unknown/unsupported: (None, "none")
        """
        underlying_dataset = self._dataset._dataset

        # For OXE datasets, prefer global statistics if available
        if hasattr(underlying_dataset, "global_statistics"):
            if underlying_dataset.global_statistics is not None:
                return underlying_dataset.global_statistics, "global"

        # Fall back to per-dataset statistics
        if hasattr(underlying_dataset, "dataset_statistics"):
            stats = underlying_dataset.dataset_statistics
            if stats is not None:
                return stats, "per-dataset"

        return None, "none"

    def save_dataloader_state(self, checkpoint_dir: str) -> str:
        """Save the dataloader state including iterator position.

        Args:
            checkpoint_dir: Directory to save the checkpoint files.
                           Supports both local paths and GCS paths (gs://...).

        Returns:
            The checkpoint prefix path used for saving.

        Note:
            - This uses TensorFlow's tf.train.Checkpoint to save the iterator state
            - The checkpoint includes the iterator position and batch count
            - Requires persistent_iterator=True when creating the dataloader
            - Limitations: Cannot checkpoint iterators with external state (e.g., tf.py_function)
            - Checkpoint files may be large due to buffering from shuffle/prefetch operations
            - For GCS paths, ensure you have sufficient local temporary storage space

        Example:
            >>> loader.save_dataloader_state("./checkpoints/dataloader")
            './checkpoints/dataloader/ckpt-1'
            >>> loader.save_dataloader_state("gs://my-bucket/checkpoints/dataloader")
            'gs://my-bucket/checkpoints/dataloader/ckpt-1'
        """
        if not self._persistent_iterator:
            raise ValueError(
                "Dataloader must be created with persistent_iterator=True to support checkpointing. "
                "Please recreate the dataloader with this flag enabled."
            )

        # Use tf.io.gfile for GCS compatibility
        if not tf.io.gfile.exists(checkpoint_dir):
            tf.io.gfile.makedirs(checkpoint_dir)

        # Get the persistent TensorFlow iterator from the dataset
        if self._iterator is None:
            if hasattr(self._dataset, "get_or_create_tf_iterator"):
                self._iterator = self._dataset.get_or_create_tf_iterator()
            else:
                raise ValueError(
                    "Dataset does not support persistent iterators. "
                    "Please ensure you're using IterableTransformedDataset with persistent_iterator=True."
                )

        # Create checkpoint with iterator and batch counter
        step = tf.Variable(self._seen_batches, dtype=tf.int64, name="batch_counter")
        self._checkpoint = tf.train.Checkpoint(step=step, iterator=self._iterator)

        # Save the checkpoint
        checkpoint_prefix = tf.io.gfile.join(checkpoint_dir, "ckpt")
        save_path = self._checkpoint.save(checkpoint_prefix)
        logging.info(f"Saved dataloader state to {save_path} (batch {self._seen_batches})")

        return save_path

    def load_dataloader_state(self, checkpoint_dir: str) -> int:
        """Load the dataloader state from a checkpoint.

        Args:
            checkpoint_dir: Directory containing the checkpoint files.
                           Supports both local paths and GCS paths (gs://...).

        Returns:
            The number of batches that were seen when the checkpoint was saved.

        Raises:
            ValueError: If no checkpoint is found in the specified directory or
                       if persistent_iterator was not enabled.

        Note:
            - This restores the iterator to the exact position when saved
            - The dataloader will resume from where it left off
            - Must be called before starting iteration
            - Requires persistent_iterator=True when creating the dataloader

        Example:
            >>> batches_seen = loader.load_dataloader_state("./checkpoints/dataloader")
            >>> print(f"Resuming from batch {batches_seen}")
            >>> batches_seen = loader.load_dataloader_state("gs://my-bucket/checkpoints/dataloader")
            >>> print(f"Resuming from batch {batches_seen}")
        """
        if not self._persistent_iterator:
            raise ValueError(
                "Dataloader must be created with persistent_iterator=True to support checkpointing. "
                "Please recreate the dataloader with this flag enabled."
            )

        # Get the persistent TensorFlow iterator from the dataset
        if self._iterator is None:
            if hasattr(self._dataset, "get_or_create_tf_iterator"):
                self._iterator = self._dataset.get_or_create_tf_iterator()
            else:
                raise ValueError(
                    "Dataset does not support persistent iterators. "
                    "Please ensure you're using IterableTransformedDataset with persistent_iterator=True."
                )

        # Create checkpoint object
        step = tf.Variable(0, dtype=tf.int64, name="batch_counter")
        self._checkpoint = tf.train.Checkpoint(step=step, iterator=self._iterator)

        # Find the latest checkpoint
        latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_ckpt is None:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")

        # Restore the checkpoint
        status = self._checkpoint.restore(latest_ckpt)
        status.expect_partial()  # Suppress warnings about unmatched objects

        # Get the restored batch count
        self._seen_batches = int(step.numpy())
        logging.info(f"Restored dataloader state from {latest_ckpt} (batch {self._seen_batches})")

        return self._seen_batches

    def get_batches_seen(self) -> int:
        """Get the number of batches seen so far.

        Returns:
            The count of batches processed.
        """
        return self._seen_batches

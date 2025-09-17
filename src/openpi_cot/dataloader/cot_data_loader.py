# openpi_cot/training/cot_data_loader.py
import dataclasses
import logging
import os
from typing import Literal

import jax
import openpi.models.model as _model
import openpi.training.data_loader as up  # upstream module
import openpi.transforms as up_tf

from openpi_cot.dataloader.cot_rlds_dataset import CombinedCoTRldsDataset
from openpi_cot.dataloader.cot_rlds_dataset import DroidCoTRldsDataset
from openpi_cot.dataloader.cot_rlds_dataset import OXECoTRldsDatasets
from openpi_cot.models.adapters.model_adapter import CoTObservation
from openpi_cot.policies.combined_cot_policy import CombinedCoTInputs
from openpi_cot.policies.droid_cot_policy import DroidCoTInputs  # the one transform you condition on
import openpi_cot.training.config as _config


def _create_rlds_dataset(
    data_cfg: _config.CoTDataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool,
    split_seed: int,
    seed: int,
    max_samples: int | None,
    split: str,
) -> up.Dataset:
    # Per-host batching; avoids redundant slicing work in multi-process setups
    local_bsz = max(1, batch_size // jax.process_count())

    # Some configs may expose an optional `dataset_type` attribute.
    dataset_type = getattr(data_cfg, "dataset_type", "droid")

    if dataset_type == "droid":
        return DroidCoTRldsDataset(
            data_dir=data_cfg.rlds_data_dir,
            language_action_dir=data_cfg.language_action_dir,
            batch_size=local_bsz,
            shuffle=shuffle,
            action_chunk_size=action_horizon,
            action_space=data_cfg.action_space,
            shuffle_buffer_size=data_cfg.shuffle_buffer_size,
            split_seed=split_seed,
            max_samples=max_samples,
            seed=seed,
            config=data_cfg,
            split=split,
        )
    if dataset_type == "oxe":
        # OXE requires additional fields; only proceed if present, else fall back upstream.
        oxe_required = (
            hasattr(data_cfg, "data_mix") and data_cfg.data_mix is not None,
            hasattr(data_cfg, "resize_resolution") and data_cfg.resize_resolution is not None,
        )
        rlds_data_dir = getattr(data_cfg, "rlds_data_dir", None)
        if all(oxe_required) and rlds_data_dir is not None:
            return OXECoTRldsDatasets(
                config=data_cfg,
                rlds_data_dir=rlds_data_dir,
                data_mix=data_cfg.data_mix,
                resize_resolution=data_cfg.resize_resolution,
                action_chunk_size=action_horizon,
                batch_size=local_bsz,
                shuffle=shuffle,
                shuffle_buffer_size=data_cfg.shuffle_buffer_size,
                seed=seed,
                split=split,
                max_samples=max_samples,
                use_wrist_image=getattr(data_cfg, "use_wrist_image", False),
            )
        logging.warning(
            "dataset_type='oxe' selected but required fields missing; falling back to upstream RLDS loader."
        )
    if dataset_type == "combined":
        # Combined requires both DROID and OXE fields; validate and proceed if present.
        rlds_data_dir = getattr(data_cfg, "rlds_data_dir", None)
        has_droid = (
            getattr(data_cfg, "rlds_data_dir", None) is not None
            and getattr(data_cfg, "language_action_dir", None) is not None
        )
        has_oxe = (
            rlds_data_dir is not None
            and hasattr(data_cfg, "data_mix")
            and data_cfg.data_mix is not None
            and hasattr(data_cfg, "resize_resolution")
            and data_cfg.resize_resolution is not None
        )
        if has_droid and has_oxe:
            return CombinedCoTRldsDataset(
                # Top-level
                batch_size=local_bsz,
                shuffle=shuffle,
                shuffle_buffer_size=data_cfg.shuffle_buffer_size,
                max_samples=max_samples,
                seed=seed,
                split=split,
                use_wrist_image=getattr(data_cfg, "use_wrist_image", False),
                droid_weight=getattr(data_cfg, "droid_weight", 1.0),
                # DROID-specific (Raw)
                data_dir=data_cfg.rlds_data_dir,
                language_action_dir=data_cfg.language_action_dir,
                config=data_cfg,
                action_chunk_size=action_horizon,
                action_space=data_cfg.action_space,
                split_seed=split_seed,
                # OXE-specific (Raw)
                rlds_data_dir=rlds_data_dir,
                data_mix=data_cfg.data_mix,
                resize_resolution=data_cfg.resize_resolution,
            )
        logging.warning(
            "dataset_type='combined' selected but required fields missing; falling back to upstream RLDS loader."
        )

    return up.create_rlds_dataset(data_cfg, action_horizon, local_bsz, shuffle=shuffle)


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

    tx = [
        *data_cfg.repack_transforms.inputs,
        *data_cfg.data_transforms.inputs,
        up_tf.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
        *data_cfg.model_transforms.inputs,
    ]

    if split is not None and split != "train":
        new_tx = []
        for t in tx:
            if isinstance(t, (DroidCoTInputs, CombinedCoTInputs)):
                new_tx.append(dataclasses.replace(t, wrist_image_dropout_prob=0.0, text_state_dropout_prob=0.0))
            else:
                new_tx.append(t)
        tx = new_tx

    return tx


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
) -> up.DataLoader[tuple[CoTObservation, _model.Actions]]:
    # Avoid import-time side effects:
    # Only clear LEROBOT_HOME if we are about to construct a LeRobot dataset.
    if config.data.repo_id not in (None, "fake") and config.data.rlds_data_dir is None:
        os.environ.pop("LEROBOT_HOME", None)

    data_cfg = config.data.create(config.assets_dirs, config.model)
    logging.info("data_config: %s", data_cfg)

    # If RLDS, follow the RLDS path with our two hooks; else, fall back to upstream torch loader
    if data_cfg.rlds_data_dir is not None:
        if framework == "pytorch":
            raise NotImplementedError("PyTorch RLDS data loader is not supported yet")

        # 1) dataset
        ds = _create_rlds_dataset(
            data_cfg,
            config.model.action_horizon,
            config.batch_size,
            shuffle=shuffle,
            split_seed=seed,
            seed=seed,
            max_samples=max_samples if max_samples is not None else getattr(data_cfg, "max_samples", None),
            split=split,
        )

        # 2) transforms (split-aware)
        tx = _make_iterable_transforms(data_cfg, skip_norm_stats=data_cfg.norm_stats is None, split=split)
        iterable = up.IterableTransformedDataset(ds, tx, is_batched=True)

        return CoTRLDSDataLoader(iterable, sharding=sharding, num_batches=num_batches, data_cfg=data_cfg)

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
        skip_norm_stats=skip_norm_stats,
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
    ):
        self._dataset = dataset
        self._num_batches = num_batches
        self._data_cfg = data_cfg
        self._n_proc = jax.process_count()
        self._proc_idx = jax.process_index()

        if sharding is None:
            sharding = jax.sharding.PositionalSharding(jax.local_devices())
        self._sharding = sharding

    def _to_device(self, batch):
        def put(x):
            if not (hasattr(x, "shape") and x.shape):
                return x
            if isinstance(self._sharding, jax.sharding.NamedSharding):
                # Assemble a global jax.Array across processes.
                return jax.make_array_from_process_local_data(self._sharding, x)
            # Per-host sharding (PositionalSharding etc.).
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
            yield CoTObservation.from_dict(batch), batch["actions"]

    def data_config(self) -> _config.CoTDataConfig:
        return self._data_cfg

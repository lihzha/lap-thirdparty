from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import subprocess
import time
from typing import Protocol

from etils import epath
import jax
from openpi.shared import array_typing as at
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future
import tensorflow as tf

from openpi_cot.dataloader import cot_data_loader as _data_loader
from openpi_cot.shared.adapters import normalize_adapter as _normalize_adapter
from openpi_cot.training import utils as training_utils


def _delete_gcs_prefix_with_gsutil(uri: str) -> None:
    """Delete a GCS URI prefix using gsutil recursively.

    Raises a CalledProcessError if deletion fails.
    """
    cmd = ["gsutil", "-m", "rm", "-r", uri]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logging.info("gsutil deletion output for %s:\n%s", uri, result.stdout)


def _make_async_options(async_enable: bool, async_timeout_secs: int | None) -> ocp.AsyncOptions | None:
    if not async_enable:
        return None
    kwargs = {}
    if async_timeout_secs is not None:
        kwargs["timeout_secs"] = async_timeout_secs
    return ocp.AsyncOptions(**kwargs)


def create_checkpoint_manager(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    async_timeout_secs: int | None = 7200,
    async_enable: bool = True,
) -> ocp.CheckpointManager:
    checkpoint_dir = epath.Path(checkpoint_dir)
    async_options = _make_async_options(async_enable, async_timeout_secs)
    return ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=async_options,
        ),
    )


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    async_timeout_secs: int | None = 7200,
    async_enable: bool = True,
) -> tuple[ocp.CheckpointManager, bool]:
    logging.info(f"Checkpoint_dir:{checkpoint_dir}")
    checkpoint_dir = epath.Path(checkpoint_dir)
    logging.info(f"Checkpoint_dir:{checkpoint_dir}")
    resuming = False
    is_gcs = str(checkpoint_dir).startswith("gs://")
    exists = tf.io.gfile.exists(str(checkpoint_dir)) if is_gcs else checkpoint_dir.exists()
    if exists:
        if overwrite:
            try:
                if is_gcs:
                    # Use gsutil to delete the GCS prefix recursively.
                    _delete_gcs_prefix_with_gsutil(str(checkpoint_dir))
                    # Recreate the prefix to ensure later writes succeed.
                    tf.io.gfile.makedirs(str(checkpoint_dir))
                else:
                    checkpoint_dir.rmtree()
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
            except Exception as e:
                logging.warning(
                    "Failed to wipe checkpoint directory %s due to %s. Proceeding without wiping.",
                    checkpoint_dir,
                    e,
                )
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    if is_gcs:
        tf.io.gfile.makedirs(str(checkpoint_dir))
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = create_checkpoint_manager(
        checkpoint_dir,
        keep_period=keep_period,
        async_timeout_secs=async_timeout_secs,
        async_enable=async_enable,
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def _extract_directory(checkpoint_manager: ocp.CheckpointManager) -> str:
    directory_attr = getattr(checkpoint_manager, "directory", None)
    if directory_attr is None:
        directory_attr = getattr(checkpoint_manager, "_directory", None)
    return str(directory_attr) if directory_attr is not None else "<unknown>"


def _extract_keep_period(checkpoint_manager: ocp.CheckpointManager) -> int | None:
    options = getattr(checkpoint_manager, "options", None)
    if options is None:
        options = getattr(checkpoint_manager, "_options", None)
    return getattr(options, "keep_period", None)


def _extract_async_timeout(checkpoint_manager: ocp.CheckpointManager) -> int | None:
    options = getattr(checkpoint_manager, "options", None)
    if options is None:
        options = getattr(checkpoint_manager, "_options", None)
    async_opts = getattr(options, "async_options", None)
    if async_opts is None:
        return None
    return getattr(async_opts, "timeout_secs", None)


def _has_async_enabled(checkpoint_manager: ocp.CheckpointManager) -> bool:
    options = getattr(checkpoint_manager, "options", None)
    if options is None:
        options = getattr(checkpoint_manager, "_options", None)
    if options is None:
        return False
    return getattr(options, "async_options", None) is not None


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    max_retries: int = 0,
    retry_delay_secs: float = 0.0,
    retry_backoff: float = 1.0,
    fallback_to_sync: bool = False,
    async_timeout_secs: int | None = None,
    keep_period: int | None = None,
) -> ocp.CheckpointManager:
    start_time = time.perf_counter()
    directory_str = _extract_directory(checkpoint_manager)
    logging.info("Checkpoint save start | step=%d | dir=%s", step, directory_str)

    if keep_period is None:
        keep_period = _extract_keep_period(checkpoint_manager)
    if async_timeout_secs is None:
        async_timeout_secs = _extract_async_timeout(checkpoint_manager)

    attempt = 0
    delay_secs = retry_delay_secs
    manager_to_use = checkpoint_manager

    while True:
        attempt += 1
        try:

            def save_assets(directory: epath.Path):
                # Save the normalization stats.
                data_config = data_loader.data_config()
                norm_stats = data_config.norm_stats
                if norm_stats is not None and data_config.asset_id is not None:
                    _normalize_adapter.save(str(directory / data_config.asset_id), norm_stats)

            # Split params that can be used for inference into a separate item.
            with at.disable_typechecking():
                train_state, params = _split_params(state)
            items = {
                "assets": save_assets,
                "train_state": train_state,
                "params": {"params": params},
            }
            manager_to_use.save(step, items)

            # # Save TF iterator state if data loader supports it
            # if hasattr(data_loader, "save_iterator_checkpoint"):
            #     tf_iter_dir = f"{_extract_directory(manager_to_use)}/{step}/tf_iterator"
            #     data_loader.save_iterator_checkpoint(tf_iter_dir, step=step)

            duration = time.perf_counter() - start_time
            logging.info(
                "Checkpoint save complete | step=%d | dir=%s | duration=%.2fs",
                step,
                _extract_directory(manager_to_use),
                duration,
            )
            return manager_to_use
        except KeyboardInterrupt:
            raise
        except Exception as err:
            duration = time.perf_counter() - start_time
            logging.warning(
                "Checkpoint save failed | step=%d | dir=%s | duration=%.2fs | attempt=%d | error=%s",
                step,
                _extract_directory(manager_to_use),
                duration,
                attempt,
                err,
            )
            if fallback_to_sync and manager_to_use is checkpoint_manager and _has_async_enabled(manager_to_use):
                logging.info("Retrying checkpoint save with synchronous manager (async disabled).")
                manager_to_use = create_checkpoint_manager(
                    _extract_directory(checkpoint_manager),
                    keep_period=keep_period,
                    async_timeout_secs=async_timeout_secs,
                    async_enable=False,
                )
                continue

            if attempt > max_retries:
                logging.error(
                    "Checkpoint save exhausted retries | step=%d | attempts=%d",
                    step,
                    attempt,
                )
                raise

            if delay_secs > 0:
                logging.info("Sleeping %.1fs before retrying checkpoint save", delay_secs)
                time.sleep(delay_secs)
                if retry_backoff > 1.0:
                    delay_secs *= retry_backoff


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    # del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )

    # # Restore TF iterator state if data loader supports it
    # if hasattr(data_loader, "restore_iterator_checkpoint"):
    #     # Determine which step to restore from
    #     restore_step = step if step is not None else checkpoint_manager.latest_step()
    #     if restore_step is not None:
    #         tf_iter_dir = f"{_extract_directory(checkpoint_manager)}/{restore_step}/tf_iterator"
    #         data_loader.restore_iterator_checkpoint(tf_iter_dir)

    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(
    assets_dir: epath.Path | str, asset_id: str
) -> dict[str, _normalize_adapter.ExtendedNormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize_adapter.load(str(norm_stats_dir))
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(
    state: training_utils.TrainState,
) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])

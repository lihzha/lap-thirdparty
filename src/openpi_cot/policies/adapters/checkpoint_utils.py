import dataclasses
import logging

from etils import epath
from openpi.shared import array_typing as at
import orbax.checkpoint as ocp
from openpi_cot.training import utils as training_utils

def create_checkpoint_manager(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    async_timeout_secs: int | None = 7200,
    async_enable: bool = True,
) -> ocp.CheckpointManager:
    checkpoint_dir = epath.Path(checkpoint_dir)
    return ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
        ),
    )

def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    async_timeout_secs: int | None = 7200,
    async_enable: bool = True,
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir)
    logging.info(f"Checkpoint_dir:{checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = create_checkpoint_manager(
        checkpoint_dir,
        keep_period=keep_period,
        async_timeout_secs=async_timeout_secs,
        async_enable=async_enable,
    )

    return mngr, True


def restore_params(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    step: int | None = None,
    train_state_sharding: training_utils.TrainState | None = None,
) -> training_utils.TrainState:
    """Restore training state and dataloader state from checkpoint.

    Args:
        checkpoint_manager: The checkpoint manager
        state: Training state template to restore into
        data_loader: Data loader to restore iterator state
        step: Specific checkpoint step to restore (None = latest)
        train_state_sharding: Optional sharding tree to restore with explicit placement (e.g. evaluation)

    Returns:
        Restored training state
    """
    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        _, params = _split_params(state)
     
        _, params_sharding = _split_params(train_state_sharding)
        restore_args = ocp.args.Composite(
            params=ocp.args.PyTreeRestore(
                item={"params": params},
                transforms={},
                restore_args=ocp.checkpoint_utils.construct_restore_args(
                    {"params": params}, sharding_tree={"params": params_sharding}
                ),
            ),
        )
        restored = checkpoint_manager.restore(step, args=restore_args)

    return restored["params"]

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
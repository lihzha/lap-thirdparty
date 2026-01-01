from __future__ import annotations

import logging
from typing import Any

import flax.nnx as nnx
import jax
from openpi.models import model as _model
import openpi.policies.policy as _policy
import openpi.shared.array_typing as at
from openpi.training import config as _config
import openpi.transforms as up_transforms

from openpi_cot.policies.adapters.policy_adaptor import CoTPolicy
import openpi_cot.transforms as transforms

# Lazy imports to avoid loading TensorFlow (and allocating GPU memory) at import time
# These will be imported when create_trained_policy() is actually called
# from openpi_cot.shared.download import maybe_download
# from openpi_cot.training import checkpoints as _checkpoints


def load_model_from_train_state(config, checkpoint_dir):
    from flax import traverse_util
    from openpi.training import optimizer as _optimizer

    from openpi_cot.policies.adapters.checkpoint_utils import initialize_checkpoint_dir
    from openpi_cot.policies.adapters.checkpoint_utils import restore_params
    import openpi_cot.training.mh_sharding as sharding
    import openpi_cot.training.utils as training_utils

    rng = jax.random.key(config.seed)
    eval_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Human-readable mesh overview
    sharding.log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", sharding.format_sharding(data_sharding))
    logging.info("Replicated sharding spec: %s", sharding.format_sharding(replicated_sharding))

    # Initialize checkpoint manager
    checkpoint_manager, _ = initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=config.keep_period,
        async_timeout_secs=config.checkpoint_async_timeout_secs,
        async_enable=False,  # No async for evaluation
    )

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike) -> training_utils.TrainState:
        # Initialize the model
        model = config.model.create(rng)
        params = nnx.state(model)

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
    train_state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    # Log available checkpoints and determine which to load
    available_steps = list(checkpoint_manager.all_steps())
    logging.info(f"Available checkpoints: {available_steps}")

    if config.eval_checkpoint_step is None:
        latest_step = checkpoint_manager.latest_step()
        logging.info(f"No checkpoint step specified (--eval_checkpoint_step), using latest: {latest_step}")
        checkpoint_step_to_load = latest_step
    else:
        checkpoint_step_to_load = config.eval_checkpoint_step
        if checkpoint_step_to_load not in available_steps:
            raise ValueError(
                f"Requested checkpoint step {checkpoint_step_to_load} not found. Available steps: {available_steps}"
            )
        logging.info(f"Loading specified checkpoint step: {checkpoint_step_to_load}")

    # checkpoint_step_to_load = 20000

    # Restore checkpoint using the same helper as training (supports explicit sharding)
    params = restore_params(
        checkpoint_manager,
        train_state_shape,
        step=checkpoint_step_to_load,
        train_state_sharding=train_state_sharding,
    )
    params = params.to_pure_dict()
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    params = traverse_util.unflatten_dict(flat_params)

    model = config.model.load(params)

    return model


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: str,
    *,
    repack_transforms: up_transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, up_transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    # Lazy import to avoid loading TensorFlow at module import time
    from openpi_cot.shared.download import maybe_download

    repack_transforms = repack_transforms or up_transforms.Group()
    checkpoint_dir = maybe_download(str(checkpoint_dir))
    # model = load_model_from_train_state(train_config, checkpoint_dir)
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params"))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        # Lazy import to avoid loading TensorFlow and other heavy dependencies at module import time
        from openpi_cot.training import checkpoints as _checkpoints

        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            up_transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, normalization_type=getattr(data_config, "action_proprio_normalization_type", "normal")
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, normalization_type=getattr(data_config, "action_proprio_normalization_type", "normal")
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=False,
        pytorch_device=None,
    )


def create_trained_policy_cot(*args, sample_kwargs: dict | None = None, **kwargs) -> CoTPolicy:
    """Build the standard policy via upstream, then wrap with CoTPolicy."""
    base = create_trained_policy(*args, **kwargs)
    return CoTPolicy(base, sample_kwargs=sample_kwargs)


def create_trained_policy_from_model(
    model: _model.BaseModel,
    train_config: _config.TrainConfig,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    base = _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=False,
        pytorch_device=None,
    )
    return CoTPolicy(base, sample_kwargs=sample_kwargs)

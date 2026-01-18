"""Evaluate token accuracy and/or rollout performance on a checkpoint using validation data."""

import dataclasses
import logging
import math
import os
import platform

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils as mh
import numpy as np
from openpi.models import model as _model
from openpi.models.model import Observation
import openpi.shared.array_typing as at
from openpi.training import optimizer as _optimizer
from rail_tpu_utils import prevent_cross_region
import tqdm_loggable.auto as tqdm
import wandb

import openpi_cot.datasets.cot_data_loader as _data_loader
from openpi_cot.models.model_adapter import CoTObservation
import openpi_cot.training.checkpoints as _checkpoints
import openpi_cot.training.config as _config
import openpi_cot.training.mh_sharding as sharding
import openpi_cot.training.utils as training_utils
import openpi_cot.training.vis_tools as vis_tools

EMA_VALUE_RTOL = 5e-2
EMA_VALUE_ATOL = 1e-3


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
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    # Only initialize wandb in the main process
    if jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    eval_tag = f"eval_{config.eval_mode}"
    wandb.init(
        name=f"{config.exp_name}_{eval_tag}",
        config=dataclasses.asdict(config),
        project=config.project_name,
        tags=["evaluation", config.eval_mode],
    )


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


def _state_to_pytree(state):
    if state is None:
        return None
    if hasattr(state, "to_pure_dict"):
        return state.to_pure_dict()
    return state


def _is_numeric_leaf(value) -> bool:
    return isinstance(value, (jax.Array, np.ndarray)) or np.isscalar(value)


def _ensure_params_similarity(
    params,
    ema_params,
    *,
    value_rtol: float = EMA_VALUE_RTOL,
    value_atol: float = EMA_VALUE_ATOL,
) -> None:
    """Ensure that `params` and `ema_params` are shape/dtype compatible and numerically close."""
    params_tree = _state_to_pytree(params)
    ema_tree = _state_to_pytree(ema_params)

    at.check_pytree_equality(
        expected=params_tree,
        got=ema_tree,
        check_shapes=True,
        check_dtypes=True,
    )

    params_with_path, _ = jax.tree_util.tree_flatten_with_path(params_tree)
    ema_leaves, _ = jax.tree_util.tree_flatten(ema_tree)

    for (path, param_leaf), ema_leaf in zip(params_with_path, ema_leaves):
        if _is_numeric_leaf(param_leaf):
            param_arr = jnp.asarray(param_leaf)
            ema_arr = jnp.asarray(ema_leaf)

            if jnp.issubdtype(param_arr.dtype, jnp.bool_):
                values_equal = bool(jnp.array_equal(param_arr, ema_arr))
                if not values_equal:
                    raise ValueError(
                        f"EMA params mismatch at {jax.tree_util.keystr(path)}: boolean values differ between params "
                        "and ema_params"
                    )
                continue

            are_close = bool(jnp.allclose(param_arr, ema_arr, rtol=value_rtol, atol=value_atol))
            if not are_close:
                max_diff = float(jnp.max(jnp.abs(param_arr - ema_arr)))
                raise ValueError(
                    f"EMA params mismatch at {jax.tree_util.keystr(path)}: max abs diff {max_diff:.6f} exceeds "
                    f"rtol={value_rtol} / atol={value_atol}"
                )
        elif param_leaf != ema_leaf:
            raise ValueError(
                f"EMA params mismatch at {jax.tree_util.keystr(path)}: values differ ({param_leaf!r} vs {ema_leaf!r})"
            )

    logging.info("Verified EMA parameters match main parameters (rtol=%s, atol=%s)", value_rtol, value_atol)


class TokenAccuracyEvaluator:
    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[CoTObservation, _model.Actions],
    ) -> dict[str, at.Array]:
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        @at.typecheck
        def eval_fn(
            model: _model.BaseModel,
            rng: at.KeyArrayLike,
            observation: CoTObservation,
            actions: _model.Actions,
        ):
            loss, metrics = model.compute_loss(rng, observation, actions, train=False, verbose_mode=True)
            return loss, metrics

        eval_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        loss, metrics = eval_fn(model, eval_rng, observation, actions)
        logging.info(metrics.keys())

        info = {
            "loss": loss,
        }
        for k, v in metrics.items():
            if k in {"lang_loss", "langact_loss", "per_sample_loss", "labels"}:
                continue
            info[k] = v

        return info


class RolloutEvaluator:
    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[CoTObservation, _model.Actions],
    ) -> jax.Array:
        """Sample language action tokens for rollout evaluation.

        Returns:
            id_buf: Sampled token IDs [batch, seq_len, 1]
            t_final: Final sequence length
        """
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        # Prepare eval batch (remove language action from ground truth)
        observation, _ = batch

        # Sample language action tokens
        output_tokens = model.sample_tokens(rng, observation)

        return output_tokens


class TokenVisualizationEvaluator:
    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[CoTObservation, _model.Actions],
    ) -> dict[str, at.Array]:
        """Compute loss and return predictions and labels for visualization.

        Returns:
            Dictionary containing loss, predictions, labels, and token_mask
        """
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        eval_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch

        loss, metrics = model.compute_loss_with_decoded_tokens(
            eval_rng, observation, actions, train=False, verbose_mode=False
        )

        info = {
            "loss": loss,
            "predictions": metrics["predictions"],
            "labels": metrics["labels"],
            "token_mask": metrics["token_mask"],
        }

        # Include other metrics if available
        for key in ["token_accuracy", "critical_token_accuracy", "number_token_accuracy", "direction_token_accuracy"]:
            if key in metrics:
                info[key] = metrics[key]

        return info


class TrainLossEvaluator:
    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[CoTObservation, _model.Actions],
    ) -> tuple[jax.Array, dict[str, at.Array]]:
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        loss, metrics = model.compute_loss(rng, batch[0], batch[1], train=True, stage_config=None)
        return loss, metrics


class ValidationLossEvaluator:
    """Validation loss evaluator that exactly matches train.py's ValidationStepRunner."""

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

        eval_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch

        # Call compute_loss to get per-sample metrics for dataset tracking
        # Note: We use the model in eval mode but request per-sample metrics by passing train=True
        # This is to enable dataset-level tracking during validation
        # Pass verbose_mode=True to enable detailed metrics for validation
        verbose_mode = self.config.model.verbose_mode
        val_loss, val_metrics = model.compute_loss(
            eval_rng, observation, actions, train=False, verbose_mode=verbose_mode
        )

        val_metrics["val_loss"] = val_loss

        return val_metrics


def load_model_from_params_directly(
    config: _config.TrainConfig,
    checkpoint_dir: epath.Path,
    mesh: jax.sharding.Mesh,
    init_rng: at.KeyArrayLike,
    train_state_shape: training_utils.TrainState,
    train_state_sharding: training_utils.TrainState,
) -> training_utils.TrainState:
    """Load model directly from params directory, similar to policy_config_adapter.create_trained_policy.
    
    This mode loads params directly from checkpoint_dir / "params" using _model.restore_params,
    then creates a model using config.model.load(). This is simpler than loading the full
    TrainState and can be useful when you only need the model for inference.
    
    Args:
        config: Training configuration
        checkpoint_dir: Checkpoint directory path
        mesh: Device mesh for sharding
        init_rng: Random key for initialization
        train_state_shape: Shape template for the train state
        train_state_sharding: Sharding specification for the train state
        
    Returns:
        TrainState with loaded model
    """
    from openpi.training import optimizer as _optimizer
    
    logging.info("Loading model directly from params (eval_load_params_directly=True)")
    
    # Determine checkpoint step directory
    # CheckpointManager uses step directories like "step_1000", "step_2000", etc.
    if config.eval_checkpoint_step is not None:
        checkpoint_step_dir = checkpoint_dir / config.eval_checkpoint_step
    else:
        # Find latest checkpoint step
        checkpoint_dirs = [d for d in checkpoint_dir.iterdir()]
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoint steps found in {checkpoint_dir}")
        checkpoint_step_dir = max(checkpoint_dirs, key=lambda d: int(d.name))  # type: ignore
        logging.info(f"No checkpoint step specified, using latest: {checkpoint_step_dir.name}")
    
    params_path = checkpoint_step_dir / "params"
    if not params_path.exists():
        raise ValueError(f"Params directory not found: {params_path}")
    
    # Get step number from checkpoint directory name
    step = int(checkpoint_step_dir.name)
    
    # Load params directly using _model.restore_params, same as policy_config_adapter.py
    params = _model.restore_params(params_path)
    
    # Create model from params
    model = config.model.load(params)
    
    # Create TrainState from the loaded model
    params_state = nnx.state(model)
    model_def = nnx.graphdef(model)
    
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    ema_decay, ema_params_enabled = config.get_ema_init()
    
    train_state = training_utils.TrainState(
        step=step,
        params=params_state,
        model_def=model_def,
        tx=tx,
        opt_state=tx.init(params_state.filter(config.trainable_filter)),
        ema_decay=ema_decay,
        ema_params=None if not ema_params_enabled else params_state,
    )
    
    logging.info(f"Loaded model from params at step {step}")
    return train_state


def main(config: _config.TrainConfig):
    init_logging()
    effective_fsdp_devices = init_tpu(config)

    rng = jax.random.key(config.seed)
    eval_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Human-readable mesh overview
    sharding.log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", sharding.format_sharding(data_sharding))
    logging.info("Replicated sharding spec: %s", sharding.format_sharding(replicated_sharding))

    init_wandb(config, enabled=config.wandb_enabled)

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    ema_decay, ema_params_enabled = config.get_ema_init()

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
            ema_decay=ema_decay,
            ema_params=None if not ema_params_enabled else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    train_state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    # Load model using either direct params loading or full state restoration
    if config.eval_load_params_directly:
        checkpoint_dir = epath.Path(config.checkpoint_dir)
        train_state = load_model_from_params_directly(
            config,
            checkpoint_dir,
            mesh,
            init_rng,
            train_state_shape,
            train_state_sharding,
        )
    else:
        # Initialize checkpoint manager
        checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=False,  # Never overwrite for evaluation
            resume=True,  # Always resume for evaluation
            async_timeout_secs=config.checkpoint_async_timeout_secs,
            async_enable=False,  # No async for evaluation
        )

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

        # Restore checkpoint using the same helper as training (supports explicit sharding)
        train_state = _checkpoints.restore_state(
            checkpoint_manager,
            train_state_shape,
            data_loader=None,
            step=checkpoint_step_to_load,
            train_state_sharding=train_state_sharding,
        )

    # if train_state.ema_params is not None:
    #     ema_rtol = EMA_VALUE_RTOL
    #     if train_state.ema_decay is not None:
    #         ema_rtol = max(ema_rtol, 1.0 - float(train_state.ema_decay))
    #     _ensure_params_similarity(
    #         train_state.params,
    #         train_state.ema_params,
    #         value_rtol=ema_rtol,
    #         value_atol=EMA_VALUE_ATOL,
    #     )

    # # Use EMA params for evaluation if available (matches training checkpoint layout)
    if train_state.ema_params is not None and config.eval_use_ema:
        logging.info("Using EMA params for evaluation")
        train_state = dataclasses.replace(train_state, params=train_state.ema_params)

    logging.info(f"Loaded checkpoint at step {train_state.step}")
    sharding.log_param_sharding_actual(train_state.params)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        split=config.eval_split,
        seed=config.seed,
        max_samples=getattr(config.data, "val_max_samples", None),
    )

    # Determine number of evaluation batches
    num_eval_batches = config.num_eval_batches
    if num_eval_batches is None:
        if getattr(config.data, "val_max_samples", None):
            process_count = getattr(jax, "process_count", lambda: 1)()
            local_bs = max(1, config.batch_size // process_count)
            num_eval_batches = math.ceil(config.data.val_max_samples / local_bs)
        else:
            # Default to 1000 batches if not specified
            num_eval_batches = 1000

    logging.info(f"Evaluating over {num_eval_batches} batches with mode: {config.eval_mode}")

    results = {}

    # Token accuracy evaluation
    if config.eval_mode in ["token_accuracy", "both"]:
        logging.info("Running token accuracy evaluation...")
        token_results = evaluate_token_accuracy(
            config,
            eval_rng,
            train_state,
            train_state_sharding,
            data_loader,
            mesh,
            data_sharding,
            replicated_sharding,
            num_eval_batches,
        )
        results.update(token_results)

    # Rollout evaluation
    if config.eval_mode in ["rollout", "both"]:
        logging.info("Running rollout evaluation...")
        rollout_results = evaluate_rollout(
            config,
            eval_rng,
            train_state,
            train_state_sharding,
            data_loader,
            mesh,
            data_sharding,
            replicated_sharding,
            num_eval_batches,
        )
        results.update(rollout_results)

    # Token visualization evaluation
    if config.eval_mode == "token_visualization":
        logging.info("Running token visualization evaluation...")
        viz_results = evaluate_token_visualization(
            config,
            eval_rng,
            train_state,
            train_state_sharding,
            data_loader,
            mesh,
            data_sharding,
            replicated_sharding,
            num_eval_batches,
        )
        results.update(viz_results)

    if config.eval_mode == "train_loss":
        logging.info("Running train loss evaluation...")
        loss, metrics = evaluate_loss(
            config,
            eval_rng,
            train_state,
            train_state_sharding,
            data_loader,
            mesh,
            data_sharding,
            replicated_sharding,
            num_eval_batches,
        )
        results.update({"eval/train_loss/loss": float(loss)})
        for key, value in metrics.items():
            results[f"eval/train_loss/{key}"] = float(value)

    # Validation loss evaluation (matches train.py's ValidationStepRunner exactly)
    if config.eval_mode == "val_loss":
        logging.info("Running validation loss evaluation (matching train.py)...")
        val_results = evaluate_validation_loss(
            config,
            eval_rng,
            train_state,
            train_state_sharding,
            data_loader,
            mesh,
            data_sharding,
            replicated_sharding,
            num_eval_batches,
        )
        results.update(val_results)

    # Log final results
    logging.info("=" * 80)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 80)
    for key, value in results.items():
        logging.info(f"{key:40s}: {value}")
    logging.info("=" * 80)

    # Log to wandb
    if jax.process_index() == 0 and config.wandb_enabled:
        wandb.log(results, step=int(train_state.step))
        wandb.summary.update(results)

    return results


def evaluate_token_accuracy(
    config: _config.TrainConfig,
    eval_rng: at.KeyArrayLike,
    train_state: training_utils.TrainState,
    train_state_sharding,
    data_loader,
    mesh,
    data_sharding,
    replicated_sharding,
    num_eval_batches: int,
) -> dict[str, float]:
    """Evaluate token accuracy."""
    evaluator = TokenAccuracyEvaluator(config)
    peval_step = jax.jit(
        evaluator,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
    )

    data_iter = iter(data_loader)

    pbar = tqdm.tqdm(
        range(num_eval_batches),
        total=num_eval_batches,
        dynamic_ncols=True,
        desc="Token accuracy evaluation",
        disable=(jax.process_index() != 0),
    )

    number_token_losses = []
    direction_token_losses = []
    other_token_losses = []

    number_token_accuracies = []
    direction_token_accuracies = []
    all_token_accuracies = []

    with sharding.set_mesh(mesh):
        for batch_idx in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                logging.info(f"Reached end of dataset at batch {batch_idx}")
                break

            eval_info = peval_step(eval_rng, train_state, batch)
            # Bring sharded arrays to host memory for local computation.
            per_token_loss = jnp.asarray(training_utils.to_local_array(eval_info["per_token_loss"]))
            all_token_accuracies.append(jnp.asarray(training_utils.to_local_array(eval_info["token_accuracy"])))
            if not config.model.use_fast:
                number_mask = jnp.asarray(training_utils.to_local_array(batch[0].number_token_mask))[:, 1:]
                direction_mask = jnp.asarray(training_utils.to_local_array(batch[0].direction_token_mask))[:, 1:]
                tokenized_langact_mask = jnp.asarray(training_utils.to_local_array(batch[0].tokenized_langact_mask))[
                    :, 1:
                ]

                def _masked_mean(mask):
                    mask_sum = jnp.sum(mask)
                    return jnp.sum(per_token_loss * mask) / mask_sum

                number_token_loss = _masked_mean(number_mask)
                direction_token_loss = _masked_mean(direction_mask)
                other_token_mask = jnp.logical_and(
                    tokenized_langact_mask,
                    jnp.logical_not(
                        jnp.logical_or(
                            number_mask,
                            direction_mask,
                        )
                    ),
                )
                other_token_loss = _masked_mean(other_token_mask)
                number_token_losses.append(number_token_loss)
                direction_token_losses.append(direction_token_loss)
                other_token_losses.append(other_token_loss)

                number_token_accuracies.append(
                    jnp.asarray(training_utils.to_local_array(eval_info["number_token_accuracy"]))
                )
                direction_token_accuracies.append(
                    jnp.asarray(training_utils.to_local_array(eval_info["direction_token_accuracy"]))
                )

    if not config.model.use_fast:
        return {
            "eval/number_token_loss": float(jnp.mean(jnp.array(number_token_losses))),
            "eval/direction_token_loss": float(jnp.mean(jnp.array(direction_token_losses))),
            "eval/other_token_loss": float(jnp.mean(jnp.array(other_token_losses))),
            "eval/number_token_accuracy": float(jnp.mean(jnp.array(number_token_accuracies))),
            "eval/direction_token_accuracy": float(jnp.mean(jnp.array(direction_token_accuracies))),
            "eval/token_accuracy": float(jnp.mean(jnp.array(all_token_accuracies))),
        }
    return {
        "eval/token_accuracy": float(jnp.mean(jnp.array(all_token_accuracies))),
    }


def evaluate_rollout(
    config: _config.TrainConfig,
    eval_rng: at.KeyArrayLike,
    train_state: training_utils.TrainState,
    train_state_sharding,
    data_loader,
    mesh,
    data_sharding,
    replicated_sharding,
    num_eval_batches: int,
    max_logged_imgs: int = 100,
) -> dict[str, float]:
    """Evaluate rollout performance (language action prediction accuracy)."""
    evaluator = RolloutEvaluator(config)
    # Note: batch input uses data sharding because loaders are process-sharded.
    peval_step = jax.jit(
        evaluator,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(data_sharding),
    )
    # peval_step = evaluator
    # Get tokenizer for decoding
    tokenizer = data_loader.tokenizer

    data_iter = iter(data_loader)
    images_to_log = []
    num_logged_imgs = 0

    pbar = tqdm.tqdm(
        range(num_eval_batches),
        total=num_eval_batches,
        dynamic_ncols=True,
        desc="Rollout evaluation",
        disable=(jax.process_index() != 0),
    )

    with sharding.set_mesh(mesh):
        for batch_idx in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                logging.info(f"Reached end of dataset at batch {batch_idx}")
                break

            # Materialize process-local data to avoid non-addressable access in Python.
            batch_local = jax.tree_util.tree_map(training_utils.to_local_array, batch)

            # Compute a global max cut index so all hosts pad to the same length.
            langact_mask = batch_local[0].tokenized_langact_mask
            if langact_mask is None:
                global_max_cut_idx = None
            else:
                local_max = 0
                for i in range(langact_mask.shape[0]):
                    true_indices = np.where(langact_mask[i])[0]
                    cut_idx = int(true_indices[0]) if true_indices.size > 0 else int(langact_mask.shape[1])
                    local_max = max(local_max, cut_idx)
                gathered = mh.process_allgather(np.array([local_max], dtype=np.int32))
                global_max_cut_idx = int(np.max(np.asarray(gathered)))

            # Prepare eval batch (remove language actions) and shard for JIT.
            eval_batch = vis_tools.prepare_eval_batch(batch_local, global_max_cut_idx=global_max_cut_idx)

            def _to_sharded(x):
                if not (hasattr(x, "shape") and x.shape):
                    return x
                if hasattr(x, "dtype") and (x.dtype == np.object_ or getattr(x.dtype, "kind", None) in ("U", "S")):
                    return x
                return jax.make_array_from_process_local_data(data_sharding, x)

            eval_batch_sharded = jax.tree_util.tree_map(_to_sharded, eval_batch)

            # Run rollout evaluation
            output_tokens = peval_step(eval_rng, train_state, eval_batch_sharded)

            k_local = min(config.batch_size, batch_local[0].state.shape[0])

            # Process results on host
            if jax.process_index() == 0:
                output_tokens_local = training_utils.to_local_array(output_tokens)
                gt_texts, pred_texts = vis_tools.eval_step(batch_local, output_tokens_local, tokenizer, k_local)

                imgs_to_log = eval_batch[0].images["base_0_rgb"][:k_local]

                for i in range(k_local):
                    gt_text = gt_texts[i]
                    pred_text = pred_texts[i]
                    img = imgs_to_log[i]

                    img_to_log = vis_tools.create_rollout_visualization(
                        img,
                        gt_text,
                        pred_text,
                    )
                    images_to_log.append(wandb.Image(img_to_log))
            num_logged_imgs += k_local
            if num_logged_imgs >= max_logged_imgs:
                break

    return {"eval/rollout": images_to_log}


def evaluate_loss(
    config: _config.TrainConfig,
    eval_rng: at.KeyArrayLike,
    train_state: training_utils.TrainState,
    train_state_sharding,
    data_loader,
    mesh,
    data_sharding,
    replicated_sharding,
    num_eval_batches: int,
) -> dict[str, float]:
    """Evaluate rollout performance (language action prediction accuracy)."""
    evaluator = TrainLossEvaluator(config)
    # Note: batch input uses replicated sharding because prepare_eval_batch is called outside JIT
    # peval_step = jax.jit(
    #     evaluator,
    #     in_shardings=(replicated_sharding, train_state_sharding, replicated_sharding),
    #     out_shardings=(replicated_sharding),
    # )
    peval_step = evaluator
    # Get tokenizer for decoding

    data_iter = iter(data_loader)
    pbar = tqdm.tqdm(
        range(num_eval_batches),
        total=num_eval_batches,
        dynamic_ncols=True,
        desc="Rollout evaluation",
        disable=(jax.process_index() != 0),
    )

    with sharding.set_mesh(mesh):
        for batch_idx in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                logging.info(f"Reached end of dataset at batch {batch_idx}")
                break

            loss, metrics = peval_step(eval_rng, train_state, batch)
    return loss, metrics


def evaluate_validation_loss(
    config: _config.TrainConfig,
    eval_rng: at.KeyArrayLike,
    train_state: training_utils.TrainState,
    train_state_sharding,
    data_loader,
    mesh,
    data_sharding,
    replicated_sharding,
    num_eval_batches: int,
) -> dict[str, float]:
    """Evaluate validation loss, matching train.py's ValidationStepRunner exactly."""
    evaluator = ValidationLossEvaluator(config)
    pval_step = jax.jit(
        evaluator,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    data_iter = iter(data_loader)
    val_infos = []

    pbar = tqdm.tqdm(
        range(num_eval_batches),
        total=num_eval_batches,
        dynamic_ncols=True,
        desc="Validation loss evaluation",
        disable=(jax.process_index() != 0),
    )

    with sharding.set_mesh(mesh):
        for batch_idx in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                logging.info(f"Reached end of dataset at batch {batch_idx}")
                break

            val_info = pval_step(eval_rng, train_state, batch)
            val_infos.append(val_info)

            # Update progress bar with running loss
            if val_infos and (batch_idx + 1) % 10 == 0:
                recent_losses = [float(jax.device_get(info["val_loss"])) for info in val_infos[-min(10, len(val_infos)) :]]
                pbar.set_postfix({"val_loss": f"{np.mean(recent_losses):.4f}"})

    # Aggregate metrics across all batches
    results = {}
    if val_infos:
        # Get all metric keys from the first batch
        metric_keys = val_infos[0].keys()

        for key in metric_keys:
            # Skip non-scalar metrics
            values = []
            for info in val_infos:
                if key in info:
                    val = jax.device_get(info[key])
                    # Only aggregate scalar values
                    if np.isscalar(val) or (hasattr(val, "shape") and val.shape == ()):
                        values.append(float(val))
            if values:
                results[f"eval/val_loss/{key}"] = float(np.mean(values))

    results["eval/val_loss/num_batches"] = len(val_infos)

    return results


def evaluate_token_visualization(
    config: _config.TrainConfig,
    eval_rng: at.KeyArrayLike,
    train_state: training_utils.TrainState,
    train_state_sharding,
    data_loader,
    mesh,
    data_sharding,
    replicated_sharding,
    num_eval_batches: int,
) -> dict[str, float]:
    """Evaluate and visualize token predictions vs ground truth."""
    evaluator = TokenVisualizationEvaluator(config)
    peval_step = jax.jit(
        evaluator,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
    )

    # Get tokenizer for decoding
    tokenizer = data_loader.tokenizer

    data_iter = iter(data_loader)
    all_accuracies = []
    max_samples_to_log = 10  # Log only a few samples for readability

    pbar = tqdm.tqdm(
        range(num_eval_batches),
        total=num_eval_batches,
        dynamic_ncols=True,
        desc="Token visualization evaluation",
        disable=(jax.process_index() != 0),
    )

    with sharding.set_mesh(mesh):
        for batch_idx in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                logging.info(f"Reached end of dataset at batch {batch_idx}")
                break

            eval_info = peval_step(eval_rng, train_state, batch)

            # Process results on host
            if jax.process_index() == 0:
                predictions = jax.device_get(eval_info["predictions"])  # [batch, seq_len]
                labels = jax.device_get(eval_info["labels"])  # [batch, seq_len]
                token_mask = jax.device_get(eval_info["token_mask"])  # [batch, seq_len]

                # Log first few samples
                if batch_idx < max_samples_to_log:
                    batch_size = predictions.shape[0]
                    for sample_idx in range(min(2, batch_size)):  # Log up to 2 samples per batch
                        pred_tokens = predictions[sample_idx]
                        label_tokens = labels[sample_idx]
                        mask = token_mask[sample_idx]

                        # Only decode valid tokens
                        valid_pred = pred_tokens[mask > 0]
                        valid_label = label_tokens[mask > 0]

                        # Decode tokens
                        pred_text = tokenizer.decode(valid_pred.tolist())
                        label_text = tokenizer.decode(valid_label.tolist())

                        # Log to console
                        logging.info("=" * 80)
                        logging.info(f"Batch {batch_idx}, Sample {sample_idx}")
                        logging.info("-" * 80)
                        logging.info(f"Ground Truth: {label_text}")
                        logging.info(f"Prediction:   {pred_text}")
                        logging.info("=" * 80)

                        # Log to wandb
                        if config.wandb_enabled:
                            wandb.log(
                                {
                                    f"eval/token_viz/batch_{batch_idx}_sample_{sample_idx}/ground_truth": label_text,
                                    f"eval/token_viz/batch_{batch_idx}_sample_{sample_idx}/prediction": pred_text,
                                },
                                step=int(train_state.step),
                            )

                # Track accuracy
                if "token_accuracy" in eval_info:
                    all_accuracies.append(float(jax.device_get(eval_info["token_accuracy"])))

            # Update progress bar
            if all_accuracies and (batch_idx + 1) % 10 == 0:
                recent_acc = all_accuracies[-min(10, len(all_accuracies)) :]
                pbar.set_postfix({"token_acc": f"{np.mean(recent_acc):.4f}"})

    # Compute final statistics
    results = {}
    if all_accuracies:
        results["eval/token_viz/mean_accuracy"] = float(np.mean(all_accuracies))
        results["eval/token_viz/std_accuracy"] = float(np.std(all_accuracies))

    results["eval/token_viz/num_batches"] = len(all_accuracies) if all_accuracies else 0

    return results


if __name__ == "__main__":
    config = _config.cli()
    main(config)

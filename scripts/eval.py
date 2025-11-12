"""Evaluate token accuracy and/or rollout performance on a checkpoint using validation data."""

import dataclasses
import logging
import math
import os
import platform

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
import openpi.shared.array_typing as at
from rail_tpu_utils import prevent_cross_region
import tqdm_loggable.auto as tqdm
import wandb

import openpi_cot.dataloader.cot_data_loader as _data_loader
from openpi_cot.models.adapters.model_adapter import CoTObservation
import openpi_cot.training.checkpoints as _checkpoints
import openpi_cot.training.config as _config
import openpi_cot.training.mh_sharding as sharding
import openpi_cot.training.utils as training_utils
import openpi_cot.training.vis_tools as vis_tools


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
            per_sample_loss, token_accuracy, critical_token_accuracy = model.compute_loss(
                rng, observation, actions, train=False
            )
            return jnp.mean(per_sample_loss), (per_sample_loss, token_accuracy, critical_token_accuracy)

        eval_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        loss, (per_sample_loss, token_accuracy, critical_token_accuracy) = eval_fn(
            model, eval_rng, observation, actions
        )

        info = {
            "loss": loss,
            "per_sample_loss": per_sample_loss,
            "token_accuracy": token_accuracy,
            "critical_token_accuracy": critical_token_accuracy,
        }
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
    ) -> tuple[jax.Array, jax.Array]:
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
        id_buf, t_final = model.sample_reasoning(observation)

        return id_buf, t_final


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

    # Initialize checkpoint manager
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,  # Never overwrite for evaluation
        resume=True,  # Always resume for evaluation
        async_timeout_secs=config.checkpoint_async_timeout_secs,
        async_enable=False,  # No async for evaluation
    )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        split="val",
        seed=config.seed,
        max_samples=getattr(config.data, "val_max_samples", None),
    )

    # Initialize train state shape and sharding
    from openpi.training import optimizer as _optimizer

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

    # Initialize concrete train state with proper sharding for checkpoint restoration
    logging.info("Initializing train state for checkpoint restoration...")
    train_state = jax.jit(
        init,
        in_shardings=replicated_sharding,
        out_shardings=train_state_sharding,
    )(init_rng)

    # Restore checkpoint with explicit sharding for device mismatch handling
    import orbax.checkpoint as ocp

    with at.disable_typechecking():
        # Split params for restoration
        def _split_params(state):
            """Split params from train state for separate checkpointing."""
            params = state.params
            state_without_params = dataclasses.replace(state, params=None)
            return state_without_params, params

        def _merge_params(state, params_dict):
            """Merge params back into train state after restoration."""
            params = params_dict.get("params")
            return dataclasses.replace(state, params=params)

        train_state_without_params, params = _split_params(train_state)

        # Get sharding for restoration
        train_state_sharding_without_params, params_sharding = _split_params(train_state_sharding)

        # Create restore args with explicit shardings to handle device mismatch
        restore_args = ocp.args.Composite(
            train_state=ocp.args.PyTreeRestore(
                item=train_state_without_params,
                transforms={},
                restore_args=ocp.checkpoint_utils.construct_restore_args(
                    train_state_without_params,
                    sharding_tree=train_state_sharding_without_params,
                ),
            ),
            params=ocp.args.PyTreeRestore(
                item={"params": params},
                transforms={},
                restore_args=ocp.checkpoint_utils.construct_restore_args(
                    {"params": params},
                    sharding_tree={"params": params_sharding},
                ),
            ),
        )

        restored = checkpoint_manager.restore(
            config.eval_checkpoint_step,
            args=restore_args,
        )

        train_state = _merge_params(restored["train_state"], restored["params"])

        # # Use EMA params for evaluation if available (they typically perform better)
        # if train_state.ema_params is not None:
        #     logging.info("Using EMA params for evaluation")
        #     train_state = dataclasses.replace(train_state, params=train_state.ema_params)
        # else:
        #     logging.info("EMA params not available, using regular params for evaluation")

    logging.info(f"Loaded checkpoint at step {train_state.step}")
    sharding.log_param_sharding_actual(train_state.params)

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
    eval_infos = []

    pbar = tqdm.tqdm(
        range(num_eval_batches),
        total=num_eval_batches,
        dynamic_ncols=True,
        desc="Token accuracy evaluation",
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
            eval_infos.append(eval_info)

            # Log intermediate results
            if (batch_idx + 1) % 10 == 0:
                stacked_infos = common_utils.stack_forest(eval_infos[-10:])
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                pbar.set_postfix(
                    {
                        "loss": f"{reduced_info['loss']:.4f}",
                        "token_acc": f"{reduced_info['token_accuracy']:.4f}",
                    }
                )

    # Compute final statistics
    stacked_infos = common_utils.stack_forest(eval_infos)
    final_results = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

    # Compute additional statistics
    token_accuracies = jax.device_get(stacked_infos["token_accuracy"])
    losses = jax.device_get(stacked_infos["loss"])

    return {
        "eval/token_accuracy/mean": float(final_results["token_accuracy"]),
        "eval/token_accuracy/loss": float(final_results["loss"]),
        "eval/token_accuracy/std": float(jnp.std(token_accuracies)),
        "eval/token_accuracy/loss_std": float(jnp.std(losses)),
        "eval/token_accuracy/min": float(jnp.min(token_accuracies)),
        "eval/token_accuracy/max": float(jnp.max(token_accuracies)),
        "eval/num_batches_evaluated": len(eval_infos),
        "eval/checkpoint_step": int(train_state.step),
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
) -> dict[str, float]:
    """Evaluate rollout performance (language action prediction accuracy)."""
    evaluator = RolloutEvaluator(config)
    # Note: batch input uses replicated sharding because prepare_eval_batch is called outside JIT
    peval_step = jax.jit(
        evaluator,
        in_shardings=(replicated_sharding, train_state_sharding, replicated_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
    )

    # Get tokenizer for decoding
    tokenizer = data_loader.tokenizer

    data_iter = iter(data_loader)
    l2_cm_values_all = []
    images_to_log = []
    num_images_logged = 0
    max_images_to_log = 64

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

            # Prepare eval batch (remove language actions) and replicate for JIT
            eval_batch = vis_tools.prepare_eval_batch(batch)
            # Replicate the batch to match expected sharding
            eval_batch_replicated = jax.device_put(eval_batch, replicated_sharding)

            breakpoint()

            # Run rollout evaluation
            id_buf, t_final = peval_step(eval_rng, train_state, eval_batch_replicated)

            # Process results on host
            if jax.process_index() == 0:
                k_local = min(config.batch_size, batch[0].state.shape[0])
                l2_cm_values, to_log = vis_tools.eval_step(batch, id_buf, t_final, tokenizer, k_local)

                if l2_cm_values:
                    l2_cm_values_all.extend(l2_cm_values)

                # Collect images for logging
                if to_log and num_images_logged < max_images_to_log:
                    for img in to_log:
                        if num_images_logged >= max_images_to_log:
                            break
                        images_to_log.append(wandb.Image(img))
                        num_images_logged += 1

            # Update progress bar
            if l2_cm_values_all and (batch_idx + 1) % 10 == 0:
                recent_l2 = l2_cm_values_all[-min(100, len(l2_cm_values_all)) :]
                pbar.set_postfix(
                    {
                        "mean_l2_cm": f"{np.mean(recent_l2):.2f}",
                        "n_samples": len(l2_cm_values_all),
                    }
                )

    # Compute final statistics
    if not l2_cm_values_all:
        logging.warning("No valid rollout samples were evaluated!")
        return {
            "eval/rollout/mean_l2_cm": float("nan"),
            "eval/rollout/std_l2_cm": float("nan"),
            "eval/rollout/num_samples": 0,
        }

    l2_array = np.array(l2_cm_values_all)
    results = {
        "eval/rollout/mean_l2_cm": float(np.mean(l2_array)),
        "eval/rollout/std_l2_cm": float(np.std(l2_array)),
        "eval/rollout/median_l2_cm": float(np.median(l2_array)),
        "eval/rollout/min_l2_cm": float(np.min(l2_array)),
        "eval/rollout/max_l2_cm": float(np.max(l2_array)),
        "eval/rollout/num_samples": len(l2_cm_values_all),
    }

    # Log images to wandb
    if images_to_log and jax.process_index() == 0 and config.wandb_enabled:
        wandb.log({"eval/rollout/predictions": images_to_log}, step=int(train_state.step))

    return results


if __name__ == "__main__":
    config = _config.cli()
    main(config)

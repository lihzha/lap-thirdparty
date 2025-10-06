import dataclasses
import logging
import math
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
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

    wandb.init(
        name=f"{config.exp_name}_token_accuracy_eval",
        config=dataclasses.asdict(config),
        project=config.project_name,
        tags=["evaluation", "token_accuracy"],
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
            per_sample_loss = model.compute_loss(rng, observation, actions, train=False)
            token_accuracy = getattr(model, "token_accuracy", None)
            token_acc_value = token_accuracy.value if token_accuracy is not None else jnp.array(0.0)
            return jnp.mean(per_sample_loss), (per_sample_loss, token_acc_value)

        eval_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        loss, (per_sample_loss, token_accuracy) = eval_fn(model, eval_rng, observation, actions)

        info = {
            "loss": loss,
            "per_sample_loss": per_sample_loss,
            "token_accuracy": token_accuracy,
        }
        return info


def load_checkpoint(
    config: _config.TrainConfig,
    mesh: jax.sharding.Mesh,
    checkpoint_path: epath.Path,
) -> tuple[training_utils.TrainState, Any]:
    """Load checkpoint from the specified path."""
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

    rng = jax.random.key(config.seed)
    train_state_shape = jax.eval_shape(init, rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    # Initialize checkpoint manager
    checkpoint_manager = _checkpoints.OrbaxCheckpointManager(
        checkpoint_path,
        keep_period=None,
        async_timeout_secs=None,
        async_enable=False,
    )

    # Restore the checkpoint
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, None)

    logging.info(f"Loaded checkpoint from {checkpoint_path}, step: {train_state.step}")

    return train_state, state_sharding


def main(config: _config.TrainConfig, checkpoint_path: str, num_eval_batches: int | None = None):
    init_logging()
    effective_fsdp_devices = init_tpu(config)

    rng = jax.random.key(config.seed)
    eval_rng = rng

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Human-readable mesh overview
    sharding.log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", sharding.format_sharding(data_sharding))
    logging.info("Replicated sharding spec: %s", sharding.format_sharding(replicated_sharding))

    init_wandb(config, enabled=config.wandb_enabled)

    # Load checkpoint
    ckpt_path = epath.Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")

    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    train_state, train_state_sharding = load_checkpoint(config, mesh, ckpt_path)

    logging.info(f"Loaded train state at step {train_state.step}")
    sharding.log_param_sharding_actual(train_state.params)

    # Create validation data loader for evaluation
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        split="val",
        seed=config.seed,
        max_samples=getattr(config.data, "val_max_samples", None),
    )

    logging.info("Initialized validation data loader")

    # Create evaluator
    evaluator = TokenAccuracyEvaluator(config)
    peval_step = jax.jit(
        evaluator,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
    )

    # Determine number of evaluation batches
    if num_eval_batches is None:
        if getattr(config.data, "val_max_samples", None):
            process_count = getattr(jax, "process_count", lambda: 1)()
            local_bs = max(1, config.batch_size // process_count)
            num_eval_batches = math.ceil(config.data.val_max_samples / local_bs)
        else:
            # Default to 1000 batches if not specified
            num_eval_batches = 1000

    logging.info(f"Evaluating over {num_eval_batches} batches")

    # Run evaluation
    data_iter = iter(data_loader)
    eval_infos = []

    pbar = tqdm.tqdm(
        range(num_eval_batches),
        total=num_eval_batches,
        dynamic_ncols=True,
        desc="Evaluating token accuracy",
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

    results = {
        "mean_token_accuracy": float(final_results["token_accuracy"]),
        "mean_loss": float(final_results["loss"]),
        "std_token_accuracy": float(jnp.std(token_accuracies)),
        "std_loss": float(jnp.std(losses)),
        "min_token_accuracy": float(jnp.min(token_accuracies)),
        "max_token_accuracy": float(jnp.max(token_accuracies)),
        "num_batches_evaluated": len(eval_infos),
        "checkpoint_step": int(train_state.step),
    }

    # Log results
    logging.info("=" * 80)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 80)
    for key, value in results.items():
        logging.info(f"{key:30s}: {value}")
    logging.info("=" * 80)

    # Log to wandb
    if jax.process_index() == 0 and config.wandb_enabled:
        wandb.log(results)
        wandb.summary.update(results)

    return results


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate token accuracy on a checkpoint")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=None,
        help="Number of batches to evaluate (default: auto-detect from config)",
    )

    args, remaining_args = parser.parse_known_args()

    # Load config using the remaining arguments
    config = _config.cli(remaining_args)

    # Run evaluation
    results = main(config, args.checkpoint_path, args.num_batches)

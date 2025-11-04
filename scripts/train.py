import dataclasses
import datetime
import logging
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
import openpi_cot.training.log_util as log_util
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


def process_and_log_metrics(
    step: int,
    infos: list[dict[str, at.Array]],
    batch: tuple[CoTObservation | Observation, _model.Actions],
    dataset_stats_tracker: log_util.DatasetStatsTracker,
    dataset_info_buffer: log_util.LocalDatasetInfoBuffer,
    config: _config.TrainConfig,
    host_batch_cache: "HostBatchCache | None" = None,
    dataset_log_tracker: vis_tools.DatasetLogTracker | None = None,
    tok: PaligemmaCoTTokenizer | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """
    Unified function to process and log training/validation metrics.

    Args:
        step: Current training step
        infos: List of metric dictionaries to aggregate
        batch: Current batch data
        dataset_stats_tracker: Tracker for dataset-level statistics
        dataset_info_buffer: Buffer for local dataset info
        config: Training configuration
        host_batch_cache: Cache for host batches (only for training with langact)
        dataset_log_tracker: Tracker for dataset logging counts (only for training with langact)
        tok: Tokenizer for decoding (only for training with langact)
        prefix: Prefix for metric names (empty for train, "val_" for validation)

    Returns:
        Dictionary of reduced metrics
    """
    # Gather and update dataset stats from buffered local data
    dataset_info_buffer.gather_and_update_stats(dataset_stats_tracker)

    # Stack and reduce metrics
    stacked_infos = common_utils.stack_forest(infos)
    reduce_overrides = {
        "grad_norm": jnp.mean,
        "loss": jnp.mean,
        "param_norm": jnp.mean,
    }
    reduced_info = {}

    # Process metrics: average metrics go directly to logging, per_sample metrics are skipped
    for key, value in stacked_infos.items():
        if "per_sample_loss" in key:
            reduced_info[f"{prefix}max_{key}"] = jnp.max(value)
        elif "per_sample" in key:
            # Skip per_sample_* metrics - they're only used for dataset-level statistics
            continue
        else:
            # All other metrics are averaged and logged directly
            # For validation, add prefix to non-prefixed keys
            metric_key = key if key.startswith(prefix) else f"{prefix}{key}"
            reduced_info[metric_key] = reduce_overrides.get(key, jnp.mean)(value)

    reduced_info = jax.device_get(reduced_info)

    # Add dataset statistics to logging
    dataset_metrics = dataset_stats_tracker.get_metrics(prefix=prefix)
    reduced_info.update(dataset_metrics)

    info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
    mode = "val" if prefix else "train"
    logging.info(f"Step {step} ({mode}): {info_str}")

    if jax.process_index() == 0:
        wandb.log(reduced_info, step=step)

        # Create and log dataset statistics bar plots
        if dataset_stats_tracker.dataset_stats:
            plots = log_util.create_dataset_stats_plots(
                dataset_stats_tracker.dataset_stats, dataset_stats_tracker.cumulative_stats
            )
            log_util.log_dataset_plots(plots, step, prefix=prefix)

        # Training-specific logging: random examples and dataset log counts
        if not prefix and config.model.enable_langact_training:
            if host_batch_cache is not None and tok is not None and dataset_log_tracker is not None:
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

    return reduced_info


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
            metrics = model.compute_loss(rng, observation, actions, train=True, stage_config=stage_config)
            return jnp.mean(metrics["per_sample_loss"]), metrics

        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        diff_state = nnx.DiffState(0, self.config.trainable_filter)
        (loss, loss_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )

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
            "grad_norm": optax.global_norm(grads),
            "param_norm": optax.global_norm(kernel_params),
            **loss_metrics,
        }

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

        eval_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch

        # Call compute_loss to get per-sample metrics for dataset tracking
        # Note: We use the model in eval mode but request per-sample metrics by passing train=True
        # This is to enable dataset-level tracking during validation
        val_metrics = model.compute_loss(eval_rng, observation, actions, train=True)

        result = {}

        # Process all metrics from compute_loss
        for key, value in val_metrics.items():
            if key == "per_sample_loss":
                # total_loss is per-sample during validation, store it for dataset tracking
                result["per_sample_loss"] = value
                result["val_loss"] = jnp.mean(value)
            else:
                # Include all other metrics (accuracy, etc.)
                result[key] = value

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

    # ptrain_step = train_runner

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
            out_shardings=replicated_sharding,
        )
        # Determine how many validation batches to evaluate each time.
        # If a fixed validation subset size is configured, compute batches from it;
        # otherwise fall back to a heuristic constant divided by global batch size.
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
    dataset_stats_tracker = log_util.DatasetStatsTracker()
    dataset_info_buffer = log_util.LocalDatasetInfoBuffer(tok)

    # Track current training stage for transition detection
    current_stage_idx = None
    if config.training_schedule is not None:
        # Find the initial stage
        for i, stage in enumerate(config.training_schedule.stages):
            if stage.start_step <= start_step and (stage.end_step is None or start_step < stage.end_step):
                current_stage_idx = i
                logging.info(f"Starting training at step {start_step} in stage {i}")
                break

    num_val_batches = None

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

        # Buffer local dataset info (no multihost gathering at every step)
        # NOTE: We track pred_ and langact_ metrics separately for dataset-level stats
        log_util.buffer_dataset_metrics_from_batch(dataset_info_buffer, batch, info)

        if step % config.log_interval == 0:
            # Use unified logging function for training metrics
            process_and_log_metrics(
                step=step,
                infos=infos,
                batch=batch,
                dataset_stats_tracker=dataset_stats_tracker,
                dataset_info_buffer=dataset_info_buffer,
                config=config,
                host_batch_cache=host_batch_cache,
                dataset_log_tracker=dataset_log_tracker,
                tok=tok,
                prefix="",
            )

            infos = []
            # Reset dataset stats tracker to only track the next log_interval window
            dataset_stats_tracker.reset()
        # Periodic validation
        if config.do_val and step % getattr(config, "val_interval", 500) == 0:
            # Initialize validation dataset trackers
            val_dataset_stats_tracker = log_util.DatasetStatsTracker()
            val_dataset_info_buffer = log_util.LocalDatasetInfoBuffer(tok)

            with sharding.set_mesh(mesh):
                val_infos = []
                # Recreate a fresh iterator to ensure the same fixed validation subset each time.
                val_iter = iter(val_loader)

                # If num_val_batches not yet determined, iterate until StopIteration to count
                if num_val_batches is None:
                    logging.info("First validation run - determining total number of validation batches...")
                    val_batch_count = 0
                    try:
                        while True:
                            val_batch = next(val_iter)
                            val_info = pval_step(train_rng, train_state, val_batch)
                            val_info_local = jax.device_get(val_info)
                            val_infos.append(val_info_local)
                            log_util.buffer_dataset_metrics_from_batch(val_dataset_info_buffer, val_batch, val_info)
                            val_batch_count += 1
                    except StopIteration:
                        num_val_batches = val_batch_count
                        logging.info(f"Determined validation dataset has {num_val_batches} batches")
                else:
                    # Subsequent validation runs: use progress bar with known batch count
                    val_pbar = tqdm.tqdm(
                        range(num_val_batches),
                        initial=0,
                        total=num_val_batches,
                        dynamic_ncols=True,
                        disable=(jax.process_index() != 0),
                    )
                    try:
                        for _ in val_pbar:
                            val_batch = next(val_iter)
                            val_info = pval_step(train_rng, train_state, val_batch)
                            val_info_local = jax.device_get(val_info)
                            val_infos.append(val_info_local)
                            log_util.buffer_dataset_metrics_from_batch(val_dataset_info_buffer, val_batch, val_info)
                    except StopIteration:
                        logging.warning(
                            f"Validation ended early at {len(val_infos)} batches (expected {num_val_batches}). "
                            "This may indicate dataset size changed."
                        )

                # Use unified logging function for validation metrics
                process_and_log_metrics(
                    step=step,
                    infos=val_infos,
                    batch=val_batch,  # Use last val_batch for dataset info
                    dataset_stats_tracker=val_dataset_stats_tracker,
                    dataset_info_buffer=val_dataset_info_buffer,
                    config=config,
                    host_batch_cache=None,
                    dataset_log_tracker=None,
                    tok=None,
                    prefix="val_",
                )

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

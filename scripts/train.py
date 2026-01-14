import dataclasses
from dataclasses import replace
import datetime
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import matplotlib
from openpi.models import model as _model
from openpi.models.model import Observation
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import optax
from rail_tpu_utils import prevent_cross_region
import tqdm_loggable.auto as tqdm
import wandb

import openpi_cot.datasets.cot_data_loader as _data_loader
from openpi_cot.models.model_adapter import CoTObservation
from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer
import openpi_cot.training.checkpoints as _checkpoints
import openpi_cot.training.config as _config
from openpi_cot.training.eval_helpers import eval_checkpoint
import openpi_cot.training.log_util as log_util
import openpi_cot.training.mh_sharding as sharding
import openpi_cot.training.utils as training_utils
import openpi_cot.training.vis_tools as vis_tools
import openpi_cot.training.weight_loaders as _weight_loaders

matplotlib.use("Agg")  # Use non-interactive backend for remote environments


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


def _validate_loaded_params(expected: at.Params, got: at.Params, *, allow_partial: bool) -> dict[tuple[str, ...], Any]:
    flat_expected = traverse_util.flatten_dict(expected)
    flat_got = {k: v for k, v in traverse_util.flatten_dict(got).items() if not isinstance(v, jax.ShapeDtypeStruct)}

    unexpected = [k for k in flat_got if k not in flat_expected]
    if unexpected:
        sample = ", ".join("/".join(k) for k in unexpected[:8])
        raise ValueError(f"Loaded params contain unexpected keys (sample): {sample}")

    mismatches = []
    for key, value in flat_got.items():
        expected_value = flat_expected[key]
        if hasattr(value, "shape") and hasattr(expected_value, "shape") and value.shape != expected_value.shape:
            mismatches.append((key, f"shape {value.shape} != {expected_value.shape}"))
        if hasattr(value, "dtype") and hasattr(expected_value, "dtype") and value.dtype != expected_value.dtype:
            mismatches.append((key, f"dtype {value.dtype} != {expected_value.dtype}"))

    if mismatches:
        sample = ", ".join("/".join(k) + f" ({reason})" for k, reason in mismatches[:8])
        raise ValueError(f"Loaded params do not match expected shapes/dtypes (sample): {sample}")

    missing = set(flat_expected) - set(flat_got)
    all_missing = ", ".join("/".join(k) for k in list(missing))
    logging.info(f"Loaded params missing required keys: {all_missing}")
    if missing:
        if not allow_partial:
            all_missing = ", ".join("/".join(k) for k in list(missing))
            raise ValueError(f"Loaded params missing required keys: {all_missing}")
        logging.info("Weight loader missing %d params; using random init for them.", len(missing))

    return flat_got


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params, *, allow_partial: bool
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    flat_loaded = _validate_loaded_params(params_shape, loaded_params, allow_partial=allow_partial)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(flat_loaded)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    ema_decay, ema_params_enabled = config.get_ema_init()

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
            ema_decay=ema_decay,
            ema_params=None if not ema_params_enabled else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader,
        train_state_shape.params.to_pure_dict(),
        allow_partial=config.allow_partial_weights,
    )
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
        step: at.Int[at.ArrayLike, ""],
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
            loss, metrics = model.compute_loss(
                rng, observation, actions, train=True, stage_config=stage_config, return_augmented_images=True
            )
            return loss, metrics

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

        # Debug-friendly grad stats: keep the original (likely bf16) norm, plus a float32 norm for overflow checks.
        grad_norm_bf16 = optax.global_norm(grads)
        grad_norm_f32 = optax.global_norm(jax.tree.map(lambda g: g.astype(jnp.float32), grads))

        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
        ema_decay, ema_enabled = self.config.get_ema_decay_for_step(step)

        def _apply_ema(s):
            return dataclasses.replace(
                s,
                ema_params=jax.tree.map(
                    lambda old, new: ema_decay * old + (1 - ema_decay) * new,
                    s.ema_params,
                    new_params,
                ),
            )

        new_state = jax.lax.cond(ema_enabled, _apply_ema, lambda s: s, new_state)

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
            "grad_norm": grad_norm_bf16,
            "grad_norm_f32": grad_norm_f32,
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
        # Pass verbose_mode=True to enable detailed metrics for validation
        verbose_mode = self.config.model.verbose_mode
        val_loss, val_metrics = model.compute_loss(
            eval_rng, observation, actions, train=False, verbose_mode=verbose_mode
        )

        val_metrics["val_loss"] = val_loss

        return val_metrics


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

    log_util.log_mem("Before init train state")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)

    log_util.log_mem("After init train state")

    logging.info(f"Initialized train state (param shapes):\n{training_utils.array_tree_to_info(train_state.params)}")
    sharding.log_param_sharding_planned(train_state_sharding)
    sharding.log_param_sharding_actual(train_state.params)

    data_loader: _data_loader.CoTRLDSDataLoader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        # seed=config.seed + start_step,
        seed=config.seed,
        persistent_iterator=True,
    )
    tok = PaligemmaCoTTokenizer(max_len=300)

    # Initialize dataset log tracker for uniform sample logging across datasets
    dataset_log_tracker = vis_tools.DatasetLogTracker(tokenizer=tok)

    data_iter = iter(data_loader)
    log_util.log_mem("Before getting batch")
    batch = next(data_iter)
    vis_tools.vis_batch(batch, tok=tok, step=0)
    log_util.log_mem("After getting batch")
    logging.info("Successfully initialized dataloader and retrieved first batch")
    logging.info(f"Initialized data loader (shapes):\n{training_utils.array_tree_to_info(batch)}")
    sharding.log_batch_sharding(batch)
    # Restore checkpoint BEFORE creating iterator to ensure dataloader state is restored correctly
    dataloader_restored = False
    if resuming:
        try:
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader=data_loader)
            dataloader_restored = True
            logging.info("Successfully restored checkpoint and dataloader state")
        except Exception as e:
            logging.error(f"Failed to restore dataloader state: {e}")
            dataloader_restored = False

    # Get start step after restoring checkpoint (if resuming)
    start_step = int(train_state.step)

    if config.model.enable_langact_training and config.use_validation and config.use_eval:
        eval_data_loader = _data_loader.create_data_loader(
            replace(
                config,
                model=replace(config.model, verbose_mode=True),
            ),
            sharding=data_sharding,
            shuffle=False,
            split="val",
            seed=config.seed,
            max_samples=getattr(config.data, "val_max_samples", None),
            persistent_iterator=False,
        )

        eval_checkpoint(
            train_state,
            config,
            mesh,
            data_sharding,
            replicated_sharding,
            eval_data_loader,
            jax.random.fold_in(train_rng, train_state.step),
            train_state_sharding,
        )

    train_runner = TrainingStepRunner(config)
    ptrain_step = jax.jit(
        train_runner,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, None),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    if config.use_validation:
        hash_tables_cache = data_loader.dataset.hash_tables

        val_data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            split="val",
            seed=config.seed,
            max_samples=getattr(config.data, "val_max_samples", None),
            hash_tables=hash_tables_cache,
            persistent_iterator=False,
        )

        # franka_val_data_loader = _data_loader.create_data_loader(
        #     replace(
        #         config,
        #         # model=replace(config.model, verbose_mode=True),
        #         batch_size=128,
        #         data=replace(config.data, data_mix="franka_dataset", val_fraction=1.0),
        #     ),
        #     sharding=data_sharding,
        #     shuffle=False,
        #     split="val",
        #     seed=config.seed,
        #     max_samples=getattr(config.data, "val_max_samples", None),
        #     hash_tables=hash_tables_cache,
        #     persistent_iterator=False,
        # )

        num_val_batches = val_data_loader.num_val_batches()
        logging.info(f"Initial number of validation batches (from loader): {num_val_batches}")
        # Try to get dataset statistics
        val_dataset = getattr(val_data_loader, "dataset", None)
        if val_dataset:
            if hasattr(val_dataset, "dataset_statistics"):
                logging.info(f"Validation dataset statistics: {val_dataset.dataset_statistics}")
            if hasattr(val_dataset, "dataset_length"):
                logging.info(f"Validation dataset length: {val_dataset.dataset_length}")
        logging.info("=" * 80)

        # Try to obtain the tokenizer from the transform pipeline for decoding
        # tok = data_loader.tokenizer
        val_runner = ValidationStepRunner(config)
        pval_step = jax.jit(
            val_runner,
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    # Log preemption tracking information to wandb
    if jax.process_index() == 0:
        preemption_info = {
            "preemption/start_timestamp": training_start_timestamp,
            "preemption/is_resuming": float(resuming),
            "preemption/start_step": start_step,
            "preemption/dataloader_restored": float(dataloader_restored),
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
    host_batch_cache = vis_tools.HostBatchCache()
    val_host_batch_cache = vis_tools.HostBatchCache()
    verbose_mode = config.model.verbose_mode
    dataset_stats_tracker = log_util.DatasetStatsTracker() if verbose_mode else None
    dataset_info_buffer = log_util.LocalDatasetInfoBuffer(tok) if verbose_mode else None

    for step in pbar:
        # Profiling: Time training step
        # train_start = time.perf_counter()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch, step)

        # Extract augmented_images before appending to infos (to avoid reduction issues)
        augmented_images = info.pop("augmented_images", None)
        infos.append(info)

        if verbose_mode:
            # Buffer local dataset info (no multihost gathering at every step)
            # NOTE: We track pred_ and langact_ metrics separately for dataset-level stats
            log_util.buffer_dataset_metrics_from_batch(dataset_info_buffer, batch, info)
        
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

        if step % config.log_interval == 0:
            # Visualize augmented images (after augmentation in compute_loss)
            if augmented_images is not None:
                vis_tools.vis_augmented_images(augmented_images, step=step, prefix="train", num_samples=4)

            # Use unified logging function for training metrics
            log_util.process_and_log_metrics(
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
                verbose_mode=verbose_mode,
            )
            infos = []
            # Reset dataset stats tracker to only track the next log_interval window
            if verbose_mode:
                dataset_stats_tracker.reset()
        # Periodic validation
        if config.use_validation and step % getattr(config, "val_interval", 500) == 0:
            # Initialize validation dataset trackers
            val_dataset_stats_tracker = log_util.DatasetStatsTracker() if verbose_mode else None
            val_dataset_info_buffer = log_util.LocalDatasetInfoBuffer(tok) if verbose_mode else None

            with sharding.set_mesh(mesh):
                val_infos = []
                # Recreate a fresh iterator to ensure the same fixed validation subset each time.
                val_iter = iter(val_data_loader)

                val_batch = None

                # Subsequent validation runs: use progress bar with known batch count
                val_pbar = tqdm.tqdm(
                    range(num_val_batches),
                    initial=0,
                    total=num_val_batches,
                    dynamic_ncols=True,
                    disable=(jax.process_index() != 0),
                )
                for _ in val_pbar:
                    val_batch = next(val_iter)
                    val_info = pval_step(train_rng, train_state, val_batch)
                    # val_info_local = jax.device_get(val_info)
                    # val_infos.append(val_info_local)
                    val_infos.append(val_info)
                    if verbose_mode:
                        log_util.buffer_dataset_metrics_from_batch(val_dataset_info_buffer, val_batch, val_info)
                # Use unified logging function for validation metrics
                if val_batch:
                    log_util.process_and_log_metrics(
                        step=step,
                        infos=val_infos,
                        batch=val_batch,  # Use last val_batch for dataset info
                        dataset_stats_tracker=val_dataset_stats_tracker,
                        dataset_info_buffer=val_dataset_info_buffer,
                        config=config,
                        host_batch_cache=val_host_batch_cache,
                        dataset_log_tracker=dataset_log_tracker,
                        tok=tok,
                        prefix="val_",
                        verbose_mode=verbose_mode,
                    )

                # val_infos = []
                # # Recreate a fresh iterator to ensure the same fixed validation subset each time.
                # franka_val_iter = iter(franka_val_data_loader)

                # # Subsequent validation runs: use progress bar with known batch count
                # val_pbar = tqdm.tqdm(
                #     range(20),
                #     initial=0,
                #     total=20,
                #     dynamic_ncols=True,
                #     disable=(jax.process_index() != 0),
                # )
                # for _ in val_pbar:
                #     val_batch = next(franka_val_iter)
                #     val_info = pval_step(train_rng, train_state, val_batch)
                #     # val_info_local = jax.device_get(val_info)
                #     # val_infos.append(val_info_local)
                #     val_infos.append(val_info)

                # log_util.process_and_log_metrics(
                #     step=step,
                #     infos=val_infos,
                #     batch=val_batch,  # Use last val_batch for dataset info
                #     dataset_stats_tracker=val_dataset_stats_tracker,
                #     dataset_info_buffer=val_dataset_info_buffer,
                #     config=config,
                #     host_batch_cache=val_host_batch_cache,
                #     dataset_log_tracker=dataset_log_tracker,
                #     tok=tok,
                #     prefix="franka_val_",
                #     verbose_mode=False,
                # )

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps:
            if config.model.enable_langact_training and config.use_validation and config.use_eval:
                eval_checkpoint(
                    train_state,
                    config,
                    mesh,
                    data_sharding,
                    replicated_sharding,
                    eval_data_loader,
                    jax.random.fold_in(train_rng, train_state.step),
                    train_state_sharding,
                )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())

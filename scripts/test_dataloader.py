import dataclasses
import logging
import os
import platform

import etils.epath as epath
import jax
from jax.experimental import multihost_utils as mh
import numpy as np
from rail_tpu_utils import prevent_cross_region
import wandb

import openpi_cot.training.checkpoints as _checkpoints
import openpi_cot.training.config as _config
import openpi_cot.training.cot_data_loader as _data_loader
import openpi_cot.training.mh_sharding as sharding
import openpi_cot.training.utils as training_utils


def _is_tpu_runtime() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


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


def main(config: _config.TrainConfig):
    if (
        ("v6" in config.name and config.fsdp_devices > 8)
        or ("v4" in config.name and config.fsdp_devices > 4)
        or ("v5" in config.name and config.fsdp_devices > 8)
    ):
        jax.distributed.initialize()
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
    init_logging()
    logging.info(f"Local devices: {local_devices}, Global devices: {global_devices}, Process count: {process_count}")
    if process_count == 1:
        # Choose the largest divisor of available devices not exceeding configured fsdp_devices
        target = min(config.fsdp_devices, max(1, local_devices))
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
        assert global_devices % effective_fsdp_devices == 0

    logging.info(f"Running on: {platform.node()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

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
    )

    init_wandb(
        config,
        resuming=False,
        enabled=config.wandb_enabled,
        rewind_to_step=getattr(config, "rewind_to_step", None),
    )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
    )

    data_iter = iter(data_loader)
    # Fetch the correct first batch, advancing the iterator on resume
    logging.info("Before getting batch")

    batch = next(data_iter)
    logging.info("After getting batch")
    logging.info(f"Initialized data loader (shapes):\n{training_utils.array_tree_to_info(batch)}")
    # Sharding details for the first batch
    sharding.log_batch_sharding(batch)

    # Removed unused tokenizer extraction from private members

    for step in range(10):
        batch = next(data_iter)

        # ----- Inspect sharding on-device (before converting to local host array) -----
        # batch[0] is your CoTObservation; pick any leading-batched tensor on device:
        first_cam_key = next(iter(batch[0].images))
        imgs = batch[0].images[first_cam_key]  # jax.Array with NamedSharding(("data","fsdp"))

        # Log the global sharding once
        if step == 0 and jax.process_index() == 0:
            logging.info("JAX sharding for images: %s", imgs.sharding)
            try:
                viz = jax.debug.visualize_array_sharding(imgs)
                logging.info("Array sharding layout:\n%s", viz)
            except Exception:
                pass

        # Each host only owns its addressable shards. Confirm per-device microbatch size:
        # Expect 4 here with fsdp_devices=32, P=8, D=4, global_batch=128.
        local_device_shards = imgs.addressable_shards  # only the 4 shards on this host
        local_shapes = [s.data.shape for s in local_device_shards]
        if step == 0:
            for sd, shp in zip([s.device for s in local_device_shards], local_shapes):
                logging.info(f"[host {jax.process_index()}] shard on {sd}: {shp}")
            # Optional hard assertion: per-device leading dim is identical across local devices
            lead_dims = {shp[0] for shp in local_shapes}
            assert len(lead_dims) == 1, f"Inconsistent local shard sizes: {local_shapes}"
            logging.info(f"[host {jax.process_index()}] per-device microbatch = {next(iter(lead_dims))}")

        # ----- Your existing host-local view + (optional) global gather for logging -----
        local_imgs = training_utils.to_local_array(imgs)  # host-local: shape (local_bsz, ...)
        if getattr(jax, "process_count", lambda: 1)() > 1:
            # All hosts must participate
            global_imgs = mh.process_allgather(local_imgs, tiled=True)
        else:
            global_imgs = local_imgs

        if jax.process_index() == 0:
            logging.info(f"Images shape (global assembled for logging): {global_imgs.shape}")
            imgs_u8 = ((np.asarray(global_imgs) + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
            wandb.log({"camera_views": [wandb.Image(img) for img in imgs_u8]}, step=step)


if __name__ == "__main__":
    main(_config.cli())

"""Test script for dataloader checkpoint save/restore functionality.

This script mimics the training flow but focuses only on dataloader checkpointing,
without any model or training logic.

Usage:
    python scripts/test_dataloader_checkpoint.py --config-name=local
"""

import logging
import os
import platform
import tempfile
from pathlib import Path

import etils.epath as epath
import jax
import numpy as np
from rail_tpu_utils import prevent_cross_region

import openpi_cot.dataloader.cot_data_loader as _data_loader
import openpi_cot.training.config as _config
import openpi_cot.training.mh_sharding as sharding


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
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def init_tpu(config: _config.TrainConfig):
    """Initialize TPU/GPU environment."""

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
        os.environ["CURL_CA_BUNDLE"] = "/etc/pki/tls/certs/ca-bundle.crt"

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


def collect_batch_signature(batch):
    """Collect a signature of a batch for comparison."""
    obs, actions = batch
    # Get first image from first camera
    first_cam_key = next(iter(obs.images))
    first_image = obs.images[first_cam_key]

    # Create signature: mean of first image and first action
    signature = {
        "image_mean": float(np.mean(first_image)),
        "image_std": float(np.std(first_image)),
        "action_mean": float(np.mean(actions)),
        "action_std": float(np.std(actions)),
        "image_shape": first_image.shape,
        "action_shape": actions.shape,
    }
    return signature


def test_dataloader_checkpoint(config: _config.TrainConfig):
    """Test dataloader checkpoint save and restore."""
    init_logging()
    effective_fsdp_devices = init_tpu(config)

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    # Create temporary checkpoint directory
    temp_dir = tempfile.mkdtemp(prefix="dataloader_checkpoint_test_")
    checkpoint_dir = Path(temp_dir) / "iterator_checkpoint"
    logging.info(f"Using temporary checkpoint directory: {checkpoint_dir}")

    try:
        # ========== Phase 1: Initial data loading and checkpoint save ==========
        logging.info("=" * 80)
        logging.info("PHASE 1: Initial data loading")
        logging.info("=" * 80)

        data_loader1 = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
            seed=config.seed,
            persistent_iterator=True,
        )

        logging.info(f"Created data loader with persistent_iterator=True")

        # Check if persistent iterator was successfully initialized
        if hasattr(data_loader1, "_dataset") and hasattr(data_loader1._dataset, "persistent_iterator"):
            logging.info(f"Persistent iterator status: {data_loader1._dataset.persistent_iterator}")
            if data_loader1._dataset._tf_iterator is not None:
                logging.info("✓ TF iterator initialized successfully")
            else:
                logging.warning("✗ TF iterator is None - checkpointing may not work")

        # Create iterator and consume some batches
        data_iter1 = iter(data_loader1)
        logging.info("Created iterator")

        num_batches_before_checkpoint = 10
        batches_before = []
        logging.info(f"Consuming {num_batches_before_checkpoint} batches before checkpoint...")

        for i in range(num_batches_before_checkpoint):
            batch = next(data_iter1)
            sig = collect_batch_signature(batch)
            batches_before.append(sig)
            if i == 0:
                logging.info(f"  Batch {i}: image_shape={sig['image_shape']}, action_shape={sig['action_shape']}")
            else:
                logging.info(f"  Batch {i}: image_mean={sig['image_mean']:.4f}, action_mean={sig['action_mean']:.4f}")

        # Save checkpoint
        logging.info("Saving iterator checkpoint...")
        data_loader1.save_iterator_checkpoint(str(checkpoint_dir))

        # Verify checkpoint files were created
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*"))
            logging.info(f"✓ Checkpoint saved: {len(checkpoint_files)} files created")
            for f in checkpoint_files:
                logging.info(f"  - {f.name}")
        else:
            logging.error("✗ Checkpoint directory not created!")
            return False

        # Get the next few batches (these should match after restoration)
        num_batches_after_checkpoint = 5
        expected_batches = []
        logging.info(f"Consuming {num_batches_after_checkpoint} batches after checkpoint (expected after restore)...")

        for i in range(num_batches_after_checkpoint):
            batch = next(data_iter1)
            sig = collect_batch_signature(batch)
            expected_batches.append(sig)
            logging.info(
                f"  Expected batch {i}: image_mean={sig['image_mean']:.4f}, action_mean={sig['action_mean']:.4f}"
            )

        # Clean up first loader
        del data_iter1
        del data_loader1

        # ========== Phase 2: Simulate restart and restore checkpoint ==========
        logging.info("")
        logging.info("=" * 80)
        logging.info("PHASE 2: Simulating restart and checkpoint restore")
        logging.info("=" * 80)

        # Create new data loader (simulates restart)
        data_loader2 = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
            seed=config.seed,  # Same seed
            persistent_iterator=True,
        )

        logging.info("Created new data loader (simulating restart)")

        # Restore checkpoint BEFORE creating iterator
        logging.info("Restoring iterator checkpoint...")
        data_loader2.restore_iterator_checkpoint(str(checkpoint_dir))
        logging.info("✓ Checkpoint restored")

        # Create iterator AFTER restoration
        data_iter2 = iter(data_loader2)
        logging.info("Created iterator after restoration")

        # Get batches and compare with expected
        logging.info(f"Verifying {num_batches_after_checkpoint} batches match expected...")
        all_match = True

        for i in range(num_batches_after_checkpoint):
            batch = next(data_iter2)
            sig = collect_batch_signature(batch)
            expected = expected_batches[i]

            # Compare signatures
            image_mean_match = np.isclose(sig["image_mean"], expected["image_mean"], rtol=1e-5, atol=1e-6)
            action_mean_match = np.isclose(sig["action_mean"], expected["action_mean"], rtol=1e-5, atol=1e-6)
            shape_match = sig["image_shape"] == expected["image_shape"] and sig["action_shape"] == expected["action_shape"]

            match = image_mean_match and action_mean_match and shape_match

            if match:
                logging.info(
                    f"  ✓ Batch {i} matches: "
                    f"image_mean={sig['image_mean']:.4f} (expected {expected['image_mean']:.4f}), "
                    f"action_mean={sig['action_mean']:.4f} (expected {expected['action_mean']:.4f})"
                )
            else:
                logging.error(
                    f"  ✗ Batch {i} MISMATCH: "
                    f"image_mean={sig['image_mean']:.4f} vs {expected['image_mean']:.4f}, "
                    f"action_mean={sig['action_mean']:.4f} vs {expected['action_mean']:.4f}"
                )
                all_match = False

        # ========== Phase 3: Results ==========
        logging.info("")
        logging.info("=" * 80)
        logging.info("TEST RESULTS")
        logging.info("=" * 80)

        if all_match:
            logging.info("✓✓✓ SUCCESS: All batches after restoration match expected values!")
            logging.info("✓ Iterator state was correctly saved and restored")
            logging.info("✓ Dataloader checkpoint functionality works correctly")
            return True
        else:
            logging.error("✗✗✗ FAILURE: Some batches did not match expected values")
            logging.error("✗ Iterator state may not have been properly saved/restored")
            return False

    finally:
        # Cleanup
        import shutil

        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temporary directory: {temp_dir}")


def main(config: _config.TrainConfig):
    """Main entry point."""
    success = test_dataloader_checkpoint(config)
    if success:
        logging.info("🎉 Test passed!")
        exit(0)
    else:
        logging.error("❌ Test failed!")
        exit(1)


if __name__ == "__main__":
    main(_config.cli())

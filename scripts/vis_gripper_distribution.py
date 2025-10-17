#!/usr/bin/env python3
"""
Lightweight gripper state value distribution visualizer.

Extracts gripper states (last dimension of non-padded state) from dataset batches
and visualizes their distribution.

Usage:
    python scripts/vis_gripper_distribution.py CONFIG_NAME --data.data-mix=DATASET_NAME
    python scripts/vis_gripper_distribution.py pi_combined_cot_v4 --data.data-mix=bc_z --num-batches=100
"""

import logging
import os

import etils.epath as epath
import jax
import jax.experimental.multihost_utils as multihost_utils
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

import openpi_cot.dataloader.cot_data_loader as _data_loader
import openpi_cot.training.config as _config
import openpi_cot.training.mh_sharding as sharding

try:
    import wandb
except ImportError:
    wandb = None


def _safe_device_get(arr):
    """Safely get array to host, handling multi-host sharded arrays."""
    if arr is None:
        return None
    try:
        # Try direct device_get first (works for single-host or local shards)
        return jax.device_get(arr)
    except RuntimeError as e:
        if "non-addressable" in str(e):
            # Array spans multiple hosts, need to gather
            try:
                gathered = multihost_utils.process_allgather(arr, tiled=True)
                return jax.device_get(gathered)
            except Exception as gather_error:
                logging.warning(f"Failed to gather array: {gather_error}")
                return None
        raise


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

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


def extract_gripper_states(batch, state_padding_mask=None) -> np.ndarray:
    """
    Extract gripper states from a batch.

    Args:
        batch: Batch from the data loader
        state_padding_mask: Optional padding mask to filter out padded values

    Returns:
        Array of gripper state values (non-padded)
    """
    obs = batch[0]
    state = _safe_device_get(obs.state)
    if state is None:
        return np.array([])
    # Last dimension is gripper
    gripper_states = state[..., 6]
    gripper_states = gripper_states.flatten()
    return gripper_states


def extract_gripper_actions(batch) -> np.ndarray:
    """
    Extract gripper actions from a batch.

    Args:
        batch: Batch from the data loader

    Returns:
        Array of gripper action values (non-padded)
    """
    actions = batch[1]
    actions_data = _safe_device_get(actions)
    if actions_data is None:
        return np.array([])
    # Last dimension is gripper action
    gripper_actions = actions_data[..., 6]
    gripper_actions = gripper_actions.flatten()
    return gripper_actions


def get_image_at_index(all_batches, global_idx, camera_key="primary"):
    """
    Extract image at a specific global flattened index across all batches.

    Args:
        all_batches: List of batches
        global_idx: Global flattened index
        camera_key: Camera view to extract (default "primary")

    Returns:
        Image as numpy array (uint8) or None if not found
    """
    # Calculate which batch and local index
    cumulative = 0
    for batch in all_batches:
        obs = batch[0]
        state = _safe_device_get(obs.state)
        if state is None:
            continue
        batch_size, seq_len = state.shape[0], state.shape[1]
        seq_len = 1
        batch_total = batch_size * seq_len

        if cumulative + batch_total > global_idx:
            # This batch contains the index
            local_idx = global_idx - cumulative
            batch_idx = local_idx // seq_len
            seq_idx = local_idx % seq_len

            # Get image from this batch
            if hasattr(obs, "images") and camera_key in obs.images:
                img = _safe_device_get(obs.images[camera_key][batch_idx, seq_idx])
                if img is None:
                    return None, None
                # Convert from [-1, 1] to [0, 255]
                img_u8 = np.asarray(((img + 1.0) * 0.5 * 255.0).clip(0, 255), dtype=np.uint8)
                return img_u8, state[batch_idx, 6]  # Return image and gripper value

        cumulative += batch_total

    return None, None


def plot_gripper_distribution(gripper_values: np.ndarray, dataset_name: str):
    """
    Create visualization of gripper state distribution.

    Args:
        gripper_values: Array of gripper state values
        dataset_name: Name of the dataset

    Returns:
        matplotlib Figure object
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Gripper State Distribution: {dataset_name}", fontsize=16, fontweight="bold")

    # 1. Histogram with KDE
    ax1 = axes[0, 0]
    ax1.hist(gripper_values, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")

    # Only attempt KDE if we have sufficient variance in the data
    if len(gripper_values) > 1 and np.std(gripper_values) > 1e-10:
        try:
            from scipy import stats

            kde = stats.gaussian_kde(gripper_values)
            x_range = np.linspace(gripper_values.min(), gripper_values.max(), 200)
            ax1.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
            ax1.legend()
        except (ImportError, np.linalg.LinAlgError):
            # Skip KDE if scipy not available or data is degenerate (all same values)
            pass

    ax1.set_xlabel("Gripper State Value", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Distribution with KDE", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # 2. Box plot
    ax2 = axes[0, 1]
    box_data = ax2.boxplot(gripper_values, vert=True, patch_artist=True, widths=0.5)
    box_data["boxes"][0].set_facecolor("lightblue")
    box_data["medians"][0].set_color("red")
    box_data["medians"][0].set_linewidth(2)
    ax2.set_ylabel("Gripper State Value", fontsize=12)
    ax2.set_title("Box Plot", fontsize=13, fontweight="bold")
    ax2.set_xticklabels(["Gripper State"])
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.5)

    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_values = np.sort(gripper_values)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax3.plot(sorted_values, cumulative, linewidth=2, color="green")
    ax3.set_xlabel("Gripper State Value", fontsize=12)
    ax3.set_ylabel("Cumulative Probability", fontsize=12)
    ax3.set_title("Cumulative Distribution Function", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Compute statistics
    stats_text = f"""
    Dataset Statistics
    {"=" * 35}

    Sample Count:    {len(gripper_values):,}

    Mean:            {np.mean(gripper_values):.6f}
    Median:          {np.median(gripper_values):.6f}
    Std Dev:         {np.std(gripper_values):.6f}

    Min:             {np.min(gripper_values):.6f}
    Max:             {np.max(gripper_values):.6f}
    Range:           {np.ptp(gripper_values):.6f}

    Q1 (25%):        {np.percentile(gripper_values, 25):.6f}
    Q3 (75%):        {np.percentile(gripper_values, 75):.6f}
    IQR:             {np.percentile(gripper_values, 75) - np.percentile(gripper_values, 25):.6f}

    Unique Values:   {len(np.unique(gripper_values)):,}

    % Near 0 (<0.1): {100 * np.mean(gripper_values < 0.1):.2f}%
    % Near 1 (>0.9): {100 * np.mean(gripper_values > 0.9):.2f}%
    """

    ax4.text(
        0.1,
        0.9,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    return fig


def main(config: _config.TrainConfig):
    init_logging()

    # Get dataset name from config
    dataset_name = getattr(config.data, "data_mix", "unknown")
    num_batches = 200

    logging.info(f"Analyzing gripper distribution for dataset: {dataset_name}")
    logging.info(f"Will process {num_batches} batches")

    # Initialize wandb
    wandb_enabled = bool(getattr(config, "wandb_enabled", False)) and wandb is not None
    if not wandb_enabled:
        if bool(getattr(config, "wandb_enabled", False)) and wandb is None:
            logging.warning("wandb requested but not installed; falling back to local image dumps.")
    else:
        wandb_mode = "online" if os.environ.get("WANDB_DISABLED", "false").lower() not in {"1", "true"} else "offline"
        run_name = f"gripper-dist-{dataset_name}"
        exp_name = getattr(config, "exp_name", None)
        if exp_name:
            run_name = f"{run_name}-{exp_name}"
        wandb_config = {
            "config_name": config.name,
            "exp_name": exp_name,
            "dataset": dataset_name,
            "num_batches": num_batches,
            "project_name": getattr(config, "project_name", "openpi-cot"),
        }
        wandb.init(
            project=getattr(config, "project_name", "openpi-cot"),
            name=run_name,
            config=wandb_config,
            reinit=True,
            mode=wandb_mode,
        )
        if hasattr(wandb.run, "log_code"):
            wandb.run.log_code(str(epath.Path(__file__).parent.parent))

    # Initialize JAX
    process_count = getattr(jax, "process_count", lambda: 1)()
    local_devices = getattr(jax, "local_device_count", lambda: 1)()
    global_devices = getattr(jax, "device_count", lambda: local_devices)()

    logging.info(f"Local devices: {local_devices}, Global devices: {global_devices}, Process count: {process_count}")

    # Determine effective FSDP devices
    if process_count == 1:
        target = min(config.fsdp_devices, max(1, local_devices))
        effective_fsdp_devices = 1
        for d in range(target, 0, -1):
            if global_devices % d == 0:
                effective_fsdp_devices = d
                break
    else:
        effective_fsdp_devices = config.fsdp_devices
        assert global_devices % effective_fsdp_devices == 0

    # Create mesh and sharding
    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    # Create data loader
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
    )

    # Collect gripper states and track batches for min/max visualization
    logging.info("Collecting gripper states from batches...")
    all_gripper_states = []
    all_gripper_actions = []
    all_batches = []  # Store batches to find min/max later

    data_iter = iter(data_loader)

    for batch_idx in range(num_batches):
        try:
            batch = next(data_iter)
            gripper_states = extract_gripper_states(batch)
            gripper_actions = extract_gripper_actions(batch)

            if len(gripper_states) > 0:
                all_gripper_states.append(gripper_states)
                all_batches.append(batch)  # Store the batch
                logging.info(f"Batch {batch_idx + 1}/{num_batches}: Collected {len(gripper_states)} gripper states")
            else:
                logging.warning(f"Batch {batch_idx + 1}/{num_batches}: No gripper states found")

            if len(gripper_actions) > 0:
                all_gripper_actions.append(gripper_actions)
                logging.info(f"Batch {batch_idx + 1}/{num_batches}: Collected {len(gripper_actions)} gripper actions")
            else:
                logging.warning(f"Batch {batch_idx + 1}/{num_batches}: No gripper actions found")

        except StopIteration:
            logging.warning(f"Data iterator exhausted at batch {batch_idx + 1}")
            break
        except Exception as e:
            logging.error(f"Error processing batch {batch_idx + 1}: {e}")
            continue

    if not all_gripper_states:
        logging.error("No gripper states collected! Check dataset configuration.")
        return

    # Concatenate all collected states
    all_gripper_states = np.concatenate(all_gripper_states)
    logging.info(f"Total gripper states collected: {len(all_gripper_states):,}")

    # Concatenate all collected actions
    if all_gripper_actions:
        all_gripper_actions = np.concatenate(all_gripper_actions)
        logging.info(f"Total gripper actions collected: {len(all_gripper_actions):,}")

        # Check if states and actions have same length (they should if extracted from same batches)
        if len(all_gripper_actions) != len(all_gripper_states):
            logging.warning(f"Mismatch: {len(all_gripper_actions)} actions vs {len(all_gripper_states)} states")
            logging.warning("This may cause issues with image extraction for actions!")
        else:
            logging.info("✓ Gripper actions and states have matching lengths")
    else:
        all_gripper_actions = np.array([])
        logging.warning("No gripper actions collected!")

    # Find top-k min/max gripper states and their locations
    num_examples = 10  # Number of min/max examples to visualize

    # Get indices of top-k smallest and largest values
    min_indices = np.argpartition(all_gripper_states, num_examples)[:num_examples]
    max_indices = np.argpartition(all_gripper_states, -num_examples)[-num_examples:]

    # Sort them by value for nicer display
    min_indices = min_indices[np.argsort(all_gripper_states[min_indices])]
    max_indices = max_indices[np.argsort(all_gripper_states[max_indices])[::-1]]

    logging.info(f"Finding {num_examples} min and {num_examples} max gripper examples...")
    logging.info(f"Min gripper values: {all_gripper_states[min_indices]}")
    logging.info(f"Max gripper values: {all_gripper_states[max_indices]}")

    # Extract images for min/max gripper states
    logging.info("Extracting images for min/max gripper states...")

    # Detect available camera keys from the first batch
    camera_keys_to_try = ["base_0_rgb", "left_wrist_0_rgb"]
    if len(all_batches) > 0:
        obs = all_batches[0][0]
        if hasattr(obs, "images"):
            available_cameras = list(obs.images.keys())
            logging.info(f"Available camera keys in dataset: {available_cameras}")
            camera_keys_to_try = available_cameras
        else:
            logging.warning("No images attribute found in observations")
    else:
        logging.warning("No batches available to detect camera keys")

    camera_images = {}  # {camera_key: {"min": [...], "max": [...]}}

    for cam_key in camera_keys_to_try:
        min_images_cam = []
        max_images_cam = []

        # Extract min examples
        for idx in min_indices:
            img, gripper_val = get_image_at_index(all_batches, idx, cam_key)
            if img is not None:
                min_images_cam.append((img, gripper_val, idx))

        # Extract max examples
        for idx in max_indices:
            img, gripper_val = get_image_at_index(all_batches, idx, cam_key)
            if img is not None:
                max_images_cam.append((img, gripper_val, idx))

        # Limit to num_examples
        min_images_cam = min_images_cam[:num_examples]
        max_images_cam = max_images_cam[:num_examples]

        if len(min_images_cam) > 0 or len(max_images_cam) > 0:
            camera_images[cam_key] = {
                "min": min_images_cam,
                "max": max_images_cam,
            }
            logging.info(f"Camera '{cam_key}': Found {len(min_images_cam)} min and {len(max_images_cam)} max images")

    logging.info(f"Extracted images from {len(camera_images)} camera views for gripper states")

    # Extract images for min/max gripper actions
    camera_images_actions = {}  # {camera_key: {"min": [...], "max": [...]}}

    if len(all_gripper_actions) > 0:
        # Find top-k min/max gripper actions and their locations
        action_min_indices = np.argpartition(all_gripper_actions, num_examples)[:num_examples]
        action_max_indices = np.argpartition(all_gripper_actions, -num_examples)[-num_examples:]

        # Sort them by value for nicer display
        action_min_indices = action_min_indices[np.argsort(all_gripper_actions[action_min_indices])]
        action_max_indices = action_max_indices[np.argsort(all_gripper_actions[action_max_indices])[::-1]]

        logging.info(f"Finding {num_examples} min and {num_examples} max gripper action examples...")
        logging.info(f"Min gripper action values: {all_gripper_actions[action_min_indices]}")
        logging.info(f"Max gripper action values: {all_gripper_actions[action_max_indices]}")
        logging.info(f"Min gripper action indices: {action_min_indices}")
        logging.info(f"Max gripper action indices: {action_max_indices}")
        logging.info(f"Total gripper actions: {len(all_gripper_actions)}, Total gripper states: {len(all_gripper_states)}")

        # Extract images for min/max gripper actions
        logging.info("Extracting images for min/max gripper actions...")
        logging.info(f"Using camera keys: {camera_keys_to_try}")
        logging.info(f"Number of batches available: {len(all_batches)}")

        # Debug: Check if indices are within valid range
        total_samples = sum(
            _safe_device_get(batch[0].state).shape[0] if _safe_device_get(batch[0].state) is not None else 0
            for batch in all_batches
        )
        logging.info(f"Total samples across all batches: {total_samples}")
        if len(action_min_indices) > 0:
            logging.info(f"Sample action min index to lookup: {action_min_indices[0]} (value: {all_gripper_actions[action_min_indices[0]]:.6f})")

        for cam_key in camera_keys_to_try:
            min_images_cam = []
            max_images_cam = []

            # Extract min examples
            found_count = 0
            for idx in action_min_indices:
                img, gripper_val = get_image_at_index(all_batches, idx, cam_key)
                if img is not None:
                    # Get the actual action value instead of state value
                    action_val = all_gripper_actions[idx]
                    min_images_cam.append((img, action_val, idx))
                    found_count += 1
            logging.info(f"Camera '{cam_key}': Found {found_count}/{len(action_min_indices)} min action images")

            # Extract max examples
            found_count = 0
            for idx in action_max_indices:
                img, gripper_val = get_image_at_index(all_batches, idx, cam_key)
                if img is not None:
                    # Get the actual action value instead of state value
                    action_val = all_gripper_actions[idx]
                    max_images_cam.append((img, action_val, idx))
                    found_count += 1
            logging.info(f"Camera '{cam_key}': Found {found_count}/{len(action_max_indices)} max action images")

            # Limit to num_examples
            min_images_cam = min_images_cam[:num_examples]
            max_images_cam = max_images_cam[:num_examples]

            if len(min_images_cam) > 0 or len(max_images_cam) > 0:
                camera_images_actions[cam_key] = {
                    "min": min_images_cam,
                    "max": max_images_cam,
                }
                logging.info(f"Camera '{cam_key}': Found {len(min_images_cam)} min and {len(max_images_cam)} max action images")

        logging.info(f"Extracted images from {len(camera_images_actions)} camera views for gripper actions")

    # Print summary statistics
    logging.info("=" * 60)
    logging.info(f"Gripper State Statistics for {dataset_name}")
    logging.info("=" * 60)
    logging.info(f"Sample count:     {len(all_gripper_states):,}")
    logging.info(f"Mean:             {np.mean(all_gripper_states):.6f}")
    logging.info(f"Median:           {np.median(all_gripper_states):.6f}")
    logging.info(f"Std:              {np.std(all_gripper_states):.6f}")
    logging.info(f"Min:              {np.min(all_gripper_states):.6f}")
    logging.info(f"Max:              {np.max(all_gripper_states):.6f}")
    logging.info(f"% Near 0 (<0.1):  {100 * np.mean(all_gripper_states < 0.1):.2f}%")
    logging.info(f"% Near 1 (>0.9):  {100 * np.mean(all_gripper_states > 0.9):.2f}%")
    logging.info("=" * 60)

    # Print gripper action statistics
    if len(all_gripper_actions) > 0:
        logging.info("")
        logging.info("=" * 60)
        logging.info(f"Gripper Action Statistics for {dataset_name}")
        logging.info("=" * 60)
        logging.info(f"Sample count:     {len(all_gripper_actions):,}")
        logging.info(f"Mean:             {np.mean(all_gripper_actions):.6f}")
        logging.info(f"Median:           {np.median(all_gripper_actions):.6f}")
        logging.info(f"Std:              {np.std(all_gripper_actions):.6f}")
        logging.info(f"Min:              {np.min(all_gripper_actions):.6f}")
        logging.info(f"Max:              {np.max(all_gripper_actions):.6f}")
        logging.info(f"% Near -1 (<-0.9): {100 * np.mean(all_gripper_actions < -0.9):.2f}%")
        logging.info(f"% Near 0 (±0.1):   {100 * np.mean(np.abs(all_gripper_actions) < 0.1):.2f}%")
        logging.info(f"% Near 1 (>0.9):   {100 * np.mean(all_gripper_actions > 0.9):.2f}%")
        logging.info("=" * 60)

    # Create visualization
    fig = plot_gripper_distribution(all_gripper_states, dataset_name)

    # Log to wandb or save locally
    if wandb_enabled and wandb is not None:
        log_dict = {
            "gripper_distribution": wandb.Image(fig),
            "stats/state/sample_count": len(all_gripper_states),
            "stats/state/mean": np.mean(all_gripper_states),
            "stats/state/median": np.median(all_gripper_states),
            "stats/state/std": np.std(all_gripper_states),
            "stats/state/min": np.min(all_gripper_states),
            "stats/state/max": np.max(all_gripper_states),
            "stats/state/pct_near_0": 100 * np.mean(all_gripper_states < 0.1),
            "stats/state/pct_near_1": 100 * np.mean(all_gripper_states > 0.9),
        }

        # Add gripper action statistics
        if len(all_gripper_actions) > 0:
            log_dict.update(
                {
                    "stats/action/sample_count": len(all_gripper_actions),
                    "stats/action/mean": np.mean(all_gripper_actions),
                    "stats/action/median": np.median(all_gripper_actions),
                    "stats/action/std": np.std(all_gripper_actions),
                    "stats/action/min": np.min(all_gripper_actions),
                    "stats/action/max": np.max(all_gripper_actions),
                    "stats/action/pct_near_neg1": 100 * np.mean(all_gripper_actions < -0.9),
                    "stats/action/pct_near_0": 100 * np.mean(np.abs(all_gripper_actions) < 0.1),
                    "stats/action/pct_near_1": 100 * np.mean(all_gripper_actions > 0.9),
                }
            )

        # Add min/max images from all cameras if available
        for cam_key, images_dict in camera_images.items():
            min_images = images_dict["min"]
            max_images = images_dict["max"]

            # Clean camera key for wandb (replace underscores with dashes for better display)
            cam_name = cam_key.replace("_", "-")

            if len(min_images) > 0:
                log_dict[f"examples/{cam_name}/min"] = [
                    wandb.Image(img, caption=f"Min #{i + 1}: {val:.6f} (idx={idx})")
                    for i, (img, val, idx) in enumerate(min_images)
                ]
                logging.info(f"Logging {len(min_images)} min gripper images for {cam_key}")

            if len(max_images) > 0:
                log_dict[f"examples/{cam_name}/max"] = [
                    wandb.Image(img, caption=f"Max #{i + 1}: {val:.6f} (idx={idx})")
                    for i, (img, val, idx) in enumerate(max_images)
                ]
                logging.info(f"Logging {len(max_images)} max gripper images for {cam_key}")

        if len(camera_images) == 0:
            logging.warning("No gripper state images found from any camera")

        # Add min/max action images from all cameras if available
        for cam_key, images_dict in camera_images_actions.items():
            min_images = images_dict["min"]
            max_images = images_dict["max"]

            # Clean camera key for wandb (replace underscores with dashes for better display)
            cam_name = cam_key.replace("_", "-")

            if len(min_images) > 0:
                log_dict[f"examples/{cam_name}/action-min"] = [
                    wandb.Image(img, caption=f"Action Min #{i + 1}: {val:.6f} (idx={idx})")
                    for i, (img, val, idx) in enumerate(min_images)
                ]
                logging.info(f"Logging {len(min_images)} min gripper action images for {cam_key}")

            if len(max_images) > 0:
                log_dict[f"examples/{cam_name}/action-max"] = [
                    wandb.Image(img, caption=f"Action Max #{i + 1}: {val:.6f} (idx={idx})")
                    for i, (img, val, idx) in enumerate(max_images)
                ]
                logging.info(f"Logging {len(max_images)} max gripper action images for {cam_key}")

        if len(camera_images_actions) == 0 and len(all_gripper_actions) > 0:
            logging.warning("No gripper action images found from any camera")

        wandb.log(log_dict)
        logging.info("Logged gripper distribution and examples to wandb")
        plt.close(fig)
    else:
        # Fallback: save locally
        output_dir = os.environ.get("OPENPI_OUTPUT_DIR", ".")
        save_path = os.path.join(output_dir, f"gripper_dist_{dataset_name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved plot to: {save_path}")
        plt.close(fig)

        # Save min/max images locally for all cameras
        for cam_key, images_dict in camera_images.items():
            min_images = images_dict["min"]
            max_images = images_dict["max"]

            for i, (img, val, idx) in enumerate(min_images):
                min_img_path = os.path.join(output_dir, f"gripper_min_{cam_key}_{i:02d}_{dataset_name}.png")
                plt.imsave(min_img_path, img)
                logging.info(f"Saved {cam_key} min gripper image #{i + 1} (val={val:.6f}) to: {min_img_path}")

            for i, (img, val, idx) in enumerate(max_images):
                max_img_path = os.path.join(output_dir, f"gripper_max_{cam_key}_{i:02d}_{dataset_name}.png")
                plt.imsave(max_img_path, img)
                logging.info(f"Saved {cam_key} max gripper image #{i + 1} (val={val:.6f}) to: {max_img_path}")

        # Save min/max action images locally for all cameras
        for cam_key, images_dict in camera_images_actions.items():
            min_images = images_dict["min"]
            max_images = images_dict["max"]

            for i, (img, val, idx) in enumerate(min_images):
                min_img_path = os.path.join(output_dir, f"gripper_action_min_{cam_key}_{i:02d}_{dataset_name}.png")
                plt.imsave(min_img_path, img)
                logging.info(f"Saved {cam_key} min gripper action image #{i + 1} (val={val:.6f}) to: {min_img_path}")

            for i, (img, val, idx) in enumerate(max_images):
                max_img_path = os.path.join(output_dir, f"gripper_action_max_{cam_key}_{i:02d}_{dataset_name}.png")
                plt.imsave(max_img_path, img)
                logging.info(f"Saved {cam_key} max gripper action image #{i + 1} (val={val:.6f}) to: {max_img_path}")

    # Finish wandb run
    if wandb_enabled and wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.finish()


if __name__ == "__main__":
    # Parse config from command line
    config = _config.cli()
    main(config)

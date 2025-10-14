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

    # Get state - typically shape [batch_size, seq_len, state_dim]
    if hasattr(obs, "state") and obs.state is not None:
        state = jax.device_get(obs.state)
        # Last dimension is gripper
        gripper_states = state[..., -1]

        # Apply padding mask if available
        if state_padding_mask is not None:
            mask = jax.device_get(state_padding_mask)
            gripper_states = gripper_states[~mask.astype(bool)]
        else:
            # Flatten all values
            gripper_states = gripper_states.flatten()

        return gripper_states

    # Fallback: try to get gripper_state directly
    if hasattr(obs, "gripper_state") and obs.gripper_state is not None:
        gripper_states = jax.device_get(obs.gripper_state)
        return gripper_states.flatten()

    return np.array([])


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
    if len(gripper_values) > 1:
        try:
            from scipy import stats

            kde = stats.gaussian_kde(gripper_values)
            x_range = np.linspace(gripper_values.min(), gripper_values.max(), 200)
            ax1.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
            ax1.legend()
        except ImportError:
            pass  # Skip KDE if scipy not available
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

    # Collect gripper states
    logging.info("Collecting gripper states from batches...")
    all_gripper_states = []

    data_iter = iter(data_loader)

    for batch_idx in range(num_batches):
        try:
            batch = next(data_iter)
            gripper_states = extract_gripper_states(batch)

            if len(gripper_states) > 0:
                all_gripper_states.append(gripper_states)
                logging.info(f"Batch {batch_idx + 1}/{num_batches}: Collected {len(gripper_states)} gripper states")
            else:
                logging.warning(f"Batch {batch_idx + 1}/{num_batches}: No gripper states found")

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

    # Create visualization
    fig = plot_gripper_distribution(all_gripper_states, dataset_name)

    # Log to wandb or save locally
    if wandb_enabled and wandb is not None:
        wandb.log(
            {
                "gripper_distribution": wandb.Image(fig),
                "stats/sample_count": len(all_gripper_states),
                "stats/mean": np.mean(all_gripper_states),
                "stats/median": np.median(all_gripper_states),
                "stats/std": np.std(all_gripper_states),
                "stats/min": np.min(all_gripper_states),
                "stats/max": np.max(all_gripper_states),
                "stats/pct_near_0": 100 * np.mean(all_gripper_states < 0.1),
                "stats/pct_near_1": 100 * np.mean(all_gripper_states > 0.9),
            }
        )
        logging.info("Logged gripper distribution to wandb")
        plt.close(fig)
    else:
        # Fallback: save locally
        output_dir = os.environ.get("OPENPI_OUTPUT_DIR", ".")
        save_path = os.path.join(output_dir, f"gripper_dist_{dataset_name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved plot to: {save_path}")
        plt.close(fig)

    # Finish wandb run
    if wandb_enabled and wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.finish()


if __name__ == "__main__":
    # Parse config from command line
    config = _config.cli()

    # Add num_batches argument if not present
    if not hasattr(config, "num_batches"):
        import sys

        for i, arg in enumerate(sys.argv):
            if arg == "--num-batches" and i + 1 < len(sys.argv):
                config.num_batches = int(sys.argv[i + 1])
                break
        else:
            config.num_batches = 50  # Default

    main(config)

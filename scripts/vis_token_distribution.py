"""Visualize token distributions from the dataloader.

This script analyzes and visualizes:
1. Number tokens distribution
2. Number + direction tokens distribution
3. Per-dataset token distributions
4. State value distributions
"""

from collections import Counter
from collections import defaultdict
import logging
import os
import platform
import re

import etils.epath as epath
import jax
import jax.experimental.multihost_utils as multihost_utils
import matplotlib.pyplot as plt
import numpy as np
from rail_tpu_utils import prevent_cross_region
import wandb

import openpi_cot.dataloader.cot_data_loader as _data_loader
from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer
import openpi_cot.training.config as _config
import openpi_cot.training.mh_sharding as sharding


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
    logger.handlers[0].setFormatter(formatter)


def _safe_device_get(arr):
    """Safely get array to host, handling multi-host sharded arrays."""
    if arr is None:
        return None
    try:
        return jax.device_get(arr)
    except RuntimeError as e:
        if "non-addressable" in str(e):
            try:
                gathered = multihost_utils.process_allgather(arr, tiled=True)
                return jax.device_get(gathered)
            except Exception as gather_error:
                logging.warning(f"Failed to gather array: {gather_error}")
                return None
        raise


def _is_tpu_runtime() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


def extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numbers (including negative) from text."""
    # Match integers and floats, including negative numbers
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, text)
    return [float(m) for m in matches if m]


def extract_direction_tokens(text: str) -> list[str]:
    """Extract direction keywords from text."""
    directions = ["forward", "backward", "back", "left", "right", "up", "down", "open", "close"]
    found = []
    text_lower = text.lower()
    for direction in directions:
        # Count occurrences
        count = text_lower.count(direction)
        found.extend([direction] * count)
    return found


def parse_prompt_tokens(prompt_text: str) -> dict:
    """Parse a prompt to extract state values, action numbers, and direction tokens.

    Expected format: "Task: ..., State: 1 3 51 122 -1 235 89, Action: move forward 1 cm, ..."
    """
    result = {"state_values": [], "action_numbers": [], "direction_tokens": []}

    # Extract state section
    state_match = re.search(r"State:\s*([^,]+?)(?:,|$)", prompt_text, re.IGNORECASE)
    if state_match:
        state_text = state_match.group(1)
        result["state_values"] = extract_numbers_from_text(state_text)

    # Extract action section
    action_match = re.search(r"Action:\s*(.+?)$", prompt_text, re.IGNORECASE | re.DOTALL)
    if action_match:
        action_text = action_match.group(1)
        result["action_numbers"] = extract_numbers_from_text(action_text)
        result["direction_tokens"] = extract_direction_tokens(action_text)

    return result


def decode_prompt_strings(obs, tokenizer) -> list[str]:
    """Extract and decode the prompt tokens per example."""
    if not hasattr(obs, "tokenized_prompt") or obs.tokenized_prompt is None:
        return []

    tokens = _safe_device_get(obs.tokenized_prompt)
    if tokens is None:
        return []

    # For full prompt, we don't use the mask (or we use inverse of langact mask)
    # But for simplicity, let's just decode all tokens
    out: list[str] = []
    for i in range(tokens.shape[0]):
        try:
            # Decode full prompt - filter out padding (0 tokens)
            valid_tokens = tokens[i][tokens[i] != 0]
            text = tokenizer.decode(valid_tokens.astype(np.int32))
        except Exception:
            text = ""
        out.append(text)
    return out


def decode_dataset_names(obs, tokenizer) -> list[str]:
    """Decode dataset names from tokenized_dataset_name."""
    if not hasattr(obs, "tokenized_dataset_name") or obs.tokenized_dataset_name is None:
        return []

    names = _safe_device_get(obs.tokenized_dataset_name)
    if names is None:
        return []

    out: list[str] = []
    for i in range(names.shape[0]):
        try:
            name = tokenizer.decode(names[i].astype(np.int32))
            name = name.strip()
        except Exception:
            name = "unknown"
        out.append(name)
    return out


def create_histogram(values: list, title: str, xlabel: str, color: str, bins: int = 30):
    """Create a histogram plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor="black")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def create_histogram_from_bins(bin_edges, bin_counts, title: str, xlabel: str, color: str):
    """Create a histogram plot from precomputed bins."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_centers, bin_counts, width=width * 0.9, color=color, alpha=0.7, edgecolor="black")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


class RunningStats:
    """Track running statistics (mean, std, count) without storing all values."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean

    def update(self, value):
        """Update statistics with a new value using Welford's online algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean if self.count > 0 else 0.0

    def get_std(self):
        return np.sqrt(self.M2 / self.count) if self.count > 1 else 0.0

    def get_variance(self):
        return self.M2 / self.count if self.count > 1 else 0.0


def create_bar_plot(labels: list[str], values: list[float], title: str, ylabel: str, color: str):
    """Create a bar plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), values, color=color)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def main(config: _config.TrainConfig):
    """Analyze and visualize token distributions from the dataloader.

    Memory-efficient implementation:
    - Uses fixed-size histograms instead of storing all raw values
    - Uses Welford's online algorithm for running statistics (mean, std)
    - Only unbounded storage is Counter for discrete number token values
    - Memory usage is O(num_bins + num_datasets + num_unique_values), not O(num_samples)
    """
    num_batches = 50  # Number of batches to analyze

    wandb_enabled = bool(getattr(config, "wandb_enabled", False)) and wandb is not None
    if wandb_enabled:
        wandb_mode = "online" if os.environ.get("WANDB_DISABLED", "false").lower() not in {"1", "true"} else "offline"
        run_name = f"vis-token-dist-{config.name}"
        exp_name = getattr(config, "exp_name", None)
        if exp_name:
            run_name = f"{run_name}-{exp_name}"
        wandb.init(
            project=getattr(config, "project_name", "openpi-cot"),
            name=run_name,
            config={"config_name": config.name, "exp_name": exp_name},
            reinit=True,
            mode=wandb_mode,
        )

    # Initialize JAX distributed if needed
    if ("v6" in config.name and config.fsdp_devices > 8) or ("v4" in config.name and config.fsdp_devices > 4):
        jax.distributed.initialize()

    data_dir = save_dir = config.data.rlds_data_dir
    cache_dir = os.environ.get("OPENPI_DATA_HOME", None)
    if _is_tpu_runtime() and (str(data_dir).startswith("gs://") or str(save_dir).startswith("gs://")):
        prevent_cross_region(data_dir, save_dir)
        if cache_dir is not None:
            prevent_cross_region(cache_dir, save_dir)

    # Setup devices
    process_count = getattr(jax, "process_count", lambda: 1)()
    local_devices = getattr(jax, "local_device_count", lambda: 1)()
    global_devices = getattr(jax, "device_count", lambda: local_devices)()
    init_logging()
    logging.info(f"Local devices: {local_devices}, Global devices: {global_devices}, Process count: {process_count}")

    if process_count == 1:
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

    # Create mesh and data loader
    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
    )
    tok = PaligemmaCoTTokenizer(max_len=300)

    # Initialize tracking structures - using histograms and running stats for memory efficiency
    # Histograms for count distributions (bins from 0 to 50)
    num_token_hist_bins = np.arange(0, 4, 1)
    num_token_hist_counts = np.zeros(len(num_token_hist_bins) - 1, dtype=np.int64)

    # Histogram for state values (bins from -500 to 500)
    state_value_hist_bins = np.arange(-1, 257, 1)
    state_value_hist_counts = np.zeros(len(state_value_hist_bins) - 1, dtype=np.int64)

    # Running stats for summary statistics
    num_token_stats = RunningStats()
    state_value_stats = RunningStats()

    # Counter for individual number token values (memory efficient for discrete values)
    number_token_value_counter = Counter()

    # Per-dataset tracking using running stats
    dataset_num_token_stats = defaultdict(RunningStats)
    dataset_state_value_stats = defaultdict(RunningStats)
    dataset_number_token_value_counter = defaultdict(Counter)

    data_iter = iter(data_loader)
    logging.info("Starting token distribution analysis...")

    for batch_idx in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            logging.info(f"Reached end of data after {batch_idx} batches")
            break

        obs = batch[0]

        # Decode prompts and dataset names
        prompt_texts = decode_prompt_strings(obs, tok)
        dataset_names = decode_dataset_names(obs, tok)

        for i, prompt_text in enumerate(prompt_texts):
            dataset_name = dataset_names[i] if i < len(dataset_names) else "unknown"

            # Parse tokens
            parsed = parse_prompt_tokens(prompt_text)

            num_count = len(parsed["action_numbers"])
            state_values = parsed["state_values"]

            # Update global histograms
            if num_count < len(num_token_hist_bins):
                num_token_hist_counts[min(num_count, len(num_token_hist_counts) - 1)] += 1

            # Update running stats
            num_token_stats.update(num_count)

            # Update state value histogram and stats
            for state_val in state_values:
                # Find bin index for state value
                bin_idx = np.searchsorted(state_value_hist_bins, state_val) - 1
                if 0 <= bin_idx < len(state_value_hist_counts):
                    state_value_hist_counts[bin_idx] += 1
                state_value_stats.update(state_val)

            # Count individual number token values
            for num_value in parsed["action_numbers"]:
                # Round to nearest integer for cleaner distribution
                number_token_value_counter[int(round(num_value))] += 1

            # Per-dataset running stats
            dataset_num_token_stats[dataset_name].update(num_count)
            for state_val in state_values:
                dataset_state_value_stats[dataset_name].update(state_val)

            # Per-dataset number token value counter
            for num_value in parsed["action_numbers"]:
                dataset_number_token_value_counter[dataset_name][int(round(num_value))] += 1

        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Processed {batch_idx + 1}/{num_batches} batches")

    logging.info(f"Analysis complete. Processed {num_token_stats.count} robot samples")
    logging.info(f"Found {len(dataset_num_token_stats)} unique datasets")

    # Create visualizations
    plots = {}

    # 1. Global number tokens distribution
    if num_token_stats.count > 0:
        plots["num_tokens_hist"] = create_histogram_from_bins(
            num_token_hist_bins,
            num_token_hist_counts,
            "Number Tokens Distribution (Global)",
            "Number of Number Tokens per Sample",
            "steelblue",
        )
        logging.info(
            f"Number tokens - Mean: {num_token_stats.get_mean():.2f}, Std: {num_token_stats.get_std():.2f}, Count: {num_token_stats.count}"
        )

    # 3. Global state values distribution
    if state_value_stats.count > 0:
        plots["state_values_hist"] = create_histogram_from_bins(
            state_value_hist_bins,
            state_value_hist_counts,
            "State Values Distribution (Global)",
            "State Value",
            "mediumseagreen",
        )
        logging.info(
            f"State values - Mean: {state_value_stats.get_mean():.2f}, Std: {state_value_stats.get_std():.2f}, Count: {state_value_stats.count}"
        )

    # 4. Number token value frequency distribution
    if number_token_value_counter:
        # Get top 50 most common number values for better visualization
        most_common = number_token_value_counter.most_common(50)
        num_values = [val for val, count in most_common]
        num_counts = [count for val, count in most_common]

        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(range(len(num_values)), num_counts, color="mediumpurple")
        ax.set_xlabel("Number Token Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Number Token Value Frequency Distribution (Top 50)", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(num_values)))
        ax.set_xticklabels([str(v) for v in num_values], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        # Add count labels on bars
        for bar, count in zip(bars, num_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, str(count), ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plots["number_value_freq"] = fig

        logging.info(f"Found {len(number_token_value_counter)} unique number token values")
        logging.info(f"Top 10 most common number values: {number_token_value_counter.most_common(10)}")

    # 5. Per-dataset number token value frequency distribution
    if dataset_number_token_value_counter:
        datasets_sorted = sorted(
            dataset_number_token_value_counter.keys(),
            key=lambda x: sum(dataset_number_token_value_counter[x].values()),
            reverse=True,
        )

        # Create subplots - one per dataset (top 20 values per dataset)
        num_datasets = len(datasets_sorted)
        cols = min(3, num_datasets)  # Max 3 columns
        rows = (num_datasets + cols - 1) // cols  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if num_datasets == 1:
            axes = np.array([axes])
        axes = axes.flatten() if num_datasets > 1 else axes

        for idx, dataset_name in enumerate(datasets_sorted):
            ax = axes[idx]
            counter = dataset_number_token_value_counter[dataset_name]

            # Get top 20 most common number values for this dataset
            most_common = counter.most_common(20)
            if most_common:
                num_values = [val for val, count in most_common]
                num_counts = [count for val, count in most_common]

                bars = ax.bar(range(len(num_values)), num_counts, color="coral")
                ax.set_xlabel("Number Token Value", fontsize=10)
                ax.set_ylabel("Frequency", fontsize=10)
                ax.set_title(f"{dataset_name} (Top 20)", fontsize=11, fontweight="bold")
                ax.set_xticks(range(len(num_values)))
                ax.set_xticklabels([str(v) for v in num_values], rotation=45, ha="right", fontsize=8)
                ax.grid(axis="y", alpha=0.3)

                # Add count labels on bars for datasets with few values
                if len(num_values) <= 10:
                    for bar, count in zip(bars, num_counts):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height,
                            str(count),
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

        # Hide extra subplots
        for idx in range(num_datasets, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plots["dataset_number_value_freq"] = fig

        logging.info(f"Created per-dataset number token value frequency plots for {num_datasets} datasets")

    # 6. Per-dataset average number tokens
    if dataset_num_token_stats:
        dataset_names_sorted = sorted(
            dataset_num_token_stats.keys(), key=lambda x: dataset_num_token_stats[x].get_mean(), reverse=True
        )
        avg_num_counts = [dataset_num_token_stats[name].get_mean() for name in dataset_names_sorted]
        plots["dataset_num_tokens"] = create_bar_plot(
            dataset_names_sorted,
            avg_num_counts,
            "Average Number Tokens per Dataset",
            "Average Number Tokens",
            "steelblue",
        )

    # 7. Per-dataset state value statistics
    if dataset_state_value_stats:
        dataset_names_sorted = sorted(
            dataset_state_value_stats.keys(), key=lambda x: dataset_state_value_stats[x].count, reverse=True
        )
        avg_state_values = [dataset_state_value_stats[name].get_mean() for name in dataset_names_sorted]
        plots["dataset_state_means"] = create_bar_plot(
            dataset_names_sorted,
            avg_state_values,
            "Average State Value per Dataset",
            "Average State Value",
            "mediumseagreen",
        )

    # Log or save plots
    if wandb_enabled and wandb is not None:
        log_dict = {}
        for plot_name, plot_fig in plots.items():
            log_dict[f"token_dist/{plot_name}"] = wandb.Image(plot_fig)
        wandb.log(log_dict)
        logging.info("Logged plots to wandb")
        wandb.finish()
    else:
        # Save plots locally
        output_dir = epath.Path("./token_distribution_plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        for plot_name, plot_fig in plots.items():
            plot_fig.savefig(output_dir / f"{plot_name}.png", dpi=150, bbox_inches="tight")
        logging.info(f"Saved plots to {output_dir}")

    # Close all figures
    for plot_fig in plots.values():
        plt.close(plot_fig)

    logging.info("Token distribution analysis complete!")


if __name__ == "__main__":
    main(_config.cli())

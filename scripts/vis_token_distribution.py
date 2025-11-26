"""Visualize token distributions from the dataloader.

This script analyzes and visualizes:
1. Number tokens distribution
2. Number + direction tokens distribution
3. Per-dataset token distributions
4. State value distributions
"""

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
    directions = ["forward", "backward", "left", "right", "up", "down", "open", "close"]
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

    # Initialize tracking structures
    all_num_token_counts = []  # Count of number tokens per sample
    all_num_dir_token_counts = []  # Count of number + direction tokens per sample
    all_state_values = []  # All state values

    # Per-dataset tracking
    dataset_num_counts = defaultdict(list)
    dataset_num_dir_counts = defaultdict(list)
    dataset_state_values = defaultdict(list)

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

        breakpoint()

        for i, prompt_text in enumerate(prompt_texts):
            dataset_name = dataset_names[i] if i < len(dataset_names) else "unknown"

            # Parse tokens
            parsed = parse_prompt_tokens(prompt_text)

            breakpoint()

            num_count = len(parsed["action_numbers"])
            dir_count = len(parsed["direction_tokens"])
            num_dir_count = num_count + dir_count
            state_values = parsed["state_values"]

            # Global tracking
            all_num_token_counts.append(num_count)
            all_num_dir_token_counts.append(num_dir_count)
            all_state_values.extend(state_values)

            # Per-dataset tracking
            dataset_num_counts[dataset_name].append(num_count)
            dataset_num_dir_counts[dataset_name].append(num_dir_count)
            dataset_state_values[dataset_name].extend(state_values)

        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Processed {batch_idx + 1}/{num_batches} batches")

    logging.info(f"Analysis complete. Processed {len(all_num_token_counts)} robot samples")
    logging.info(f"Found {len(dataset_num_counts)} unique datasets")

    # Create visualizations
    plots = {}

    # 1. Global number tokens distribution
    if all_num_token_counts:
        plots["num_tokens_hist"] = create_histogram(
            all_num_token_counts,
            "Number Tokens Distribution (Global)",
            "Number of Number Tokens per Sample",
            "steelblue",
            bins=20,
        )
        logging.info(
            f"Number tokens - Mean: {np.mean(all_num_token_counts):.2f}, Std: {np.std(all_num_token_counts):.2f}"
        )

    # 2. Global number + direction tokens distribution
    if all_num_dir_token_counts:
        plots["num_dir_tokens_hist"] = create_histogram(
            all_num_dir_token_counts,
            "Number + Direction Tokens Distribution (Global)",
            "Number of (Number + Direction) Tokens per Sample",
            "coral",
            bins=20,
        )
        logging.info(
            f"Number + Direction tokens - Mean: {np.mean(all_num_dir_token_counts):.2f}, Std: {np.std(all_num_dir_token_counts):.2f}"
        )

    # 3. Global state values distribution
    if all_state_values:
        plots["state_values_hist"] = create_histogram(
            all_state_values, "State Values Distribution (Global)", "State Value", "mediumseagreen", bins=50
        )
        logging.info(f"State values - Mean: {np.mean(all_state_values):.2f}, Std: {np.std(all_state_values):.2f}")

    # 4. Per-dataset average number tokens
    if dataset_num_counts:
        dataset_names_sorted = sorted(
            dataset_num_counts.keys(), key=lambda x: np.mean(dataset_num_counts[x]), reverse=True
        )
        avg_num_counts = [np.mean(dataset_num_counts[name]) for name in dataset_names_sorted]
        plots["dataset_num_tokens"] = create_bar_plot(
            dataset_names_sorted,
            avg_num_counts,
            "Average Number Tokens per Dataset",
            "Average Number Tokens",
            "steelblue",
        )

    # 5. Per-dataset average number + direction tokens
    if dataset_num_dir_counts:
        dataset_names_sorted = sorted(
            dataset_num_dir_counts.keys(), key=lambda x: np.mean(dataset_num_dir_counts[x]), reverse=True
        )
        avg_num_dir_counts = [np.mean(dataset_num_dir_counts[name]) for name in dataset_names_sorted]
        plots["dataset_num_dir_tokens"] = create_bar_plot(
            dataset_names_sorted,
            avg_num_dir_counts,
            "Average (Number + Direction) Tokens per Dataset",
            "Average (Number + Direction) Tokens",
            "coral",
        )

    # 6. Per-dataset state value statistics
    if dataset_state_values:
        dataset_names_sorted = sorted(
            dataset_state_values.keys(), key=lambda x: len(dataset_state_values[x]), reverse=True
        )
        avg_state_values = [
            np.mean(dataset_state_values[name]) if dataset_state_values[name] else 0 for name in dataset_names_sorted
        ]
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

"""Shared logging utilities for training and validation metrics.

This module provides unified functionality for:
1. Dataset-level metrics tracking (per dataset, per category)
2. Global metrics aggregation (across all samples)
3. Multi-host metric gathering
4. Visualization and plotting
"""

import logging
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
import openpi.shared.array_typing as at

import openpi_cot.training.utils as training_utils


class DatasetStatsTracker:
    """Tracks per-dataset loss and token accuracy (micro-averaged) across training/validation.

    Supports tracking separate statistics for different sample types (e.g., 'pred' and 'langact').
    """

    def __init__(self):
        # Use micro-averaging: track total correct tokens and total tokens per (dataset, sample_type)
        # Key format: (dataset_name, sample_type) -> stats dict
        self.dataset_stats = {}  # {(dataset_name, sample_type): {stats}}
        # Track cumulative stats (never reset) - for since-start averages
        self.cumulative_stats = {}  # {(dataset_name, sample_type): {stats}}

    def update(
        self,
        dataset_names: list[str],
        losses: np.ndarray,
        critical_token_data: tuple[np.ndarray, np.ndarray] | None = None,
        number_token_data: tuple[np.ndarray, np.ndarray] | None = None,
        direction_token_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_type: str = "langact",
    ):
        """Update statistics with new batch data using micro-averaging.

        Args:
            dataset_names: List of dataset names for each sample
            losses: Array of per-sample losses
            critical_token_data: Tuple of (correct_counts, total_counts) for critical tokens (optional)
            number_token_data: Tuple of (correct_counts, total_counts) for number tokens (optional)
            direction_token_data: Tuple of (correct_counts, total_counts) for direction tokens (optional)
            sample_type: Type of samples (e.g., 'pred', 'langact')
        """
        for idx, name in enumerate(dataset_names):
            key = (name, sample_type)

            # Initialize if needed
            if key not in self.dataset_stats:
                self.dataset_stats[key] = self._create_empty_stats()
            if key not in self.cumulative_stats:
                self.cumulative_stats[key] = self._create_empty_stats()

            # Update loss and count
            self.dataset_stats[key]["total_loss"] += float(losses[idx])
            self.dataset_stats[key]["count"] += 1
            self.cumulative_stats[key]["total_loss"] += float(losses[idx])
            self.cumulative_stats[key]["count"] += 1

            # Micro-averaging: accumulate token-level counts
            if critical_token_data is not None:
                correct_counts, total_counts = critical_token_data
                self._update_token_stats(self.dataset_stats[key], "critical", correct_counts[idx], total_counts[idx])
                self._update_token_stats(self.cumulative_stats[key], "critical", correct_counts[idx], total_counts[idx])

            if number_token_data is not None:
                correct_counts, total_counts = number_token_data
                self._update_token_stats(self.dataset_stats[key], "number", correct_counts[idx], total_counts[idx])
                self._update_token_stats(self.cumulative_stats[key], "number", correct_counts[idx], total_counts[idx])

            if direction_token_data is not None:
                correct_counts, total_counts = direction_token_data
                self._update_token_stats(self.dataset_stats[key], "direction", correct_counts[idx], total_counts[idx])
                self._update_token_stats(
                    self.cumulative_stats[key], "direction", correct_counts[idx], total_counts[idx]
                )

    @staticmethod
    def _create_empty_stats() -> dict[str, Any]:
        """Create an empty statistics dictionary."""
        return {
            "total_loss": 0.0,
            "count": 0,
            "critical_correct": 0,
            "critical_total": 0,
            "number_correct": 0,
            "number_total": 0,
            "direction_correct": 0,
            "direction_total": 0,
        }

    @staticmethod
    def _update_token_stats(stats_dict: dict, token_type: str, correct: int | float, total: int | float):
        """Update token statistics in a stats dictionary."""
        stats_dict[f"{token_type}_correct"] += int(correct)
        stats_dict[f"{token_type}_total"] += int(total)

    def get_metrics(self, prefix: str = "") -> dict[str, float]:
        """Get current average losses and micro-averaged token accuracies for all datasets and sample types.

        Args:
            prefix: Optional prefix for all metric names (e.g., "val_" for validation)

        Returns:
            Dictionary of metrics with appropriate prefixes
        """
        metrics = {}

        # Current interval metrics
        for (dataset_name, sample_type), stats in self.dataset_stats.items():
            if stats["count"] > 0:
                type_prefix = f"{sample_type}_" if sample_type != "all" else ""
                base_key = f"{prefix}dataset/{dataset_name}/{type_prefix}"

                # Loss and count
                metrics[f"{base_key}avg_loss"] = stats["total_loss"] / stats["count"]
                metrics[f"{base_key}count"] = stats["count"]

                # Token accuracies
                self._add_token_accuracy_metrics(metrics, base_key, stats, "critical")
                self._add_token_accuracy_metrics(metrics, base_key, stats, "number")
                self._add_token_accuracy_metrics(metrics, base_key, stats, "direction")

        # Cumulative (since-start) metrics
        for (dataset_name, sample_type), stats in self.cumulative_stats.items():
            if stats["count"] > 0:
                type_prefix = f"{sample_type}_" if sample_type != "all" else ""
                base_key = f"{prefix}dataset/{dataset_name}/{type_prefix}"

                # Cumulative loss and count
                metrics[f"{base_key}cumulative_avg_loss"] = stats["total_loss"] / stats["count"]
                metrics[f"{base_key}cumulative_count"] = stats["count"]

                # Cumulative token accuracies
                self._add_token_accuracy_metrics(metrics, base_key, stats, "critical", cumulative=True)
                self._add_token_accuracy_metrics(metrics, base_key, stats, "number", cumulative=True)
                self._add_token_accuracy_metrics(metrics, base_key, stats, "direction", cumulative=True)

        return metrics

    @staticmethod
    def _add_token_accuracy_metrics(
        metrics: dict, base_key: str, stats: dict, token_type: str, cumulative: bool = False
    ):
        """Add token accuracy metrics to the metrics dictionary."""
        total_key = f"{token_type}_total"
        correct_key = f"{token_type}_correct"

        if stats[total_key] > 0:
            accuracy = stats[correct_key] / stats[total_key]
            prefix = "cumulative_" if cumulative else ""
            metrics[f"{base_key}{prefix}avg_{token_type}_token_acc"] = accuracy
            if not cumulative:
                metrics[f"{base_key}{token_type}_token_count"] = stats[total_key]

    def reset(self):
        """Reset accumulated statistics for the current log interval. Cumulative counts are preserved."""
        self.dataset_stats = {}


class LocalDatasetInfoBuffer:
    """Buffers local dataset info per step (without multihost gathering). Only gathers at log_interval.

    Supports buffering data for different sample types (e.g., 'pred', 'langact').
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Accumulate local tokenized names, losses, and token-level counts across steps
        # Each buffer is a list of (data, sample_type) tuples
        self.tokenized_names_buffer = []
        self.losses_buffer = []
        # For micro-averaging: buffer (correct_count, total_count) tuples
        self.critical_correct_buffer = []
        self.critical_total_buffer = []
        self.number_correct_buffer = []
        self.number_total_buffer = []
        self.direction_correct_buffer = []
        self.direction_total_buffer = []

    def add_local_batch(
        self,
        tokenized_dataset_names: at.Array,
        per_sample_losses: at.Array,
        per_sample_critical_token_data: tuple[at.Array, at.Array] | None = None,
        per_sample_number_token_data: tuple[at.Array, at.Array] | None = None,
        per_sample_direction_token_data: tuple[at.Array, at.Array] | None = None,
        sample_type: str = "langact",
    ):
        """Add local batch data (no multihost gathering here).

        Args:
            tokenized_dataset_names: Local tokenized dataset names
            per_sample_losses: Local per-sample losses
            per_sample_critical_token_data: Tuple of (correct_counts, total_counts) for critical tokens (optional)
            per_sample_number_token_data: Tuple of (correct_counts, total_counts) for number tokens (optional)
            per_sample_direction_token_data: Tuple of (correct_counts, total_counts) for direction tokens (optional)
            sample_type: Type of samples (e.g., 'pred', 'langact')
        """
        # Convert to numpy and store with sample_type
        self.tokenized_names_buffer.append(
            (np.asarray(training_utils.to_local_array(tokenized_dataset_names)), sample_type)
        )
        self.losses_buffer.append((np.asarray(training_utils.to_local_array(per_sample_losses)), sample_type))

        # Buffer token-level counts for micro-averaging
        if per_sample_critical_token_data is not None:
            correct_counts, total_counts = per_sample_critical_token_data
            self.critical_correct_buffer.append(
                (np.asarray(training_utils.to_local_array(correct_counts)), sample_type)
            )
            self.critical_total_buffer.append((np.asarray(training_utils.to_local_array(total_counts)), sample_type))

        if per_sample_number_token_data is not None:
            correct_counts, total_counts = per_sample_number_token_data
            self.number_correct_buffer.append((np.asarray(training_utils.to_local_array(correct_counts)), sample_type))
            self.number_total_buffer.append((np.asarray(training_utils.to_local_array(total_counts)), sample_type))

        if per_sample_direction_token_data is not None:
            correct_counts, total_counts = per_sample_direction_token_data
            self.direction_correct_buffer.append(
                (np.asarray(training_utils.to_local_array(correct_counts)), sample_type)
            )
            self.direction_total_buffer.append((np.asarray(training_utils.to_local_array(total_counts)), sample_type))

    def gather_and_update_stats(self, dataset_stats_tracker: DatasetStatsTracker) -> None:
        """Gather buffered data from all hosts and update dataset statistics.

        This should be called at log_interval to batch multihost communication.
        Handles data grouped by sample_type.
        """
        # Check if any process has data - all processes must agree to proceed or skip
        process_count = jax.process_count()
        has_local_data = len(self.tokenized_names_buffer) > 0

        if process_count > 1:
            # Gather has_data flags from all processes
            has_data_array = np.array([has_local_data], dtype=bool)
            all_has_data = jax.experimental.multihost_utils.process_allgather(has_data_array)
            any_process_has_data = np.any(all_has_data)

            # If no process has data, all can skip
            if not any_process_has_data:
                return

            # If some but not all processes have data, those without data need to participate
            # but with empty buffers - they'll be handled below
        else:
            if not has_local_data:
                return

        # Group buffered data by sample_type
        data_by_type = {}
        for (names, sample_type), (losses, _) in zip(self.tokenized_names_buffer, self.losses_buffer):
            if sample_type not in data_by_type:
                data_by_type[sample_type] = {
                    "names": [],
                    "losses": [],
                    "critical_correct": [],
                    "critical_total": [],
                    "number_correct": [],
                    "number_total": [],
                    "direction_correct": [],
                    "direction_total": [],
                }
            data_by_type[sample_type]["names"].append(names)
            data_by_type[sample_type]["losses"].append(losses)

        # Add token count data grouped by sample_type
        for (correct, sample_type), (total, _) in zip(self.critical_correct_buffer, self.critical_total_buffer):
            if sample_type in data_by_type:
                data_by_type[sample_type]["critical_correct"].append(correct)
                data_by_type[sample_type]["critical_total"].append(total)

        for (correct, sample_type), (total, _) in zip(self.number_correct_buffer, self.number_total_buffer):
            if sample_type in data_by_type:
                data_by_type[sample_type]["number_correct"].append(correct)
                data_by_type[sample_type]["number_total"].append(total)

        for (correct, sample_type), (total, _) in zip(self.direction_correct_buffer, self.direction_total_buffer):
            if sample_type in data_by_type:
                data_by_type[sample_type]["direction_correct"].append(correct)
                data_by_type[sample_type]["direction_total"].append(total)

        # Synchronize sample_types across all processes to ensure deterministic order
        # Known sample types are 'langact' and 'pred' - use fixed order for simplicity
        all_possible_types = ["langact", "pred"]

        if process_count > 1:
            # Use a simple boolean array to indicate which types each process has
            local_has_types = np.array([
                "langact" in data_by_type,
                "pred" in data_by_type
            ], dtype=np.int32)

            # Gather flags from all processes
            all_has_types = jax.experimental.multihost_utils.process_allgather(local_has_types)
            all_has_types = np.asarray(all_has_types)  # [num_processes, 2]

            # Determine which types any process has (OR across processes)
            any_has_types = np.any(all_has_types, axis=0)

            # Build list of types to process
            unique_types = [t for t, has in zip(all_possible_types, any_has_types) if has]
        else:
            unique_types = sorted(data_by_type.keys())

        # Process each sample_type separately
        for sample_type in unique_types:
            # Check if this process has data for this sample_type
            has_data = sample_type in data_by_type

            if has_data:
                data = data_by_type[sample_type]
                # Concatenate data for this sample_type
                all_local_names = np.concatenate(data["names"], axis=0)
                all_local_losses = np.concatenate(data["losses"], axis=0)
                local_seq_len = all_local_names.shape[-1] if all_local_names.size > 0 else 0
            else:
                # Will create empty arrays after broadcasting shape info
                all_local_names = None
                all_local_losses = None
                local_seq_len = 0
                data = {
                    "critical_correct": [],
                    "critical_total": [],
                    "number_correct": [],
                    "number_total": [],
                    "direction_correct": [],
                    "direction_total": [],
                }

            # Broadcast sequence length so all processes know the correct shape for empty arrays
            if process_count > 1:
                seq_len_array = np.array([local_seq_len], dtype=np.int32)
                all_seq_lens = jax.experimental.multihost_utils.process_allgather(seq_len_array)
                # Get the max non-zero seq_len (should be consistent across processes that have data)
                seq_len = int(np.max(all_seq_lens))

                # Create properly shaped empty arrays if this process has no data
                if not has_data:
                    all_local_names = np.empty((0, seq_len), dtype=np.int32)
                    all_local_losses = np.empty((0,), dtype=np.float32)

                # Multihost gather - all processes must participate
                all_names = jax.experimental.multihost_utils.process_allgather(all_local_names)
                all_losses = jax.experimental.multihost_utils.process_allgather(all_local_losses)

                # Flatten: [num_processes, batch_per_process, seq_len] -> [total_batch, seq_len]
                all_names = np.asarray(all_names).reshape(-1, seq_len)
                all_losses = np.asarray(all_losses).flatten()
            else:
                if not has_data:
                    continue
                all_names = all_local_names
                all_losses = all_local_losses

            # Skip update if there's no data after gathering (all processes had empty data)
            if all_names.shape[0] == 0:
                continue

            # Gather token counts (only processes with data will contribute)
            all_critical_data = self._gather_token_data(data, "critical", process_count)
            all_number_data = self._gather_token_data(data, "number", process_count)
            all_direction_data = self._gather_token_data(data, "direction", process_count)

            # Decode gathered tokenized names
            decoded_names = self._decode_names(all_names)

            # Update stats with token-level data for this sample_type
            # Only the process with rank 0 should update to avoid duplicate updates
            if process_count == 1 or jax.process_index() == 0:
                dataset_stats_tracker.update(
                    decoded_names,
                    all_losses,
                    all_critical_data,
                    all_number_data,
                    all_direction_data,
                    sample_type=sample_type,
                )

        # Clear buffers
        self.tokenized_names_buffer = []
        self.losses_buffer = []
        self.critical_correct_buffer = []
        self.critical_total_buffer = []
        self.number_correct_buffer = []
        self.number_total_buffer = []
        self.direction_correct_buffer = []
        self.direction_total_buffer = []

    def _gather_token_data(
        self, data: dict, token_type: str, process_count: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Gather token count data across hosts.

        All processes must participate in gather even if they have no data.
        """
        correct_key = f"{token_type}_correct"
        total_key = f"{token_type}_total"

        has_data = bool(data[correct_key] and data[total_key])

        if has_data:
            all_local_correct = np.concatenate(data[correct_key], axis=0)
            all_local_total = np.concatenate(data[total_key], axis=0)
        else:
            # Empty arrays for processes without this token type data
            all_local_correct = np.empty((0,), dtype=np.int32)
            all_local_total = np.empty((0,), dtype=np.int32)

        if process_count > 1:
            # All processes must participate in gather
            all_correct = jax.experimental.multihost_utils.process_allgather(all_local_correct)
            all_total = jax.experimental.multihost_utils.process_allgather(all_local_total)
            all_correct = np.asarray(all_correct).flatten()
            all_total = np.asarray(all_total).flatten()

            # Return None if no process had data
            if all_correct.size == 0:
                return None
        else:
            all_correct = all_local_correct
            all_total = all_local_total

            if not has_data:
                return None

        return (all_correct, all_total)

    def _decode_names(self, all_names: np.ndarray) -> list[str]:
        """Decode tokenized dataset names."""
        decoded_names = []
        for i in range(all_names.shape[0]):
            try:
                name = self.tokenizer.decode(all_names[i])
                name = name.strip()
                decoded_names.append(name)
            except Exception as e:
                logging.warning(f"Failed to decode dataset name for sample {i}: {e}")
                decoded_names.append("unknown")
        return decoded_names


def create_bar_plot(
    dataset_names: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    color: str,
    value_format: str = "{:.4f}",
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Helper function to create a bar plot with consistent styling.

    Args:
        dataset_names: List of dataset names for x-axis
        values: List of values for y-axis
        ylabel: Label for y-axis
        title: Plot title
        color: Bar color
        value_format: Format string for value labels
        ylim: Optional y-axis limits

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(dataset_names)), values, color=color)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(dataset_names)))
    ax.set_xticklabels(dataset_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        label = value_format.format(value) if isinstance(value, float) else str(int(value))
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def create_dataset_stats_plots(
    dataset_stats: dict[tuple[str, str], dict[str, float]],
    cumulative_stats: dict[tuple[str, str], dict[str, float]] | None = None,
) -> dict[str, plt.Figure]:
    """Create bar plots for dataset statistics.

    Args:
        dataset_stats: Dictionary mapping (dataset_name, sample_type) tuples to stats dicts
        cumulative_stats: Optional dictionary with same structure for cumulative metrics

    Returns:
        Dictionary with plots for interval and cumulative metrics
    """
    if not dataset_stats:
        return {}

    # Extract dataset names and statistics (group by dataset name, across all sample types)
    dataset_names = sorted(set(name for name, _ in dataset_stats))

    # For now, create plots for "all" sample type if it exists
    # TODO: Consider creating separate plots per sample_type
    plots = {}

    # Filter stats for "all" sample type
    all_stats = {name: stats for (name, stype), stats in dataset_stats.items() if stype == "all"}

    if not all_stats:
        return {}

    counts = [all_stats[name]["count"] for name in dataset_names if name in all_stats]
    avg_losses = [
        all_stats[name]["total_loss"] / all_stats[name]["count"] if all_stats[name]["count"] > 0 else 0.0
        for name in dataset_names
        if name in all_stats
    ]

    # Compute token accuracies
    avg_critical_accs = []
    avg_number_accs = []
    avg_direction_accs = []

    for name in dataset_names:
        if name not in all_stats:
            continue
        stats = all_stats[name]

        critical_acc = stats["critical_correct"] / stats["critical_total"] if stats["critical_total"] > 0 else 0.0
        avg_critical_accs.append(critical_acc)

        number_acc = stats["number_correct"] / stats["number_total"] if stats["number_total"] > 0 else 0.0
        avg_number_accs.append(number_acc)

        direction_acc = stats["direction_correct"] / stats["direction_total"] if stats["direction_total"] > 0 else 0.0
        avg_direction_accs.append(direction_acc)

    # Filter dataset_names to only include those in all_stats
    dataset_names_filtered = [name for name in dataset_names if name in all_stats]

    # Sort by count (descending) for better visualization
    sorted_indices = np.argsort(counts)[::-1]
    dataset_names_sorted = [dataset_names_filtered[i] for i in sorted_indices]
    counts_sorted = [counts[i] for i in sorted_indices]
    avg_losses_sorted = [avg_losses[i] for i in sorted_indices]
    avg_critical_accs_sorted = [avg_critical_accs[i] for i in sorted_indices]
    avg_number_accs_sorted = [avg_number_accs[i] for i in sorted_indices]
    avg_direction_accs_sorted = [avg_direction_accs[i] for i in sorted_indices]

    # Create interval plots
    plots["counts"] = create_bar_plot(
        dataset_names_sorted,
        counts_sorted,
        "Number of Examples",
        "Dataset Example Counts",
        "steelblue",
        value_format="{:.0f}",
    )

    plots["avg_loss"] = create_bar_plot(
        dataset_names_sorted,
        avg_losses_sorted,
        "Average Loss",
        "Dataset Average Loss",
        "coral",
    )

    # Create accuracy plots if data is available
    if any(acc > 0 for acc in avg_critical_accs_sorted):
        plots["avg_critical_token_acc"] = create_bar_plot(
            dataset_names_sorted,
            avg_critical_accs_sorted,
            "Average Critical Token Accuracy",
            "Dataset Average Critical Token Accuracy",
            "mediumseagreen",
            ylim=(0, 1),
        )

    if any(acc > 0 for acc in avg_number_accs_sorted):
        plots["avg_number_token_acc"] = create_bar_plot(
            dataset_names_sorted,
            avg_number_accs_sorted,
            "Average Number Token Accuracy",
            "Dataset Average Number Token Accuracy",
            "skyblue",
            ylim=(0, 1),
        )

    if any(acc > 0 for acc in avg_direction_accs_sorted):
        plots["avg_direction_token_acc"] = create_bar_plot(
            dataset_names_sorted,
            avg_direction_accs_sorted,
            "Average Direction Token Accuracy",
            "Dataset Average Direction Token Accuracy",
            "lightcoral",
            ylim=(0, 1),
        )

    # Create cumulative plots if requested
    if cumulative_stats:
        cumulative_all_stats = {name: stats for (name, stype), stats in cumulative_stats.items() if stype == "all"}

        if cumulative_all_stats:
            cumulative_counts = [cumulative_all_stats.get(name, {}).get("count", 0) for name in dataset_names_sorted]
            cumulative_avg_losses = [
                cumulative_all_stats[name]["total_loss"] / cumulative_all_stats[name]["count"]
                if name in cumulative_all_stats and cumulative_all_stats[name]["count"] > 0
                else 0.0
                for name in dataset_names_sorted
            ]

            cumulative_avg_critical_accs = [
                cumulative_all_stats[name]["critical_correct"] / cumulative_all_stats[name]["critical_total"]
                if name in cumulative_all_stats and cumulative_all_stats[name]["critical_total"] > 0
                else 0.0
                for name in dataset_names_sorted
            ]

            cumulative_avg_number_accs = [
                cumulative_all_stats[name]["number_correct"] / cumulative_all_stats[name]["number_total"]
                if name in cumulative_all_stats and cumulative_all_stats[name]["number_total"] > 0
                else 0.0
                for name in dataset_names_sorted
            ]

            cumulative_avg_direction_accs = [
                cumulative_all_stats[name]["direction_correct"] / cumulative_all_stats[name]["direction_total"]
                if name in cumulative_all_stats and cumulative_all_stats[name]["direction_total"] > 0
                else 0.0
                for name in dataset_names_sorted
            ]

            # Create cumulative plots
            plots["cumulative_counts"] = create_bar_plot(
                dataset_names_sorted,
                cumulative_counts,
                "Cumulative Number of Examples (Since Start)",
                "Dataset Cumulative Example Counts (Since Start)",
                "mediumseagreen",
                value_format="{:.0f}",
            )

            plots["cumulative_avg_loss"] = create_bar_plot(
                dataset_names_sorted,
                cumulative_avg_losses,
                "Cumulative Average Loss (Since Start)",
                "Dataset Cumulative Average Loss (Since Start)",
                "coral",
            )

            if any(acc > 0 for acc in cumulative_avg_critical_accs):
                plots["cumulative_avg_critical_token_acc"] = create_bar_plot(
                    dataset_names_sorted,
                    cumulative_avg_critical_accs,
                    "Cumulative Avg Critical Token Accuracy (Since Start)",
                    "Dataset Cumulative Average Critical Token Accuracy (Since Start)",
                    "mediumseagreen",
                    ylim=(0, 1),
                )

            if any(acc > 0 for acc in cumulative_avg_number_accs):
                plots["cumulative_avg_number_token_acc"] = create_bar_plot(
                    dataset_names_sorted,
                    cumulative_avg_number_accs,
                    "Cumulative Avg Number Token Accuracy (Since Start)",
                    "Dataset Cumulative Average Number Token Accuracy (Since Start)",
                    "skyblue",
                    ylim=(0, 1),
                )

            if any(acc > 0 for acc in cumulative_avg_direction_accs):
                plots["cumulative_avg_direction_token_acc"] = create_bar_plot(
                    dataset_names_sorted,
                    cumulative_avg_direction_accs,
                    "Cumulative Avg Direction Token Accuracy (Since Start)",
                    "Dataset Cumulative Average Direction Token Accuracy (Since Start)",
                    "lightcoral",
                    ylim=(0, 1),
                )

    return plots


def buffer_dataset_metrics_from_batch(
    buffer: LocalDatasetInfoBuffer,
    batch: tuple,
    info: dict[str, at.Array],
):
    """Buffer dataset-level metrics from a training/validation batch.

    This function extracts per-sample metrics from the info dict and buffers them
    for later gathering across hosts. It handles both pred and langact sample types.

    Args:
        buffer: LocalDatasetInfoBuffer to add data to
        batch: Tuple of (observation, actions)
        info: Dictionary containing per-sample metrics from compute_loss
    """
    obs = batch[0]

    # Check if batch has dataset names
    if not hasattr(obs, "tokenized_dataset_name"):
        return

    try:
        # Buffer prediction samples separately
        if "pred_per_sample_loss" in info:
            pred_critical_data = None
            if "pred_per_sample_critical_correct" in info and "pred_per_sample_critical_total" in info:
                pred_critical_data = (info["pred_per_sample_critical_correct"], info["pred_per_sample_critical_total"])

            pred_number_data = None
            if "pred_per_sample_number_correct" in info and "pred_per_sample_number_total" in info:
                pred_number_data = (info["pred_per_sample_number_correct"], info["pred_per_sample_number_total"])

            pred_direction_data = None
            if "pred_per_sample_direction_correct" in info and "pred_per_sample_direction_total" in info:
                pred_direction_data = (
                    info["pred_per_sample_direction_correct"],
                    info["pred_per_sample_direction_total"],
                )

            buffer.add_local_batch(
                obs.tokenized_dataset_name,
                info["pred_per_sample_loss"],
                pred_critical_data,
                pred_number_data,
                pred_direction_data,
                sample_type="pred",
            )

        # Buffer langact samples separately
        if "langact_per_sample_loss" in info:
            langact_critical_data = None
            if "langact_per_sample_critical_correct" in info and "langact_per_sample_critical_total" in info:
                langact_critical_data = (
                    info["langact_per_sample_critical_correct"],
                    info["langact_per_sample_critical_total"],
                )

            langact_number_data = None
            if "langact_per_sample_number_correct" in info and "langact_per_sample_number_total" in info:
                langact_number_data = (
                    info["langact_per_sample_number_correct"],
                    info["langact_per_sample_number_total"],
                )

            langact_direction_data = None
            if "langact_per_sample_direction_correct" in info and "langact_per_sample_direction_total" in info:
                langact_direction_data = (
                    info["langact_per_sample_direction_correct"],
                    info["langact_per_sample_direction_total"],
                )

            buffer.add_local_batch(
                obs.tokenized_dataset_name,
                info["langact_per_sample_loss"],
                langact_critical_data,
                langact_number_data,
                langact_direction_data,
                sample_type="langact",
            )

    except Exception as e:
        logging.warning(f"Failed to buffer dataset info: {e}")


def log_dataset_plots(plots: dict[str, plt.Figure], step: int, prefix: str = ""):
    """Log dataset statistics plots to wandb.

    Args:
        plots: Dictionary of plot figures
        step: Training step
        prefix: Optional prefix for wandb keys (e.g., "val_")
    """
    import wandb

    if not plots or jax.process_index() != 0:
        return

    log_dict = {}

    # Interval plots
    for plot_name in [
        "counts",
        "avg_loss",
        "avg_critical_token_acc",
        "avg_number_token_acc",
        "avg_direction_token_acc",
    ]:
        if plot_name in plots:
            log_dict[f"{prefix}dataset_stats/interval_{plot_name}_plot"] = wandb.Image(plots[plot_name])

    # Cumulative plots
    for plot_name in [
        "cumulative_counts",
        "cumulative_avg_loss",
        "cumulative_avg_critical_token_acc",
        "cumulative_avg_number_token_acc",
        "cumulative_avg_direction_token_acc",
    ]:
        if plot_name in plots:
            log_dict[f"{prefix}dataset_stats/{plot_name}_plot"] = wandb.Image(plots[plot_name])

    if log_dict:
        wandb.log(log_dict, step=step)

    # Close all figures to prevent memory leaks
    for plot_fig in plots.values():
        plt.close(plot_fig)

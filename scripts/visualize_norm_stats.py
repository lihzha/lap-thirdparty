#!/usr/bin/env python3
"""
Visualize norm statistics across all datasets.
Reads all norm_stats.json files and creates comparison plots.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_all_norm_stats(norm_stats_dir):
    """Load all norm_stats.json files from the directory."""
    norm_stats_dir = Path(norm_stats_dir)
    datasets = {}

    for json_file in sorted(norm_stats_dir.glob("*.json")):
        # Parse filename: OXE_{dataset_name}_{version}_norm_stats.json
        filename = json_file.stem  # Remove .json
        filename = filename.removesuffix("_norm_stats")  # Remove _norm_stats suffix
        filename = filename.removeprefix("OXE_")  # Remove OXE_ prefix

        with open(json_file) as f:
            data = json.load(f)
            datasets[filename] = data["norm_stats"]
            print(f"Loaded: {filename}")

    return datasets


def plot_dataset_sizes(datasets, output_dir):
    """Plot number of transitions and trajectories for each dataset."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    dataset_names = list(datasets.keys())
    state_transitions = [datasets[d]["state"]["num_transitions"] for d in dataset_names]
    state_trajectories = [datasets[d]["state"]["num_trajectories"] for d in dataset_names]

    # Plot transitions
    bars1 = ax1.barh(range(len(dataset_names)), state_transitions, color="steelblue")
    ax1.set_yticks(range(len(dataset_names)))
    ax1.set_yticklabels(dataset_names, fontsize=8)
    ax1.set_xlabel("Number of Transitions")
    ax1.set_title("Dataset Sizes - Number of Transitions")
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, state_transitions)):
        ax1.text(val, i, f" {val:,}", va="center", fontsize=7)

    # Plot trajectories
    bars2 = ax2.barh(range(len(dataset_names)), state_trajectories, color="coral")
    ax2.set_yticks(range(len(dataset_names)))
    ax2.set_yticklabels(dataset_names, fontsize=8)
    ax2.set_xlabel("Number of Trajectories")
    ax2.set_title("Dataset Sizes - Number of Trajectories")
    ax2.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, state_trajectories)):
        ax2.text(val, i, f" {val:,}", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_sizes.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'dataset_sizes.png'}")
    plt.close()


def plot_dimension_statistics(datasets, output_dir, stat_type="state"):
    """Plot statistics (mean, std, q01, q99) for each dimension across datasets."""
    dataset_names = list(datasets.keys())

    # Get max number of dimensions
    max_dims = max(len(datasets[d][stat_type]["mean"]) for d in dataset_names)

    # Prepare data
    means = np.zeros((len(dataset_names), max_dims))
    stds = np.zeros((len(dataset_names), max_dims))
    q01s = np.zeros((len(dataset_names), max_dims))
    q99s = np.zeros((len(dataset_names), max_dims))

    for i, dataset in enumerate(dataset_names):
        stats = datasets[dataset][stat_type]
        n_dims = len(stats["mean"])
        means[i, :n_dims] = stats["mean"]
        stds[i, :n_dims] = stats["std"]
        q01s[i, :n_dims] = stats["q01"]
        q99s[i, :n_dims] = stats["q99"]

    # Create subplots for each dimension
    n_cols = 4
    n_rows = (max_dims + n_cols - 1) // n_cols

    # Plot means
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if max_dims > 1 else [axes]

    for dim in range(max_dims):
        ax = axes[dim]
        values = means[:, dim]
        bars = ax.barh(range(len(dataset_names)), values)
        ax.set_yticks(range(len(dataset_names)))
        ax.set_yticklabels(dataset_names, fontsize=6)
        ax.set_xlabel("Mean", fontsize=8)
        ax.set_title(f"{stat_type.capitalize()} Dim {dim} - Mean", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

        # Color code by value
        vmin, vmax = values.min(), values.max()
        for bar, val in zip(bars, values):
            if vmax > vmin:
                color_intensity = (val - vmin) / (vmax - vmin)
                bar.set_color(plt.cm.RdYlBu_r(color_intensity))

    # Hide unused subplots
    for idx in range(max_dims, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"{stat_type}_means.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / f'{stat_type}_means.png'}")
    plt.close()

    # Plot stds
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if max_dims > 1 else [axes]

    for dim in range(max_dims):
        ax = axes[dim]
        values = stds[:, dim]
        bars = ax.barh(range(len(dataset_names)), values)
        ax.set_yticks(range(len(dataset_names)))
        ax.set_yticklabels(dataset_names, fontsize=6)
        ax.set_xlabel("Std Dev", fontsize=8)
        ax.set_title(f"{stat_type.capitalize()} Dim {dim} - Std", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

        # Color code by value
        vmin, vmax = values.min(), values.max()
        for bar, val in zip(bars, values):
            if vmax > vmin:
                color_intensity = (val - vmin) / (vmax - vmin)
                bar.set_color(plt.cm.YlOrRd(color_intensity))

    # Hide unused subplots
    for idx in range(max_dims, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"{stat_type}_stds.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / f'{stat_type}_stds.png'}")
    plt.close()

    # Plot ranges (q99 - q01)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if max_dims > 1 else [axes]

    for dim in range(max_dims):
        ax = axes[dim]
        q01_vals = q01s[:, dim]
        q99_vals = q99s[:, dim]
        ranges = q99_vals - q01_vals

        # Create horizontal bar chart showing range
        for i, dataset in enumerate(dataset_names):
            ax.barh(i, ranges[i], left=q01_vals[i], alpha=0.7)

        ax.set_yticks(range(len(dataset_names)))
        ax.set_yticklabels(dataset_names, fontsize=6)
        ax.set_xlabel("Value Range", fontsize=8)
        ax.set_title(f"{stat_type.capitalize()} Dim {dim} - Range (q01-q99)", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    # Hide unused subplots
    for idx in range(max_dims, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"{stat_type}_ranges.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / f'{stat_type}_ranges.png'}")
    plt.close()


def plot_dimension_count_comparison(datasets, output_dir):
    """Compare number of dimensions across datasets for state and actions."""
    dataset_names = list(datasets.keys())
    state_dims = [len(datasets[d]["state"]["mean"]) for d in dataset_names]
    action_dims = [len(datasets[d]["actions"]["mean"]) for d in dataset_names]

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(dataset_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, state_dims, width, label="State Dims", color="steelblue")
    bars2 = ax.bar(x + width / 2, action_dims, width, label="Action Dims", color="coral")

    ax.set_ylabel("Number of Dimensions")
    ax.set_title("State and Action Dimensions by Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "dimension_counts.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'dimension_counts.png'}")
    plt.close()


def generate_summary_table(datasets, output_dir):
    """Generate a summary table with key statistics."""
    with open(output_dir / "summary.txt", "w") as f:
        f.write("=" * 100 + "\n")
        f.write("DATASET STATISTICS SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        for dataset_name in sorted(datasets.keys()):
            stats = datasets[dataset_name]
            f.write(f"\n{dataset_name}\n")
            f.write("-" * 100 + "\n")
            f.write(f"  State Dimensions:      {len(stats['state']['mean'])}\n")
            f.write(f"  Action Dimensions:     {len(stats['actions']['mean'])}\n")
            f.write(f"  Num Transitions:       {stats['state']['num_transitions']:,}\n")
            f.write(f"  Num Trajectories:      {stats['state']['num_trajectories']:,}\n")

            if stats["state"]["num_trajectories"] > 0:
                avg_traj_len = stats["state"]["num_transitions"] / stats["state"]["num_trajectories"]
                f.write(f"  Avg Trajectory Length: {avg_traj_len:.1f}\n")

            f.write("\n")

    print(f"Saved: {output_dir / 'summary.txt'}")


def main():
    # Set up directories
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    norm_stats_dir = project_dir / "norm_stats"
    output_dir = project_dir / "norm_stats_visualizations"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading norm stats from: {norm_stats_dir}")

    if not norm_stats_dir.exists():
        print(f"Error: {norm_stats_dir} does not exist!")
        print("Please run download_norm_stats.py first.")
        return 1

    # Load all datasets
    datasets = load_all_norm_stats(norm_stats_dir)

    if not datasets:
        print("No datasets found!")
        return 1

    print(f"\nLoaded {len(datasets)} datasets")
    print(f"Saving visualizations to: {output_dir}\n")

    # Generate visualizations
    print("Generating visualizations...")
    plot_dataset_sizes(datasets, output_dir)
    plot_dimension_count_comparison(datasets, output_dir)
    plot_dimension_statistics(datasets, output_dir, stat_type="state")
    plot_dimension_statistics(datasets, output_dir, stat_type="actions")
    generate_summary_table(datasets, output_dir)

    print(f"\n✓ All visualizations saved to {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""Plot average success rate vs epoch for each policy from evaluation_results.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    # Load the evaluation results
    results_path = Path(__file__).parent.parent / "evaluation_results.json"
    with open(results_path) as f:
        data = json.load(f)

    # Set up the plot with a clean style
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define colors and markers for different policies
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "p", "*", "h"]

    # Plot each policy
    for idx, (policy_name, epochs_data) in enumerate(data.items()):
        # Extract epoch numbers and average success rates
        epochs = sorted([int(e) for e in epochs_data.keys()])
        avg_success_rates = [epochs_data[str(e)]["average"] for e in epochs]

        # Plot the curve
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # Create a cleaner display name
        display_name = policy_name.replace("_", " ").replace("libero finetune", "").strip()
        if display_name == "ours":
            display_name = "Ours (KI)"
        elif display_name == "rawaction":
            display_name = "Raw Action"
        elif "noki" in display_name:
            display_name = display_name.replace("noki", "(no KI)")
        print(display_name)
        if display_name == "ours  (no KI) langweight04" or display_name == "fast  (no KI) langweight01":
            plt.plot(
                epochs,
                avg_success_rates,
                marker=marker,
                markersize=8,
                linewidth=2,
                color=color,
                label=display_name,
            )

    # Customize the plot
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average Success Rate", fontsize=12)
    plt.title("LIBERO Evaluation: Average Success Rate vs Training Epoch", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)

    # Set axis limits
    plt.ylim(0.0, 1.0)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save the figure
    output_path = Path(__file__).parent.parent / "evaluation_results_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Also save as PDF for higher quality
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to: {pdf_path}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()

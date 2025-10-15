"""Diagnostic script to understand why some dimensions have std=0 in your dataset.

Usage:
    python diagnose_dataset_std_zero.py --norm_stats_path /path/to/norm_stats.json

This will help you understand if std=0 is due to:
1. Truly constant data (expected)
2. Float32 precision loss (needs investigation)
3. Data preprocessing issues
"""

import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_stats_path", required=True, help="Path to norm_stats.json")
    args = parser.parse_args()

    # Load the norm stats
    with open(args.norm_stats_path, "r") as f:
        data = json.load(f)

    print("=" * 80)
    print("NORM STATS DIAGNOSIS")
    print("=" * 80)
    print()

    # Check actions
    action_stats = data["norm_stats"]["actions"]
    action_mean = np.array(action_stats["mean"])
    action_std = np.array(action_stats["std"])
    action_q01 = np.array(action_stats["q01"])
    action_q99 = np.array(action_stats["q99"])

    print(f"Actions: {len(action_mean)} dimensions")
    print(f"  Samples: {action_stats.get('num_transitions', 'N/A')} transitions")
    print(f"  Trajectories: {action_stats.get('num_trajectories', 'N/A')}")
    print()

    zero_action_dims = np.where(action_std == 0)[0]
    if len(zero_action_dims) > 0:
        print(f"⚠️  {len(zero_action_dims)}/{len(action_mean)} action dimensions have std=0:")
        for dim in zero_action_dims:
            print(f"    Dim {dim}: mean={action_mean[dim]:.6f}, std=0.0, range=[{action_q01[dim]:.6f}, {action_q99[dim]:.6f}]")
            if action_q01[dim] == action_q99[dim]:
                print(f"             → TRULY CONSTANT (q01 == q99)")
            else:
                print(f"             → SUSPICIOUS! Quantiles differ but std=0")
        print()
    else:
        print("✅ All action dimensions have std > 0")
        print()

    # Check state
    state_stats = data["norm_stats"]["state"]
    state_mean = np.array(state_stats["mean"])
    state_std = np.array(state_stats["std"])
    state_q01 = np.array(state_stats["q01"])
    state_q99 = np.array(state_stats["q99"])

    print(f"State: {len(state_mean)} dimensions")
    print(f"  Samples: {state_stats.get('num_transitions', 'N/A')} transitions")
    print()

    if len(state_mean) == 0:
        print("  (Empty state - state_encoding == NONE)")
        print()
    else:
        zero_state_dims = np.where(state_std == 0)[0]
        if len(zero_state_dims) > 0:
            print(f"⚠️  {len(zero_state_dims)}/{len(state_mean)} state dimensions have std=0:")
            for dim in zero_state_dims:
                print(f"    Dim {dim}: mean={state_mean[dim]:.6f}, std=0.0, range=[{state_q01[dim]:.6f}, {state_q99[dim]:.6f}]")
                if state_q01[dim] == state_q99[dim]:
                    print(f"             → TRULY CONSTANT (q01 == q99)")
                else:
                    print(f"             → SUSPICIOUS! Quantiles differ but std=0")
            print()
        else:
            print("✅ All state dimensions have std > 0")
            print()

    # Summary
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("If q01 == q99 for std=0 dims → Data is TRULY constant (valid!)")
    print("  Examples: gripper always closed, unused joint, camera zoom fixed")
    print()
    print("If q01 != q99 for std=0 dims → Float32 precision issue!")
    print("  The quantile estimation detected variation, but variance calc collapsed")
    print("  This suggests std/mean ratio is below float32 precision (~1e-7)")
    print()
    print("NEXT STEP: Sample raw data from these dimensions to verify")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract policy performance from evaluation logs.

Parses openpi-<job_id>.log for policy name and epoch,
and libero-<job_id>.err for success rates across task suites.

Output: JSON file with results organized by policy name and epoch.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def extract_policy_info(log_path: Path) -> tuple:
    """
    Extract policy name and epoch from openpi-<job_id>.log.

    Looks for line: "Epoch dir: .../policy_name/epoch"

    Returns:
        Tuple of (policy_name, epoch) or None if not found.
    """
    try:
        content = log_path.read_text()
    except Exception as e:
        print(f"Warning: Could not read {log_path}: {e}")
        return None

    # Pattern: Epoch dir: <path>/policy_name/epoch
    match = re.search(r"Epoch dir:\s*(.+)", content)
    if not match:
        return None

    epoch_dir = match.group(1).strip()
    parts = epoch_dir.rstrip("/").split("/")

    if len(parts) < 2:
        return None

    epoch = parts[-1]
    policy_name = parts[-2]

    return policy_name, epoch


def extract_success_rates(err_path: Path) -> list:
    """
    Extract success rates from libero-<job_id>.err.

    Looks for lines: "INFO:root:Total success rate: <float>"

    Returns:
        List of 4 success rates or None if not exactly 4 found.
    """
    try:
        content = err_path.read_text()
    except Exception as e:
        print(f"Warning: Could not read {err_path}: {e}")
        return None

    # Pattern: INFO:root:Total success rate: <float>
    matches = re.findall(r"INFO:root:Total success rate: ([\d.]+)", content)

    if len(matches) != 4:
        return None

    return [float(rate) for rate in matches]


def extract_all_results(logs_dir: Path) -> dict:
    """
    Extract results from all log pairs in the logs directory.

    Returns:
        Dictionary with structure:
        {
            "policy_name": {
                "epoch": {
                    "libero_spatial": float,
                    "libero_object": float,
                    "libero_goal": float,
                    "libero_10": float,
                    "average": float
                }
            }
        }
    """
    results = defaultdict(dict)

    # Find all openpi-<job_id>.log files
    log_files = list(logs_dir.glob("openpi-*.log"))

    for log_file in log_files:
        # Extract job_id from filename
        match = re.match(r"openpi-(\d+)\.log", log_file.name)
        if not match:
            continue

        job_id = match.group(1)

        # Find corresponding err file
        err_file = logs_dir / f"libero-{job_id}.err"
        if not err_file.exists():
            print(f"Skipping {job_id}: libero-{job_id}.err not found")
            continue

        # Extract policy info
        policy_info = extract_policy_info(log_file)
        if policy_info is None:
            print(f"Skipping {job_id}: Could not extract policy info from {log_file.name}")
            continue

        policy_name, epoch = policy_info

        # Extract success rates
        success_rates = extract_success_rates(err_file)
        if success_rates is None:
            print(f"Skipping {job_id}: Could not extract 4 success rates from {err_file.name}")
            continue

        # Build result entry
        result_entry = {"job_id": job_id}
        for suite, rate in zip(TASK_SUITES, success_rates):
            result_entry[suite] = rate
        result_entry["average"] = sum(success_rates) / len(success_rates)

        # Check for duplicate entries
        if epoch in results[policy_name]:
            print(f"Warning: Duplicate entry for {policy_name}/{epoch} (job {job_id}), overwriting")

        results[policy_name][epoch] = result_entry
        print(f"Extracted: {policy_name}/{epoch} (job {job_id})")

    return dict(results)


def main():
    parser = argparse.ArgumentParser(description="Extract policy performance from evaluation logs.")
    parser.add_argument(
        "--logs-dir", type=Path, default=Path("logs"), help="Directory containing log files (default: logs/)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Output JSON file (default: evaluation_results.json)",
    )

    args = parser.parse_args()

    if not args.logs_dir.exists():
        print(f"Error: Logs directory '{args.logs_dir}' not found")
        return 1

    print(f"Scanning {args.logs_dir} for evaluation logs...")
    results = extract_all_results(args.logs_dir)

    if not results:
        print("No valid results found.")
        return 1

    # Sort results by policy name and epoch
    sorted_results = {}
    for policy_name in sorted(results.keys()):
        sorted_results[policy_name] = {}
        for epoch in sorted(results[policy_name].keys(), key=lambda x: int(x) if x.isdigit() else x):
            sorted_results[policy_name][epoch] = results[policy_name][epoch]

    # Write to JSON
    with open(args.output, "w") as f:
        json.dump(sorted_results, f, indent=2)

    print(f"\nResults written to {args.output}")
    print(f"Found {len(sorted_results)} policies:")
    for policy_name, epochs in sorted_results.items():
        print(f"  {policy_name}: {len(epochs)} checkpoint(s)")

    return 0


if __name__ == "__main__":
    exit(main())

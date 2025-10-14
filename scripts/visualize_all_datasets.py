#!/usr/bin/env python3
"""
Script to visualize all datasets in oxe_pi_magic_soup_with_other_states_with_bimanual mixture.
Usage: python scripts/visualize_all_datasets.py [--start-from DATASET_NAME] [--only DATASET_NAME]
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# All datasets from oxe_pi_magic_soup_with_other_states_with_bimanual mixture
DATASETS = [
    "kuka",
    "bc_z",
    "droid",
    "fractal20220817_data",
    "bridge_v2_oxe",
    "taco_play",
    "jaco_play",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "utaustin_mutex",
    "berkeley_fanuc_manipulation",
    "cmu_stretch",
    "fmb",
    "dobbe",
    "berkeley_autolab_ur5",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "roboturk",
    "austin_buds_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "viola",
    "agibot_dataset",
    "sample_r1_lite",
]

# Base command template
BASE_CMD = (
    "tpu v4 \"source ~/.zshrc && cd openpi-cot && git checkout main && git pull origin main && "
    "uv run --group tpu scripts/vis_oxe_dataset.py pi_combined_cot_v4 "
    "--exp-name=vis_dataset_{dataset} --fsdp-devices=4 --batch-size=16 "
    "--data.shuffle-buffer-size=400 --model.max-token-len=180 "
    "--model.enable-prediction-training --data.no-use-json-actions "
    "--data.data-mix={dataset} --model.prompt-format=schema_compact\""
)


def run_visualization(dataset: str, log_file: Path) -> bool:
    """Run visualization for a single dataset."""
    cmd = BASE_CMD.format(dataset=dataset)

    print(f"\nRunning: {cmd}", flush=True)
    log_file.write(f"\nRunning: {cmd}\n")
    log_file.flush()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}", flush=True)
        log_file.write(f"Error: {e}\n")
        log_file.flush()
        return False


def main():
    # Parse command line arguments
    start_from = None
    only_dataset = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--start-from" and i + 1 < len(args):
            start_from = args[i + 1]
            i += 2
        elif args[i] == "--only" and i + 1 < len(args):
            only_dataset = args[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {args[i]}")
            print("Usage: python scripts/visualize_all_datasets.py [--start-from DATASET_NAME] [--only DATASET_NAME]")
            sys.exit(1)

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"dataset_visualization_{timestamp}.log")

    # Determine which datasets to process
    if only_dataset:
        if only_dataset not in DATASETS:
            print(f"Error: Dataset '{only_dataset}' not found in mixture")
            print(f"Available datasets: {', '.join(DATASETS)}")
            sys.exit(1)
        datasets_to_process = [only_dataset]
    elif start_from:
        if start_from not in DATASETS:
            print(f"Error: Dataset '{start_from}' not found in mixture")
            print(f"Available datasets: {', '.join(DATASETS)}")
            sys.exit(1)
        start_idx = DATASETS.index(start_from)
        datasets_to_process = DATASETS[start_idx:]
    else:
        datasets_to_process = DATASETS

    # Start processing
    with open(log_file, "w") as log:
        start_time = datetime.now()
        print(f"Starting dataset visualization at {start_time}")
        print(f"Total datasets to process: {len(datasets_to_process)}")
        print(f"Log file: {log_file}")
        print("=" * 60)

        log.write(f"Starting dataset visualization at {start_time}\n")
        log.write(f"Total datasets to process: {len(datasets_to_process)}\n")
        log.write("=" * 60 + "\n")
        log.flush()

        results = {}

        for idx, dataset in enumerate(datasets_to_process, 1):
            total = len(datasets_to_process)
            print(f"\n[{idx}/{total}] Processing dataset: {dataset}")
            print(f"Started at: {datetime.now()}")

            log.write(f"\n[{idx}/{total}] Processing dataset: {dataset}\n")
            log.write(f"Started at: {datetime.now()}\n")
            log.flush()

            success = run_visualization(dataset, log)
            results[dataset] = success

            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{status}: {dataset}")
            print(f"Finished at: {datetime.now()}")
            print("-" * 60)

            log.write(f"{status}: {dataset}\n")
            log.write(f"Finished at: {datetime.now()}\n")
            log.write("-" * 60 + "\n")
            log.flush()

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        successful = sum(1 for v in results.values() if v)
        failed = len(results) - successful

        summary = f"""
{'=' * 60}
SUMMARY
{'=' * 60}
Total datasets processed: {len(results)}
Successful: {successful}
Failed: {failed}
Duration: {duration}
Completed at: {end_time}

Results by dataset:
"""
        for dataset, success in results.items():
            status = "✓" if success else "✗"
            summary += f"  {status} {dataset}\n"

        print(summary)
        log.write(summary)

        print(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    main()

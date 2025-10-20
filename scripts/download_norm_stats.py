#!/usr/bin/env python3
"""
Download all norm_stats.json files from GCS bucket gs://pi0-cot/OXE/*/*/norm_stats.json
"""

import os
import subprocess
import json
from pathlib import Path


def main():
    # Output directory for downloaded norm stats
    output_dir = Path(__file__).parent.parent / "norm_stats"
    output_dir.mkdir(exist_ok=True)

    print(f"Downloading norm stats to: {output_dir}")

    # List all norm_stats.json files in the GCS bucket
    gcs_pattern = "gs://pi0-cot/OXE/*/*/norm_stats.json"
    print(f"Searching for files matching: {gcs_pattern}")

    try:
        # Use gsutil to list all matching files
        result = subprocess.run(
            ["gsutil", "ls", gcs_pattern],
            capture_output=True,
            text=True,
            check=True
        )

        file_paths = result.stdout.strip().split('\n')
        file_paths = [f for f in file_paths if f]  # Remove empty strings

        print(f"Found {len(file_paths)} norm_stats.json files")

        # Download each file
        for i, gcs_path in enumerate(file_paths, 1):
            # Extract dataset name and version from path
            # Format: gs://pi0-cot/OXE/dataset_name/version/norm_stats.json
            parts = gcs_path.split('/')
            dataset_name = parts[-3]
            version = parts[-2]

            # Create output filename
            output_filename = f"OXE_{dataset_name}_{version}_norm_stats.json"
            output_path = output_dir / output_filename

            print(f"[{i}/{len(file_paths)}] Downloading {dataset_name}/{version}...")

            try:
                # Download the file
                subprocess.run(
                    ["gsutil", "cp", gcs_path, str(output_path)],
                    check=True,
                    capture_output=True
                )
                print(f"  ✓ Saved to {output_path}")

                # Validate JSON
                with open(output_path, 'r') as f:
                    json.load(f)

            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to download: {e}")
            except json.JSONDecodeError:
                print(f"  ✗ Invalid JSON, removing file")
                output_path.unlink()

        print(f"\n✓ Download complete! Files saved to {output_dir}")
        print(f"  Total: {len(list(output_dir.glob('*.json')))} files")

    except subprocess.CalledProcessError as e:
        print(f"Error listing files: {e}")
        print(f"Make sure gsutil is installed and you have access to gs://pi0-cot/")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

#!/bin/bash

set -e

# Define special case datasets
LOCAL_DATASETS=("kuka" "berkeley_autolab_ur5" "utaustin_mutex" "berkeley_fanuc_manipulation")
FMB_DATASET="fmb"
IGNORE_DATASETS=("stanford_hydra_dataset_converted_externally_to_rlds" "agibot_dataset" "sample_r1_lite")

# Identify all datasets in gs://pi0-cot/OXE/
echo "Identifying datasets in gs://pi0-cot/OXE/..."
datasets=$(gsutil ls gs://pi0-cot/OXE/ | sed 's|gs://pi0-cot/OXE/||' | sed 's|/$||')

for dataset_path in $datasets; do
    dataset_name=$(basename "$dataset_path")

    # Skip ignored datasets
    if [[ " ${IGNORE_DATASETS[@]} " =~ " ${dataset_name} " ]]; then
        echo "Skipping ignored dataset: $dataset_name"
        continue
    fi

    echo "Processing dataset: $dataset_name"

    # Get all versions for this dataset
    versions=$(gsutil ls "gs://pi0-cot/OXE/$dataset_name/" | sed "s|gs://pi0-cot/OXE/$dataset_name/||" | sed 's|/$||')

    for version in $versions; do
        echo "  Version: $version"

        target="gs://v6_east1d/OXE/$dataset_name/$version"

        # Check if target already exists
        if gsutil -q stat "$target/" 2>/dev/null; then
            echo "  Skipping (already exists): $target"
            continue
        fi

        # Check if this is the fmb dataset
        if [ "$dataset_name" == "$FMB_DATASET" ]; then
            echo "  Copying from gs://xembodiment_data/fmb_dataset/1.0.0 -> $target"
            gsutil -m cp -r "gs://xembodiment_data/fmb_dataset/1.0.0" "$target"
        # Check if this is a local dataset
        elif [[ " ${LOCAL_DATASETS[@]} " =~ " ${dataset_name} " ]]; then
            source="/n/fs/vla-mi/datasets/OXE/$dataset_name/$version"
            echo "  Copying from $source -> $target"
            gsutil -m cp -r "$source" "$target"
        else
            source="gs://gresearch/robotics/$dataset_name/$version"
            echo "  Copying from $source -> $target"
            gsutil -m cp -r "$source" "$target"
        fi

        echo "  Completed: $dataset_name/$version"
    done
done

echo "All datasets copied successfully!"

#!/bin/bash

set -e

# Define special case datasets
LOCAL_DATASETS=("kuka" "berkeley_autolab_ur5" "stanford_hydra_dataset_converted_externally_to_rlds" "utaustin_mutex" "berkeley_fanuc_manipulation")
FMB_DATASET="fmb"

# Identify all datasets in gs://pi0-cot/OXE/
echo "Identifying datasets in gs://pi0-cot/OXE/..."
datasets=$(gsutil ls gs://pi0-cot/OXE/ | sed 's|gs://pi0-cot/OXE/||' | sed 's|/$||')

for dataset_path in $datasets; do
    dataset_name=$(basename "$dataset_path")

    echo "Processing dataset: $dataset_name"

    # Get all versions for this dataset
    versions=$(gsutil ls "gs://pi0-cot/OXE/$dataset_name/" | sed "s|gs://pi0-cot/OXE/$dataset_name/||" | sed 's|/$||')

    for version in $versions; do
        echo "  Version: $version"

        target="gs://v6_east1d/OXE/$dataset_name/$version"

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

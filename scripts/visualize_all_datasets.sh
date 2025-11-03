#!/bin/bash

# Script to visualize all datasets in oxe_pi_magic_soup_with_other_states_with_bimanual mixture
# Usage: ./scripts/visualize_all_datasets.sh

# Define all datasets from the mixture (in order as they appear)
DATASETS=(
    # "kuka"
    # "bc_z"
    "droid"
    # "fractal20220817_data"
    # "bridge_v2_oxe"
    "taco_play"
    # "jaco_play"
    "furniture_bench_dataset_converted_externally_to_rlds"
    # "utaustin_mutex"
    "berkeley_fanuc_manipulation"
    # "cmu_stretch"
    "fmb"
    # "dobbe"
    "berkeley_autolab_ur5"
    # "dlr_edan_shared_control_converted_externally_to_rlds"
    # "roboturk"
    # "austin_buds_dataset_converted_externally_to_rlds"
    # "austin_sailor_dataset_converted_externally_to_rlds"
    # "austin_sirius_dataset_converted_externally_to_rlds"
    # "viola"
    # "molmoact_dataset"
    # "agibot_large_dataset"
    # "sample_r1_lite"
)

# Log file for tracking progress
LOG_FILE="dataset_visualization_$(date +%Y%m%d_%H%M%S).log"

echo "Starting dataset visualization at $(date)" | tee -a "$LOG_FILE"
echo "Total datasets to process: ${#DATASETS[@]}" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Counter for progress tracking
COUNTER=0
TOTAL=${#DATASETS[@]}

# Iterate through each dataset
for DATASET in "${DATASETS[@]}"; do
    COUNTER=$((COUNTER + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "[$COUNTER/$TOTAL] Processing dataset: $DATASET" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"

    # Run the visualization command
    tpu v4 "source ~/.zshrc && cd openpi-cot && git checkout main && git pull origin main && uv run --group tpu scripts/vis_oxe_dataset.py pi_combined_cot_v4 --exp-name=${DATASET} --fsdp-devices=4 --batch-size=16 --data.shuffle-buffer-size=4000 --model.max-token-len=180 --model.enable-prediction-training --data.no-use-json-actions --data.data-mix=${DATASET} --data.language_action_format_name=with_rotation --model.prompt_format=pi05" 2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Successfully processed: $DATASET" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed to process: $DATASET (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
    fi

    echo "Finished at: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "All dataset visualizations completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"

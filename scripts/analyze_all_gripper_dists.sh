#!/bin/bash

# Script to analyze gripper distribution for all datasets in oxe_pi_magic_soup_with_other_states_with_bimanual mixture
# Usage: ./scripts/analyze_all_gripper_dists.sh [NUM_BATCHES]


# Define all datasets from the mixture (in order as they appear)
DATASETS=(
    "kuka"
    "bc_z"
    "droid"
    "fractal20220817_data"
    "bridge_v2_oxe"
    "taco_play"
    "jaco_play"
    "furniture_bench_dataset_converted_externally_to_rlds"
    "utaustin_mutex"
    "berkeley_fanuc_manipulation"
    "cmu_stretch"
    "fmb"
    "dobbe"
    "berkeley_autolab_ur5"
    "dlr_edan_shared_control_converted_externally_to_rlds"
    "roboturk"
    "austin_buds_dataset_converted_externally_to_rlds"
    "austin_sailor_dataset_converted_externally_to_rlds"
    "austin_sirius_dataset_converted_externally_to_rlds"
    "viola"
    "agibot_dataset"
    "sample_r1_lite"
)

# Log file for tracking progress
LOG_FILE="gripper_analysis_$(date +%Y%m%d_%H%M%S).log"

echo "Starting gripper distribution analysis at $(date)" | tee -a "$LOG_FILE"
echo "Total datasets to process: ${#DATASETS[@]}" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Counter for progress tracking
COUNTER=0
TOTAL=${#DATASETS[@]}

# Iterate through each dataset
for DATASET in "${DATASETS[@]}"; do
    COUNTER=$((COUNTER + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "[$COUNTER/$TOTAL] Analyzing gripper distribution: $DATASET" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"

    # Run the gripper analysis command
    tpu v4 "source ~/.zshrc && cd openpi-cot && git checkout main && git pull origin main && uv run --group tpu scripts/vis_gripper_distribution.py pi_combined_cot_v4 --exp-name=gripper_dist_${DATASET} --fsdp-devices=4 --batch-size=16 --data.shuffle-buffer-size=400 --model.max-token-len=180 --model.enable-prediction-training --data.no-use-json-actions --data.data-mix=${DATASET} --model.prompt-format=schema_compact --data.vis-dataset" 2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Successfully analyzed: $DATASET" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed to analyze: $DATASET (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
    fi

    echo "Finished at: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "All gripper distribution analyses completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Results are logged to wandb. Check your wandb project for visualizations!"

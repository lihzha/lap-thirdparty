#!/bin/bash
# Submit LIBERO evaluation jobs for multiple policies
# Usage: ./submit_libero_evals.sh

set -euo pipefail

# Define policies to evaluate
# Format: "POLICY_CONFIG|EPOCH_DIR|POLICY_TYPE|JOB_NAME"
POLICIES=(
  "pi_combined_fast_cot_local|checkpoints/pi_combined_fast_cot_v5europe/libero_fast_overfitting_boundsq99noclip_val_pi05/5500|COT_FT|cot_pi05ft_5500"
  "pi_combined_fast_cot_local|checkpoints/pi_combined_fast_cot_v5/libero_fast_overfitting_boundsq99noclip_val/5500|COT_FT|cot_ft_5500"
  "pi_combined_fast_cot_local|checkpoints/pi_combined_fast_cot_v5/libero_fast_overfitting_boundsq99noclip_val/18500|COT_FT|cot_ft_18500"
)

# Common settings
CONTROL_MODE="OSC_POSE"

echo "Submitting LIBERO evaluation jobs..."
echo "========================================"

for policy_entry in "${POLICIES[@]}"; do
  # Parse the policy entry
  IFS='|' read -r policy_config epoch_dir policy_type job_name <<< "$policy_entry"
  
  echo "Submitting job: ${job_name}"
  echo "  Policy config: ${policy_config}"
  echo "  Epoch dir: ${epoch_dir}"
  echo "  Policy type: ${policy_type}"
  
  # Submit the job with environment variables
  job_id=$(sbatch \
    --job-name="libero_${job_name}" \
    --export=ALL,POLICY_CONFIG="${policy_config}",EPOCH_DIR="${epoch_dir}",POLICY_TYPE="${policy_type}",CONTROL_MODE="${CONTROL_MODE}" \
    libero_eval.sh | awk '{print $4}')
  
  echo "  Submitted job ID: ${job_id}"
  echo "----------------------------------------"
done

echo "All jobs submitted!"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs in: logs/"

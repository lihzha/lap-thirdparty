#!/bin/bash
# Submit LIBERO evaluation jobs for multiple policies
# Usage: ./submit_libero_evals.sh

set -euo pipefail

# Define policies to evaluate
# Format: "POLICY_CONFIG|EPOCH_DIR|POLICY_TYPE|JOB_NAME"
POLICIES=(
  # "pi_combined_fast_cot_local|checkpoints/pi_combined_fast_cot_v5europe/libero_fast_overfitting_boundsq99noclip_val_pi05/5500|COT_FT|cot_pi05ft_5500"
  # "pi_combined_fast_cot_local|checkpoints/pi_combined_fast_cot_v5/libero_fast_overfitting_boundsq99noclip_val/5500|COT_FT|cot_ft_5500"
  # "pi_combined_fast_cot_local|checkpoints/pi_combined_fast_cot_v5/libero_fast_overfitting_boundsq99noclip_val/18500|COT_FT|cot_ft_18500"
  # "paligemma_boundsq99|gs://v5_central1_a/checkpoints/pi_combined_cot_v5/rawaction_libero_finetune/10000|COT_FT|raw_finetune_10000"
  # "paligemma_boundsq99|gs://v5_central1_a/checkpoints/pi_combined_cot_v5/ours_libero_finetune/2000|COT_FT|ours_finetune_2000"
  # "paligemma_boundsq99|gs://v5_central1_a/checkpoints/pi_combined_cot_v5/ours_libero_finetune_noki/14000|COT_FT|ours_finetune_noki_14000"
  # "paligemma_boundsq99|gs://v5_central1_a/checkpoints/pi_combined_cot_v5/ours_libero_finetune_noki/18000|COT_FT|ours_finetune_noki_18000"
  # "paligemma_boundsq99|gs://v5_central1_a/checkpoints/pi_combined_cot_v5/ours_libero_finetune_noki_langweight04/18000|COT_FT|ours_finetune_noki_langweight04_18000"
  # "paligemma_boundsq99|gs://v5_central1_a/checkpoints/pi_combined_cot_v5/ours_libero_finetune_noki_langweight08/18000|COT_FT|ours_finetune_noki_langweight08_18000"
  # "gemma3_combined_cot_v4|gs://pi0-cot/checkpoints/gemma3_combined_cot_v4/gemma_try/12000|COT_FT|gemma_try_12000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight01/2000|COT_FT|fast_finetune_noki_langweight01_2000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight01/6000|COT_FT|fast_finetune_noki_langweight01_6000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight01/8000|COT_FT|fast_finetune_noki_langweight01_8000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight01/10000|COT_FT|fast_finetune_noki_langweight01_10000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight01/14000|COT_FT|fast_finetune_noki_langweight01_14000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight04/2000|COT_FT|fast_finetune_noki_langweight04_2000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight04/6000|COT_FT|fast_finetune_noki_langweight04_6000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight04/8000|COT_FT|fast_finetune_noki_langweight04_8000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight04/10000|COT_FT|fast_finetune_noki_langweight04_10000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight04/14000|COT_FT|fast_finetune_noki_langweight04_14000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight08/2000|COT_FT|fast_finetune_noki_langweight08_2000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight08/6000|COT_FT|fast_finetune_noki_langweight08_6000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight08/8000|COT_FT|fast_finetune_noki_langweight08_8000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight08/10000|COT_FT|fast_finetune_noki_langweight08_10000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight08/14000|COT_FT|fast_finetune_noki_langweight08_14000"
  # "pi_combined_vla0_v6|gs://v6_east1d/checkpoints/pi_combined_vla0_v6/vla0_2048/30000|COT_FT|vla0_30000_raw"
  # "pi_combined_vla0_v6|gs://v6_east1d/checkpoints/pi_combined_vla0_v6/vla0_2048/30000|COT_FT|vla0_30000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight08/18000|COT_FT|fast_finetune_noki_langweight08_18000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight04/18000|COT_FT|fast_finetune_noki_langweight04_18000"
  # "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight01/18000|COT_FT|fast_finetune_noki_langweight01_18000"

  "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight08_first/500|COT_FT|fast_finetune_noki_langweight08_500"
  "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight08_first/1000|COT_FT|fast_finetune_noki_langweight08_1000"
  "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight08_first/1500|COT_FT|fast_finetune_noki_langweight08_1500"

  "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight01_first/500|COT_FT|fast_finetune_noki_langweight01_500"
  "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight01_first/1000|COT_FT|fast_finetune_noki_langweight01_1000"
  "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight01_first/1500|COT_FT|fast_finetune_noki_langweight01_1500"

  "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight04_first/500|COT_FT|fast_finetune_noki_langweight04_500"
  "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight04_first/1000|COT_FT|fast_finetune_noki_langweight04_1000"
  "pi_combined_fast_cot_local|gs://v5_central1_a/checkpoints/pi_combined_fast_cot_v5/fast_libero_finetune_noki_langweight04_first/1500|COT_FT|fast_finetune_noki_langweight04_1500"
  
  "paligemma_boundsq99|gs://v5_central1_a/checkpoints/pi_combined_cot_v5/ours_libero_finetune_noki_langweight08_first/500|COT_FT|ours_finetune_noki_500"
  "paligemma_boundsq99|gs://v5_central1_a/checkpoints/pi_combined_cot_v5/ours_libero_finetune_noki_langweight08_first/1000|COT_FT|ours_finetune_noki_1000"
  "paligemma_boundsq99|gs://v5_central1_a/checkpoints/pi_combined_cot_v5/ours_libero_finetune_noki_langweight08_first/1500|COT_FT|ours_finetune_noki_1500"

  "paligemma_boundsq99|gs://v5_europewest4/checkpoints/pi_combined_cot_v5europe/ours_libero_finetune_noki_langweight01_first/500|COT_FT|ours_finetune_noki_500"
  "paligemma_boundsq99|gs://v5_europewest4/checkpoints/pi_combined_cot_v5europe/ours_libero_finetune_noki_langweight01_first/1000|COT_FT|ours_finetune_noki_1000"
  "paligemma_boundsq99|gs://v5_europewest4/checkpoints/pi_combined_cot_v5europe/ours_libero_finetune_noki_langweight01_first/1500|COT_FT|ours_finetune_noki_1500"

  "paligemma_boundsq99|gs://v5_europewest4/checkpoints/pi_combined_cot_v5europe/ours_libero_finetune_noki_langweight04_first/500|COT_FT|ours_finetune_noki_500"
  "paligemma_boundsq99|gs://v5_europewest4/checkpoints/pi_combined_cot_v5europe/ours_libero_finetune_noki_langweight04_first/1000|COT_FT|ours_finetune_noki_1000"
  "paligemma_boundsq99|gs://v5_europewest4/checkpoints/pi_combined_cot_v5europe/ours_libero_finetune_noki_langweight04_first/1500|COT_FT|ours_finetune_noki_1500"
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

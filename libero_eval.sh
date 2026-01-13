#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/openpi-%j.log
#SBATCH --error=logs/openpi-%j.err

set -uo pipefail  # Removed -e to allow proper cleanup on background process failure
cd "$SLURM_SUBMIT_DIR"

# Accept parameters from environment or use defaults
POLICY_CONFIG="${POLICY_CONFIG:-pi_combined_fast_cot_local}"
EPOCH_DIR="${EPOCH_DIR:-checkpoints/pi_combined_fast_cot_v5europe/libero_fast_overfitting_boundsq99noclip_val_pi05/5500}"
POLICY_TYPE="${POLICY_TYPE:-COT_FT}"
CONTROL_MODE="${CONTROL_MODE:-OSC_POSE}"

export JAX_PLATFORMS=cuda
export OPENPI_DATA_HOME="/n/fs/robot-data/cache/openpi"
export LIBERO_CONFIG_PATH=third_party/openpi/third_party/libero
PYTHONPATH=${PYTHONPATH:-}
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/third_party/openpi/third_party/libero"

# Ensure log directory exists
mkdir -p logs

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Policy config: ${POLICY_CONFIG}"
echo "Epoch dir: ${EPOCH_DIR}"
echo "Policy type: ${POLICY_TYPE}"
echo "Control mode: ${CONTROL_MODE}"
echo "========================================"

# Start the policy server in background with separate logs
uv run --group cuda --active scripts/serve_policy.py \
  policy:checkpoint --policy.config="${POLICY_CONFIG}" --policy.dir="${EPOCH_DIR}" \
  > "logs/serve-${SLURM_JOB_ID}.out" 2> "logs/serve-${SLURM_JOB_ID}.err" &
policy_pid=$!
echo "Started policy server with PID $policy_pid"

# Give the server time to start up
sleep 30

# Function to cleanup on exit
cleanup() {
  echo "Cleaning up processes..."
  kill $policy_pid 2>/dev/null || true
}
trap cleanup EXIT

# Task suites to evaluate (excluding libero_90)
TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# Loop through each task suite
for task_suite in "${TASK_SUITES[@]}"; do
  echo "========================================"
  echo "Starting evaluation for: ${task_suite}"
  echo "========================================"
  
  (
    source scripts/libero/.venv/bin/activate
    python scripts/libero/main.py \
      --args.policy-type "${POLICY_TYPE}" \
      --args.control-mode "${CONTROL_MODE}" \
      --args.task-suite-name "${task_suite}"
  ) >> "logs/libero-${SLURM_JOB_ID}.out" 2>> "logs/libero-${SLURM_JOB_ID}.err"
  
  libero_exit=$?
  echo "Task suite ${task_suite} finished with exit code ${libero_exit}"
  
  if [ $libero_exit -ne 0 ]; then
    echo "WARNING: ${task_suite} failed with exit code ${libero_exit}"
  fi
done

echo "All task suites completed!"

# Kill the server after all tasks finish
kill $policy_pid 2>/dev/null || true
wait $policy_pid 2>/dev/null || true

exit 0

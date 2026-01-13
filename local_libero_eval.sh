#!/bin/bash
# Run both policy server and LIBERO client locally in a single terminal
# Usage: ./local_libero_eval.sh [OPTIONS]
#
# Options (via environment variables or command line):
#   POLICY_CONFIG  - Policy configuration name (default: pi_combined_fast_cot_local)
#   EPOCH_DIR      - Checkpoint directory path
#   POLICY_TYPE    - Policy type: COT_FT, COT, RAW (default: COT_FT)
#   CONTROL_MODE   - Control mode: OSC_POSE, JOINT (default: OSC_POSE)
#   TASK_SUITES    - Comma-separated task suites (default: libero_spatial,libero_object,libero_goal,libero_10)
#
# Examples:
#   ./local_libero_eval.sh
#   ./local_libero_eval.sh EPOCH_DIR=checkpoints/my_model/1000
#   ./local_libero_eval.sh EPOCH_DIR=gs://bucket/path POLICY_CONFIG=my_config TASK_SUITES=libero_spatial

set -uo pipefail  # Removed -e to allow proper cleanup on background process failure

# Get script directory and cd to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse KEY=VALUE arguments from command line
for arg in "$@"; do
  if [[ "$arg" == *"="* ]]; then
    key="${arg%%=*}"
    value="${arg#*=}"
    export "$key"="$value"
  fi
done

# Accept parameters from environment or use defaults
POLICY_CONFIG="${POLICY_CONFIG:-pi_combined_fast_cot_local}"
EPOCH_DIR="${EPOCH_DIR:-checkpoints/pi_combined_fast_cot_v5europe/libero_fast_overfitting_boundsq99noclip_val_pi05/5500}"
POLICY_TYPE="${POLICY_TYPE:-COT_FT}"
CONTROL_MODE="${CONTROL_MODE:-OSC_POSE}"
TASK_SUITES_STR="${TASK_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"

# Parse task suites from comma-separated string
IFS=',' read -ra TASK_SUITES <<< "$TASK_SUITES_STR"

export JAX_PLATFORMS=cuda
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/n/fs/robot-data/cache/openpi}"
export LIBERO_CONFIG_PATH=third_party/openpi/third_party/libero
PYTHONPATH=${PYTHONPATH:-}
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/third_party/openpi/third_party/libero"

echo "========================================"
echo "Local LIBERO Evaluation"
echo "========================================"
echo "Policy config: ${POLICY_CONFIG}"
echo "Epoch dir: ${EPOCH_DIR}"
echo "Policy type: ${POLICY_TYPE}"
echo "Control mode: ${CONTROL_MODE}"
echo "Task suites: ${TASK_SUITES[*]}"
echo "========================================"

# Function to cleanup on exit
cleanup() {
  echo ""
  echo "Cleaning up processes..."
  if [ -n "${policy_pid:-}" ]; then
    kill $policy_pid 2>/dev/null || true
    wait $policy_pid 2>/dev/null || true
    echo "Policy server stopped."
  fi
}
trap cleanup EXIT INT TERM

# Start the policy server in background (output goes to terminal)
echo "Starting policy server..."
uv run --group cuda --active scripts/serve_policy.py \
  policy:checkpoint --policy.config="${POLICY_CONFIG}" --policy.dir="${EPOCH_DIR}" &
policy_pid=$!
echo "Started policy server with PID $policy_pid"

# Wait for server to be ready (check if port 8000 is listening)
SERVER_PORT="${SERVER_PORT:-8000}"
MAX_WAIT="${MAX_WAIT:-600}"  # Max 10 minutes for checkpoint download
echo "Waiting for server to be ready on port ${SERVER_PORT} (max ${MAX_WAIT}s)..."

waited=0
while ! nc -z localhost "$SERVER_PORT" 2>/dev/null; do
  # Check if server process died
  if ! kill -0 $policy_pid 2>/dev/null; then
    echo "ERROR: Policy server died during startup."
    exit 1
  fi
  
  if [ $waited -ge $MAX_WAIT ]; then
    echo "ERROR: Timeout waiting for server to be ready."
    exit 1
  fi
  
  sleep 5
  waited=$((waited + 5))
  echo "  Still waiting... (${waited}s elapsed)"
done

echo "Policy server is ready! (took ${waited}s)"

# Track success/failure
declare -A results

# Loop through each task suite
for task_suite in "${TASK_SUITES[@]}"; do
  echo ""
  echo "========================================"
  echo "Starting evaluation for: ${task_suite}"
  echo "========================================"
  
  (
    source scripts/libero/.venv/bin/activate
    python scripts/libero/main.py \
      --args.policy-type "${POLICY_TYPE}" \
      --args.control-mode "${CONTROL_MODE}" \
      --args.task-suite-name "${task_suite}"
  )
  
  libero_exit=$?
  
  if [ $libero_exit -eq 0 ]; then
    echo "✓ ${task_suite} completed successfully"
    results["$task_suite"]="SUCCESS"
  else
    echo "✗ ${task_suite} failed with exit code ${libero_exit}"
    results["$task_suite"]="FAILED (exit $libero_exit)"
  fi
done

echo ""
echo "========================================"
echo "Evaluation Summary"
echo "========================================"
for task_suite in "${TASK_SUITES[@]}"; do
  echo "  ${task_suite}: ${results[$task_suite]}"
done
echo "========================================"
echo "All task suites completed!"

# Kill the server
kill $policy_pid 2>/dev/null || true
wait $policy_pid 2>/dev/null || true

exit 0

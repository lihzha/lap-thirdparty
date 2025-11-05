#!/bin/bash
set -e  # Exit on error

# Track server PID for cleanup
SERVER_PID=""

gscp() {
  local src="$1"
  local dst="$2"

  # Ensure dest dir exists for multi-source copy
  if [[ "$dst" != gs://* ]]; then
    mkdir -p "$dst"
  fi

  # Force directory semantics for multi-source copy
  dst="${dst%/}/"

  # Copy both directories
  gsutil -m cp -r "${src}/assets" "${src}/params" "${dst}"
}

# Trap to cleanup server on script exit (Ctrl+C, error, etc.)
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Cleaning up server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup INT TERM EXIT

CONFIG_NAME=pi05_libero_finetune_v4_freezevlm
CKPT_NAME=libero_combined_finetune_freezevlm
CKPT_BASE_DIR=~/projects/openpi-cot/checkpoints/$CONFIG_NAME/$CKPT_NAME
GCS_PATH=gs://pi0-cot/checkpoints/$CONFIG_NAME/$CKPT_NAME

# Allow user to specify epoch or list first one
if [ -n "$1" ]; then
    TEST_EPOCH="$1"
    echo "Testing with user-specified epoch: $TEST_EPOCH"
else
    # List and get first epoch from GCS
    echo "Listing epoch folders from $GCS_PATH..."
    # Filter for directories ending with /, exclude parent dir and files, only numeric folder names
    TEST_EPOCH=$(gsutil ls "$GCS_PATH/" 2>/dev/null | grep -E "/$" | grep -v "^${GCS_PATH}/$" | while read -r line; do
        epoch=$(basename "$line")
        # Only include if epoch looks like a number (actual epoch folders)
        if [ -n "$epoch" ] && [ "$epoch" != "$CKPT_NAME" ] && [[ "$epoch" =~ ^[0-9]+$ ]]; then
            echo "$epoch"
        fi
    done | head -n 1)

    if [ -z "$TEST_EPOCH" ]; then
        echo "No epoch subdirectories found in GCS. Using checkpoint directory directly."
        TEST_EPOCH="."
    else
        echo "Found first epoch: $TEST_EPOCH"
    fi
fi

# Create base checkpoint directory if it doesn't exist
mkdir -p "$CKPT_BASE_DIR"

echo "=========================================="
echo "Testing download for epoch: $TEST_EPOCH"
echo "=========================================="

# Determine checkpoint paths
if [ "$TEST_EPOCH" = "." ]; then
    EPOCH_DIR=$CKPT_BASE_DIR
    GCS_EPOCH_PATH=$GCS_PATH
else
    EPOCH_DIR=$CKPT_BASE_DIR/$TEST_EPOCH
    GCS_EPOCH_PATH=$GCS_PATH/$TEST_EPOCH
fi


mkdir -p "$EPOCH_DIR"
echo $EPOCH_DIR
# Clean up any existing directory first
echo "Cleaning up any existing checkpoint at $EPOCH_DIR..."
rm -rf "$EPOCH_DIR" 2>/dev/null || true


# Download this specific epoch from GCS
echo "Downloading epoch from $GCS_EPOCH_PATH..."
if ! gscp $GCS_EPOCH_PATH $EPOCH_DIR 2>/dev/null; then
    echo "ERROR: Failed to download epoch $TEST_EPOCH from GCS."
    exit 1
fi

# Verify the checkpoint directory exists and has content
if [ ! -d "$EPOCH_DIR" ] || [ -z "$(ls -A "$EPOCH_DIR" 2>/dev/null)" ]; then
    echo "ERROR: Downloaded checkpoint directory is missing or empty."
    rm -rf "$EPOCH_DIR" 2>/dev/null || true
    exit 1
fi

echo "Successfully downloaded checkpoint for epoch $TEST_EPOCH"
echo "Checkpoint location: $EPOCH_DIR"
echo ""
echo "Contents:"
ls -lh "$EPOCH_DIR"
echo ""

# Test starting the policy server
echo "=========================================="
echo "Testing policy server startup..."
echo "=========================================="

LOG_FILE="/tmp/policy_server_test.log"
echo "Starting policy server in background..."
JAX_PLATFORMS=cuda OPENPI_DATA_HOME=~/.cache/openpi uv run --group cuda --active scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero_finetune_local --policy.dir=$EPOCH_DIR --policy.type=raw > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

echo "Policy server started with PID: $SERVER_PID (log: $LOG_FILE)"

# Wait for server to be ready (check for up to 120 seconds)
echo "Waiting for server to be ready..."
SERVER_READY=false
for i in {1..120}; do
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "ERROR: Server process died! Check log: $LOG_FILE"
        echo "Last 50 lines of log:"
        tail -n 50 "$LOG_FILE"
        exit 1
    fi

    # Check if server is responding via health check endpoint
    if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
        echo "Server is ready after ${i} seconds (health check passed)"
        SERVER_READY=true
        break
    fi

    # Also check for "Creating server" log message as a fallback
    if grep -q "Creating server" "$LOG_FILE" 2>/dev/null; then
        # Give it a few more seconds after seeing the log
        sleep 3
        if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
            echo "Server is ready after $((i+3)) seconds (health check passed)"
            SERVER_READY=true
            break
        fi
    fi

    sleep 1
done

if [ "$SERVER_READY" = false ]; then
    echo "ERROR: Server did not become ready after 120 seconds"
    echo "Last 50 lines of log:"
    tail -n 50 "$LOG_FILE"
    exit 1
fi

echo ""
echo "=========================================="
echo "Running evaluation..."
echo "=========================================="

source /home/irom-lab/projects/openpi-cot/scripts/libero/.venv/bin/activate
LIBERO_CONFIG_PATH=/home/irom-lab/projects/openpi-cot/third_party/openpi/third_party/libero PYTHONPATH=$PYTHONPATH:$PWD/third_party/openpi/third_party/libero python scripts/libero/main.py --args.policy-type FT --args.control-mode OSC_POSE
EVAL_STATUS=$?

echo ""
echo "=========================================="
echo "Shutting down policy server..."
echo "=========================================="
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
SERVER_PID=""  # Reset so cleanup trap doesn't try again

if [ $EVAL_STATUS -eq 0 ]; then
    echo "Evaluation completed successfully for epoch: $TEST_EPOCH"
else
    echo "Evaluation failed for epoch: $TEST_EPOCH (exit code: $EVAL_STATUS)"
    exit $EVAL_STATUS
fi

echo ""
echo "=========================================="
echo "Cleaning up downloaded checkpoint..."
echo "=========================================="
if [ "$TEST_EPOCH" = "." ]; then
    rm -rf "$CKPT_BASE_DIR" 2>/dev/null || true
else
    rm -rf "$EPOCH_DIR" 2>/dev/null || true
fi
echo "Cleaned up: $EPOCH_DIR"

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "=========================================="

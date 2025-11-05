#!/bin/bash
set -e  # Exit on error

# Track server PID for cleanup
SERVER_PID=""

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
CKPT_BASE_DIR=~/projects/openpi-cot/checkpoints/$CONFIG_NAME
GCS_PATH=gs://pi0-cot/checkpoints/$CONFIG_NAME/$CKPT_NAME

# 1. List all epoch folders from GCS
echo "Listing epoch folders from $GCS_PATH..."
all_epochs=()

# Use gsutil to list directories (ending with /)
while IFS= read -r line; do
    # Extract just the epoch name (last part of the path)
    epoch=$(basename "$line")
    # Skip empty lines and the parent directory itself
    # Only include if epoch looks like a number (actual epoch folders)
    if [ -n "$epoch" ] && [ "$epoch" != "$CKPT_NAME" ] && [[ "$epoch" =~ ^[0-9]+$ ]]; then
        all_epochs+=("$epoch")
    fi
done < <(gsutil ls "$GCS_PATH/" 2>/dev/null | grep -E "/$" | grep -v "^${GCS_PATH}/$")

# If no subdirectories found, try downloading the checkpoint directory directly
if [ ${#all_epochs[@]} -eq 0 ]; then
    echo "No epoch subdirectories found in GCS. Will try to use checkpoint directory directly."
    all_epochs=(".")
fi

echo "Found ${#all_epochs[@]} epoch(s) to evaluate: ${all_epochs[*]}"

# Create base checkpoint directory if it doesn't exist
mkdir -p "$CKPT_BASE_DIR"

# 2. Eval sequentially - download each epoch, eval, then cleanup
for epoch in "${all_epochs[@]}"; do
    echo "=========================================="
    echo "Evaluating epoch: $epoch"
    echo "=========================================="

    # Determine checkpoint paths
    if [ "$epoch" = "." ]; then
        EPOCH_DIR=$CKPT_BASE_DIR/$CKPT_NAME
        GCS_EPOCH_PATH=$GCS_PATH
    else
        EPOCH_DIR=$CKPT_BASE_DIR/$CKPT_NAME/$epoch
        GCS_EPOCH_PATH=$GCS_PATH/$epoch
    fi

    # Download this specific epoch from GCS
    echo "Downloading epoch from $GCS_EPOCH_PATH to $EPOCH_DIR..."
    if ! gscp "$GCS_EPOCH_PATH" "$(dirname "$EPOCH_DIR")" 2>/dev/null; then
        echo "ERROR: Failed to download epoch $epoch from GCS. Skipping this epoch."
        continue
    fi

    # Verify the checkpoint directory exists and has content
    if [ ! -d "$EPOCH_DIR" ] || [ -z "$(ls -A "$EPOCH_DIR" 2>/dev/null)" ]; then
        echo "ERROR: Downloaded checkpoint directory is missing or empty. Skipping epoch $epoch."
        rm -rf "$EPOCH_DIR" 2>/dev/null || true
        continue
    fi

    echo "Successfully downloaded checkpoint for epoch $epoch"

    echo "Step 1: Starting policy server in background..."
    LOG_FILE="/tmp/policy_server_${epoch}.log"
    uv run --active scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero_eval --policy.dir=$EPOCH_DIR > "$LOG_FILE" 2>&1 &
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
            rm -rf "$EPOCH_DIR" 2>/dev/null || true
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
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        SERVER_PID=""
        rm -rf "$EPOCH_DIR" 2>/dev/null || true
        continue
    fi

    echo ""
    echo "Step 2: Running evaluation..."
    python scripts/libero/main.py --policy-type FT --control-model OSC_POSE
    EVAL_STATUS=$?

    echo ""
    echo "Step 3: Shutting down policy server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    SERVER_PID=""  # Reset so cleanup trap doesn't try again

    if [ $EVAL_STATUS -eq 0 ]; then
        echo "Evaluation completed successfully for epoch: $epoch"
    else
        echo "Evaluation failed for epoch: $epoch (exit code: $EVAL_STATUS)"
    fi

    echo ""
    echo "Step 4: Cleaning up downloaded checkpoint..."
    if [ "$epoch" = "." ]; then
        # Clean up the entire checkpoint directory
        rm -rf "$CKPT_BASE_DIR/$CKPT_NAME" 2>/dev/null || true
    else
        # Clean up just this epoch directory
        rm -rf "$EPOCH_DIR" 2>/dev/null || true
    fi
    echo "Cleaned up: $EPOCH_DIR"

    echo "Waiting 5 seconds before next epoch..."
    sleep 5
done

# Disable trap as we're done
trap - INT TERM EXIT
echo "All epochs evaluated!"
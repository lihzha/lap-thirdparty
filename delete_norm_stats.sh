#!/bin/bash
# delete_norm_stats.sh
# Deletes all norm_stats.json files under gs://pi0-cot/OXE/*/*/

set -euo pipefail

BUCKET="gs://pi0-cot/OXE"

echo "Finding all norm_stats.json files under $BUCKET ..."
FILES=$(gsutil ls -r "$BUCKET/**/norm_stats.json" 2>/dev/null || true)

if [ -z "$FILES" ]; then
    echo "No norm_stats.json files found."
    exit 0
fi

echo "Found the following files:"
echo "$FILES"

read -p "Are you sure you want to delete these files? [y/N] " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Deleting files..."
    echo "$FILES" | xargs -n 1 gsutil rm
    echo "Deletion complete."
else
    echo "Aborted."
fi

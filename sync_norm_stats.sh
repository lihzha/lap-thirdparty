#!/usr/bin/env bash
set -euo pipefail

SRC_BUCKET="gs://v6_east1d/OXE"
DST_BUCKET="gs://v5_central1_a/OXE"
MISSING_LOG="missing_norm_stats.txt"

declare -A TRANSMITTED=()

: > "${MISSING_LOG}"

# Find all norm_stats.json files under the source bucket.
while IFS= read -r norm_path; do
  # Expect path like: gs://v6_east1d/OXE/<dataset>/<version>/norm_stats.json
  rel_path="${norm_path#${SRC_BUCKET}/}"
  dataset_name="${rel_path%%/*}"
  rest="${rel_path#*/}"
  version="${rest%%/*}"

  dst_prefix="${DST_BUCKET}/${dataset_name}/${version}"
  if gsutil -q stat "${dst_prefix}/**" >/dev/null 2>&1; then
    gsutil -m cp "${norm_path}" "${dst_prefix}/norm_stats.json"
    TRANSMITTED["${dataset_name}/${version}"]=1
  fi
done < <(gsutil ls "${SRC_BUCKET}/*/*/norm_stats.json" 2>/dev/null || true)

# Record destination folders that did not receive updated norm stats.
while IFS= read -r dst_path; do
  rel_path="${dst_path#${DST_BUCKET}/}"
  dataset_name="${rel_path%%/*}"
  rest="${rel_path#*/}"
  version="${rest%%/*}"
  key="${dataset_name}/${version}"

  if [[ -z "${TRANSMITTED[${key}]:-}" ]]; then
    echo "${key}" >> "${MISSING_LOG}"
  fi
done < <(gsutil ls -d "${DST_BUCKET}/*/*/" 2>/dev/null || true)

echo "Done. Missing entries recorded in ${MISSING_LOG}"

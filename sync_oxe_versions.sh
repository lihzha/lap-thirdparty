#!/usr/bin/env bash
set -euo pipefail

SRC_INDEX_BUCKET="gs://v6_east1d/OXE"
SRC_DATA_BUCKET="gs://gresearch/robotics"
DST_BUCKET="gs://v5_central1_a/OXE"
MISSING_LOG="missing_versions.txt"
SKIP_DATASETS=(
  droid
  agibot_dataset
  agibot_large_dataset
  coco_captions
  droid_100
  lvis
  paco_ego4d
  paco_lvis
  roboturk
  sample_r1_lite
  ucsd_kitchen_dataset_externally_to_rlds
)

: > "${MISSING_LOG}"

# List dataset folders under the index bucket.
while IFS= read -r dataset_path; do
  dataset_name="$(basename "${dataset_path%/}")"
  for skip in "${SKIP_DATASETS[@]}"; do
    if [[ "${dataset_name}" == "${skip}" ]]; then
      continue 2
    fi
  done

  # List versions inside the dataset folder.
  while IFS= read -r version_path; do
    version="$(basename "${version_path%/}")"
    echo "Processing ${dataset_name} version ${version}..."
    if [[ "${dataset_name}" == "bc_z" && "${version}" != "0.1.0" ]]; then
      continue
    fi
    src_path="${SRC_DATA_BUCKET}/${dataset_name}/${version}"
    dst_path="${DST_BUCKET}/${dataset_name}/${version}"

    if gsutil -q stat "${src_path}/**" >/dev/null 2>&1; then
      gsutil -m cp -r "${src_path}" "${dst_path}"
    else
      echo "${dataset_name}/${version}" >> "${MISSING_LOG}"
    fi
  done < <(gsutil ls -d "${dataset_path}*/" 2>/dev/null || true)
done < <(gsutil ls -d "${SRC_INDEX_BUCKET}/*/" 2>/dev/null || true)

echo "Done. Missing entries recorded in ${MISSING_LOG}"

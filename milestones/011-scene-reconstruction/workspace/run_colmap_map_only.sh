#!/bin/bash
# CPU-only: Mapper step (runs on jdp-mac after GPU extract+match on outpost)
#
# Usage: ./run_colmap_map_only.sh <scene>
#   e.g. ./run_colmap_map_only.sh 001-patio-fisheye
#
# Prereqs:
#   - database.db from outpost (GPU extract+match)
#   - images/ directory with the same frames
#
# This runs locally on jdp-mac (no Docker, no GPU needed)

set -euo pipefail

SCENE="${1:?Usage: $0 <scene-name>}"
BASE_DIR="${COLMAP_DATA_DIR:-/var/krabby/workspace/milestones/011-scene-reconstruction/data}"
SCENE_DIR="${BASE_DIR}/scenes/${SCENE}"
IMAGE_DIR="${SCENE_DIR}/images"
SPARSE_DIR="${SCENE_DIR}/sparse"
DB_PATH="${SCENE_DIR}/database.db"

if [ ! -f "${DB_PATH}" ]; then
    echo "ERROR: Database not found: ${DB_PATH}"
    echo "Run GPU extract+match on outpost first, then ship database.db here."
    exit 1
fi

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "ERROR: Images not found: ${IMAGE_DIR}"
    exit 1
fi

NUM_IMAGES=$(ls "${IMAGE_DIR}"/*.jpg 2>/dev/null | wc -l)
echo "=== COLMAP Mapper (CPU, jdp-mac) ==="
echo "Scene:    ${SCENE}"
echo "Database: ${DB_PATH}"
echo "Images:   ${NUM_IMAGES}"
echo "Output:   ${SPARSE_DIR}"
echo ""

rm -rf "${SPARSE_DIR}"
mkdir -p "${SPARSE_DIR}"

echo "Running mapper..."
time colmap mapper \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGE_DIR}" \
    --output_path "${SPARSE_DIR}" \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_extra_params 1 \
    --Mapper.init_max_error 8 \
    --Mapper.abs_pose_min_num_inliers 15 \
    --Mapper.init_min_num_inliers 50 \
    --Mapper.multiple_models 0

echo ""
echo "=== Results ==="
for d in "${SPARSE_DIR}"/*/; do
    colmap model_analyzer --path "$d" 2>&1
done

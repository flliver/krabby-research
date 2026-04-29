#!/bin/bash
# GPU-only: Feature extraction + exhaustive matching
# Run on outpost (GPU), then ship database to jdp-mac for mapper
#
# Usage: ./run_colmap_match_only.sh <scene> [camera_model]
#   e.g. ./run_colmap_match_only.sh 001-patio-fisheye SIMPLE_RADIAL_FISHEYE
#
# Output: database.db (ship to jdp-mac for mapper step)

set -euo pipefail

SCENE="${1:?Usage: $0 <scene-name> [camera_model]}"
CAMERA_MODEL="${2:-SIMPLE_RADIAL}"
SCENE_DIR="/data/scenes/${SCENE}"
IMAGE_DIR="${SCENE_DIR}/images"
DB_PATH="${SCENE_DIR}/database.db"

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "ERROR: Image directory not found: ${IMAGE_DIR}"
    exit 1
fi

NUM_IMAGES=$(ls "${IMAGE_DIR}"/*.jpg 2>/dev/null | wc -l)
echo "=== COLMAP GPU Extract + Match ==="
echo "Scene:        ${SCENE}"
echo "Camera model: ${CAMERA_MODEL}"
echo "Images:       ${NUM_IMAGES}"
echo ""

rm -f "${DB_PATH}"

echo "[1/2] Feature extraction (GPU SIFT)..."
time colmap feature_extractor \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGE_DIR}" \
    --ImageReader.camera_model "${CAMERA_MODEL}" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.max_num_features 8192

echo ""
echo "[2/2] Exhaustive matching (GPU)..."
time colmap exhaustive_matcher \
    --database_path "${DB_PATH}" \
    --SiftMatching.use_gpu 1

echo ""
echo "=== Done — GPU steps complete ==="
echo "Database: ${DB_PATH}"
echo ""
echo "Ship to jdp-mac for mapper:"
echo "  scp <outpost>:~/outposts/krabby/data/011-scene-reconstruction/scenes/${SCENE}/database.db \\"
echo "    /var/krabby/workspace/milestones/011-scene-reconstruction/data/scenes/${SCENE}/"

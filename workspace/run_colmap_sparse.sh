#!/bin/bash
# T0: COLMAP Sparse Reconstruction
# Input:  /data/scenes/<scene>/images/  (extracted JPEG frames)
# Output: /data/scenes/<scene>/sparse/  (camera poses + sparse point cloud)
#
# Usage: ./run_colmap_sparse.sh <scene> [camera_model]
#   e.g. ./run_colmap_sparse.sh 001-patio-fisheye SIMPLE_RADIAL_FISHEYE
#         ./run_colmap_sparse.sh 002-patio-dewarped SIMPLE_RADIAL
#
# Camera models:
#   SIMPLE_RADIAL_FISHEYE - for fisheye/ultra-wide lenses (DJI Action 3 native)
#   SIMPLE_RADIAL         - for dewarped video or normal lenses
#   OPENCV_FISHEYE        - full fisheye model (more params, less stable init)

set -euo pipefail

SCENE="${1:?Usage: $0 <scene-name> [camera_model]}"
CAMERA_MODEL="${2:-SIMPLE_RADIAL}"
SCENE_DIR="/data/scenes/${SCENE}"
IMAGE_DIR="${SCENE_DIR}/images"
SPARSE_DIR="${SCENE_DIR}/sparse"
DB_PATH="${SCENE_DIR}/database.db"

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "ERROR: Image directory not found: ${IMAGE_DIR}"
    exit 1
fi

NUM_IMAGES=$(ls "${IMAGE_DIR}"/*.jpg 2>/dev/null | wc -l)
echo "=== COLMAP Sparse Reconstruction ==="
echo "Scene:        ${SCENE}"
echo "Camera model: ${CAMERA_MODEL}"
echo "Images:       ${NUM_IMAGES} frames in ${IMAGE_DIR}"
echo "Output:       ${SPARSE_DIR}"
echo ""

# Clean previous run
rm -rf "${SPARSE_DIR}"/* "${DB_PATH}"
mkdir -p "${SPARSE_DIR}"

# Detect CUDA support in COLMAP
USE_GPU=1
if colmap help 2>&1 | grep -q "without CUDA"; then
    echo "NOTE: COLMAP built without CUDA — using CPU"
    USE_GPU=0
fi

# Step 1: Feature extraction
# Single camera — all frames from same video/device
echo "[1/3] Feature extraction (SIFT)..."
colmap feature_extractor \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGE_DIR}" \
    --ImageReader.camera_model "${CAMERA_MODEL}" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu "${USE_GPU}" \
    --SiftExtraction.max_num_features 8192

# Step 2: Feature matching
# Sequential matching for video/hyperlapse frames with generous overlap
echo "[2/3] Feature matching (sequential, overlap=15)..."
colmap sequential_matcher \
    --database_path "${DB_PATH}" \
    --SiftMatching.use_gpu "${USE_GPU}" \
    --SequentialMatching.overlap 15

# Step 3: Sparse reconstruction (mapper)
echo "[3/3] Sparse reconstruction (mapper)..."
colmap mapper \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGE_DIR}" \
    --output_path "${SPARSE_DIR}" \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_extra_params 1 \
    --Mapper.init_max_error 8 \
    --Mapper.abs_pose_min_num_inliers 15 \
    --Mapper.init_min_num_inliers 50 \
    --Mapper.multiple_models 0

# Report results
echo ""
echo "=== Done ==="
if [ -d "${SPARSE_DIR}/0" ]; then
    echo "Sparse model written to: ${SPARSE_DIR}/0/"
    echo ""
    echo "Model stats:"
    colmap model_analyzer --path "${SPARSE_DIR}/0" 2>&1 || true
else
    echo "WARNING: No reconstruction found. Check image quality / overlap."
    ls -la "${SPARSE_DIR}/"
fi

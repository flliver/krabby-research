#!/bin/bash
# T1: COLMAP Dense Reconstruction (MVS)
# Input:  /data/scenes/<scene>/sparse/0/  (from T0)
# Output: /data/scenes/<scene>/dense/fused.ply  (dense point cloud)
#
# Usage: ./run_colmap_dense.sh <scene>
#   e.g. ./run_colmap_dense.sh 002-patio-dewarped
#
# Requires: GPU for patch_match_stereo (CPU fallback is extremely slow)

set -euo pipefail

SCENE="${1:?Usage: $0 <scene-name>}"
SCENE_DIR="/data/scenes/${SCENE}"
IMAGE_DIR="${SCENE_DIR}/images"
SPARSE_DIR="${SCENE_DIR}/sparse/0"
DENSE_DIR="${SCENE_DIR}/dense"

if [ ! -d "${SPARSE_DIR}" ]; then
    echo "ERROR: Sparse model not found: ${SPARSE_DIR}"
    echo "Run run_colmap_sparse.sh first."
    exit 1
fi

echo "=== COLMAP Dense Reconstruction (MVS) ==="
echo "Scene:  ${SCENE}"
echo "Sparse: ${SPARSE_DIR}"
echo "Output: ${DENSE_DIR}"
echo ""

mkdir -p "${DENSE_DIR}"

# Step 1: Undistort images using estimated camera parameters
echo "[1/3] Undistorting images..."
colmap image_undistorter \
    --image_path "${IMAGE_DIR}" \
    --input_path "${SPARSE_DIR}" \
    --output_path "${DENSE_DIR}" \
    --output_type COLMAP

# Step 2: Stereo (patch match) — computes dense depth maps
# This is the GPU-intensive step
echo "[2/3] Patch match stereo (dense depth maps)..."
colmap patch_match_stereo \
    --workspace_path "${DENSE_DIR}" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

# Step 3: Fuse depth maps into dense point cloud
echo "[3/3] Stereo fusion (fusing depth maps → point cloud)..."
colmap stereo_fusion \
    --workspace_path "${DENSE_DIR}" \
    --workspace_format COLMAP \
    --output_path "${DENSE_DIR}/fused.ply"

# Report results
echo ""
echo "=== Done ==="
if [ -f "${DENSE_DIR}/fused.ply" ]; then
    POINT_COUNT=$(python3 -c "
import struct
with open('${DENSE_DIR}/fused.ply', 'rb') as f:
    header = b''
    while True:
        line = f.readline()
        header += line
        if b'element vertex' in line:
            count = int(line.split()[-1])
            print(count)
            break
" 2>/dev/null || echo "unknown")
    FILE_SIZE=$(du -sh "${DENSE_DIR}/fused.ply" | cut -f1)
    echo "Dense point cloud: ${DENSE_DIR}/fused.ply"
    echo "Points: ${POINT_COUNT}"
    echo "Size:   ${FILE_SIZE}"
    echo ""
    echo "Next step: ./run_mesh_conditioning.sh ${SCENE}"
else
    echo "WARNING: No fused.ply produced. Check patch_match_stereo output."
fi

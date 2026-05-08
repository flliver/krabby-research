#!/bin/bash
# VGGT: Feed-forward 3D reconstruction (CVPR 2025 Best Paper)
# Input:  /data/scenes/<scene>/images/  (extracted JPEG frames)
# Output: /data/scenes/<scene>/vggt_output/  (COLMAP-format sparse + point cloud)
#
# Usage: ./run_vggt.sh <scene>
#   e.g. ./run_vggt.sh 001-patio-fisheye
#
# For 16 GB VRAM GPUs, uses reduced query params to fit in memory.

set -euo pipefail

SCENE="${1:?Usage: $0 <scene-name>}"
SCENE_DIR="/data/scenes/${SCENE}"
IMAGE_DIR="${SCENE_DIR}/images"
OUTPUT_DIR="${SCENE_DIR}/vggt_output"

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "ERROR: Image directory not found: ${IMAGE_DIR}"
    exit 1
fi

NUM_IMAGES=$(ls "${IMAGE_DIR}"/*.jpg 2>/dev/null | wc -l)
echo "=== VGGT Reconstruction ==="
echo "Scene:  ${SCENE}"
echo "Images: ${NUM_IMAGES}"
echo "Output: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

cd /opt/VGGT

echo "[1/2] Running VGGT reconstruction..."
time python demo_colmap.py \
    --scene_dir "${SCENE_DIR}" \
    --use_ba \
    --max_query_pts 2048 \
    --query_frame_num 5

echo ""
echo "[2/2] Exporting point cloud..."
# VGGT outputs to scene_dir/sparse/ in COLMAP format
if [ -d "${SCENE_DIR}/sparse" ]; then
    cp -r "${SCENE_DIR}/sparse" "${OUTPUT_DIR}/"

    # Also export as PLY for visualization
    python -c "
import numpy as np
import struct, os

sparse_dir = '${SCENE_DIR}/sparse'
output_ply = '${OUTPUT_DIR}/vggt_cloud.ply'

# Read COLMAP binary points3D
pts_file = os.path.join(sparse_dir, 'points3D.bin')
if os.path.exists(pts_file):
    with open(pts_file, 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        points = []
        colors = []
        for _ in range(num_points):
            pid = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<3d', f.read(24))
            rgb = struct.unpack('<3B', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            track_len = struct.unpack('<Q', f.read(8))[0]
            f.read(track_len * 8)  # skip track
            points.append(xyz)
            colors.append(rgb)

        points = np.array(points)
        colors = np.array(colors)

        with open(output_ply, 'w') as pf:
            pf.write('ply\nformat ascii 1.0\n')
            pf.write(f'element vertex {len(points)}\n')
            pf.write('property float x\nproperty float y\nproperty float z\n')
            pf.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
            pf.write('end_header\n')
            for p, c in zip(points, colors):
                pf.write(f'{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')

        print(f'Exported {len(points)} points to {output_ply}')
else:
    print('No points3D.bin found — check demo_colmap.py output')
"

    ls -lh "${OUTPUT_DIR}/"
else
    echo "WARNING: No sparse/ output found"
    ls -la "${SCENE_DIR}/"
fi

echo ""
echo "=== Done ==="

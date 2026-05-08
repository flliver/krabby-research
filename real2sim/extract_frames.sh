#!/bin/bash
# Extract frames from video for COLMAP input
# Input:  /data/videos/<video_file>
# Output: /data/scenes/<scene>/images/
#
# Usage: ./extract_frames.sh <video_file> <scene_name> [fps]
#   e.g. ./extract_frames.sh DJI_20260411135818_0010_D.MP4 scene-001 2

set -euo pipefail

VIDEO_FILE="${1:?Usage: $0 <video_file> <scene_name> [fps]}"
SCENE="${2:?Usage: $0 <video_file> <scene_name> [fps]}"
FPS="${3:-2}"

VIDEO_PATH="/data/videos/${VIDEO_FILE}"
IMAGE_DIR="/data/scenes/${SCENE}/images"

if [ ! -f "${VIDEO_PATH}" ]; then
    echo "ERROR: Video not found: ${VIDEO_PATH}"
    exit 1
fi

mkdir -p "${IMAGE_DIR}"

echo "=== Frame Extraction ==="
echo "Video:  ${VIDEO_PATH}"
echo "Output: ${IMAGE_DIR}"
echo "FPS:    ${FPS}"
echo ""

# Extract at specified fps, full resolution, highest quality JPEG
ffmpeg -i "${VIDEO_PATH}" \
    -vf "fps=${FPS}" \
    -q:v 1 \
    "${IMAGE_DIR}/frame_%04d.jpg" \
    -y

NUM_FRAMES=$(ls "${IMAGE_DIR}"/*.jpg | wc -l)
TOTAL_SIZE=$(du -sh "${IMAGE_DIR}" | cut -f1)

echo ""
echo "=== Done ==="
echo "Extracted ${NUM_FRAMES} frames (${TOTAL_SIZE})"
echo ""
echo "Next step: ./run_colmap_sparse.sh ${SCENE}"

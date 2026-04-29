#!/bin/bash
# Run MASt3R-SLAM on a video with live-streamed log output.
#
# Usage: run_mast3r.sh <scene-name> [video-name]
#   scene-name : name under /data/scenes/ (output dir)
#   video-name : video file under /data/videos/ (default: <scene-name>.mp4)
#
# Example:
#   run_mast3r.sh 004-sky-house-dining
#   run_mast3r.sh 004-sky-house-dining 004-sky-house-dining.mp4
#
# Tail the log live in another shell:
#   ssh jeremy@<outpost> "tail -f ~/outposts/krabby/data/011-scene-reconstruction/logs/mast3r-<scene>-run.log"

set -euo pipefail

SCENE="${1:?Usage: $0 <scene-name> [video-name]}"
VIDEO="${2:-${SCENE}.mp4}"

DATA_ROOT="$HOME/outposts/krabby/data/011-scene-reconstruction"
SCENE_DIR="$DATA_ROOT/scenes/$SCENE"
VIDEO_PATH="$DATA_ROOT/videos/$VIDEO"
LOG="$DATA_ROOT/logs/mast3r-${SCENE}-run.log"

if [ ! -f "$VIDEO_PATH" ]; then
    echo "ERROR: video not found: $VIDEO_PATH"
    exit 1
fi

mkdir -p "$SCENE_DIR/mast3r_output" "$DATA_ROOT/logs"

echo "=== MASt3R-SLAM run ==="
echo "Scene:  $SCENE"
echo "Video:  $VIDEO_PATH ($(ls -lh "$VIDEO_PATH" | awk '{print $5}'))"
echo "Output: $SCENE_DIR/mast3r_output/"
echo "Log:    $LOG"
echo
echo "Tail live with:"
echo "  ssh jeremy@$(hostname) \"tail -f $LOG\""
echo

# Key flags for live monitoring:
#   -e PYTHONUNBUFFERED=1      -- flush every print() immediately
#   --shm-size=8g              -- avoid Python multiprocessing deadlock
#   no | tail in the bash -c   -- write everything to log

sg docker -c "docker run --rm --gpus all --shm-size=8g \
  -e PYTHONUNBUFFERED=1 \
  -v $DATA_ROOT:/data \
  krabby-mast3r \
  bash -c '
    cd /opt/MASt3R-SLAM
    echo \"=== MASt3R-SLAM: $SCENE ===\"
    echo \"GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader)\"
    echo \"Started: \$(date)\"
    echo
    time python main.py \
      --dataset /data/videos/$VIDEO \
      --config config/base.yaml \
      --no-viz \
      --save-as /data/scenes/$SCENE/mast3r_output/${SCENE}
    echo
    echo \"=== Output ===\"
    ls -lh /data/scenes/$SCENE/mast3r_output/${SCENE}/ 2>/dev/null || echo \"No output dir\"
    echo \"Finished: \$(date)\"
  '" > "$LOG" 2>&1 &

PID=$!
echo "Background PID: $PID"
echo "Container PID:  $(sleep 1 && pgrep -f 'docker run.*krabby-mast3r' | tail -1)"

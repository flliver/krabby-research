#!/bin/bash
# Extract a candidate-pool of frames from scene 004 (sky-house-dining)
# evenly spaced across the 3:47 source video, at 1024×576 resolution.
#
# Usage: extract_candidates.sh <N>
#   N — number of frames to extract (default 500)
#
# Runs the krabby-mast3r container's ffmpeg, since the bbeeprz host has
# no native ffmpeg. The data dir is bind-mounted as /data inside.
#
# Source: ~/outposts/krabby/data/011-scene-reconstruction/videos/004-sky-house-dining.mp4
# Output: ~/outposts/krabby/data/011-scene-reconstruction/frames/004-sfm-scaling-<N>/

set -euo pipefail
N=${1:-500}
DATA_DIR=$HOME/outposts/krabby/data/011-scene-reconstruction
DEST=frames/004-sfm-scaling-$N
DURATION_SEC=227   # 3:47 source video, sky-house-dining

mkdir -p $DATA_DIR/$DEST
docker run --rm \
  -v $DATA_DIR:/data \
  krabby-mast3r:latest \
  ffmpeg -hide_banner -loglevel warning \
    -i /data/videos/004-sky-house-dining.mp4 \
    -vf "fps=$N/$DURATION_SEC,scale=1024:-2" -q:v 2 \
    /data/$DEST/frame_%04d.jpg

echo "Extracted $(ls $DATA_DIR/$DEST | wc -l) frames into $DATA_DIR/$DEST/"
echo "Disk: $(du -sh $DATA_DIR/$DEST | cut -f1)"

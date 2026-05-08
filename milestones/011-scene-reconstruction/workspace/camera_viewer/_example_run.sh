#!/bin/bash
# Example invocation. Adjust paths to where your cameras.json + frames live.
#
# If the cameras.json was produced on bbeeprz/tbeeprz, rsync these two trees
# down first:
#
#   rsync -av bbeeprz:/home/jeremy/outposts/krabby/data/011-scene-reconstruction/sfm-scaling-out/n060/mast3r_sfm/ ~/work/sfm-n060/
#   rsync -av bbeeprz:/home/jeremy/outposts/krabby/data/011-scene-reconstruction/frames/004-sfm-scaling-500/      ~/work/frames-500/
#
# (The cameras.json's filepaths point at /data/... inside the bbeeprz container,
# so we override with --frames to point at the local copy.)

set -euo pipefail
cd "$(dirname "$0")"

CAMERAS=${1:-~/work/sfm-n060/cameras.json}
FRAMES=${2:-~/work/frames-500}
OUT=${3:-./selected_frames.json}
PORT=${4:-8080}

# One-time setup:
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements.txt

# If you haven't sourced your venv:
[ -d .venv ] && source .venv/bin/activate

python viewer.py \
    --cameras "$CAMERAS" \
    --frames "$FRAMES" \
    --output "$OUT" \
    --port "$PORT"

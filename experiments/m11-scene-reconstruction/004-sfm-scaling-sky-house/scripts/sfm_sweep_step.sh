#!/bin/bash
# Usage: sfm-sweep-step.sh <N>
set -uo pipefail
N=$1
NPAD=$(printf "%03d" $N)
OUT=/data/sfm-scaling-out/n$NPAD
HOST_OUT=$HOME/outposts/krabby/data/011-scene-reconstruction/sfm-scaling-out/n$NPAD
LOG_VRAM=/tmp/vram-n$NPAD.log
LOG_OUT=/tmp/run-n$NPAD.log

rm -rf $HOST_OUT $LOG_VRAM $LOG_OUT 2>/dev/null

# Background VRAM poller (1 Hz)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > $LOG_VRAM &
POLLER=$!

T0=$(date +%s)
docker exec matcha-build bash -c "
  source /opt/matcha/bin/activate
  export PYTHONPATH=/opt/MAtCha:/opt/MAtCha/mast3r:/opt/MAtCha/mast3r/dust3r:/opt/MAtCha/2d-gaussian-splatting:/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd /opt/MAtCha
  python train.py \
    -s /data/frames/004-sfm-scaling-500 \
    -o $OUT \
    --sfm_only \
    --n_images $N \
    --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \
    --depthanything_encoder vitl 2>&1
" > $LOG_OUT 2>&1
RC=$?
T1=$(date +%s)
WALL=$((T1 - T0))

kill $POLLER 2>/dev/null
PEAK=$(awk '{ if ($1 > max) max = $1 } END { print max }' $LOG_VRAM)

# Count cameras correctly
N_CAMS="?"
if [ -f $HOST_OUT/mast3r_sfm/cameras.json ]; then
  N_CAMS=$(python3 - <<'PYEOF' "$HOST_OUT/mast3r_sfm/cameras.json"
import json, sys
d = json.load(open(sys.argv[1]))
if isinstance(d, dict) and "filepaths" in d:
    print(len(d["filepaths"]))
elif isinstance(d, list):
    print(len(d))
else:
    print(len(d.keys()))
PYEOF
)
fi

echo "RESULT N=$N RC=$RC WALL_SEC=$WALL PEAK_VRAM_MIB=$PEAK CAMS=$N_CAMS"
echo "---last 5 lines of MAtCha output---"
tail -5 $LOG_OUT

#!/bin/bash
# Run remaining N values sequentially. Bail on first failure.
set -uo pipefail
RESULTS=/tmp/sfm-sweep-results.tsv
echo -e "N\tRC\tWALL_SEC\tPEAK_VRAM_MIB\tCAMS" > $RESULTS

# N=24 and N=60 are already done by the time this runs; just record their results
for N in 24 60; do
  NPAD=$(printf "%03d" $N)
  HOST_OUT=$HOME/outposts/krabby/data/011-scene-reconstruction/sfm-scaling-out/n$NPAD
  if [ -f $HOST_OUT/mast3r_sfm/cameras.json ]; then
    CAMS=$(python3 -c "import json; d=json.load(open('$HOST_OUT/mast3r_sfm/cameras.json')); print(len(d['filepaths']))")
    PEAK=$(awk '{ if ($1 > max) max = $1 } END { print max }' /tmp/vram-n$NPAD.log 2>/dev/null || echo 0)
    # Best-effort wall reconstruction from RESULT line in old log
    WALL=$(grep -oE 'WALL_SEC=[0-9]+' /tmp/run-n$NPAD-summary.log 2>/dev/null | tail -1 | cut -d= -f2)
    [ -z "$WALL" ] && WALL="?"
    echo -e "$N\t0\t$WALL\t$PEAK\t$CAMS" >> $RESULTS
  fi
done

# Run remaining N values
for N in 120 200 300 500; do
  echo "=== N=$N ==="
  /tmp/sfm-sweep-step.sh $N 2>&1 | tail -8
  RC=$?
  NPAD=$(printf "%03d" $N)
  HOST_OUT=$HOME/outposts/krabby/data/011-scene-reconstruction/sfm-scaling-out/n$NPAD
  if [ -f $HOST_OUT/mast3r_sfm/cameras.json ]; then
    CAMS=$(python3 -c "import json; d=json.load(open('$HOST_OUT/mast3r_sfm/cameras.json')); print(len(d['filepaths']))")
  else
    CAMS="?"
    echo "[FAILED at N=$N — cameras.json missing] Stopping sweep."
    break
  fi
done

echo
echo "=== Final results table ==="
cat $RESULTS

#!/bin/bash
# Wait until N=300 completes (cameras.json exists), then kill the chain script
# to prevent the queued N=500 from running (we know it'll OOM).
N300_OUT=$HOME/outposts/krabby/data/011-scene-reconstruction/sfm-scaling-out/n300/mast3r_sfm/cameras.json
CHAIN_PID=720300

while [ ! -f $N300_OUT ]; do
  sleep 10
done
echo "N=300 cameras.json detected at $(date). Killing chain PID $CHAIN_PID to prevent N=500 OOM."
kill $CHAIN_PID 2>&1
sleep 3
ps -p $CHAIN_PID 2>&1 | tail -1

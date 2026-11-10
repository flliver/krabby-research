#!/bin/bash
# Serial runner for the stand re-pairing (STAND_REPAIR.md): `rsd_stand.py step S` for each S given, in order; stops at
# the first rc != 0, or before a step when the STOP_QUEUE file exists. Sentinels in the queue log:
# RSDSTQ BEGIN | START | END | STOP | DONE. Launch detached:
#   setsid nohup bash rsd_stand_queue.sh refs sn_train sn_rows sn_read ... > /dev/null 2>&1 < /dev/null &
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rod_speed_dr/stand
Q=$OUT/queue.log
mkdir -p "$OUT"
echo "RSDSTQ BEGIN $(date '+%F %T') pid=$$ steps: $*" >> "$Q"
for s in "$@"; do
  if [ -e "$OUT/STOP_QUEUE" ]; then echo "RSDSTQ STOP $s (STOP_QUEUE) $(date '+%F %T')" >> "$Q"; exit 9; fi
  echo "RSDSTQ START $s $(date '+%F %T')" >> "$Q"
  "$PY" -I -B "$HERE/rsd_stand.py" step "$s" >> "$Q" 2>&1
  rc=$?
  echo "RSDSTQ END $s rc=$rc $(date '+%F %T')" >> "$Q"
  if [ "$rc" != 0 ]; then echo "RSDSTQ STOP $s rc=$rc $(date '+%F %T')" >> "$Q"; exit "$rc"; fi
done
echo "RSDSTQ DONE $(date '+%F %T')" >> "$Q"

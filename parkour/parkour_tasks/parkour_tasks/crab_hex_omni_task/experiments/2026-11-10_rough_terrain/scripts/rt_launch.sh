#!/bin/bash
# Rough-terrain campaign: start rt_queue.py DETACHED (setsid nohup; survives the session) for one block.
#   bash rt_launch.sh BLOCK [rt_queue.py args...]        BLOCK = smoke | baseline | train | gate
#   e.g.  bash rt_launch.sh smoke
#         bash rt_launch.sh gate --new-ckpt /abs/.../model_34990.pt
# Log: <OUT>/logs/queue_<BLOCK>.log (an existing log is rotated to .<mtime>). Prints "RT_LAUNCH pid=... log=...".
# Watch it with rt_wait.sh LOG FROM_LINE (log sentinel, never pgrep). A dry pass is `python -I -B rt_queue.py
# --block BLOCK --dry` in the foreground (nothing launches), not this script.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
BLOCK=${1:?block: smoke|baseline|train|gate}; shift
case "$BLOCK" in smoke|baseline|train|gate) ;; *) echo "RT_LAUNCH refused: unknown block $BLOCK"; exit 2 ;; esac
for x in "$@"; do [ "$x" = --dry ] && { echo "RT_LAUNCH refused: --dry runs in the foreground"; exit 2; }; done
OUT=${RT_TP_OUT:-/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rough_terrain}
mkdir -p "$OUT/logs"
LOG=$OUT/logs/queue_$BLOCK.log
# Refuse BEFORE rotating while another queue holds the lock: rotating a live queue's log moves it out from under
# its watcher (2026-11-10 smoke relaunch). The lock is rt_queue.py's flock file, probed non-blocking here.
LOCK=$("$PY" -I -B -c "import sys; sys.path.insert(0, '$HERE'); import rt_machinery as M; print(M.LOCK_PATH)")
if [ -e "$LOCK" ] && ! flock -n "$LOCK" true; then
  echo "RT_LAUNCH refused: another queue holds $LOCK (its log is still $LOG); nothing rotated"; exit 3
fi
[ -e "$LOG" ] && mv "$LOG" "$LOG.$(stat -c %Y "$LOG")"
setsid nohup env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" -I -B "$HERE/rt_queue.py" --block "$BLOCK" "$@" \
  > "$LOG" 2>&1 < /dev/null &
echo "RT_LAUNCH pid=$! block=$BLOCK log=$LOG args=$*"

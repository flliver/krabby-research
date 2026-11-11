#!/bin/bash
# Short arm (ARM.md, Amendment A): start rt_short.py DETACHED (setsid nohup; survives the session) for one block.
#   bash rt_short_launch.sh BLOCK [rt_short.py args...]        BLOCK = strain | ssel | sconf1 | sconf2
#   e.g.  bash rt_short_launch.sh strain
#         bash rt_short_launch.sh ssel
# Same shape as scripts/rt_launch.sh: the lock probe BEFORE rotating the log, setsid nohup, env -u PYTHONPATH.
# Log: <OUT>/logs/queue_<BLOCK>.log (an existing log is rotated to .<mtime>). Prints "RT_LAUNCH pid=... log=...".
# Watch it with scripts/rt_wait.sh LOG FROM_LINE (log sentinel, never pgrep). A dry pass is `python -I -B rt_short.py
# --block BLOCK --dry` in the foreground (nothing launches), not this script (--dry is refused here). sconf1 / sconf2
# refuse inside rt_short.py unless results/short_arm/selection.json is committed (--test-pick is dry-only).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCR=/home/nickmagus/krabby/krabby-research/.claude/worktrees/rough-terrain/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-11-10_rough_terrain/scripts
PY=/home/nickmagus/krabby/isaac_venv/bin/python
BLOCK=${1:?block: strain|ssel|sconf1|sconf2}; shift
case "$BLOCK" in strain|ssel|sconf1|sconf2) ;; *) echo "RT_LAUNCH refused: unknown block $BLOCK"; exit 2 ;; esac
for x in "$@"; do
  case "$x" in
    --dry|--test-pick|--test-pick=*) echo "RT_LAUNCH refused: $x is a foreground dry-pass option (python -I -B rt_short.py --block $BLOCK --dry ...)"; exit 2 ;;
  esac
done
OUT=${RT_TP_OUT:-/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rough_terrain}
mkdir -p "$OUT/logs"
LOG=$OUT/logs/queue_$BLOCK.log
# Refuse BEFORE rotating while another queue holds the lock: rotating a live queue's log moves it out from under
# its watcher (2026-11-10 smoke relaunch). The lock is rt_queue.py's flock file, probed non-blocking here.
LOCK=$("$PY" -I -B -c "import sys; sys.path.insert(0, '$SCR'); import rt_machinery as M; print(M.LOCK_PATH)")
if [ -e "$LOCK" ] && ! flock -n "$LOCK" true; then
  echo "RT_LAUNCH refused: another queue holds $LOCK (its log is still $LOG); nothing rotated"; exit 3
fi
[ -e "$LOG" ] && mv "$LOG" "$LOG.$(stat -c %Y "$LOG")"
setsid nohup env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" -I -B "$HERE/rt_short.py" --block "$BLOCK" "$@" \
  > "$LOG" 2>&1 < /dev/null &
echo "RT_LAUNCH pid=$! block=$BLOCK log=$LOG args=$*"

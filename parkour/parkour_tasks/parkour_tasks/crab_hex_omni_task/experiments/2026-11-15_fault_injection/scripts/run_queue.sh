#!/bin/bash
# Fault-injection sweep: start fault_queue.py DETACHED for one phase (setsid nohup; survives the session), and a
# one-shot log-sentinel watcher. NEVER pgrep / ps in the watcher: the log is the only signal.
#
#   bash run_queue.sh PHASE [fault_queue.py run args...]   PHASE = smoke|nominals|es1|es2|kind|c1|closing
#       e.g. bash run_queue.sh smoke
#            bash run_queue.sh nominals --extra L000:C
#            bash run_queue.sh es2 --es2-cells S1_2,S1_perm,...
#       Log: $FS_TP_OUT/logs/queue_<PHASE>.log (default TP_OUT = MAIN parkour/outputs/fault_sweep; an existing log
#       is rotated to .<mtime>). Prints "FSQ_LAUNCH pid=... phase=... log=...".
#   bash run_queue.sh --wait LOG [MAX_HOURS]               block until the log's sentinel line
#       "QUEUE END status=<ok|stop> phase=<p> ..." (the runner's last line, also after a crash; if the runner dies
#       without one, this wrapper appends "QUEUE END status=stop ... (runner exited rc=N with no sentinel)").
#       Prints "FSQ_WAIT <the QUEUE END line>"; exit 0 on status=ok, 1 on status=stop, 2 on MAX_HOURS (default 8)
#       with the last QROW line (a silent hang is noticed). Poll 30 s.
# A dry pass runs in the foreground, not here:  python -I -B fault_queue.py run --phase PHASE --dry-run [--dry-gate]
# The runner composes no row env itself: gate_row.sh strips every parent KRABBY_* and sets PYTHONPATH; the runner is
# started with PYTHONPATH unset anyway.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAMP=$(cd "$HERE/.." && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=${FS_TP_OUT:-/home/nickmagus/krabby/krabby-research/parkour/outputs/fault_sweep}
RES=${FS_RESULTS:-$CAMP/results}             # FS_RESULTS: a scratch results dir (tests only)
LOCK=$RES/.queue.lock
POLL_S=30

if [ "${1:-}" = --wait ]; then
  LOG=${2:?usage: run_queue.sh --wait LOG [MAX_HOURS]}; MAXH=${3:-8}
  case "$MAXH" in ''|*[!0-9]*) echo "FSQ_WAIT bad MAX_HOURS '$MAXH'"; exit 2 ;; esac
  end=$(( $(date +%s) + MAXH * 3600 ))
  while :; do
    if [ -f "$LOG" ]; then
      hit=$(grep -E '^QUEUE END status=' "$LOG" | tail -n 1)
      if [ -n "$hit" ]; then
        echo "FSQ_WAIT $hit"
        case "$hit" in "QUEUE END status=ok "*|"QUEUE END status=ok") exit 0 ;; *) exit 1 ;; esac
      fi
    fi
    if [ "$(date +%s)" -ge "$end" ]; then
      echo "FSQ_WAIT TIMEOUT after ${MAXH} h; last: $(grep -E '^QROW ' "$LOG" 2>/dev/null | tail -n 1)"; exit 2
    fi
    sleep "$POLL_S"
  done
fi

PHASE=${1:?usage: run_queue.sh PHASE [args] | --wait LOG [MAX_HOURS]}; shift
case "$PHASE" in smoke|nominals|es1|es2|kind|c1|closing) ;; *) echo "FSQ_LAUNCH refused: unknown phase $PHASE"; exit 2 ;; esac
for x in "$@"; do
  case "$x" in
    --dry-run|--dry-gate) echo "FSQ_LAUNCH refused: $x runs in the foreground"; exit 2 ;;
    --launcher|--launcher=*|--fault-read|--fault-read=*|--results|--results=*)
      [ "${FS_QUEUE_TEST:-0}" = 1 ] || { echo "FSQ_LAUNCH refused: $x is a test option (FS_QUEUE_TEST=1)"; exit 2; } ;;
  esac
done
[ -f "$RES/STOP.json" ] && { echo "FSQ_LAUNCH refused: $RES/STOP.json exists (clear it by hand)"; exit 1; }
mkdir -p "$OUT/logs" "$RES"
LOG=$OUT/logs/queue_$PHASE.log
# Refuse BEFORE rotating while another runner holds the lock (rotating a live log moves it out from under its watcher).
if [ -e "$LOCK" ] && ! flock -n "$LOCK" true; then
  echo "FSQ_LAUNCH refused: another queue holds $LOCK; nothing rotated"; exit 3
fi
[ -e "$LOG" ] && mv "$LOG" "$LOG.$(stat -c %Y "$LOG")"
# The inner bash guarantees a sentinel even if python dies by a signal before writing its own.
setsid nohup env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 bash -c '
  py=$1; q=$2; log=$3; phase=$4; out=$5; shift 5
  "$py" -I -B "$q" run --phase "$phase" --tp-out "$out" "$@" >> "$log" 2>&1 < /dev/null
  rc=$?
  tail -n 1 "$log" | grep -q "^QUEUE END status=" || echo "QUEUE END status=stop phase=$phase (runner exited rc=$rc with no sentinel)" >> "$log"
' fsq "$PY" "$HERE/fault_queue.py" "$LOG" "$PHASE" "$OUT" ${FS_RESULTS:+--results "$RES"} "$@" > /dev/null 2>&1 < /dev/null &
echo "FSQ_LAUNCH pid=$! phase=$PHASE log=$LOG args=$*"
echo "  watch: bash $HERE/run_queue.sh --wait $LOG"

#!/bin/bash
# P2 steps 7-8 (p2/P2.md): ONE detached chain for the es9+ student read -- the front rows, the 84 rows, the CPU
# near-miss read, the one-sided relaunch rows and the CPU verdict (p2/p2_read.py, called, never re-implemented).
# One Isaac process at a time (p2_queue.sh's conventions, which this follows step for step).
#
#   bash p2_read_queue.sh [--from STEP]     start the chain DETACHED (setsid nohup); prints P2RQ_LAUNCH
#   bash p2_read_queue.sh --dry [--from STEP]   print the chain and run p2_read.py --dry (CPU). Nothing launches,
#                                           nothing is written, NO lock is probed (the lock files are not opened)
#   STEP = front rows nearmiss relaunch verdict
#
# The chain (each step's output -> <OUT>/logs/p2rq/<step>.log; its key lines are copied into the queue log):
#   front     p2_read.py --block front      GPU, 5 rows: the C1S es9 rows of ledge_up, pits, rubble, hills, gravel
#                                           (Left 22). A failed front row (rc 1: a role or env refusal) STOPS the
#                                           chain: "P2_READ STOP front ..." (notify and ask; 79 rows not burnt)
#   rows      p2_read.py --block rows       GPU, the 84 rows (the front 5 skip, complete at the pin). A failed row
#                                           is counted and the chain goes on (the near-miss read says INCOMPLETE)
#   nearmiss  p2_read.py --near-miss        CPU -> results/p2/read_near_miss.json; rc 0 NONE, 10 DUE, 11 STOP (over
#                                           the 16-row cap: notify and ask), 12 INCOMPLETE, 13 HALT (a same-row pair
#                                           beyond a class signature: stops P2)
#   relaunch  p2_read.py --block relaunch   GPU, only after nearmiss rc 10: the _R1 rows it marks due (<= 16), BEFORE
#                                           the stand starts (Left 18)
#   verdict   p2_read.py --verdict          CPU -> results/p2/read.json; always runs (also after STOP / HALT /
#                                           INCOMPLETE, so the numbers exist); rc 0 computed, 12, 13, 14 PENDING
# The chain stops only on a STOP_QUEUE file (results/p2/STOP_QUEUE or results/STOP_QUEUE, read between steps), a
# preflight refusal, a front-row failure, a GPU block ending rc 20 (REFUSED: no row launched) / 21 (HALT: DDC growth
# or guard stop), or a guard breach before an Isaac step. It never starts the stand (step 9 is the parent's).
#
# Preflight (CPU, before ANY step, also on a --from restart; any refusal -> "P2RQ FAIL chain (preflight: ...)"):
# results/p2/pin.json exists; build_check.py check --pin-path it; p2_read.py --inputs-check --block rows (the pick
# from results/p2/sel.json committed and clean, every policy by sha, frozen files, no foreign es9+ stamp). The DDC
# is reported: above P2RQ_DDC_WARN_GB (70) a "P2RQ WARN" line (P2.md Left 14: moving it aside between selection and
# the read is the user's yes; never inside the read); above P2RQ_DDC_MAX_GB (78) the guard stops before an Isaac step.
#
# Sentinels in <OUT>/read_queue.log (column 0; everything else is indented or stays in the step's own log):
#   P2RQ START chain ... | P2RQ START <step> | P2RQ DONE <step> rc=N | P2RQ FAIL <step> rc=N (...) | P2RQ SKIP <step>
#   P2RQ DONE chain fail=N | P2RQ FAIL chain (...)          -- the last line of every chain
#   P2RQ PREFLIGHT OK|BAD ... | P2RQ WARN ...
#   P2_READ VERDICT ... / P2_READ NEARMISS ... / P2_READ STOP ... / P2_READ HALT ... / P2_READ REFUSED ... /
#   P2_INPUTS ... / BUILD_CHECK ...                         copied from the steps (the verdict line is notified at
#                                                           once by the parent, Left 17)
# Watch it with a log-sentinel watcher on "^P2RQ (DONE|FAIL) chain"; never pgrep.
#
# Guards: the chain lock <OUT>/p2_read_queue.lock (one read chain; probed only by a real launch, BEFORE the log is
# rotated); before every Isaac step: wait while the campaign's .isaac.lock is held (the distill / another queue:
# flock probe) or any Isaac python runs (rt_machinery.isaac_patterns, `ps -ww -eo args`, never pgrep -f), then DDC
# <= P2RQ_DDC_MAX_GB (78) and free disk >= P2RQ_FREE_MIN_GB (30); rt_queue repeats the busy wait and the guards before
# every row (RTQ_DDC_MAX_GB / RTQ_FREE_MIN_GB passed down). The chain runs with PYTHONPATH unset.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAMP=$(cd "$HERE/.." && pwd)
SCR=$CAMP/scripts
W=$(cd "$CAMP/../../../../../.." && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rough_terrain_p2
QLOG=$OUT/read_queue.log
SLOG=$OUT/logs/p2rq
READ=$HERE/p2_read.py
P2PIN=$CAMP/results/p2/pin.json
CHAIN_LOCK=$OUT/p2_read_queue.lock
ISAAC_LOCK=$CAMP/.isaac.lock
STOPS=("$CAMP/results/p2/STOP_QUEUE" "$CAMP/results/STOP_QUEUE")
DDC=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache
DDC_MAX_GB=${P2RQ_DDC_MAX_GB:-78}
DDC_WARN_GB=${P2RQ_DDC_WARN_GB:-70}
FREE_MIN_GB=${P2RQ_FREE_MIN_GB:-30}
STEPS=(front rows nearmiss relaunch verdict)
POLL_S=30

cmd_of() {
  case "$1" in
    front) echo "$PY -I -B $READ --block front" ;;
    rows) echo "$PY -I -B $READ --block rows" ;;
    nearmiss) echo "$PY -I -B $READ --near-miss" ;;
    relaunch) echo "$PY -I -B $READ --block relaunch" ;;
    verdict) echo "$PY -I -B $READ --verdict" ;;
  esac
}
when_of() {
  case "$1" in
    front) echo "always; a failed row STOPS the chain (P2_READ STOP front); rc 20 / 21 ends it" ;;
    rows) echo "after front rc 0; rc 20 / 21 ends the chain; a failed row is counted, the chain goes on" ;;
    nearmiss) echo "always after rows (CPU)" ;;
    relaunch) echo "only if nearmiss rc=10 (DUE); rc 20 / 21 ends the chain" ;;
    verdict) echo "always (CPU; also after STOP / HALT / INCOMPLETE)" ;;
  esac
}
is_isaac() { case "$1" in front|rows|relaunch) return 0 ;; *) return 1 ;; esac; }

isaac_pats() {
  env -u PYTHONPATH "$PY" -I -B -c "import sys; sys.path.insert(0, '$SCR'); import rt_machinery as M; i, n = M.isaac_patterns(); print(i); print(n)"
}
busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
isaac_lock_held() { [ -e "$ISAAC_LOCK" ] && ! flock -n "$ISAAC_LOCK" true; }
stop_file() { local f; for f in "${STOPS[@]}"; do [ -e "$f" ] && { echo "$f"; return 0; }; done; return 1; }
ddc_gb() { local g; g=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); echo "${g:-0}"; }
free_gb() {
  local p=$1
  while [ ! -e "$p" ] && [ "$p" != / ]; do p=$(dirname "$p"); done
  df -BG --output=avail "$p" 2>/dev/null | tail -1 | tr -dc '0-9'
}
guard_problems() {
  local g f p
  g=$(ddc_gb)
  [ "$g" -le "${DDC_MAX_GB%%.*}" ] || echo "Kit DerivedDataCache ${g} GB > ${DDC_MAX_GB} GB (moving it aside is the user's call)"
  for p in "$OUT" "$W"; do
    f=$(free_gb "$p"); f=${f:-0}
    [ "$f" -ge "${FREE_MIN_GB%%.*}" ] || echo "free disk ${f} GB < ${FREE_MIN_GB} GB at $p"
  done
}

# ------------------------------------------------------------------------------------------------- the dry pass
dry() {  # CPU only: no lock file is opened, no du, no process wait
  local from=$1 on=0 s
  echo "# P2 read chain (dry): nothing launches, nothing is written, no lock is probed"
  echo "# queue log $QLOG ; step logs $SLOG/<step>.log ; chain lock $CHAIN_LOCK ; campaign lock $ISAAC_LOCK"
  echo "# STOP files: ${STOPS[*]}"
  echo "# guards before each Isaac step: DDC <= $DDC_MAX_GB GB (warn > $DDC_WARN_GB), free >= $FREE_MIN_GB GB (not measured in a dry pass)"
  echo "  preflight    CPU    test -f $P2PIN; build_check.py check --pin-path $P2PIN; p2_read.py --inputs-check --block rows"
  [ -z "$from" ] && on=1
  for s in "${STEPS[@]}"; do
    [ "$s" = "$from" ] && on=1
    if [ "$on" = 0 ]; then echo "  [--from] skip $s"; continue; fi
    printf '  %-10s %-6s %s\n' "$s" "$(is_isaac "$s" && echo GPU || echo CPU)" "$(cmd_of "$s")"
    printf '  %-10s        when: %s\n' "" "$(when_of "$s")"
  done
  echo "#"; echo "# ===== p2_read.py --dry (slot plan and counts)"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$PY" -I -B "$READ" --dry 2>&1 \
    | grep -E '^(#|P2_READ)' | sed 's/^/    /'
  echo "    (p2_read.py --dry rc=${PIPESTATUS[0]})"
  echo "# end of dry pass"
}

# ------------------------------------------------------------------------------------------------- the chain
say() { echo "$*" | tee -a "$QLOG" >/dev/null; }
copy_keys() {  # LOGFILE FROM_LINE
  tail -n +"$2" "$1" | grep -E '^(P2_READ|P2_INPUTS (OK|BAD|WARN|INFO)|BUILD_CHECK)' | cut -c1-2000 >> "$QLOG"
  tail -n +"$2" "$1" | grep -E '^(RTQ_START|RTQ_DONE|RTQ [^ ]+ rc=[1-9]|RTQ [^ ]+ rc=0 \(stamp|  TPROW END)' \
    | sed 's/^/    /' >> "$QLOG"
}
run_cmd() {  # STEP CMD -> rc
  local s=$1 c=$2 log=$SLOG/$1.log n rc
  mkdir -p "$SLOG"
  [ -f "$log" ] || : > "$log"
  n=$(( $(wc -l < "$log") + 1 ))
  echo "### $(date -Iseconds) $c" >> "$log"
  # shellcheck disable=SC2086
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 RTQ_DDC_MAX_GB="$DDC_MAX_GB" RTQ_FREE_MIN_GB="$FREE_MIN_GB" $c >> "$log" 2>&1 < /dev/null
  rc=$?
  copy_keys "$log" "$n"
  return "$rc"
}
preflight() {  # -> PF_BAD / PF_WHY
  local c rc g
  PF_BAD=0; PF_WHY=""
  if [ ! -f "$P2PIN" ]; then PF_BAD=1; PF_WHY="no $P2PIN"; fi
  for c in "$PY -I -B $SCR/build_check.py check --pin-path $P2PIN" "$PY -I -B $READ --inputs-check --block rows"; do
    run_cmd preflight "$c"; rc=$?
    if [ "$rc" != 0 ]; then PF_BAD=1; PF_WHY="${PF_WHY:+$PF_WHY; }${c#"$PY -I -B "} rc=$rc"; fi
  done
  g=$(ddc_gb)
  if [ "$g" -gt "${DDC_WARN_GB%%.*}" ]; then
    say "P2RQ WARN Kit DerivedDataCache ${g} GB > ${DDC_WARN_GB} GB (P2.md Left 14: move it aside before the read -- the user's yes; the guard stops at ${DDC_MAX_GB})"
  fi
  say "P2RQ PREFLIGHT $([ "$PF_BAD" = 0 ] && echo OK || echo "BAD -- $PF_WHY") ddc=${g}GB"
}
wait_isaac_free() {
  local s=$1 waited=0 gp
  while isaac_lock_held || busy; do
    if [ $((waited % 600)) = 0 ]; then
      say "  # [$s] waiting: $(isaac_lock_held && echo "$ISAAC_LOCK held")$(busy && echo " an Isaac python runs: $(ps -ww -eo pid=,args= | grep -E "$ISAAC_PAT" | grep -vE "$NONKIT_PAT" | head -1 | cut -c1-160)")"
    fi
    sleep "$POLL_S"; waited=$((waited + POLL_S))
  done
  gp=$(guard_problems)
  if [ -n "$gp" ]; then say "P2RQ FAIL $s rc=5 (guard: $(echo "$gp" | tr '\n' ';'))"; return 5; fi
  return 0
}

chain() {
  local from=$1 on=0 s rc fail=0 nrc=-1 sf
  exec 9>>"$CHAIN_LOCK"
  flock -n 9 || { say "P2RQ FAIL chain (another read chain holds $CHAIN_LOCK)"; exit 3; }
  say "P2RQ START chain from=${from:-front} pid=$$ at=$(date -Iseconds) out=$OUT ddc_max=$DDC_MAX_GB free_min=$FREE_MIN_GB"
  [ -f "$READ" ] || { say "P2RQ FAIL chain (no $READ)"; exit 2; }
  preflight
  if [ "$PF_BAD" != 0 ]; then say "P2RQ FAIL chain (preflight: $PF_WHY; no step ran)"; exit 1; fi
  [ -z "$from" ] && on=1
  case "$from" in relaunch) nrc=10 ;; esac       # a --from relaunch restart: p2_read.py replans from a FRESH record
  for s in "${STEPS[@]}"; do
    [ "$s" = "$from" ] && on=1
    [ "$on" = 1 ] || continue
    if sf=$(stop_file); then say "P2RQ FAIL chain (stopped by $sf before $s)"; exit 1; fi
    if [ "$s" = relaunch ] && [ "$nrc" != 10 ]; then say "P2RQ SKIP relaunch (nearmiss rc=$nrc: no relaunch due)"; continue; fi
    if is_isaac "$s"; then
      wait_isaac_free "$s" || { say "P2RQ FAIL chain (guard before $s)"; exit 1; }
    fi
    say "P2RQ START $s at=$(date -Iseconds) cmd=$(cmd_of "$s")"
    run_cmd "$s" "$(cmd_of "$s")"; rc=$?
    if is_isaac "$s" && { [ "$rc" = 20 ] || [ "$rc" = 21 ]; }; then
      say "P2RQ FAIL $s rc=$rc ($([ "$rc" = 20 ] && echo "REFUSED: no row launched" || echo "HALT: DDC-growth or guard stop"); log $SLOG/$s.log)"
      say "P2RQ FAIL chain ($s rc=$rc)"; exit 1
    fi
    case "$s" in
      front)
        if [ "$rc" != 0 ]; then
          say "P2_READ STOP front rc=$rc (a front row failed: a role / env refusal; the 79 other rows were not launched; notify and ask; log $SLOG/front.log)"
          say "P2RQ FAIL chain (front rc=$rc)"; exit 1
        fi
        say "P2RQ DONE front rc=0 at=$(date -Iseconds)" ;;
      nearmiss)
        nrc=$rc
        case "$rc" in
          0|10|11|12|13) say "P2RQ DONE nearmiss rc=$rc ($(case $rc in 0) echo NONE;; 10) echo DUE;; 11) echo "STOP: notify and ask";; 12) echo INCOMPLETE;; 13) echo "HALT: stops P2";; esac))" ;;
          *) say "P2RQ FAIL nearmiss rc=$rc (log $SLOG/nearmiss.log)"; fail=$((fail + 1)) ;;
        esac ;;
      verdict)
        case "$rc" in
          0|12|13|14) say "P2RQ DONE verdict rc=$rc ($(case $rc in 0) echo COMPUTED;; 12) echo INCOMPLETE;; 13) echo HALT;; 14) echo PENDING;; esac))" ;;
          *) say "P2RQ FAIL verdict rc=$rc (log $SLOG/verdict.log)"; fail=$((fail + 1)) ;;
        esac ;;
      *)
        if [ "$rc" = 0 ]; then say "P2RQ DONE $s rc=0 at=$(date -Iseconds)"
        else say "P2RQ FAIL $s rc=$rc (log $SLOG/$s.log; the chain goes on)"; fail=$((fail + 1)); fi ;;
    esac
  done
  say "P2RQ DONE chain fail=$fail at=$(date -Iseconds)"
}

# ------------------------------------------------------------------------------------------------- entry
DRY=0; FROM=""; CHAIN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry) DRY=1; shift ;;
    --from) FROM=${2:-}; shift 2 ;;
    --chain) CHAIN=1; shift ;;
    *) echo "P2RQ_LAUNCH refused: unknown option $1"; exit 2 ;;
  esac
done
if [ -n "$FROM" ]; then
  ok=0; for s in "${STEPS[@]}"; do [ "$s" = "$FROM" ] && ok=1; done
  [ "$ok" = 1 ] || { echo "P2RQ_LAUNCH refused: --from $FROM is not one of ${STEPS[*]}"; exit 2; }
fi
if [ "$DRY" = 1 ]; then dry "$FROM"; exit 0; fi

PATS=$(isaac_pats) || { echo "P2RQ_LAUNCH refused: cannot read the Isaac patterns from rt_machinery"; exit 2; }
ISAAC_PAT=$(echo "$PATS" | sed -n 1p); NONKIT_PAT=$(echo "$PATS" | sed -n 2p)
[ -n "$ISAAC_PAT" ] && [ -n "$NONKIT_PAT" ] || { echo "P2RQ_LAUNCH refused: empty Isaac patterns"; exit 2; }
if [ "$CHAIN" = 1 ]; then chain "$FROM"; exit 0; fi

mkdir -p "$OUT" "$SLOG"
if [ -e "$CHAIN_LOCK" ] && ! flock -n "$CHAIN_LOCK" true; then
  echo "P2RQ_LAUNCH refused: another read chain holds $CHAIN_LOCK (its log is still $QLOG); nothing rotated"; exit 3
fi
[ -e "$QLOG" ] && mv "$QLOG" "$QLOG.$(stat -c %Y "$QLOG")"
setsid nohup env -u PYTHONPATH bash "$HERE/p2_read_queue.sh" --chain ${FROM:+--from "$FROM"} \
  > "$OUT/p2_read_queue.nohup.out" 2>&1 < /dev/null &
echo "P2RQ_LAUNCH pid=$! log=$QLOG from=${FROM:-front}"

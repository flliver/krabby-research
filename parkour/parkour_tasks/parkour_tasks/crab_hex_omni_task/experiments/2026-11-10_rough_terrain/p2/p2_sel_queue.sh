#!/bin/bash
# P2 step 6 (p2/P2.md): ONE detached chain -- wait for the distill RT_ST_s19, then c1's teacher rows -> the prescreen ->
# the rank -> full block(s) -> the decision, then the ONNX export of the pick. Every GPU step is p2_select.py (the RT
# row machinery: rt_queue.main, gate_row.sh, the results/p2/pin.json check); nothing here launches Kit itself. One Isaac
# process at a time. The chain decides nothing of its own beyond the exit codes p2_select.py pre-registers.
#
#   bash p2_sel_queue.sh [--from STEP]          start the chain DETACHED (setsid nohup); prints P2SQ_LAUNCH
#   bash p2_sel_queue.sh --dry [--deep]         print the chain (commands, guards, the state now). Nothing launches,
#                                               nothing is written, NO lock is probed or created; --deep also runs
#                                               p2_select.py --dry (every block) and p2_onnx.py --dry.
#   STEP = wait_distill teacher prescreen rank full1 decide1 full2 decide2 onnx
#
# The chain (each step's output -> <OUT>/logs/p2sq/<step>.log; its key lines are copied into the queue log):
#   wait_distill CPU  until "TPDISTILL END RT_ST_s19 rc=N" in <OUT>/distill/distill.log (poll 60 s); N != 0 -> FAIL
#                     chain (two distill misses = HALT is the parent's read); then until the step-3/distill chain's
#                     lock <OUT>/p2_queue.lock is free (that chain ends right after the distill)
#   teacher      GPU  p2_select.py --block teacher     c1's 8 teacher rows (suite@1/2 reused by proof, side22 not run)
#   prescreen    GPU  p2_select.py --block prescreen   5 rows x 5 candidates
#   rank         CPU  p2_select.py --rank              rc 0 -> on; else (a prescreen row missing / invalid) FAIL chain
#   full1        GPU  p2_select.py --block full        the block next_action names (11 + 4 at 90 kg + 4 WATCH)
#   decide1      CPU  p2_select.py --decide            -> results/p2/sel.json; rc 0 PICK -> onnx; 10 NEXT -> full2
#                     (NEXT naming the block just run = one of its rows failed -> FAIL chain); 11 HALT -> chain ends
#                     (P2SQ HALT, no ONNX: notify and ask); 12 INCOMPLETE / 13 STOP -> FAIL chain
#   full2        GPU  p2_select.py --block full        only after decide1 rc 10
#   decide2      CPU  p2_select.py --decide            0 PICK -> onnx; 11 HALT -> chain ends; else FAIL chain
#   onnx         CPU  p2_onnx.py --from-sel            the pick's ONNX pair, parity <= 1e-4, io == c31000's ->
#                                                      results/p2/onnx_PICK.json (CPU only, CUDA hidden)
# A failed row (a block rc 1) never stops the chain by itself: the next CPU read reports what is missing. The chain
# stops on a STOP_QUEUE file (results/p2/STOP_QUEUE or results/STOP_QUEUE, read between steps), a block ending rc 20
# (REFUSED: no row launched) / 21 (HALT: rt_queue's DDC-growth or guard stop), a guard breach before a GPU step, or the
# read verdicts above.
#
# Sentinels in <OUT>/sel_queue.log (column 0):
#   P2SQ START chain ... | P2SQ START <step> | P2SQ DONE <step> rc=N | P2SQ FAIL <step> rc=N (...) | P2SQ SKIP <step> (...)
#   P2SQ PICK <cand> sha256=... | P2SQ HALT (...) | P2SQ DONE chain fail=N | P2SQ FAIL chain (...)  -- the last line
#   P2_SEL ... / P2_SEL_RANK ... / P2_SEL_WATCH ... / P2_SEL_PLAN ... / P2_SEL_INPUTS (OK|BAD|WARN) / P2_ONNX END ...
# Watch it with a log-sentinel watcher on "^P2SQ (DONE|FAIL) chain|^P2SQ HALT|^P2SQ PICK"; never pgrep.
#
# Guards (p2_queue.sh's conventions): the chain lock <OUT>/p2_sel_queue.lock (one selection chain), probed BEFORE the
# queue log is rotated; before every GPU step: wait while the campaign's .isaac.lock is held (flock probe) or any Isaac
# python runs (rt_machinery.isaac_patterns: distill.sh's anchored pattern + rt_kit_*.py; `ps -ww -eo args`, never
# pgrep -f), then Kit DerivedDataCache <= P2SQ_DDC_MAX_GB (78) and free disk >= P2SQ_FREE_MIN_GB (30) on the outputs and
# the worktree, else FAIL + stop (moving the cache aside is the user's call; PLAN item 14: check it before the read).
# rt_queue repeats the busy wait and the guards before every row (RTQ_DDC_MAX_GB / RTQ_FREE_MIN_GB passed down). The
# chain runs with PYTHONPATH unset (gate_row.sh sets this worktree's PYTHONPATH for its Kit child; drivers run -I).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAMP=$(cd "$HERE/.." && pwd)
SCR=$CAMP/scripts
W=$(cd "$CAMP/../../../../../.." && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rough_terrain_p2
QLOG=$OUT/sel_queue.log
SLOG=$OUT/logs/p2sq
SEL=$HERE/p2_select.py
ONNX=$HERE/p2_onnx.py
DISTILL_LOG=$OUT/distill/distill.log
RUN_NAME=RT_ST_s19
P2_CHAIN_LOCK=$OUT/p2_queue.lock
CHAIN_LOCK=$OUT/p2_sel_queue.lock
ISAAC_LOCK=$CAMP/.isaac.lock
STOPS=("$CAMP/results/p2/STOP_QUEUE" "$CAMP/results/STOP_QUEUE")
DDC=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache
DDC_MAX_GB=${P2SQ_DDC_MAX_GB:-78}
FREE_MIN_GB=${P2SQ_FREE_MIN_GB:-30}
STEPS=(wait_distill teacher prescreen rank full1 decide1 full2 decide2 onnx)
POLL_S=30
DISTILL_POLL_S=60

cmd_of() {
  case "$1" in
    wait_distill) echo "(wait) grep 'TPDISTILL END $RUN_NAME rc=' $DISTILL_LOG; flock -n $P2_CHAIN_LOCK" ;;
    teacher) echo "$PY -I -B $SEL --block teacher" ;;
    prescreen) echo "$PY -I -B $SEL --block prescreen" ;;
    rank) echo "$PY -I -B $SEL --rank" ;;
    full1|full2) echo "$PY -I -B $SEL --block full" ;;
    decide1|decide2) echo "$PY -I -B $SEL --decide" ;;
    onnx) echo "$PY -I -B $ONNX --from-sel" ;;
  esac
}
is_isaac() { case "$1" in teacher|prescreen|full1|full2) return 0 ;; *) return 1 ;; esac; }
when_of() {
  case "$1" in
    wait_distill) echo "always; END rc != 0 -> FAIL chain" ;;
    teacher|prescreen) echo "always; rc 20 / 21 ends the chain, rc 1 (a failed row) goes on" ;;
    rank) echo "always; rc != 0 -> FAIL chain" ;;
    full1) echo "after rank rc 0; rc 20 / 21 ends the chain" ;;
    decide1) echo "after full1; 0 PICK -> onnx, 10 NEXT (another cand) -> full2, 11 HALT -> end, else FAIL" ;;
    full2) echo "only after decide1 rc 10 (NEXT, another candidate)" ;;
    decide2) echo "only after full2; 0 PICK -> onnx, 11 HALT -> end, else FAIL" ;;
    onnx) echo "only after a PICK" ;;
  esac
}
step_idx() { local i=0 s; for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && { echo "$i"; return 0; }; i=$((i + 1)); done; echo 0; }

isaac_pats() {
  env -u PYTHONPATH "$PY" -I -B -c "import sys; sys.path.insert(0, '$SCR'); import rt_machinery as M; i, n = M.isaac_patterns(); print(i); print(n)"
}
busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
busy_line() { ps -ww -eo args= | grep -E "$ISAAC_PAT" | grep -vE "$NONKIT_PAT" | head -1 | cut -c1-160; }   # args only: the pattern is ^-anchored
lock_held() { [ -e "$1" ] && ! flock -n "$1" true; }   # RUN MODE ONLY (the dry pass never probes a lock)
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
distill_rc() { grep -oE "TPDISTILL END $RUN_NAME rc=-?[0-9]+" "$DISTILL_LOG" 2>/dev/null | tail -1 | sed 's/.*rc=//'; }

# ------------------------------------------------------------------------------------------------- the dry pass
dry() {
  local deep=$1 from=$2 on=0 s gp sf rc
  echo "# P2 selection chain (dry): nothing launches, nothing is written, no lock is probed"
  echo "# queue log $QLOG ; step logs $SLOG/<step>.log ; chain lock $CHAIN_LOCK ; waits on $P2_CHAIN_LOCK and $ISAAC_LOCK"
  echo "# STOP files: ${STOPS[*]}"
  echo "# guards before each GPU step: DDC <= $DDC_MAX_GB GB (now $(ddc_gb) GB), free >= $FREE_MIN_GB GB (now $(free_gb "$OUT") GB at $OUT, $(free_gb "$W") GB at $W)"
  gp=$(guard_problems); [ -z "$gp" ] && echo "# guards now: OK" || echo "# guards now: WOULD STOP -- $gp"
  if busy; then echo "# Isaac busy now: YES (the chain would wait): $(busy_line)"; else echo "# Isaac busy now: no"; fi
  if sf=$(stop_file); then echo "# STOP file present: $sf (a launch would stop before its first step)"; fi
  rc=$(distill_rc); echo "# distill $RUN_NAME: $([ -n "$rc" ] && echo "ended rc=$rc" || echo "running (no END line yet; wait_distill would wait)")"
  for f in "$SEL" "$ONNX"; do [ -f "$f" ] && echo "# script $f (present)" || echo "# script $f MISSING"; done
  echo "#"
  [ -z "$from" ] && on=1
  for s in "${STEPS[@]}"; do
    [ "$s" = "$from" ] && on=1
    if [ "$on" = 0 ]; then echo "  [--from] skip $s"; continue; fi
    printf '  %-13s %-4s %s\n' "$s" "$(is_isaac "$s" && echo GPU || echo CPU)" "$(cmd_of "$s")"
    printf '  %-13s      when: %s\n' "" "$(when_of "$s")"
  done
  if [ "$deep" = 1 ]; then
    echo "#"; echo "# ===== deep: p2_select.py --dry (teacher, prescreen, full; KRABBY_ env lines dropped)"
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" -I -B "$SEL" --dry 2>&1 | grep -E '^(# P2|#   |RTQ|P2_SEL|BUILD_CHECK|TPROW)' | cut -c1-240 | sed 's/^/    /'
    echo "#"; echo "# ===== deep: p2_select.py --decide --dry"
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" -I -B "$SEL" --decide --dry 2>&1 | cut -c1-240 | sed 's/^/    /'
    echo "#"; echo "# ===== deep: p2_onnx.py --dry"
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PY" -I -B "$ONNX" --dry 2>&1 | cut -c1-240 | sed 's/^/    /'
  fi
  echo "# end of dry pass"
}

# ------------------------------------------------------------------------------------------------- the chain
say() { echo "$*" | tee -a "$QLOG" >/dev/null; }
copy_keys() {  # LOGFILE FROM_LINE
  tail -n +"$2" "$1" | grep -E '^(P2_SEL|P2_ONNX END|BUILD_CHECK (OK|BAD))' | cut -c1-2000 >> "$QLOG"
  tail -n +"$2" "$1" | grep -E '^(RTQ_START|RTQ_DONE|RTQ [^ ]+ rc=[1-9]|RTQ [^ ]+ rc=0 \(stamp|  TPROW END)' | sed 's/^/    /' >> "$QLOG"
}
LAST_LOG=""; LAST_FROM=1
run_step() {  # STEP -> the command's rc
  local s=$1 c log n rc
  c=$(cmd_of "$s"); log=$SLOG/$s.log
  mkdir -p "$SLOG"; [ -f "$log" ] || : > "$log"
  n=$(( $(wc -l < "$log") + 1 ))
  echo "### $(date -Iseconds) $c" >> "$log"
  # shellcheck disable=SC2086
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 RTQ_DDC_MAX_GB="$DDC_MAX_GB" RTQ_FREE_MIN_GB="$FREE_MIN_GB" $c >> "$log" 2>&1 < /dev/null
  rc=$?
  copy_keys "$log" "$n"
  LAST_LOG=$log; LAST_FROM=$n
  return "$rc"
}
last_line() { tail -n +"$LAST_FROM" "$LAST_LOG" | grep -E "$1" | tail -1; }
wait_isaac_free() {
  local s=$1 waited=0 gp
  while lock_held "$ISAAC_LOCK" || busy; do
    if [ $((waited % 600)) = 0 ]; then
      say "  # [$s] waiting: $(lock_held "$ISAAC_LOCK" && echo "$ISAAC_LOCK held")$(busy && echo " an Isaac python runs: $(busy_line)")"
    fi
    sleep "$POLL_S"; waited=$((waited + POLL_S))
  done
  gp=$(guard_problems)
  if [ -n "$gp" ]; then say "P2SQ FAIL $s rc=5 (guard: $(echo "$gp" | tr '\n' ';'))"; return 5; fi
  return 0
}
wait_distill() {
  local rc waited=0
  until rc=$(distill_rc) && [ -n "$rc" ]; do
    if [ $((waited % 3600)) = 0 ]; then say "  # [wait_distill] waiting for 'TPDISTILL END $RUN_NAME' in $DISTILL_LOG"; fi
    if stop_file >/dev/null; then say "P2SQ FAIL chain (stopped by $(stop_file) while waiting for the distill)"; exit 1; fi
    sleep "$DISTILL_POLL_S"; waited=$((waited + DISTILL_POLL_S))
  done
  [ "$rc" = 0 ] || { say "P2SQ FAIL wait_distill rc=$rc (distill.sh END $RUN_NAME rc=$rc)"; say "P2SQ FAIL chain (the distill ended rc=$rc; no selection row ran)"; exit 1; }
  waited=0
  while lock_held "$P2_CHAIN_LOCK"; do
    if [ $((waited % 600)) = 0 ]; then say "  # [wait_distill] waiting for the step-3/distill chain to release $P2_CHAIN_LOCK"; fi
    sleep "$POLL_S"; waited=$((waited + POLL_S))
  done
  return 0
}
plan_cand() { last_line '^P2_SEL_PLAN full cand=' | sed -E 's/^P2_SEL_PLAN full cand=([^ ]+).*/\1/'; }
next_cand() { last_line '^P2_SEL NEXT cand=' | sed -E 's/^P2_SEL NEXT cand=([^ ]+).*/\1/'; }

chain() {
  local from=$1 on=0 s rc fail=0 full_due=0 full2_due=0 picked=0 c1="" c2="" sf
  exec 9>>"$CHAIN_LOCK"
  flock -n 9 || { say "P2SQ FAIL chain (another selection chain holds $CHAIN_LOCK)"; exit 3; }
  say "P2SQ START chain from=${from:-wait_distill} pid=$$ at=$(date -Iseconds) out=$OUT ddc_max=$DDC_MAX_GB free_min=$FREE_MIN_GB"
  for f in "$SEL" "$ONNX"; do [ -f "$f" ] || { say "P2SQ FAIL chain (no $f)"; exit 2; }; done
  [ -z "$from" ] && on=1
  case "$from" in full1) full_due=1 ;; full2) full2_due=1 ;; decide2) full2_due=1 ;; onnx) picked=1 ;; esac
  for s in "${STEPS[@]}"; do
    [ "$s" = "$from" ] && on=1
    [ "$on" = 1 ] || continue
    if sf=$(stop_file); then say "P2SQ FAIL chain (stopped by $sf before $s)"; exit 1; fi
    case "$s" in
      full1) [ "$full_due" = 1 ] || { say "P2SQ SKIP $s (rank did not pass)"; continue; } ;;
      decide1) : ;;
      full2) [ "$full2_due" = 1 ] || { say "P2SQ SKIP $s (decide1 named no second block)"; continue; } ;;
      decide2) [ "$full2_due" = 1 ] || { say "P2SQ SKIP $s (full2 did not run)"; continue; } ;;
      onnx) [ "$picked" = 1 ] || { say "P2SQ SKIP $s (no pick)"; continue; } ;;
    esac
    if [ "$s" = wait_distill ]; then
      say "P2SQ START $s at=$(date -Iseconds)"; wait_distill; say "P2SQ DONE $s rc=0 at=$(date -Iseconds) (distill END rc=0)"; continue
    fi
    if is_isaac "$s"; then wait_isaac_free "$s" || { say "P2SQ FAIL chain (guard before $s)"; exit 1; }; fi
    say "P2SQ START $s at=$(date -Iseconds) cmd=$(cmd_of "$s")"
    run_step "$s"; rc=$?
    case "$s" in
      teacher|prescreen|full1|full2)
        if [ "$rc" = 20 ] || [ "$rc" = 21 ]; then
          say "P2SQ FAIL $s rc=$rc ($([ "$rc" = 20 ] && echo "REFUSED: no row launched" || echo "HALT: DDC-growth or guard stop"); log $SLOG/$s.log)"
          say "P2SQ FAIL chain ($s rc=$rc)"; exit 1
        fi
        [ "$s" = full1 ] && c1=$(plan_cand)
        [ "$s" = full2 ] && c2=$(plan_cand)
        if [ "$rc" = 0 ]; then say "P2SQ DONE $s rc=0 at=$(date -Iseconds)"; else say "P2SQ FAIL $s rc=$rc (a row failed; the chain goes on; log $SLOG/$s.log)"; fail=$((fail + 1)); fi ;;
      rank)
        if [ "$rc" = 0 ]; then say "P2SQ DONE rank rc=0 ($(last_line '^P2_SEL_RANK ranked=' | cut -c1-300))"; full_due=1
        else say "P2SQ FAIL rank rc=$rc ($(last_line '^P2_SEL_RANK' | cut -c1-300))"; say "P2SQ FAIL chain (rank rc=$rc: a prescreen row is missing or invalid; notify)"; exit 1; fi ;;
      decide1|decide2)
        case "$rc" in
          0) picked=1; full2_due=0
             say "P2SQ PICK $(last_line '^P2_SEL PICK' | cut -c13-400)"; say "P2SQ DONE $s rc=0 (PICK)" ;;
          10) n=$(next_cand)
              if [ "$s" = decide2 ] || { [ -n "$c1" ] && [ "$n" = "$c1" ]; }; then
                say "P2SQ FAIL $s rc=10 (NEXT names $n again: a row of its block failed; notify)"; say "P2SQ FAIL chain ($s: block rows incomplete)"; exit 1
              fi
              full2_due=1; say "P2SQ DONE $s rc=10 (NEXT cand=$n: block 1 failed; block 2 of <= 2)" ;;
          11) say "P2SQ HALT ($s: $(last_line '^P2_SEL HALT' | cut -c1-400); no ONNX; notify and ask)"; say "P2SQ DONE chain fail=$fail at=$(date -Iseconds) (HALT)"; exit 0 ;;
          *) say "P2SQ FAIL $s rc=$rc ($(last_line '^P2_SEL (INCOMPLETE|STOP)' | cut -c1-400))"; say "P2SQ FAIL chain ($s rc=$rc; notify)"; exit 1 ;;
        esac ;;
      onnx)
        if [ "$rc" = 0 ]; then say "P2SQ DONE onnx rc=0 ($(last_line '^P2_ONNX END' | cut -c1-300))"
        else say "P2SQ FAIL onnx rc=$rc ($(last_line '^P2_ONNX' | cut -c1-300))"; fail=$((fail + 1)); fi ;;
    esac
  done
  say "P2SQ DONE chain fail=$fail at=$(date -Iseconds)"
}

# ------------------------------------------------------------------------------------------------- entry
DRY=0; DEEP=0; FROM=""; CHAIN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry) DRY=1; shift ;;
    --deep) DEEP=1; shift ;;
    --from) FROM=${2:-}; shift 2 ;;
    --chain) CHAIN=1; shift ;;
    *) echo "P2SQ_LAUNCH refused: unknown option $1"; exit 2 ;;
  esac
done
if [ -n "$FROM" ]; then
  ok=0; for s in "${STEPS[@]}"; do [ "$s" = "$FROM" ] && ok=1; done
  [ "$ok" = 1 ] || { echo "P2SQ_LAUNCH refused: --from $FROM is not one of ${STEPS[*]}"; exit 2; }
fi
PATS=$(isaac_pats) || { echo "P2SQ_LAUNCH refused: cannot read the Isaac patterns from rt_machinery"; exit 2; }
ISAAC_PAT=$(echo "$PATS" | sed -n 1p); NONKIT_PAT=$(echo "$PATS" | sed -n 2p)
[ -n "$ISAAC_PAT" ] && [ -n "$NONKIT_PAT" ] || { echo "P2SQ_LAUNCH refused: empty Isaac patterns"; exit 2; }

if [ "$DRY" = 1 ]; then dry "$DEEP" "$FROM"; exit 0; fi
if [ "$CHAIN" = 1 ]; then chain "$FROM"; exit 0; fi

mkdir -p "$OUT" "$SLOG"
if lock_held "$CHAIN_LOCK"; then
  echo "P2SQ_LAUNCH refused: another selection chain holds $CHAIN_LOCK (its log is still $QLOG); nothing rotated"; exit 3
fi
[ -e "$QLOG" ] && mv "$QLOG" "$QLOG.$(stat -c %Y "$QLOG")"
setsid nohup env -u PYTHONPATH bash "$HERE/p2_sel_queue.sh" --chain ${FROM:+--from "$FROM"} \
  > "$OUT/p2_sel_queue.nohup.out" 2>&1 < /dev/null &
echo "P2SQ_LAUNCH pid=$! log=$QLOG from=${FROM:-wait_distill}"

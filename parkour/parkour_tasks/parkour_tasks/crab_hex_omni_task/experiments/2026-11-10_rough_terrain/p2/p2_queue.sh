#!/bin/bash
# P2 steps 3 -> 4 (p2/P2.md): ONE detached chain -- the step-3 blocks (smoke, d-prime, bridge + its relaunches, the
# CPU verdicts) and then the distill (p2/p2_distill.sh, called, never re-implemented). One Isaac process at a time.
#
#   bash p2_queue.sh [--from STEP] [--distill-args "ARGS"]   start the chain DETACHED (setsid nohup); prints P2Q_LAUNCH
#   bash p2_queue.sh --dry [--deep] [--from STEP]            print the chain (every command, the guards, the locks);
#                                                            --deep also runs every block's own dry pass and
#                                                            p2_distill.sh --dry-run. Nothing launches, nothing written.
#   STEP = smoke smoke_check dprime bridge judge0 bridge_r1 judge1 bridge_r2 judge2 read distill
#
# The chain (each step's output -> <OUT>/logs/p2q/<step>.log; its key lines are copied into the queue log):
#   smoke       p2_step3.py --block smoke                 GPU, 4 rows (es90, 4 episodes, two launches)
#   smoke_check p2_step3.py --check-smoke                 CPU; FAIL -> dprime / bridge / judges are SKIPPED (their
#                                                         rows would fail the same way) and the chain goes on
#   dprime      p2_step3.py --block dprime                GPU, 4 rows
#   bridge      p2_step3.py --block bridge                GPU, 4 rows
#   judgeK      p2_step3.py --judge-bridge                CPU -> results/p2/bridge_judge.json; rc 0 PASS, 10 RELAUNCH,
#                                                         11 STOP, 12 INCOMPLETE
#   bridge_rK   p2_step3.py --block bridge --relaunch K   GPU, only after judge(K-1) rc 10 (only the rows it marks due)
#   read        p2_step3.py --read                        CPU -> results/p2/step3.json (smoke, d', trigger, bridge)
#   distill     flock -n <campaign>/.isaac.lock bash p2_distill.sh ARGS   GPU, 10-15 h (default ARGS
#               "--name RT_ST_s19 --checkpoint <c1 model_29500> --wait-idle"; --distill-args or P2Q_DISTILL_ARGS
#               replace them)
# A bridge STOP, a d-prime TRIGGER / CANNOT_GAIN, a smoke FAIL or a failed row NEVER stop the chain: the distill runs
# (never idle the GPU; notify-and-ask is the parent's job -- it reads the sentinels below). The chain stops only on a
# STOP_QUEUE file (results/p2/STOP_QUEUE or results/STOP_QUEUE, read between steps), a preflight refusal, a step-3
# block ending rc 20 / 21 (below), a guard breach before an Isaac step, or a missing script.
#
# Preflight (CPU, before ANY step, also on a --from restart; any refusal -> "P2Q FAIL chain (preflight: ...)" and exit:
# no step runs and the distill does NOT start): results/p2/pin.json exists (PLAN step 2); build_check.py check
# --pin-path it; p2_step3.py --inputs-check --block X for every step-3 block the chain still runs (smoke/dprime/bridge).
# A step-3 block that ends rc 20 (REFUSED: non-zero with no row launched -- a stop_on_fail check, a plan refusal, the
# .isaac.lock refusal, the c31000 identity refusal) or rc 21 (HALT: rt_queue's DDC-growth / guard stop) also ends the
# chain before the distill ("P2Q FAIL chain (<step> rc=20|21 ...)"); a block whose launched row failed (rc 1) is
# counted and the chain goes on.
#
# Sentinels in <OUT>/queue.log (column 0; everything else a step prints is indented or stays in its own log):
#   P2Q START chain ... | P2Q START <step> | P2Q DONE <step> rc=0 | P2Q FAIL <step> rc=N (...) | P2Q SKIP <step> (...)
#   P2Q DONE chain fail=N | P2Q FAIL chain (...)          -- the last line of every chain
#   P2Q PREFLIGHT OK|BAD ... and the preflight's BUILD_CHECK ... / P2_INPUTS ... lines
#   P2_SMOKE ... / P2_DPRIME ... / P2_BRIDGE ... / P2_STEP3 ... / P2_INPUTS BAD ...   copied from the CPU steps
# Watch it with a log-sentinel watcher (e.g. scripts/rt_wait.sh-style tail on "^P2Q (DONE|FAIL) chain"); never pgrep.
#
# Guards (rt_short_launch.sh / rt_kit_run.sh / rt_queue.py conventions): the chain lock <OUT>/p2_queue.lock (one P2
# chain), probed BEFORE the queue log is rotated; before every Isaac step: wait while the campaign's .isaac.lock is held
# by another queue (flock probe) or any Isaac python runs (rt_machinery.isaac_patterns: distill.sh's anchored pattern
# + rt_kit_*.py, `ps -ww -eo args`, never pgrep -f), then Kit DerivedDataCache <= P2Q_DDC_MAX_GB (78) and free disk
# >= P2Q_FREE_MIN_GB (30) on the outputs and the worktree, else P2Q FAIL + stop (moving the cache aside is the user's
# call). rt_queue repeats the busy wait and the DDC / disk guard before every row (RTQ_DDC_MAX_GB / RTQ_FREE_MIN_GB =
# the same values, passed down). The chain runs with PYTHONPATH unset (gate_row.sh and p2_distill.sh each set this
# worktree's PYTHONPATH for their Kit child; the drivers run python -I).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAMP=$(cd "$HERE/.." && pwd)
SCR=$CAMP/scripts
W=$(cd "$CAMP/../../../../../.." && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rough_terrain_p2
QLOG=$OUT/queue.log
SLOG=$OUT/logs/p2q
STEP3=$HERE/p2_step3.py
P2PIN=$CAMP/results/p2/pin.json
DISTILL=$HERE/p2_distill.sh
CHAIN_LOCK=$OUT/p2_queue.lock
ISAAC_LOCK=$CAMP/.isaac.lock
STOPS=("$CAMP/results/p2/STOP_QUEUE" "$CAMP/results/STOP_QUEUE")
DDC=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache
DDC_MAX_GB=${P2Q_DDC_MAX_GB:-78}
FREE_MIN_GB=${P2Q_FREE_MIN_GB:-30}
C1_CK=$W/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19/model_29500.pt
C1_SHA=6d1f29be8bf4504fd53fd20018dbab38ff60c6ccb32d7bd885f3ab15eaaed3c1
DISTILL_ARGS=${P2Q_DISTILL_ARGS:-"--name RT_ST_s19 --checkpoint $C1_CK --wait-idle"}
STEPS=(smoke smoke_check dprime bridge judge0 bridge_r1 judge1 bridge_r2 judge2 read distill)
POLL_S=30

cmd_of() {  # STEP -> the command (one line, words split by the caller)
  case "$1" in
    smoke) echo "$PY -I -B $STEP3 --block smoke" ;;
    smoke_check) echo "$PY -I -B $STEP3 --check-smoke" ;;
    dprime) echo "$PY -I -B $STEP3 --block dprime" ;;
    bridge) echo "$PY -I -B $STEP3 --block bridge" ;;
    judge0|judge1|judge2) echo "$PY -I -B $STEP3 --judge-bridge" ;;
    bridge_r1) echo "$PY -I -B $STEP3 --block bridge --relaunch 1" ;;
    bridge_r2) echo "$PY -I -B $STEP3 --block bridge --relaunch 2" ;;
    read) echo "$PY -I -B $STEP3 --read" ;;
    distill) echo "flock -n $ISAAC_LOCK bash $DISTILL $DISTILL_ARGS" ;;
  esac
}
step_idx() { local i=0 s; for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && { echo "$i"; return 0; }; i=$((i + 1)); done; echo 0; }
preflight_cmds() {  # FROM -> the preflight's python commands (after the pin-file test), one per line
  local f; f=$(step_idx "${1:-smoke}")
  if [ -f "$P2PIN" ]; then echo "$PY -I -B $SCR/build_check.py check --pin-path $P2PIN"; fi
  if [ "$f" -le "$(step_idx smoke)" ]; then echo "$PY -I -B $STEP3 --inputs-check --block smoke"; fi
  if [ "$f" -le "$(step_idx dprime)" ]; then echo "$PY -I -B $STEP3 --inputs-check --block dprime"; fi
  if [ "$f" -le "$(step_idx bridge_r2)" ]; then echo "$PY -I -B $STEP3 --inputs-check --block bridge"; fi
  return 0
}
is_isaac() { case "$1" in smoke|dprime|bridge|bridge_r1|bridge_r2|distill) return 0 ;; *) return 1 ;; esac; }
when_of() {
  case "$1" in
    smoke) echo "always; rc 20 (REFUSED, no row launched) / 21 (DDC-growth or guard HALT) ends the chain" ;;
    dprime|bridge) echo "skipped if smoke_check failed; rc 20 / 21 ends the chain" ;;
    judge0) echo "skipped if smoke_check failed" ;;
    bridge_r1) echo "only if judge0 rc=10 (RELAUNCH); plans only the rows judge0 marks due at R1; rc 20 / 21 ends the chain" ;;
    judge1) echo "only if bridge_r1 ran" ;;
    bridge_r2) echo "only if judge1 rc=10 (RELAUNCH); plans only the rows judge1 marks due at R2; rc 20 / 21 ends the chain" ;;
    judge2) echo "only if bridge_r2 ran" ;;
    read) echo "always (records whatever exists)" ;;
    distill) echo "always (a bridge STOP / d' trigger / smoke FAIL never stops it)" ;;
    *) echo "always" ;;
  esac
}

isaac_pats() {  # ISAAC_PAT and NONKIT_PAT, one per line, from rt_machinery (distill.sh's, + rt_kit_*.py)
  env -u PYTHONPATH "$PY" -I -B -c "import sys; sys.path.insert(0, '$SCR'); import rt_machinery as M; i, n = M.isaac_patterns(); print(i); print(n)"
}
busy() {  # 0 while any Isaac python runs (anchored ps -ww -eo args; never pgrep -f)
  ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"
}
isaac_lock_held() { [ -e "$ISAAC_LOCK" ] && ! flock -n "$ISAAC_LOCK" true; }
stop_file() { local f; for f in "${STOPS[@]}"; do [ -e "$f" ] && { echo "$f"; return 0; }; done; return 1; }
ddc_gb() { local g; g=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); echo "${g:-0}"; }
free_gb() {  # PATH (its nearest existing ancestor: the output root may not exist before the first launch)
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
dry() {
  local deep=$1 from=$2 on=0 s
  echo "# P2 chain (dry): nothing launches, nothing is written"
  echo "# queue log $QLOG ; step logs $SLOG/<step>.log ; chain lock $CHAIN_LOCK ; campaign lock $ISAAC_LOCK"
  echo "# STOP files: ${STOPS[*]}"
  echo "# guards before each Isaac step: DDC <= $DDC_MAX_GB GB (now $(ddc_gb) GB), free >= $FREE_MIN_GB GB (now $(free_gb "$OUT") GB at $OUT, $(free_gb "$W") GB at $W)"
  local gp; gp=$(guard_problems); [ -z "$gp" ] && echo "# guards now: OK" || echo "# guards now: WOULD STOP -- $gp"
  if [ -e "$CHAIN_LOCK" ] && ! flock -n "$CHAIN_LOCK" true; then echo "# chain lock: HELD (a P2 chain runs; a launch would be refused)"; else echo "# chain lock: free"; fi
  if isaac_lock_held; then echo "# campaign .isaac.lock: HELD (the chain would wait)"; else echo "# campaign .isaac.lock: free"; fi
  if busy; then echo "# Isaac busy: YES (the chain would wait): $(ps -ww -eo pid=,args= | grep -E "$ISAAC_PAT" | grep -vE "$NONKIT_PAT" | head -2)"; else echo "# Isaac busy: no"; fi
  local sf; if sf=$(stop_file); then echo "# STOP file present: $sf (a launch would stop before its first step)"; fi
  if [ -f "$C1_CK" ]; then
    local sha; sha=$(sha256sum "$C1_CK" | cut -d' ' -f1)
    [ "$sha" = "$C1_SHA" ] && echo "# c1 $C1_CK sha256 ${sha:0:8} == pre-registered" || echo "# c1 $C1_CK sha256 ${sha:0:8} != pre-registered ${C1_SHA:0:8} (p2_distill.sh refuses)"
  else echo "# c1 checkpoint MISSING: $C1_CK"; fi
  [ -f "$DISTILL" ] && echo "# distill script: $DISTILL (present)" || echo "# distill script: $DISTILL MISSING (the distill step would FAIL)"
  echo "#"
  echo "  preflight    CPU    before any step (also on --from); any refusal ends the chain before the distill"
  echo "                      test -f $P2PIN"
  preflight_cmds "$from" | sed "s|^$PY -I -B |                      |"
  preflight "$from" dry
  echo "                      preflight now: $([ "$PF_BAD" = 0 ] && echo "OK (a launch would proceed)" || echo "WOULD REFUSE -- $PF_WHY")"
  [ -z "$from" ] && on=1
  for s in "${STEPS[@]}"; do
    [ "$s" = "$from" ] && on=1
    if [ "$on" = 0 ]; then echo "  [--from] skip $s"; continue; fi
    printf '  %-12s %-6s %s\n' "$s" "$(is_isaac "$s" && echo GPU || echo CPU)" "$(cmd_of "$s")"
    printf '  %-12s        when: %s\n' "" "$(when_of "$s")"
  done
  if [ "$deep" = 1 ]; then
    for b in "smoke" "dprime" "bridge" "bridge --relaunch 1" "bridge --relaunch 2"; do
      echo "#"; echo "# ===== deep: p2_step3.py --block $b --dry"
      # shellcheck disable=SC2086
      env -u PYTHONPATH "$PY" -I -B "$STEP3" --block $b --dry 2>&1 | grep -vE '^KRABBY_' | sed 's/^/    /'
    done
    echo "#"; echo "# ===== deep: p2_distill.sh $DISTILL_ARGS --dry-run"
    if [ -f "$DISTILL" ]; then
      # shellcheck disable=SC2086
      env -u PYTHONPATH bash "$DISTILL" $DISTILL_ARGS --dry-run 2>&1 | sed 's/^/    /'
      echo "    (p2_distill.sh --dry-run rc=${PIPESTATUS[0]})"
    else echo "    (absent)"; fi
  fi
  echo "# end of dry pass"
}

# ------------------------------------------------------------------------------------------------- the chain
say() { echo "$*" | tee -a "$QLOG" >/dev/null; }
copy_keys() {  # STEP LOGFILE FROM_LINE: sentinel lines at column 0, the queue's own lines indented
  tail -n +"$3" "$2" | grep -E '^(P2_SMOKE|P2_DPRIME|P2_BRIDGE|P2_STEP3|P2_INPUTS (OK|BAD|WARN)|BUILD_CHECK)' | cut -c1-2000 >> "$QLOG"
  tail -n +"$3" "$2" | grep -E '^(RTQ_START|RTQ_DONE|RTQ [^ ]+ rc=[1-9]|RTQ [^ ]+ rc=0 \(stamp|  TPROW END|TPDISTILL|p2_distill.sh:|# WOULD REFUSE)' \
    | sed 's/^/    /' >> "$QLOG"
}
run_step() { run_cmd "$1" "$(cmd_of "$1")"; }
run_cmd() {  # STEP CMD -> the command's rc (output -> <SLOG>/<STEP>.log; key lines -> the queue log)
  local s=$1 c=$2 log=$SLOG/$1.log n rc
  mkdir -p "$SLOG"
  [ -f "$log" ] || : > "$log"
  n=$(( $(wc -l < "$log") + 1 ))
  echo "### $(date -Iseconds) $c" >> "$log"
  # shellcheck disable=SC2086
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 RTQ_DDC_MAX_GB="$DDC_MAX_GB" RTQ_FREE_MIN_GB="$FREE_MIN_GB" $c >> "$log" 2>&1 < /dev/null
  rc=$?
  copy_keys "$s" "$log" "$n"
  return "$rc"
}
PF_BAD=0; PF_WHY=""
preflight() {  # FROM MODE(run|dry) -> sets PF_BAD (0 = every check passed) and PF_WHY; dry prints, run logs
  local from=$1 mode=$2 c rc out
  PF_BAD=0; PF_WHY=""
  if [ ! -f "$P2PIN" ]; then
    PF_BAD=1; PF_WHY="no $P2PIN (PLAN step 2: build_check.py pin --pin-path results/p2/pin.json on a clean tree)"
    if [ "$mode" = dry ]; then echo "                      BAD: $PF_WHY"; else say "P2Q PREFLIGHT BAD $PF_WHY"; fi
  fi
  while IFS= read -r c; do
    [ -n "$c" ] || continue
    if [ "$mode" = dry ]; then   # read-only: build_check check without --json; --inputs-check only prints
      # shellcheck disable=SC2086
      out=$(env -u PYTHONPATH -u P2_DRY PYTHONDONTWRITEBYTECODE=1 $c 2>&1 < /dev/null); rc=$?
      echo "$out" | grep -E '^(BUILD_CHECK|P2_INPUTS (OK|BAD|WARN))' | cut -c1-300 | sed 's/^/                      > /'
      echo "                      rc=$rc: ${c#"$PY -I -B "}"
    else
      run_cmd preflight "$c"; rc=$?
    fi
    if [ "$rc" != 0 ]; then PF_BAD=1; PF_WHY="${PF_WHY:+$PF_WHY; }${c#"$PY -I -B "} rc=$rc"; fi
  done < <(preflight_cmds "$from")
  if [ "$mode" != dry ]; then say "P2Q PREFLIGHT $([ "$PF_BAD" = 0 ] && echo OK || echo "BAD -- $PF_WHY")"; fi
  return 0
}
wait_isaac_free() {  # STEP: wait for the campaign lock and for no Isaac python; then the guards (non-zero = breach)
  local s=$1 waited=0 gp
  while isaac_lock_held || busy; do
    if [ $((waited % 600)) = 0 ]; then
      say "  # [$s] waiting: $(isaac_lock_held && echo "$ISAAC_LOCK held by another queue")$(busy && echo " an Isaac python runs: $(ps -ww -eo pid=,args= | grep -E "$ISAAC_PAT" | grep -vE "$NONKIT_PAT" | head -1 | cut -c1-160)")"
    fi
    sleep "$POLL_S"; waited=$((waited + POLL_S))
  done
  gp=$(guard_problems)
  if [ -n "$gp" ]; then say "P2Q FAIL $s rc=5 (guard: $(echo "$gp" | tr '\n' ';'))"; return 5; fi
  return 0
}

chain() {
  local from=$1 on=0 s rc fail=0 smoke_ok=1 jrc=-1 ran_r1=0 ran_r2=0 sf
  exec 9>>"$CHAIN_LOCK"
  flock -n 9 || { say "P2Q FAIL chain (another P2 chain holds $CHAIN_LOCK)"; exit 3; }
  say "P2Q START chain from=${from:-smoke} pid=$$ at=$(date -Iseconds) out=$OUT distill_args=\"$DISTILL_ARGS\" ddc_max=$DDC_MAX_GB free_min=$FREE_MIN_GB"
  [ -f "$STEP3" ] || { say "P2Q FAIL chain (no $STEP3)"; exit 2; }
  preflight "$from" run
  if [ "$PF_BAD" != 0 ]; then say "P2Q FAIL chain (preflight: $PF_WHY; no step ran, the distill did not start)"; exit 1; fi
  [ -z "$from" ] && on=1
  # a --from restart into the bridge tail: the relaunch blocks plan only from a FRESH results/p2/bridge_judge.json
  # (p2_step3.py refuses a stale one), so the chain lets them run and they launch exactly the rows still due
  case "$from" in
    bridge_r1) jrc=10 ;;
    judge1) ran_r1=1 ;;
    bridge_r2) ran_r1=1; jrc=10 ;;
    judge2) ran_r1=1; ran_r2=1 ;;
  esac
  for s in "${STEPS[@]}"; do
    [ "$s" = "$from" ] && on=1
    [ "$on" = 1 ] || continue
    if sf=$(stop_file); then say "P2Q FAIL chain (stopped by $sf before $s)"; exit 1; fi
    case "$s" in
      dprime|bridge|judge0) if [ "$smoke_ok" = 0 ]; then say "P2Q SKIP $s (smoke_check failed)"; continue; fi ;;
      bridge_r1) if [ "$jrc" != 10 ]; then say "P2Q SKIP $s (judge0 rc=$jrc: no relaunch due)"; continue; fi ;;
      judge1) if [ "$ran_r1" = 0 ]; then say "P2Q SKIP $s (bridge_r1 did not run)"; continue; fi ;;
      bridge_r2) if [ "$ran_r1" = 0 ] || [ "$jrc" != 10 ]; then say "P2Q SKIP $s (judge1 rc=$jrc / bridge_r1 ran=$ran_r1: no relaunch due)"; continue; fi ;;
      judge2) if [ "$ran_r2" = 0 ]; then say "P2Q SKIP $s (bridge_r2 did not run)"; continue; fi ;;
      distill) if [ ! -f "$DISTILL" ]; then say "P2Q FAIL distill rc=2 (no $DISTILL)"; fail=$((fail + 1)); continue; fi ;;
    esac
    if is_isaac "$s"; then
      wait_isaac_free "$s" || { say "P2Q FAIL chain (guard before $s)"; exit 1; }
    fi
    say "P2Q START $s at=$(date -Iseconds) cmd=$(cmd_of "$s")"
    run_step "$s"; rc=$?
    case "$s" in
      smoke|dprime|bridge|bridge_r1|bridge_r2)
        if [ "$rc" = 20 ] || [ "$rc" = 21 ]; then
          say "P2Q FAIL $s rc=$rc ($([ "$rc" = 20 ] && echo "REFUSED: no row launched" || echo "HALT: DDC-growth or guard stop"); log $SLOG/$s.log)"
          say "P2Q FAIL chain ($s rc=$rc; the distill did not start)"; exit 1
        fi ;;
    esac
    case "$s" in
      smoke_check) [ "$rc" = 0 ] || smoke_ok=0 ;;
      judge0|judge1|judge2) jrc=$rc ;;
      bridge_r1) ran_r1=1 ;;
      bridge_r2) ran_r2=1 ;;
    esac
    case "$s" in
      judge0|judge1|judge2)
        case "$rc" in
          0|10|11|12) say "P2Q DONE $s rc=$rc ($(case $rc in 0) echo PASS;; 10) echo RELAUNCH;; 11) echo STOP;; 12) echo INCOMPLETE;; esac))" ;;
          *) say "P2Q FAIL $s rc=$rc (log $SLOG/$s.log)"; fail=$((fail + 1)) ;;
        esac ;;
      *)
        if [ "$rc" = 0 ]; then say "P2Q DONE $s rc=0 at=$(date -Iseconds)"
        else say "P2Q FAIL $s rc=$rc (log $SLOG/$s.log; the chain goes on)"; fail=$((fail + 1)); fi ;;
    esac
  done
  say "P2Q DONE chain fail=$fail at=$(date -Iseconds)"
}

# ------------------------------------------------------------------------------------------------- entry
DRY=0; DEEP=0; FROM=""; CHAIN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry) DRY=1; shift ;;
    --deep) DEEP=1; shift ;;
    --from) FROM=${2:-}; shift 2 ;;
    --distill-args) DISTILL_ARGS=${2:-}; shift 2 ;;
    --chain) CHAIN=1; shift ;;
    *) echo "P2Q_LAUNCH refused: unknown option $1"; exit 2 ;;
  esac
done
if [ -n "$FROM" ]; then
  ok=0; for s in "${STEPS[@]}"; do [ "$s" = "$FROM" ] && ok=1; done
  [ "$ok" = 1 ] || { echo "P2Q_LAUNCH refused: --from $FROM is not one of ${STEPS[*]}"; exit 2; }
fi
case "$DISTILL_ARGS" in *--dry-run*) echo "P2Q_LAUNCH refused: --dry-run in the distill args (use --dry)"; exit 2 ;; esac
PATS=$(isaac_pats) || { echo "P2Q_LAUNCH refused: cannot read the Isaac patterns from rt_machinery"; exit 2; }
ISAAC_PAT=$(echo "$PATS" | sed -n 1p); NONKIT_PAT=$(echo "$PATS" | sed -n 2p)
[ -n "$ISAAC_PAT" ] && [ -n "$NONKIT_PAT" ] || { echo "P2Q_LAUNCH refused: empty Isaac patterns"; exit 2; }

if [ "$DRY" = 1 ]; then dry "$DEEP" "$FROM"; exit 0; fi
if [ "$CHAIN" = 1 ]; then chain "$FROM"; exit 0; fi

# launcher: refuse BEFORE rotating the log while another chain holds the lock (rt_short_launch.sh's rule)
mkdir -p "$OUT" "$SLOG"
if [ -e "$CHAIN_LOCK" ] && ! flock -n "$CHAIN_LOCK" true; then
  echo "P2Q_LAUNCH refused: another P2 chain holds $CHAIN_LOCK (its log is still $QLOG); nothing rotated"; exit 3
fi
[ -e "$QLOG" ] && mv "$QLOG" "$QLOG.$(stat -c %Y "$QLOG")"
setsid nohup env -u PYTHONPATH bash "$HERE/p2_queue.sh" --chain ${FROM:+--from "$FROM"} --distill-args "$DISTILL_ARGS" \
  > "$OUT/p2_queue.nohup.out" 2>&1 < /dev/null &
echo "P2Q_LAUNCH pid=$! log=$QLOG from=${FROM:-smoke} distill_args=\"$DISTILL_ARGS\""

#!/usr/bin/env bash
# Stand-network campaign -- TABLE STAND-UP TEST launcher (sn_table_test.py). One Isaac process at a time, under a memory
# cap, detached: a queue runner (setsid nohup) runs the jobs SERIALLY, each `systemd-run --user --scope -p MemoryMax=24G
# -p MemorySwapMax=0 timeout --signal=TERM -k K T <isaac python> sn_table_test.py --headless ...` (memories: video memory
# hazard, Kit DerivedDataCache corruption, GPU queue chaining).
#
#   sn_table_test.sh --smoke [--smoke-no-video] [--dry-run] [--wait-idle] [--foreground]
#       3 jobs, 2 s each: stand_net on rod_min and r on knee_folded (2 envs, no video: both poses, both routed paths),
#       then a 1-env s1 video job on rod_min (--enable_cameras, the env-0 camera, captions, the streaming mp4 writer:
#       the render / memory-hazard path is exercised before the 15-s videos); --smoke-no-video drops the video job
#       (--smoke-video is accepted and is the default)
#   sn_table_test.sh [--pose rod_min|knee_folded|both] [--num-envs 32] [--episode-s 15] [--seed 1]
#                    [--controllers "stand_net r s1 zero"] [--videos "stand_net r s1"] [--dry-run] [--wait-idle]
#                    [--foreground]
#       the FULL run: for each pose (default both), for each controller: N envs (--jitter) headless stats (zero = the
#       no-controller baseline: the actuator layer's slew alone); then for each pose, a 1-env video per --videos
#       controller (env 0 = the nominal start); then the merge table
#   sn_table_test.sh --run-queue STAMP_DIR     (internal: the detached queue runner)
#
# Every run gets a stamp dir $BASE/<date>_<smoke|full>/ (BASE = SN_TABLE_OUT, default parkour/logs/videos/sn_table):
#   facts.json (written before anything launches: the retracted poses, spawn z, clearances, R at start; a refusal here
#   launches nothing), jobs.txt, queue.log (sentinels "SNTABLE JOB START|END <name> rc=..", "SNTABLE QUEUE END rc=.."),
#   queue.pid, <job>.log / .unit, <job>_summary.json, <job>_traces.npz, video jobs' <job>.mp4, and at the end
#   table_test_summary.json (the merge); $BASE/LATEST names the newest stamp.
# "SNTABLE QUEUE END rc=N": 0 every job ran and the merge wrote table_test_summary.json; 1 a job failed (the queue
#   stopped; a partial merge is still attempted); 3 another Isaac process ran too long; 4 STOP file; 5 DerivedDataCache
#   too big / no jobs.txt; 6 every job ran but the MERGE failed (no table_test_summary.json). A watcher reads success
#   only from rc=0.
# Refusals / stops: exit 3 while ANY Isaac python runs (anchored `ps -ww -eo args` pattern, never `pgrep -f`; the
# Kit-free sn_table_test --emit-facts / --merge and sn_dagger --read / --emit-env calls are not Isaac) unless
# --wait-idle; the runner re-checks before EVERY job and waits (30-s polls, up to SN_TABLE_WAIT_MAX s, default 3600)
# while another Isaac process runs; exit 5 when the Kit DerivedDataCache exceeds SN_DDC_MAX_GB (default 80); `touch
# STAMP/STOP` (or BASE/STOP_QUEUE) stops the queue before its next job; stop a running job with `systemctl --user stop
# <unit>.scope` (SIGTERM: the script closes its mp4 / env / Kit), never SIGKILL. The queue stops at the first job that
# fails (rc != 0). SN_TABLE_TIMEOUT (default 1800 s) bounds each job with SIGTERM, SIGKILL SN_TABLE_KILL_AFTER s later.
# NOTE: other launchers' busy patterns (sn_record_video.sh, sn_dagger.sh, run_arm.sh) do not know sn_table_test.py.
# Exit codes: 0 launched (or dry run / foreground queue done), 3 busy, 5 usage / refusal, 7 the queue ended before its
# first job became visible or it never became visible (read queue.log); --foreground: the queue's rc.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)  # this checkout's root, from the script's own location
PY=${SN_PY:-/home/nickmagus/krabby/isaac_venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$HERE/sn_table_test.py
SELF=$HERE/$(basename "$0")
BASE=${SN_TABLE_OUT:-$R/parkour/logs/videos/sn_table}
MEM=${SN_TABLE_MEMORY_MAX:-24G}
DDC=${SN_DDC:-/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache}
DDC_MAX_GB=${SN_DDC_MAX_GB:-80}
TMO=${SN_TABLE_TIMEOUT:-1800}
KILL_AFTER=${SN_TABLE_KILL_AFTER:-120}
WAIT_MAX=${SN_TABLE_WAIT_MAX:-3600}
UP_WAIT=${SN_UP_WAIT:-180}
# the bracketed [i] keeps the pattern from matching a grep / shell line that quotes it
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|evaluation|rsl_rl/train|train|play|play_stream_video|demo|demo_crab_hex_student|record_categories|sn_record_video|sn_dagger|sn_table_test|harvest_omni_rsi_bank|p2_smoke|zero_agent|verify_crab_[a-z_]+|debug_actuator_resolution|training_timeline_probe|crab_hex_phase_cfg_dump)\.py'
NONKIT_PAT='(sn_dagger\.py .*--(read|emit-env)([ =]|$))|(sn_table_test\.py .*--(emit-facts|merge)([ =]|$))'

die() { echo "sn_table_test.sh: $*" >&2; exit 5; }
busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
show_busy() { ps -ww -eo pid,args | grep -E "^ *[0-9]+ ${ISAAC_PAT#^}" | grep -vE "$NONKIT_PAT" >&2; }
ddc_gb() { local g; g=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); echo "${g:-0}"; }

# ------------------------------------------------------------------ the queue runner (detached child)
run_queue() {
  local STAMP=$1 QLOG=$1/queue.log rc_all=0 NAME VIDEO ARGS waited rc UNIT g
  say() { echo "[$(date -Is)] $*" >> "$QLOG"; }
  [ -f "$STAMP/jobs.txt" ] || { say "SNTABLE QUEUE END rc=5 (no jobs.txt)"; return 5; }
  export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 TERM=${TERM:-xterm}
  # the script replaces every KRABBY_* from the eval stamp before Kit; start it from a clean base anyway
  for v in $(env | sed -n 's/^\(KRABBY_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
  say "SNTABLE QUEUE START stamp=$STAMP jobs=$(grep -c . "$STAMP/jobs.txt")"
  while IFS='|' read -r NAME VIDEO ARGS; do
    [ -n "$NAME" ] || continue
    if [ -f "$STAMP/STOP" ] || [ -f "$BASE/STOP_QUEUE" ]; then say "STOP file present: the queue stops before $NAME"; rc_all=4; break; fi
    waited=0
    while busy; do
      if [ "$waited" -ge "$WAIT_MAX" ]; then say "another Isaac process for ${WAIT_MAX} s: the queue stops before $NAME"; rc_all=3; break 2; fi
      sleep 30; waited=$((waited + 30))
    done
    g=$(ddc_gb)
    if [ "$g" -gt "$DDC_MAX_GB" ]; then say "Kit DerivedDataCache ${g} GB > ${DDC_MAX_GB}: the queue stops before $NAME"; rc_all=5; break; fi
    UNIT="sn-table-${NAME//[^A-Za-z0-9_-]/_}-$(date +%s)"
    echo "$UNIT" > "$STAMP/$NAME.unit"
    APP=(--headless)
    [ "$VIDEO" = 1 ] && APP+=(--enable_cameras)
    # shellcheck disable=SC2206
    JARGS=($ARGS)
    CMD=(systemd-run --user --scope -q "--unit=$UNIT" -p "MemoryMax=$MEM" -p MemorySwapMax=0
         timeout --signal=TERM -k "$KILL_AFTER" "$TMO" "$PY" "$SCRIPT" "${APP[@]}" "${JARGS[@]}" --out "$STAMP" --prefix "$NAME")
    { echo "[$(date -Is)] job $NAME unit=$UNIT.scope DerivedDataCache=${g}GB"; printf '%q ' "${CMD[@]}"; echo; } >> "$STAMP/$NAME.log"
    say "SNTABLE JOB START $NAME unit=$UNIT.scope"
    "${CMD[@]}" >> "$STAMP/$NAME.log" 2>&1 < /dev/null
    rc=$?
    say "SNTABLE JOB END $NAME rc=$rc"
    if [ "$rc" -ne 0 ]; then rc_all=1; say "job $NAME failed (rc=$rc): the queue stops (read $STAMP/$NAME.log)"; break; fi
  done < "$STAMP/jobs.txt"
  if ! CUDA_VISIBLE_DEVICES="" "$PY" -B "$SCRIPT" --merge "$STAMP" >> "$QLOG" 2>&1; then
    say "merge failed: no table_test_summary.json (read the lines above)"
    [ "$rc_all" -eq 0 ] && rc_all=6
  fi
  say "SNTABLE QUEUE END rc=$rc_all stamp=$STAMP"
  return "$rc_all"
}

if [ "${1:-}" = "--run-queue" ]; then
  [ $# -eq 2 ] || die "--run-queue STAMP_DIR"
  run_queue "$2"
  exit $?
fi

# ------------------------------------------------------------------ the launcher
DRY=0; WAIT=0; FG=0; SMOKE=0; SMOKE_VIDEO=1; POSE=both; NENV=32; EPS=15; SEED=1
CTRLS="stand_net r s1 zero"; VIDS="stand_net r s1"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --wait-idle) WAIT=1; shift ;;
    --foreground) FG=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --smoke-video) SMOKE_VIDEO=1; shift ;;
    --smoke-no-video) SMOKE_VIDEO=0; shift ;;
    --pose) [ $# -ge 2 ] || die "--pose needs a value"; POSE=$2; shift 2 ;;
    --num-envs) [ $# -ge 2 ] || die "--num-envs needs a value"; NENV=$2; shift 2 ;;
    --episode-s) [ $# -ge 2 ] || die "--episode-s needs a value"; EPS=$2; shift 2 ;;
    --seed) [ $# -ge 2 ] || die "--seed needs a value"; SEED=$2; shift 2 ;;
    --controllers) [ $# -ge 2 ] || die "--controllers needs a value"; CTRLS=$2; shift 2 ;;
    --videos) [ $# -ge 2 ] || die "--videos needs a value"; VIDS=$2; shift 2 ;;
    *) die "unknown option $1" ;;
  esac
done
case "$POSE" in rod_min|knee_folded) POSES=$POSE ;; both) POSES="rod_min knee_folded" ;; *) die "--pose rod_min|knee_folded|both, got '$POSE'" ;; esac
case "$NENV" in ''|*[!0-9]*|0) die "--num-envs must be a positive integer, got '$NENV'" ;; esac
case "$SEED" in ''|*[!0-9]*) die "--seed must be a non-negative integer, got '$SEED'" ;; esac
case "$EPS" in ''|*[!0-9.]*) die "--episode-s must be a positive number, got '$EPS'" ;; esac
for c in $CTRLS $VIDS; do case "$c" in stand_net|r|s1|zero) ;; *) die "unknown controller '$c'" ;; esac; done
case "$TMO" in ''|*[!0-9]*|0) die "SN_TABLE_TIMEOUT must be a positive integer of seconds, got '$TMO'" ;; esac
case "$KILL_AFTER" in ''|*[!0-9]*|0) die "SN_TABLE_KILL_AFTER must be a positive integer of seconds, got '$KILL_AFTER'" ;; esac
case "$WAIT_MAX" in ''|*[!0-9]*|0) die "SN_TABLE_WAIT_MAX must be a positive integer of seconds, got '$WAIT_MAX'" ;; esac
case "$BASE" in *[[:space:]*?\[\]]*) die "output base has whitespace / glob characters: $BASE" ;; esac
[ -f "$SCRIPT" ] || die "missing $SCRIPT"
[ -x "$PY" ] || die "no python at $PY"

JOBS=()
if [ "$SMOKE" = 1 ]; then
  MODE=smoke
  JOBS+=("smoke_stand_net_rod_min|0|--controller stand_net --pose rod_min --num-envs 2 --episode-s 2 --seed $SEED --jitter --smoke")
  JOBS+=("smoke_r_knee_folded|0|--controller r --pose knee_folded --num-envs 2 --episode-s 2 --seed $SEED --jitter --smoke")
  [ "$SMOKE_VIDEO" = 1 ] && JOBS+=("smoke_video_s1_rod_min|1|--controller s1 --pose rod_min --num-envs 1 --episode-s 2 --seed $SEED --video --smoke")
else
  MODE=full
  for p in $POSES; do for c in $CTRLS; do
    JOBS+=("stats_${c}_${p}|0|--controller $c --pose $p --num-envs $NENV --episode-s $EPS --seed $SEED --jitter")
  done; done
  for p in $POSES; do for c in $VIDS; do
    JOBS+=("video_${c}_${p}|1|--controller $c --pose $p --num-envs 1 --episode-s $EPS --seed $SEED --video")
  done; done
fi
STAMP=$BASE/$(date +%Y-%m-%d_%H-%M-%S)_$MODE
FIRST=${JOBS[0]%%|*}

if [ "$DRY" = 1 ]; then
  echo "# stamp: $STAMP  (nothing created)  DerivedDataCache: $(ddc_gb) GB  per-job timeout ${TMO} s (+${KILL_AFTER} s KILL)"
  echo "export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1"
  for j in "${JOBS[@]}"; do
    NAME=${j%%|*}; rest=${j#*|}; VIDEO=${rest%%|*}; ARGS=${rest#*|}
    APP="--headless"; [ "$VIDEO" = 1 ] && APP="--headless --enable_cameras"
    echo "systemd-run --user --scope -q --unit=sn-table-$NAME-<epoch> -p MemoryMax=$MEM -p MemorySwapMax=0 timeout --signal=TERM -k $KILL_AFTER $TMO $PY $SCRIPT $APP $ARGS --out $STAMP --prefix $NAME"
  done
  echo "# then: CUDA_VISIBLE_DEVICES= $PY -B $SCRIPT --merge $STAMP"
  exit 0
fi

if busy; then
  if [ "$WAIT" = 1 ]; then
    echo "sn_table_test.sh: another Isaac process runs; the queue runner will wait (30-s polls, <= ${WAIT_MAX} s per job)" >&2
  else
    echo "sn_table_test.sh: another Isaac process is running (one at a time); use --wait-idle to queue behind it:" >&2
    show_busy
    exit 3
  fi
fi
G=$(ddc_gb)
[ "$G" -gt "$DDC_MAX_GB" ] && die "Kit DerivedDataCache is ${G} GB (> ${DDC_MAX_GB}): check the kit logs for omni.datastore migration loops and quarantine it first"

mkdir -p "$STAMP" || die "cannot create $STAMP"
printf '%s\n' "${JOBS[@]}" > "$STAMP/jobs.txt"
# the facts first (CPU, no Kit): a refusal here launches nothing
CUDA_VISIBLE_DEVICES="" "$PY" -B "$SCRIPT" --emit-facts --out "$STAMP" > "$STAMP/facts.log" 2>&1 \
  || { cat "$STAMP/facts.log" >&2; die "--emit-facts failed (see $STAMP/facts.log); nothing launched"; }
echo "$STAMP" > "$BASE/LATEST"
QLOG=$STAMP/queue.log
echo "[$(date -Is)] sn_table_test.sh mode=$MODE poses=$POSES num_envs=$NENV episode_s=$EPS seed=$SEED controllers='$CTRLS' videos='$VIDS' DerivedDataCache=${G}GB" >> "$QLOG"

if [ "$FG" = 1 ]; then
  run_queue "$STAMP"
  rc=$?
  echo "sn_table_test.sh: queue done rc=$rc stamp=$STAMP (table: $STAMP/table_test_summary.json)"
  exit "$rc"
fi
setsid nohup bash "$SELF" --run-queue "$STAMP" > /dev/null 2>&1 < /dev/null &
echo $! > "$STAMP/queue.pid"
# return only once the first job's python is visible (an Isaac launch started right after must see it busy)
mine() { ps -ww -eo pid,args | grep -E "^ *[0-9]+ ${ISAAC_PAT#^}" | grep -E -- "--prefix $FIRST( |\$)" | awk '{print $1}' | head -1; }
waited=0
PID=$(mine)
until [ -n "$PID" ]; do
  if grep -q "SNTABLE QUEUE END" "$QLOG" 2>/dev/null; then
    echo "sn_table_test.sh: the queue ended before its first job became visible: $(grep 'SNTABLE QUEUE END' "$QLOG" | tail -1); see $QLOG" >&2
    exit 7
  fi
  if [ "$WAIT" != 1 ] && [ "$waited" -ge "$UP_WAIT" ]; then
    echo "sn_table_test.sh: the first job is not visible after ${UP_WAIT} s (queue pid $(cat "$STAMP/queue.pid")); check $QLOG" >&2
    exit 7
  fi
  if [ "$WAIT" = 1 ] && [ "$waited" -ge "$UP_WAIT" ]; then
    echo "sn_table_test.sh: queue pid $(cat "$STAMP/queue.pid") is waiting for the GPU (see $QLOG)"
    exit 0
  fi
  sleep 1; waited=$((waited + 1)); PID=$(mine)
done
echo "sn_table_test.sh: queue pid $(cat "$STAMP/queue.pid") stamp=$STAMP; first job $FIRST python pid=$PID (visible after ${waited} s)"
echo "sn_table_test.sh: done when 'SNTABLE QUEUE END rc=...' appears in $QLOG; table -> $STAMP/table_test_summary.json"

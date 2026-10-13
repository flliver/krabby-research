#!/usr/bin/env bash
# Stand-height campaign -- TABLE / GROUND STAND-UP TEST launcher (sh_table_test.py; K3, PLAN.md H5 / H7). Adapted from
# the closed predecessor's sn_table_test.sh (copied, never edited). One Isaac process at a time, under a memory cap,
# detached: a queue runner (setsid nohup) runs the jobs SERIALLY, each `systemd-run --user --scope -p MemoryMax=24G
# -p MemorySwapMax=0 timeout --signal=TERM T <isaac python> sh_table_test.py --headless ...` (CHARTER section 2
# running rules: one Isaac process, no SIGKILL of Kit -- the timeout sends SIGTERM only, never a follow-up KILL (the
# predecessor's `-k` is dropped: a SIGKILL mid cache write corrupts the Kit DerivedDataCache) -- the DerivedDataCache
# check, MemoryMax, the streaming recorder, no `pgrep -f` self-match).
#
#   sh_table_test.sh --smoke [--smoke-no-video] [--stand-ckpt P] [--dry-run] [--wait-idle] [--foreground]
#       2 stats jobs (r_h and stand_net, 2 envs on each of 3 cells: a 25-in table knee_folded, the ground rod_min, an
#       18-in table random fold; 2-s episodes, --smoke), then a 1-env 2-s r_h video job on ground knee_folded (the
#       render / memory-hazard path); --smoke-no-video drops the video job
#   sh_table_test.sh [--seeds "1 2"] [--controllers "r_h stand_net posture_only s1 amask_old"] [--cells all]
#                    [--envs-per-cell 100] [--split none|surface] [--stand-ckpt P] [--label L]
#                    [--videos "r_h stand_net"] [--video-cells "table_0p635__knee_folded ground__knee_folded
#                    ground__rod_min"] [--video-seed 1] [--dry-run] [--wait-idle] [--foreground]
#       the FULL K3 run: per env seed, per controller one stats job holding every cell (--split surface: one job per
#       surface), --jitter, the pinned K3 episode; then the 1-env videos; then the merge (K3 vs r_h, the R_h floor)
#   sh_table_test.sh --run-queue STAMP_DIR     (internal: the detached queue runner)
#
# Every run gets a stamp dir $BASE/<date>_<smoke|full>/ (BASE = SH_TABLE_OUT, default parkour/logs/videos/sh_table):
#   facts.json (first, Kit-free: a refusal here launches nothing), jobs.txt, queue.log (sentinels "SHTABLE JOB
#   START|END <name> rc=..", "SHTABLE QUEUE END rc=.."), queue.pid, <job>.log / .unit, <job>__<cell>.json per cell,
#   <job>_summary.json, <job>_traces.npz, video jobs' <job>.mp4, and at the end table_test_summary.json (the merge);
#   $BASE/LATEST names the newest stamp.
# "SHTABLE QUEUE END rc=N": 0 every job ran and the merge wrote table_test_summary.json; 1 a job failed (the queue
#   stopped; a partial merge is still attempted); 3 another Isaac process ran too long; 4 STOP file; 5 DerivedDataCache
#   too big / no jobs.txt; 6 every job ran but the MERGE failed. A watcher reads success only from rc=0.
# Refusals / stops: exit 3 while ANY Isaac python runs (anchored `ps -ww -eo args` pattern, never `pgrep -f`; the
# Kit-free --emit-facts / --merge calls and the CPU-only sh_facts / sh_kin / sh_core / sh_table_core are not Isaac)
# unless --wait-idle; the runner re-checks before EVERY job; exit 5 when the Kit DerivedDataCache exceeds SH_DDC_MAX_GB
# (default 80); `touch STAMP/STOP` (or BASE/STOP_QUEUE) stops the queue before its next job; stop a running job with
# `systemctl --user stop <unit>.scope` (SIGTERM: the script closes its mp4 / env / Kit), never SIGKILL. SH_TABLE_TIMEOUT
# (default 3600 s) bounds each job with SIGTERM only (the script's SIGTERM handler closes Kit); a job that still hangs
# is stopped by hand with `systemctl --user stop <unit>.scope`.
# Exit codes: 0 launched (or dry run / foreground queue done), 3 busy, 5 usage / refusal, 7 the queue ended before its
# first job became visible or it never became visible; --foreground: the queue's rc.
set -u
R=/home/nickmagus/krabby/krabby-research
PY=${SH_PY:-/home/nickmagus/krabby/isaac_venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$HERE/sh_table_test.py
SELF=$HERE/$(basename "$0")
BASE=${SH_TABLE_OUT:-$R/parkour/logs/videos/sh_table}
MEM=${SH_TABLE_MEMORY_MAX:-24G}
DDC=${SH_DDC:-/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache}
DDC_MAX_GB=${SH_DDC_MAX_GB:-80}
TMO=${SH_TABLE_TIMEOUT:-3600}
WAIT_MAX=${SH_TABLE_WAIT_MAX:-3600}
UP_WAIT=${SH_UP_WAIT:-180}
# the bracketed [i] keeps the pattern from matching a grep / shell line that quotes it
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|evaluation|rsl_rl/train|train|play|play_stream_video|demo|demo_crab_hex_student|record_categories|sn_record_video|sn_dagger|sn_table_test|sn_stand_net_eval_crab_hex_gait|p2r_mode_eval_crab_hex_gait|harvest_omni_rsi_bank|p2_smoke|zero_agent|verify_crab_[a-z_]+|debug_actuator_resolution|training_timeline_probe|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+)\.py'
NONKIT_PAT='(s[hn]_dagger\.py .*--(read|emit-env)([ =]|$))|(sn_table_test\.py .*--(emit-facts|merge)([ =]|$))|(sh_table_test\.py .*--(emit-facts|merge)([ =]|$))|(/sh_(facts|kin|core|table_core|eval_common|read|bridge_read|observe_read)\.py( |$))'

die() { echo "sh_table_test.sh: $*" >&2; exit 5; }
busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
show_busy() { ps -ww -eo pid,args | grep -E "^ *[0-9]+ ${ISAAC_PAT#^}" | grep -vE "$NONKIT_PAT" >&2; }
ddc_gb() { local g; g=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); echo "${g:-0}"; }

# ------------------------------------------------------------------ the queue runner (detached child)
run_queue() {
  local STAMP=$1 QLOG=$1/queue.log rc_all=0 NAME VIDEO ARGS waited rc UNIT g
  say() { echo "[$(date -Is)] $*" >> "$QLOG"; }
  [ -f "$STAMP/jobs.txt" ] || { say "SHTABLE QUEUE END rc=5 (no jobs.txt)"; return 5; }
  export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 TERM=${TERM:-xterm}
  # the script replaces every KRABBY_* from the eval stamp before Kit; start it from a clean base anyway
  for v in $(env | sed -n 's/^\(KRABBY_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
  say "SHTABLE QUEUE START stamp=$STAMP jobs=$(grep -c . "$STAMP/jobs.txt")"
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
    UNIT="sh-table-${NAME//[^A-Za-z0-9_-]/_}-$(date +%s)"
    echo "$UNIT" > "$STAMP/$NAME.unit"
    APP=(--headless)
    [ "$VIDEO" = 1 ] && APP+=(--enable_cameras)
    # shellcheck disable=SC2206
    JARGS=($ARGS)
    CMD=(systemd-run --user --scope -q "--unit=$UNIT" -p "MemoryMax=$MEM" -p MemorySwapMax=0
         timeout --signal=TERM "$TMO" "$PY" "$SCRIPT" "${APP[@]}" "${JARGS[@]}" --out "$STAMP" --prefix "$NAME")
    { echo "[$(date -Is)] job $NAME unit=$UNIT.scope DerivedDataCache=${g}GB"; printf '%q ' "${CMD[@]}"; echo; } >> "$STAMP/$NAME.log"
    say "SHTABLE JOB START $NAME unit=$UNIT.scope"
    "${CMD[@]}" >> "$STAMP/$NAME.log" 2>&1 < /dev/null
    rc=$?
    say "SHTABLE JOB END $NAME rc=$rc"
    if [ "$rc" -ne 0 ]; then rc_all=1; say "job $NAME failed (rc=$rc): the queue stops (read $STAMP/$NAME.log)"; break; fi
  done < "$STAMP/jobs.txt"
  if ! CUDA_VISIBLE_DEVICES="" "$PY" -B "$SCRIPT" --merge "$STAMP" >> "$QLOG" 2>&1; then
    say "merge failed: no table_test_summary.json (read the lines above)"
    [ "$rc_all" -eq 0 ] && rc_all=6
  fi
  say "SHTABLE QUEUE END rc=$rc_all stamp=$STAMP"
  return "$rc_all"
}

if [ "${1:-}" = "--run-queue" ]; then
  [ $# -eq 2 ] || die "--run-queue STAMP_DIR"
  run_queue "$2"
  exit $?
fi

# ------------------------------------------------------------------ the launcher
DRY=0; WAIT=0; FG=0; SMOKE=0; SMOKE_VIDEO=1; SEEDS="1 2"; NENV=100; CELLS=all; SPLIT=none; STAND_CKPT=""; LABEL=""
CTRLS="r_h stand_net posture_only s1 amask_old"; VIDS="r_h stand_net"; VSEED=1
VCELLS="table_0p635__knee_folded ground__knee_folded ground__rod_min"
SURFACES="table_0p4572 table_0p635 table_0p762 ground"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --wait-idle) WAIT=1; shift ;;
    --foreground) FG=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --smoke-video) SMOKE_VIDEO=1; shift ;;
    --smoke-no-video) SMOKE_VIDEO=0; shift ;;
    --seeds) [ $# -ge 2 ] || die "--seeds needs a value"; SEEDS=$2; shift 2 ;;
    --controllers) [ $# -ge 2 ] || die "--controllers needs a value"; CTRLS=$2; shift 2 ;;
    --cells) [ $# -ge 2 ] || die "--cells needs a value"; CELLS=$2; shift 2 ;;
    --envs-per-cell) [ $# -ge 2 ] || die "--envs-per-cell needs a value"; NENV=$2; shift 2 ;;
    --split) [ $# -ge 2 ] || die "--split needs a value"; SPLIT=$2; shift 2 ;;
    --stand-ckpt) [ $# -ge 2 ] || die "--stand-ckpt needs a value"; STAND_CKPT=$2; shift 2 ;;
    --label) [ $# -ge 2 ] || die "--label needs a value"; LABEL=$2; shift 2 ;;
    --videos) [ $# -ge 2 ] || die "--videos needs a value"; VIDS=$2; shift 2 ;;
    --video-cells) [ $# -ge 2 ] || die "--video-cells needs a value"; VCELLS=$2; shift 2 ;;
    --video-seed) [ $# -ge 2 ] || die "--video-seed needs a value"; VSEED=$2; shift 2 ;;
    *) die "unknown option $1" ;;
  esac
done
case "$NENV" in ''|*[!0-9]*|0) die "--envs-per-cell must be a positive integer, got '$NENV'" ;; esac
case "$VSEED" in ''|*[!0-9]*) die "--video-seed must be a non-negative integer, got '$VSEED'" ;; esac
for s in $SEEDS; do case "$s" in ''|*[!0-9]*) die "--seeds must be non-negative integers, got '$SEEDS'" ;; esac; done
[ -n "$SEEDS" ] || die "--seeds is empty"
case "$SPLIT" in none|surface) ;; *) die "--split none|surface, got '$SPLIT'" ;; esac
for c in $CTRLS $VIDS; do case "$c" in r_h|stand_net|posture_only|s1|amask_old) ;; *) die "unknown controller '$c'" ;; esac; done
case "$CELLS" in *[[:space:]]*) die "--cells is one comma list (no spaces), got '$CELLS'" ;; esac
case "$STAND_CKPT" in *[[:space:]*?\[\]]*) die "--stand-ckpt has whitespace / glob characters: $STAND_CKPT" ;; esac
case "$LABEL" in ''|*[!A-Za-z0-9_.-]*) [ -z "$LABEL" ] || die "--label must be [A-Za-z0-9_.-]+, got '$LABEL'" ;; esac
case "$TMO" in ''|*[!0-9]*|0) die "SH_TABLE_TIMEOUT must be a positive integer of seconds, got '$TMO'" ;; esac
case "$WAIT_MAX" in ''|*[!0-9]*|0) die "SH_TABLE_WAIT_MAX must be a positive integer of seconds, got '$WAIT_MAX'" ;; esac
case "$BASE" in *[[:space:]*?\[\]]*) die "output base has whitespace / glob characters: $BASE" ;; esac
[ -f "$SCRIPT" ] || die "missing $SCRIPT"
[ -x "$PY" ] || die "no python at $PY"

# per-controller extra args: the stand checkpoint and the label reach stand_net only
xargs_for() {
  local c=$1 x=""
  if [ "$c" = stand_net ]; then
    [ -n "$STAND_CKPT" ] && x="$x --stand-ckpt $STAND_CKPT"
    [ -n "$LABEL" ] && x="$x --label $LABEL"
  fi
  echo "$x"
}

JOBS=()
if [ "$SMOKE" = 1 ]; then
  MODE=smoke
  SC="table_0p635__knee_folded,ground__rod_min,table_0p4572__random"
  for c in r_h stand_net; do
    JOBS+=("smoke_${c}|0|--controller $c --cells $SC --envs-per-cell 2 --episode-s 2 --seed 1 --jitter --smoke$(xargs_for $c)")
  done
  [ "$SMOKE_VIDEO" = 1 ] && JOBS+=("smoke_video_r_h_ground__knee_folded|1|--controller r_h --cells ground__knee_folded --envs-per-cell 1 --episode-s 2 --seed 1 --video --smoke")
else
  MODE=full
  if [ "$SPLIT" = surface ]; then
    if [ "$CELLS" = all ]; then GROUPS_=$SURFACES; else GROUPS_=$(echo "$CELLS" | tr ',' ' '); fi
  else
    GROUPS_=$CELLS
  fi
  for s in $SEEDS; do for c in $CTRLS; do for gq in $GROUPS_; do
    if [ "$SPLIT" = surface ]; then n="stats_${c}_s${s}_${gq}"; else n="stats_${c}_s${s}"; fi
    JOBS+=("${n}|0|--controller $c --cells $gq --envs-per-cell $NENV --seed $s --jitter$(xargs_for $c)")
  done; done; done
  for c in $VIDS; do for vc in $VCELLS; do
    JOBS+=("video_${c}_${vc}|1|--controller $c --cells $vc --envs-per-cell 1 --seed $VSEED --video$(xargs_for $c)")
  done; done
fi
STAMP=$BASE/$(date +%Y-%m-%d_%H-%M-%S)_$MODE
FIRST=${JOBS[0]%%|*}

if [ "$DRY" = 1 ]; then
  echo "# stamp: $STAMP  (nothing created)  DerivedDataCache: $(ddc_gb) GB  per-job timeout ${TMO} s (SIGTERM only)"
  echo "export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1"
  echo "CUDA_VISIBLE_DEVICES= $PY -B $SCRIPT --emit-facts --out $STAMP"
  for j in "${JOBS[@]}"; do
    NAME=${j%%|*}; rest=${j#*|}; VIDEO=${rest%%|*}; ARGS=${rest#*|}
    APP="--headless"; [ "$VIDEO" = 1 ] && APP="--headless --enable_cameras"
    echo "systemd-run --user --scope -q --unit=sh-table-$NAME-<epoch> -p MemoryMax=$MEM -p MemorySwapMax=0 timeout --signal=TERM $TMO $PY $SCRIPT $APP $ARGS --out $STAMP --prefix $NAME"
  done
  echo "# then: CUDA_VISIBLE_DEVICES= $PY -B $SCRIPT --merge $STAMP"
  exit 0
fi

if busy; then
  if [ "$WAIT" = 1 ]; then
    echo "sh_table_test.sh: another Isaac process runs; the queue runner will wait (30-s polls, <= ${WAIT_MAX} s per job)" >&2
  else
    echo "sh_table_test.sh: another Isaac process is running (one at a time); use --wait-idle to queue behind it:" >&2
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
echo "[$(date -Is)] sh_table_test.sh mode=$MODE seeds='$SEEDS' cells=$CELLS envs_per_cell=$NENV split=$SPLIT controllers='$CTRLS' videos='$VIDS' video_cells='$VCELLS' stand_ckpt='${STAND_CKPT:-default}' DerivedDataCache=${G}GB" >> "$QLOG"

if [ "$FG" = 1 ]; then
  run_queue "$STAMP"
  rc=$?
  echo "sh_table_test.sh: queue done rc=$rc stamp=$STAMP (table: $STAMP/table_test_summary.json)"
  exit "$rc"
fi
setsid nohup bash "$SELF" --run-queue "$STAMP" > /dev/null 2>&1 < /dev/null &
echo $! > "$STAMP/queue.pid"
# return only once the first job's python is visible (an Isaac launch started right after must see it busy)
mine() { ps -ww -eo pid,args | grep -E "^ *[0-9]+ ${ISAAC_PAT#^}" | grep -E -- "--prefix $FIRST( |\$)" | awk '{print $1}' | head -1; }
waited=0
PID=$(mine)
until [ -n "$PID" ]; do
  if grep -q "SHTABLE QUEUE END" "$QLOG" 2>/dev/null; then
    echo "sh_table_test.sh: the queue ended before its first job became visible: $(grep 'SHTABLE QUEUE END' "$QLOG" | tail -1); see $QLOG" >&2
    exit 7
  fi
  if [ "$WAIT" != 1 ] && [ "$waited" -ge "$UP_WAIT" ]; then
    echo "sh_table_test.sh: the first job is not visible after ${UP_WAIT} s (queue pid $(cat "$STAMP/queue.pid")); check $QLOG" >&2
    exit 7
  fi
  if [ "$WAIT" = 1 ] && [ "$waited" -ge "$UP_WAIT" ]; then
    echo "sh_table_test.sh: queue pid $(cat "$STAMP/queue.pid") is waiting for the GPU (see $QLOG)"
    exit 0
  fi
  sleep 1; waited=$((waited + 1)); PID=$(mine)
done
echo "sh_table_test.sh: queue pid $(cat "$STAMP/queue.pid") stamp=$STAMP; first job $FIRST python pid=$PID (visible after ${waited} s)"
echo "sh_table_test.sh: done when 'SHTABLE QUEUE END rc=...' appears in $QLOG; table -> $STAMP/table_test_summary.json"

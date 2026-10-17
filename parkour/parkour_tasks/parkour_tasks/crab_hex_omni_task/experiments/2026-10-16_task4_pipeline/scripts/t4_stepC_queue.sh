#!/bin/bash
# PLAN step 8 (Phase C preflight, GPU ~26 GPU-h): the step-8 queue. ONE job at a time from the job table
# (scripts/t4_stepC_jobs.py: C1 inertness -> C2 eval bridge -> C3 payload -> C4 camera / student -> C5 sweep anchors ->
# C6 stand -> C7 full-graph dry pass; cheapest disqualifier first), each job's script written to
# preflight/stepC/jobs/<name>.sh and run here, its log in preflight/stepC/logs/<name>.log. The pass rules are
# preflight/STEPC_RULES.md (applied by scripts/t4_stepC_read.py --check); the summary is t4_stepC_read.py (no args).
#
#   t4_stepC_queue.sh [--only C1,C2,...]          run (resumable: complete jobs are skipped, by their outputs)
#   t4_stepC_queue.sh --list [--only ...]         the job list, status and GPU-h -- launches NOTHING
#   t4_stepC_queue.sh --retry NAME|failed         clear a failed / STOPPED job (and its group's checks), then exit
#   t4_stepC_queue.sh --accept NAME --why TEXT    accept a STOP (recorded), then exit
# Run it detached:  setsid nohup bash t4_stepC_queue.sh > /dev/null 2>&1 &   (watch preflight/stepC/stepC_queue.log)
# Knobs: T4C_DRY=reduced|official (C7 spec; default reduced), T4C_DRIVER_ARGS (extra run_candidates args, e.g.
# --accept-code-change WHY after a STOP-C fix), TP_NOTIFY_CMD (called with each STOP / END line).
#
# Sentinels (one line each, appended to preflight/stepC/stepC_queue.log):
#   T4C START | T4C JOB <name> ... | T4C DONE <name> | T4C SKIP <name> | T4C FAIL <name> (report-only: the queue goes
#   on; jobs that need it are blocked) | T4C STOP <name> <why> (a STOP-C trigger or an unresolved STOP: the queue
#   EXITS 3) | T4C END <why>
# Safety: one Isaac process at a time (anchored `ps -ww -eo args` pattern, never `pgrep -f`: the pattern below cannot
# match this script, the job table or the readers -- none runs under isaac_venv/bin/python as an Isaac script name);
# every Kit launch is memory-capped by its launcher (systemd-run --scope MemoryMax=48G) and bounded with
# `timeout --signal=TERM`; nothing here ever sends SIGKILL; a STOP_QUEUE file in preflight/stepC stops it between jobs.
set -u
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
C=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline
SC=$C/preflight/stepC
JOBS=$C/scripts/t4_stepC_jobs.py
MAKE=$C/scripts/t4_make_stepC.py
LOG=$SC/stepC_queue.log
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+|bc_dump_cfgs|t4_dump_cfgs)\.py'
NONKIT_PAT='(s[hn]_dagger\.py .*--(read|emit-env)([ =]|$))|(s[hn]_table_test\.py .*--(emit-facts|merge)([ =]|$))|(/s[hn]_(facts|kin|core|table_core|eval_common|read|bridge_read|observe_read|calib|common)\.py( |$))'

ONLY=""; MODE=run; NAME=""; WHY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY=$2; shift 2 ;;
    --list) MODE=list; shift ;;
    --retry) MODE=retry; NAME=$2; shift 2 ;;
    --accept) MODE=accept; NAME=$2; shift 2 ;;
    --why) WHY=$2; shift 2 ;;
    *) echo "t4_stepC_queue.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

# a clean environment: no KRABBY_* / stand-tool / pipeline variable of the caller reaches any job (each job script
# sets exactly what it needs); TP_NOTIFY_CMD and the T4C_* knobs are kept
for v in $(env | grep -oE '^(KRABBY_|SN_WALKER_|P2_|P2R_|SH_|SN_STAND_|TP_|T4_)[A-Za-z0-9_]*' | sort -u); do
  [ "$v" = TP_NOTIFY_CMD ] || unset "$v"
done
export PYTHONDONTWRITEBYTECODE=1

ONLYARG=(); [ -n "$ONLY" ] && ONLYARG=(--only "$ONLY")
case "$MODE" in
  list) exec "$PY" -I "$JOBS" list "${ONLYARG[@]}" ;;
  retry) exec "$PY" -I "$JOBS" retry "$NAME" ;;
  accept) exec "$PY" -I "$JOBS" accept "$NAME" --why "$WHY" ;;
esac

mkdir -p "$SC/logs" "$SC/jobs"
say() { local line="T4C $* $(date -Is)"; echo "$line" >> "$LOG"; echo "$line"; }
notify() { [ -n "${TP_NOTIFY_CMD:-}" ] && $TP_NOTIFY_CMD "$1" > /dev/null 2>&1 || true; }
busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }

# one queue at a time: the pidfile holds OUR pid and /proc/<pid>/cmdline must be this script
PIDF=$SC/queue.pid
if [ -f "$PIDF" ]; then
  op=$(head -n 1 "$PIDF" 2>/dev/null)
  if [ -n "$op" ] && [ -r "/proc/$op/cmdline" ] && tr '\0' ' ' < "/proc/$op/cmdline" | grep -q 't4_stepC_queue\.sh'; then
    echo "t4_stepC_queue.sh: a queue is already running (pid $op)" >&2; exit 3
  fi
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

# the static inputs (candidates, env files, shape manifests, dry spec) must be the generator's, byte for byte
if ! "$PY" -I "$MAKE" --check > /dev/null 2>&1; then
  "$PY" -I "$MAKE" >> "$LOG" 2>&1 || { say STOP make "t4_make_stepC.py failed"; exit 5; }
fi
say START "pid=$$ only=${ONLY:-all} dry=${T4C_DRY:-reduced}"
while true; do
  if [ -f "$SC/STOP_QUEUE" ]; then say STOP queue "STOP_QUEUE present (remove it to resume)"; exit 4; fi
  line=$("$PY" -I "$JOBS" next "${ONLYARG[@]}" 2>> "$LOG") || { say STOP jobs "the job table failed (see $LOG)"; exit 5; }
  set -- $line
  kind=${1:-}
  case "$kind" in
    END) say END "${*:2}"; notify "T4C END ${*:2}"; exit 0 ;;
    STOP) say STOP "${*:2}"; notify "T4C STOP ${*:2}"; exit 3 ;;
    JOB) ;;
    *) say STOP jobs "unparseable next: $line"; exit 5 ;;
  esac
  name=$2; isaac=$3; gh=$4
  if [ "$isaac" = 1 ] && busy; then
    say WAIT "$name another Isaac process is running"
    while busy; do sleep 30; done
  fi
  say JOB "$name gpu_h=$gh script=$SC/jobs/$name.sh"
  t0=$(date +%s)
  bash "$SC/jobs/$name.sh" >> "$SC/logs/$name.log" 2>&1 < /dev/null
  rc=$?
  res=$("$PY" -I "$JOBS" mark "$name" "$rc" 2>> "$LOG")
  wall=$(( $(date +%s) - t0 ))
  case "${res%% *}" in
    DONE) say DONE "$name rc=$rc wall_s=$wall ${res#DONE $name}" ;;
    FAIL) say FAIL "$name rc=$rc wall_s=$wall (report-only; dependents blocked) ${res#FAIL $name}" ;;
    STOP) say STOP "$name rc=$rc wall_s=$wall ${res#STOP $name}"; notify "T4C STOP $name ${res#STOP $name}"; exit 3 ;;
    *) say STOP "$name" "mark failed: $res"; exit 5 ;;
  esac
done

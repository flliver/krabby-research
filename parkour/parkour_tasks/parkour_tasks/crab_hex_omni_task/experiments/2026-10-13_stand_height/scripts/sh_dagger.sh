#!/usr/bin/env bash
# Stand-height campaign (2026-10-13_stand_height), PLAN steps 5 / 6 / 7 -- launch ONE sh_dagger.py run on the GPU.
# Copied from the closed predecessor's sn_dagger.sh (sha256 9bc34b2d...; never edited): names, run dir, busy pattern and
# options adapted. NOT run by the builder: the orchestrator runs it, serially, one Isaac process at a time.
#
#   sh_dagger.sh [--dry-run] [--detach] [--wait-idle] NAME -- <sh_dagger.py args>
#     NAME            [A-Za-z0-9_]+ ; the run dir is $RUNS/NAME (must not exist / be empty: one job, no resume)
#     <args>          --mode dagger [--arm mask] --seed S --iterations N [--num-envs 1024] [--smoke [--smoke-iters 100]]
#                     [--deadband D] (the student-executed residual deadband, CHARTER 2026-10-14; default 0 = off)
#                     [--trim-branch [--trim-min X]] (the trim weight category, CHARTER 2026-10-14; default off)
#                     (--out is the script's; --headless is added; AppLauncher args such as --device pass through)
#     --dry-run       print the environment lines and the command, launch nothing
#     --detach        setsid nohup; returns once the trainer python is VISIBLE to the busy checks (ps), so an eval
#                     started right after cannot overlap it (exit 7 if it ended first or is not visible after
#                     SH_UP_WAIT s, default 120); default: foreground, the trainer's rc
#     --wait-idle     wait (30-s polls) instead of refusing while another Isaac process runs
#
# What it does (CHARTER section 2 running rules; memories: Kit cache, video / memory hazards, GPU queue chaining):
#   1. refuses on a STOP_QUEUE file, a bad NAME, a non-empty run dir, an --out / --emit-env / --read in the args;
#   2. one Isaac process at a time: refuses (exit 3) while ANY Isaac python runs -- the evals (eval_crab_hex_gait and
#      every launcher named *eval_crab_hex_gait), train.py, run_phases / run_candidates, a trainer of this campaign or
#      the predecessor's, the table tests, the streaming recorders (NOT the Kit-free `--read` / `--emit-env` calls) --
#      anchored `ps -ww -eo args` match, never `pgrep -f` (it would match its own command line);
#   3. prints the Kit DerivedDataCache size (refuses above SH_DDC_MAX_GB, default 80; quarantine per the memory note);
#   4. composes the process environment with `sh_dagger.py --emit-env` (Kit-free; the predecessor trainer's env,
#      unchanged: every KRABBY_* / P2R_* / SN_STAND_* unset, then the preset 2a + plant + the FULL P2__S1_s18 overlay +
#      the four D2 / D5 changes, OMNI_KIT_ACCEPT_EULA=yes, TERM=xterm, PYTHONUNBUFFERED=1);
#   5. runs `systemd-run --user --scope -q --unit=sh-dagger-NAME-<epoch> -p MemoryMax=48G -p MemorySwapMax=0
#      timeout --signal=TERM <SH_TIMEOUT> <isaac python> sh_dagger.py --headless <args> --out $RUNS/NAME` (SIGTERM is
#      the signal the trainer handles; never SIGKILL. SH_TIMEOUT seconds, default 3600 with --smoke else 14400 --
#      size it from the smoke's s_per_iter x --iterations + margin for the real runs), stdout + stderr appended to
#      $RUNS/NAME.log, the unit name in $RUNS/NAME.unit. STOP it with `systemctl --user stop <unit>.scope`.
#   Sentinel lines "SHDAGGER START|END NAME ..." go to $SHOUT/sh_dagger.log (the watcher's log).
# Exit codes: the trainer's (0 done, 1 refused after Kit start / crashed, 2 refused before Kit, 4 interrupted,
# 6 smoke FAIL) or 124 (the SH_TIMEOUT bound fired) in the foreground; 3 busy; 4 STOP_QUEUE; 5 usage / refusal here;
# 0 after a detached launch whose trainer is visible; 7 a detached launch that ended before it became visible.
set -u
R=/home/nickmagus/krabby/krabby-research
PY=${SH_PY:-/home/nickmagus/krabby/isaac_venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
TRAINER=$HERE/sh_dagger.py
SHOUT=${SH_OUT:-$R/parkour/outputs/omni_plan/stand_height}
RUNS=$SHOUT/runs
SNOUT=$R/parkour/outputs/omni_plan/stand_network
P2OUT=$R/parkour/outputs/omni_plan/phase2
QLOG=$SHOUT/sh_dagger.log
MEM=${SH_MEMORY_MAX:-48G}
DDC=${SH_DDC:-/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache}
DDC_MAX_GB=${SH_DDC_MAX_GB:-80}
ISAAC_PAT='^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|rsl_rl/train|s[hn]_dagger|s[hn]_table_test|s[hn]_record_video|training_timeline_probe|crab_hex_phase_cfg_dump)\.py'
#: the Kit-free trainer calls (CPU reads, env composition): never an Isaac process
NONKIT_PAT='s[hn]_dagger\.py .*--(read|emit-env)([ =]|$)'
UP_WAIT=${SH_UP_WAIT:-120}

die() { echo "sh_dagger.sh: $*" >&2; exit 5; }
DRY=0; DETACH=0; WAIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --detach) DETACH=1; shift ;;
    --wait-idle) WAIT=1; shift ;;
    --) die "NAME missing before --" ;;
    -*) die "unknown option $1" ;;
    *) break ;;
  esac
done
[ $# -ge 1 ] || die "usage: sh_dagger.sh [--dry-run] [--detach] [--wait-idle] NAME -- <sh_dagger.py args>"
NAME=$1; shift
case "$NAME" in ''|*[!A-Za-z0-9_]*) die "NAME '$NAME' must be [A-Za-z0-9_]+" ;; esac
[ "${1:-}" = "--" ] || die "expected '--' after NAME"
shift
ARGS=("$@")
for a in "${ARGS[@]}"; do
  case "$a" in
    --out|--out=*|--emit-env|--read|--read=*) die "'$a' is not allowed here (the script owns the run dir)" ;;
  esac
done
OUT=$RUNS/$NAME
LOG=$RUNS/$NAME.log
case "$OUT" in *[[:space:]*?\[\]]*) die "run dir path has whitespace / glob characters: $OUT" ;; esac
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then die "run dir $OUT is not empty (one job, no resume)"; fi
[ -f "$TRAINER" ] || die "no trainer $TRAINER"
if [ -f "$SHOUT/STOP_QUEUE" ] || [ -f "$SNOUT/STOP_QUEUE" ] || [ -f "$P2OUT/STOP_QUEUE" ]; then
  echo "sh_dagger.sh: STOP_QUEUE present" >&2; exit 4
fi

busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
if [ "$DRY" != 1 ]; then
  if busy; then
    if [ "$WAIT" = 1 ]; then
      while busy; do sleep 30; done
    else
      echo "sh_dagger.sh: another Isaac process is running (one at a time); use --wait-idle to queue" >&2
      ps -ww -eo pid,args | grep -E "^ *[0-9]+ ${ISAAC_PAT#^}" | grep -vE "$NONKIT_PAT" >&2
      exit 3
    fi
  fi
fi

DDC_GB=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1)
DDC_GB=${DDC_GB:-0}
if [ "$DDC_GB" -gt "$DDC_MAX_GB" ]; then
  die "Kit DerivedDataCache is ${DDC_GB} GB (> ${DDC_MAX_GB}): check kit logs for omni.datastore migration loops and quarantine it first"
fi

SMOKE=0
for a in "${ARGS[@]}"; do [ "$a" = --smoke ] && SMOKE=1; done
if [ "$SMOKE" = 1 ]; then TMO=${SH_TIMEOUT:-3600}; else TMO=${SH_TIMEOUT:-14400}; fi
case "$TMO" in ''|*[!0-9]*|0) die "SH_TIMEOUT must be a positive integer of seconds, got '$TMO'" ;; esac

ENVLINES=$("$PY" -B "$TRAINER" --emit-env) || die "environment composition refused (see above)"
UNIT="sh-dagger-${NAME}-$(date +%s)"
CMD=(systemd-run --user --scope -q "--unit=$UNIT" -p "MemoryMax=$MEM" -p MemorySwapMax=0
     timeout --signal=TERM "$TMO" "$PY" "$TRAINER" --headless "${ARGS[@]}" --out "$OUT")
if [ "$DRY" = 1 ]; then
  printf '%s\n' "$ENVLINES"
  printf '%q ' "${CMD[@]}"; echo
  echo "# log: $LOG  unit: $UNIT.scope  DerivedDataCache: ${DDC_GB} GB  timeout: ${TMO} s"
  exit 0
fi

eval "$ENVLINES"
mkdir -p "$RUNS"
echo "$UNIT" > "$RUNS/$NAME.unit"
say() { echo "[$(date -Is)] $*" >> "$QLOG"; }
{ echo "[$(date -Is)] sh_dagger.sh $NAME unit=$UNIT.scope DerivedDataCache=${DDC_GB}GB"; printf '%q ' "${CMD[@]}"; echo; } >> "$LOG"
say "SHDAGGER START $NAME unit=$UNIT.scope out=$OUT args=${ARGS[*]}"
if [ "$DETACH" = 1 ]; then
  END="SHDAGGER END $NAME unit=$UNIT.scope"
  setsid nohup bash -c '"$@" >> "$0" 2>&1 < /dev/null; rc=$?; echo "[$(date -Is)] '"$END"' rc=$rc" >> "'"$QLOG"'"' \
    "$LOG" "${CMD[@]}" > /dev/null 2>&1 < /dev/null &
  PID=$!
  up() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -vE "$NONKIT_PAT" | grep -qE -- "--out $OUT( |\$)"; }
  waited=0
  until up; do
    if grep -qF "$END " "$QLOG" 2>/dev/null; then
      echo "sh_dagger.sh: the detached run ended before it became visible: $(grep -F "$END " "$QLOG" | tail -1); see $LOG" >&2
      exit 7
    fi
    if [ "$waited" -ge "$UP_WAIT" ]; then
      echo "sh_dagger.sh: the trainer is not visible after ${UP_WAIT} s (pid=$PID unit=$UNIT.scope); check $LOG before launching anything else" >&2
      exit 7
    fi
    sleep 1; waited=$((waited + 1))
  done
  echo "sh_dagger.sh: detached pid=$PID unit=$UNIT.scope log=$LOG out=$OUT (trainer visible after ${waited} s)"
  exit 0
fi
"${CMD[@]}" >> "$LOG" 2>&1 < /dev/null
rc=$?
say "SHDAGGER END $NAME rc=$rc"
exit "$rc"

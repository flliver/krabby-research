#!/usr/bin/env bash
# Stand-height campaign -- generate the step-8 raise / lower schedule and record its video (VIDEO ONLY; read by no
# gate). Adapted copy of the closed predecessor's 2026-10-11_stand_network/scripts/sn_record_video.sh (never edited).
# The recorder is an Isaac process: one at a time, under a memory cap, never SIGKILLed, detached (memories: video
# memory hazard, Kit DerivedDataCache corruption, GPU queue chaining; CHARTER section 2: never SIGKILL Kit).
#
#   sh_record_video.sh [--dry-run] [--overwrite] [--smoke | --seed N] [--total-s S] [--prefix P] [-- <recorder args>]
#     (default)       the fixed RAISE / LOWER plan (~155 s): videos/scenarios_sh_video_raise.yaml, row sh_raise__omni
#     --smoke         the ~20 s smoke plan (walk -> 1.20 -> 1.00 -> walk: a raise, an in-stand lower, a restore):
#                     videos/scenarios_sh_video_smoke.yaml, row sh_smoke__omni
#     --seed N        the RANDOM plan with seed N (random walks, random stand heights / lengths):
#                     videos/scenarios_sh_video_random.yaml, row sh_random__omni
#     --total-s S     random plan only: the target video length in s (default 150)
#     --prefix P      output prefix (default sh_raise / sh_smoke / sh_random_s<N>)
#     --overwrite     allow replacing an existing <prefix>_sequence.mp4
#     --dry-run       print the schedule (nothing written), whether an Isaac process is running, the environment and
#                     the command; launch nothing
#     -- <args>       passed to sh_record_video.py -- the head goes here, e.g.
#                     -- --stand-ckpt /abs/.../runs/SH_T18/model_5000.pt --stand-mask 1 --deadband 0.25
#
# What it does:
#   1. refuses (exit 3) while ANY Isaac python runs -- evals (*eval_crab_hex_gait.py), DAgger trainers, table tests,
#      video recorders, train / play / demo scripts, drivers, any campaign-local sh_*.py except the Kit-free modes /
#      CPU scripts (NONKIT_PAT: sh_eval_rows.sh's list + this video's schedule generator and core) -- matched on
#      `ps -ww -eo args` with an anchored, bracketed pattern (never `pgrep -f`: it would match its own command line);
#   2. refuses (exit 5) when the Kit DerivedDataCache exceeds SH_DDC_MAX_GB (default 80 GB: the corruption precedent);
#   3. writes the schedule (`sh_video_schedule.py`, CPU only): videos/scenarios_sh_video_<plan>.yaml + the JSON;
#   4. runs `systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 timeout --signal=TERM <T>
#      <isaac python> sh_record_video.py --headless --enable_cameras --manifest ... --sequence ... --out-dir $OUT
#      --prefix P` with OMNI_KIT_ACCEPT_EULA=yes, PYTHONUNBUFFERED=1, detached via `setsid nohup`; returns once the
#      recorder python is visible (its pid -> $OUT/P.pid, the scope unit -> $OUT/P.unit, the log -> $OUT/P.log).
#      SH_VIDEO_TIMEOUT seconds (default 5400; --smoke 1800) bounds the run with SIGTERM only (the recorder unwinds:
#      the mp4 writer closes, Kit closes). NO `timeout -k`: the campaign never SIGKILLs Kit (the predecessor's -k is
#      dropped on purpose). A Kit hung inside C++ ignores the TERM's SystemExit until control returns; then stop it
#      with `systemctl --user stop <unit>.scope` (the CHARTER's stop path), never with a SIGKILL of your own.
# Exit codes: 0 launched (or dry run), 3 busy, 5 usage / refusal, 7 the recorder ended before becoming visible or
# never became visible (read the log).
set -u
R=/home/nickmagus/krabby/krabby-research
PY=${SH_PY:-/home/nickmagus/krabby/isaac_venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
GEN=$HERE/sh_video_schedule.py
REC=$HERE/sh_record_video.py
VID=$(cd "$HERE/.." && pwd)/videos
OUT=${SH_VIDEO_OUT:-$R/parkour/logs/videos/sh_stand_height}
MEM=${SH_VIDEO_MEMORY_MAX:-24G}
DDC=${SH_DDC:-/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache}
DDC_MAX_GB=${SH_DDC_MAX_GB:-80}
UP_WAIT=${SH_UP_WAIT:-120}
# sh_eval_rows.sh's busy list (trainers, table tests, video recorders, train / play, evals, drivers, any campaign-local
# sh_*.py); the bracketed [i] keeps the pattern from matching a grep / shell line that quotes it
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|evaluation|rsl_rl/train|train|play|play_stream_video|demo|demo_crab_hex_student|record_categories|sn_record_video|sn_dagger|sn_table_test|sn_stand_net_eval_crab_hex_gait|p2r_mode_eval_crab_hex_gait|harvest_omni_rsi_bank|p2_smoke|zero_agent|verify_crab_[a-z_]+|debug_actuator_resolution|training_timeline_probe|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+)\.py'
# Kit-free modes and CPU-only campaign scripts (sh_eval_rows.sh's list + this video's CPU helpers); the CPU scripts match
# by absolute OR relative path (e.g. `python sh_read.py --spec ...` run from the scripts dir)
NONKIT_PAT='(s[hn]_dagger\.py .*--(read|emit-env)([ =]|$))|(s[hn]_table_test\.py .*--(emit-facts|merge)([ =]|$))|((^|[ /])sh_(facts|kin|core|table_core|eval_common|read|bridge_read|observe_read|video_schedule|video_core)\.py( |$))'

die() { echo "sh_record_video.sh: $*" >&2; exit 5; }
DRY=0; OVERWRITE=0; SMOKE=0; SEED=""; TOTAL=""; PREFIX=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --seed) [ $# -ge 2 ] || die "--seed needs a value"; SEED=$2; shift 2 ;;
    --total-s) [ $# -ge 2 ] || die "--total-s needs a value"; TOTAL=$2; shift 2 ;;
    --prefix) [ $# -ge 2 ] || die "--prefix needs a value"; PREFIX=$2; shift 2 ;;
    --) shift; break ;;
    *) die "unknown option $1 (recorder args go after --)" ;;
  esac
done
EXTRA=("$@")
[ "$SMOKE" = 1 ] && [ -n "$SEED" ] && die "--smoke and --seed are exclusive (the smoke is a fixed plan)"
if [ -n "$SEED" ]; then
  case "$SEED" in *[!0-9]*) die "--seed must be a non-negative integer, got '$SEED'" ;; esac
  PLAN=random; TOTAL=${TOTAL:-150}; PREFIX=${PREFIX:-sh_random_s$SEED}; TMO=${SH_VIDEO_TIMEOUT:-5400}
  case "$TOTAL" in ''|*[!0-9.]*) die "--total-s must be a positive number, got '$TOTAL'" ;; esac
  GEN_ARGS=(--plan random --seed "$SEED" --total-s "$TOTAL" --out-dir "$VID")
else
  [ -n "$TOTAL" ] && die "--total-s applies to the random plan only (--seed N)"
  if [ "$SMOKE" = 1 ]; then
    PLAN=smoke; PREFIX=${PREFIX:-sh_smoke}; TMO=${SH_VIDEO_TIMEOUT:-1800}
  else
    PLAN=raise; PREFIX=${PREFIX:-sh_raise}; TMO=${SH_VIDEO_TIMEOUT:-5400}
  fi
  GEN_ARGS=(--plan "$PLAN" --out-dir "$VID")
fi
YAML=$VID/scenarios_sh_video_$PLAN.yaml; ROW=sh_${PLAN}__omni
case "$PREFIX" in ''|*[!A-Za-z0-9_.-]*) die "--prefix must be [A-Za-z0-9_.-]+, got '$PREFIX'" ;; esac
case "$TMO" in ''|*[!0-9]*|0) die "SH_VIDEO_TIMEOUT must be a positive integer of seconds, got '$TMO'" ;; esac
case "$OUT" in *[[:space:]*?\[\]]*) die "output dir has whitespace / glob characters: $OUT" ;; esac
for a in "${EXTRA[@]}"; do
  case "$a" in
    --manifest|--manifest=*|--sequence|--sequence=*|--out-dir|--out-dir=*|--prefix|--prefix=*)
      die "'$a' is set by this launcher" ;;
  esac
done
[ -f "$GEN" ] && [ -f "$REC" ] || die "missing $GEN or $REC"
[ -x "$PY" ] || die "no python at $PY"
MP4=$OUT/${PREFIX}_sequence.mp4
if [ -e "$MP4" ] && [ "$OVERWRITE" != 1 ] && [ "$DRY" != 1 ]; then
  die "$MP4 exists (pass --overwrite, or a new --prefix)"
fi

busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
BUSY=0; busy && BUSY=1
if [ "$DRY" != 1 ] && [ "$BUSY" = 1 ]; then
  echo "sh_record_video.sh: another Isaac process is running (one at a time):" >&2
  ps -ww -eo pid,args | grep -E "^ *[0-9]+ ${ISAAC_PAT#^}" | grep -vE "$NONKIT_PAT" >&2
  exit 3
fi

DDC_GB=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1)
DDC_GB=${DDC_GB:-0}
if [ "$DDC_GB" -gt "$DDC_MAX_GB" ]; then
  die "Kit DerivedDataCache is ${DDC_GB} GB (> ${DDC_MAX_GB}): check the kit logs for omni.datastore migration loops and quarantine it first"
fi

if [ "$DRY" = 1 ]; then
  CUDA_VISIBLE_DEVICES="" "$PY" -B "$GEN" "${GEN_ARGS[@]}" --dry-run || die "schedule generation failed"
else
  CUDA_VISIBLE_DEVICES="" "$PY" -B "$GEN" "${GEN_ARGS[@]}" || die "schedule generation failed"
fi

UNIT="sh-video-${PREFIX//[^A-Za-z0-9_-]/_}-$(date +%s)"
LOG=$OUT/$PREFIX.log
CMD=(systemd-run --user --scope -q "--unit=$UNIT" -p "MemoryMax=$MEM" -p MemorySwapMax=0
     timeout --signal=TERM "$TMO" "$PY" "$REC" --headless --enable_cameras
     --manifest "$YAML" --sequence "$ROW" --out-dir "$OUT" --prefix "$PREFIX" "${EXTRA[@]}")
if [ "$DRY" = 1 ]; then
  echo "# isaac busy: $([ "$BUSY" = 1 ] && echo 'YES (a real launch refuses, exit 3)' || echo no)"
  echo "export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1"
  printf '%q ' "${CMD[@]}"; echo
  echo "# log: $LOG  pid: $OUT/$PREFIX.pid  unit: $UNIT.scope  DerivedDataCache: ${DDC_GB} GB  timeout: ${TMO} s (SIGTERM only; stop by hand: systemctl --user stop $UNIT.scope)"
  exit 0
fi

export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 TERM=${TERM:-xterm}
mkdir -p "$OUT"
echo "$UNIT" > "$OUT/$PREFIX.unit"
rm -f "$OUT/$PREFIX.pid"
{ echo "[$(date -Is)] sh_record_video.sh plan=$PLAN seed=${SEED:-none} total_s=${TOTAL:-plan} unit=$UNIT.scope DerivedDataCache=${DDC_GB}GB"
  printf '%q ' "${CMD[@]}"; echo; } >> "$LOG"
END="SHVIDEO END $PREFIX unit=$UNIT.scope"
setsid nohup bash -c '"$@" >> "$0" 2>&1 < /dev/null; rc=$?; echo "[$(date -Is)] '"$END"' rc=$rc" >> "$0"' \
  "$LOG" "${CMD[@]}" > /dev/null 2>&1 < /dev/null &
WRAP=$!
# return only once THIS recorder's python is visible (an Isaac launch started right after must see it busy)
mine() { ps -ww -eo pid,args | grep -E "^ *[0-9]+ ${ISAAC_PAT#^}" | grep -E -- "--prefix $PREFIX( |\$)" | awk '{print $1}' | head -1; }
waited=0
PID=$(mine)
until [ -n "$PID" ]; do
  if grep -qF "$END " "$LOG" 2>/dev/null; then
    echo "sh_record_video.sh: the recorder ended before it became visible: $(grep -F "$END " "$LOG" | tail -1); see $LOG" >&2
    exit 7
  fi
  if [ "$waited" -ge "$UP_WAIT" ]; then
    echo "sh_record_video.sh: the recorder is not visible after ${UP_WAIT} s (wrapper pid=$WRAP unit=$UNIT.scope); check $LOG" >&2
    exit 7
  fi
  sleep 1; waited=$((waited + 1)); PID=$(mine)
done
echo "$PID" > "$OUT/$PREFIX.pid"
echo "sh_record_video.sh: detached recorder pid=$PID unit=$UNIT.scope log=$LOG mp4=$MP4 (visible after ${waited} s)"
echo "sh_record_video.sh: done when '$END rc=...' appears in the log; sanity -> $OUT/${PREFIX}_sequence_sanity.json"

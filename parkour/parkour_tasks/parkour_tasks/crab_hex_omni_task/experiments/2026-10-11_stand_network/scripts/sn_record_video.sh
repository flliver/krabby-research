#!/usr/bin/env bash
# Stand-network campaign -- generate the random-command schedule and record its video (VIDEO ONLY; read by no gate).
# The recorder is an Isaac process: one at a time, under a memory cap, detached (memories: video memory hazard,
# Kit DerivedDataCache corruption, GPU queue chaining).
#
#   sn_record_video.sh [--dry-run] [--overwrite] [--seed N] [--total-s S] [--prefix P] [--smoke] [-- <recorder args>]
#     --seed N        the schedule seed (default 20261013)
#     --total-s S     video length in s = prologue + holds (default 180; --smoke: 20)
#     --prefix P      output prefix (default sn_random_s<seed>; --smoke: sn_smoke_s<seed>)
#     --smoke         a 20-s schedule (settle, ONE move_turn, one stand -- the stand-after-turn path) written to
#                     videos/scenarios_sn_video_smoke.yaml, row sn_random_smoke__omni
#     --overwrite     allow replacing an existing <prefix>_sequence.mp4
#     --dry-run       print the schedule (nothing written), the environment and the command; launch nothing
#     -- <args>       passed to sn_record_video.py (e.g. --no-arrows, --no-captions, --stand-rule slot)
#
# What it does:
#   1. refuses (exit 3) while ANY Isaac python runs -- evals (*eval_crab_hex_gait.py), train / play / record / demo
#      scripts, run_phases / run_candidates, sn_dagger.py (not its Kit-free --read / --emit-env) -- matched on
#      `ps -ww -eo args` with an anchored, bracketed pattern (never `pgrep -f`: it would match its own command line);
#   2. refuses (exit 5) when the Kit DerivedDataCache exceeds SN_DDC_MAX_GB (default 80 GB: the corruption precedent);
#   3. writes the schedule (`sn_video_schedule.py`, CPU only): videos/scenarios_sn_video[_smoke].yaml + the JSON;
#   4. runs `systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 timeout --signal=TERM -k <K> <T>
#      <isaac python> sn_record_video.py --headless --enable_cameras --manifest ... --sequence ... --out-dir $OUT
#      --prefix P` with OMNI_KIT_ACCEPT_EULA=yes, PYTHONUNBUFFERED=1, detached via `setsid nohup`; returns once the
#      recorder python is visible (its pid -> $OUT/P.pid, the scope unit -> $OUT/P.unit, the log -> $OUT/P.log).
#      SN_VIDEO_TIMEOUT seconds (default 5400; --smoke 1800) bounds the run with SIGTERM (the recorder unwinds:
#      the mp4 writer closes, Kit closes). The recorder turns SIGTERM into a Python SystemExit, which runs only once
#      control returns from Kit / PhysX C++ code, so a Kit hung INSIDE a render or step never unwinds: SIGKILL
#      follows SN_VIDEO_KILL_AFTER seconds later (default 120) to free the scope and the GPU (the mp4 is lost then
#      either way). STOP a run by hand with `systemctl --user stop <unit>.scope`, never with a SIGKILL of your own.
# Exit codes: 0 launched (or dry run), 3 busy, 5 usage / refusal, 7 the recorder ended before becoming visible or
# never became visible (read the log).
#
# NOTE 2026-10-26 (comment only; the code below is unchanged): the busy check's ISAAC_PAT predates sn_table_test.py and
# the successor stand-height campaign's scripts (2026-10-13_stand_height/scripts: sh_dagger.py, sh_table_test.py,
# sh_record_video.py, ...) and does NOT see them, so this launcher would start a second Kit beside a running table test,
# stand-height trainer or video (the DerivedDataCache-corruption scenario). Before launching, check by hand that no
# Isaac python runs (`ps -ww -eo args | grep '[i]saac_venv/bin/python'`), or read the "# isaac busy:" line of
# `2026-10-13_stand_height/scripts/sh_record_video.sh --dry-run`, whose pattern covers both campaigns.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)  # this checkout's root, from the script's own location
PY=${SN_PY:-/home/nickmagus/krabby/isaac_venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
GEN=$HERE/sn_video_schedule.py
REC=$HERE/sn_record_video.py
VID=$(cd "$HERE/.." && pwd)/videos
OUT=${SN_VIDEO_OUT:-$R/parkour/logs/videos/sn_stand_network}
MEM=${SN_VIDEO_MEMORY_MAX:-24G}
DDC=${SN_DDC:-/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache}
DDC_MAX_GB=${SN_DDC_MAX_GB:-80}
UP_WAIT=${SN_UP_WAIT:-120}
KILL_AFTER=${SN_VIDEO_KILL_AFTER:-120}
# the bracketed [i] keeps the pattern from matching a grep / shell line that quotes it
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|evaluation|rsl_rl/train|train|play|play_stream_video|demo|demo_crab_hex_student|record_categories|sn_record_video|sn_dagger|harvest_omni_rsi_bank|p2_smoke|zero_agent|verify_crab_[a-z_]+|debug_actuator_resolution|training_timeline_probe|crab_hex_phase_cfg_dump)\.py'
NONKIT_PAT='sn_dagger\.py .*--(read|emit-env)([ =]|$)'

die() { echo "sn_record_video.sh: $*" >&2; exit 5; }
DRY=0; OVERWRITE=0; SMOKE=0; SEED=20261013; TOTAL=""; PREFIX=""
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
case "$SEED" in ''|*[!0-9]*) die "--seed must be a non-negative integer, got '$SEED'" ;; esac
if [ "$SMOKE" = 1 ]; then
  TOTAL=${TOTAL:-20}; PREFIX=${PREFIX:-sn_smoke_s$SEED}; TAG=(--tag smoke --kinds move_turn)
  YAML=$VID/scenarios_sn_video_smoke.yaml; ROW=sn_random_smoke__omni; TMO=${SN_VIDEO_TIMEOUT:-1800}
else
  TOTAL=${TOTAL:-180}; PREFIX=${PREFIX:-sn_random_s$SEED}; TAG=()
  YAML=$VID/scenarios_sn_video.yaml; ROW=sn_random__omni; TMO=${SN_VIDEO_TIMEOUT:-5400}
fi
case "$TOTAL" in ''|*[!0-9.]*) die "--total-s must be a positive number, got '$TOTAL'" ;; esac
case "$PREFIX" in ''|*[!A-Za-z0-9_.-]*) die "--prefix must be [A-Za-z0-9_.-]+, got '$PREFIX'" ;; esac
case "$TMO" in ''|*[!0-9]*|0) die "SN_VIDEO_TIMEOUT must be a positive integer of seconds, got '$TMO'" ;; esac
case "$KILL_AFTER" in ''|*[!0-9]*|0) die "SN_VIDEO_KILL_AFTER must be a positive integer of seconds, got '$KILL_AFTER'" ;; esac
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
if [ "$DRY" != 1 ] && busy; then
  echo "sn_record_video.sh: another Isaac process is running (one at a time):" >&2
  ps -ww -eo pid,args | grep -E "^ *[0-9]+ ${ISAAC_PAT#^}" | grep -vE "$NONKIT_PAT" >&2
  exit 3
fi

DDC_GB=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1)
DDC_GB=${DDC_GB:-0}
if [ "$DDC_GB" -gt "$DDC_MAX_GB" ]; then
  die "Kit DerivedDataCache is ${DDC_GB} GB (> ${DDC_MAX_GB}): check the kit logs for omni.datastore migration loops and quarantine it first"
fi

GEN_ARGS=(--seed "$SEED" --total-s "$TOTAL" "${TAG[@]}" --out-dir "$VID")
if [ "$DRY" = 1 ]; then
  CUDA_VISIBLE_DEVICES="" "$PY" -B "$GEN" "${GEN_ARGS[@]}" --dry-run || die "schedule generation failed"
else
  CUDA_VISIBLE_DEVICES="" "$PY" -B "$GEN" "${GEN_ARGS[@]}" || die "schedule generation failed"
fi

UNIT="sn-video-${PREFIX//[^A-Za-z0-9_-]/_}-$(date +%s)"
LOG=$OUT/$PREFIX.log
CMD=(systemd-run --user --scope -q "--unit=$UNIT" -p "MemoryMax=$MEM" -p MemorySwapMax=0
     timeout --signal=TERM -k "$KILL_AFTER" "$TMO" "$PY" "$REC" --headless --enable_cameras
     --manifest "$YAML" --sequence "$ROW" --out-dir "$OUT" --prefix "$PREFIX" "${EXTRA[@]}")
if [ "$DRY" = 1 ]; then
  echo "export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1"
  printf '%q ' "${CMD[@]}"; echo
  echo "# log: $LOG  pid: $OUT/$PREFIX.pid  unit: $UNIT.scope  DerivedDataCache: ${DDC_GB} GB  timeout: ${TMO} s (SIGKILL ${KILL_AFTER} s after the TERM)"
  exit 0
fi

export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 TERM=${TERM:-xterm}
mkdir -p "$OUT"
echo "$UNIT" > "$OUT/$PREFIX.unit"
rm -f "$OUT/$PREFIX.pid"
{ echo "[$(date -Is)] sn_record_video.sh seed=$SEED total_s=$TOTAL smoke=$SMOKE unit=$UNIT.scope DerivedDataCache=${DDC_GB}GB"
  printf '%q ' "${CMD[@]}"; echo; } >> "$LOG"
END="SNVIDEO END $PREFIX unit=$UNIT.scope"
setsid nohup bash -c '"$@" >> "$0" 2>&1 < /dev/null; rc=$?; echo "[$(date -Is)] '"$END"' rc=$rc" >> "$0"' \
  "$LOG" "${CMD[@]}" > /dev/null 2>&1 < /dev/null &
WRAP=$!
# return only once THIS recorder's python is visible (an Isaac launch started right after must see it busy)
mine() { ps -ww -eo pid,args | grep -E "^ *[0-9]+ ${ISAAC_PAT#^}" | grep -E -- "--prefix $PREFIX( |\$)" | awk '{print $1}' | head -1; }
waited=0
PID=$(mine)
until [ -n "$PID" ]; do
  if grep -qF "$END " "$LOG" 2>/dev/null; then
    echo "sn_record_video.sh: the recorder ended before it became visible: $(grep -F "$END " "$LOG" | tail -1); see $LOG" >&2
    exit 7
  fi
  if [ "$waited" -ge "$UP_WAIT" ]; then
    echo "sn_record_video.sh: the recorder is not visible after ${UP_WAIT} s (wrapper pid=$WRAP unit=$UNIT.scope); check $LOG" >&2
    exit 7
  fi
  sleep 1; waited=$((waited + 1)); PID=$(mine)
done
echo "$PID" > "$OUT/$PREFIX.pid"
echo "sn_record_video.sh: detached recorder pid=$PID unit=$UNIT.scope log=$LOG mp4=$MP4 (visible after ${waited} s)"
echo "sn_record_video.sh: done when '$END rc=...' appears in the log; sanity -> $OUT/${PREFIX}_sequence_sanity.json"

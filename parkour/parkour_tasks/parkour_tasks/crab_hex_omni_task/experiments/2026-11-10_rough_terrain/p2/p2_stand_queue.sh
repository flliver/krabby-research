#!/bin/bash
# P2 step 9 (p2/P2.md): the stand re-pair queue -- ONE serial runner over `p2_stand.py step S` (rsd_stand_queue.sh
# re-pointed: the queue log, STOP_QUEUE and locks under <MAIN>/parkour/outputs/omni_plan/rough_terrain_p2/stand,
# the records under results/p2/stand; the walker is c1, the names REFS/SN/SH/SH_RH _RT_6d1f29be).
#
#   bash p2_stand_queue.sh --dry [STEP ...]   print the queue (steps, commands, roots, guards, locks) and run
#                                             `p2_stand.py dry`: READS ONLY -- no lock probed, nothing written or
#                                             launched; exit = p2_stand.py dry's (0 OK / 1 BAD) or 2 (usage)
#   bash p2_stand_queue.sh [STEP ...]         start the runner DETACHED (setsid nohup); prints P2STQ_LAUNCH
#   bash p2_stand_queue.sh --run [STEP ...]   the foreground runner (what the detached launch runs)
#   STEP = preflight refs sn_train sn_rows sn_read sh_train sh_table sh_rows rh_rows sh_read bh
#   (default: preflight refs sn_train sn_rows sn_read sh_train sh_table sh_rows rh_rows sh_read)
#
# The runner stops at the first step with rc != 0 (4 = a decide-record-notify stop: a gate HALT the rules do not
# cover, a premise STOP, a guard; 1 = infrastructure), or before a step when a STOP_QUEUE file exists:
#   <OUT>/STOP_QUEUE, results/p2/STOP_QUEUE, results/STOP_QUEUE.
# Every step resumes (finished rows / runs / reads are skipped), so a re-launch with the remaining steps continues.
#
# Guards:
#   * one stand queue: flock -n <OUT>/p2_stand_queue.lock (refused when held: "P2STQ REFUSED ...");
#   * the DerivedDataCache: REFUSED above P2STQ_DDC_MAX_GB (78; an override may only lower it) at the start, and
#     p2_stand.py's guard repeats it (with >= 30 GB free, the results/p2/pin.json build check and the Isaac-busy wait)
#     before every Isaac launch. Moving the cache aside (renamed, never deleted) is the PARENT's job (P2.md step 9,
#     settled 6d): the queue only refuses and reports the size;
#   * one Isaac process at a time: the runner waits for, then holds, the campaign's .isaac.lock for its whole run
#     (the lock the distill and every rt_queue block take), so no P2 / RT block starts mid-stand; inside, every
#     launcher also waits while any Isaac python runs (ps -ww patterns, never pgrep -f).
# Stops are SIGTERM only (p2_stand.py / rsd_stand.train: SIGTERM, 120 s, SIGTERM, 120 s; never SIGKILL).
#
# Sentinels in <OUT>/queue.log (column 0):
#   P2STQ BEGIN | LOCK | WAIT | START <s> | END <s> rc=N | STOP <s> (...) | REFUSED (...) | DONE
#   P2ST <step> END rc=N ... (each step's own last line)   P2ST NOTIFY ...   P2ST BODY_HEIGHT ...   P2ST PREFLIGHT ...
# Watch with a log-sentinel watcher on "^P2STQ (DONE|STOP|REFUSED)" and "^P2ST NOTIFY" (never pgrep).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAMP=$(cd "$HERE/.." && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
STAND_PY=$HERE/p2_stand.py
OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rough_terrain_p2/stand
Q=$OUT/queue.log
QLOCK=$OUT/p2_stand_queue.lock
ISAAC_LOCK=$CAMP/.isaac.lock
STOPS=("$OUT/STOP_QUEUE" "$CAMP/results/p2/STOP_QUEUE" "$CAMP/results/STOP_QUEUE")
DDC=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache
DDC_MAX_GB=78
if [ -n "${P2STQ_DDC_MAX_GB:-}" ]; then
  case "$P2STQ_DDC_MAX_GB" in *[!0-9]*) echo "P2STQ REFUSED P2STQ_DDC_MAX_GB must be an integer" >&2; exit 2 ;; esac
  [ "$P2STQ_DDC_MAX_GB" -lt "$DDC_MAX_GB" ] && DDC_MAX_GB=$P2STQ_DDC_MAX_GB
fi
FREE_MIN_GB=30
POLL_S=30
DEFAULT_STEPS=(preflight refs sn_train sn_rows sn_read sh_train sh_table sh_rows rh_rows sh_read)
VALID=" preflight refs sn_train sn_rows sn_read sh_train sh_table sh_rows rh_rows sh_read bh "

MODE=launch
case "${1:-}" in
  --dry) MODE=dry; shift ;;
  --run) MODE=run; shift ;;
  -h|--help) awk 'NR > 1 && /^set -u/ {exit} NR > 1 {print}' "$0"; exit 0 ;;
esac
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=("${DEFAULT_STEPS[@]}")
for s in "${STEPS[@]}"; do
  case "$VALID" in *" $s "*) ;; *) echo "P2STQ REFUSED unknown step '$s' (valid:$VALID)" >&2; exit 2 ;; esac
done

ddc_gb() { local g; g=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); echo "${g:-0}"; }
free_gb() { local p=$1; while [ ! -e "$p" ]; do p=$(dirname "$p"); done; df --output=avail -B1G "$p" | tail -1 | tr -d ' '; }
stop_hit() { local f; for f in "${STOPS[@]}"; do [ -e "$f" ] && { echo "$f"; return 0; }; done; return 1; }
disk_refusal() {  # -> prints the refusal, exit 0 when refused
  local d f
  d=$(ddc_gb); f=$(free_gb "$OUT")
  if [ "$d" -gt "$DDC_MAX_GB" ]; then
    echo "DerivedDataCache ${d} GB > ${DDC_MAX_GB} GB at $DDC: move it aside first (renamed, never deleted; the parent's job, P2.md step 9)"
    return 0
  fi
  if [ "$f" -lt "$FREE_MIN_GB" ]; then echo "only ${f} GB free (floor ${FREE_MIN_GB} GB)"; return 0; fi
  return 1
}

if [ "$MODE" = dry ]; then
  echo "P2STQ DRY queue (serial, stop at the first rc != 0): ${STEPS[*]}"
  for s in "${STEPS[@]}"; do echo "P2STQ DRY step $s: env -u PYTHONPATH $PY -I -B $STAND_PY step $s >> $Q 2>&1"; done
  echo "P2STQ DRY queue log $Q; STOP files ${STOPS[*]}"
  echo "P2STQ DRY locks (NOT probed in --dry): flock -n $QLOCK (one stand queue), then wait for + hold flock $ISAAC_LOCK"
  echo "P2STQ DRY DerivedDataCache $(ddc_gb) GB (refuse > ${DDC_MAX_GB}); free $(free_gb "$OUT") GB (floor ${FREE_MIN_GB})"
  if r=$(disk_refusal); then echo "P2STQ DRY WOULD-REFUSE $r"; fi
  if sf=$(stop_hit); then echo "P2STQ DRY WOULD-STOP $sf exists"; fi
  for f in "$STAND_PY" "$PY"; do [ -e "$f" ] || { echo "P2STQ DRY BAD missing $f"; exit 1; }; done
  bash -n "$0" && echo "P2STQ DRY bash -n OK"
  env -u PYTHONPATH "$PY" -I -B "$STAND_PY" dry
  exit $?
fi

if [ "$MODE" = launch ]; then
  for f in "$STAND_PY" "$PY"; do [ -e "$f" ] || { echo "P2STQ REFUSED missing $f" >&2; exit 2; }; done
  if r=$(disk_refusal); then echo "P2STQ REFUSED $r" >&2; exit 5; fi
  if sf=$(stop_hit); then echo "P2STQ REFUSED $sf exists" >&2; exit 9; fi
  mkdir -p "$OUT"
  setsid nohup bash "$0" --run "${STEPS[@]}" >> "$OUT/queue.nohup.out" 2>&1 < /dev/null &
  echo "P2STQ_LAUNCH pid=$! steps=${STEPS[*]} log=$Q (watch: ^P2STQ (DONE|STOP|REFUSED), ^P2ST NOTIFY)"
  exit 0
fi

# ------------------------------------------------------------------------------------------------- the runner
mkdir -p "$OUT"
exec 8>>"$QLOCK"
if ! flock -n 8; then echo "P2STQ REFUSED another stand queue holds $QLOCK $(date '+%F %T')" >> "$Q"; exit 3; fi
echo "P2STQ BEGIN $(date '+%F %T') pid=$$ steps: ${STEPS[*]}" >> "$Q"
if r=$(disk_refusal); then echo "P2STQ REFUSED $r $(date '+%F %T')" >> "$Q"; exit 5; fi
echo "P2STQ DDC $(ddc_gb) GB (guard ${DDC_MAX_GB}); free $(free_gb "$OUT") GB" >> "$Q"
exec 9>>"$ISAAC_LOCK"
waited=0
until flock -n 9; do
  [ "$waited" = 0 ] && echo "P2STQ WAIT $ISAAC_LOCK is held (another P2 / RT queue or the distill) $(date '+%F %T')" >> "$Q"
  waited=1
  if sf=$(stop_hit); then echo "P2STQ STOP (STOP_QUEUE $sf while waiting for the Isaac lock) $(date '+%F %T')" >> "$Q"; exit 9; fi
  sleep "$POLL_S"
done
echo "P2STQ LOCK $ISAAC_LOCK held $(date '+%F %T')" >> "$Q"
for s in "${STEPS[@]}"; do
  if sf=$(stop_hit); then echo "P2STQ STOP $s (STOP_QUEUE $sf) $(date '+%F %T')" >> "$Q"; exit 9; fi
  echo "P2STQ START $s $(date '+%F %T') ddc=$(ddc_gb)GB" >> "$Q"
  env -u PYTHONPATH "$PY" -I -B "$STAND_PY" step "$s" >> "$Q" 2>&1
  rc=$?
  echo "P2STQ END $s rc=$rc $(date '+%F %T')" >> "$Q"
  if [ "$rc" != 0 ]; then echo "P2STQ STOP $s rc=$rc $(date '+%F %T')" >> "$Q"; exit "$rc"; fi
done
echo "P2STQ DONE $(date '+%F %T')" >> "$Q"

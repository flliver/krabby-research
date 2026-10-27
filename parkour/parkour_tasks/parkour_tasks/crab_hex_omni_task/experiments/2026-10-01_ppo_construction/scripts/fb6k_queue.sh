#!/usr/bin/env bash
# FB6k + INC0 queue (CHARTER 11.6): INC0 block -> the INC0 attribution gate -> the FB6k block (drawn order) ->
# the extension decision -> the x9k forks. Serial, one Isaac process at a time (each block is one run_arm.sh); every
# record goes to $OUT/arms_fb6k (verdict.py's step-3 dir and x6k's stay untouched). The INC0 gate STOPS before any
# FB6k GPU is spent when INC0 is FAST/FAST in >= 2 of 3 (or unreadable): that is a stop-and-ask (CHARTER 11.5).
# STOP_QUEUE (written by run_arm.sh on a wiring/dose kill, or by hand) stops between blocks.
#
# Usage (detached): setsid nohup bash scripts/fb6k_queue.sh > $OUT/fb6k_queue.log 2>&1 < /dev/null &
C=$(cd "$(dirname "$0")/.." && pwd)
S=$C/scripts
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=$R/parkour/outputs/omni_plan/ppo_construction
LOG=$OUT/queue.log
A=$OUT/arms_fb6k
INC0="C_lr6k__INC0_s11 C_lr6k__INC0_s12 C_lr6k__INC0_s13"
#: drawn once (CHARTER 11.2): random.Random(20261004), seed-block-major, B/C0 shuffled within each seed
FB="FB6k__C0_s16 FB6k__B_s16 FB6k__C0_s17 FB6k__B_s17 FB6k__B_s18 FB6k__C0_s18"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
stopped() { [ -f "$OUT/STOP_QUEUE" ]; }
block_ok() {  # RC NAME -> continue only when run_arm.sh ended normally (0); 4 = STOP_QUEUE, 6 = launch refused
  if [ "$1" != 0 ]; then say "STOP fb6k_queue: run_arm.sh rc $1 in the $2 block -- NOTIFY"; exit 4; fi
  stopped && { say "STOP fb6k_queue: STOP_QUEUE after the $2 block -- NOTIFY"; exit 4; }
  return 0
}
mkdir -p "$A"

say "fb6k_queue start (CHARTER 11): INC0 [$INC0] -> INC0 gate -> FB6k [$FB] -> extensions"
stopped && { say "STOP fb6k_queue: STOP_QUEUE present before the INC0 block"; exit 4; }
ARMS_DIR=$A EXTRA_LR_EVALS="3000 4500" ACCEPT_WHY="FB6k + INC0 (CHARTER 11): candidates v5" \
  bash "$S/run_arm.sh" $INC0
block_ok $? INC0

g=$("$PY" "$S/fb6k_read.py" --arms "$A" --inc0-gate 2>&1); rc=$?
say "INC0GATE rc $rc :: $(echo "$g" | tail -1 | cut -c1-500)"
if [ "$rc" != 0 ]; then
  say "STOP fb6k_queue: the INC0 gate (CHARTER 11.5) stops before FB6k -- NOTIFY stop and ask"
  exit 4
fi

# ADVISE_DEAD=1: the DEAD-at-1000 kill is uncalibrated on the FB pool (CHARTER 11.3): logged, never stops a run
ADVISE_DEAD=1 ARMS_DIR=$A EXTRA_LR_EVALS="3000 4500" ACCEPT_WHY="" bash "$S/run_arm.sh" $FB
block_ok $? FB6k

out=$("$PY" "$S/fb6k_read.py" --arms "$A" --extensions 2>&1); rc=$?
if [ "$rc" != 0 ]; then say "STOP fb6k_queue: fb6k_read.py --extensions rc $rc -- NOTIFY :: $(echo "$out" | tail -1 | cut -c1-300)"; exit 4; fi
ext=$(echo "$out" | tail -1)
say "EXTENSIONS (CHARTER 11.4): ${ext:-none}"
if [ -n "$ext" ]; then
  ARMS_DIR=$A FORK_SRC_ARMS=$A RESUME_READ=1 EXTRA_LR_EVALS="6100 7500" ACCEPT_WHY="" bash "$S/run_arm.sh" $ext
  block_ok $? x9k
fi
say "fb6k_queue end -- read with scripts/fb6k_read.py (CHARTER 11.4)"

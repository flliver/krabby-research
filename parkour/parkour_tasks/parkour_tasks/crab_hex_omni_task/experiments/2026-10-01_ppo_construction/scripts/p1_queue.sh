#!/usr/bin/env bash
# P1-B6 queue (CHARTER 12.6): t3 block (6000 its from scratch, drawn order) -> the t3 stop gate -> the tail block
# (c4 -> c5 -> c6 -> c6x, each tail forking its seed's t3 head) -> the c6 pre-cut both-hands reads -> W's re-score.
# Serial, one Isaac process at a time (every step is one run_arm.sh); records in $OUT/arms_p1. The t3 gate STOPS before
# the ~25 GPU-h tail block when formation fails once turning shares the pool (CHARTER 12.4). Any run_arm.sh or reader
# error stops the queue with a NOTIFY line; STOP_QUEUE (a wiring/dose kill, or by hand) stops between steps.
#
# Usage (detached): setsid nohup bash scripts/p1_queue.sh > $OUT/p1_queue.log 2>&1 < /dev/null &
C=$(cd "$(dirname "$0")/.." && pwd)
S=$C/scripts
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=$R/parkour/outputs/omni_plan/ppo_construction
LOG=$OUT/queue.log
A=$OUT/arms_p1
#: drawn once (CHARTER 12.2): random.Random(20261005), seed-block-major, B/C0 shuffled within each seed
T3="P1B6__C0_t3_s16 P1B6__B_t3_s16 P1B6__C0_t3_s17 P1B6__B_t3_s17 P1B6__C0_t3_s18 P1B6__B_t3_s18"
TAILS="P1B6__C0_tail_s16 P1B6__B_tail_s16 P1B6__C0_tail_s17 P1B6__B_tail_s17 P1B6__C0_tail_s18 P1B6__B_tail_s18"
#: W (the omni phase-1 head of record, clock-free) and the env of its own suite stamp (clock hidden, as trained)
W_CK=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt
W_ST=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29
SUITE_MAN=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
stopped() { [ -f "$OUT/STOP_QUEUE" ]; }
block_ok() {  # RC NAME -> continue only when the step ended normally (0)
  if [ "$1" != 0 ]; then say "STOP p1_queue: rc $1 in the $2 step -- NOTIFY"; exit 4; fi
  stopped && { say "STOP p1_queue: STOP_QUEUE after the $2 step -- NOTIFY"; exit 4; }
  return 0
}
mkdir -p "$A"

say "p1_queue start (CHARTER 12): t3 [$T3] -> t3 gate -> tails [$TAILS] -> c6 pre-cut reads -> W re-score"
stopped && { say "STOP p1_queue: STOP_QUEUE present before the t3 block"; exit 4; }
# ADVISE_DEAD=1: the DEAD-at-1000 kill is uncalibrated on this pool (CHARTER 12.3, as 11.3)
ADVISE_DEAD=1 ARMS_DIR=$A EXTRA_LR_EVALS="4500" ACCEPT_WHY="P1-B6 (CHARTER 12): candidates v6" \
  bash "$S/run_arm.sh" $T3
block_ok $? t3

g=$("$PY" "$S/p1_read.py" --arms "$A" --t3-gate 2>&1); rc=$?
say "T3GATE rc $rc :: $(echo "$g" | tail -1 | cut -c1-500)"
if [ "$rc" != 0 ]; then say "STOP p1_queue: the t3 gate (CHARTER 12.4) stops before the tails -- NOTIFY stop and ask"; exit 4; fi

STOP_ON_REF_STOP=1 ARMS_DIR=$A FORK_SRC_ARMS=$A RESUME_READ=1 EXTRA_LR_EVALS="13000" ACCEPT_WHY="" bash "$S/run_arm.sh" $TAILS
block_ok $? tail

for ref in $TAILS; do
  out=$("$PY" "$S/p1_read.py" --c6 "$ref" 2>&1); rc=$?
  if [ "$rc" != 0 ]; then say "STOP p1_queue: no c6 head/stamp for $ref -- NOTIFY :: $(echo "$out" | tail -1 | cut -c1-300)"; exit 4; fi
  read -r ck st <<< "$(echo "$out" | tail -1)"
  ARMS_DIR=$A bash "$S/run_arm.sh" --lr-eval "${ref}_c6" "$ck" "$st"
  block_ok $? "c6 pre-cut $ref"
done

ARMS_DIR=$A bash "$S/run_arm.sh" --lr-eval W_p1b6 "$W_CK" "$W_ST"
block_ok $? "W lr"
ARMS_DIR=$A bash "$S/run_arm.sh" --row-eval W_p1b6 "$W_CK" "$W_ST" "$SUITE_MAN" suite__omni
block_ok $? "W suite"
say "p1_queue end -- read with scripts/p1_read.py (CHARTER 12.5)"

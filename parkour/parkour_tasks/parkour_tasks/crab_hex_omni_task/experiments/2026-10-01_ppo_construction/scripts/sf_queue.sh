#!/usr/bin/env bash
# "Run 1-3" queue (CHARTER 13.5): (1) turn-band diagnostics (evals only) -> (2) t3x forks of the stopped P1-B6 B t3 heads
# -> the t3x rule (records + notifies; never stops this queue) -> (3) side-first c4 (FB6k B heads forked into the P1 c4
# stage) -> the c4 gate -> side-first c5 -> c6 -> c6x -> c6 pre-cut + turn-band reads -> the C0 trigger -> (conditional)
# the C0 side-first tails. Serial, one Isaac process at a time; records in $OUT/arms_p1. Any run_arm.sh or reader error
# stops the queue with a NOTIFY line; STOP_QUEUE stops between steps.
#
# Usage (detached): setsid nohup bash scripts/sf_queue.sh > $OUT/sf_queue.log 2>&1 < /dev/null &
# Resume after a stop: FROM_STEP=turnband|t3x|c4|rest|c0 skips the earlier steps (finished references must not be relaunched:
# run_arm.sh would find no fresh head and stop).
C=$(cd "$(dirname "$0")/.." && pwd)
S=$C/scripts
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=$R/parkour/outputs/omni_plan/ppo_construction
LOG=$OUT/queue.log
A=$OUT/arms_p1
FB=$OUT/arms_fb6k
BAND_MAN=$C/scenarios_turn_band.yaml
SUITE_MAN=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml
W_CK=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt
W_ST=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29
T3="P1B6__C0_t3_s16 P1B6__B_t3_s16 P1B6__C0_t3_s17 P1B6__B_t3_s17 P1B6__C0_t3_s18 P1B6__B_t3_s18"
T3X="P1B6__B_t3x_s16 P1B6__B_t3x_s17 P1B6__B_t3x_s18 P1B6__C0_t3x_s16"
SFB4="SF__B_c4_s16 SF__B_c4_s17 SF__B_c4_s18"
SFBR="SF__B_rest_s16 SF__B_rest_s17 SF__B_rest_s18"
SFC4="SF__C0_c4_s16 SF__C0_c4_s17 SF__C0_c4_s18"
SFCR="SF__C0_rest_s16 SF__C0_rest_s17 SF__C0_rest_s18"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
stopped() { [ -f "$OUT/STOP_QUEUE" ]; }
block_ok() {  # RC NAME -> continue only when the step ended normally (0)
  if [ "$1" != 0 ]; then say "STOP sf_queue: rc $1 in the $2 step -- NOTIFY"; exit 4; fi
  stopped && { say "STOP sf_queue: STOP_QUEUE after the $2 step -- NOTIFY"; exit 4; }
  return 0
}
recf() {  # REF ARMS FIELD -> a field of that reference's record
  python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]) or '')" "$2/$1.record.json" "$3" 2>/dev/null
}
band() {  # NAME CHECKPOINT ENV_STAMP -> the turn-band row as <NAME>_tb
  ARMS_DIR=$A bash "$S/run_arm.sh" --row-eval "${1}_tb" "$2" "$3" "$BAND_MAN" turn_band__omni
  block_ok $? "turn band $1"
}
band_head() {  # REF ARMS -> the turn-band row on that reference's head with its own driver-stamp env
  local ck st; ck=$(recf "$1" "$2" checkpoint); st=$(recf "$1" "$2" driver_stamp)
  [ -n "$ck" ] && [ -n "$st" ] || { say "STOP sf_queue: no checkpoint/stamp for $1 -- NOTIFY"; exit 4; }
  band "$1" "$ck" "$st"
}
c6_reads() {  # REST_REF -> c6 pre-cut lr + turn band at c6 and c6x
  local out rc ck st
  out=$("$PY" "$S/p1_read.py" --c6 "$1" 2>&1); rc=$?
  if [ "$rc" != 0 ]; then say "STOP sf_queue: no c6 head/stamp for $1 -- NOTIFY :: $(echo "$out" | tail -1 | cut -c1-300)"; exit 4; fi
  read -r ck st <<< "$(echo "$out" | tail -1)"
  ARMS_DIR=$A bash "$S/run_arm.sh" --lr-eval "${1}_c6" "$ck" "$st"
  block_ok $? "c6 pre-cut $1"
  band "${1}_c6" "$ck" "$st"
  band_head "$1" "$A"
}
FROM_STEP=${FROM_STEP:-turnband}
step_n() { case "$1" in turnband) echo 1;; t3x) echo 2;; c4) echo 3;; rest) echo 4;; c0) echo 5;; *) echo 0;; esac; }
FROM=$(step_n "$FROM_STEP"); [ "$FROM" -ge 1 ] || { echo "bad FROM_STEP $FROM_STEP" >&2; exit 2; }
want() { [ "$(step_n "$1")" -ge "$FROM" ]; }
mkdir -p "$A"
say "sf_queue start (CHARTER 13): turn band -> t3x [$T3X] -> side-first c4 [$SFB4] -> c4 gate -> rest [$SFBR] -> C0 trigger"
stopped && { say "STOP sf_queue: STOP_QUEUE present at start"; exit 4; }

# (1) turn-band diagnostics: the six t3 heads, the FB6k B heads, W (lr + suite + band), sampled suites (diagnostic only)
if want turnband; then
for ref in $T3; do band_head "$ref" "$A"; done
for s in 16 17 18; do band_head "FB6k__B_s$s" "$FB"; done
ARMS_DIR=$A ACCEPT_WHY="" bash "$S/run_arm.sh" --lr-eval W_p1b6 "$W_CK" "$W_ST"; block_ok $? "W lr"
ARMS_DIR=$A bash "$S/run_arm.sh" --row-eval W_p1b6 "$W_CK" "$W_ST" "$SUITE_MAN" suite__omni; block_ok $? "W suite"
band W_p1b6 "$W_CK" "$W_ST"
for ref in P1B6__B_t3_s16 P1B6__B_t3_s17 P1B6__B_t3_s18 P1B6__C0_t3_s16; do
  EVAL_EXTRA_ARGS="--sample-actions" ARMS_DIR=$A bash "$S/run_arm.sh" --row-eval "${ref}_sampled" \
    "$(recf "$ref" "$A" checkpoint)" "$(recf "$ref" "$A" driver_stamp)" "$SUITE_MAN" suite__omni
  block_ok $? "sampled suite $ref"
done
say "TURNDIAG done (CHARTER 13.2) -- read with scripts/sf_read.py"
fi

# (2) t3x: +3000 on the stopped t3 heads (B s16-18, C0 s16)
if want t3x; then
STOP_ON_REF_STOP=1 ARMS_DIR=$A FORK_SRC_ARMS=$A RESUME_READ=1 EXTRA_LR_EVALS="6100 7500" \
  ACCEPT_WHY="t3x + side-first (CHARTER 13): candidates v7" bash "$S/run_arm.sh" $T3X
block_ok $? t3x
for ref in $T3X; do band_head "$ref" "$A"; done
g=$("$PY" "$S/sf_read.py" --t3x-rule 2>&1)
say "T3XRULE :: $(echo "$g" | tail -1 | cut -c1-500)"
case "$g" in *'"PLATEAU"'*) ;; *) say "NOTIFY t3x rule: $(echo "$g" | tail -1 | cut -c1-200) -- the user's call (CHARTER 13.3); the side-first block continues";; esac
fi

# (3) side-first: c4 from the FB6k B heads, then the c4 gate
if want c4; then
STOP_ON_REF_STOP=1 ARMS_DIR=$A FORK_SRC_ARMS=$FB RESUME_READ=1 ACCEPT_WHY="" bash "$S/run_arm.sh" $SFB4
block_ok $? "side-first c4"
g=$("$PY" "$S/sf_read.py" --c4-gate 2>&1); rc=$?
say "C4GATE rc $rc :: $(echo "$g" | tail -1 | cut -c1-400)"
if [ "$rc" != 0 ]; then say "STOP sf_queue: the c4 gate (CHARTER 13.4) -- NOTIFY stop and ask"; exit 4; fi
fi
if want rest; then
STOP_ON_REF_STOP=1 ARMS_DIR=$A FORK_SRC_ARMS=$A RESUME_READ=1 EXTRA_LR_EVALS="13000" ACCEPT_WHY="" bash "$S/run_arm.sh" $SFBR
block_ok $? "side-first rest"
for ref in $SFBR; do c6_reads "$ref"; done
fi

# the C0 side-first tails only if B's c6x is FAST/FAST (hands) in >= 2/3
g=$("$PY" "$S/sf_read.py" --c0-trigger 2>&1); rc=$?
say "C0TRIGGER rc $rc :: $(echo "$g" | tail -1 | cut -c1-300)"
if [ "$rc" = 0 ]; then
  STOP_ON_REF_STOP=1 ARMS_DIR=$A FORK_SRC_ARMS=$FB RESUME_READ=1 ACCEPT_WHY="" bash "$S/run_arm.sh" $SFC4
  block_ok $? "side-first C0 c4"
  STOP_ON_REF_STOP=1 ARMS_DIR=$A FORK_SRC_ARMS=$A RESUME_READ=1 EXTRA_LR_EVALS="13000" ACCEPT_WHY="" bash "$S/run_arm.sh" $SFCR
  block_ok $? "side-first C0 rest"
  for ref in $SFCR; do c6_reads "$ref"; done
elif [ "$rc" != 3 ]; then
  say "STOP sf_queue: sf_read --c0-trigger rc $rc -- NOTIFY"; exit 4
fi
say "sf_queue end -- read with scripts/sf_read.py (CHARTER 13.4)"

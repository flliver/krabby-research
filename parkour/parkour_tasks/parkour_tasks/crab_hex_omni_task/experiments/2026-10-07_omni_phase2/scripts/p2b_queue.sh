#!/usr/bin/env bash
# Stage p2b queue (P2B_PLAN.md / CHARTER 10): refs -> code-inertness -> the s18 screen (C0, CL, TG) -> STOP and ask
# -> the replicates (the kept levers at s16 then s17) -> the recipe read -> STOP and ask. Modelled on p2_queue.sh:
# serial, one Isaac process at a time (run_arm.sh's anchored idle check before every launch/eval), the same flock
# file ($OUT/p2_queue.lock -- a phase-2 queue and a p2b queue never run together either), the same say/STOP/NOTIFY
# conventions (the watcher push-notifies any line containing " NOTIFY " or " STOP "), and FROM_STEP resume.
#
#   refs       for seed in 18, 16, 17: REF_S1_s<seed>'s reads -- the seed's S1 p2a head (arms/P2__S1_s<seed>.record.json
#              checkpoint + driver_stamp), on the new (p2b) code: lr; suite__omni at env_seed 1 (s18 also at 2, the
#              suite band); field_light / ring_obst at env_seeds 1 and 2; field_hard at env_seeds 1-4 (n = 400);
#              field_turn / field_suite at env_seed 1.
#   inertness  p2_read.py --p2b-inertness (CHARTER 10 step 3: each REF must reproduce its p2a reads exactly on the new
#              code -- a deterministic eval with the knobs off). Anything but rc 0 STOPS and asks.
#   screen     P2__C0_s18, P2__CL_s18, P2__TG_s18 (run_arm.sh stops on any kill/refusal), each with the standard head
#              reads (field_hard at env_seeds 1-4) plus the intermediate reads at the p2b checkpoints (p2_live.py
#              --p2b-checkpoints, a fork of model_19994: 21000 / 22500 / 24000) -- then p2_read.py --p2b-screen and a
#              STOP: the user may prune P2B_LEVERS before the replicates. Resume with FROM_STEP=reps.
#   reps       the levers kept in $P2B_LEVERS (default "CL TG"), at s16 then s17: P2__<lever>_s16 for every lever,
#              then P2__<lever>_s17 for every lever, same reads as the screen.
#   recipe     p2_read.py --p2b-recipe (CHARTER 10 / P2B_PLAN.md step 6: the CHARTER 6.2 ladder per lever at 3 seeds)
#              -- NOTIFY the reading, then STOP: the head-of-record proposal (step 7) and the bake are the user's.
#
# Usage (detached): setsid nohup bash scripts/p2b_queue.sh > $OUT/p2b_queue.log 2>&1 < /dev/null &
# Resume after a stop: FROM_STEP=refs|inertness|screen|reps|recipe skips the earlier steps (a finished reference must
# never be relaunched: run_arm.sh would resume it, not fork it again).
set -u
C=$(cd "$(dirname "$0")/.." && pwd)
S=$C/scripts
R=/home/nickmagus/krabby/krabby-research
PY=${P2_PY:-/home/nickmagus/krabby/isaac_venv/bin/python}
OUT=${P2_OUT:-$R/parkour/outputs/omni_plan/phase2}
LOG=$OUT/queue.log
A=$OUT/arms
P2MAN=$C/scenarios_p2.yaml
OMNIMAN=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml
FROWS12="field_light__omni ring_obst__omni"            # the rows that stay at env_seeds 1,2 under p2b (field_hard
                                                        # alone widens to 1-4: CHARTER 10 / P2B_PLAN.md step 3/6)
RUN_ARM=${RUN_ARM:-$S/run_arm.sh}
READ=${READ:-$S/p2_read.py}
LIVE=${LIVE:-$S/p2_live.py}
FORK_MODEL_N=19994                                     # every p2b arm forks its seed's S1 p2a head model_19994
P2B_LEVERS=${P2B_LEVERS:-"CL TG"}
MECH=${MECH:-$S/p2_mech.py}
# the code-state guard compares every row eval / launch with REF_S1_s18's suite sidecar (the p2b code), not the ANCW
# anchor's (p2a code). Until that stamp exists (the first refs rows) the guard has no reference and requires a clean
# tree only, exactly as the p2a anchors' first rows did (review of the step-1 diff, 2026-10-08).
export P2_GUARD_REF=${P2_GUARD_REF:-REF_S1_s18}
mkdir -p "$A"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
# the SAME lock file as p2_queue.sh (review wf_b1fbf4b2, 2026-10-07): a phase-2 queue and a p2b queue would otherwise
# race run_arm.sh's 30-s idle poll and put two Isaac processes on the GPU together.
exec 9>"$OUT/p2_queue.lock"
if ! flock -n 9; then
  say "STOP p2b_queue: another p2_queue holds $OUT/p2_queue.lock (FROM_STEP=${FROM_STEP:-refs} refused) -- NOTIFY"
  echo "p2_queue already running ($OUT/p2_queue.lock)" >&2
  exit 3
fi
stopped() { [ -f "$OUT/STOP_QUEUE" ]; }
block_ok() {  # RC NAME -> continue only when the step ended normally (0)
  if [ "$1" != 0 ]; then say "STOP p2b_queue: rc $1 in the $2 step -- NOTIFY"; exit 4; fi
  stopped && { say "STOP p2b_queue: STOP_QUEUE after the $2 step -- NOTIFY"; exit 4; }
  return 0
}
recf() {  # REF FIELD -> a field of that reference's record
  python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]) or '')" "$A/$1.record.json" "$2" 2>/dev/null
}
row() {  # NAME CK STAMP MANIFEST SCENARIO ENV_SEED
  bash "$RUN_ARM" --row-eval "$1" "$2" "$3" "$4" "$5" "$6"
  block_ok $? "$5 env_seed $6 of $1"
}
lr() {  # NAME CK STAMP
  bash "$RUN_ARM" --lr-eval "$1" "$2" "$3"
  block_ok $? "lr of $1"
}
field_reads_p2b() {  # NAME CK STAMP -> the p2b row reads: field_light/ring_obst at 1,2; field_hard at 1-4; turn/suite at 1
  local r s
  for r in $FROWS12; do row "$1" "$2" "$3" "$P2MAN" "$r" 1; row "$1" "$2" "$3" "$P2MAN" "$r" 2; done
  for s in 1 2 3 4; do row "$1" "$2" "$3" "$P2MAN" field_hard__omni "$s"; done
  row "$1" "$2" "$3" "$P2MAN" field_turn__omni 1
  row "$1" "$2" "$3" "$P2MAN" field_suite__omni 1
}
head_reads_p2b() {  # REF -> lr + the p2b field rows of the arm head, with its own driver suite-stamp env
  local ck st
  ck=$(recf "$1" checkpoint); st=$(recf "$1" driver_stamp)
  [ -n "$ck" ] && [ -n "$st" ] || { say "STOP p2b_queue: no checkpoint/stamp recorded for $1 -- NOTIFY"; exit 4; }
  lr "$1" "$ck" "$st"
  field_reads_p2b "$1" "$ck" "$st"
}
m_reads_p2b() {  # REF FORK_MODEL_N -> the p2b intermediate reads (lr + suite + field_hard at env_seed 1) at the three
  # p2b checkpoints (p2_live.py --p2b-checkpoints; P2B_PLAN.md step 1/4: fork_first + 1005/2505/4005)
  local ck st rd n mck
  ck=$(recf "$1" checkpoint); st=$(recf "$1" driver_stamp); rd=$(dirname "$ck")
  mck=$("$PY" "$LIVE" --p2b-checkpoints "$2") || { say "STOP p2b_queue: p2_live --p2b-checkpoints $2 failed -- NOTIFY"; exit 4; }
  for n in $mck; do
    if [ ! -f "$rd/model_$n.pt" ]; then say "note p2b_queue: $1 has no model_$n.pt (intermediate read skipped) -- NOTIFY"; continue; fi
    lr "${1}_m$n" "$rd/model_$n.pt" "$st"
    row "${1}_m$n" "$rd/model_$n.pt" "$st" "$OMNIMAN" suite__omni 1
    row "${1}_m$n" "$rd/model_$n.pt" "$st" "$P2MAN" field_hard__omni 1
  done
}
arm_p2b() {  # REF -> train it (run_arm.sh stops on any kill/refusal) then its p2b reads
  STOP_ON_REF_STOP=1 bash "$RUN_ARM" "$1"
  block_ok $? "arm $1"
  head_reads_p2b "$1"
  m_reads_p2b "$1" "$FORK_MODEL_N"
}
refs() {  # SEED -> REF_S1_s<seed>'s reads, from the S1 p2a head's own checkpoint + driver_stamp (arms record;
  # CHARTER 10 / P2B_PLAN.md step 3: the new code's reference, re-run so every p2b comparison stays INVALID-EVAL-valid)
  local seed=$1 name=REF_S1_s$1 ck st r s
  ck=$(recf "P2__S1_s${seed}" checkpoint); st=$(recf "P2__S1_s${seed}" driver_stamp)
  [ -n "$ck" ] && [ -n "$st" ] || { say "STOP p2b_queue: no checkpoint/stamp recorded for P2__S1_s${seed} (CHARTER 10 refs) -- NOTIFY"; exit 4; }
  lr "$name" "$ck" "$st"
  row "$name" "$ck" "$st" "$OMNIMAN" suite__omni 1
  [ "$seed" = 18 ] && row "$name" "$ck" "$st" "$OMNIMAN" suite__omni 2
  for r in $FROWS12; do row "$name" "$ck" "$st" "$P2MAN" "$r" 1; row "$name" "$ck" "$st" "$P2MAN" "$r" 2; done
  for s in 1 2 3 4; do row "$name" "$ck" "$st" "$P2MAN" field_hard__omni "$s"; done
  row "$name" "$ck" "$st" "$P2MAN" field_turn__omni 1
  row "$name" "$ck" "$st" "$P2MAN" field_suite__omni 1
}
FROM_STEP=${FROM_STEP:-refs}
step_n() { case "$1" in refs) echo 1;; inertness) echo 2;; screen) echo 3;; reps) echo 4;; recipe) echo 5;; *) echo 0;; esac; }
FROM=$(step_n "$FROM_STEP"); [ "$FROM" -ge 1 ] || { echo "bad FROM_STEP $FROM_STEP" >&2; exit 2; }
want() { [ "$(step_n "$1")" -ge "$FROM" ]; }
say "p2b_queue start (CHARTER 10 / P2B_PLAN.md) from $FROM_STEP: refs -> inertness -> screen (C0/CL/TG s18) -> reps ($P2B_LEVERS at s16,s17) -> recipe"
stopped && { say "STOP p2b_queue: STOP_QUEUE present at start"; exit 4; }

# (1) references on the new code
if want refs; then
for s in 18 16 17; do refs "$s"; done
fi

# (2) code-inertness: every REF must reproduce its p2a reads exactly (CHARTER 10 step 3)
if want inertness; then
g=$("$PY" "$READ" --p2b-inertness 2>&1); rc=$?
say "P2B INERTNESS rc $rc :: $(echo "$g" | tail -1 | cut -c1-600)"
[ "$rc" = 0 ] || { say "STOP p2b_queue: the REF code-inertness check (CHARTER 10 step 3: each REF must reproduce its p2a reads exactly) -- NOTIFY"; exit 4; }
fi

# (3) the s18 screen: C0 (continuation), CL (clearance), TG (tilt guard) -- then stop and ask (the user may prune)
if want screen; then
for arm in C0 CL TG; do arm_p2b "P2__${arm}_s18"; done
g=$("$PY" "$READ" --p2b-screen 2>&1); rc=$?
say "P2B SCREEN rc $rc :: $(echo "$g" | tail -1 | cut -c1-800)"
"$PY" "$MECH" --heads REF_S1_s18,P2__C0_s18,P2__CL_s18,P2__TG_s18 --out "$OUT/p2b_mech_screen.json" >/dev/null 2>&1
say "P2B MECH (descriptive, P2B_PLAN.md step 6) rc $? -> $OUT/p2b_mech_screen.json"
say "STOP p2b_queue: s18 screen done (C0, CL, TG) -- the user may prune levers before the replicates (P2B_LEVERS); resume with FROM_STEP=reps -- NOTIFY"
exit 4
fi

# (4) replicates of every kept lever, at s16 then s17 (CHARTER 10 / P2B_PLAN.md step 5)
if want reps; then
say "p2b_queue replicates: levers [$P2B_LEVERS] at s16 then s17"
for s in 16 17; do
  for lv in $P2B_LEVERS; do
    arm_p2b "P2__${lv}_s${s}"
  done
done
fi

# (5) the recipe read (CHARTER 10 / P2B_PLAN.md step 6) -- then stop and ask: the head-of-record proposal (step 7)
# and the bake are the user's
g=$("$PY" "$READ" --p2b-recipe 2>&1); rc=$?
say "P2B RECIPE rc $rc :: $(echo "$g" | tail -1 | cut -c1-800)"
mh="REF_S1_s16,REF_S1_s17,REF_S1_s18,P2__C0_s18"; for lv in $P2B_LEVERS; do mh="$mh,P2__${lv}_s16,P2__${lv}_s17,P2__${lv}_s18"; done
"$PY" "$MECH" --heads "$mh" --out "$OUT/p2b_mech_recipe.json" >/dev/null 2>&1
say "P2B MECH (descriptive) rc $? -> $OUT/p2b_mech_recipe.json"
say "STOP p2b_queue: replicates done -- the recipe read is logged; the head-of-record proposal (CHARTER 10 / P2B_PLAN.md step 7) and the bake are the user's -- read with scripts/p2_read.py --p2b-recipe --p2b-md PATH -- NOTIFY"
exit 4

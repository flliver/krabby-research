#!/usr/bin/env bash
# Stage p2s queue (P2S_PLAN.md steps 5-7 / CHARTER 11): refs -> inertness -> the s18 screen (ST2; ST1 only if ST2 reads
# STILL) -> STOP and ask -> the replicate references (REFS_S1_s16 / _s17) -> their inertness -> the kept dose at s16, s17
# -> the recipe read -> STOP and ask. A copy of p2b_queue.sh's structure: serial, one Isaac process at a time (run_arm.sh's
# anchored idle check before every launch / eval), the same flock file ($OUT/p2_queue.lock -- a phase-2, p2b and p2s
# queue never run together), the same say / STOP / NOTIFY conventions (the watcher push-notifies any line containing
# " NOTIFY " or " STOP "), and FROM_STEP resume.
#
#   refs       REFS_S1_s18 (the S1 p2a head, arms/P2__S1_s18.record.json checkpoint + driver_stamp) and C0R_s18 (the
#              p2b C0 head, arms/P2__C0_s18.record.json) on the NEW (p2s) code: every p2b read row -- lr; suite__omni at
#              env_seeds 1 AND 2; field_light / ring_obst at 1 and 2; field_hard at 1-4 (n = 400); field_turn /
#              field_suite at 1 -- plus walkstop__omni (scenarios_p2s.yaml) at 1 and 2. REFS first: its suite es1 stamp
#              is the code-state guard's new reference (P2_GUARD_REF=REFS_S1_s18, re-pointed as p2b re-pointed it to
#              REF_S1_s18 after its code move; until that stamp exists the guard requires a clean tree only).
#   inertness  p2_read.py --p2s-inertness --p2s-seeds 18 (P2S_PLAN step 5: every row REFS_S1_s18 / C0R_s18 share with
#              the p2b stamps REF_S1_s18 / P2__C0_s18 reproduces that read EXACTLY; any difference STOPS and asks).
#   st2        the ST2 s18 arm ($P2S_ST2_S18, default P2__ST2_s18; run_arm.sh stops on any kill / refusal), its reads
#              (every refs row except suite es1, which is the arm's driver stamp). The FIRST launch of this queue
#              invocation passes --accept-code-change via ACCEPT_CODE_WHY (the p2s step-3 code move, e1a08baf), ONCE;
#              the code-state guard still refuses any other move. A p2s_flag (the notify-only termination flag, CHARTER
#              11) set during the arm STOPS and asks after its reads.
#   st1        ST1 s18 ($P2S_ST1_S18) ONLY if ST2 reads STILL, with or without COST (p2_read.py --p2s-is-still); else
#              skipped and logged.
#   screen     p2_read.py --p2s-screen (verdicts, attribution vs C0R, C0R's own clause values, reported items) +
#              --p2s-md $OUT/p2s_screen.md, then STOP and ask -- either way (ST2 NO EFFECT / GAMED ends the screen).
#              Resume with FROM_STEP=refsrep (P2S_DOSE = the kept dose, default ST2).
#   refsrep    REFS_S1_s16 then REFS_S1_s17 on the new code (the same rows as refs), then p2_read.py --p2s-inertness
#              --p2s-seeds 16,17 -> any difference STOPS.
#   rep16      the kept dose at s16 ($P2S_REP_S16, default P2__<P2S_DOSE>_s16; labelled "lever + continuation": no
#              same-seed control), the same reads; a p2s_flag stops after the arm (resume FROM_STEP=rep17).
#   rep17      the kept dose at s17 ($P2S_REP_S17), likewise (resume FROM_STEP=recipe).
#   recipe     p2_read.py --p2s-recipe (P2S_PLAN step 7: the CHARTER 6.2 rules with STILL / PARTIAL / COST mapped) +
#              --p2s-md $OUT/p2s_recipe.md -- NOTIFY the reading, then STOP: the head proposal (step 8) and the bake are
#              the user's.
#
# FRESH CHAIN NAMES AFTER A KILL: run_arm.sh records a killed reference ("killed" in arms/<ref>.record.json) and a
# relaunch under the same name would RESUME the stopped run, so this queue REFUSES (STOP) to launch any reference whose
# record says killed. Rerun under a fresh chain (P2__ST<k>R<j>_s<seed> in candidates.yaml) named through P2S_ST2_S18 /
# P2S_ST1_S18 / P2S_REP_S16 / P2S_REP_S17 (p2_read.py reads the same variables). A reference that already has a
# (non-killed) record is refused too: it was trained -- resume past its step instead.
#
# Usage (detached): setsid nohup bash scripts/p2s_queue.sh > $OUT/p2s_queue.log 2>&1 < /dev/null &
# Resume after a stop: FROM_STEP=refs|inertness|st2|st1|screen|refsrep|rep16|rep17|recipe skips the earlier steps (a
# finished arm must never be relaunched: run_arm.sh would resume it, not fork it again).
set -u
C=$(cd "$(dirname "$0")/.." && pwd)
S=$C/scripts
R=/home/nickmagus/krabby/krabby-research
PY=${P2_PY:-/home/nickmagus/krabby/isaac_venv/bin/python}
OUT=${P2_OUT:-$R/parkour/outputs/omni_plan/phase2}
LOG=$OUT/queue.log
A=$OUT/arms
P2MAN=$C/scenarios_p2.yaml
P2SMAN=$C/scenarios_p2s.yaml
OMNIMAN=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml
FROWS12="field_light__omni ring_obst__omni"            # the rows read at env_seeds 1,2 (field_hard widens to 1-4)
WSROW=walkstop__omni
RUN_ARM=${RUN_ARM:-$S/run_arm.sh}
READ=${READ:-$S/p2_read.py}
export P2S_DOSE=${P2S_DOSE:-ST2}
export P2S_ST2_S18=${P2S_ST2_S18:-P2__ST2_s18}
export P2S_ST1_S18=${P2S_ST1_S18:-P2__ST1_s18}
export P2S_REP_S16=${P2S_REP_S16:-P2__${P2S_DOSE}_s16}
export P2S_REP_S17=${P2S_REP_S17:-P2__${P2S_DOSE}_s17}
# the code-state guard compares every row eval / launch with REFS_S1_s18's suite sidecar (the p2s code), not REF_S1_s18's
# (p2b code): re-pointed exactly as p2b_queue.sh re-pointed it after the p2b step-1 code move
export P2_GUARD_REF=${P2_GUARD_REF:-REFS_S1_s18}
# --accept-code-change for the driver's recorded code_sha (e3fe70fb, p2b) -> the p2s step-3 code: passed ONCE, on this
# invocation's first launch (run_arm.sh: a no-op when the code_sha is unchanged; the code guard refuses any other move)
ACC_ONCE=${P2S_ACCEPT_CODE_WHY-"p2s step-3 code move (reward_stand_support / penalty_stand_slide knobs, e1a08baf; CHARTER 11, P2S_PLAN step 5)"}
mkdir -p "$A"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
# the SAME lock file as p2_queue.sh / p2b_queue.sh: two queues would race run_arm.sh's idle poll onto one GPU
exec 9>"$OUT/p2_queue.lock"
if ! flock -n 9; then
  say "STOP p2s_queue: another p2_queue holds $OUT/p2_queue.lock (FROM_STEP=${FROM_STEP:-refs} refused) -- NOTIFY"
  echo "p2_queue already running ($OUT/p2_queue.lock)" >&2
  exit 3
fi
stopped() { [ -f "$OUT/STOP_QUEUE" ]; }
block_ok() {  # RC NAME -> continue only when the step ended normally (0)
  if [ "$1" != 0 ]; then say "STOP p2s_queue: rc $1 in the $2 step -- NOTIFY"; exit 4; fi
  stopped && { say "STOP p2s_queue: STOP_QUEUE after the $2 step -- NOTIFY"; exit 4; }
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
field_reads_p2s() {  # NAME CK STAMP -> the p2b field rows + walkstop: field_light / ring_obst at 1,2; field_hard at 1-4;
  # field_turn / field_suite at 1; walkstop__omni (scenarios_p2s.yaml, flat) at 1,2
  local r s
  for r in $FROWS12; do row "$1" "$2" "$3" "$P2MAN" "$r" 1; row "$1" "$2" "$3" "$P2MAN" "$r" 2; done
  for s in 1 2 3 4; do row "$1" "$2" "$3" "$P2MAN" field_hard__omni "$s"; done
  row "$1" "$2" "$3" "$P2MAN" field_turn__omni 1
  row "$1" "$2" "$3" "$P2MAN" field_suite__omni 1
  for s in 1 2; do row "$1" "$2" "$3" "$P2SMAN" "$WSROW" "$s"; done
}
ref_reads() {  # NAME SOURCE_REF -> a p2s reference (REFS / C0R): the SOURCE's checkpoint + driver_stamp env, every row
  local name=$1 ck st
  ck=$(recf "$2" checkpoint); st=$(recf "$2" driver_stamp)
  [ -n "$ck" ] && [ -n "$st" ] || { say "STOP p2s_queue: no checkpoint/stamp recorded for $2 (the source of $name) -- NOTIFY"; exit 4; }
  lr "$name" "$ck" "$st"
  row "$name" "$ck" "$st" "$OMNIMAN" suite__omni 1
  row "$name" "$ck" "$st" "$OMNIMAN" suite__omni 2
  field_reads_p2s "$name" "$ck" "$st"
}
head_reads_p2s() {  # REF -> lr + suite es2 + the field rows + walkstop of a trained arm head, with its driver stamp env
  local ck st
  ck=$(recf "$1" checkpoint); st=$(recf "$1" driver_stamp)
  [ -n "$ck" ] && [ -n "$st" ] || { say "STOP p2s_queue: no checkpoint/stamp recorded for $1 -- NOTIFY"; exit 4; }
  lr "$1" "$ck" "$st"
  row "$1" "$ck" "$st" "$OMNIMAN" suite__omni 2
  field_reads_p2s "$1" "$ck" "$st"
}
refuse_recorded() {  # REF -> STOP when a record exists: killed (fresh chain names after a kill) or already trained
  local k
  [ -f "$A/$1.record.json" ] || return 0
  k=$(recf "$1" killed)
  if [ -n "$k" ]; then
    say "STOP p2s_queue: $1 was killed ($k) -- relaunching it would RESUME the stopped run; rerun under a fresh chain name (P2__ST<k>R<j>_s<seed>, via P2S_ST2_S18 / P2S_ST1_S18 / P2S_REP_S16 / P2S_REP_S17) -- NOTIFY"
  else
    say "STOP p2s_queue: $1 already has a record ($A/$1.record.json) -- it was trained; never relaunch it, resume past its step -- NOTIFY"
  fi
  exit 4
}
arm_p2s() {  # REF -> train it (run_arm.sh stops on any kill/refusal), its reads, then stop if its p2s_flag was set
  refuse_recorded "$1"
  ACCEPT_CODE_WHY="$ACC_ONCE" STOP_ON_REF_STOP=1 bash "$RUN_ARM" "$1"
  local rc=$?
  ACC_ONCE=""
  block_ok "$rc" "arm $1"
  head_reads_p2s "$1"
  if [ -f "$A/$1.p2s_flag" ]; then
    say "STOP p2s_queue: $1 carries the notify-only p2s termination flag ($(tr '\t' ' ' < "$A/$1.p2s_flag" | cut -c1-300)); its reads are done -- the user decides before the next launch (resume with FROM_STEP=${2:-next}) -- NOTIFY"
    exit 4
  fi
}
inert() {  # SEEDS -> p2_read.py --p2s-inertness; anything but rc 0 STOPS and asks
  local g rc
  g=$("$PY" "$READ" --p2s-inertness --p2s-seeds "$1" 2>&1); rc=$?
  say "P2S INERTNESS ($1) rc $rc :: $(echo "$g" | tail -1 | cut -c1-600)"
  [ "$rc" = 0 ] || { say "STOP p2s_queue: the reference inertness check at seeds $1 (P2S_PLAN step 5: every row shared with p2b must reproduce its p2b read exactly) -- NOTIFY"; exit 4; }
}
FROM_STEP=${FROM_STEP:-refs}
step_n() { case "$1" in refs) echo 1;; inertness) echo 2;; st2) echo 3;; st1) echo 4;; screen) echo 5;; refsrep) echo 6;;
                        rep16) echo 7;; rep17) echo 8;; recipe) echo 9;; *) echo 0;; esac; }
FROM=$(step_n "$FROM_STEP"); [ "$FROM" -ge 1 ] || { echo "bad FROM_STEP $FROM_STEP" >&2; exit 2; }
want() { [ "$(step_n "$1")" -ge "$FROM" ]; }
# UNTIL_STEP=<step>: stop (and notify) after that step -- used to run the references while a user decision on a later
# step is pending (2026-10-08: the f stop-rule question before the ST2 launch)
UNTIL=$(step_n "${UNTIL_STEP:-recipe}"); [ "$UNTIL" -ge 1 ] || { echo "bad UNTIL_STEP $UNTIL_STEP" >&2; exit 2; }
until_stop() {  # STEP -> exit 4 (STOP + NOTIFY) when STEP is the UNTIL_STEP
  if [ "$(step_n "$1")" -ge "$UNTIL" ] && [ "$UNTIL" -lt 9 ]; then
    say "STOP p2s_queue: UNTIL_STEP=${UNTIL_STEP} reached after the $1 step -- resume with FROM_STEP=<next step> -- NOTIFY"; exit 4
  fi
}
say "p2s_queue start (CHARTER 11 / P2S_PLAN.md) from $FROM_STEP: refs (REFS_S1_s18, C0R_s18) -> inertness -> st2 ($P2S_ST2_S18) -> st1 ($P2S_ST1_S18, only if ST2 reads STILL) -> screen STOP -> refsrep (REFS_S1_s16/s17) -> rep16 ($P2S_REP_S16) -> rep17 ($P2S_REP_S17; dose $P2S_DOSE) -> recipe STOP; guard ref $P2_GUARD_REF"
stopped && { say "STOP p2s_queue: STOP_QUEUE present at start"; exit 4; }
# SMOKES_OK is NOT a gate here, as in p2b_queue.sh (CHARTER 5's smokes were not re-run after either code move); its
# state is logged for the record only
g=$("$PY" "$READ" --smokes-ok-check 2>&1); say "note p2s_queue: SMOKES_OK check rc $? (informational, not a gate -- as in p2b) :: $(echo "$g" | tail -1 | cut -c1-300)"

# (1) the s18 references on the new code: REFS_S1_s18 first (the guard's reference), then C0R_s18
if want refs; then
ref_reads REFS_S1_s18 P2__S1_s18
ref_reads C0R_s18 P2__C0_s18
fi

# (2) inertness: REFS_S1_s18 / C0R_s18 reproduce REF_S1_s18 / P2__C0_s18 exactly on every shared row
if want inertness; then
inert 18
fi
want refs && until_stop refs
want inertness && until_stop inertness

# (3) the ST2 screen arm
if want st2; then
arm_p2s "$P2S_ST2_S18" st1
fi

# (4) ST1 only if ST2 reads STILL (with or without COST)
if want st1; then
g=$("$PY" "$READ" --p2s-is-still "$P2S_ST2_S18" 2>&1); rc=$?
say "P2S ST2 READS STILL? rc $rc :: $(echo "$g" | tail -1 | cut -c1-400)"
# fable review 2026-10-08: rc 4 is "not STILL" ONLY when the read is a final verdict; a PENDING / INVALID read, a
# reader crash or an argparse error must STOP, never silently skip ST1
vv=$(echo "$g" | tail -1 | python3 -c "import json,sys
try: d=json.loads(sys.stdin.read()); print(d.get('verdict') or '')
except Exception: print('')" 2>/dev/null)
case "$rc:$vv" in
  0:*) arm_p2s "$P2S_ST1_S18" screen ;;
  "4:NO EFFECT"|"4:PARTIAL"|"4:GAMED"|"4:COST"|"4:FLAT FAIL"|"4:OBSTACLE LOSS"|"4:HAND LOSS"|"4:DOSE")
    say "p2s_queue: $P2S_ST2_S18 reads $vv (not STILL) -- ST1 is not run (P2S_PLAN step 6)" ;;
  *) say "STOP p2s_queue: the ST2 STILL read is not a final verdict (rc $rc, verdict '$vv') -- fix the read, then resume with FROM_STEP=st1 -- NOTIFY"; exit 4 ;;
esac
fi

# (5) the s18 screen read -- then stop and ask, either way
if want screen; then
g=$("$PY" "$READ" --p2s-screen --p2s-md "$OUT/p2s_screen.md" 2>&1); rc=$?
echo "$g" | tail -1 > "$OUT/p2s_screen.json"
say "P2S SCREEN rc $rc :: $(echo "$g" | tail -1 | cut -c1-800) -> $OUT/p2s_screen.json, $OUT/p2s_screen.md"
say "STOP p2s_queue: s18 screen done ($P2S_ST2_S18$( [ -f "$A/$P2S_ST1_S18.record.json" ] && echo ", $P2S_ST1_S18")) -- the user picks the kept dose (P2S_DOSE) before the replicates; resume with FROM_STEP=refsrep -- NOTIFY"
exit 4
fi

# (6) the replicate references on the new code, then their inertness
if want refsrep; then
ref_reads REFS_S1_s16 P2__S1_s16
ref_reads REFS_S1_s17 P2__S1_s17
inert 16,17
fi

# (7) the kept dose at s16, then (8) at s17 ("lever + continuation")
if want rep16; then
say "p2s_queue replicates: dose $P2S_DOSE -> $P2S_REP_S16 then $P2S_REP_S17"
arm_p2s "$P2S_REP_S16" rep17
fi
if want rep17; then
arm_p2s "$P2S_REP_S17" recipe
fi

# (9) the recipe read -- then stop and ask: the head proposal (step 8) and the bake are the user's
g=$("$PY" "$READ" --p2s-recipe --p2s-md "$OUT/p2s_recipe.md" 2>&1); rc=$?
say "P2S RECIPE rc $rc :: $(echo "$g" | tail -1 | cut -c1-800)"
say "STOP p2s_queue: replicates done -- the recipe read is logged ($OUT/p2s_recipe.md); the head proposal (P2S_PLAN step 8) and the bake are the user's -- NOTIFY"
exit 4

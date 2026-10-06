#!/usr/bin/env bash
# Omni phase-2 queue (CHARTER 5 / 6 / 8): anchors -> A1 s18 (the gate) -> A0 s18 -> S1 -> F1 -> A1 s16, A1 s17
# -> the recipe read. E1 was dropped by the user on 2026-10-06 (CHARTER 4 replay gate FAIL; CHARTER 9): its chain keeps
# a placeholder and the sibling loop never runs it. Serial, one Isaac process at a time (run_arm.sh's anchored idle
# check before every launch and every eval). Records in $OUT/arms, eval stamps in $OUT/lr. Every stop writes a
# "STOP p2_queue" / NOTIFY line to $LOG (the watcher push-notifies it) and exits 4; $OUT/STOP_QUEUE stops the queue
# between steps.
#
#   gates    (pre-launch audit B1 / M2, 2026-10-06) nothing runs without $OUT/SMOKES_OK: at the start, whatever
#            FROM_STEP, p2_read.py --smokes-ok-check must pass (the sentinel p2_read.py --write-smokes-ok writes only
#            after the CHARTER 5 smokes S1-S4 (scripts/p2_smoke.py), S5 (field_light at 7 episodes, zero stand-hold
#            falls) and the CAL1 fork smoke's dump audit passed, on the live code state). Before arm P2__A1_s18,
#            p2_read.py --cal2-gate must pass (P2__CAL2_s18's record: smoke_audit PASS, not killed, and its PROMOTE ==
#            the A1 chain's KRABBY_TERRAIN_PROMOTE in candidates.yaml). Either failing is STOP + NOTIFY, exit 4.
#            Every row-eval / lr-eval / arm launch also passes run_arm.sh's code-state guard (M1; exit 7 -> STOP here).
#   anchors  the three c6x heads (anchors.yaml) with their own c6x suite-stamp env, every CHARTER 5 row:
#            suite__omni (s18 also at env_seed 2: the suite band), side22_25lr__omni, slow__omni (the flat completion
#            ceiling), field_light / field_hard / ring_obst at env_seeds 1 AND 2, field_turn, field_suite -> then
#            p2_read.py --anchors-check (discrimination + inertness): anything but ok STOPS and asks.
#   a1s18    P2__A1_s18 -> head reads -> intermediate reads -> p2_read.py --gate: anything but PASS STOPS and asks.
#   a0       P2__A0_s18 -> head reads -> p2_read.py --field-vs-ring (descriptive, logged; NOTIFY, no stop).
#   s1|f1    P2__<arm>_s18 -> head reads -> p2_read.py --sibling <arm>: EVERY sibling verdict STOPS and asks (CHARTER
#            6.4); resume with FROM_STEP=<next>. (e1 is still a valid FROM_STEP value; no step runs it.)
#   a1rep    P2__A1_s16, P2__A1_s17 -> head reads each.
#   recipe   p2_read.py --recipe (CHARTER 6.2 + the 6.5 p2b trigger): NOTIFY the reading; STOP when p2b triggers or a
#            seed is still PENDING.
# Head reads of an arm head: side22_25lr (lr eval), field_light / field_hard / ring_obst at env_seeds 1 and 2,
# field_turn and field_suite at env_seed 1, all with the arm's own driver suite-stamp env (its suite read IS that driver
# stamp; run_arm.sh gave it its sidecar). Intermediate reads at model_16000 / 17500 / 19000: lr, suite and field_hard
# at env_seed 1 (n = 100), then p2_read.py --m-screen names the ones that beat the head by more than max(0.10, 2 band):
# each gets field_hard at env_seed 2 (CHARTER 7).
#
# Usage (detached): setsid nohup bash scripts/p2_queue.sh > $OUT/p2_queue.log 2>&1 < /dev/null &
# Resume after a stop: FROM_STEP=anchors|a1s18|a0|s1|e1|f1|a1rep|recipe skips the earlier steps (a finished reference
# must never be relaunched: run_arm.sh would resume it). run_arm.sh passes --accept-candidates-change (its default ACCEPT_WHY) on each invocation's first launch.
C=$(cd "$(dirname "$0")/.." && pwd)
S=$C/scripts
R=/home/nickmagus/krabby/krabby-research
PY=${P2_PY:-/home/nickmagus/krabby/isaac_venv/bin/python}
OUT=${P2_OUT:-$R/parkour/outputs/omni_plan/phase2}
LOG=$OUT/queue.log
A=$OUT/arms
P2MAN=$C/scenarios_p2.yaml
OMNIMAN=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml
FROWS="field_light__omni field_hard__omni ring_obst__omni"
MCK="16000 17500 19000"
RUN_ARM=${RUN_ARM:-$S/run_arm.sh}
READ=${READ:-$S/p2_read.py}
mkdir -p "$A"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
stopped() { [ -f "$OUT/STOP_QUEUE" ]; }
block_ok() {  # RC NAME -> continue only when the step ended normally (0)
  if [ "$1" != 0 ]; then say "STOP p2_queue: rc $1 in the $2 step -- NOTIFY"; exit 4; fi
  stopped && { say "STOP p2_queue: STOP_QUEUE after the $2 step -- NOTIFY"; exit 4; }
  return 0
}
recf() {  # REF FIELD -> a field of that reference's record
  python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]) or '')" "$A/$1.record.json" "$2" 2>/dev/null
}
anc() {  # SEED FIELD -> anchors.yaml field (checkpoint / env_stamp made absolute)
  python3 - "$C/anchors.yaml" "$1" "$2" "$R" <<'PY'
import sys, yaml
a = yaml.safe_load(open(sys.argv[1]))["anchors"][int(sys.argv[2])][sys.argv[3]]
print(f"{sys.argv[4]}/{a}" if sys.argv[3] in ("checkpoint", "env_stamp") else a)
PY
}
row() {  # NAME CK STAMP MANIFEST SCENARIO ENV_SEED
  bash "$RUN_ARM" --row-eval "$1" "$2" "$3" "$4" "$5" "$6"
  block_ok $? "$5 env_seed $6 of $1"
}
lr() {  # NAME CK STAMP
  bash "$RUN_ARM" --lr-eval "$1" "$2" "$3"
  block_ok $? "lr of $1"
}
field_reads() {  # NAME CK STAMP -> the field / ring rows of one head (both env_seeds for the completion rows)
  local r
  for r in $FROWS; do row "$1" "$2" "$3" "$P2MAN" "$r" 1; row "$1" "$2" "$3" "$P2MAN" "$r" 2; done
  row "$1" "$2" "$3" "$P2MAN" field_turn__omni 1
  row "$1" "$2" "$3" "$P2MAN" field_suite__omni 1
}
head_reads() {  # REF -> lr + the field rows of the arm head, with its own driver suite-stamp env
  local ck st
  ck=$(recf "$1" checkpoint); st=$(recf "$1" driver_stamp)
  [ -n "$ck" ] && [ -n "$st" ] || { say "STOP p2_queue: no checkpoint/stamp recorded for $1 -- NOTIFY"; exit 4; }
  lr "$1" "$ck" "$st"
  field_reads "$1" "$ck" "$st"
}
m_reads() {  # REF -> the CHARTER 7 intermediate reads (+ the env_seed-2 field_hard of those that beat the head)
  local ck st rd n m
  ck=$(recf "$1" checkpoint); st=$(recf "$1" driver_stamp); rd=$(dirname "$ck")
  for n in $MCK; do
    if [ ! -f "$rd/model_$n.pt" ]; then say "note p2_queue: $1 has no model_$n.pt (intermediate read skipped) -- NOTIFY"; continue; fi
    lr "${1}_m$n" "$rd/model_$n.pt" "$st"
    row "${1}_m$n" "$rd/model_$n.pt" "$st" "$OMNIMAN" suite__omni 1
    row "${1}_m$n" "$rd/model_$n.pt" "$st" "$P2MAN" field_hard__omni 1
  done
  for m in $("$PY" "$READ" --m-screen "$1" 2>/dev/null); do
    n=${m##*_m}
    say "M-SCREEN $m beats the head on field_hard at n=100: env_seed-2 read (CHARTER 7)"
    row "$m" "$rd/model_$n.pt" "$st" "$P2MAN" field_hard__omni 2
  done
}
arm() {  # REF -> train it (run_arm.sh stops on any kill / refusal) then its reads
  STOP_ON_REF_STOP=1 bash "$RUN_ARM" "$1"
  block_ok $? "arm $1"
  head_reads "$1"
  m_reads "$1"
}
FROM_STEP=${FROM_STEP:-anchors}
step_n() { case "$1" in anchors) echo 1;; a1s18) echo 2;; a0) echo 3;; s1) echo 4;; e1) echo 5;; f1) echo 6;; a1rep) echo 7;; recipe) echo 8;; *) echo 0;; esac; }
FROM=$(step_n "$FROM_STEP"); [ "$FROM" -ge 1 ] || { echo "bad FROM_STEP $FROM_STEP" >&2; exit 2; }
want() { [ "$(step_n "$1")" -ge "$FROM" ]; }
say "p2_queue start (CHARTER 8) from $FROM_STEP: anchors -> A1 s18 gate -> A0 -> S1 -> F1 -> A1 s16/s17 -> recipe (E1 dropped)"
stopped && { say "STOP p2_queue: STOP_QUEUE present at start"; exit 4; }

# (0) the CHARTER 5 smokes gate (pre-launch audit B1): no anchor, arm or eval without a valid $OUT/SMOKES_OK
g=$("$PY" "$READ" --smokes-ok-check --out "$OUT" 2>&1); rc=$?
say "SMOKES_OK rc $rc :: $(echo "$g" | tail -1 | cut -c1-600)"
[ "$rc" = 0 ] || { say "STOP p2_queue: no valid $OUT/SMOKES_OK (CHARTER 5 smokes S1-S5 + the CAL1 fork smoke must pass first; scripts/p2_smoke.py, then p2_read.py --smoke-s5 / --write-smokes-ok) -- NOTIFY"; exit 4; }

# (1) anchors (CHARTER 5)
if want anchors; then
for s in 18 16 17; do
  n=$(anc "$s" name); ck=$(anc "$s" checkpoint); st=$(anc "$s" env_stamp)
  row "$n" "$ck" "$st" "$OMNIMAN" suite__omni 1
  [ "$s" = 18 ] && row "$n" "$ck" "$st" "$OMNIMAN" suite__omni 2
  lr "$n" "$ck" "$st"
  row "$n" "$ck" "$st" "$OMNIMAN" slow__omni 1
  field_reads "$n" "$ck" "$st"
done
g=$("$PY" "$READ" --anchors-check 2>&1); rc=$?
say "ANCHORS rc $rc :: $(echo "$g" | tail -1 | cut -c1-600)"
[ "$rc" = 0 ] || { say "STOP p2_queue: the CHARTER 5 anchor checks (discrimination / inertness) -- NOTIFY stop and ask"; exit 4; }
fi

# (2) A1 s18, the gate (CHARTER 6.2)
if want a1s18; then
g=$("$PY" "$READ" --cal2-gate --out "$OUT" 2>&1); rc=$?
say "CAL2 GATE rc $rc :: $(echo "$g" | tail -1 | cut -c1-600)"
[ "$rc" = 0 ] || { say "STOP p2_queue: P2__CAL2_s18 is not a passed calibration smoke at the A1 PROMOTE values (CHARTER 5 step 4; pre-launch audit M2) -- NOTIFY"; exit 4; }
arm P2__A1_s18
g=$("$PY" "$READ" --gate 2>&1); rc=$?
say "GATE A1 s18 rc $rc :: $(echo "$g" | tail -1 | cut -c1-600)"
[ "$rc" = 0 ] || { say "STOP p2_queue: A1 s18 is not PASS (CHARTER 6.2) -- NOTIFY stop and ask"; exit 4; }
say "NOTIFY A1 s18 PASS: the gate continues to A0 / the siblings"
fi

# (3) A0 ring control (CHARTER 6.3, descriptive)
if want a0; then
arm P2__A0_s18
g=$("$PY" "$READ" --field-vs-ring 2>&1)
say "NOTIFY FIELD-VS-RING (descriptive, CHARTER 6.3) :: $(echo "$g" | tail -1 | cut -c1-800)"
fi

# (4-6) siblings: every verdict stops and asks (CHARTER 6.4)
for sib in S1 F1; do   # E1 dropped by the user 2026-10-06 (CHARTER 4 replay gate FAIL; CHARTER 9)
  step=$(echo "$sib" | tr 'A-Z' 'a-z')
  if want "$step"; then
    arm "P2__${sib}_s18"
    g=$("$PY" "$READ" --sibling "$sib" 2>&1)
    say "SIBLING $sib :: $(echo "$g" | tail -1 | cut -c1-800)"
    nxt=$(case $sib in S1) echo f1;; F1) echo a1rep;; esac)
    say "STOP p2_queue: sibling $sib verdict (CHARTER 6.4) -- NOTIFY stop and ask; resume with FROM_STEP=$nxt"
    exit 4
  fi
done

# (7) A1 replicates
if want a1rep; then
arm P2__A1_s16
arm P2__A1_s17
fi

# (8) recipe read + p2b trigger
g=$("$PY" "$READ" --recipe 2>&1); rc=$?
say "RECIPE rc $rc :: $(echo "$g" | tail -1 | cut -c1-800)"
if [ "$rc" != 0 ]; then say "STOP p2_queue: recipe read PENDING or the p2b trigger fired (CHARTER 6.5) -- NOTIFY stop and ask"; exit 4; fi
say "NOTIFY p2_queue end: recipe read logged; the head-of-record proposal (CHARTER 7) is the judges' -- read with scripts/p2_read.py --md"

#!/usr/bin/env bash
# Side-bisect decision tree (user 2026-09-30; pre-registered in CHARTER "Decision tree after C_lr"):
#   C_lr OK (FAST both hands)          -> CFG = C_lr
#   C_lr FAIL (SLOW/DEAD both hands)   -> run C_lr_10 (+ s2 if ambiguous): OK -> CFG = C_lr_10; FAIL -> BOTH_FAIL (stop; analysis)
#   anything else ambiguous            -> STOP for a manual decision (decision defaults)
#   with CFG: FB_<CFG> (+ fwd/back, suite eval + both-hands side eval), then P1_<CFG> (full phase 1, 12000 its),
#   then both-hands side eval of P1's c6x head and of W for comparison. Then DONE (the session waits for the user).
set -u
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
C=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect
L=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/launch_candidates.sh
EVAL=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
LRMAN=$C/scenarios_side_lr.yaml
EV=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect
RUNS=$R/parkour/logs/rsl_rl/crab_hex_omni_flat_walk
OUT=$R/parkour/outputs/omni_plan/side_bisect/arms
LROUT=$R/parkour/outputs/omni_plan/side_bisect/lr
PB=/home/nickmagus/.claude/plans/side-speed-bisection.evidence/phaseB
LOG=$R/parkour/outputs/omni_plan/side_bisect/queue.log
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
W_CK=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt
W_ST=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29
mkdir -p $OUT $LROUT
say() { echo "[$(date -Is)] $*" >> $LOG; }
wait_unit() { until ! systemctl --user is-active -q $1; do sleep 30; done; }
drivers_idle() { while pgrep -f "isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py" >/dev/null; do sleep 30; done; }
cls_of() { python3 -c "import json;print(json.load(open('$1'))['$2'].get('$3'))"; }
decide() {  # L R -> OK / FAIL / AMBIG
  if [ "$1" = FAST ] && [ "$2" = FAST ]; then echo OK
  elif { [ "$1" = SLOW ] || [ "$1" = DEAD ]; } && { [ "$2" = SLOW ] || [ "$2" = DEAD ]; }; then echo FAIL
  else echo AMBIG; fi
}
any_stamp_of() {  # RUN_DIR -> newest complete eval stamp (any row) of that run
  local rd=$1 tag; tag=$(basename $rd | cut -d_ -f3-)
  ls -d $EV/$tag/*/seed001/*/ 2>/dev/null | while read d; do
    grep -q "$(basename $rd)" $d/run_meta.json 2>/dev/null && [ -f $d/complete.json ] && echo $d; done | tail -1
}
newest_run_after() {  # EPOCH [PATTERN]
  local t=$1 pat=${2:-}
  for d in $(ls -dt $RUNS/*${pat}*/ 2>/dev/null | head -8); do
    [ $(stat -c %Y $d) -ge $t ] && { echo ${d%/}; return; }
  done
}
FIRST_ACC=1
ACC=(--accept-candidates-change "side bisect 2026-09-30: user's decision tree after C_lr (C_lr_10; FB_*; P1_*), per the CHARTER")
launch() {  # REF -> UNIT, T0
  local ref=$1 lout args=()
  if [ $FIRST_ACC = 1 ]; then args=("${ACC[@]}"); fi
  for try in $(seq 40); do
    lout=$(cd $R && $L --campaign-dir $C --reference-chain $ref "${args[@]}" 2>&1)
    UNIT=$(echo "$lout" | head -1)
    case "$UNIT" in cands-*) break ;; esac
    say "launch of $ref refused (try $try): $(echo "$lout" | tail -1 | cut -c1-200); retry in 60 s"
    UNIT=""; sleep 60
  done
  if [ -z "$UNIT" ]; then say "STOP $ref: launch refused 40 times"; exit 5; fi
  FIRST_ACC=0
  T0=$(date +%s)
  say "launched $ref unit $UNIT"
}
lr_eval() {  # NAME CHECKPOINT ENV_STAMP -> sets LRL LRR
  local name=$1 ck=$2 st=$3 envs d
  envs=$(python3 -c "import json;m=json.load(open('$st/run_meta.json'));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))")
  drivers_idle
  ( cd $R/parkour && env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH=$R/parkour/parkour_tasks:$R/parkour $envs \
      $CAP timeout 3600 $PY $EVAL --headless --manifest $LRMAN --scenario side22_25lr__omni \
      --checkpoint $ck --save-raw --no-plot --output-root $LROUT/$name ) > $LROUT/$name.log 2>&1
  d=$(ls -d $LROUT/$name/side22_25lr__omni/seed001/*/ 2>/dev/null | tail -1)
  if [ -z "$d" ] || [ ! -f $d/complete.json ]; then say "STOP $name: lr eval produced no complete stamp"; exit 7; fi
  $PY $PB/side22_classify.py --json $OUT/${name}_lr.json $name=$d 2>&1 | grep -E "lat_|CLASS" >> $LOG
  LRL=$(cls_of $OUT/${name}_lr.json $name class); LRR=$(cls_of $OUT/${name}_lr.json $name class_r)
  say "EVENT ${name}_lr CLASS_L $LRL CLASS_R $LRR"
}
gate_arm() {  # REF (single-stage side-only arm) -> sets VERDICT
  local ref=$1 rd st
  launch $ref
  rd=""; for _ in $(seq 60); do rd=$(newest_run_after $T0); [ -n "$rd" ] && break; sleep 15; done
  [ -z "$rd" ] && { say "STOP $ref: no run dir"; exit 4; }
  say "$ref run dir $rd"
  wait_unit $UNIT
  tail -2 $C/orchestrator.log | cut -c1-300 >> $LOG
  st=$(any_stamp_of $rd); [ -z "$st" ] && { say "STOP $ref: no complete eval stamp"; exit 6; }
  lr_eval $ref $rd/model_2999.pt $st
  VERDICT=$(decide $LRL $LRR)
  say "EVENT $ref VERDICT $VERDICT"
}

say "p1-queue armed (waits for the C_lr queue to end)"
until grep -q "DONE clr""-queue" $LOG; do sleep 30; done
drivers_idle
V1=$(decide "$(cls_of $OUT/C_lr_lr.json C_lr class)" "$(cls_of $OUT/C_lr_lr.json C_lr class_r)")
if [ -f $OUT/C_lr_s2_lr.json ]; then
  V2=$(decide "$(cls_of $OUT/C_lr_s2_lr.json C_lr_s2 class)" "$(cls_of $OUT/C_lr_s2_lr.json C_lr_s2 class_r)")
  if [ $V1 = OK ] && [ $V2 = OK ]; then V1=OK; elif [ $V2 = FAIL ] && [ $V1 != OK ]; then V1=FAIL; else V1=AMBIG; fi
fi
say "EVENT C_lr VERDICT $V1"
cp $PB/candidates_v11.yaml $C/candidates.yaml
say "applied candidates_v11.yaml"
CFG=""
if [ $V1 = OK ]; then CFG=C_lr
elif [ $V1 = FAIL ]; then
  gate_arm C_lr_10; V3=$VERDICT
  if [ $V3 = AMBIG ]; then
    gate_arm C_lr_10_s2; V4=$VERDICT
    if [ $V4 = OK ] && [ $V3 != FAIL ]; then V3=AMBIG; fi   # s3 ambiguous + s2 FAST: still a manual call
    [ $V4 = FAIL ] && V3=FAIL
  fi
  if [ $V3 = OK ]; then CFG=C_lr_10
  elif [ $V3 = FAIL ]; then say "EVENT BOTH_FAIL (C_lr and C_lr_10) -> analysis + next steps by the session"; say "DONE p1-queue"; exit 0
  else say "STOP C_lr_10: ambiguous ($V3) -> manual decision"; exit 8; fi
else say "STOP C_lr: ambiguous -> manual decision"; exit 8; fi

say "EVENT CFG $CFG -> FB_$CFG then P1_$CFG"
# fwd/back added (suite eval by the driver; side gate by the both-hands eval)
launch FB_$CFG
rd=""; for _ in $(seq 60); do rd=$(newest_run_after $T0); [ -n "$rd" ] && break; sleep 15; done
say "FB_$CFG run dir $rd"; wait_unit $UNIT; tail -2 $C/orchestrator.log | cut -c1-300 >> $LOG
st=$(any_stamp_of $rd); [ -z "$st" ] && { say "STOP FB_$CFG: no complete eval stamp"; exit 6; }
lr_eval FB_$CFG $rd/model_2999.pt $st
say "EVENT FB_$CFG VERDICT $(decide $LRL $LRR)"
# full phase 1 (t3 -> c4 -> c5 -> c6 -> c6x), one driver unit
launch P1_$CFG
wait_unit $UNIT; tail -3 $C/orchestrator.log | cut -c1-300 >> $LOG
rd=$(newest_run_after $T0 _c6x_); [ -z "$rd" ] && { say "STOP P1_$CFG: no c6x run dir"; exit 6; }
ck=$(ls $rd/model_*.pt | sort -t_ -k2 -n | tail -1)
st=$(any_stamp_of $rd); [ -z "$st" ] && { say "STOP P1_$CFG: no c6x suite stamp"; exit 6; }
say "P1_$CFG c6x head $ck"
lr_eval P1_$CFG $ck $st
say "EVENT P1_$CFG VERDICT $(decide $LRL $LRR)"
lr_eval W_ref $W_CK $W_ST
say "EVENT W_ref VERDICT $(decide $LRL $LRR)"
say "DONE p1-queue"

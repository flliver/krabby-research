#!/usr/bin/env bash
# Side-bisect: canonical harvest -> two-direction bank -> C_lr_rsi (user 2026-09-30; CHARTER "Mirror canonicalization").
# Functions derived from queue_p1.sh.
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
ACC=(--accept-candidates-change "side bisect 2026-09-30: C_lr_10_rsi (user request), per the CHARTER")
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
  # campaign default for DEAD: diagnose + seed 2 -> treat a DEAD hand as ambiguous so the tree runs the seed-2 arm
  if [ "$LRL" = DEAD ] || [ "$LRR" = DEAD ]; then [ "$VERDICT" = FAIL ] && VERDICT=AMBIG; fi
  say "EVENT $ref VERDICT $VERDICT"
}

drivers_idle
cp $PB/candidates_v13.yaml $C/candidates.yaml
say "c10rsi-queue start; applied candidates_v13.yaml"
gate_arm C_lr_10_rsi
if [ $VERDICT = AMBIG ]; then
  say "rule: C_lr_10_rsi ambiguous -> C_lr_10_rsi_s2"
  gate_arm C_lr_10_rsi_s2
fi
say "DONE c10rsi-queue"

#!/usr/bin/env bash
# Side-bisect next steps (user 2026-09-29 "run the next steps"; rules pre-registered in CHARTER "Next steps").
# Waits for the C_mo_s1 classification line, applies candidates_v7.yaml, then runs LRfa and LR4 (seed 3), each followed
# by a by-hand both-hands eval (scenarios_side_lr.yaml) with the env_vars of the arm's own driver eval. Adds seed 2 for
# an LR arm whose hands disagree or read INTERMEDIATE, and C_nb + C_nr if C_mo's 3-seed majority is not FAST.
# Log: queue.log ("EVENT ..." per classified arm, "DONE lr-queue" at the end).
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
mkdir -p $OUT $LROUT
say() { echo "[$(date -Is)] $*" >> $LOG; }
wait_unit() { until ! systemctl --user is-active -q $1; do sleep 30; done; }
drivers_idle() { while pgrep -f "isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py" >/dev/null; do sleep 30; done; }

stamp_of() {  # RUN_DIR -> newest complete side22_25 stamp for that run
  local rd=$1 tag
  tag=$(basename $rd | cut -d_ -f3-)
  ls -d $EV/$tag/side22_25__omni/seed001/*/ 2>/dev/null | while read d; do
    grep -q "$(basename $rd)" $d/run_meta.json 2>/dev/null && [ -f $d/complete.json ] && echo $d; done | tail -1
}

cls_of() {  # JSON NAME KEY
  python3 -c "import json;print(json.load(open('$1'))['$2'].get('$3'))"
}

newest_run_after() {
  local t=$1
  for d in $(ls -dt $RUNS/*/ | head -5); do
    [ $(stat -c %Y $d) -ge $t ] && { echo ${d%/}; return; }
  done
}

run_arm() {  # REF [launcher args] -> sets RD
  local ref=$1; shift
  local t0 unit lout
  for try in $(seq 40); do
    lout=$(cd $R && $L --campaign-dir $C --reference-chain $ref "$@" 2>&1)
    unit=$(echo "$lout" | head -1)
    case "$unit" in cands-*) break ;; esac
    say "launch of $ref refused (try $try): $(echo "$lout" | tail -1 | cut -c1-200); retry in 60 s"
    unit=""; sleep 60
  done
  if [ -z "$unit" ]; then say "STOP $ref: launch refused 40 times"; exit 5; fi
  t0=$(date +%s)
  say "launched $ref unit $unit"
  RD=""
  for _ in $(seq 60); do RD=$(newest_run_after $t0); [ -n "$RD" ] && break; sleep 15; done
  if [ -z "$RD" ]; then say "STOP $ref: no run dir appeared"; exit 4; fi
  say "$ref run dir $RD"
  wait_unit $unit
  tail -2 $C/orchestrator.log | cut -c1-300 >> $LOG
  local st; st=$(stamp_of $RD)
  if [ -z "$st" ]; then say "STOP $ref: no complete side22_25 stamp"; exit 6; fi
  $PY $PB/side22_classify.py --json $OUT/$ref.json $ref=$st 2>&1 | grep -E "lat_|CLASS" >> $LOG
  say "EVENT $ref CLASS $(cls_of $OUT/$ref.json $ref class)"
}

lr_eval() {  # REF RUN_DIR -> both-hands eval + classify; echoes "classL classR"
  local ref=$1 rd=$2 st envs
  st=$(stamp_of $rd)
  envs=$(python3 -c "import json;m=json.load(open('$st/run_meta.json'));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))")
  drivers_idle
  ( cd $R/parkour && env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH=$R/parkour/parkour_tasks:$R/parkour $envs \
      $CAP timeout 3600 $PY $EVAL --headless --manifest $LRMAN --scenario side22_25lr__omni \
      --checkpoint $rd/model_2999.pt --save-raw --no-plot --output-root $LROUT/$ref ) > $LROUT/$ref.log 2>&1
  local d; d=$(ls -d $LROUT/$ref/side22_25lr__omni/seed001/*/ 2>/dev/null | tail -1)
  if [ -z "$d" ] || [ ! -f $d/complete.json ]; then say "STOP $ref: lr eval produced no complete stamp"; exit 7; fi
  $PY $PB/side22_classify.py --json $OUT/${ref}_lr.json $ref=$d 2>&1 | grep -E "lat_|CLASS" >> $LOG
  local cl cr; cl=$(cls_of $OUT/${ref}_lr.json $ref class); cr=$(cls_of $OUT/${ref}_lr.json $ref class_r)
  say "EVENT ${ref}_lr CLASS_L $cl CLASS_R $cr"
  LRL=$cl; LRR=$cr
}

say "lr-queue armed; waiting for the C_mo seed-1 classification"
until grep -q "EVENT C_mo_s1 CLASS" $LOG; do sleep 30; done
drivers_idle
S1=$(grep "EVENT C_mo_s1 CLASS" $LOG | tail -1 | awk '{print $NF}')
cp $PB/candidates_v7.yaml $C/candidates.yaml
say "applied candidates_v7.yaml; C_mo_s1 class $S1"
ACC=(--accept-candidates-change "side bisect 2026-09-29: two-direction side test LR4/LRfa (+ seed-2 refs) and C_nb/C_nr kept available, per the CHARTER 'Next steps'")

EXTRA=()
first=1
for ref in LRfa LR4; do
  if [ $first = 1 ]; then run_arm $ref "${ACC[@]}"; first=0; else run_arm $ref; fi
  lr_eval $ref $RD
  if [ "$LRL" != "$LRR" ] || [ "$LRL" = "INTERMEDIATE" ] || [ "$LRR" = "INTERMEDIATE" ]; then
    EXTRA+=(${ref}_s2); say "rule: $ref hands $LRL/$LRR -> queue ${ref}_s2"
  fi
done
# C_mo majority over s3 INTERMEDIATE, s2 FAST, s1 $S1
if [ "$S1" != "FAST" ]; then EXTRA+=(C_nb C_nr); say "rule: C_mo majority not FAST (s1 $S1) -> queue C_nb C_nr"; fi
for ref in "${EXTRA[@]}"; do
  run_arm $ref
  case $ref in LR*) lr_eval $ref $RD ;; esac
done
say "DONE lr-queue"

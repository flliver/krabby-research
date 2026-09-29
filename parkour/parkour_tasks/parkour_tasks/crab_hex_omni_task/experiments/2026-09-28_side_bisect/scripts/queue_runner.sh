#!/usr/bin/env bash
# Side-speed bisection queue runner (decision defaults, CHARTER "Decision defaults"; user autonomy rule 2026-09-28).
# 1. wait for the running arm R (unit $1, run dir $2) to exit; classify it;
# 2. apply candidates_v2.yaml (C split + seed-2 references) once R's driver is gone;
# 3. run the pre-decided queue serially, classifying each arm:
#      R SLOW          -> C_mo C_band C_rsi C_s2
#      R FAST          -> O_s2, then STOP (interpretation "either bundle suffices" needs a seed-2 O first)
#      R other         -> R_s2 C_mo C_band C_rsi C_s2
# Every event is a line in $LOG; "EVENT" lines mark an arm classified, "STOP"/"DONE" end the runner.
set -u
RUNIT=$1; RRD=$2
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
C=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect
L=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/launch_candidates.sh
EV=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect
RUNS=$R/parkour/logs/rsl_rl/crab_hex_omni_flat_walk
OUT=$R/parkour/outputs/omni_plan/side_bisect/arms
PB=/home/nickmagus/.claude/plans/side-speed-bisection.evidence/phaseB
LOG=$R/parkour/outputs/omni_plan/side_bisect/queue.log
mkdir -p $OUT
say() { echo "[$(date -Is)] $*" >> $LOG; }

wait_unit() { until ! systemctl --user is-active -q $1; do sleep 30; done; }

classify() {  # NAME RUN_DIR -> echoes class; writes $OUT/NAME.json
  local name=$1 rd=$2 tag stamp
  tag=$(basename $rd | cut -d_ -f3-)
  stamp=$(ls -d $EV/$tag/side22_25__omni/seed001/*/ 2>/dev/null | while read d; do
    grep -q "$(basename $rd)" $d/run_meta.json 2>/dev/null && echo $d; done | tail -1)
  if [ -z "$stamp" ] || [ ! -f "$stamp/complete.json" ]; then echo "NOSTAMP"; return; fi
  $PY $PB/side22_classify.py --json $OUT/$name.json $name=$stamp 2>&1 | grep -E "lat_|CLASS" >> $LOG
  python3 -c "import json;print(json.load(open('$OUT/$name.json'))['$name'].get('class'))"
}

newest_run_after() {  # epoch -> newest run dir created after it
  local t=$1
  for d in $(ls -dt $RUNS/*/ | head -5); do
    [ $(stat -c %Y $d) -ge $t ] && { echo ${d%/}; return; }
  done
}

run_arm() {  # REF [extra launcher args...] -> echoes class
  local ref=$1; shift
  local t0 unit rd cls
  t0=$(date +%s)
  local lout
  lout=$(cd $R && $L --campaign-dir $C --reference-chain $ref "$@" 2>&1)
  unit=$(echo "$lout" | head -1)
  say "launched $ref unit $unit :: $(echo "$lout" | tail -1)"
  rd=""
  for _ in $(seq 60); do rd=$(newest_run_after $t0); [ -n "$rd" ] && break; sleep 15; done
  if [ -z "$rd" ]; then say "STOP $ref: no run dir appeared (unit $unit: $(systemctl --user is-active $unit))"; exit 4; fi
  say "$ref run dir $rd"
  wait_unit $unit
  tail -2 $C/orchestrator.log | cut -c1-300 >> $LOG
  cls=$(classify $ref $rd)
  say "EVENT $ref CLASS $cls"
  echo $cls
}

say "runner start; waiting for R ($RUNIT)"
wait_unit $RUNIT
RCLS=$(classify R $RRD)
say "EVENT R CLASS $RCLS"
cp $PB/candidates_v2.yaml $C/candidates.yaml
say "applied candidates_v2.yaml (C split + seed-2 references)"
ACC=(--accept-candidates-change "side bisect 2026-09-29: C read FAST at s3; add the C split (C_mo/C_band/C_rsi) and seed-2 references per the CHARTER decision defaults")
case $RCLS in
  SLOW) Q=(C_mo C_band C_rsi C_s2) ;;
  FAST) Q=(O_s2) ;;
  *)    Q=(R_s2 C_mo C_band C_rsi C_s2) ;;
esac
say "queue: ${Q[*]}"
first=1
for ref in "${Q[@]}"; do
  if [ $first = 1 ]; then run_arm $ref "${ACC[@]}" > /dev/null; first=0; else run_arm $ref > /dev/null; fi
done
if [ "$RCLS" = "FAST" ]; then say "STOP R FAST: O_s2 done, the 'either suffices' reading needs a decision"; else say "DONE queue"; fi

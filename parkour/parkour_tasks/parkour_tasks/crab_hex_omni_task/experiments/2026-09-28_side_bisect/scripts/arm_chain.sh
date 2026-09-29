#!/usr/bin/env bash
# Wait for an arm's driver unit to exit, classify its side22_25 eval stamp, and launch NEXT only when the
# finished arm is SLOW (the plan's pre-registered path: O SLOW -> C, C done -> R). Any other class, a missing
# stamp, or a stamp from the wrong checkpoint launches nothing and exits for a decision.
# Usage: arm_chain.sh UNIT RUN_DIR NAME [NEXT_ARM [NEXT_SEED]]
set -u
U=$1; RD=$2; NAME=$3; NEXT=${4:-}; NSEED=${5:-3}
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
C=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect
EV=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect
OUT=$R/parkour/outputs/omni_plan/side_bisect/arms
mkdir -p $OUT
until ! systemctl --user is-active -q $U; do sleep 30; done
echo "== unit $U exited $(date -Is)"
tail -4 $C/orchestrator.log
TAG=$(basename $RD | cut -d_ -f3-)
STAMP=$(ls -d $EV/$TAG/side22_25__omni/seed001/*/ 2>/dev/null | while read d; do
  grep -q "$(basename $RD)" $d/run_meta.json 2>/dev/null && echo $d; done | tail -1)
if [ -z "$STAMP" ] || [ ! -f "$STAMP/complete.json" ]; then echo "NO COMPLETE STAMP for $RD under $EV/$TAG"; exit 2; fi
echo "stamp $STAMP"
$PY ~/.claude/plans/side-speed-bisection.evidence/phaseB/side22_classify.py --json $OUT/$NAME.json $NAME=$STAMP 2>&1 | grep -E "ckpt|lat_|CLASS"
CLS=$(python3 -c "import json;print(json.load(open('$OUT/$NAME.json'))['$NAME'].get('class'))")
echo "CLASS $NAME $CLS"
if [ "$CLS" = "SLOW" ] && [ -n "$NEXT" ]; then
  cd $R && $C/../tools/launch_candidates.sh --campaign-dir $C --reference-chain $NEXT --seed $NSEED 2>&1 | tail -2
  echo "LAUNCHED $NEXT s$NSEED $(date -Is)"
else
  echo "NOT LAUNCHING (class $CLS, next '${NEXT}')"
fi

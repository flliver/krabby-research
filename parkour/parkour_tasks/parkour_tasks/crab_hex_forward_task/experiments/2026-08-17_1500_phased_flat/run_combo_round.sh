#!/bin/bash
# Phase B combo round (user-approved 2026-08-17): six combo screens, serial, overnight.
# Each arm: 3k-iter resume from the C2 base -> gait_eval on the pinned flat scenario ->
# one summary line to results_combo.tsv. Emits one progress line per arm stage.
set -u
CAMP=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-17_1500_phased_flat
BASE=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-15_1530_task1_velocity/screen_C2_airtime/logs/rsl_rl/crab_hex_flat_walk/2026-08-15_18-01-57/model_2999.pt
PY=/home/nickmagus/krabby/isaac_venv/bin/python
TRAIN=/home/nickmagus/krabby/krabby-research/parkour/scripts/rsl_rl/train.py
EVAL=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
EVALROOT=/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/v1/flat_walk_forward/seed001
RESULTS=$CAMP/results_combo.tsv
echo -e "arm\tratio\ttripod\tcompletion\treward\tfailure\treversal_income\trev_per_s" > $RESULTS

declare -A ENVS
ARMS=(CB1_pow_rev03 CB2_pow_rev03_spin02 CB3_pow_spin02 CB4_pow_rev06 CB5_rev03_spin02_lock01 CB6_all)
ENVS[CB1_pow_rev03]="-E KRABBY_POWER_W=-0.001 -E KRABBY_REVERSAL_W=-0.3"
ENVS[CB2_pow_rev03_spin02]="-E KRABBY_POWER_W=-0.001 -E KRABBY_REVERSAL_W=-0.3 -E KRABBY_SPIN_REWARD_W=0.2"
ENVS[CB3_pow_spin02]="-E KRABBY_POWER_W=-0.001 -E KRABBY_REVERSAL_W=0 -E KRABBY_SPIN_REWARD_W=0.2"
ENVS[CB4_pow_rev06]="-E KRABBY_POWER_W=-0.001 -E KRABBY_REVERSAL_W=-0.6"
ENVS[CB5_rev03_spin02_lock01]="-E KRABBY_REVERSAL_W=-0.3 -E KRABBY_SPIN_REWARD_W=0.2 -E KRABBY_PHASE_LOCK_W=0.1"
ENVS[CB6_all]="-E KRABBY_POWER_W=-0.001 -E KRABBY_REVERSAL_W=-0.3 -E KRABBY_SPIN_REWARD_W=0.2 -E KRABBY_PHASE_LOCK_W=0.1"

for ARM in "${ARMS[@]}"; do
  D=$CAMP/$ARM; mkdir -p $D; cd $D
  echo "[$(date +%H:%M)] $ARM: training"
  OMNI_KIT_ACCEPT_EULA=yes systemd-run --user --scope -q -p MemoryMax=45G \
    ${ENVS[$ARM]} -E OMNI_KIT_ACCEPT_EULA=yes timeout 10800 \
    $PY $TRAIN --task Isaac-Crab-Hex-Flat-Walk-v0 --headless --num_envs 256 --seed 1 \
    --resume --checkpoint $BASE --max_iterations 3000 > $D/train.log 2>&1
  TEXIT=$?
  CK=$(ls $D/logs/rsl_rl/crab_hex_flat_walk/*/model_5998.pt 2>/dev/null | head -1)
  if [ -z "$CK" ]; then
    echo "[$(date +%H:%M)] $ARM: TRAIN FAILED (exit $TEXIT, no model_5998)"
    echo -e "$ARM\tTRAIN_FAIL\t-\t-\t-\t-\t-\t-" >> $RESULTS
    continue
  fi
  echo "[$(date +%H:%M)] $ARM: eval"
  cd /home/nickmagus/krabby/krabby-research/parkour
  OMNI_KIT_ACCEPT_EULA=yes systemd-run --user --scope -q -p MemoryMax=45G \
    -E OMNI_KIT_ACCEPT_EULA=yes timeout 1800 \
    $PY $EVAL --headless --scenario flat_walk_forward --checkpoint $CK \
    --allow-checkpoint-sha-mismatch --no-plot > $D/gait_eval.log 2>&1
  R=$(ls -td $EVALROOT/* | head -1)
  $PY - "$ARM" "$D/train.log" "$R" "$RESULTS" <<'PYEOF'
import json,glob,sys,re,statistics as st
arm,tlog,R,res=sys.argv[1:5]
try:
    a=json.load(open(R+'/scenario_metrics.json'))['aggregate']
    ratio=round(a['shaft_one_direction_ratio']['median'],4)
    tripod=round(a['tripod_score']['median'],3)
    comp=a['schedule_completion_rate']
    eps=[json.load(open(f)) for f in glob.glob(R+'/metrics/episode_*.json')]
    rev=round(st.median([e['shaft_spin']['reversals_per_s_median'] for e in eps if e.get('shaft_spin')]),2)
except Exception as ex:
    ratio=tripod=comp=rev=f'EVAL_FAIL'
txt=open(tlog,errors='ignore').read()
def last(pat):
    m=re.findall(pat,txt)
    return m[-1] if m else '-'
reward=last(r'Mean reward:\s*([\d.+-]+)')
fail=last(r'crab_failure[^\d-]*([\d.]+)')
ri=last(r'penalty_motor_direction_reversal[^\d-]*(-?[\d.]+)')
open(res,'a').write(f"{arm}\t{ratio}\t{tripod}\t{comp}\t{reward}\t{fail}\t{ri}\t{rev}\n")
print(f"[done] {arm}: ratio={ratio} tripod={tripod} completion={comp}")
PYEOF
done
echo "[$(date +%H:%M)] COMBO ROUND COMPLETE"
cat $RESULTS

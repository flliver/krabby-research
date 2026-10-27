#!/usr/bin/env bash
# Step A (CHARTER "Mirror canonicalization"): canonical evals of the fast one-direction heads.
set -u
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
C=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect
EVAL=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
ARMS=$R/parkour/outputs/omni_plan/side_bisect/arms
OUT=$R/parkour/outputs/omni_plan/side_bisect/canon
LOG=$R/parkour/outputs/omni_plan/side_bisect/queue.log
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
mkdir -p $OUT
say() { echo "[$(date -Is)] $*" >> $LOG; }
drivers_idle() { while pgrep -f "isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py" >/dev/null; do sleep 20; done; }

run() {  # HEAD MODE MANIFEST SCENARIO [extra]
  local head=$1 mode=$2 man=$3 sid=$4; shift 4
  local st ck envs
  st=$(python3 -c "import json;print(json.load(open('$ARMS/$head.json'))['$head']['stamp'])")
  ck=$(python3 -c "import json;print(json.load(open('$ARMS/$head.json'))['$head']['checkpoint'])")
  envs=$(python3 -c "import json;m=json.load(open('$st/run_meta.json'));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))")
  drivers_idle
  ( cd $R/parkour && env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH=$R/parkour/parkour_tasks:$R/parkour $envs \
      $CAP timeout 3600 $PY $EVAL --headless --manifest $man --scenario $sid --checkpoint $ck --save-raw --no-plot \
      --canon $mode "$@" --output-root $OUT/$head/$mode ) > $OUT/${head}_${mode}_${sid}.log 2>&1
  local rc=$?
  local d; d=$(ls -d $OUT/$head/$mode/$sid/seed001/*/ 2>/dev/null | tail -1)
  local canon; canon=$([ -n "$d" ] && python3 -c "import json;m=json.load(open('$d/run_meta.json'));c=m.get('canon');print('canon', c.get('mode'), 'flips', c.get('flips_total', c.get('total_flips')) if c else None)" 2>/dev/null)
  say "canon-eval $head $mode $sid rc $rc stamp $d $canon"
}

say "canon-evals start"
# smoke
run C_mo_s2 wave $C/scenarios_side_lr.yaml side22_25lr__omni --episodes 4
S=$(ls -d $OUT/C_mo_s2/wave/side22_25lr__omni/seed001/*/ 2>/dev/null | tail -1)
if [ -z "$S" ] || ! python3 -c "import json,sys;m=json.load(open('$S/run_meta.json'));sys.exit(0 if m.get('canon') else 1)"; then
  say "STOP canon smoke: no canon block (see $OUT/C_mo_s2_wave_side22_25lr__omni.log)"; exit 3; fi
python3 -c "import numpy as np,glob;f=sorted(glob.glob('$S/raw/episode_*.npz'));z=np.load(f[-1]);print('canon_flag' in z.files, z['canon_flag'].mean() if 'canon_flag' in z.files else None)" >> $LOG 2>&1
rm -rf $OUT/C_mo_s2/wave   # the smoke is not a result
say "canon smoke OK"
for head in C_mo_s2 C C_s2 C_mo_s1; do
  run $head wave $C/scenarios_side_lr.yaml side22_25lr__omni
  run $head cmd  $C/scenarios_side_lr.yaml side22_25lr__omni
  run $head cmd  $C/scenarios_side_reversal.yaml rev_direct__omni
  run $head cmd  $C/scenarios_side_reversal.yaml rev_pause__omni
done
say "DONE canon-evals"

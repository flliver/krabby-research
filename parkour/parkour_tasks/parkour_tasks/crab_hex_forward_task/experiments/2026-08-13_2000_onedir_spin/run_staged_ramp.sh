#!/bin/bash
# Staged-ramp experiment (2026-08-15, user-approved): engineer the one observed
# walking->spin transition. Phase A (5k): walk-first — schedule -0.1, reversal OFF,
# lock OFF. Phase B (resume +10k): reversal -0.3 + lock +0.1 + mechanical-power -0.001
# (replayed selector: spin ~42% cheaper than oscillation). 2 seeds, serial.
# Gates: ratio >= 0.8 AND completion >= 0.9, slip < 15%.
set -u
CAMP=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_2000_onedir_spin
PY=/home/nickmagus/krabby/isaac_venv/bin/python
TRAIN=/home/nickmagus/krabby/krabby-research/parkour/scripts/rsl_rl/train.py
EVAL=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
SUMMARY=$CAMP/staged_ramp_summary.md
echo "# Staged ramp — started $(date '+%F %T')" >> $SUMMARY
echo "| seed | phase | reward | 1-dir ratio | completion | slip | tripod | verdict |" >> $SUMMARY
echo "|------|-------|--------|-------------|------------|------|--------|---------|" >> $SUMMARY

run_eval () {  # $1=dir  $2=ckpt  $3=seed  $4=phase  $5=reward
  cd $1
  OMNI_KIT_ACCEPT_EULA=yes systemd-run --user --scope -q -p MemoryMax=45G \
    -E OMNI_KIT_ACCEPT_EULA=yes timeout 1800 $PY $EVAL --headless \
    --scenario flat_walk_forward --checkpoint $2 --allow-checkpoint-sha-mismatch \
    > gait_eval_$4.log 2>&1 || { echo "| $3 | $4 | $5 | EVAL FAIL | - | - | - | ERROR |" >> $SUMMARY; return 1; }
  R=$(ls -td /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/v1/flat_walk_forward/seed001/*/ | head -1)
  $PY - << PYEOF >> $SUMMARY
import json
s = json.load(open("$R/scenario_metrics.json"))
a = s.get("aggregate", s)
ratio = a["shaft_one_direction_ratio"]["median"]
comp = a["schedule_completion_rate"]
slip = a["slip_ratio"]["median"] or -1
trip = a["tripod_score"]["median"] if a["tripod_score"]["median"] is not None else -1
v = "WINNER" if (ratio >= 0.8 and comp >= 0.9 and slip < 0.15) else ("partial" if ratio >= 0.3 else "oscillator")
print(f"| $3 | $4 | $5 | {ratio:.3f} | {comp:.2f} | {slip:.3f} | {trip:.3f} | {v} |")
PYEOF
}

for SEED in 1 2; do
  # --- Phase A: walk-first ---
  DA=$CAMP/ramp_seed${SEED}_phaseA
  mkdir -p $DA && cd $DA
  echo "[ramp] seed $SEED phase A start $(date '+%T')"
  OMNI_KIT_ACCEPT_EULA=yes systemd-run --user --scope -q -p MemoryMax=45G \
    -E KRABBY_REVERSAL_W=0.0 -E KRABBY_CAM_SCHED_W=-0.1 -E KRABBY_PHASE_LOCK_W=0.0 \
    -E OMNI_KIT_ACCEPT_EULA=yes \
    timeout 10800 $PY $TRAIN --task Isaac-Crab-Hex-Flat-Walk-v0 --headless \
    --num_envs 256 --seed $SEED --max_iterations 5000 > train.log 2>&1 || {
      echo "| $SEED | A | TRAIN FAIL | - | - | - | - | ERROR |" >> $SUMMARY; continue; }
  RA=$(tr '\r' '\n' < train.log | grep "Mean reward" | tail -1 | awk '{print $NF}')
  CKA=$DA/$(ls -d logs/rsl_rl/crab_hex_flat_walk/*/ | head -1)model_4999.pt
  run_eval $DA $CKA $SEED A $RA

  # --- Phase B: ramp spin pressure on the established walker ---
  DB=$CAMP/ramp_seed${SEED}_phaseB
  mkdir -p $DB && cd $DB
  echo "[ramp] seed $SEED phase B start $(date '+%T')"
  OMNI_KIT_ACCEPT_EULA=yes systemd-run --user --scope -q -p MemoryMax=45G \
    -E KRABBY_REVERSAL_W=-0.3 -E KRABBY_CAM_SCHED_W=-0.1 -E KRABBY_PHASE_LOCK_W=0.1 \
    -E KRABBY_POWER_W=-0.001 -E OMNI_KIT_ACCEPT_EULA=yes \
    timeout 21600 $PY $TRAIN --task Isaac-Crab-Hex-Flat-Walk-v0 --headless \
    --num_envs 256 --seed $SEED --max_iterations 10000 --resume --checkpoint $CKA \
    > train.log 2>&1 || {
      echo "| $SEED | B | TRAIN FAIL | - | - | - | - | ERROR |" >> $SUMMARY; continue; }
  RB=$(tr '\r' '\n' < train.log | grep "Mean reward" | tail -1 | awk '{print $NF}')
  CKB=$DB/$(ls -d logs/rsl_rl/crab_hex_flat_walk/*/ | head -1)model_14999.pt
  run_eval $DB $CKB $SEED B $RB
done
echo "" >> $SUMMARY
echo "Finished $(date '+%F %T')" >> $SUMMARY
echo "[ramp] ALL DONE"

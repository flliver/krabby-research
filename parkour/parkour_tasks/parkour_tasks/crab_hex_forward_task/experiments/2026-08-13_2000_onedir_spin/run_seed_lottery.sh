#!/bin/bash
# Seed-basin lottery (plan of record, user-approved overnight run 2026-08-14).
# 7 seeds x 5k iters, round-4 shaping (reversal -0.3 baked; schedule -0.1; lock +0.1),
# gait eval per seed, spin one-direction ratio as the selector.
# Precedent: parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-12_1550_seed_basin_search.
set -u
CAMP=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_2000_onedir_spin
PY=/home/nickmagus/krabby/isaac_venv/bin/python
TRAIN=/home/nickmagus/krabby/krabby-research/parkour/scripts/rsl_rl/train.py
EVAL=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
SUMMARY=$CAMP/lottery_summary.md
echo "# Seed-basin lottery — started $(date '+%F %T')" >> $SUMMARY
echo "| seed | reward@5k | 1-dir ratio | completion | slip | tripod | verdict |" >> $SUMMARY
echo "|------|-----------|-------------|------------|------|--------|---------|" >> $SUMMARY

for SEED in 2 3 4 5 6 7 8; do
  D=$CAMP/lottery_seed$SEED
  mkdir -p $D && cd $D
  echo "[lottery] seed $SEED train start $(date '+%T')"
  OMNI_KIT_ACCEPT_EULA=yes systemd-run --user --scope -q -p MemoryMax=45G \
    -E KRABBY_CAM_SCHED_W=-0.1 -E KRABBY_PHASE_LOCK_W=0.1 -E OMNI_KIT_ACCEPT_EULA=yes \
    timeout 10800 $PY $TRAIN --task Isaac-Crab-Hex-Flat-Walk-v0 --headless \
    --num_envs 256 --seed $SEED --max_iterations 5000 > train.log 2>&1
  TR=$?
  if [ $TR -ne 0 ]; then
    echo "| $SEED | TRAIN FAIL ($TR) | - | - | - | - | ERROR |" >> $SUMMARY
    echo "[lottery] seed $SEED TRAIN FAILED ($TR)"
    continue
  fi
  REWARD=$(tr '\r' '\n' < train.log | grep "Mean reward" | tail -1 | awk '{print $NF}')
  CKPT=$(ls -d logs/rsl_rl/crab_hex_flat_walk/*/ | head -1)model_4999.pt
  echo "[lottery] seed $SEED eval start $(date '+%T')"
  OMNI_KIT_ACCEPT_EULA=yes systemd-run --user --scope -q -p MemoryMax=45G \
    -E OMNI_KIT_ACCEPT_EULA=yes timeout 1800 $PY $EVAL --headless \
    --scenario flat_walk_forward --checkpoint $CKPT --allow-checkpoint-sha-mismatch \
    > gait_eval.log 2>&1
  EV=$?
  if [ $EV -ne 0 ]; then
    echo "| $SEED | $REWARD | EVAL FAIL ($EV) | - | - | - | ERROR |" >> $SUMMARY
    continue
  fi
  R=$(ls -td /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/v1/flat_walk_forward/seed001/*/ | head -1)
  LINE=$($PY - << PYEOF
import json
s = json.load(open("$R/scenario_metrics.json"))
a = s.get("aggregate", s)
ratio = a["shaft_one_direction_ratio"]["median"]
comp = a["schedule_completion_rate"]
slip = a["slip_ratio"]["median"] if a["slip_ratio"]["median"] is not None else -1
trip = a["tripod_score"]["median"] if a["tripod_score"]["median"] is not None else -1
verdict = "WINNER" if ratio >= 0.8 else ("partial" if ratio >= 0.3 else "oscillator")
print(f"| $SEED | $REWARD | {ratio:.3f} | {comp:.2f} | {slip:.3f} | {trip:.3f} | {verdict} |")
PYEOF
)
  echo "$LINE" >> $SUMMARY
  echo "[lottery] seed $SEED done: $LINE"
done
echo "" >> $SUMMARY
echo "Finished $(date '+%F %T')" >> $SUMMARY
echo "[lottery] ALL DONE"

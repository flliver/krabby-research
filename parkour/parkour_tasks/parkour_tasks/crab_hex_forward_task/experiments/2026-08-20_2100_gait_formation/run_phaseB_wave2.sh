#!/usr/bin/env bash
# Phase B wave 2 (2026-08-21): dense-gradient bootstrap ladder. Wave-1 found the tripod
# crossing term gradient-dead at the creep-shuffle (income fired <=15/1000 iters, max 1e-4).
# Ladder: excess-contact penalty + airtime pay CAUSE stepping; TRIPOD_MIN_AMP 0.05 lets
# embryonic swaps pay; tripod 0.3 remains the target. Controls: each dense term alone.
set -u
cd "$(dirname "$0")/../../parkour"
PY=/home/nickmagus/krabby/isaac_venv/bin/python
D=../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-20_2100_gait_formation
M=parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_v2.yaml
export OMNI_KIT_ACCEPT_EULA=yes TERM=xterm

BASE="KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_TRACK_L1_W=-1.0 KRABBY_TRACK_SIGMA2=0.25"
LADDER="KRABBY_EXCESS_CONTACT_W=-0.4 KRABBY_AIRTIME_W=1.2 KRABBY_TRIPOD_W=0.3 KRABBY_TRIPOD_MIN_AMP=0.05"

declare -A ARMS=(
  [B_XC]="KRABBY_EXCESS_CONTACT_W=-0.4"
  [B_AIR]="KRABBY_AIRTIME_W=1.2"
  [B_DENSE_T30]="$LADDER"
  [B_DENSE_T30C]="$LADDER KRABBY_CAM_CLIP_LO=0.1"
  [B_DENSE_T30S]="$LADDER KRABBY_TRACK_SIGMA2=0.1"
)
ORDER=(B_DENSE_T30 B_DENSE_T30C B_DENSE_T30S B_XC B_AIR)

for arm in "${ORDER[@]}"; do
  delta=${ARMS[$arm]}
  echo "=== $arm : $delta"
  env $BASE $delta timeout 3600 $PY scripts/rsl_rl/train.py \
    --task Isaac-Crab-Hex-Flat-Walk-v0 --headless --num_envs 256 --seed 1 \
    --max_iterations 1000 > "$D/${arm}_train.log" 2>&1
  echo "${arm}_TRAIN_EXIT=$?"
  RUN=$(ls -t logs/rsl_rl/crab_hex_flat_walk/ | head -1)
  CKPT=$(ls logs/rsl_rl/crab_hex_flat_walk/$RUN/ | grep -E "^model_[0-9]+\.pt$" | sort -t_ -k2 -n | tail -1)
  echo "${arm}_RUN=$RUN CKPT=$CKPT"
  env $BASE $delta timeout 1500 $PY \
    parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py \
    --headless --manifest $M --scenario flat_walk_slow_v2 \
    --checkpoint "logs/rsl_rl/crab_hex_flat_walk/$RUN/$CKPT" --no-plot \
    --output-root logs/rsl_rl/gait_eval/gait_formation > "$D/${arm}_eval.log" 2>&1
  echo "${arm}_EVAL_EXIT=$?"
  grep -E "tripod_score |tracking_ratio |schedule_completion|terminations" "$D/${arm}_eval.log" | tail -4
  nz=$(grep -oE "reward_tripod_schedule: [0-9.]+" "$D/${arm}_train.log" | grep -cv ": 0.0000")
  echo "${arm}_TRIPOD_INCOME_NONZERO_ITERS=$nz"
done
echo "PHASE_B_WAVE2_DONE"

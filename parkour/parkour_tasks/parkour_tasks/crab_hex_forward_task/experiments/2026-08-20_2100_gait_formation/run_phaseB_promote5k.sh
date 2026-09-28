#!/usr/bin/env bash
# Phase B promotion wave (5k, 2026-08-21 overnight): the four best formation screens at
# consolidation scale. The scale curve says survival consolidates 1k->5k+; the open question
# is whether stepping survives that consolidation or creep reasserts.
set -u
cd "$(dirname "$0")/../../parkour"
PY=/home/nickmagus/krabby/isaac_venv/bin/python
D=../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-20_2100_gait_formation
M=parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_v2.yaml
export OMNI_KIT_ACCEPT_EULA=yes TERM=xterm

BASE="KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_TRACK_L1_W=-1.0 KRABBY_TRACK_SIGMA2=0.25"

declare -A ARMS=(
  [P_AIR]="KRABBY_AIRTIME_W=1.2"
  [P_AIR_T30]="KRABBY_AIRTIME_W=1.2 KRABBY_TRIPOD_W=0.3 KRABBY_TRIPOD_MIN_AMP=0.05"
  [P_DENSE_T30S]="KRABBY_EXCESS_CONTACT_W=-0.4 KRABBY_AIRTIME_W=1.2 KRABBY_TRIPOD_W=0.3 KRABBY_TRIPOD_MIN_AMP=0.05 KRABBY_TRACK_SIGMA2=0.1"
  [P_T30S]="KRABBY_TRIPOD_W=0.3 KRABBY_TRACK_SIGMA2=0.1"
)
ORDER=(P_AIR P_AIR_T30 P_DENSE_T30S P_T30S)

for arm in "${ORDER[@]}"; do
  delta=${ARMS[$arm]}
  echo "=== $arm : $delta"
  env $BASE $delta timeout 10800 $PY scripts/rsl_rl/train.py \
    --task Isaac-Crab-Hex-Flat-Walk-v0 --headless --num_envs 256 --seed 1 \
    --max_iterations 5000 > "$D/${arm}_train.log" 2>&1
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
echo "PHASE_B_PROMOTE_DONE"

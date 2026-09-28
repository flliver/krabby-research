#!/usr/bin/env bash
# PLAN E Phase 4 wave F: lift arms on the baked clock+RSI stack.
set -u
cd "$(dirname "$0")/../../parkour"
PY=/home/nickmagus/krabby/isaac_venv/bin/python
D=../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2
M=parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_v2.yaml
export OMNI_KIT_ACCEPT_EULA=yes TERM=xterm

BANK="$PWD/../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/rsi_bank_E1.npz"
STACK="KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_TRACK_L1_W=-1.0 KRABBY_TRACK_SIGMA2=0.1 KRABBY_CLOCK_W=1.0 KRABBY_RSI_FRAC=0.2 KRABBY_RSI_BANK=$BANK"

declare -A ARMS=(
  [F1_clear1]="KRABBY_FOOT_CLEAR_FLAT=1 KRABBY_FOOT_CLEAR_W=1.0"
  [F2_clear2]="KRABBY_FOOT_CLEAR_FLAT=1 KRABBY_FOOT_CLEAR_W=2.0"
  [F3_clear1_min]="KRABBY_FOOT_CLEAR_FLAT=1 KRABBY_FOOT_CLEAR_W=1.0 KRABBY_SWING_MIN_CLEAR_W=-0.4"
)
ORDER=(F1_clear1 F2_clear2 F3_clear1_min)

for arm in "${ORDER[@]}"; do
  delta=${ARMS[$arm]}
  echo "=== $arm : $delta"
  env $STACK $delta timeout 10800 $PY scripts/rsl_rl/train.py \
    --task Isaac-Crab-Hex-Flat-Walk-v0 --headless --num_envs 256 --seed 1 \
    --max_iterations 5000 > "$D/${arm}_train.log" 2>&1
  echo "${arm}_TRAIN_EXIT=$?"
  RUN=$(ls -t logs/rsl_rl/crab_hex_flat_walk/ | head -1)
  CKPT=$(ls logs/rsl_rl/crab_hex_flat_walk/$RUN/ | grep -E "^model_[0-9]+\.pt$" | sort -t_ -k2 -n | tail -1)
  echo "${arm}_RUN=$RUN CKPT=$CKPT"
  env $STACK $delta timeout 1500 $PY \
    parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py \
    --headless --manifest $M --scenario flat_walk_slow_v2 \
    --checkpoint "logs/rsl_rl/crab_hex_flat_walk/$RUN/$CKPT" --no-plot \
    --output-root logs/rsl_rl/gait_eval/gait_formation_v2 > "$D/${arm}_eval.log" 2>&1
  echo "${arm}_EVAL_EXIT=$?"
  grep -E "tripod_score |tracking_ratio |schedule_completion|terminations" "$D/${arm}_eval.log" | tail -4
done
echo "WAVE_F_DONE"

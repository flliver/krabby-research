#!/usr/bin/env bash
# Phase B screen wave 1 (2026-08-21, post Phase-A bake): from-scratch 1k screens on the
# C5 survival stack. Ranked on tripod formation + tracking_ratio (completion = constraint).
# Priority arm per PLAN D: clamp + tripod income (the lit-review "never clamp alone" recipe).
# Serial per GPU rule; each arm: 1k train then n=100 eval on flat_walk_slow_v2 (in-range band).
set -u
cd "$(dirname "$0")/../../parkour"
PY=/home/nickmagus/krabby/isaac_venv/bin/python
D=../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-20_2100_gait_formation
M=parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/eval/scenarios_v2.yaml
export OMNI_KIT_ACCEPT_EULA=yes TERM=xterm

BASE="KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_TRACK_L1_W=-1.0 KRABBY_TRACK_SIGMA2=0.25"

declare -A ARMS=(
  [B_T30C]="KRABBY_TRIPOD_W=0.3 KRABBY_CAM_CLIP_LO=0.1"
  [B_T30]="KRABBY_TRIPOD_W=0.3"
  [B_T15]="KRABBY_TRIPOD_W=0.15"
  [B_T50]="KRABBY_TRIPOD_W=0.5"
  [B_CLAMP]="KRABBY_CAM_CLIP_LO=0.1"
  [B_T30S]="KRABBY_TRIPOD_W=0.3 KRABBY_TRACK_SIGMA2=0.1"
)
ORDER=(B_T30C B_T30 B_T15 B_T50 B_CLAMP B_T30S)

for arm in "${ORDER[@]}"; do
  delta=${ARMS[$arm]}
  # B_T30S overrides SIGMA2: later assignment wins with env, so append delta after BASE.
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
done
echo "PHASE_B_WAVE1_DONE"

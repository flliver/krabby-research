#!/usr/bin/env bash
# Chain #6: 2b1 as a goal-income DOSE RAMP (25% -> 100%), critic reset at boundaries.
set -u
cd "$(dirname "$0")/../../parkour"
PY=/home/nickmagus/krabby/isaac_venv/bin/python
D=../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-26_0200_teacher_handoff
RESET=../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-17_1500_phased_flat/B1_critic_reset/make_critic_reset.py
export OMNI_KIT_ACCEPT_EULA=yes TERM=xterm
BASE="KRABBY_LIN_VEL_X=0.20:0.40 KRABBY_CLOCK_W=1.0 KRABBY_MIN_ACTUAL_SPEED=0.08"
GATE=0.35

leg () {
  local name=$1 mode=$2 extra=$3 ckpt=$4 cap=$5
  local chunk=1 fail=1.0 ok=0
  while [ $chunk -le $cap ]; do
    echo "=== $name chunk $chunk"
    env $BASE $extra KRABBY_HEX_TEACHER_MODE=$mode timeout -k 60 14400 $PY scripts/rsl_rl/train.py \
      --task Isaac-Crab-Hex-Teacher-v0 --headless --num_envs 256 --seed 2 \
      --max_iterations 2000 --resume --checkpoint "$ckpt" > "$D/${name}_c${chunk}_train.log" 2>&1
    echo "${name}_c${chunk}_EXIT=$?"
    RUN=$(ls -t logs/rsl_rl/crab_hex_teacher/ | head -1)
    CK=$(ls logs/rsl_rl/crab_hex_teacher/$RUN/ | grep -E "^model_[0-9]+\.pt$" | sort -t_ -k2 -n | tail -1)
    ckpt="$PWD/logs/rsl_rl/crab_hex_teacher/$RUN/$CK"
    fail=$(grep -oE "Episode_Termination/crab_failure: [0-9.]+" "$D/${name}_c${chunk}_train.log" | tail -20 | awk -F': ' '{s+=$2} END {print s/NR}')
    echo "${name}_c${chunk}: failure_tail20=$fail | ckpt=$RUN/$CK"
    ok=$(python3 -c "print(1 if $fail < $GATE else 0)")
    [ "$ok" = "1" ] && { echo "${name}_GATE_PASS at chunk $chunk"; break; }
    chunk=$((chunk+1))
  done
  [ "$ok" != "1" ] && echo "${name}_GATE_FAIL (failure $fail)"
  LEG_CKPT="$ckpt"; LEG_OK="$ok"
}

START="$PWD/logs/rsl_rl/crab_hex_teacher/2026-08-26_12-38-32/model_26899_critic_reset.pt"
leg 2b1_q "2b1" "KRABBY_GOAL_VEL_W=0.19 KRABBY_YAW_W=0.05" "$START" 3
[ "$LEG_OK" != "1" ] && { echo "CHAIN_STOPPED_AT_2B1_QUARTER"; exit 1; }
leg 2b1_full "2b1" "" "$LEG_CKPT" 3
[ "$LEG_OK" != "1" ] && { echo "CHAIN_STOPPED_AT_2B1_FULL"; exit 1; }
RC="${LEG_CKPT%.pt}_critic_reset.pt"; $PY $RESET --src "$LEG_CKPT" --dst "$RC"
leg 2b2 "2b2" "" "$RC" 4
echo "CHAIN_V3_DONE (2b2 gate: $LEG_OK, final=$LEG_CKPT)"

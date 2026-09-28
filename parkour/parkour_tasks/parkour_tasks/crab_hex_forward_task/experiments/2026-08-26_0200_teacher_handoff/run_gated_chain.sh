#!/usr/bin/env bash
# Gated teacher hand-off chain #2 (2026-08-26): promote a stage only when its failure
# rate clears the gate; keep training in 2k chunks up to a cap otherwise.
set -u
cd "$(dirname "$0")/../../parkour"
PY=/home/nickmagus/krabby/isaac_venv/bin/python
D=../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-26_0200_teacher_handoff
export OMNI_KIT_ACCEPT_EULA=yes TERM=xterm
BAND="KRABBY_LIN_VEL_X=0.20:0.40 KRABBY_CLOCK_W=1.0 KRABBY_MIN_ACTUAL_SPEED=0.08"
GATE=0.35

run_stage () {
  local name=$1 mode=$2 start_ckpt=$3 cap_chunks=$4
  local ckpt="$start_ckpt" chunk=1 fail=1.0
  while [ $chunk -le $cap_chunks ]; do
    echo "=== $name chunk $chunk (resume=$ckpt)"
    env $BAND KRABBY_HEX_TEACHER_MODE=$mode timeout -k 60 14400 $PY scripts/rsl_rl/train.py \
      --task Isaac-Crab-Hex-Teacher-v0 --headless --num_envs 256 --seed 2 \
      --max_iterations 2000 --resume --checkpoint "$ckpt" > "$D/${name}_c${chunk}_train.log" 2>&1
    echo "${name}_c${chunk}_EXIT=$?"
    RUN=$(ls -t logs/rsl_rl/crab_hex_teacher/ | head -1)
    CK=$(ls logs/rsl_rl/crab_hex_teacher/$RUN/ | grep -E "^model_[0-9]+\.pt$" | sort -t_ -k2 -n | tail -1)
    ckpt="$PWD/logs/rsl_rl/crab_hex_teacher/$RUN/$CK"
    fail=$(grep -oE "Episode_Termination/crab_failure: [0-9.]+" "$D/${name}_c${chunk}_train.log" | tail -20 | awk -F': ' '{s+=$2} END {print s/NR}')
    ep=$(grep -oE "Mean episode length: [0-9.]+" "$D/${name}_c${chunk}_train.log" | tail -1)
    echo "${name}_c${chunk}: failure_tail20=$fail | $ep | ckpt=$RUN/$CK"
    ok=$(python3 -c "print(1 if $fail < $GATE else 0)")
    if [ "$ok" = "1" ]; then echo "${name}_GATE_PASS at chunk $chunk"; break; fi
    chunk=$((chunk+1))
  done
  if [ "$ok" != "1" ]; then echo "${name}_GATE_FAIL after $cap_chunks chunks (failure $fail)"; fi
  STAGE_CKPT="$ckpt"
  STAGE_OK="$ok"
}

run_stage bridge bridge "$PWD/logs/rsl_rl/crab_hex_flat_walk/2026-08-25_19-09-15/model_24900_critic_reset.pt" 4
[ "$STAGE_OK" != "1" ] && { echo "CHAIN_STOPPED_AT_BRIDGE"; exit 1; }
run_stage 2b1 2b1 "$STAGE_CKPT" 4
[ "$STAGE_OK" != "1" ] && { echo "CHAIN_STOPPED_AT_2B1"; exit 1; }
run_stage 2b2 2b2 "$STAGE_CKPT" 4
echo "GATED_CHAIN_DONE (2b2 gate: $STAGE_OK, final=$STAGE_CKPT)"

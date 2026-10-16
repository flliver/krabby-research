#!/bin/bash
# Train body reference chains one after another (PLAN steps 5, 6, 8) through the candidates driver, and read every
# stage end before the next stage starts.
#
#   bc_train_queue.sh [--smoke] CHAIN [CHAIN ...]      (each CHAIN is a references: name in body_cmd_train/)
#
# Per chain: wait for any other Isaac python -> launch_candidates.sh --reference-chain CHAIN --seed <its seed> ->
# wait for the driver's systemd unit to end -> bc_after_stage.py CHAIN (dump audit + stage-end read).
# Lines for the watcher:  BCTRAIN STAGE <chain> rc=<rc>   (0 OK, 2 NOTIFY = decide-record-notify, continue)
#                         BCTRAIN STOP <chain> rc=<rc>    (3 = a PLAN STOP point, 4 = audit / record failure)
#                         BCTRAIN END
# The queue never kills anything: a STOP ends the queue after the stage that triggered it.
set -u
R=/home/nickmagus/krabby/krabby-research
C=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-15_body_cmd
T=$C/body_cmd_train
L=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/launch_candidates.sh
PY=/home/nickmagus/krabby/isaac_venv/bin/python
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+)\.py'
smoke=""
[ "${1:-}" = "--smoke" ] && { smoke="--smoke"; shift; }
mkdir -p "$C/stage_reads"
echo "BCTRAIN START $(date -Is) chains=$*"
for chain in "$@"; do
  seed=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['references'][sys.argv[2]]['seed'])" "$T/candidates.yaml" "$chain") || { echo "BCTRAIN STOP $chain rc=4 (unknown chain)"; exit 4; }
  while ps -eo args | grep -qE "$ISAAC_PAT"; do sleep 30; done
  # BC_DRIVER_ARGS: extra run_candidates args (e.g. --accept-code-change WHY), word-split on purpose
  out=$("$L" --campaign-dir "$T" --reference-chain "$chain" --seed "$seed" ${BC_DRIVER_ARGS:-} 2>&1)
  unit=$(printf '%s\n' "$out" | grep -E '^cands-' | head -n 1)
  [ -n "$unit" ] || { echo "BCTRAIN STOP $chain rc=4 (launcher printed no unit: $(printf '%s' "$out" | tail -n 2 | tr '\n' ' '))"; exit 4; }
  echo "BCTRAIN LAUNCH $chain seed=$seed unit=$unit $(date -Is)"
  sleep 20
  while systemctl --user is-active -q "$unit"; do sleep 60; done
  "$PY" -I "$C/scripts/bc_after_stage.py" --chain "$chain" $smoke > "$C/stage_reads/$chain.log" 2>&1
  rc=$?
  if [ "$rc" = 0 ] || [ "$rc" = 2 ]; then
    echo "BCTRAIN STAGE $chain rc=$rc $(date -Is)"
  else
    echo "BCTRAIN STOP $chain rc=$rc $(date -Is)"
    exit "$rc"
  fi
done
echo "BCTRAIN END $(date -Is)"

#!/bin/bash
# DR stress sweep, step 4: the dose smoke, one Kit process per knob group (G0 none, G1 plant, G2 sensor, G3 push).
# The env is composed like tools/pipeline/gate_row.sh composes a row's: every KRABBY_* of the parent is removed, then
# the F19 p3b.env keys, then the group's KRABBY_STRESS_* keys. Memory-capped scope, SIGTERM-only timeout; waits while
# any Isaac python runs (the anchored `ps -ww -eo args` pattern, never `pgrep -f`).
#
#   smoke.sh OUT_DIR G0|G1|G2|G3 ...
# Prints "SMOKE END group=<g> rc=<rc> json=<path>" per group.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(cd "$HERE/../../../../../../.." && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
X=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments
BASE_ENV=$X/2026-10-16_task4_pipeline/F19/envs/p3b.env
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|stress_smoke|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+|bc_dump_cfgs|t4_dump_cfgs)\.py'
out=$1; shift
mkdir -p "$out"
group_keys() {
  case "$1" in
    G0) echo "" ;;
    G1) echo "KRABBY_STRESS_ROD_SPEED=0.7 KRABBY_STRESS_CAM=0.8 KRABBY_STRESS_ROD_EFFORT=0.85 KRABBY_STRESS_KPKD=1.25 KRABBY_STRESS_LEG_MASS=1.15 KRABBY_STRESS_MU=0.4 KRABBY_STRESS_VXY_ZERO=1" ;;
    G2) echo "KRABBY_STRESS_ACT_DELAY=3 KRABBY_STRESS_OBS_NOISE=2 KRABBY_STRESS_CUR_SCALE=0.85 KRABBY_STRESS_CUR_NOISE=0.1 KRABBY_STRESS_VXY_NOISE=0.1 KRABBY_STRESS_DEPTH_LAG=3" ;;
    G3) echo "KRABBY_STRESS_PUSH_DV=0.35" ;;
    *) echo "BAD" ;;
  esac
}
for g in "$@"; do
  keys=$(group_keys "$g")
  [ "$keys" = BAD ] && { echo "SMOKE END group=$g rc=2 json= (unknown group)"; continue; }
  envs=$(grep -E '^KRABBY_[A-Z0-9_]+=' "$BASE_ENV" | tr '\n' ' ')
  unset_args=$(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')
  rec=""
  [ "$g" = G0 ] && rec="--record $out/G0_record.npz"
  [ "$g" = G3 ] && rec="--steps 640"
  while ps -ww -eo args | grep -qE "$ISAAC_PAT"; do sleep 20; done
  ( cd "$R/parkour" && env $unset_args OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 \
      PYTHONPATH="$R/parkour/parkour_tasks:$R/parkour" $envs $keys \
      $CAP timeout --signal=TERM 1200 "$PY" -u "$HERE/stress_smoke.py" --headless --group "$g" \
      --out "$out/$g.json" $rec ) > "$out/$g.log" 2>&1
  echo "SMOKE END group=$g rc=$? json=$out/$g.json"
done

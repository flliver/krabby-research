#!/usr/bin/env bash
set -euo pipefail

KRABBY_ROOT="${KRABBY_ROOT:-/home/sanjay/Projects/krabby}"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <path-to-usda> <path-to-model.pt>" >&2
  exit 2
fi

USD_PATH="$1"
CHECKPOINT_PATH="$2"

if [[ ! -f "$USD_PATH" ]]; then
  echo "USD file not found: $USD_PATH" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint file not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

export KRABBY_HEX_USD_PATH="$USD_PATH"
export KRABBY_HEX_SPAWN_Z="${KRABBY_HEX_SPAWN_Z:-1.05}"
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"

cd "$KRABBY_ROOT/krabby-research/parkour"

echo "Playing crab hex flat-walk baseline"
echo "  USD:        $KRABBY_HEX_USD_PATH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Spawn Z:    $KRABBY_HEX_SPAWN_Z"
echo
echo "Tip: stop large headless training jobs before GUI play to avoid GPU memory pressure."

"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/play.py" \
  --task Isaac-Crab-Hex-Flat-Walk-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$CHECKPOINT_PATH"

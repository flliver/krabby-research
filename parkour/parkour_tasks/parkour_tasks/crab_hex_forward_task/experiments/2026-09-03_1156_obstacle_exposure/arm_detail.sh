#!/usr/bin/env bash
# Per-arm detail: termination-anatomy probe roll of the arm's checkpoint with its own stack,
# then the comparison table + gate ledger. Writes detail_<ARM>.md. Usage: arm_detail.sh <ARM>
set -u
ARM="$1"; D=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-03_1156_obstacle_exposure
PY=/home/nickmagus/krabby/isaac_venv/bin/python; PK=/home/nickmagus/krabby/krabby-research/parkour
SAFE=$(echo "$ARM" | tr '+' '_'); OUT="$D/probe_${SAFE}_termination"; LOG="$D/${SAFE}_termination_probe.log"
read -r CK EPS < <($PY - "$ARM" <<'PYEOF'
import json, sys
st = json.load(open("/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-03_1156_obstacle_exposure/state.json"))
name = sys.argv[1]
rec = st["arms"].get(name) or st.get("wave2", {}).get(name)
print(rec["ckpt"], rec["extra"].get("KRABBY_EPISODE_S", "20"))
PYEOF
)
$PY - "$ARM" > "$D/env_stack_${SAFE}.sh" <<'PYEOF'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("rx", "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-03_1156_obstacle_exposure/run_exposure.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
st = json.load(open(m.STATE)); name = sys.argv[1]
rec = st["arms"].get(name) or st.get("wave2", {}).get(name)
for k, v in m.campaign_stack(rec["extra"]).items():
    print(f"export {k}={v}")
print("export OMNI_KIT_ACCEPT_EULA=yes TERM=xterm")
PYEOF
if [ ! -f "$OUT/summary.json" ]; then
  ( cd "$PK" && source "$D/env_stack_${SAFE}.sh" && $PY "$D/training_timeline_probe.py" --headless --checkpoint "$CK" \
      --num_envs 64 --steps 4000 --modes stochastic --out "$OUT" --label "${SAFE}_termination" > "$LOG" 2>&1 ) &
  for _ in $(seq 150); do  # up to 50 min; the probe's teardown hangs, so stop it once the summary exists
    sleep 20
    if [ -f "$OUT/summary.json" ]; then sleep 30; pkill -f "training_timeline_prob[e]"; break; fi
    if grep -q "^Traceback" "$LOG" 2>/dev/null; then sleep 20; pkill -f "training_timeline_prob[e]"; break; fi
  done
fi
{
  echo "# Detail — arm $ARM ($(date '+%Y-%m-%d %H:%M'))"
  echo; $PY "$D/arm_table.py" "$ARM"; echo
  echo "## Miss anatomy (probe roll of the arm's head, its own training stack, stochastic policy; episode ${EPS} s)"
  if [ -f "$OUT/summary.json" ]; then $PY "$D/analyze_termination.py" "$OUT" "$EPS"; else echo "probe failed — see $LOG"; fi
} > "$D/detail_${SAFE}.md"
echo "detail ready: $D/detail_${SAFE}.md"

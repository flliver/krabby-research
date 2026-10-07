#!/usr/bin/env bash
# Anchor-discrimination probe (CHARTER 9, 2026-10-06): 3 heads x 3 harder field_hard variants, 50 episodes, env_seed 1.
# Serial, one Isaac process at a time, through run_arm.sh --row-eval (code-state guard + sidecar). Sentinel: PROBE_END.
C=$(cd "$(dirname "$0")/.." && pwd)
R=/home/nickmagus/krabby/krabby-research
OUT=$R/parkour/outputs/omni_plan/phase2
MAN=$C/probes/scenarios_probe_hard.yaml
say() { echo "[$(date -Is)] $*" >> "$OUT/queue.log"; }
say "PROBE start: anchor-discrimination probe (rows: probe_ramp_max, probe_wall_hard, probe_wall_max; 50 eps)"
for row in probe_wall_hard__omni probe_ramp_max__omni probe_wall_max__omni; do
  for s in 17 16 18; do
    ck=$R/$(/home/nickmagus/krabby/isaac_venv/bin/python -c "import yaml;print(yaml.safe_load(open('$C/anchors.yaml'))['anchors'][$s]['checkpoint'])")
    st=$R/$(/home/nickmagus/krabby/isaac_venv/bin/python -c "import yaml;print(yaml.safe_load(open('$C/anchors.yaml'))['anchors'][$s]['env_stamp'])")
    [ -f "$OUT/STOP_QUEUE" ] && { say "PROBE stopped: STOP_QUEUE"; echo PROBE_END >> "$OUT/probe_hard.done"; exit 3; }
    bash "$C/scripts/run_arm.sh" --row-eval "PROBE_s$s" "$ck" "$st" "$MAN" "$row" 1 --episodes 50 || say "PROBE row failed: s$s $row"
  done
done
say "PROBE end"
echo PROBE_END >> "$OUT/probe_hard.done"

#!/usr/bin/env bash
# Morphology x exposure launcher (systemd-run, same hardening as the PLAN G/H launchers).
# Pass-through args: --skip-smoke, --start-p2 [--top A15,A20,A15+B], --seed2 <cfg>.
for _ in $(seq 120); do
  pgrep -f "isaac_venv/bin/python [^ ]*(eval_crab_hex_gait|rsl_rl/train|training_timeline_probe)\.py" >/dev/null || break
  sleep 5
done
exec systemd-run --user --unit "morph-x-exposure-$(date +%s)" -p OOMPolicy=continue \
  --working-directory /home/nickmagus/krabby/krabby-research \
  timeout -k 300 345600 \
  /home/nickmagus/krabby/isaac_venv/bin/python \
  "$(cd "$(dirname "$0")" && pwd)/run_morph_exposure.py" "$@"

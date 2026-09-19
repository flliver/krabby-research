#!/usr/bin/env bash
for _ in $(seq 120); do
  pgrep -f "isaac_venv/bin/python [^ ]*(eval_crab_hex_gait|rsl_rl/train|training_timeline_probe)\.py" >/dev/null || break
  sleep 5
done
exec systemd-run --user --unit "a15b-lineage-$(date +%s)" -p OOMPolicy=continue \
  --working-directory /home/nickmagus/krabby/krabby-research \
  timeout -k 300 345600 \
  /home/nickmagus/krabby/isaac_venv/bin/python \
  "$(cd "$(dirname "$0")" && pwd)/run_lineage.py" "$@"

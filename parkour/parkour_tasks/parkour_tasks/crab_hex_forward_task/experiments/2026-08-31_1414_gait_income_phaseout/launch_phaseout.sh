#!/usr/bin/env bash
# PLAN G launcher: run_phaseout.py under systemd-run (same hardening as the PLAN F
# launcher it forks: teardown wait, service mode not --scope, OOMPolicy=continue).
# Pass-through args reach the orchestrator (e.g. --start-rounds after the user
# approves the post-scout schedule).
for _ in $(seq 120); do
  pgrep -f "isaac_venv/bin/python" >/dev/null || break
  sleep 5
done
exec systemd-run --user --unit "gait-phaseout-$(date +%s)" -p OOMPolicy=continue \
  --working-directory /home/nickmagus/krabby/krabby-research \
  timeout -k 300 172800 \
  /home/nickmagus/krabby/isaac_venv/bin/python \
  "$(cd "$(dirname "$0")" && pwd)/run_phaseout.py" "$@"

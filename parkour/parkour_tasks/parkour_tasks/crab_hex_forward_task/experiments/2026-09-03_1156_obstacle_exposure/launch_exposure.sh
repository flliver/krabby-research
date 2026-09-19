#!/usr/bin/env bash
# PLAN H launcher: run_exposure.py under systemd-run (same hardening as the PLAN G launcher:
# teardown wait, service mode not --scope, OOMPolicy=continue). Pass-through args reach the
# orchestrator (--start-smoke / --start-wave2 / --start-d after the user decisions).
for _ in $(seq 120); do
  pgrep -f "isaac_venv/bin/python" >/dev/null || break
  sleep 5
done
exec systemd-run --user --unit "obstacle-exposure-$(date +%s)" -p OOMPolicy=continue \
  --working-directory /home/nickmagus/krabby/krabby-research \
  timeout -k 300 259200 \
  /home/nickmagus/krabby/isaac_venv/bin/python \
  "$(cd "$(dirname "$0")" && pwd)/run_exposure.py" "$@"

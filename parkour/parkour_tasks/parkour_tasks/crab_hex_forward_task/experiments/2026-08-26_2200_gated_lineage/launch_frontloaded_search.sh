#!/usr/bin/env bash
# PLAN F launcher: the orchestrator under systemd-run with a hard-kill backstop.
exec systemd-run --user --scope -p MemoryMax=45G \
  timeout -k 300 172800 \
  /home/nickmagus/krabby/isaac_venv/bin/python \
  "$(dirname "$0")/run_frontloaded_search.py"

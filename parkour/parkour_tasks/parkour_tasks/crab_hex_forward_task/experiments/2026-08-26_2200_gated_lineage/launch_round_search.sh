#!/usr/bin/env bash
# PLAN F launcher: the orchestrator under systemd-run with a hard-kill backstop.
# Never start while a previous trainer still holds memory — a relaunch that overlaps
# an Isaac teardown gets OOM-killed (observed 2026-08-27 08:51, cost 6 h dead time).
for _ in $(seq 120); do
  pgrep -f "isaac_venv/bin/python" >/dev/null || break
  sleep 5
done
# Service mode (not --scope): fully detached from the invoking shell — a scope dies
# with the caller's process group (observed 2026-08-27 14:50).
# OOMPolicy=continue: an OOM-killed Isaac subprocess must not take the orchestrator
# unit down with it (systemd's default stop policy did exactly that, 2026-08-28 17:30).
exec systemd-run --user --unit "round-search-$(date +%s)" -p OOMPolicy=continue \
  --working-directory /home/nickmagus/krabby/krabby-research \
  timeout -k 300 172800 \
  /home/nickmagus/krabby/isaac_venv/bin/python \
  "$(cd "$(dirname "$0")" && pwd)/run_round_search.py"

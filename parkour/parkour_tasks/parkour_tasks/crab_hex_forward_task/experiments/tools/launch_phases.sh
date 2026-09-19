#!/usr/bin/env bash
# Launch run_phases.py in its own systemd user scope (isolated from the Claude app cgroup;
# OOMPolicy=continue so one OOM-killed subprocess does not take the driver down).
#   launch_phases.sh --campaign-dir <dir> --plant A15+B --phases 3a,3b --from-checkpoint <pt> ...
# Unit name: phases-<campaign dir basename>-<epoch>. Watch with heartbeat_phases.sh <unit> <dir>.
set -euo pipefail
CDIR=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--campaign-dir" ]]; then CDIR="${args[$((i + 1))]}"; fi
done
[[ -n "$CDIR" ]] || { echo "usage: $0 --campaign-dir <dir> [run_phases.py args]" >&2; exit 2; }
for _ in $(seq 120); do
  pgrep -f "isaac_venv/bin/python [^ ]*(eval_crab_hex_gait|rsl_rl/train|training_timeline_probe|crab_hex_phase_cfg_dump)\.py" >/dev/null || break
  sleep 5
done
UNIT="phases-$(basename "$CDIR")-$(date +%s)"
echo "$UNIT"
exec systemd-run --user --unit "$UNIT" -p OOMPolicy=continue \
  --working-directory /home/nickmagus/krabby/krabby-research \
  timeout -k 300 345600 \
  /home/nickmagus/krabby/isaac_venv/bin/python \
  "$(cd "$(dirname "$0")" && pwd)/run_phases.py" "$@"

#!/usr/bin/env bash
# Launch run_candidates.py (the candidate ladder) in its own systemd user scope.
#   launch_candidates.sh --campaign-dir <dir> [--max-wall-h H] [--accept-code-change WHY] ...
# Unit name: cands-<campaign dir basename>-<epoch>, echoed on stdout and passed to the driver as
# --unit so the ledger CLI can say whether an override lands now or at the next launch.
# Watch with: heartbeat_phases.sh <unit> <campaign dir> 1800
#
# Differences from launch_phases.sh, all deliberate:
#   * NO `timeout` wrapper. The ladder is a many-day campaign with no wall-clock cap (user decision);
#     a cap belongs on ONE relaunch (`--max-wall-h`), which parks cleanly at await_user:relaunch.
#   * MemoryMax=48G / MemorySwapMax=0 / OOMPolicy=continue: Isaac under a cap on this 59 GB box, with
#     one OOM-killed eval child leaving the DRIVER alive to record the infra signature (the plan's
#     infra detector reads this cgroup's memory.events oom_kill counter).
#   * KillMode=mixed + TimeoutStopSec=360: `systemctl --user stop <unit>` SIGTERMs the driver only
#     (not the whole cgroup), so the driver's own handler stops the training/eval child with SIGTERM
#     first -- a SIGKILL of a live Kit corrupts its DerivedDataCache and every later start balloons.
#     360 s covers the worst case: one poll tick (120 s) + the child's SIGTERM grace (120 s) + the post-SIGKILL
#     wait (60 s) + the save -- about 300 s.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)  # this checkout's root
CDIR=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--campaign-dir" ]]; then CDIR="${args[$((i + 1))]}"; fi
done
[[ -n "$CDIR" ]] || { echo "usage: $0 --campaign-dir <dir> [run_candidates.py args]" >&2; exit 2; }

# One Isaac process at a time: wait up to 10 min for any other training/eval/probe to exit, and
# refuse to start beside another campaign driver (two drivers would share one GPU and one ledger).
if pgrep -f "isaac_venv/bin/python [^ ]*(run_phases|run_candidates)\.py" >/dev/null; then
  echo "another campaign driver is already running (run_phases.py / run_candidates.py) -- refusing" >&2
  exit 3
fi
for _ in $(seq 120); do
  pgrep -f "isaac_venv/bin/python [^ ]*(eval_crab_hex_gait|rsl_rl/train|training_timeline_probe|crab_hex_phase_cfg_dump)\.py" >/dev/null || break
  sleep 5
done

UNIT="cands-$(basename "$CDIR")-$(date +%s)"
echo "$UNIT"
exec systemd-run --user --unit "$UNIT" \
  -p OOMPolicy=continue \
  -p MemoryMax=48G \
  -p MemorySwapMax=0 \
  -p KillMode=mixed \
  -p TimeoutStopSec=360 \
  --working-directory "$REPO_ROOT" \
  /home/nickmagus/krabby/isaac_venv/bin/python \
  "$(cd "$(dirname "$0")" && pwd)/run_candidates.py" --unit "$UNIT" "$@"

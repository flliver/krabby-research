#!/usr/bin/env bash
# Heartbeat for the PLAN H orchestrator (heartbeat skill format):
#   <symbol> <HH:MM> | ETA <estimate> | <short status>
# usage: heartbeat_exposure.sh <systemd-user-unit> [interval_s]
UNIT="$1"; INTERVAL="${2:-1800}"
D=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-03_1156_obstacle_exposure
STATE="$D/state.json"
prev=""; prev_t=$(date +%s); prev_stage=""
phase_of() { python3 -c "import json;print(json.load(open('$STATE')).get('phase','?'))" 2>/dev/null || echo "?"; }
human_phase() {
  case "$1" in
    a) echo "Phase A: timeline probes, trench-only eval, widened baselines";;
    await_user_a) echo "Phase A finished, paused for the user (start smoke + wave 1 with --start-smoke)";;
    smoke) echo "Phase B smoke runs (unarmed then armed, 200 iterations each)";;
    wave1) echo "Wave 1 arms (C0 control, C3, C1, C4, C2, C5; 5000 iterations each)";;
    await_user_c) echo "Wave 1 finished, paused for the user (--start-wave2)";;
    wave2) echo "Wave 2 pairings and the seed-2 confirmation";;
    await_user_d) echo "Wave 2 finished, paused for the user (--start-d)";;
    d) echo "Phase D replay 5k to 30k with the winner armed";;
    done) echo "campaign complete";;
    *) echo "phase $1";;
  esac
}
while true; do
  now=$(date +%H:%M); t=$(date +%s)
  phase=$(phase_of)
  alive=0; systemctl --user is-active --quiet "$UNIT" && alive=1
  # newest job log (train / probe / eval) tells the stage inside the phase
  latest=$(ls -t "$D"/*_train.log "$D"/*_probe.log "$D"/*_obst.log "$D"/*_canary.log 2>/dev/null | head -1)
  stage=$(basename "${latest:-none}" | sed 's/\.log$//')
  cur=0; total=0
  case "$stage" in
    *_train) cur=$(grep -c "Mean reward" "$latest" 2>/dev/null || echo 0); total=5000; [[ "$stage" == *smoke* ]] && total=200;;
    *_probe) cur=$(grep -c "\[probe\]" "$latest" 2>/dev/null || echo 0); total=3;;
    *_obst|*_canary) cur=$(grep -c "episode" "$latest" 2>/dev/null || echo 0); total=100;;
  esac
  if [ "$phase" = "done" ]; then echo "🏁 $now | ETA n/a | Campaign complete — see REPORT.md"; exit 0; fi
  if [ "$alive" = "0" ]; then
    case "$phase" in
      await_user_*) echo "⏸️ $now | ETA n/a | $(human_phase "$phase")"; exit 0;;
      *) echo "❌ $now | ETA n/a | Orchestrator unit $UNIT is not running while state says '$phase' — check journalctl --user -u $UNIT and $stage.log"; exit 0;;
    esac
  fi
  eta="unknown"; sym="✅"
  if [ -n "$prev" ] && [ "$stage" = "$prev_stage" ] && [ "$cur" -gt "$prev" ] && [ "$total" -gt 0 ]; then
    rate=$(( (cur - prev) * 3600 / (t - prev_t + 1) ))  # units per hour
    [ "$rate" -gt 0 ] && eta="~$(( (total - cur) * 60 / rate ))m"
  fi
  if [ -n "$prev" ] && [ "$stage" = "$prev_stage" ] && [ "$cur" -le "$prev" ] && [[ "$stage" == *_train ]]; then
    sym="⚠️"; detail="no new training iterations since the last beat"
  else
    detail="$(human_phase "$phase"); current job $stage at $cur of $total"
  fi
  prev=$cur; prev_t=$t; prev_stage=$stage
  echo "$sym $now | ETA $eta | $detail"
  sleep "$INTERVAL"
done

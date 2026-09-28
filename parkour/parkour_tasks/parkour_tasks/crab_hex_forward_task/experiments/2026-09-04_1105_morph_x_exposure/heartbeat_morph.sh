#!/usr/bin/env bash
# Heartbeat for the morphology x exposure orchestrator: <symbol> <HH:MM> | ETA <est> | <status>
UNIT="$1"; INTERVAL="${2:-1800}"
D=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-04_1105_morph_x_exposure
STATE="$D/state.json"; prev=""; prev_t=$(date +%s); prev_stage=""
phase_of() { python3 -c "import json;print(json.load(open('$STATE')).get('phase','?'))" 2>/dev/null || echo "?"; }
human() { case "$1" in
  smoke) echo "armed smoke on A15 (200 iterations + one eval)";;
  p1) echo "P1: from-scratch formation with walking slots, 8 configurations";;
  await_user_a) echo "P1 finished, paused at STOP A for the user (--start-p2 [--top ...])";;
  p2) echo "P2: golden + top 3, 40-s episodes, two 5k stages each";;
  await_user_b) echo "P2 finished, paused at STOP B for the user (--seed2 <cfg>)";;
  seed2) echo "seed-2 confirmation of the user's pick";;
  halt_smoke) echo "smoke FAILED — halted";;
  done) echo "campaign complete";;
  *) echo "phase $1";; esac; }
while true; do
  now=$(date +%H:%M); t=$(date +%s); phase=$(phase_of)
  alive=0; systemctl --user is-active --quiet "$UNIT" && alive=1
  latest=$(ls -t "$D"/logs/*_train.log "$D"/logs/*_eval.log "$D"/logs/*_obst.log 2>/dev/null | head -1)
  stage=$(basename "${latest:-none}" | sed 's/\.log$//'); cur=0; total=0
  case "$stage" in
    *_train) cur=$(grep -c "Mean reward" "$latest" 2>/dev/null || echo 0); total=5000; [[ "$stage" == smoke* ]] && total=200;;
    *_eval|*_obst) cur=$(grep -c "episode" "$latest" 2>/dev/null || echo 0); total=100;;
  esac
  if [ "$phase" = "done" ]; then echo "🏁 $now | ETA n/a | Campaign complete — run assemble_final.py"; exit 0; fi
  if [ "$alive" = "0" ]; then case "$phase" in
      await_user_*) echo "⏸️ $now | ETA n/a | $(human "$phase")"; exit 0;;
      halt_smoke) echo "❌ $now | ETA n/a | $(human "$phase") — see REPORT.md"; exit 0;;
      *) echo "❌ $now | ETA n/a | Unit $UNIT not running while state says '$phase' — journalctl --user -u $UNIT; last job $stage"; exit 0;; esac; fi
  eta="unknown"; sym="✅"
  if [ -n "$prev" ] && [ "$stage" = "$prev_stage" ] && [ "$cur" -gt "$prev" ] && [ "$total" -gt 0 ]; then
    rate=$(( (cur - prev) * 3600 / (t - prev_t + 1) )); [ "$rate" -gt 0 ] && eta="~$(( (total - cur) * 60 / rate ))m"; fi
  if [ -n "$prev" ] && [ "$stage" = "$prev_stage" ] && [ "$cur" -le "$prev" ] && [[ "$stage" == *_train ]]; then
    sym="⚠️"; detail="no new training iterations since the last beat ($stage at $cur)"; else
    detail="$(human "$phase"); current job $stage at $cur of $total"; fi
  prev=$cur; prev_t=$t; prev_stage=$stage
  echo "$sym $now | ETA $eta | $detail"; sleep "$INTERVAL"; done

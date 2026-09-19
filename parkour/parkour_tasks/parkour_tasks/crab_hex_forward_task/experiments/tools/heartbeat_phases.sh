#!/usr/bin/env bash
# Heartbeat for run_phases.py: <symbol> <HH:MM> | ETA <est> | <status>
#   heartbeat_phases.sh <systemd unit> <campaign dir> [interval s, default 1800]
UNIT="$1"; D="$2"; INTERVAL="${3:-1800}"
STATE="$D/state.json"; prev=""; prev_t=$(date +%s); prev_job=""
field() { python3 -c "import json,sys;d=json.load(open('$STATE'));print(d.get('current',{}).get('$1','') if '$1'!='phase' else d.get('phase','?'))" 2>/dev/null || echo "?"; }
while true; do
  now=$(date +%H:%M); t=$(date +%s); phase=$(field phase); tag=$(field tag); total=$(field iters); kind=$(field kind)
  alive=0; systemctl --user is-active --quiet "$UNIT" && alive=1
  latest=$(ls -t "$D"/logs/*.log 2>/dev/null | head -1); job=$(basename "${latest:-none}" .log); cur=0
  case "$job" in
    *_train) cur=$(grep -c "Learning iteration" "$latest" 2>/dev/null || echo 0);;
    *_eval) cur=$(grep -c "episode" "$latest" 2>/dev/null || echo 0); total=100;;
  esac
  if [ "$phase" = "done" ]; then echo "🏁 $now | ETA n/a | Campaign complete — see $D/REPORT.md"; exit 0; fi
  if [ "$alive" = "0" ]; then case "$phase" in
      await_user*) echo "⏸️ $now | ETA n/a | ${phase#await_user:} finished, paused for the user (see $D/notify.log)"; exit 0;;
      halt*) echo "❌ $now | ETA n/a | HALTED: ${phase#halt:} — see $D/REPORT.md"; exit 0;;
      *) echo "❌ $now | ETA n/a | unit $UNIT not running while state says '$phase' — journalctl --user -u $UNIT; last job $job"; exit 0;; esac; fi
  eta="unknown"; sym="✅"
  if [ -n "$prev" ] && [ "$job" = "$prev_job" ] && [ "$cur" -gt "$prev" ] && [ "${total:-0}" -gt 0 ] 2>/dev/null; then
    rate=$(( (cur - prev) * 3600 / (t - prev_t + 1) )); [ "$rate" -gt 0 ] && eta="~$(( (total - cur) * 60 / rate ))m"; fi
  if [ -n "$prev" ] && [ "$job" = "$prev_job" ] && [ "$cur" -le "$prev" ] && [[ "$job" == *_train ]]; then
    sym="⚠️"; detail="no new iterations since the last beat ($job at $cur)"; else
    detail="phase ${tag:-?} ($kind); job $job at $cur of ${total:-?}"; fi
  prev=$cur; prev_t=$t; prev_job=$job
  echo "$sym $now | ETA $eta | $detail"; sleep "$INTERVAL"; done

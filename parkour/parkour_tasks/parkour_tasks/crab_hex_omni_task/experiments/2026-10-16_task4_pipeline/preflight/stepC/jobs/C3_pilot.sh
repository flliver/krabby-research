# C3_pilot [C3 driver] 3000-it payload pilot from S1 at the D5 training values (save cadence 100: +500k reads)
set -u
out=$(bash /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/launch_candidates.sh --campaign-dir /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/preflight/stepC --reference-chain PP_p3a_s18 --seed 18 ${T4C_DRIVER_ARGS:-} 2>&1)
unit=$(printf '%s\n' "$out" | grep -E '^cands-' | head -n 1)
[ -n "$unit" ] || { echo "launcher printed no unit: $(printf '%s' "$out" | tail -n 3 | tr '\n' ' ')"; exit 4; }
echo "unit=$unit"
sleep 20
deadline=$(( $(date +%s) + 21600 ))
while systemctl --user is-active -q "$unit"; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "wall bound 6.0 h: systemctl --user stop $unit (the driver's SIGTERM path; never SIGKILL)"
    systemctl --user stop "$unit"; exit 124
  fi
  sleep 30
done
exit 0

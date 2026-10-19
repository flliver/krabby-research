# C1_PC_s1pay00 [C1 driver] 20-it s1 smoke, payload ARMED 0:0 / ZERO_FRAC 1 (shared tags vs C1_PC_s1)
set -u
out=$(bash /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/launch_candidates.sh --campaign-dir /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/preflight/stepC --reference-chain PC_s1pay00_s18 --seed 18 ${T4C_DRIVER_ARGS:-} 2>&1)
unit=$(printf '%s\n' "$out" | grep -E '^cands-' | head -n 1)
[ -n "$unit" ] || { echo "launcher printed no unit: $(printf '%s' "$out" | tail -n 3 | tr '\n' ' ')"; exit 4; }
echo "unit=$unit"
sleep 20
deadline=$(( $(date +%s) + 3600 ))
while systemctl --user is-active -q "$unit"; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "wall bound 1.0 h: systemctl --user stop $unit (the driver's SIGTERM path; never SIGKILL)"
    systemctl --user stop "$unit"; exit 124
  fi
  sleep 30
done
exit 0

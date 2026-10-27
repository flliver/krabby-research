#!/bin/bash
# Start step 5 only after the step-4b eval queue has ended (one Isaac process at a time; no gap race).
until grep -q "STEP4B END" /home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/body_cmd/step4b_queue.log; do sleep 30; done
exec /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-15_body_cmd/scripts/step5_smoke.sh

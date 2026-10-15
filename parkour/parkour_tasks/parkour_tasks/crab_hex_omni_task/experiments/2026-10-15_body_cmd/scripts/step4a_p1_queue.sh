#!/bin/bash
# Step 4a (PLAN P1): S1 s18 + body_hold on the envelope grid, es1 + es2 -> the envelope freeze (CHARTER rule).
S=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-15_body_cmd/scripts
CK=/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt
MB=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni_body.yaml
echo "STEP4A START $(date -Is)"
for es in 1 2; do $S/bc_eval_row.sh P1_S1_s18_bh $CK /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-15_body_cmd/envs/S1_suite.env $MB grid__omni_body $es --command-adapter body_hold; done
echo "STEP4A END $(date -Is)"

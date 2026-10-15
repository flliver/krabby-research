#!/bin/bash
# Step 3 (PLAN): G0 inertness -- S1 at the new code / eval sha must reproduce REFS_S1_s18 bitwise -- plus the harness
# proof (S1 + body_hold on suite__omni_body). One row at a time (bc_eval_row.sh waits for any other Isaac python).
S=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-15_body_cmd/scripts
CK=/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt
MO=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml
MP=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2/scenarios_p2.yaml
MB=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni_body.yaml
echo "STEP3 START $(date -Is)"
$S/bc_eval_row.sh G0_S1_s18 $CK /home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24 $MO suite__omni 1
$S/bc_eval_row.sh G0_S1_s18 $CK /home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/field_hard__omni/seed001/2026-10-09_03-47-57 $MP field_hard__omni 1
$S/bc_eval_row.sh G0w_S1_s18 $CK /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-15_body_cmd/envs/S1_suite_frame_world.env $MO suite__omni 1
$S/bc_eval_row.sh PROOF_S1_s18_bh $CK /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-15_body_cmd/envs/S1_suite.env $MB suite__omni_body 1 --command-adapter body_hold
echo "STEP3 END $(date -Is)"

#!/bin/bash
# STOP-C C4_depth diagnostic (CHARTER 2026-10-17): the depth and zeroed-depth 1000-it pilots on the other field rows
# (field_light, ring_obst, field_turn) + the zero pilot's suite, Student-Play student role, through TP/gate_row.sh
# exactly as the C4 rows. Report-only; one Isaac process at a time (gate_row.sh waits).
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)  # this checkout's root, from the script's own location
EXP=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments
PL=$EXP/2026-10-16_task4_pipeline
ROW=$EXP/tools/pipeline/gate_row.sh
M2=$EXP/2026-10-07_omni_phase2/scenarios_p2.yaml
MO=$EXP/eval/scenarios_omni.yaml
S=$R/parkour/logs/rsl_rl/crab_hex_omni_student
declare -A CK=([pdepth]=$S/2026-10-17_08-57-17_T4C4_pilot_depth/model_20993.pt [pzero]=$S/2026-10-17_10-54-37_T4C4_pilot_zero/model_20993.pt)
export TP_OUT=$R/parkour/outputs/omni_plan/task4/stepC TP_ROW_TIMEOUT=5400
export TP_STRIP_KEYS='KRABBY_ACTION_STD_MAX KRABBY_ACTION_STD_RESET KRABBY_ENTROPY_COEF KRABBY_ACTION_BOUND_LOSS KRABBY_ACTION_BOUND_RAMP KRABBY_ACTION_BOUND KRABBY_PPO_SCHEDULE KRABBY_PPO_LR KRABBY_PPO_DESIRED_KL KRABBY_PPO_KL_STOP KRABBY_PPO_HIST_LR KRABBY_PPO_HAND_DIAG KRABBY_ANCHOR_W KRABBY_ANCHOR_CKPT KRABBY_STAND_PRIOR_W KRABBY_STAND_PRIOR_TARGET KRABBY_RESTANCE KRABBY_BC_W KRABBY_BC_BANK KRABBY_BC_BATCH KRABBY_OMNI_MIRROR_IMAGES KRABBY_SYM_LOSS_COEF'
LOG=$PL/preflight/stepC/depthdiag.log
echo "DD START $(date -Is)" >> "$LOG"
for h in pdepth pzero; do
  for fam_es in field_light:1 field_light:2 ring_obst:1 ring_obst:2 field_turn:1; do
    fam=${fam_es%%:*}; es=${fam_es##*:}
    bash "$ROW" "T4C4D_$h" "${CK[$h]}" "$PL/R18/envs/p2a.$fam.env" "$M2" "${fam}__omni" "$es" \
      --task Isaac-Crab-Hex-Omni-Student-Play-v0 --policy-role student > /dev/null 2>&1
    echo "DD ROW $h $fam@$es rc=$? $(date -Is)" >> "$LOG"
  done
done
bash "$ROW" T4C4D_pzero "${CK[pzero]}" "$PL/R18/envs/p2a.env" "$MO" suite__omni 1 --task Isaac-Crab-Hex-Omni-Student-Play-v0 \
  --policy-role student --trained-categories stand,fwd,side,back,diag,turn,move_turn > /dev/null 2>&1
echo "DD ROW pzero suite@1 rc=$? $(date -Is)" >> "$LOG"
echo "DD END $(date -Is)" >> "$LOG"

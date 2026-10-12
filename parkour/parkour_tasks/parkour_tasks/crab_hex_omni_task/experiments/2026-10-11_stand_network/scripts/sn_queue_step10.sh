#!/usr/bin/env bash
# Stand-network step 10 (user go 2026-10-12): A-mask replicates at stand-net training seeds 16 and 17 -- same config
# as the pilot (sn_dagger --mode dagger --arm mask --iterations 4000; judged ckpt model_4000), each followed by the
# step-9 eval rows. S1 s18 stays the walking head. Serial, one Isaac process at a time. Sentinels "SNQ10 ..." in
# $SNOUT/sn_queue10.log; a STOP_QUEUE file in $SNOUT stops before the next stage.
set -u
R=/home/nickmagus/krabby/krabby-research
S=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-11_stand_network/scripts
SNOUT=$R/parkour/outputs/omni_plan/stand_network
LOG=$SNOUT/sn_queue10.log
ROWS="suite__omni@1 suite__omni@2 walkstop__omni@1 walkstop__omni@2 field_hard__omni@1 field_hard__omni@2 field_hard__omni@3 field_hard__omni@4 field_turn__omni@1"
say() { echo "SNQ10 $* $(date -Is)" >> "$LOG"; }
stopq() { [ -f "$SNOUT/STOP_QUEUE" ] && { say STOP STOP_QUEUE present; exit 4; }; }
say START step10
for seed in 16 17; do
  name=SN_Amask_s$seed
  stopq
  bash "$S/sn_dagger.sh" "$name" -- --mode dagger --arm mask --seed "$seed" --iterations 4000; rc=$?
  say train $name rc=$rc
  [ "$rc" = 0 ] && [ -f "$SNOUT/runs/$name/model_4000.pt" ] || { say STOP train $name rc=$rc; exit 6; }
  stopq
  # shellcheck disable=SC2086
  bash "$S/sn_eval_rows.sh" "$name" "$SNOUT/runs/$name/model_4000.pt" 1 $ROWS; rc=$?
  say evals $name rc=$rc
  [ "$rc" = 0 ] || { say STOP evals $name rc=$rc; exit 5; }
done
say DONE step10

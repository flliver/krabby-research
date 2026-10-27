#!/usr/bin/env bash
# Stand-network GPU queue, user go 2026-10-12: step-9 evals of the pilot SN_Amask_s18 (model_4000, mask 1) first,
# then step 8 (train SN_Afull_s18, 4000 its, mask 0 -- same config as the pilot), then its step-9 evals. Serial, one
# Isaac process at a time. The step-8 premise check (mid-run, notify-only + STOP-and-ask) is read by the session from
# the trainer's stands.jsonl; a STOP_QUEUE file in $SNOUT stops before the next stage. Sentinels "SNQ9 ..." in
# $SNOUT/sn_queue89.log.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)  # this checkout's root, from the script's own location
S=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-11_stand_network/scripts
SNOUT=$R/parkour/outputs/omni_plan/stand_network
LOG=$SNOUT/sn_queue89.log
ROWS="suite__omni@1 suite__omni@2 walkstop__omni@1 walkstop__omni@2 field_hard__omni@1 field_hard__omni@2 field_hard__omni@3 field_hard__omni@4 field_turn__omni@1"
say() { echo "SNQ9 $* $(date -Is)" >> "$LOG"; }
stopq() { [ -f "$SNOUT/STOP_QUEUE" ] && { say STOP STOP_QUEUE present; exit 4; }; }
evals() {  # NAME CKPT MASK
  stopq
  # shellcheck disable=SC2086
  bash "$S/sn_eval_rows.sh" "$1" "$2" "$3" $ROWS; local rc=$?
  say evals "$1" rc=$rc
  [ "$rc" = 0 ] || { say STOP evals "$1" rc=$rc; exit 5; }
}
say START step89
evals SN_Amask_s18 "$SNOUT/runs/SN_Amask_s18/model_4000.pt" 1
stopq
bash "$S/sn_dagger.sh" SN_Afull_s18 -- --mode dagger --arm full --seed 18 --iterations 4000; rc=$?
say train SN_Afull_s18 rc=$rc
[ "$rc" = 0 ] || { say STOP train SN_Afull_s18 rc=$rc; exit 6; }
[ -f "$SNOUT/runs/SN_Afull_s18/model_4000.pt" ] || { say STOP no model_4000 for SN_Afull_s18; exit 6; }
evals SN_Afull_s18 "$SNOUT/runs/SN_Afull_s18/model_4000.pt" 0
say DONE step89

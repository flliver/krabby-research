#!/usr/bin/env bash
# Stand-network GPU queue up to (not including) step 7 -- PLAN.md steps 3 (identity bridge + GPU swap-live), 6 (smokes
# + smoke-checkpoint walkstop rows + offline parity). Serial, one Isaac process at a time; each stage's own script
# refuses to start while another Isaac driver runs. A1 (6b) is launched separately once the smoke's s_per_iter has
# set its iteration count. Sentinel lines "SNQ ..." in $SNOUT/sn_queue.log (the session watches DONE / STOP).
set -u
R=/home/nickmagus/krabby/krabby-research
S=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-11_stand_network/scripts
C=$(cd "$S/.." && pwd)
SNOUT=$R/parkour/outputs/omni_plan/stand_network
PY=/home/nickmagus/krabby/isaac_venv/bin/python
S1=$R/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt
LOG=$SNOUT/sn_queue.log
mkdir -p "$SNOUT"
say() { echo "SNQ $* $(date -Is)" >> "$LOG"; }
stopq() { [ -f "$SNOUT/STOP_QUEUE" ] && { say STOP STOP_QUEUE present; exit 4; }; }
cpu() { ( cd "$R" && CUDA_VISIBLE_DEVICES="" PYTHONPATH=parkour/parkour_tasks:parkour "$PY" "$@" ); }
say START pre7
# ---- step 3: identity bridge (S1 as stand ckpt, mask 0) on suite / walkstop / field_hard es1
stopq; bash "$S/sn_eval_rows.sh"; rc=$?; say bridge-rows rc=$rc
[ "$rc" = 0 ] || { say STOP bridge rows rc=$rc; exit 5; }
cpu "$S/sn_bridge_read.py" --head SN_ID_s18 --identity --json "$C/bridge_identity.json" >> "$LOG" 2>&1; rc=$?
say bridge-read rc=$rc "(0 BITWISE, 3 MISS_SMALL, 4 STOP)"
[ "$rc" = 4 ] && { say STOP identity bridge rc=4; exit 5; }
# ---- GPU swap-live: S1 as stand ckpt with mask 1, walkstop es1 (pre-entry rows must equal REFS)
stopq; bash "$S/sn_eval_rows.sh" SN_IDM_s18 "$S1" 1 walkstop__omni@1; rc=$?; say swaplive-row rc=$rc
[ "$rc" = 0 ] || { say STOP swaplive row rc=$rc; exit 5; }
cpu "$S/sn_bridge_read.py" --head SN_IDM_s18 --rows walkstop__omni@1 --live --json "$C/bridge_swaplive.json" >> "$LOG" 2>&1; rc=$?
say swaplive-read rc=$rc; [ "$rc" = 0 ] || { say STOP swaplive read rc=$rc; exit 5; }
# ---- step 6: smokes (each exits 6 on a smoke FAIL)
for spec in "SN_SMOKE_mask_s18|--mode dagger --arm mask --seed 18 --iterations 3000 --smoke --smoke-iters 100" \
            "SN_SMOKE_full_s18|--mode dagger --arm full --seed 18 --iterations 3000 --smoke --smoke-iters 100" \
            "SN_SMOKE_A1_s18|--mode recover_probe --seed 18 --iterations 100 --smoke"; do
  stopq; name=${spec%%|*}; args=${spec#*|}
  # shellcheck disable=SC2086
  bash "$S/sn_dagger.sh" "$name" -- $args; rc=$?; say smoke $name rc=$rc
  [ "$rc" = 0 ] || { say STOP smoke $name rc=$rc; exit 6; }
done
# ---- smoke checkpoints load in stand_net and run one walkstop episode set; offline trainer->launcher parity
for arm in mask full; do
  m=1; [ "$arm" = full ] && m=0
  ck=$(ls -1 "$SNOUT/runs/SN_SMOKE_${arm}_s18"/model_*.pt 2>/dev/null | sort -V | tail -1)
  [ -n "$ck" ] || { say STOP no smoke checkpoint for $arm; exit 6; }
  stopq; bash "$S/sn_eval_rows.sh" "SN_SMK_${arm}_s18" "$ck" "$m" walkstop__omni@1; rc=$?; say smk-row $arm rc=$rc
  [ "$rc" = 0 ] || { say STOP smoke-ckpt row $arm rc=$rc; exit 6; }
  cpu "$S/sn_bridge_read.py" --head "SN_SMK_${arm}_s18" --rows walkstop__omni@1 --live \
      --json "$C/smoke_ckpt_${arm}.json" >> "$LOG" 2>&1; rc=$?; say smk-read $arm rc=$rc
  cpu "$S/sn_dagger_parity.py" "$SNOUT/runs/SN_SMOKE_${arm}_s18/smoke_capture.pt" >> "$LOG" 2>&1; rc=$?
  say parity $arm rc=$rc
done
say DONE pre7

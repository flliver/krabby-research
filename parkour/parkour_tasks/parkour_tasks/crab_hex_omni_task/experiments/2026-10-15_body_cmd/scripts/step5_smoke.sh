#!/bin/bash
# Step 5 (PLAN): train the three smoke chains (100 its each, from scratch), evaluate each head on the body rows
# (route term) and the p2a smoke head on the world suite through world_rate, then the smoke assertions.
set -u
R=/home/nickmagus/krabby/krabby-research
C=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-15_body_cmd
S=$C/scripts
PY=/home/nickmagus/krabby/isaac_venv/bin/python
MB=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni_body.yaml
MO=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml
LR=$R/parkour/outputs/omni_plan/body_cmd/lr
echo "STEP5 START $(date -Is)"
$S/bc_train_queue.sh --smoke BC_SMK_1b_s18 BC_SMK_c4_s18 BC_SMK_p2a_s18 || { echo "STEP5 STOP (train)"; exit 1; }
terms=(); wr=()
for ch in BC_SMK_1b_s18 BC_SMK_c4_s18 BC_SMK_p2a_s18; do
  ck=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ckpt'])" "$C/stage_reads/$ch.json")
  $PY -I $S/bc_head_env.py --chain $ch --out $C/envs/$ch.env
  for sc in suite__omni_body turnlevels__omni_body; do
    $S/bc_eval_row.sh $ch "$ck" $C/envs/$ch.env $MB $sc 1 || { echo "STEP5 STOP (eval $ch $sc)"; exit 1; }
    terms+=(--term "$(ls -d $LR/$ch/$sc/seed001/*/ | tail -1)")
  done
done
ck=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ckpt'])" "$C/stage_reads/BC_SMK_p2a_s18.json")
$S/bc_eval_row.sh WR_BC_SMK_p2a_s18 "$ck" $C/envs/BC_SMK_p2a_s18.env $MO suite__omni 1 --command-adapter world_rate --adapter-turns on || { echo "STEP5 STOP (world_rate eval)"; exit 1; }
wr+=(--world-rate "$(ls -d $LR/WR_BC_SMK_p2a_s18/suite__omni/seed001/*/ | tail -1)")
$PY -I $S/bc_smoke_check.py "${terms[@]}" "${wr[@]}" --chain BC_SMK_1b_s18 --chain BC_SMK_c4_s18 --chain BC_SMK_p2a_s18 > $C/step5_smoke.log 2>&1
echo "STEP5 END rc=$? $(date -Is)"

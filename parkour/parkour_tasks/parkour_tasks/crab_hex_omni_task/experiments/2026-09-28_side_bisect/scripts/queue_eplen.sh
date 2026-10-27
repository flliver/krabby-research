#!/usr/bin/env bash
# Episode-length ladder on C_lr_10_rsi (CHARTER 2026-09-30 19:09; user request). Steps 20..60 s run in order
# regardless of results; any step FAST on either hand gets its _s2 arm after all five. Both-hands gate per arm.
# Log lines: "EARLY <ref>", "EVENT <ref>_lr CLASS_L x CLASS_R y", "EVENT <ref> VERDICT v", "SKIP <ref>", "DONE eplen".
set -u
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
C=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect
L=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/launch_candidates.sh
EVAL=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
LRMAN=$C/scenarios_side_lr.yaml
EV=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect
RUNS=$R/parkour/logs/rsl_rl/crab_hex_omni_flat_walk
OUT=$R/parkour/outputs/omni_plan/side_bisect/arms
LROUT=$R/parkour/outputs/omni_plan/side_bisect/lr
LOG=$R/parkour/outputs/omni_plan/side_bisect/queue.log
PATCH=$C/scripts/candidates_v16.yaml
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
mkdir -p $OUT $LROUT
say() { echo "[$(date -Is)] $*" >> $LOG; }
wait_unit() { until ! systemctl --user is-active -q $1; do sleep 30; done; }
drivers_idle() { while ps -eo args | grep -qE "^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py"; do sleep 30; done; }
cls_of() { python3 -c "import json;print(json.load(open('$1'))['$2'].get('$3'))"; }
decide() {  # L R -> OK / FAIL / AMBIG
  if [ "$1" = FAST ] && [ "$2" = FAST ]; then echo OK
  elif { [ "$1" = SLOW ] || [ "$1" = DEAD ]; } && { [ "$2" = SLOW ] || [ "$2" = DEAD ]; }; then echo FAIL
  else echo AMBIG; fi
}
any_stamp_of() {  # RUN_DIR -> newest complete driver eval stamp (any row) of that run
  local rd=$1 tag; tag=$(basename $rd | cut -d_ -f3-)
  ls -d $EV/$tag/*/seed001/*/ 2>/dev/null | while read d; do
    grep -q "$(basename $rd)" $d/run_meta.json 2>/dev/null && [ -f $d/complete.json ] && echo $d; done | tail -1
}
newest_run_after() {
  local t=$1
  for d in $(ls -dt $RUNS/*/ 2>/dev/null | head -8); do
    [ $(stat -c %Y $d) -ge $t ] && { echo ${d%/}; return; }
  done
}
FIRST_ACC=1
ACC=(--accept-candidates-change "v16: episode-length ladder C_lr_20..60_rsi + _s2 refs (user 2026-09-30, CHARTER 19:09)")
launch() {  # REF -> UNIT, T0 ; returns 1 if refused 40 times
  local ref=$1 lout args=()
  [ $FIRST_ACC = 1 ] && args=("${ACC[@]}")
  UNIT=""
  for try in $(seq 40); do
    lout=$(cd $R && $L --campaign-dir $C --reference-chain $ref "${args[@]}" 2>&1)
    UNIT=$(echo "$lout" | head -1)
    case "$UNIT" in cands-*) break ;; esac
    say "launch of $ref refused (try $try): $(echo "$lout" | tail -1 | cut -c1-200); retry in 60 s"
    UNIT=""; sleep 60
  done
  [ -z "$UNIT" ] && { say "STOP $ref: launch refused 40 times"; return 1; }
  FIRST_ACC=0
  T0=$(date +%s)
  say "launched $ref unit $UNIT"
}
early_watch() {  # REF RUN_DIR UNIT
  local ref=$1 rd=$2 unit=$3 out last
  while :; do
    sleep 120
    out=$($PY $C/scripts/early_read.py $rd 400 600 800 1000 2>/dev/null)
    last=$(echo "$out" | sed -nE 's/.*last it ([0-9]+).*/\1/p')
    if [ -n "$last" ] && [ "$last" -ge 1050 ]; then break; fi
    systemctl --user is-active -q $unit || return
  done
  say "EARLY $ref $(echo "$out" | grep EARLY | cut -c1-120) :: $(echo "$out" | grep -E '^\s+it' | awk '{printf "it%s %s  ", $2, $4}')"
}
lr_eval() {  # NAME CHECKPOINT ENV_STAMP -> sets LRL LRR ; returns 1 on failure
  local name=$1 ck=$2 st=$3 envs d
  envs=$(python3 -c "import json;m=json.load(open('$st/run_meta.json'));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))")
  drivers_idle
  ( cd $R/parkour && env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH=$R/parkour/parkour_tasks:$R/parkour $envs \
      $CAP timeout 3600 $PY $EVAL --headless --manifest $LRMAN --scenario side22_25lr__omni \
      --checkpoint $ck --save-raw --no-plot --output-root $LROUT/$name ) > $LROUT/$name.log 2>&1
  d=$(ls -d $LROUT/$name/side22_25lr__omni/seed001/*/ 2>/dev/null | tail -1)
  if [ -z "$d" ] || [ ! -f $d/complete.json ]; then say "SKIP $name: lr eval produced no complete stamp"; return 1; fi
  $PY $C/scripts/side22_classify.py --json $OUT/${name}_lr.json $name=$d 2>&1 | grep -E "lat_|CLASS" >> $LOG
  LRL=$(cls_of $OUT/${name}_lr.json $name class); LRR=$(cls_of $OUT/${name}_lr.json $name class_r)
  say "EVENT ${name}_lr CLASS_L $LRL CLASS_R $LRR"
}
gate_arm() {  # REF -> sets VERDICT LRL LRR ; returns 1 on an arm-level failure (runner continues)
  local ref=$1 rd st
  LRL=""; LRR=""; VERDICT=""
  launch $ref || exit 5
  rd=""; for _ in $(seq 60); do rd=$(newest_run_after $T0); [ -n "$rd" ] && break; sleep 15; done
  [ -z "$rd" ] && { say "SKIP $ref: no run dir (unit $UNIT)"; wait_unit $UNIT; return 1; }
  say "$ref run dir $rd"
  early_watch $ref $rd $UNIT &
  wait_unit $UNIT
  wait
  tail -2 $C/orchestrator.log | cut -c1-300 >> $LOG
  [ -f $rd/model_2999.pt ] || { say "SKIP $ref: no model_2999.pt"; return 1; }
  st=$(any_stamp_of $rd); [ -z "$st" ] && { say "SKIP $ref: no complete driver eval stamp"; return 1; }
  lr_eval $ref $rd/model_2999.pt $st || return 1
  VERDICT=$(decide $LRL $LRR)
  say "EVENT $ref VERDICT $VERDICT"
}

say "eplen-queue start (C_lr_20..60_rsi)"
drivers_idle
cp $PATCH $C/candidates.yaml
say "applied $PATCH"
confirm=()
for E in 20 30 40 50 60; do
  ref=C_lr_${E}_rsi
  gate_arm $ref || continue
  if [ "$LRL" = FAST ] || [ "$LRR" = FAST ]; then confirm+=(${ref}_s2); say "rule: $ref FAST on a hand (L $LRL, R $LRR) -> ${ref}_s2 queued"; fi
done
say "ladder done; seed-2 queue: ${confirm[*]:-none}"
for ref in "${confirm[@]}"; do gate_arm $ref || true; done
say "DONE eplen (seed 2 ran: ${confirm[*]:-none})"

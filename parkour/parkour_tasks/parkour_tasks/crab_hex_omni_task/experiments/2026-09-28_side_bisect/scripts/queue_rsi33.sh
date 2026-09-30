#!/usr/bin/env bash
# Side-bisect RSI retry (user 2026-09-30: if C_lr_10_rsi also fails, retry both RSI arms with a 0.22-0.28 harvest and 33% RSI).
# Functions derived from queue_p1.sh.
#   C_lr OK (FAST both hands)          -> CFG = C_lr
#   C_lr FAIL (SLOW/DEAD both hands)   -> run C_lr_10 (+ s2 if ambiguous): OK -> CFG = C_lr_10; FAIL -> BOTH_FAIL (stop; analysis)
#   anything else ambiguous            -> STOP for a manual decision (decision defaults)
#   with CFG: FB_<CFG> (+ fwd/back, suite eval + both-hands side eval), then P1_<CFG> (full phase 1, 12000 its),
#   then both-hands side eval of P1's c6x head and of W for comparison. Then DONE (the session waits for the user).
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
PB=/home/nickmagus/.claude/plans/side-speed-bisection.evidence/phaseB
LOG=$R/parkour/outputs/omni_plan/side_bisect/queue.log
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
W_CK=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt
W_ST=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29
mkdir -p $OUT $LROUT
say() { echo "[$(date -Is)] $*" >> $LOG; }
wait_unit() { until ! systemctl --user is-active -q $1; do sleep 30; done; }
drivers_idle() { while pgrep -f "isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py" >/dev/null; do sleep 30; done; }
cls_of() { python3 -c "import json;print(json.load(open('$1'))['$2'].get('$3'))"; }
decide() {  # L R -> OK / FAIL / AMBIG
  if [ "$1" = FAST ] && [ "$2" = FAST ]; then echo OK
  elif { [ "$1" = SLOW ] || [ "$1" = DEAD ]; } && { [ "$2" = SLOW ] || [ "$2" = DEAD ]; }; then echo FAIL
  else echo AMBIG; fi
}
any_stamp_of() {  # RUN_DIR -> newest complete eval stamp (any row) of that run
  local rd=$1 tag; tag=$(basename $rd | cut -d_ -f3-)
  ls -d $EV/$tag/*/seed001/*/ 2>/dev/null | while read d; do
    grep -q "$(basename $rd)" $d/run_meta.json 2>/dev/null && [ -f $d/complete.json ] && echo $d; done | tail -1
}
newest_run_after() {  # EPOCH [PATTERN]
  local t=$1 pat=${2:-}
  for d in $(ls -dt $RUNS/*${pat}*/ 2>/dev/null | head -8); do
    [ $(stat -c %Y $d) -ge $t ] && { echo ${d%/}; return; }
  done
}
FIRST_ACC=1
ACC=(--accept-candidates-change "side bisect 2026-09-30: RSI retry arms C_lr_rsi33 / C_lr_10_rsi33 (bank B 0.22-0.28, RSI 0.33), per the CHARTER")
launch() {  # REF -> UNIT, T0
  local ref=$1 lout args=()
  if [ $FIRST_ACC = 1 ]; then args=("${ACC[@]}"); fi
  for try in $(seq 40); do
    lout=$(cd $R && $L --campaign-dir $C --reference-chain $ref "${args[@]}" 2>&1)
    UNIT=$(echo "$lout" | head -1)
    case "$UNIT" in cands-*) break ;; esac
    say "launch of $ref refused (try $try): $(echo "$lout" | tail -1 | cut -c1-200); retry in 60 s"
    UNIT=""; sleep 60
  done
  if [ -z "$UNIT" ]; then say "STOP $ref: launch refused 40 times"; exit 5; fi
  FIRST_ACC=0
  T0=$(date +%s)
  say "launched $ref unit $UNIT"
}
lr_eval() {  # NAME CHECKPOINT ENV_STAMP -> sets LRL LRR
  local name=$1 ck=$2 st=$3 envs d
  envs=$(python3 -c "import json;m=json.load(open('$st/run_meta.json'));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))")
  drivers_idle
  ( cd $R/parkour && env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH=$R/parkour/parkour_tasks:$R/parkour $envs \
      $CAP timeout 3600 $PY $EVAL --headless --manifest $LRMAN --scenario side22_25lr__omni \
      --checkpoint $ck --save-raw --no-plot --output-root $LROUT/$name ) > $LROUT/$name.log 2>&1
  d=$(ls -d $LROUT/$name/side22_25lr__omni/seed001/*/ 2>/dev/null | tail -1)
  if [ -z "$d" ] || [ ! -f $d/complete.json ]; then say "STOP $name: lr eval produced no complete stamp"; exit 7; fi
  $PY $PB/side22_classify.py --json $OUT/${name}_lr.json $name=$d 2>&1 | grep -E "lat_|CLASS" >> $LOG
  LRL=$(cls_of $OUT/${name}_lr.json $name class); LRR=$(cls_of $OUT/${name}_lr.json $name class_r)
  say "EVENT ${name}_lr CLASS_L $LRL CLASS_R $LRR"
}
gate_arm() {  # REF (single-stage side-only arm) -> sets VERDICT
  local ref=$1 rd st
  launch $ref
  rd=""; for _ in $(seq 60); do rd=$(newest_run_after $T0); [ -n "$rd" ] && break; sleep 15; done
  [ -z "$rd" ] && { say "STOP $ref: no run dir"; exit 4; }
  say "$ref run dir $rd"
  wait_unit $UNIT
  tail -2 $C/orchestrator.log | cut -c1-300 >> $LOG
  st=$(any_stamp_of $rd); [ -z "$st" ] && { say "STOP $ref: no complete eval stamp"; exit 6; }
  lr_eval $ref $rd/model_2999.pt $st
  VERDICT=$(decide $LRL $LRR)
  # campaign default for DEAD: diagnose + seed 2 -> treat a DEAD hand as ambiguous so the tree runs the seed-2 arm
  if [ "$LRL" = DEAD ] || [ "$LRR" = DEAD ]; then [ "$VERDICT" = FAIL ] && VERDICT=AMBIG; fi
  say "EVENT $ref VERDICT $VERDICT"
}

BANK=$C/rsi_bank_side_lr_b.npz
HOUT=$R/parkour/outputs/omni_plan/side_bisect/canon/harvest_b
verdict_of() { grep "EVENT $1 VERDICT" $LOG | tail -1 | awk '{print $NF}'; }
say "rsi33-queue armed (waits for the C_lr_10_rsi queue to end)"
until grep -q "DONE c10rsi""-queue" $LOG; do sleep 30; done
V=$(verdict_of C_lr_10_rsi); V2=$(verdict_of C_lr_10_rsi_s2)
FAILED=0
if [ "$V" = FAIL ]; then FAILED=1; elif [ "$V" = AMBIG ] && [ "$V2" = FAIL ]; then FAILED=1; fi
say "EVENT C_lr_10_rsi final (s3 $V, s2 ${V2:-none}) -> retry trigger $FAILED"
if [ $FAILED = 0 ]; then say "STOP rsi33: not triggered (C_lr_10_rsi did not fail)"; say "DONE rsi33-queue"; exit 0; fi
drivers_idle
say "rsi33: harvest B with C (canon wave, 0.22-0.28)"
st=$(python3 -c "import json;print(json.load(open('$OUT/C.json'))['C']['stamp'])")
ck=$(python3 -c "import json;print(json.load(open('$OUT/C.json'))['C']['checkpoint'])")
envs=$(python3 -c "import json;m=json.load(open('$st/run_meta.json'));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))")
( cd $R/parkour && env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH=$R/parkour/parkour_tasks:$R/parkour $envs \
    $CAP timeout 3600 $PY $EVAL --headless --manifest $C/scenarios_harvest_side_b.yaml --scenario harvest_side_lr_b__omni \
    --checkpoint $ck --save-raw --no-plot --canon wave --output-root $HOUT ) > $R/parkour/outputs/omni_plan/side_bisect/harvest_b_C.log 2>&1
hs=$(ls -d $HOUT/harvest_side_lr_b__omni/seed001/*/ 2>/dev/null | tail -1)
if [ -z "$hs" ] || [ ! -f $hs/complete.json ]; then say "STOP harvest B: no complete stamp"; exit 9; fi
say "harvest B stamp $hs"
$PY $C/scripts/side22_classify.py --json $OUT/harvest_b_C.json harvest_b_C=$hs 2>&1 | grep -E "lat_" >> $LOG
$PY $C/scripts/harvest_canon_bank.py --out $BANK --per-dir 2000 --seed 0 $hs > $R/parkour/outputs/omni_plan/side_bisect/harvest_bank_b.json 2>&1 || { say "STOP bank B conversion failed"; exit 9; }
say "EVENT bank B written $BANK"
drivers_idle
cp $PB/candidates_v14.yaml $C/candidates.yaml
say "applied candidates_v14.yaml"
gate_arm C_lr_rsi33
[ $VERDICT = AMBIG ] && { say "rule: C_lr_rsi33 ambiguous -> s2"; gate_arm C_lr_rsi33_s2; }
gate_arm C_lr_10_rsi33
[ $VERDICT = AMBIG ] && { say "rule: C_lr_10_rsi33 ambiguous -> s2"; gate_arm C_lr_10_rsi33_s2; }
say "DONE rsi33-queue"

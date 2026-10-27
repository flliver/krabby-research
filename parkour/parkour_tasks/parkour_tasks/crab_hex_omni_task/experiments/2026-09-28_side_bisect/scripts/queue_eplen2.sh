#!/usr/bin/env bash
# Episode-length ladder runner v2 (CHARTER 2026-09-30 19:09 + review 20:00). Same plan as queue_eplen.sh (steps
# 20..60 in order regardless of results; any step FAST on either hand by feet_ss -> its _s2 after all five), with the
# review's runner fixes: run dir from the orchestrator's run name (F5), unit death detected while polling with the
# journal tail logged (F1), head taken from state.json after the unit ends so an infra retry is followed (F2),
# drivers_idle before every launch (F3), X17 driver-stamp env fallback (F4), plus the direction check
# (along_cmd.py, CC-1/EL-2) logged per arm.
# Usage: queue_eplen2.sh [ADOPT_REF ADOPT_UNIT ADOPT_T0]   (adopt an arm that is already training, then continue)
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
ORCH=$C/orchestrator.log
FALLBACK_ST=$EV/seed3_X17_s1_A15pB/side22_25__omni/seed001/2026-09-30_15-43-35
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
mkdir -p $OUT $LROUT
say() { echo "[$(date -Is)] $*" >> $LOG; }
wait_unit() { until ! systemctl --user is-active -q $1; do sleep 30; done; }
drivers_idle() { while ps -eo args | grep -qE "^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py"; do sleep 30; done; }
cls_of() { python3 -c "import json;print(json.load(open('$1'))['$2'].get('$3'))"; }
decide() {
  if [ "$1" = FAST ] && [ "$2" = FAST ]; then echo OK
  elif { [ "$1" = SLOW ] || [ "$1" = DEAD ]; } && { [ "$2" = SLOW ] || [ "$2" = DEAD ]; }; then echo FAIL
  else echo AMBIG; fi
}
any_stamp_of() {
  local rd=$1 tag; tag=$(basename $rd | cut -d_ -f3-)
  ls -d $EV/$tag/*/seed001/*/ 2>/dev/null | while read d; do
    grep -q "$(basename $rd)" $d/run_meta.json 2>/dev/null && [ -f $d/complete.json ] && echo $d; done | tail -1
}
orch_run_name_after() {  # "YYYY-MM-DD HH:MM:SS" -> run name of the first training line at/after it
  python3 - "$1" "$ORCH" <<'PY'
import re, sys
t0, path = sys.argv[1], sys.argv[2]
for line in open(path, errors="replace"):
    m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (seed\d+_X\d+_s1_\w+): Isaac-", line)
    if m and m.group(1) >= t0:
        print(m.group(2)); break
PY
}
head_of() {  # REF T0_EPOCH -> ckpt recorded in state.json for REF at/after T0 (follows an infra retry)
  python3 - "$1" "$2" "$C/state.json" <<'PY'
import json, sys, datetime, os
ref, t0, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
recs = json.load(open(path)).get("ladder", {}).get("references", {}).get(ref, [])
for r in reversed(recs):
    at = datetime.datetime.fromisoformat(r["at"]).timestamp()
    if at >= t0 - 60 and os.path.exists(r["ckpt"]):
        print(r["ckpt"]); break
PY
}
FIRST_ACC=1
ACC=(--accept-candidates-change "v16: episode-length ladder C_lr_20..60_rsi + _s2 refs (user 2026-09-30, CHARTER 19:09)")
launch() {  # REF -> UNIT, T0, T0S
  local ref=$1 lout args=()
  [ $FIRST_ACC = 1 ] && args=("${ACC[@]}")
  UNIT=""
  for try in $(seq 40); do
    drivers_idle
    T0S=$(date '+%Y-%m-%d %H:%M:%S'); T0=$(date +%s)
    lout=$(cd $R && $L --campaign-dir $C --reference-chain $ref "${args[@]}" 2>&1)
    UNIT=$(echo "$lout" | head -1)
    case "$UNIT" in cands-*) break ;; esac
    say "launch of $ref refused (try $try): $(echo "$lout" | tail -1 | cut -c1-200); retry in 60 s"
    UNIT=""; sleep 60
  done
  [ -z "$UNIT" ] && { say "STOP $ref: launch refused 40 times"; return 1; }
  FIRST_ACC=0
  say "launched $ref unit $UNIT"
}
early_watch() {
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
lr_eval() {  # NAME CHECKPOINT ENV_STAMP -> sets LRL LRR
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
  $PY $C/scripts/along_cmd.py --json $OUT/${name}_along.json $name=$d 2>&1 | grep -E "lat_|CLASS" >> $LOG
  say "CHECK ${name} along-command: L $(cls_of $OUT/${name}_along.json $name class_along) R $(cls_of $OUT/${name}_along.json $name class_along_r)"
}
post_arm() {  # REF -> needs UNIT T0 set; waits, gates; sets VERDICT LRL LRR
  local ref=$1 ck rd st
  wait_unit $UNIT
  tail -2 $ORCH | cut -c1-300 >> $LOG
  ck=$(head_of $ref $T0)
  if [ -z "$ck" ]; then
    say "SKIP $ref: no head recorded in state.json after launch; unit journal: $(journalctl --user -u $UNIT -n 3 --no-pager 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
    return 1
  fi
  rd=$(dirname $ck)
  st=$(any_stamp_of $rd)
  if [ -z "$st" ]; then st=$FALLBACK_ST; say "note $ref: no driver eval stamp; using the X17 stamp env (identical eval env, review F4)"; fi
  lr_eval $ref $ck $st || return 1
  VERDICT=$(decide $LRL $LRR)
  say "EVENT $ref VERDICT $VERDICT"
}
gate_arm() {
  local ref=$1 rn rd=""
  LRL=""; LRR=""; VERDICT=""
  launch $ref || exit 5
  for _ in $(seq 90); do
    rn=$(orch_run_name_after "$T0S")
    if [ -n "$rn" ]; then rd=$(ls -dt $RUNS/*_${rn} 2>/dev/null | head -1); [ -n "$rd" ] && break; fi
    systemctl --user is-active -q $UNIT || break
    sleep 10
  done
  if [ -z "$rd" ]; then
    if systemctl --user is-active -q $UNIT; then say "note $ref: run dir not found yet (unit alive); early read skipped"
    else say "SKIP $ref: unit ended before a run dir appeared; journal: $(journalctl --user -u $UNIT -n 5 --no-pager 2>/dev/null | tr '\n' ' ' | cut -c1-400)"; return 1; fi
  else
    say "$ref run dir $rd"
    early_watch $ref $rd $UNIT &
  fi
  post_arm $ref
  local rc=$?
  wait
  return $rc
}

say "eplen-queue v2 start (args: $*)"
confirm=()
note_fast() { if [ "$LRL" = FAST ] || [ "$LRR" = FAST ]; then confirm+=(${1}_s2); say "rule: $1 FAST on a hand (L $LRL, R $LRR) -> ${1}_s2 queued"; fi; }
steps=(20 30 40 50 60)
if [ $# -ge 3 ]; then
  ref=$1; UNIT=$2; T0=$3; FIRST_ACC=0
  say "adopting $ref (unit $UNIT, launched $(date -d @$T0 -Is))"
  if post_arm $ref; then note_fast $ref; fi
  steps=(); for E in 20 30 40 50 60; do [ "C_lr_${E}_rsi" = "$ref" ] && { found=1; continue; }; [ "${found:-0}" = 1 ] && steps+=($E); done
fi
for E in "${steps[@]}"; do
  ref=C_lr_${E}_rsi
  if gate_arm $ref; then note_fast $ref; fi
done
say "ladder done; seed-2 queue: ${confirm[*]:-none}"
for ref in "${confirm[@]}"; do gate_arm $ref || true; done
say "DONE eplen (seed 2 ran: ${confirm[*]:-none})"

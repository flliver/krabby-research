#!/usr/bin/env bash
# ROOT_CAUSE.md controls (CHARTER 2026-09-30 15:40; user "Run those two controls"): C_fa then C_40, serially.
# Each arm is +y only, scored on the driver's side22_25 stamp (lat_l). INTERMEDIATE / DEAD -> that arm's _s2,
# run after both seed-3 arms. Log lines: "EARLY <ref>" at it >= 1050, "EVENT <ref> CLASS <c>", then "DONE controls".
set -u
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
C=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect
L=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/launch_candidates.sh
EV=$R/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect
RUNS=$R/parkour/logs/rsl_rl/crab_hex_omni_flat_walk
OUT=$R/parkour/outputs/omni_plan/side_bisect/arms
LOG=$R/parkour/outputs/omni_plan/side_bisect/queue.log
PATCH=$C/scripts/candidates_v15.yaml
WHY="v15: add C_40 + C_fa_s2/C_40_s2 refs (ROOT_CAUSE.md controls; user 'Run those two controls' 2026-09-30)"
mkdir -p $OUT
say() { echo "[$(date -Is)] $*" >> $LOG; }
wait_unit() { until ! systemctl --user is-active -q $1; do sleep 30; done; }

classify() {  # NAME RUN_DIR -> echoes class; writes $OUT/NAME.json
  local name=$1 rd=$2 tag stamp
  tag=$(basename $rd | cut -d_ -f3-)
  stamp=$(ls -d $EV/$tag/side22_25__omni/seed001/*/ 2>/dev/null | while read d; do
    grep -q "$(basename $rd)" $d/run_meta.json 2>/dev/null && echo $d; done | tail -1)
  if [ -z "$stamp" ] || [ ! -f "$stamp/complete.json" ]; then echo "NOSTAMP"; return; fi
  $PY $C/scripts/side22_classify.py --json $OUT/$name.json $name=$stamp 2>&1 | grep -E "lat_|CLASS" >> $LOG
  python3 -c "import json;print(json.load(open('$OUT/$name.json'))['$name'].get('class'))"
}

newest_run_after() {
  local t=$1
  for d in $(ls -dt $RUNS/*/ | head -5); do
    [ $(stat -c %Y $d) -ge $t ] && { echo ${d%/}; return; }
  done
}

early_watch() {  # REF RUN_DIR UNIT: log the early read once training passes it 1050 (or the unit ends)
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

run_arm() {  # REF [extra launcher args...] -> echoes class
  local ref=$1; shift
  local t0 unit rd cls lout
  for try in $(seq 40); do
    lout=$(cd $R && $L --campaign-dir $C --reference-chain $ref "$@" 2>&1)
    unit=$(echo "$lout" | head -1)
    case "$unit" in cands-*) break ;; esac
    say "launch of $ref refused (try $try): $(echo "$lout" | tail -1 | cut -c1-200); retry in 60 s"
    unit=""; sleep 60
  done
  if [ -z "$unit" ]; then say "STOP $ref: launch refused 40 times"; exit 5; fi
  t0=$(date +%s)
  say "launched $ref unit $unit :: $(echo "$lout" | tail -1)"
  rd=""
  for _ in $(seq 60); do rd=$(newest_run_after $t0); [ -n "$rd" ] && break; sleep 15; done
  if [ -z "$rd" ]; then say "STOP $ref: no run dir appeared (unit $unit: $(systemctl --user is-active $unit))"; exit 4; fi
  say "$ref run dir $rd"
  early_watch $ref $rd $unit &
  wait_unit $unit
  wait
  tail -2 $C/orchestrator.log | cut -c1-300 >> $LOG
  cls=$(classify $ref $rd)
  say "EVENT $ref CLASS $cls"
  echo $cls
}

say "controls runner start (C_fa, C_40)"
while ps -eo args | grep -qE "^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py"; do sleep 30; done
cp $PATCH $C/candidates.yaml
say "applied $PATCH"
c_fa=$(run_arm C_fa --accept-candidates-change "$WHY")
[ -n "$c_fa" ] || { say "STOP controls: C_fa did not run (see above)"; exit 6; }
c_40=$(run_arm C_40)
[ -n "$c_40" ] || { say "STOP controls: C_40 did not run (see above)"; exit 6; }
redo=()
case "$c_fa" in INTERMEDIATE|DEAD) redo+=(C_fa_s2) ;; esac
case "$c_40" in INTERMEDIATE|DEAD) redo+=(C_40_s2) ;; esac
say "seed-3 classes: C_fa $c_fa, C_40 $c_40; seed-2 queue: ${redo[*]:-none}"
for ref in "${redo[@]}"; do run_arm $ref > /dev/null; done
say "DONE controls (C_fa $c_fa, C_40 $c_40${redo:+; seed 2 ran: ${redo[*]}})"

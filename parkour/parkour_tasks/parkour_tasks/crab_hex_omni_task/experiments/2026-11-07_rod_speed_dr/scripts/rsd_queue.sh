#!/bin/bash
# Rod-speed DR campaign: ONE detached queue runner for the serial GPU stages (memory: gpu-queue-chaining). Each stage
# is a pre-registered rsd_gate.py / rsd_train.py command; the runner decides nothing of its own beyond the exit codes
# and the bridge verdict (PASS continues; RELAUNCH_DUE relaunches once, CHARTER 3.3; anything else stops).
#
#   rsd_queue.sh STAGE [STAGE ...]        e.g.  rsd_queue.sh refarm bridge train_tdr policy_tdr g1_tdr read_g1
#
#   refarm       old T*'s reference arm (27 rows)               bridge      the code-path bridge (+ one relaunch)
#   train_tdr    TDR fine-tune (rsd_train --arm dr)             policy_tdr  register model_28991 as TDR
#   g1_tdr       TDR's G1 rows (39)                             read_g1     results/g1.json from TDR's rows
#   train_tctl   TCTL (rsd_train --arm ctl)                     policy_tctl register model_28991 as TCTL
#   tctl         TCTL's rows (12)                               closing     TDR's closing row (L090 nominal es1, Z)
#   read_g1c     results/g1.json with TCTL
#   wait_distill wait for the student distill RSD_ST_s19 to end (rc 0, else stop)
#   sel          the selection reproduction at 90 kg -> policy PICK   g2ref  c28000's in-pin reference arm
#   bridge_c28k  the c28000 code-path bridge (+ one relaunch)   g2          the pick's G2 rows, watch cells, closing
#   g2cf         g2 after a c28000 bridge STOP diagnosed and recorded as a class flip (--bridge-class-flip)
#   read_g2      results/g2.json
#
# Sentinels (one line each, appended to $OUT/queue.log): "RSDQ STEP <stage>", "RSDQ DONE <stage> rc=<rc>",
# "RSDQ STOP <stage> <why>", "RSDQ END <stages>". A stage's stdout+stderr go to $OUT/<stage>.driver.out.
# Stop the queue between rows with `touch <campaign>/STOP_QUEUE` (rsd_gate) or `touch results/runs/<name>.stop`
# (rsd_train); the runner then stops at that stage's non-zero exit.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CD=$(cd "$HERE/.." && pwd)
W=$(cd "$HERE/../../../../../../.." && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
G=("$PY" -B -I "$HERE/rsd_gate.py")
TR=("$PY" -I -B "$HERE/rsd_train.py")
OUT=${RSD_TP_OUT:-/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rod_speed_dr}
QLOG=$OUT/queue.log
OMNI=$W/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task
CK_TSTAR=$OMNI/policy/f19_8_p3b_payload180/model_25992.pt
EF=$OMNI/experiments/2026-10-16_task4_pipeline/F19/envs/p3b.distill.env
RUNS=$W/parkour/logs/rsl_rl/crab_hex_omni_teacher
SRUNS=$W/parkour/logs/rsl_rl/crab_hex_omni_student
ST_NAME=RSD_ST_s19
mkdir -p "$OUT"

say() { echo "RSDQ $* $(date -Iseconds)" | tee -a "$QLOG"; }
stop() { say STOP "$1" "$2"; exit 1; }
run() {  # STAGE CMD...  -> the command's rc; output appended to $OUT/STAGE.driver.out
  local stage=$1; shift
  echo "### $(date -Iseconds) $*" >> "$OUT/$stage.driver.out"
  "$@" >> "$OUT/$stage.driver.out" 2>&1
}
verdict() { "$PY" -B -I -c 'import json,sys; print((json.load(open(sys.argv[1])) or {}).get("verdict"))' "$1" 2>/dev/null; }
one_run_dir() {  # NAME -> the single run dir *_NAME holding model_28991.pt, else empty
  local d n=0 hit=""
  for d in "$RUNS"/*_"$1"; do [ -d "$d" ] && { n=$((n + 1)); hit=$d; }; done
  [ "$n" -eq 1 ] && [ -f "$hit/model_28991.pt" ] && echo "$hit"
}
one_student_dir() {  # -> the single student run dir *_$ST_NAME, else empty
  local d n=0 hit=""
  for d in "$SRUNS"/*_"$ST_NAME"; do [ -d "$d" ] && { n=$((n + 1)); hit=$d; }; done
  [ "$n" -eq 1 ] && echo "$hit"
}
bridge_stage() {  # STAGE POLICY: the code-path bridge with its one relaunch (CHARTER 3.3); anything but PASS stops
  local stage=$1 pol=$2 v
  run "$stage" "${G[@]}" bridge --policy "$pol"
  v=$(verdict "$CD/results/bridge_$pol.json")
  if [ "$v" = RELAUNCH_DUE ]; then
    say DONE "$stage verdict=RELAUNCH_DUE (one relaunch, CHARTER 3.3)"
    run "$stage" "${G[@]}" bridge --policy "$pol" --relaunch
    v=$(verdict "$CD/results/bridge_$pol.json")
  fi
  [ "$v" = PASS ] || stop "$stage" "verdict=$v (results/bridge_$pol.json)"
}

[ $# -ge 1 ] || { echo "usage: rsd_queue.sh STAGE [STAGE ...]" >&2; exit 2; }
for s in "$@"; do
  case "$s" in refarm|bridge|train_tdr|policy_tdr|g1_tdr|read_g1|train_tctl|policy_tctl|tctl|closing|read_g1c) ;;
    wait_distill|sel|g2ref|bridge_c28k|g2|g2cf|read_g2) ;;
    *) echo "unknown stage $s" >&2; exit 2 ;; esac
done
say START "$*"
for s in "$@"; do
  say STEP "$s"
  case "$s" in
    refarm) run refarm "${G[@]}" refarm || stop refarm "rc=$?" ;;
    bridge)
      run bridge "${G[@]}" bridge
      v=$(verdict "$CD/results/bridge_TSTAR.json")
      if [ "$v" = RELAUNCH_DUE ]; then
        say DONE "bridge verdict=RELAUNCH_DUE (one relaunch, CHARTER 3.3)"
        run bridge "${G[@]}" bridge --relaunch
        v=$(verdict "$CD/results/bridge_TSTAR.json")
      fi
      [ "$v" = PASS ] || stop bridge "verdict=$v (results/bridge_TSTAR.json)" ;;
    train_tdr) run train_tdr "${TR[@]}" --arm dr --name RSD_TDR_s19 --checkpoint "$CK_TSTAR" --env-file "$EF" \
                 --wait-idle || stop train_tdr "rc=$? (see train/RSD_TDR_s19.log)" ;;
    policy_tdr)
      d=$(one_run_dir RSD_TDR_s19); [ -n "$d" ] || stop policy_tdr "no single RSD_TDR_s19 run dir with model_28991.pt"
      run policy_tdr "${G[@]}" policy add TDR --checkpoint "$d/model_28991.pt" --role teacher || stop policy_tdr "rc=$?" ;;
    g1_tdr) run g1_tdr "${G[@]}" g1 --policy TDR || stop g1_tdr "rc=$?" ;;
    read_g1) run read_g1 "${G[@]}" read g1 --tdr TDR || stop read_g1 "rc=$?" ;;
    train_tctl) run train_tctl "${TR[@]}" --arm ctl --name RSD_TCTL_s19 --checkpoint "$CK_TSTAR" --env-file "$EF" \
                  --wait-idle || stop train_tctl "rc=$? (see train/RSD_TCTL_s19.log)" ;;
    policy_tctl)
      d=$(one_run_dir RSD_TCTL_s19); [ -n "$d" ] || stop policy_tctl "no single RSD_TCTL_s19 run dir with model_28991.pt"
      run policy_tctl "${G[@]}" policy add TCTL --checkpoint "$d/model_28991.pt" --role teacher || stop policy_tctl "rc=$?" ;;
    tctl) run tctl "${G[@]}" tctl --policy TCTL || stop tctl "rc=$?" ;;
    closing) run closing "${G[@]}" closing --policy TDR || stop closing "rc=$?" ;;
    read_g1c) run read_g1c "${G[@]}" read g1 --tdr TDR --tctl TCTL || stop read_g1c "rc=$?" ;;
    wait_distill)
      until grep -q "TPDISTILL END $ST_NAME " "$OUT/distill/distill.log" 2>/dev/null; do sleep 60; done
      rc=$(sed -n "s/.*TPDISTILL END $ST_NAME rc=\([0-9]*\).*/\1/p" "$OUT/distill/distill.log" | tail -1)
      [ "$rc" = 0 ] || stop wait_distill "distill rc=$rc (see distill/$ST_NAME.log)" ;;
    sel)
      d=$(one_student_dir); [ -n "$d" ] || stop sel "no single student run dir *_$ST_NAME"
      run sel "${G[@]}" sel --student-run "$d" --teacher TDR || stop sel "rc=$? (results/sel.json)" ;;
    g2ref) run g2ref "${G[@]}" g2ref || stop g2ref "rc=$?" ;;
    bridge_c28k) bridge_stage bridge_c28k C28000 ;;
    g2) run g2 "${G[@]}" g2 --policy PICK || stop g2 "rc=$?" ;;
    g2cf) run g2cf "${G[@]}" g2 --policy PICK --bridge-class-flip || stop g2cf "rc=$?" ;;
    read_g2) run read_g2 "${G[@]}" read g2 --pick PICK --tdr TDR || stop read_g2 "rc=$?" ;;
  esac
  say DONE "$s rc=0"
done
say END "$*"

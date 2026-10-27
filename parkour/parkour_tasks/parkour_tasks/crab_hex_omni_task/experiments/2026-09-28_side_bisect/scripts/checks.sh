#!/usr/bin/env bash
# Side-speed bisection, Phase B steps 5-6 (plan ~/.claude/plans/side-speed-bisection.md):
#   5a knob smokes (C with the eval path, R and RC dump-only), 5b W harness check on side22_25__omni,
#   5c sideways FAST pin (1a s3/s2 at model_3000 and model_4999) on side22_25__side,
#   6  GPU-0: W on side holds at 0.15/0.22/0.30/0.35.
# Serial, one Isaac process at a time, each under a 48G memory cap. Run from the MAIN checkout (landed aa2fed42).
set -u
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
OUT=$R/parkour/outputs/omni_plan/side_bisect
EVAL=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
OMAN=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml
SMAN=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task/experiments/eval/scenarios_side.yaml
GMAN=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_gpu0_sidesweep.yaml
CAMP=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect
W=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt
SW=$R/parkour/logs/rsl_rl/crab_hex_side_flat_walk
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
BASEENV=(OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH=$R/parkour/parkour_tasks:$R/parkour)
# W's own recorded eval env (suite__omni stamp 2026-09-25_20-31-29), plan item 6.
WENV=(KRABBY_ACTION_LPF_HZ=0 KRABBY_ACTION_LPF_OBS=0 KRABBY_APEX_W=1.0 KRABBY_CLOCK_W=1.0
      KRABBY_FLAT_TERRAIN_GEOM=ring KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_OMNI_ANG_VEL_CLIP=0.2
      KRABBY_OMNI_ANG_VEL_MAX=0.3 KRABBY_OMNI_CLOCK_HIDE=1 KRABBY_OMNI_CLOCK_TURN_R=1.2
      KRABBY_OMNI_FACING_OBS_CLIP=0.6 KRABBY_OMNI_FACING_OBS_ENCODING=raw KRABBY_OMNI_RELAY_HYST=0
      KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.25:0.25 KRABBY_OMNI_STAND_HOLD=0 KRABBY_OMNI_STAND_OBS=0
      KRABBY_OMNI_STAND_PARK=0 KRABBY_OMNI_YAW_RANGE=-3.14159:3.14159 KRABBY_PLANT=A15+B
      KRABBY_PLAY_DEBUG_VIS=0 KRABBY_RSI_FRAC=0 KRABBY_TRACK_L1_W=-1.0 KRABBY_TRACK_SIGMA2=0.1)
C_OV=(KRABBY_OMNI_MIRROR_IMAGES=fa KRABBY_OMNI_CMD_OBS=scalar KRABBY_RSI_FRAC=0 KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25)
R_OV=(KRABBY_OMNI_CLOCK_COMBINE=sum KRABBY_TRACK_SIGMA2=0.1 KRABBY_TRACK_L1_W=-1.0 KRABBY_YAW_PROGRESS_W=0)
if env | grep -q '^KRABBY_'; then echo "unset every KRABBY_* first"; exit 1; fi
mkdir -p $OUT
S=$OUT/checks_summary.txt
echo "START $(date -Is) head $(git -C $R log -1 --format=%h)" >> $S

wait_clear() {
  for _ in $(seq 240); do
    pgrep -f "isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|rsl_rl/train|crab_hex_phase_cfg_dump)\.py" >/dev/null || return 0
    sleep 5
  done
  echo "another Isaac process never cleared" >> $S; exit 3
}

smoke() {  # name spec overlay...
  local name=$1 spec=$2; shift 2
  local K=$OUT/knob_smoke_$name
  mkdir -p $K && cp $CAMP/candidates.yaml $K/
  local ov=(); for kv in "$@"; do ov+=(--smoke-overlay "$kv"); done
  wait_clear
  ( cd $R && env "${BASEENV[@]}" $CAP timeout 3600 $PY \
      $R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py \
      --campaign-dir $K --smoke-stage "$spec" "${ov[@]}" ) > $K/knob_smoke.log 2>&1
  echo "smoke $name ($spec) exit $? $(date -Is)" >> $S
}

ev() {  # label outroot manifest scenario ckpt env...
  local label=$1 root=$2 man=$3 sid=$4 ck=$5; shift 5
  mkdir -p $root
  wait_clear
  ( cd $R/parkour && env "${BASEENV[@]}" "$@" $CAP timeout 3600 $PY $EVAL --headless --manifest $man \
      --scenario $sid --checkpoint $ck --save-raw --no-plot --output-root $root ) > $root/${label}.log 2>&1
  echo "eval $label exit $? $(date -Is)" >> $S
}

# 5a knob smokes
smoke C  1b:20:side22_25 "${C_OV[@]}"
smoke R  1b:20            "${R_OV[@]}"
smoke RC 1b:20            "${C_OV[@]}" "${R_OV[@]}"
# 5b W harness check
ev W_side22_25 $OUT/harness_W $OMAN side22_25__omni $W "${WENV[@]}"
# 5c sideways FAST pin
for run in 2026-09-11_10-57-55:s3 2026-09-12_00-31-09:s2; do
  d=${run%%:*}; s=${run##*:}
  for ck in model_3000 model_4999; do
    ev side_${s}_${ck} $OUT/pin_side/${s}_${ck} $SMAN side22_25__side $SW/$d/$ck.pt
  done
done
# 6 GPU-0 (report only)
WENV_G=("${WENV[@]/KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.25:0.25/KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25}")
for sp in 015 022 030 035; do
  ev W_sweep$sp $OUT/gpu0 $GMAN sidesweep${sp}__omni $W "${WENV_G[@]}"
done
echo "DONE $(date -Is)" >> $S

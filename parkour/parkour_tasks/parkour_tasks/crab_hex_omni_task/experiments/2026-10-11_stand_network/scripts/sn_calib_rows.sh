#!/usr/bin/env bash
# Stand-network campaign, PLAN.md step 2 (gate calibration): the scripted re-stance controller R on S1 s18 for the
# G4 walls rows R is missing. Runs serially, one Isaac process at a time, through the phase-2 run_arm.sh --row-eval
# path exactly as the p2r probe P2R_RSTg_s18 ran (launcher --p2r-mode restance --p2r-stand-bank, R constants of the
# probe, KRABBY_OMNI_STAND_OBS=clock, the REFS_S1_s18 suite es1 env stamp), under the new stamp name SN_RSTg_s18:
#   field_hard__omni es1  -> identity check against P2R_RSTg_s18 es1 (module/launcher shas moved since 2026-10-10)
#   field_hard__omni es2-4 -> R's field_hard to n = 400 at the current code
# The code-state guard is anchored on REFH_S1_s18 (the last inertness bridge, code_sha 14301cd2 = HEAD's).
# Log lines: SNCAL START / <row> rc=<rc> / DONE (or STOP) in $OUT/sn_calib.log; the session watches that file.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)  # this checkout's root, from the script's own location
P2=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2
OUT=$R/parkour/outputs/omni_plan/phase2
LOG=$R/parkour/outputs/omni_plan/stand_network/sn_calib.log
mkdir -p "$(dirname "$LOG")"
CK=$R/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt
ST=$OUT/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24
RST='lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,tilt_max=0.1,eps_step=0.3'
say() { echo "SNCAL $* $(date -Is)" >> "$LOG"; }
say START SN_RSTg_s18 rst=$RST
for es in 1 2 3 4; do
  if [ -f "$OUT/STOP_QUEUE" ] || [ -f "$(dirname "$LOG")/STOP_QUEUE" ]; then say STOP STOP_QUEUE present; exit 4; fi
  P2_GUARD_REF=REFH_S1_s18 \
  P2_EVAL=$P2/scripts/p2r_mode_eval_crab_hex_gait.py \
  EVAL_EXTRA_ARGS='--p2r-mode restance --p2r-stand-bank' \
  P2_ENV_EXTRA="KRABBY_RESTANCE=$RST KRABBY_OMNI_STAND_OBS=clock" \
    bash "$P2/scripts/run_arm.sh" --row-eval SN_RSTg_s18 "$CK" "$ST" "$P2/scenarios_p2.yaml" field_hard__omni "$es"
  rc=$?
  say SN_RSTg_s18 field_hard__omni es$es rc=$rc
  if [ "$rc" != 0 ]; then say STOP SN_RSTg_s18 field_hard__omni es$es rc=$rc; exit 5; fi
done
say DONE SN_RSTg_s18

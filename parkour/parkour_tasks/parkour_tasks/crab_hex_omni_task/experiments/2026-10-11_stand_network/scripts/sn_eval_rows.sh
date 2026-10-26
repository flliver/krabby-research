#!/usr/bin/env bash
# Stand-network campaign, PLAN.md step 3 -- stand_net eval rows through the phase-2 run_arm.sh --row-eval path.
# NOT run by the builder: the orchestrator runs it, serially, one Isaac process at a time (the launcher's name ends
# in eval_crab_hex_gait.py, so run_arm's drivers_idle sees it; drivers_idle does NOT see the DAgger trainer, so this
# script waits before every row while an sn_dagger.py process runs -- map risk T14; sn_dagger.sh refuses while any eval runs).
#
# Default = the IDENTITY BRIDGE (PLAN step 3; CHARTER 2026-10-12 miss rule): S1 itself as the stand checkpoint,
# --stand-mask 0, head name SN_ID_s18, rows suite__omni es1 / walkstop__omni es1 / field_hard__omni es1, the
# REFS_S1_s18 suite es1 env stamp (STAND_OBS=0, CLOCK_HIDE=1, STD_PARAM soft, INIT_STD 0.8, ellipse 0.35:0.35:0.25),
# guard anchored on REFH_S1_s18 (code_sha 14301cd2 / eval_sha 1d021625). Then, on CPU:
#   sn_bridge_read.py --head SN_ID_s18 --identity        exit 0 = bitwise = REFS on every row (3 = miss <= 1e-5:
#                                                         compare every G1-G5 statistic with
#                                                         sn_calib.py --head SN_ID_s18 --json <campaign>/bridge_calib.json
#                                                         --md <campaign>/bridge_calib.md -- never the default calib.*;
#                                                         4 = STOP and ask)
# A trained head: sn_eval_rows.sh NAME /abs/model_N.pt MASK ROW@ES ... ; then sn_bridge_read.py --head NAME --live.
#
# usage: sn_eval_rows.sh [--dry-run]                                  (the identity bridge, every default)
#        sn_eval_rows.sh [--dry-run] NAME STAND_CKPT MASK [ROW@ES ...] (an explicit NAME needs STAND_CKPT AND MASK:
#                                                                      no S1 / mask-0 default under a head's name)
# Guards (review 2026-10-12): SN_ID_s18 is S1 with mask 0 only; S1 as the stand checkpoint only under an SN_ID* name
# (SN_ID_s18 bridge, SN_IDM_s18 swap-live proof). The launcher itself refuses a trainer checkpoint whose
# infos.sn_dagger.stand_mask is not MASK, and any non-trainer checkpoint other than S1.
# Log lines "SNROWS ..." in $LOG; a STOP_QUEUE file (phase-2 OUT or the stand_network output dir) stops before a row.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)  # this checkout's root, from the script's own location
TASK=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task
P2=$TASK/experiments/2026-10-07_omni_phase2
HERE=$(cd "$(dirname "$0")" && pwd)
LAUNCHER=$HERE/sn_stand_net_eval_crab_hex_gait.py
OUT=$R/parkour/outputs/omni_plan/phase2
SNOUT=$R/parkour/outputs/omni_plan/stand_network
LOG=$SNOUT/sn_rows.log
# NOTE(M18 Task 4 pipeline step 6, 2026-10-16; DESIGN B6): the walker checkpoint and its eval env stamp come from
# SN_WALKER_CK / SN_WALKER_ST when set (a new walker, e.g. the pipeline's T*; set them together with SN_WALKER_SPEC and
# P2_REANCHOR=1 -- the pipeline's pin guard holds the code state); unset = S1 s18 and REFS_S1_s18 exactly, as before.
CK=${SN_WALKER_CK:-$R/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt}
ST=${SN_WALKER_ST:-$OUT/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24}
if [ -n "${SN_WALKER_CK:-}" ] || [ -n "${SN_WALKER_ST:-}" ]; then
  [ -n "${SN_WALKER_CK:-}" ] && [ -n "${SN_WALKER_ST:-}" ] || { echo "sn_eval_rows: SN_WALKER_CK and SN_WALKER_ST go together" >&2; exit 2; }
  case "$CK$ST" in *[[:space:]*?\[\]]*) echo "sn_eval_rows: SN_WALKER_CK / SN_WALKER_ST contain whitespace / glob characters" >&2; exit 2 ;; esac
  case "$CK" in /*) ;; *) echo "sn_eval_rows: SN_WALKER_CK must be absolute: $CK" >&2; exit 2 ;; esac
  case "$ST" in /*) ;; *) echo "sn_eval_rows: SN_WALKER_ST must be absolute: $ST" >&2; exit 2 ;; esac
fi
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
die() { echo "sn_eval_rows: $*" >&2; exit 2; }
ROWS=(suite__omni@1 walkstop__omni@1 field_hard__omni@1)
if [ $# -eq 0 ]; then
  NAME=SN_ID_s18; SCK=$CK; MASK=0                      # the identity bridge: the ONLY call with defaults
else
  [ $# -ge 3 ] || die "an explicit NAME needs STAND_CKPT and MASK too (got $# argument(s)); usage: sn_eval_rows.sh [--dry-run] [NAME STAND_CKPT MASK [ROW@ES ...]]"
  NAME=$1; SCK=$2; MASK=$3; shift 3
  if [ $# -gt 0 ]; then ROWS=("$@"); fi
fi
manifest() {
  case "$1" in
    suite__omni) echo "$TASK/experiments/eval/scenarios_omni.yaml" ;;
    field_hard__omni|field_turn__omni) echo "$P2/scenarios_p2.yaml" ;;
    walkstop__omni) echo "$P2/scenarios_p2s.yaml" ;;
    *) return 1 ;;
  esac
}
case "$NAME" in ''|*[!A-Za-z0-9_]*) die "NAME '$NAME' must be [A-Za-z0-9_]+" ;; esac
case "$MASK" in 0|1) ;; *) die "MASK must be 0 or 1, got '$MASK'" ;; esac
case "$SCK" in /*) ;; *) die "STAND_CKPT must be absolute: $SCK" ;; esac
case "$SCK$LAUNCHER" in *[[:space:]*?\[\]]*) die "a path contains whitespace / glob characters (EVAL_EXTRA_ARGS is unquoted)" ;; esac
[ -f "$SCK" ] || die "no stand checkpoint $SCK"
[ -f "$CK" ] || die "no S1 checkpoint $CK"
IS_S1=0; cmp -s "$SCK" "$CK" && IS_S1=1                # byte-identical to S1 (any path to the same file)
if [ "$NAME" = SN_ID_s18 ] && { [ "$IS_S1" != 1 ] || [ "$MASK" != 0 ]; }; then
  die "SN_ID_s18 is the identity bridge: S1 ($CK) with MASK 0 only, got $SCK mask $MASK"
fi
case "$NAME" in SN_ID*) ;; *) [ "$IS_S1" = 0 ] || die "S1 as the stand checkpoint only under an SN_ID* head name (SN_ID_s18 bridge / SN_IDM_s18 swap-live), not '$NAME'" ;; esac
[ -f "$ST/run_meta.json" ] || die "no REFS env stamp $ST"
[ -f "$LAUNCHER" ] || die "no launcher $LAUNCHER"
[ -z "${P2_ENV_EXTRA:-}" ] || die "P2_ENV_EXTRA must be empty (D5: never STAND_OBS / an env mask in P2_ENV_EXTRA), got '$P2_ENV_EXTRA'"
env | grep -q '^P2R_' && die "P2R_* variables in the environment (the launcher refuses them)"
for spec in "${ROWS[@]}"; do
  row=${spec%@*}; es=${spec##*@}
  case "$es" in ''|*[!0-9]*|0) die "row spec '$spec' must be ROW@ES" ;; esac
  manifest "$row" >/dev/null || die "no manifest known for row '$row'"
  [ -f "$(manifest "$row")" ] || die "manifest missing for $row"
done

[ "$DRY" = 1 ] || mkdir -p "$SNOUT"
say() { [ "$DRY" = 1 ] || echo "SNROWS $* $(date -Is)" >> "$LOG"; }
# map risk T14: run_arm.sh's drivers_idle sees evals only, never the DAgger trainer -> wait while sn_dagger.py runs
# (the Kit-free `sn_dagger.py --read` / `--emit-env` calls are not a trainer). `ps -ww`: plain `ps -eo args` truncates
# every line to $COLUMNS when that is exported (80 from some Python parents) and the anchored patterns never match the
# long Isaac command lines; run_arm.sh's own drivers_idle uses plain `ps -eo args` (phase-2 file, read-only), so the
# same driver wait is repeated here at unlimited width before each row.
TRAINER_PAT='^[^ ]*isaac_venv/bin/python [^ ]*sn_dagger\.py'
NONKIT_PAT='sn_dagger\.py .*--(read|emit-env)([ =]|$)'
DRIVER_PAT='^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py'
trainer_busy() { ps -ww -eo args | grep -E "$TRAINER_PAT" | grep -qvE "$NONKIT_PAT"; }
drivers_busy() { ps -ww -eo args | grep -qE "$DRIVER_PAT"; }
say START "$NAME" stand_ckpt="$SCK" mask="$MASK" rows="${ROWS[*]}"
for spec in "${ROWS[@]}"; do
  row=${spec%@*}; es=${spec##*@}; man=$(manifest "$row")
  if [ -f "$OUT/STOP_QUEUE" ] || [ -f "$SNOUT/STOP_QUEUE" ]; then say STOP STOP_QUEUE present; exit 4; fi
  cmd=(env P2_GUARD_REF=REFH_S1_s18 P2_EVAL="$LAUNCHER" P2_ENV_EXTRA=
       EVAL_EXTRA_ARGS="--stand-ckpt $SCK --stand-mask $MASK"
       bash "$P2/scripts/run_arm.sh" --row-eval "$NAME" "$CK" "$ST" "$man" "$row" "$es")
  if [ "$DRY" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; continue; fi
  if trainer_busy; then say WAIT sn_dagger.py running; while trainer_busy; do sleep 30; done; fi
  if drivers_busy; then say WAIT an eval / driver running; while drivers_busy; do sleep 30; done; fi
  "${cmd[@]}"; rc=$?
  say "$NAME" "$row" es"$es" rc="$rc"
  if [ "$rc" != 0 ]; then say STOP "$NAME" "$row" es"$es" rc="$rc"; exit 5; fi
done
say DONE "$NAME"

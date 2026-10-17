#!/usr/bin/env bash
# Stand-height campaign, PLAN.md step 4 -- stand_height eval rows through the phase-2 run_arm.sh --row-eval path.
# NOT run by the builder: the orchestrator runs it, serially, one Isaac process at a time (the launcher's name ends in
# eval_crab_hex_gait.py, so run_arm's drivers_idle sees it; drivers_idle sees evals ONLY, so before EVERY row this
# script waits while ANY other Isaac python runs: a DAgger trainer, a table / ground test, a streaming video recorder,
# a train / play script, an eval or driver -- ISAAC_PAT below, sh_table_test.sh's list; the Kit-free --read /
# --emit-env / --emit-facts / --merge calls and the campaign's CPU readers are exempt, NONKIT_PAT).
#
# Default = the IDENTITY BRIDGE (PLAN step 4): S1 itself as the stand checkpoint, --stand-mask 0, --sh-controller
# stand_net, --sh-command zero, --sh-restore armed (the new clip path, restore armed), head name SH_ID_s18, rows
# suite__omni / walkstop__omni / field_hard__omni es1 on their SHARED manifests, the REFS_S1_s18 suite es1 env stamp,
# guard anchored on REFH_S1_s18 (code_sha 14301cd2 / eval_sha 1d021625). Then, on CPU:
#   sh_bridge_read.py --head SH_ID_s18 --identity     exit 0 = bitwise = REFS on every row and raw key (3 = miss <=
#                                                      1e-5: the stand-network campaign's G1-G5 miss rule; 4 = STOP)
# K5 (sn_calib G1-G5, unchanged reads) needs a head's ZERO-COMMAND stamps on exactly these row@es -- queue them in
# one pass (sh_eval_common.K5_ROWS):
#   sh_eval_rows.sh NAME stand_net zero armed CKPT 1 suite__omni@1 suite__omni@2 walkstop__omni@1 walkstop__omni@2 \
#        field_turn__omni@1 field_hard__omni@1 field_hard__omni@2 field_hard__omni@3 field_hard__omni@4
# Controls / heads (PLAN step-4 control table):
#   sh_eval_rows.sh NAME CONTROLLER COMMAND RESTORE STAND_CKPT MASK [ROW@ES ...]
#     CONTROLLER  stand_net | amask_old (network: STAND_CKPT absolute + MASK 0|1)
#                 r_h | r_old | posture_only (scripted: STAND_CKPT and MASK are '-')
#     COMMAND     zero (zero-command rows: suite / walkstop / field_hard / field_turn) | manifest (height rows:
#                 hwalkstop / hraise,
#                 scenarios_sh.yaml); default rows by COMMAND at es1
#     RESTORE     armed | off (off = hand back at height: the K4 known-bad)
#   e.g. R_h on the height rows es1 + es2:  sh_eval_rows.sh SH_RH_ctl r_h manifest armed - - hwalkstop__omni@1
#        hwalkstop__omni@2 hraise__omni@1 hraise__omni@2 ;  then sh_bridge_read.py --head SH_RH_ctl --smoke --rows ...
# Residual deadband (CHARTER section 3, 2026-10-14 "USER DECISION at the step-6 stop: probe a residual deadband"; off
# by default): an optional leading --deadband X (0 < X < 1) before NAME adds `--sh-deadband X` to every row's launcher
# args (sh_core.apply_deadband: hip / knee residual entries below X zeroed after the +-1 clamp, before + q; recorded in
# run_meta p2r_mode.stand_height.residual_deadband). Network controllers only, never the identity bridge (no default
# NAME, no SH_ID* name, never S1 as the stand checkpoint). Give the head its own NAME (e.g. SH_P18db): sh_read refuses
# a head whose stamps mix deadbands. e.g. the step-6 probe's height rows:
#   sh_eval_rows.sh --deadband 0.25 SH_P18db stand_net manifest armed .../runs/SH_P18/model_5000.pt 1 \
#        hwalkstop__omni@1 hwalkstop__omni@2 hraise__omni@1 hraise__omni@2
# Guards: SH_ID_s18 is exactly the bridge (S1, mask 0, stand_net, zero, armed, no deadband); S1 as the stand checkpoint
# only under an SH_ID* name; a row's command must be its manifest's (zero on identity rows, manifest on height rows);
# P2_ENV_EXTRA empty; no P2R_* / SH_* / SN_STAND_* variables in the environment (the launcher's fallbacks never set
# here).
# Log lines "SHROWS ..." in $LOG; a STOP_QUEUE file (phase-2 OUT or the stand_height output dir) stops before a row.
#
# usage: sh_eval_rows.sh [--dry-run]
#        sh_eval_rows.sh [--dry-run] [--deadband X] NAME CONTROLLER COMMAND RESTORE STAND_CKPT MASK [ROW@ES ...]
set -u
R=/home/nickmagus/krabby/krabby-research
TASK=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task
EXP=$TASK/experiments
P2=$EXP/2026-10-07_omni_phase2
HERE=$(cd "$(dirname "$0")" && pwd)
CAMP=$(cd "$HERE/.." && pwd)
LAUNCHER=$HERE/sh_eval_crab_hex_gait.py
MAN_SH=$CAMP/scenarios_sh.yaml
OUT=$R/parkour/outputs/omni_plan/phase2
SHOUT=$R/parkour/outputs/omni_plan/stand_height
LOG=$SHOUT/sh_rows.log
# NOTE(M18 Task 4 pipeline step 6, 2026-10-16; DESIGN B6): the walker checkpoint and its eval env stamp come from
# SN_WALKER_CK / SN_WALKER_ST when set (a new walker, e.g. the pipeline's T*; set them together with SN_WALKER_SPEC and
# P2_REANCHOR=1 -- the pipeline's pin guard holds the code state); unset = S1 s18 and REFS_S1_s18 exactly, as before.
CK=${SN_WALKER_CK:-$R/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt}
ST=${SN_WALKER_ST:-$OUT/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24}
if [ -n "${SN_WALKER_CK:-}" ] || [ -n "${SN_WALKER_ST:-}" ]; then
  [ -n "${SN_WALKER_CK:-}" ] && [ -n "${SN_WALKER_ST:-}" ] || { echo "sh_eval_rows: SN_WALKER_CK and SN_WALKER_ST go together" >&2; exit 2; }
  case "$CK$ST" in *[[:space:]*?\[\]]*) echo "sh_eval_rows: SN_WALKER_CK / SN_WALKER_ST contain whitespace / glob characters" >&2; exit 2 ;; esac
  case "$CK" in /*) ;; *) echo "sh_eval_rows: SN_WALKER_CK must be absolute: $CK" >&2; exit 2 ;; esac
  case "$ST" in /*) ;; *) echo "sh_eval_rows: SN_WALKER_ST must be absolute: $ST" >&2; exit 2 ;; esac
fi
DRY=0; DB=""
die() { echo "sh_eval_rows: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --deadband|--deadband=*)
      [ -z "$DB" ] || die "--deadband given twice"
      if [ "$1" = --deadband ]; then
        [ $# -ge 2 ] || die "--deadband needs a value X (0 < X < 1)"
        DB=$2; shift 2
      else
        DB=${1#--deadband=}; shift
      fi
      [ -n "$DB" ] || die "--deadband needs a value X (0 < X < 1), got an empty one" ;;
    *) break ;;
  esac
done
if [ -n "$DB" ]; then
  case "$DB" in *[!0-9.]*|.|*.*.*) die "--deadband X must be a plain decimal number (0 < X < 1), got '$DB'" ;; esac
  awk -v x="$DB" 'BEGIN { exit !(x + 0 > 0 && x + 0 < 1) }' || die "--deadband X must satisfy 0 < X < 1 (omit it for 0), got '$DB'"
  [ $# -gt 0 ] || die "--deadband needs an explicit NAME CONTROLLER ...: the default run is the identity bridge (no deadband)"
fi
IDROWS=(suite__omni@1 walkstop__omni@1 field_hard__omni@1)
HROWS=(hwalkstop__omni@1 hraise__omni@1)
if [ $# -eq 0 ]; then
  NAME=SH_ID_s18; CTL=stand_net; CMD=zero; RST=armed; SCK=$CK; MASK=0   # the identity bridge: the ONLY defaults
else
  [ $# -ge 6 ] || die "an explicit NAME needs CONTROLLER COMMAND RESTORE STAND_CKPT MASK too (got $# argument(s)); usage: sh_eval_rows.sh [--dry-run] [NAME CONTROLLER COMMAND RESTORE STAND_CKPT MASK [ROW@ES ...]]"
  NAME=$1; CTL=$2; CMD=$3; RST=$4; SCK=$5; MASK=$6; shift 6
fi
case "$CMD" in zero) ROWS=("${IDROWS[@]}") ;; manifest) ROWS=("${HROWS[@]}") ;; *) die "COMMAND must be zero or manifest, got '$CMD'" ;; esac
if [ $# -gt 0 ]; then ROWS=("$@"); fi
# the zero-command rows run on their SHARED manifests (scenarios_sh.yaml stand_height.identity_rows lists the bridge's
# paths, sh_eval_common.K5_EXTRA_ZERO_ROWS the K5-only field_turn__omni: sh_eval_common.zero_command_rows);
# the height rows on scenarios_sh.yaml
manifest() {
  case "$1" in
    suite__omni) echo "$EXP/eval/scenarios_omni.yaml" ;;
    walkstop__omni) echo "$P2/scenarios_p2s.yaml" ;;
    field_hard__omni|field_turn__omni) echo "$P2/scenarios_p2.yaml" ;;
    hwalkstop__omni|hraise__omni) echo "$MAN_SH" ;;
    *) return 1 ;;
  esac
}
row_command() {
  case "$1" in
    suite__omni|walkstop__omni|field_hard__omni|field_turn__omni) echo zero ;;
    hwalkstop__omni|hraise__omni) echo manifest ;;
    *) return 1 ;;
  esac
}
case "$NAME" in ''|*[!A-Za-z0-9_]*) die "NAME '$NAME' must be [A-Za-z0-9_]+" ;; esac
case "$CTL" in stand_net|amask_old) NET=1 ;; r_h|r_old|posture_only) NET=0 ;; *) die "CONTROLLER must be stand_net / amask_old / r_h / r_old / posture_only, got '$CTL'" ;; esac
case "$RST" in armed|off) ;; *) die "RESTORE must be armed or off, got '$RST'" ;; esac
IS_S1=0
if [ "$NET" = 1 ]; then
  case "$MASK" in 0|1) ;; *) die "MASK must be 0 or 1 for a network controller, got '$MASK'" ;; esac
  case "$SCK" in /*) ;; *) die "STAND_CKPT must be absolute: $SCK" ;; esac
  case "$SCK" in *[[:space:]*?\[\]]*) die "STAND_CKPT contains whitespace / glob characters (EVAL_EXTRA_ARGS is unquoted)" ;; esac
  [ -f "$SCK" ] || die "no stand checkpoint $SCK"
  cmp -s "$SCK" "$CK" && IS_S1=1                    # byte-identical to S1 (any path to the same file)
else
  [ "$SCK" = - ] && [ "$MASK" = - ] || die "a scripted controller ($CTL) takes STAND_CKPT '-' and MASK '-', got '$SCK' '$MASK'"
fi
case "$LAUNCHER" in *[[:space:]*?\[\]]*) die "the launcher path contains whitespace / glob characters" ;; esac
if [ "$NAME" = SH_ID_s18 ] && { [ "$IS_S1" != 1 ] || [ "$MASK" != 0 ] || [ "$CTL" != stand_net ] || [ "$CMD" != zero ] || [ "$RST" != armed ] || [ -n "$DB" ]; }; then
  die "SH_ID_s18 is the identity bridge: S1 ($CK), MASK 0, stand_net, zero, armed, no --deadband only; got $SCK mask $MASK $CTL $CMD $RST deadband '${DB:-0}'"
fi
if [ -n "$DB" ]; then
  [ "$NET" = 1 ] || die "--deadband acts on a stand network's residual: CONTROLLER must be stand_net / amask_old, got '$CTL'"
  case "$NAME" in SH_ID*) die "--deadband is never the identity bridge (an SH_ID* name), got '$NAME'" ;; esac
  [ "$IS_S1" = 0 ] || die "--deadband with S1 as the stand checkpoint (the identity bridge) is refused"
fi
case "$NAME" in SH_ID*) ;; *) [ "$IS_S1" = 0 ] || die "S1 as the stand checkpoint only under an SH_ID* head name, not '$NAME'" ;; esac
[ -f "$CK" ] || die "no S1 checkpoint $CK"
[ -f "$ST/run_meta.json" ] || die "no REFS env stamp $ST"
[ -f "$LAUNCHER" ] || die "no launcher $LAUNCHER"
[ -z "${P2_ENV_EXTRA:-}" ] || die "P2_ENV_EXTRA must be empty (D5: never STAND_OBS / an env mask in P2_ENV_EXTRA), got '$P2_ENV_EXTRA'"
env | grep -q '^P2R_' && die "P2R_* variables in the environment (the launcher refuses them)"
env | grep -qE '^(SH_|SN_STAND_)' && die "SH_* / SN_STAND_* variables in the environment (the launcher's fallbacks; the rows pass CLI flags only)"
for spec in "${ROWS[@]}"; do
  row=${spec%@*}; es=${spec##*@}
  case "$es" in ''|*[!0-9]*|0) die "row spec '$spec' must be ROW@ES" ;; esac
  manifest "$row" >/dev/null || die "no manifest known for row '$row'"
  [ -f "$(manifest "$row")" ] || die "manifest missing for $row"
  [ "$(row_command "$row")" = "$CMD" ] || die "row '$row' runs with --sh-command $(row_command "$row"), not $CMD"
done
EXTRA="--sh-controller $CTL --sh-command $CMD --sh-restore $RST"
[ "$NET" = 1 ] && EXTRA="$EXTRA --stand-ckpt $SCK --stand-mask $MASK"
[ -n "$DB" ] && EXTRA="$EXTRA --sh-deadband $DB"

[ "$DRY" = 1 ] || mkdir -p "$SHOUT"
say() { [ "$DRY" = 1 ] || echo "SHROWS $* $(date -Is)" >> "$LOG"; }
# run_arm.sh's drivers_idle sees evals only -> before every row wait while ANY Isaac python runs (CHARTER section 2:
# one Isaac process at a time; two Kit processes sharing the DerivedDataCache is the 2026-09-18 corruption / OOM
# scenario). ISAAC_PAT = sh_table_test.sh's list (trainers, table tests, video recorders, train / play, evals, drivers,
# any campaign-local sh_*.py); NONKIT_PAT exempts the Kit-free modes and the CPU-only campaign scripts. `ps -ww`: plain
# `ps -eo args` truncates to $COLUMNS and the anchored patterns would never match the long Isaac command lines. Never
# `pgrep -f` (a pattern in our own command line matches itself).
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|evaluation|rsl_rl/train|train|play|play_stream_video|demo|demo_crab_hex_student|record_categories|sn_record_video|sn_dagger|sn_table_test|sn_stand_net_eval_crab_hex_gait|p2r_mode_eval_crab_hex_gait|harvest_omni_rsi_bank|p2_smoke|zero_agent|verify_crab_[a-z_]+|debug_actuator_resolution|training_timeline_probe|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+)\.py'
NONKIT_PAT='(s[hn]_dagger\.py .*--(read|emit-env)([ =]|$))|(s[hn]_table_test\.py .*--(emit-facts|merge)([ =]|$))|(/sh_(facts|kin|core|table_core|eval_common|read|bridge_read|observe_read)\.py( |$))'
TRAINER_PAT='s[hn]_dagger\.py'
isaac_busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
trainer_busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -vE "$NONKIT_PAT" | grep -qE "$TRAINER_PAT"; }
say START "$NAME" controller="$CTL" command="$CMD" restore="$RST" stand_ckpt="$SCK" mask="$MASK" deadband="${DB:-0}" rows="${ROWS[*]}"
for spec in "${ROWS[@]}"; do
  row=${spec%@*}; es=${spec##*@}; man=$(manifest "$row")
  if [ -f "$OUT/STOP_QUEUE" ] || [ -f "$SHOUT/STOP_QUEUE" ]; then say STOP STOP_QUEUE present; exit 4; fi
  cmd=(env P2_GUARD_REF=REFH_S1_s18 P2_EVAL="$LAUNCHER" P2_ENV_EXTRA= EVAL_EXTRA_ARGS="$EXTRA"
       bash "$P2/scripts/run_arm.sh" --row-eval "$NAME" "$CK" "$ST" "$man" "$row" "$es")
  if [ "$DRY" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; continue; fi
  if isaac_busy; then
    if trainer_busy; then say WAIT a DAgger trainer running; else say WAIT another Isaac process running; fi
    while isaac_busy; do sleep 30; done
  fi
  "${cmd[@]}"; rc=$?
  say "$NAME" "$row" es"$es" rc="$rc"
  if [ "$rc" != 0 ]; then say STOP "$NAME" "$row" es"$es" rc="$rc"; exit 5; fi
done
say DONE "$NAME"

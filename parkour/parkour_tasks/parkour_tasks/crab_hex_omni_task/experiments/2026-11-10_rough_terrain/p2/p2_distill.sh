#!/usr/bin/env bash
# Rough-terrain P2 step 4 (p2/P2.md Design > Distill): the depth-student distill from c1, the rod-speed DR campaign's
# rsd_distill.sh re-pointed. Through W's own experiments/tools/pipeline/distill.sh (busy check, its own
# DerivedDataCache limit, every parent KRABBY_* unset, the runner-only keys stripped, systemd-run --user --scope
# MemoryMax=48G, timeout --signal=TERM). This wrapper adds only what distill.sh cannot know:
#
#   * PYTHONPATH=<W>/parkour/parkour_tasks:<W>/parkour -- distill.sh adds none, and without it scripts/ and
#     parkour_isaaclab resolve to the MAIN checkout through the editable .pth (a quiet pin violation);
#   * TP_DISTILL_OUT=<MAIN>/parkour/outputs/omni_plan/rough_terrain_p2/distill; every other TP_* / DRS_* / RSD_* / RTQ_* /
#     RT_* variable of the caller is unset (distill.sh's defaults apply);
#   * the env file envs/P2_distill.env = RSD's envs/p3b_dr.distill.env with KRABBY_FLAT_TERRAIN_GEOM=field -> site
#     (exactly once) + RT_train.env's KRABBY_SITE_MIX, KRABBY_SITE_LADDER and KRABBY_RT_FRICTION lines verbatim (plan
#     settled 6a). It is a STATIC file committed with the pre-registration (reviewable before any number exists; a
#     launch never writes into the campaign tree). Every launch RE-COMPOSES it from its two sources -- each source
#     checked by sha (p3b_dr.distill.env = 8ba683b4, RT_train.env = results/pin.json's frozen sha) -- and refuses
#     unless the result is byte-identical (cmp) to the static file; then its KRABBY_* set must equal RT_train.env's
#     KRABBY_* set key for key and value for value (RT_train.env's only other keys are process keys distill.sh sets
#     itself: OMNI_KIT_ACCEPT_EULA, TERM, PYTHONUNBUFFERED -- and PYTHONPATH, exported here). distill.sh then strips the
#     runner-only keys (STUDENT_RUNNER_KNOBS_REFUSED), exactly as it did for RSD's student;
#   * the teacher guard (replaces RSD's g1.json guard): sha256(--checkpoint) = c1's full sha AND CHARTER.md carries the
#     2026-11-11 17:25 decision "A" entry naming 6d1f29be / model_29500;
#   * the build guard: live code_sha / eval_sha = results/pin.json's (and results/p2/pin.json's once written), clean;
#   * the CHARTER 3.5 guard: < 30 GB free or the Kit DerivedDataCache > 78 GB -> stop (exit 4; a move-aside is the
#     user's call), read just before launch (after the --wait-idle busy wait);
#   * a crashed distill is never resumed under the same name: an existing $OUT/NAME.log, run dir *_NAME (W's or MAIN's
#     logs) or results/p2/runs/NAME.distill.json is refused. A relaunch under a new name replays bit for bit (same
#     seed): fix the cause first (plan item 4).
#
#   p2_distill.sh [--name RT_ST_s19] [--checkpoint C1_ABS] [--wait-idle] [--dry-run]
#     -> distill.sh --name NAME --checkpoint C1_ABS --env-file <CD>/envs/P2_distill.env --seed 19 --iterations 5000
#                   --num-envs 192 --task Isaac-Crab-Hex-Omni-Student-v0 --experiment crab_hex_omni_student
#                   [--wait-idle] [--dry-run]
# --dry-run prints every guard verdict (a guard that would refuse prints "# WOULD REFUSE"), the composed env, the
# exported variables and distill.sh's own --dry-run (the stripped keys, the child env, the systemd-run command);
# nothing is written and nothing launches.
#
# The +10-iteration checks: once $OUT/NAME.log shows iteration 29510, run
#   /home/nickmagus/krabby/isaac_venv/bin/python -I -B <CD>/p2/p2_distill_check.py check --name NAME --teacher C1_ABS
# (D1-D5 a miss -> exit 3 -> `p2_distill_check.py stop --name NAME`; D6 N/A at +10; D7 a cost line). The curriculum
# reads at +300 / +1000: <CD>/p2/p2_curriculum.py read --name NAME --rel 300 | 1000.
#
# Exit codes: distill.sh's (the trainer's rc; 3 busy; 5 its refusals), 4 guard, 5 a refusal here.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CD=$(cd "$HERE/.." && pwd)
W=$(cd "$CD/../../../../../.." && pwd)    # CD -> experiments -> crab_hex_omni_task -> parkour_tasks x2 -> parkour -> W
MAIN=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
CHECK_PY=$HERE/p2_distill_check.py
DISTILL_SH=$W/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/pipeline/distill.sh
BASE_ENV=$W/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-11-07_rod_speed_dr/envs/p3b_dr.distill.env
BASE_SHA=8ba683b4dad7ce22d9ddeb5bb91548ee918c173323b815de75040b90278771c7
RT_ENV=$CD/envs/RT_train.env
ENV_OUT=$CD/envs/P2_distill.env
RT_KEYS="KRABBY_SITE_MIX KRABBY_SITE_LADDER KRABBY_RT_FRICTION"
C1_SHA=6d1f29be8bf4504fd53fd20018dbab38ff60c6ccb32d7bd885f3ab15eaaed3c1
C1_DEFAULT=$W/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19/model_29500.pt
OUT=$MAIN/parkour/outputs/omni_plan/rough_terrain_p2/distill
RUNS=$CD/results/p2/runs
DDC=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache
DDC_MAX_GB=78
FREE_MIN_GB=30
SEED=19; ITERS=5000; NUM=192; TASK=Isaac-Crab-Hex-Omni-Student-v0; EXPN=crab_hex_omni_student

die() { echo "p2_distill.sh: $*" >&2; exit 5; }
REFUSALS=0
refuse() {  # a real launch dies; --dry-run reports and carries on
  if [ "$DRY" = 1 ]; then echo "# WOULD REFUSE: $*"; REFUSALS=$((REFUSALS + 1)); else die "$*"; fi
}
[ -f "$W/parkour/scripts/rsl_rl/train.py" ] || die "worktree root $W has no parkour/scripts/rsl_rl/train.py"
[ -f "$DISTILL_SH" ] || die "no $DISTILL_SH"
[ -f "$CHECK_PY" ] || die "no $CHECK_PY"
NAME=RT_ST_s19; CK=$C1_DEFAULT; WAIT=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME=${2:-}; shift 2 ;;
    --checkpoint) CK=${2:-}; shift 2 ;;
    --wait-idle) WAIT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    *) die "unknown option $1 (the distill settings are fixed by p2/P2.md)" ;;
  esac
done
case "$NAME" in RT_ST_*) ;; *) die "--name must start with RT_ST_ (P2's student names)" ;; esac
case "$NAME" in *[!A-Za-z0-9_]*) die "--name must be [A-Za-z0-9_]+" ;; esac
case "$CK" in /*) ;; *) die "--checkpoint must be absolute: '$CK'" ;; esac
case "$CK" in *[[:space:]*?\[\]]*) die "--checkpoint has whitespace / glob characters" ;; esac
[ -f "$CK" ] || die "no such checkpoint $CK"
case "$(basename "$CK")" in model_[0-9]*.pt) ;; *) die "--checkpoint must be a model_<iter>.pt" ;; esac
echo "# p2_distill.sh $NAME ($( [ "$DRY" = 1 ] && echo dry-run || echo launch )) at $(date -Is)"

# ---- teacher guard: c1 by full sha + the CHARTER 2026-11-11 17:25 decision "A" record
CK_SHA=$(sha256sum "$CK" | cut -d' ' -f1)
if [ "$CK_SHA" = "$C1_SHA" ]; then echo "# guard teacher: PASS sha256 $CK_SHA = c1 ($CK)"
else refuse "sha256(--checkpoint) $CK_SHA != c1 $C1_SHA"; fi
CH=$("$PY" -I -B -c 'import sys; sys.path.insert(0, sys.argv[1]); import p2_distill_check as C
e = C.charter_entry(); print(e.split("\n", 1)[0] if e else "")' "$HERE" 2>/dev/null) || CH=""
if [ -n "$CH" ]; then echo "# guard charter: PASS '$CH' (names 6d1f29be + model_29500)"
else refuse "CHARTER.md has no 2026-11-11 17:25 decision \"A\" entry naming 6d1f29be / model_29500"; fi

# ---- build guard: the RT pin (and results/p2/pin.json once written), clean code/eval tree
PIN=$("$PY" -I -B "$CHECK_PY" pin 2>&1); PIN_RC=$?
echo "# guard build: $PIN"
[ "$PIN_RC" = 0 ] || refuse "build != pin: $PIN"

# ---- the env file: re-compose from the two sha-checked sources, cmp to the static file, then the KRABBY_* set proof
[ -f "$BASE_ENV" ] || die "no $BASE_ENV"
[ -f "$RT_ENV" ] || die "no $RT_ENV"
[ -f "$ENV_OUT" ] || die "no $ENV_OUT (the static env file is committed with the pre-registration; restore it from git)"
B_SHA=$(sha256sum "$BASE_ENV" | cut -d' ' -f1)
[ "$B_SHA" = "$BASE_SHA" ] && echo "# guard env base: PASS $BASE_ENV sha $B_SHA" || refuse "$BASE_ENV sha $B_SHA != $BASE_SHA"
RT_PIN_SHA=$("$PY" -I -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["frozen_files"]["envs/RT_train.env"])' \
             "$CD/results/pin.json") || die "cannot read RT_train.env's frozen sha from results/pin.json"
R_SHA=$(sha256sum "$RT_ENV" | cut -d' ' -f1)
[ "$R_SHA" = "$RT_PIN_SHA" ] && echo "# guard env RT: PASS $RT_ENV sha $R_SHA = results/pin.json" \
  || refuse "$RT_ENV sha $R_SHA != results/pin.json frozen $RT_PIN_SHA"
[ "$(tail -c1 "$BASE_ENV" | od -An -c | tr -d ' ')" = '\n' ] || die "$BASE_ENV does not end in a newline"
[ "$(grep -cxE 'KRABBY_FLAT_TERRAIN_GEOM=field' "$BASE_ENV")" = 1 ] || die "$BASE_ENV: KRABBY_FLAT_TERRAIN_GEOM=field not exactly once"
[ "$(grep -cE '^KRABBY_FLAT_TERRAIN_GEOM=' "$BASE_ENV")" = 1 ] || die "$BASE_ENV: KRABBY_FLAT_TERRAIN_GEOM set more than once"
for k in $RT_KEYS; do
  grep -qE "^$k=" "$BASE_ENV" && die "$BASE_ENV already carries $k"
  [ "$(grep -cE "^$k=" "$RT_ENV")" = 1 ] || die "$RT_ENV: $k not exactly once"
done
grep -qE '^KRABBY_STRESS_' "$BASE_ENV" "$RT_ENV" && die "a KRABBY_STRESS_* key in the env sources (CHARTER 2)"
compose() {
  sed 's/^KRABBY_FLAT_TERRAIN_GEOM=field$/KRABBY_FLAT_TERRAIN_GEOM=site/' "$BASE_ENV"
  for k in $RT_KEYS; do grep -E "^$k=" "$RT_ENV"; done
}
if cmp -s <(compose) "$ENV_OUT"; then
  echo "# guard env file: PASS $ENV_OUT ($(sha256sum "$ENV_OUT" | cut -c1-12)) = p3b_dr.distill.env [GEOM field->site] + RT_train.env's $RT_KEYS, byte-identical"
else
  refuse "$ENV_OUT differs from the composition (p3b_dr.distill.env [GEOM field->site] + RT_train.env's $RT_KEYS): $(diff <(compose) "$ENV_OUT" | head -5 | tr '\n' ' ')"
fi
[ "$(grep -cxE 'KRABBY_FLAT_TERRAIN_GEOM=site' "$ENV_OUT")" = 1 ] || refuse "$ENV_OUT: KRABBY_FLAT_TERRAIN_GEOM=site not exactly once"
SETDIFF=$(diff <(grep -E '^KRABBY_' "$RT_ENV" | sort) <(grep -E '^KRABBY_' "$ENV_OUT" | sort))
if [ -z "$SETDIFF" ]; then
  echo "# guard env set: PASS KRABBY_* of P2_distill.env == KRABBY_* of RT_train.env ($(grep -c '^KRABBY_' "$ENV_OUT") keys, key and value);" \
       "RT_train.env's other keys: $(grep -vE '^(KRABBY_|#|$)' "$RT_ENV" | cut -d= -f1 | tr '\n' ' ')(process keys, not in the env file)"
else
  refuse "KRABBY_* sets differ: $(echo "$SETDIFF" | tr '\n' ' ')"
fi

# ---- never resume a crashed distill under the same name
taken=""
[ -e "$OUT/$NAME.log" ] && taken="$taken $OUT/$NAME.log"
[ -e "$RUNS/$NAME.distill.json" ] && taken="$taken $RUNS/$NAME.distill.json"
for r in "$W" "$MAIN"; do
  for d in "$r"/parkour/logs/rsl_rl/*/*_"$NAME"; do
    [ -d "$d" ] && [[ "$(basename "$d")" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_${NAME}$ ]] && taken="$taken $d"
  done
done
if [ -n "$taken" ]; then refuse "name $NAME is taken:$taken (pick a new name)"
else echo "# guard name: PASS $NAME unused (no $OUT/$NAME.log, no run dir *_$NAME, no $RUNS/$NAME.distill.json)"; fi

# ---- busy wait (distill.sh's own pattern), then the disk / DerivedDataCache guard just before the launch
ISAAC_PAT=$(sed -n "s/^ISAAC_PAT='\(.*\)'\$/\1/p" "$DISTILL_SH")
NONKIT_PAT=$(sed -n "s/^NONKIT_PAT='\(.*\)'\$/\1/p" "$DISTILL_SH")
[ -n "$ISAAC_PAT" ] && [ -n "$NONKIT_PAT" ] || die "cannot read ISAAC_PAT / NONKIT_PAT from $DISTILL_SH"
busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
if [ "$DRY" != 1 ] && [ "$WAIT" = 1 ]; then while busy; do sleep 30; done; fi
if busy; then BUSY="busy (another Isaac python runs)"; else BUSY="idle"; fi
echo "# guard busy: $BUSY$( [ "$WAIT" = 1 ] && echo ' (--wait-idle)' || echo ' (distill.sh exits 3 if busy)')"
FREE_GB=$(df --output=avail -BG "$MAIN" | tail -1 | tr -dc '0-9')
DDC_GB=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); DDC_GB=${DDC_GB:-0}
if [ "${FREE_GB:-0}" -lt "$FREE_MIN_GB" ] || [ "$DDC_GB" -gt "$DDC_MAX_GB" ]; then
  echo "# guard disk: FAIL free ${FREE_GB} GB (min $FREE_MIN_GB), Kit DerivedDataCache ${DDC_GB} GB (max $DDC_MAX_GB)"
  echo "P2 DISTILL FAIL $NAME status=guard free=${FREE_GB}GB ddc=${DDC_GB}GB at=$(date -Is)" >&2
  [ "$DRY" = 1 ] || exit 4
  REFUSALS=$((REFUSALS + 1))
else
  echo "# guard disk: PASS free ${FREE_GB} GB (min $FREE_MIN_GB), Kit DerivedDataCache ${DDC_GB} GB (max $DDC_MAX_GB)"
fi

# ---- environment: the worktree PYTHONPATH, P2's output tree, no stray campaign variables
for v in $(env | grep -oE '^(TP|DRS|RSD|RTQ|RT|KIT)_[A-Za-z0-9_]+' | sort -u); do unset "$v"; done
export PYTHONPATH="$W/parkour/parkour_tasks:$W/parkour"
export TP_DISTILL_OUT="$OUT"

ARGS=(--name "$NAME" --checkpoint "$CK" --env-file "$ENV_OUT" --seed "$SEED" --iterations "$ITERS" --num-envs "$NUM"
      --task "$TASK" --experiment "$EXPN")
[ "$WAIT" = 1 ] && ARGS+=(--wait-idle)
CHECK_CMD="$PY -I -B $CHECK_PY check --name $NAME --teacher $CK"
if [ "$DRY" = 1 ]; then
  echo "# guards: $REFUSALS would refuse"
  echo "# env file $ENV_OUT: $(grep -c '^KRABBY_' "$ENV_OUT") KRABBY_* lines (distill.sh strips the runner-only ones below)"
  echo "# PYTHONPATH=$PYTHONPATH"
  echo "# TP_DISTILL_OUT=$TP_DISTILL_OUT"
  echo "# record (launch only): $PY -I -B $CHECK_PY record --name $NAME --teacher $CK --env-file $ENV_OUT -> $RUNS/$NAME.distill.json"
  echo "# exec bash $DISTILL_SH ${ARGS[*]} --dry-run"
  echo "# then, at +10 iterations (29510): $CHECK_CMD"
  echo "# curriculum reads: $PY -I -B $HERE/p2_curriculum.py read --name $NAME --rel 300 (and --rel 1000)"
  echo "# ---- distill.sh --dry-run:"
  exec bash "$DISTILL_SH" "${ARGS[@]}" --dry-run
fi
"$PY" -I -B "$CHECK_PY" record --name "$NAME" --teacher "$CK" --env-file "$ENV_OUT" \
    --launcher-argv "$0" --name "$NAME" --checkpoint "$CK" || die "cannot write the distill record"
echo "P2 DISTILL START $NAME ck=$CK sha=${CK_SHA:0:8} env=$ENV_OUT out=$OUT at=$(date -Is)"
echo "# at +10 iterations (29510): $CHECK_CMD"
exec bash "$DISTILL_SH" "${ARGS[@]}"

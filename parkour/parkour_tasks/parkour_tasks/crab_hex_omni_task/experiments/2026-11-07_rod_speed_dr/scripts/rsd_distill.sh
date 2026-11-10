#!/usr/bin/env bash
# Rod-speed DR campaign (CHARTER 3.4, G1 PASS only): the depth-student distill from TDR, through W's own
# experiments/tools/pipeline/distill.sh (busy check, its own DerivedDataCache limit, every parent KRABBY_* unset,
# systemd-run --user --scope MemoryMax=48G, timeout --signal=TERM). This wrapper only adds what distill.sh cannot know:
#
#   * PYTHONPATH=<W>/parkour/parkour_tasks:<W>/parkour -- distill.sh adds none, and without it scripts/ and
#     parkour_isaaclab resolve to the MAIN checkout through the editable .pth (the worktree hazard);
#   * TP_DISTILL_OUT=<MAIN>/parkour/outputs/omni_plan/rod_speed_dr/distill (the campaign's own output tree); every
#     other TP_* / DRS_* / RSD_* variable of the caller is unset (distill.sh's defaults apply);
#   * the campaign's distill env file envs/p3b_dr.distill.env = F19/envs/p3b.distill.env + KRABBY_DR_ROD_SPEED=0.6:1.35.
#     The knob must be IN the env file: distill.sh unsets every parent KRABBY_*. Written byte-identically every time
#     (an existing file that differs is refused);
#   * the CHARTER 3.5 guard: < 30 GB free or the Kit DerivedDataCache > 78 GB -> stop (exit 4; a prune is the user's
#     call); a busy wait before it with --wait-idle (distill.sh's own pattern), so the guard is read just before launch;
#   * a crashed distill is never resumed under the same name: an existing $OUT/NAME.log or run dir *_NAME is refused.
#
#   * CHARTER 3.4 "G1 PASS only" and "teacher = TDR": --teacher-policy TDR (the registered name) must be the TDR of
#     results/g1.json, that read's verdict PASS, and sha256(--checkpoint) = the registry's = g1.json's (a real launch
#     refuses otherwise; --dry-run reports WOULD REFUSE).
#
#   rsd_distill.sh --name NAME --checkpoint TDR_ABS --teacher-policy TDR [--wait-idle] [--dry-run]
#     -> distill.sh --name NAME --checkpoint TDR_ABS --env-file <CD>/envs/p3b_dr.distill.env --seed 19
#                   --iterations 5000 --num-envs 192 --task Isaac-Crab-Hex-Omni-Student-v0
#                   --experiment crab_hex_omni_student [--wait-idle] [--dry-run]
#   TDR = the G1 teacher: <W>/parkour/logs/rsl_rl/crab_hex_omni_teacher/<ts>_<TDR name>/model_28991.pt
#         (model_31990 after EXTEND -- the same value goes to --checkpoint here and to `check --teacher` below).
#
# The +10-iteration checks (CHARTER 3.4; a miss means kill). Once $OUT/NAME.log shows iteration iter+10, run
#   /home/nickmagus/krabby/isaac_venv/bin/python -I -B <CD>/scripts/rsd_distill_check.py check --name NAME \
#       --teacher TDR_ABS [--live]
# which reads: D1 the [KNOB] / [omni-knobs] multiset = T4F19_st_a2's + exactly the DR [KNOB] line; D2 the [DR] line =
# the predicted first draw (192 envs, seed 19, U(0.6, 1.35), this worktree's module); D3 teacher sha = TDR (the
# "Loading model checkpoint from:" path and params/agent.yaml load_checkpoint); D4 first iteration = TDR's, +10
# reached; D5 the live-watch tail read from TDR's run dir (the student live rules' T* tail); D6 with --live, the
# student live rules. Exit 3 = a miss -> `rsd_distill_check.py stop --name NAME` (SIGTERM the trainer pid, 120 s,
# SIGTERM, 120 s, SIGKILL last resort). `rsd_distill_check.py reference` runs the same checks on Task 4's
# T4F19_st_a2 (its params, log and T*'s run dir; no knob) -- the helper's self-test.
#
# Exit codes: distill.sh's (the trainer's rc; 3 busy; 5 its refusals), 4 guard, 5 a refusal here.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CD=$(cd "$HERE/.." && pwd)
W=$(cd "$CD/../../../../../.." && pwd)    # CD -> experiments -> crab_hex_omni_task -> parkour_tasks x2 -> parkour -> W
MAIN=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
DISTILL_SH=$W/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/pipeline/distill.sh
BASE_ENV=$W/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/F19/envs/p3b.distill.env
DR_VAR=KRABBY_DR_ROD_SPEED
DR_SPEC=0.6:1.35
ENV_OUT=$CD/envs/p3b_dr.distill.env
OUT=$MAIN/parkour/outputs/omni_plan/rod_speed_dr/distill
DDC=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache
DDC_MAX_GB=78
FREE_MIN_GB=30
SEED=19; ITERS=5000; NUM=192; TASK=Isaac-Crab-Hex-Omni-Student-v0; EXPN=crab_hex_omni_student

die() { echo "rsd_distill.sh: $*" >&2; exit 5; }
[ -f "$W/parkour/scripts/rsl_rl/train.py" ] || die "worktree root $W has no parkour/scripts/rsl_rl/train.py"
[ -f "$DISTILL_SH" ] || die "no $DISTILL_SH"
NAME=""; CK=""; TEACHER=""; WAIT=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME=${2:-}; shift 2 ;;
    --checkpoint) CK=${2:-}; shift 2 ;;
    --teacher-policy) TEACHER=${2:-}; shift 2 ;;
    --wait-idle) WAIT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    *) die "unknown option $1 (the distill settings are fixed by CHARTER 3.4)" ;;
  esac
done
case "$NAME" in ''|*[!A-Za-z0-9_]*) die "--name must be [A-Za-z0-9_]+" ;; DRS_*|DRST_*) die "--name uses a sweep prefix" ;; esac
case "$CK" in /*) ;; *) die "--checkpoint must be absolute: '$CK'" ;; esac
case "$CK" in *[[:space:]*?\[\]]*) die "--checkpoint has whitespace / glob characters" ;; esac
[ -f "$CK" ] || die "no such checkpoint $CK"
case "$(basename "$CK")" in model_[0-9]*.pt) ;; *) die "--checkpoint must be a model_<iter>.pt" ;; esac

# ---- CHARTER 3.4: the student stage runs on a G1 PASS only, from that read's TDR
G1_WHY=$("$PY" -I -B -c '
import hashlib, json, sys
res, name, ck = sys.argv[1:4]
def load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return {}
g1, reg = load(res + "/g1.json"), (load(res + "/policies.json").get("policies") or {})
tdr = (g1.get("policies") or {}).get("tdr") or {}
v, tname, tsha = g1.get("verdict"), tdr.get("name"), str(tdr.get("sha256"))
rsha = (reg.get(name) or {}).get("sha256")
h = hashlib.sha256(open(ck, "rb").read()).hexdigest()
why = []
if not name:
    why.append("--teacher-policy is required")
if v != "PASS":
    why.append(f"results/g1.json verdict {v!r} (G1 PASS only)")
if name and tname != name:
    why.append(f"results/g1.json was read for TDR {tname!r}, not {name!r}")
if name and rsha != h:
    why.append(f"sha256(--checkpoint) {h[:8]} != the registry policy {name!r}")
if tsha != h:
    why.append(f"sha256(--checkpoint) {h[:8]} != results/g1.json TDR {tsha[:8]}")
print("; ".join(why))
' "$CD/results" "$TEACHER" "$CK") || die "cannot check the G1 record"
if [ -n "$G1_WHY" ]; then
  [ "$DRY" = 1 ] && echo "# WOULD REFUSE: $G1_WHY" || die "$G1_WHY"
fi

# ---- the campaign's distill env file (the knob inside it)
[ -f "$BASE_ENV" ] || die "no $BASE_ENV"
[ "$(tail -c1 "$BASE_ENV" | od -An -c | tr -d ' ')" = '\n' ] || die "$BASE_ENV does not end in a newline"
grep -qE "^${DR_VAR}=" "$BASE_ENV" && die "$BASE_ENV already carries $DR_VAR"
grep -qE '^KRABBY_STRESS_' "$BASE_ENV" && die "$BASE_ENV carries a KRABBY_STRESS_* key (CHARTER 2)"
mkdir -p "$CD/envs"
TMP=$(mktemp "$CD/envs/.p3b_dr.distill.env.XXXXXX") || die "mktemp failed"
{ cat "$BASE_ENV"; printf '%s=%s\n' "$DR_VAR" "$DR_SPEC"; } > "$TMP"
if [ -f "$ENV_OUT" ]; then
  cmp -s "$TMP" "$ENV_OUT" || { rm -f "$TMP"; die "$ENV_OUT exists and differs from p3b.distill.env + $DR_VAR=$DR_SPEC"; }
  rm -f "$TMP"
else
  chmod 644 "$TMP" && mv "$TMP" "$ENV_OUT"
fi
[ "$(grep -cE "^${DR_VAR}=${DR_SPEC}\$" "$ENV_OUT")" = 1 ] || die "$ENV_OUT: $DR_VAR=$DR_SPEC not exactly once"

# ---- never resume a crashed distill under the same name
taken=""
[ -e "$OUT/$NAME.log" ] && taken="$taken $OUT/$NAME.log"
for r in "$W" "$MAIN"; do
  for d in "$r"/parkour/logs/rsl_rl/*/*_"$NAME"; do
    [ -d "$d" ] && [[ "$(basename "$d")" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_${NAME}$ ]] && taken="$taken $d"
  done
done
if [ -n "$taken" ]; then
  [ "$DRY" = 1 ] && echo "# WOULD REFUSE: name $NAME is taken:$taken" || die "name $NAME is taken:$taken (pick a new name)"
fi

# ---- busy wait (distill.sh's own pattern), then the guard just before the launch
ISAAC_PAT=$(sed -n "s/^ISAAC_PAT='\(.*\)'\$/\1/p" "$DISTILL_SH")
NONKIT_PAT=$(sed -n "s/^NONKIT_PAT='\(.*\)'\$/\1/p" "$DISTILL_SH")
[ -n "$ISAAC_PAT" ] && [ -n "$NONKIT_PAT" ] || die "cannot read ISAAC_PAT / NONKIT_PAT from $DISTILL_SH"
busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
if [ "$DRY" != 1 ] && [ "$WAIT" = 1 ]; then while busy; do sleep 30; done; fi
FREE_GB=$(df --output=avail -BG "$MAIN" | tail -1 | tr -dc '0-9')
DDC_GB=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); DDC_GB=${DDC_GB:-0}
echo "# guard: free ${FREE_GB} GB (min $FREE_MIN_GB), Kit DerivedDataCache ${DDC_GB} GB (max $DDC_MAX_GB)"
if [ "${FREE_GB:-0}" -lt "$FREE_MIN_GB" ] || [ "$DDC_GB" -gt "$DDC_MAX_GB" ]; then
  echo "RSD DISTILL FAIL $NAME status=guard free=${FREE_GB}GB ddc=${DDC_GB}GB at=$(date -Is)" >&2
  [ "$DRY" = 1 ] || exit 4
fi

# ---- environment: the worktree PYTHONPATH, the campaign's output tree, no stray campaign variables
for v in $(env | grep -oE '^(TP|DRS|RSD)_[A-Za-z0-9_]+' | sort -u); do unset "$v"; done
export PYTHONPATH="$W/parkour/parkour_tasks:$W/parkour"
export TP_DISTILL_OUT="$OUT"

ARGS=(--name "$NAME" --checkpoint "$CK" --env-file "$ENV_OUT" --seed "$SEED" --iterations "$ITERS" --num-envs "$NUM"
      --task "$TASK" --experiment "$EXPN")
[ "$WAIT" = 1 ] && ARGS+=(--wait-idle)
if [ "$DRY" = 1 ]; then
  echo "# env file $ENV_OUT ($(sha256sum "$ENV_OUT" | cut -c1-12)): $(grep -c '^KRABBY_' "$ENV_OUT") KRABBY_* lines"
  echo "# PYTHONPATH=$PYTHONPATH"
  echo "# TP_DISTILL_OUT=$TP_DISTILL_OUT"
  echo "# exec bash $DISTILL_SH ${ARGS[*]} --dry-run"
  echo "# then, at +10 iterations: $PY -I -B $HERE/rsd_distill_check.py check --name $NAME --teacher $CK --live"
  exec bash "$DISTILL_SH" "${ARGS[@]}" --dry-run
fi
"$PY" -I -B "$HERE/rsd_distill_check.py" record --name "$NAME" --teacher "$CK" --env-file "$ENV_OUT" \
    --launcher-argv "$0" --name "$NAME" --checkpoint "$CK" || die "cannot write the distill record"
echo "RSD DISTILL START $NAME ck=$CK env=$ENV_OUT out=$OUT at=$(date -Is)"
echo "# at +10 iterations: $PY -I -B $HERE/rsd_distill_check.py check --name $NAME --teacher $CK --live"
exec bash "$DISTILL_SH" "${ARGS[@]}"

#!/usr/bin/env bash
# Task 4 pipeline: launch ONE omni depth-student distillation (train.py on Isaac-Crab-Hex-Omni-Student-v0, resuming
# T*'s checkpoint) on the GPU. The SH/sh_dagger.sh launcher pattern: busy check, Kit DerivedDataCache guard, every
# KRABBY_* of the parent unset, a memory-capped `systemd-run --user --scope`, `timeout --signal=TERM` (the wall bound:
# one SIGTERM to the trainer). A running distill is stopped by the queue's pipeline.stop_trainer (CHARTER 2026-10-18):
# SIGTERM to the train.py pid ITSELF -- never the scope: a scope-wide signal also reaches `timeout`, which re-forwards
# it (the C7 hang) -- 120 s, SIGTERM again, 120 s, and SIGKILL only as the last resort (recorded + notified).
#
#   distill.sh --name NAME --checkpoint T*_CKPT --env-file ENV --seed S --iterations N [--num-envs 192]
#              [--task Isaac-Crab-Hex-Omni-Student-v0] [--experiment crab_hex_omni_student] [--wait-idle]
#              [--dry-run] [-- HYDRA_OVERRIDES...]   (e.g. -- agent.depth_encoder.zero_depth=true)
#     ENV    KEY=VALUE lines: T*'s training-process env (gen's envs/<stage>.distill.env: the preset, the plant and
#            T*'s overlay incl. the payload knobs). Every key of omni_student_spec.STUDENT_RUNNER_KNOBS_REFUSED is
#            STRIPPED (the student runner refuses them: no PPO, no symmetry loss, no entropy -- DESIGN stage S1); the
#            stripped keys are printed and written to $OUT/NAME.stripped.
# Output: the trainer's run dir parkour/logs/rsl_rl/<experiment>/<date>_NAME (train.py --run_name NAME); the log
# $OUT/NAME.log; the scope unit name in $OUT/NAME.unit (OUT = parkour/outputs/omni_plan/task4/distill, or TP_DISTILL_OUT).
# Sentinels "TPDISTILL START|END NAME ..." go to $OUT/distill.log.
# Exit codes: the trainer's rc (124 = the TP_DISTILL_TIMEOUT bound, default 172800 s), 3 busy (without --wait-idle),
# 5 usage / refusal (bad NAME, a non-absolute path, a refused knob that survived the strip, DerivedDataCache too big).
set -u
R=/home/nickmagus/krabby/krabby-research
PY=${TP_PY:-/home/nickmagus/krabby/isaac_venv/bin/python}
TRAIN=$R/parkour/scripts/rsl_rl/train.py
SPEC_PY=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/config/crab_hex/omni_student_spec.py
OUT=${TP_DISTILL_OUT:-$R/parkour/outputs/omni_plan/task4/distill}
MEM=${TP_MEMORY_MAX:-48G}
DDC=${TP_DDC:-/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache}
DDC_MAX_GB=${TP_DDC_MAX_GB:-80}
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+|bc_dump_cfgs|t4_dump_cfgs)\.py'
NONKIT_PAT='(s[hn]_dagger\.py .*--(read|emit-env)([ =]|$))|(s[hn]_table_test\.py .*--(emit-facts|merge)([ =]|$))|(/s[hn]_(facts|kin|core|table_core|eval_common|read|bridge_read|observe_read|calib|common)\.py( |$))'

die() { echo "distill.sh: $*" >&2; exit 5; }
NAME=""; CK=""; ENVF=""; SEED=""; ITERS=""; NUM=192; TASK=Isaac-Crab-Hex-Omni-Student-v0; EXPN=crab_hex_omni_student
WAIT=0; DRY=0; HYDRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME=$2; shift 2 ;;
    --checkpoint) CK=$2; shift 2 ;;
    --env-file) ENVF=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --iterations) ITERS=$2; shift 2 ;;
    --num-envs) NUM=$2; shift 2 ;;
    --task) TASK=$2; shift 2 ;;
    --experiment) EXPN=$2; shift 2 ;;
    --wait-idle) WAIT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --) shift; HYDRA=("$@"); break ;;
    *) die "unknown option $1" ;;
  esac
done
case "$NAME" in ''|*[!A-Za-z0-9_]*) die "--name must be [A-Za-z0-9_]+" ;; esac
for v in SEED ITERS NUM; do case "${!v}" in ''|*[!0-9]*) die "--seed / --iterations / --num-envs must be positive integers ($v='${!v}')" ;; esac; done
for p in "$CK" "$ENVF"; do
  case "$p" in /*) ;; *) die "path must be absolute: '$p'" ;; esac
  case "$p" in *[[:space:]*?\[\]]*) die "path has whitespace / glob characters: $p" ;; esac
  [ -f "$p" ] || die "no such file $p"
done
for h in "${HYDRA[@]}"; do case "$h" in *[[:space:]]*) die "hydra override with whitespace: '$h'" ;; esac; done

REFUSED=$(python3 - "$SPEC_PY" <<'PY'
import ast, sys
src = open(sys.argv[1]).read()
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "STUDENT_RUNNER_KNOBS_REFUSED" for t in node.targets):
        print(" ".join(ast.literal_eval(node.value)))
        break
PY
) || die "cannot read STUDENT_RUNNER_KNOBS_REFUSED from $SPEC_PY"
[ -n "$REFUSED" ] || die "STUDENT_RUNNER_KNOBS_REFUSED is empty in $SPEC_PY"
envs=$(grep -E '^KRABBY_[A-Z0-9_]+=' "$ENVF" | tr '\n' ' ')
stripped=""
for k in $REFUSED; do
  if printf '%s\n' $envs | grep -q "^$k="; then stripped="$stripped $k"; fi
  envs=$(printf '%s\n' $envs | grep -v "^$k=" | tr '\n' ' ')
done
for k in $REFUSED; do printf '%s\n' $envs | grep -q "^$k=" && die "refused knob $k survived the strip"; done
printf '%s\n' $envs | grep -q '^KRABBY_OMNI_PHASE=' || die "the env file sets no KRABBY_OMNI_PHASE (the student env is composed from T*'s)"

busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }
if [ "$DRY" != 1 ] && busy; then
  if [ "$WAIT" = 1 ]; then while busy; do sleep 30; done
  else echo "distill.sh: another Isaac process is running (one at a time); use --wait-idle" >&2; exit 3; fi
fi
DDC_GB=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); DDC_GB=${DDC_GB:-0}
[ "$DDC_GB" -le "$DDC_MAX_GB" ] || die "Kit DerivedDataCache is ${DDC_GB} GB (> ${DDC_MAX_GB}): quarantine it first"
TMO=${TP_DISTILL_TIMEOUT:-172800}
case "$TMO" in ''|*[!0-9]*|0) die "TP_DISTILL_TIMEOUT must be a positive integer" ;; esac

unset_args=$(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')
UNIT="tp-distill-${NAME}-$(date +%s)"
CMD=(systemd-run --user --scope -q "--unit=$UNIT" -p "MemoryMax=$MEM" -p MemorySwapMax=0
     timeout --signal=TERM "$TMO" "$PY" "$TRAIN" --task "$TASK" --headless --num_envs "$NUM" --seed "$SEED"
     --max_iterations "$ITERS" --resume --checkpoint "$CK" --run_name "$NAME" "${HYDRA[@]}")
if [ "$DRY" = 1 ]; then
  echo "# stripped:${stripped:- none}"
  printf '%s\n' $envs
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
mkdir -p "$OUT"
echo "$UNIT" > "$OUT/$NAME.unit"
echo "${stripped# }" > "$OUT/$NAME.stripped"
echo "[$(date -Is)] TPDISTILL START $NAME unit=$UNIT.scope ck=$CK stripped=${stripped# }" >> "$OUT/distill.log"
# the driver's own training launch (run_phases.train: cwd parkour/, the parent env minus every KRABBY_*, EULA, TERM,
# PYTHONUNBUFFERED) -- no PYTHONPATH of its own
( cd "$R/parkour" && env $unset_args OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 $envs "${CMD[@]}" ) \
    >> "$OUT/$NAME.log" 2>&1 < /dev/null
rc=$?
echo "[$(date -Is)] TPDISTILL END $NAME rc=$rc" >> "$OUT/distill.log"
exit "$rc"

#!/bin/bash
# Rod-speed DR campaign, step 4 / K1: the KNOB-OFF and KNOB-ON proofs on the cfg dumps (Kit, no simulation stepped).
# A generalisation of 2026-11-06_dr_stress_sweep/scripts/dump_check.sh: five cases instead of two, and the knob-ON checks.
#
# For each root (REF_ROOT, the main checkout before the knob, and THIS worktree) the root's OWN
# tests/integration/crab_hex_phase_cfg_dump.py builds the cfgs of one task under EXACTLY a case's env (every parent
# KRABBY_* removed, then the env file's KRABBY_* keys, empty values kept; `$REPO` in a generated env file resolved
# against the root being dumped), PYTHONPATH=<root>/parkour/parkour_tasks:<root>/parkour, in a memory-capped
# `systemd-run --user --scope`, `timeout --signal=TERM 900` (SIGTERM only), ONE Kit process at a time (it waits while
# any other Isaac python runs).
#
#   cases (env files = 2026-10-16_task4_pipeline/F19/envs/):
#     C1 Isaac-Crab-Hex-Omni-Teacher-v0      x p3b.distill.env
#     C2 Isaac-Crab-Hex-Omni-Teacher-v0      x p3b.train.env
#     C3 Isaac-Crab-Hex-Omni-Student-v0      x p3b.distill.env minus omni_student_spec.STUDENT_RUNNER_KNOBS_REFUSED
#                                              (read by AST, as distill.sh reads it)
#     C4 Isaac-Crab-Hex-Omni-Student-Play-v0 x p3b.env
#     C5 Isaac-Crab-Hex-Omni-Student-Play-v0 x p3b.field_light.env
#   knob OFF (all five): the two dumps are byte-identical after each dump's root path is replaced by <ROOT>.
#   knob ON  (this worktree only; C1 and C3, plus KRABBY_DR_ROD_SPEED=0.6:1.35:12000): a structural JSON diff against
#             the same case's knob-off dump touches exactly actions.joint_pos.rod_speed_dr and the environ key.
#   refusal  (this worktree only; C1 with the DR knob AND KRABBY_STRESS_ROD_SPEED=0.7): the dump exits with the
#             exclusivity error.
#
#   rsd_dump_check.sh REF_ROOT OUT_DIR
# Prints one "RSD_DUMP PASS|FAIL <check> -- <detail>" line per check, then "RSD_DUMP END pass=<n> fail=<n>"; exits 1
# on any FAIL. 13 Kit processes (5 x 2 knob-off + 2 knob-on + 1 refusal), about 1-2 minutes each.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(cd "$HERE/../../../../../../.." && pwd)
[ $# -eq 2 ] || { echo "usage: rsd_dump_check.sh REF_ROOT OUT_DIR" >&2; exit 2; }
REF=$(cd "$1" 2>/dev/null && pwd) || { echo "RSD_DUMP FAIL setup -- no such REF_ROOT $1" >&2; exit 2; }
OUT=$2
PY=/home/nickmagus/krabby/isaac_venv/bin/python
CMP=("$PY" -B -I "$HERE/rsd_dump_compare.py")
EX=parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task
ENVS=$R/$EX/experiments/2026-10-16_task4_pipeline/F19/envs
SPEC_REL=$EX/config/crab_hex/omni_student_spec.py
CAP=(systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0)
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|stress_smoke|rsd_smoke|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+|bc_dump_cfgs|t4_dump_cfgs)\.py'
DR_ON=0.6:1.35:12000

declare -A TASK=( [C1]=Isaac-Crab-Hex-Omni-Teacher-v0 [C2]=Isaac-Crab-Hex-Omni-Teacher-v0
                  [C3]=Isaac-Crab-Hex-Omni-Student-v0 [C4]=Isaac-Crab-Hex-Omni-Student-Play-v0
                  [C5]=Isaac-Crab-Hex-Omni-Student-Play-v0 )
declare -A ENVF=( [C1]=$ENVS/p3b.distill.env [C2]=$ENVS/p3b.train.env [C3]=$ENVS/p3b.distill.env
                  [C4]=$ENVS/p3b.env [C5]=$ENVS/p3b.field_light.env )
CASES=(C1 C2 C3 C4 C5)

npass=0; nfail=0
check() {  # PASS|FAIL name detail
  local verdict=$1 name=$2 detail=${3:-}
  echo "RSD_DUMP $verdict $name -- $detail"
  if [ "$verdict" = PASS ]; then npass=$((npass + 1)); else nfail=$((nfail + 1)); fi
}
finish() { echo "RSD_DUMP END pass=$npass fail=$nfail"; [ "$nfail" -eq 0 ]; exit $?; }

# ---- setup (no Kit)
mkdir -p "$OUT" || { echo "RSD_DUMP FAIL setup -- cannot create $OUT" >&2; exit 2; }
OUT=$(cd "$OUT" && pwd)
[ "$REF" != "$R" ] || { check FAIL setup "REF_ROOT is this checkout"; finish; }
for root in "$REF" "$R"; do
  [ -f "$root/tests/integration/crab_hex_phase_cfg_dump.py" ] || { check FAIL setup "no cfg dump script under $root"; finish; }
  [ -f "$root/$SPEC_REL" ] || { check FAIL setup "no omni_student_spec.py under $root"; finish; }
done
for c in "${CASES[@]}"; do [ -f "${ENVF[$c]}" ] || { check FAIL setup "no env file ${ENVF[$c]}"; finish; }; done
if [ -e "$REF/$EX/config/crab_hex/dr_knobs.py" ] || grep -q rod_speed_dr "$REF/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp/parkour_actions.py" 2>/dev/null; then
  check FAIL setup "REF_ROOT already carries the DR knob: it is not a pre-knob reference"; finish
fi
REFUSED=$("${CMP[@]}" refused "$R/$SPEC_REL") || { check FAIL setup "cannot read STUDENT_RUNNER_KNOBS_REFUSED"; finish; }
[ -n "$REFUSED" ] || { check FAIL setup "STUDENT_RUNNER_KNOBS_REFUSED is empty"; finish; }
REFUSED_REF=$("${CMP[@]}" refused "$REF/$SPEC_REL" 2>/dev/null)
if [ "$REFUSED" = "$REFUSED_REF" ]; then
  check PASS "C3 refused-knob list identical in both roots" "$(echo "$REFUSED" | wc -w) keys"
else
  check FAIL "C3 refused-knob list identical in both roots" "this worktree: $REFUSED | reference: $REFUSED_REF"
fi
echo "RSD_DUMP INFO ref=$REF ($(git -C "$REF" rev-parse --short HEAD 2>/dev/null)) new=$R ($(git -C "$R" rev-parse --short HEAD 2>/dev/null)) out=$OUT"

compose_env() {  # ROOT ENV_FILE [STRIP]  -> KEY=VALUE lines ($REPO -> ROOT; STRIP=1 drops the student-refused keys)
  local root=$1 src=$2 strip=${3:-0} line key k skip
  while IFS= read -r line; do
    key=${line%%=*}
    if [ "$strip" = 1 ]; then
      skip=0; for k in $REFUSED; do [ "$key" = "$k" ] && skip=1; done
      [ "$skip" = 1 ] && continue
    fi
    echo "${line//\$REPO/$root}"
  done < <(grep -E '^KRABBY_[A-Z0-9_]+=' "$src")
}

RC=0
dump() {  # ROOT CASE_ID OUT_JSON STRIP [EXTRA KEY=VALUE ...]   (sets RC; the log is OUT_JSON with .log)
  local root=$1 cid=$2 js=$3 strip=$4 k; shift 4
  local -a envs unset_args=()
  mapfile -t envs < <(compose_env "$root" "${ENVF[$cid]}" "$strip")
  while IFS= read -r k; do unset_args+=(-u "$k"); done < <(env | grep -oE '^KRABBY_[A-Z0-9_]+')
  while ps -ww -eo args | grep -qE "$ISAAC_PAT"; do sleep 20; done
  rm -f "$js"
  ( cd "$root/parkour" && env "${unset_args[@]}" OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 \
      PYTHONPATH="$root/parkour/parkour_tasks:$root/parkour" "${envs[@]}" "$@" \
      "${CAP[@]}" timeout --signal=TERM 900 "$PY" -u "$root/tests/integration/crab_hex_phase_cfg_dump.py" --headless \
      --task "${TASK[$cid]}" --out "$js" ) > "${js%.json}.log" 2>&1
  RC=$?
  echo "RSD_DUMP INFO dump root=$([ "$root" = "$R" ] && echo new || echo ref) case=$cid rc=$RC json=$js"
}

# ---- knob OFF: five cases x two roots
for c in "${CASES[@]}"; do
  strip=0; [ "$c" = C3 ] && strip=1
  dump "$REF" "$c" "$OUT/ref__$c.json" "$strip"; rc_ref=$RC
  dump "$R" "$c" "$OUT/new__$c.json" "$strip"; rc_new=$RC
  name="$c knob-off dump byte-identical to the reference (${TASK[$c]} x $(basename "${ENVF[$c]}"))"
  if [ "$rc_ref" -ne 0 ] || [ "$rc_new" -ne 0 ] || [ ! -s "$OUT/ref__$c.json" ] || [ ! -s "$OUT/new__$c.json" ]; then
    check FAIL "$name" "a dump failed: ref rc=$rc_ref, new rc=$rc_new (logs: ${OUT}/ref__$c.log, ${OUT}/new__$c.log)"
    continue
  fi
  line=$("${CMP[@]}" same "$OUT/ref__$c.json" "$OUT/new__$c.json" "$REF" "$R" | grep '^VERDICT ' | head -1)
  if [ "${line#VERDICT IDENTICAL}" != "$line" ]; then check PASS "$name" "${line#VERDICT IDENTICAL }"
  else check FAIL "$name" "${line#VERDICT }"; fi
done

# ---- knob ON: this worktree only, C1 and C3
for c in C1 C3; do
  strip=0; [ "$c" = C3 ] && strip=1
  name="$c knob-on diff touches exactly rod_speed_dr and the environ key (${TASK[$c]})"
  dump "$R" "$c" "$OUT/on__$c.json" "$strip" "KRABBY_DR_ROD_SPEED=$DR_ON"
  if [ "$RC" -ne 0 ] || [ ! -s "$OUT/on__$c.json" ] || [ ! -s "$OUT/new__$c.json" ]; then
    check FAIL "$name" "knob-on dump failed: rc=$RC (log: $OUT/on__$c.log) or the knob-off dump is missing"
    continue
  fi
  line=$("${CMP[@]}" onoff "$OUT/new__$c.json" "$OUT/on__$c.json" "${TASK[$c]}" 0.6 1.35 12000 | grep '^VERDICT ' | head -1)
  if [ "${line#VERDICT OK}" != "$line" ]; then check PASS "$name" "${line#VERDICT OK }"
  else check FAIL "$name" "${line#VERDICT }"; fi
done

# ---- refusal: DR + stress rod speed, C1
dump "$R" C1 "$OUT/refuse__C1.json" 0 "KRABBY_DR_ROD_SPEED=$DR_ON" "KRABBY_STRESS_ROD_SPEED=0.7"
name="C1 KRABBY_DR_ROD_SPEED + KRABBY_STRESS_ROD_SPEED=0.7 is refused"
line=$("${CMP[@]}" refusal "$OUT/refuse__C1.log" "$RC" | grep '^VERDICT ' | head -1)
if [ "${line#VERDICT OK}" != "$line" ] && [ ! -s "$OUT/refuse__C1.json" ]; then check PASS "$name" "${line#VERDICT OK }"
elif [ -s "$OUT/refuse__C1.json" ]; then check FAIL "$name" "a dump was written: the cfg was built with both knobs"
else check FAIL "$name" "${line#VERDICT }"; fi

finish

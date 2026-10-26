#!/bin/bash
# Task 4 pipeline: ONE gate row (a copy of 2026-10-15_body_cmd/scripts/bc_eval_row.sh, extended; the launcher class
# every recorded REFS / anchor stamp came from). The env is EXACTLY a composed KEY=VALUE file (gen writes it from the
# stage's composed env) -- every other KRABBY_* variable of the parent is removed -- plus, for a payload level row,
# a level file (the four KRABBY_PAYLOAD_* keys). Memory-capped scope; waits while any Isaac python runs (anchored
# `ps -ww -eo args` pattern, never `pgrep -f`).
#
#   gate_row.sh NAME CHECKPOINT ENV_FILE MANIFEST SCENARIO ENV_SEED [extra harness args...]
#     every path ABSOLUTE (the launch cds into parkour/; a relative path is refused, rc 2)
#   TP_OUT          the output root (required): stamps land in $TP_OUT/lr/NAME/SCENARIO/seed00<k>/<time>/
#   TP_LEVEL_FILE   optional KEY=VALUE file of KRABBY_PAYLOAD_* keys applied after ENV_FILE (refused when ENV_FILE
#                   already sets one of them)
#   TP_STRIP_KEYS   optional space-separated KRABBY_* names dropped from the composed env (the Student-Play rows:
#                   omni_student_spec.STUDENT_RUNNER_KNOBS_REFUSED, which the student runner refuses); each one
#                   dropped is recorded in tp_row.json
#   TP_ROW_TIMEOUT  seconds (default 5400; SIGTERM only, never a follow-up KILL)
#   TP_ROW_DRY=1    print the composed env (one KEY=VALUE per line), the stripped keys and the harness command, then
#                   exit 0 -- nothing waits, nothing launches (the unit tests' and the dry pass's view of a row)
# SUCCESS is judged by the STAMP, never by the harness exit code alone: rc 0 AND a stamp dir with run_meta.json +
# scenario_metrics.json + summary.md + raw/, AND complete.json on a category-suite row (the harness writes it last,
# on suite rows only). Then code_state.json, then tp_row.json (the completion marker, LAST). Otherwise rc 8.
# Prints "TPROW END name=... rc=... stamp=..." as its last line.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)  # this checkout's root, from the script's own location
PY=/home/nickmagus/krabby/isaac_venv/bin/python
EVAL=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
PYP=$R/parkour/parkour_tasks:$R/parkour
TOOLS=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+|bc_dump_cfgs|t4_dump_cfgs)\.py'
NONKIT_PAT='(s[hn]_dagger\.py .*--(read|emit-env)([ =]|$))|(s[hn]_table_test\.py .*--(emit-facts|merge)([ =]|$))|(/s[hn]_(facts|kin|core|table_core|eval_common|read|bridge_read|observe_read|calib|common)\.py( |$))'

end() { echo "TPROW END name=${name:-?} rc=$1 stamp=${2:-} ${3:-}"; exit "$1"; }
[ $# -ge 6 ] || end 2 "" "(usage: gate_row.sh NAME CHECKPOINT ENV_FILE MANIFEST SCENARIO ENV_SEED [args...])"
name=$1; ck=$2; src=$3; man=$4; sc=$5; es=$6; shift 6
[ -n "${TP_OUT:-}" ] || end 2 "" "(TP_OUT is not set)"
OUT=$TP_OUT
case "$name" in ''|*[!A-Za-z0-9_]*) end 2 "" "(NAME must be [A-Za-z0-9_]+)" ;; esac
case "$es" in ''|*[!0-9]*|0) end 2 "" "(env_seed '$es' not a positive integer)" ;; esac
for p in "$ck" "$src" "$man" "$OUT" ${TP_LEVEL_FILE:+"$TP_LEVEL_FILE"}; do
  case "$p" in /*) ;; *) end 2 "" "(relative path refused: $p)" ;; esac
  case "$p" in *[[:space:]*?\[\]]*) end 2 "" "(path with whitespace / glob characters: $p)" ;; esac
done
[ -f "$ck" ] || end 2 "" "(no checkpoint $ck)"
[ -f "$src" ] || end 2 "" "(no env file $src)"
[ -f "$man" ] || end 2 "" "(no manifest $man)"
sd=$(printf 'seed%03d' "$es")
envs=$(grep -E '^KRABBY_[A-Z0-9_]+=' "$src" | tr '\n' ' ')
dropped=""
for k in ${TP_STRIP_KEYS:-}; do
  case "$k" in KRABBY_[A-Z0-9_]*) ;; *) end 2 "" "(TP_STRIP_KEYS entry '$k' is not a KRABBY_* name)" ;; esac
  if printf '%s\n' $envs | grep -q "^$k="; then dropped="$dropped $k"; fi
  envs=$(printf '%s\n' $envs | grep -v "^$k=" | tr '\n' ' ')
done
level_envs=""
if [ -n "${TP_LEVEL_FILE:-}" ]; then
  [ -f "$TP_LEVEL_FILE" ] || end 2 "" "(no level file $TP_LEVEL_FILE)"
  level_envs=$(grep -E '^KRABBY_PAYLOAD_[A-Z0-9_]+=' "$TP_LEVEL_FILE" | tr '\n' ' ')
  for kv in $level_envs; do
    k=${kv%%=*}
    if printf '%s\n' $envs | grep -q "^$k="; then end 2 "" "(level key $k already set by $src)"; fi
  done
fi
unset_args=$(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')
if [ "${TP_ROW_DRY:-0}" = 1 ]; then
  echo "# stripped:${dropped:- none}"
  printf '%s\n' $envs $level_envs
  echo "$PY -u $EVAL --headless --manifest $man --scenario $sc --env-seed $es --checkpoint $ck --save-raw --no-plot --output-root $OUT/lr/$name $*"
  end 0 "" "(dry)"
fi
while ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; do sleep 20; done
mkdir -p "$OUT/lr/$name"
log="$OUT/lr/$name.$sc.$sd.log"
t0=$(date +%s)
( cd "$R/parkour" && env $unset_args OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH="$PYP" $envs \
    $level_envs $CAP timeout --signal=TERM "${TP_ROW_TIMEOUT:-5400}" "$PY" -u "$EVAL" --headless --manifest "$man" \
    --scenario "$sc" --env-seed "$es" --checkpoint "$ck" --save-raw --no-plot --output-root "$OUT/lr/$name" "$@" ) \
    > "$log" 2>&1
rc=$?
d=""
for c in $(ls -d "$OUT/lr/$name/$sc/$sd"/*/ 2>/dev/null); do
  [ "$(stat -c %Y "$c")" -ge "$t0" ] && d=${c%/}
done
[ "$rc" = 0 ] || end "$rc" "$d" "(harness rc $rc; log $log)"
[ -n "$d" ] || end 8 "" "(harness rc 0 but no new stamp under $OUT/lr/$name/$sc/$sd; log $log)"
for f in run_meta.json scenario_metrics.json summary.md; do
  [ -f "$d/$f" ] || end 8 "$d" "(stamp has no $f)"
done
[ -d "$d/raw" ] || end 8 "$d" "(stamp has no raw/)"
if python3 -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get('suite') else 1)" "$d/run_meta.json"; then
  [ -f "$d/complete.json" ] || end 8 "$d" "(category-suite stamp without complete.json)"
fi
( cd "$TOOLS" && "$PY" -I -c "
import json, subprocess, sys, datetime
sys.dont_write_bytecode = True
sys.path.insert(0, '.')
import run_candidates as r
st = subprocess.run(['git', '-C', str(r.REPO), 'status', '--porcelain', '--', *r.CODE_PATHSPECS, *r.EVAL_PATHSPECS],
                    capture_output=True, text=True).stdout.split('\n')
dirty = [l[3:] for l in st if l.strip() and not r.code_drop(l[3:])]
json.dump({'code_sha': r.code_sha(), 'eval_sha': r.eval_sha(), 'dirty': str(bool(dirty)), 'dirty_paths': json.dumps(dirty),
           'written_at': datetime.datetime.now().isoformat(timespec='seconds'), 'writer': 'TP gate_row.sh'},
          open(sys.argv[1] + '/code_state.json', 'w'), indent=1)
" "$d" ) || end 9 "$d" "(code_state.json not written)"
python3 - "$d" "$name" "$ck" "$src" "${TP_LEVEL_FILE:-}" "$man" "$sc" "$es" "$dropped" "$@" <<'PY' || end 9 "$d" "(tp_row.json not written)"
import json, sys, datetime
d, name, ck, src, lvl, man, sc, es, dropped = sys.argv[1:10]
json.dump({"by": "TP gate_row.sh", "name": name, "checkpoint": ck, "env_file": src, "level_file": lvl or None,
           "manifest": man, "scenario": sc, "env_seed": int(es), "stripped_keys": dropped.split(),
           "args": sys.argv[10:], "rc": 0, "written_at": datetime.datetime.now().isoformat(timespec="seconds")},
          open(d + "/tp_row.json", "w"), indent=1)
PY
end 0 "$d"

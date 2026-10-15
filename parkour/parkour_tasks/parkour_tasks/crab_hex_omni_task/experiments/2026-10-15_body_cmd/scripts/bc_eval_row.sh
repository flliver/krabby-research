#!/bin/bash
# One eval row for the body-command campaign, launched exactly like phase 2's run_arm.sh row_eval (the launcher that
# wrote the REFS_S1 stamps): the env is EXACTLY a recorded env -- an earlier stamp's run_meta env_vars, or a
# KEY=VALUE file -- with every other KRABBY_* variable removed; memory-capped scope; waits while any Isaac python runs.
#
#   bc_eval_row.sh NAME CHECKPOINT ENV_SOURCE MANIFEST SCENARIO ENV_SEED [extra eval args...]
#     ENV_SOURCE  a stamp dir (its run_meta.json env_vars) or a file of KEY=VALUE lines
# Output stamp: $OUT/lr/NAME/SCENARIO/seed00<k>/<time>/ ; a code_state.json is written into it after a clean exit.
# Prints "BCROW END name=... rc=... stamp=..." as its last line.
set -u
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
EVAL=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
PYP=$R/parkour/parkour_tasks:$R/parkour
OUT=${BC_OUT:-$R/parkour/outputs/omni_plan/body_cmd}
TOOLS=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+)\.py'

name=$1; ck=$2; src=$3; man=$4; sc=$5; es=$6; shift 6
case "$es" in ''|*[!0-9]*) echo "BCROW END name=$name rc=2 stamp= (env_seed '$es' not a positive integer)"; exit 2 ;; esac
sd=$(printf 'seed%03d' "$es")
if [ -d "$src" ]; then
  envs=$(python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))" "$src/run_meta.json")
else
  envs=$(grep -E '^KRABBY_[A-Z0-9_]+=' "$src" | tr '\n' ' ')
fi
unset_args=$(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')
while ps -eo args | grep -qE "$ISAAC_PAT"; do sleep 20; done
mkdir -p "$OUT/lr/$name"
log="$OUT/lr/$name.$sc.$sd.log"
( cd "$R/parkour" && env $unset_args OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH="$PYP" $envs \
    $CAP timeout 5400 "$PY" -u "$EVAL" --headless --manifest "$man" --scenario "$sc" --env-seed "$es" \
    --checkpoint "$ck" --save-raw --no-plot --output-root "$OUT/lr/$name" "$@" ) > "$log" 2>&1
rc=$?
d=$(ls -d "$OUT/lr/$name/$sc/$sd"/*/ 2>/dev/null | tail -1)
if [ "$rc" = 0 ] && [ -n "$d" ]; then
  ( cd "$TOOLS" && "$PY" -I -c "
import json, subprocess, sys, datetime
sys.path.insert(0, '.')
import run_candidates as r
st = subprocess.run(['git', '-C', str(r.REPO), 'status', '--porcelain', '--', *r.CODE_PATHSPECS, *r.EVAL_PATHSPECS],
                    capture_output=True, text=True).stdout.split('\n')
dirty = [l[3:] for l in st if l.strip() and not r.code_drop(l[3:])]
json.dump({'code_sha': r.code_sha(), 'eval_sha': r.eval_sha(), 'dirty': str(bool(dirty)), 'dirty_paths': json.dumps(dirty),
           'written_at': datetime.datetime.now().isoformat(timespec='seconds'), 'writer': 'body_cmd bc_eval_row.sh'},
          open(sys.argv[1] + '/code_state.json', 'w'), indent=1)
" "$d" ) || rc=9
fi
echo "BCROW END name=$name rc=$rc stamp=$d"
exit $rc

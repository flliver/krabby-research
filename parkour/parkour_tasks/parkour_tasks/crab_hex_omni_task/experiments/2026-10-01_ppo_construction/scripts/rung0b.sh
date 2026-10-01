#!/usr/bin/env bash
# Rung 0b: the C_lr_rsi cam-staging diagnostic (CHARTER section 5; plan step 2 item 4). Runs the 8 rows of
# ../scenarios_rung0b.yaml as DIRECT evals of X02 (the FAST C head), ONE Isaac process per row, serially, with the
# harness command pattern of the side bisect's queue_eplen2.sh lr_eval: X02's own driver-stamp env
# (run_meta.env_vars) in the process environment, `systemd-run --scope` MemoryMax=48G, `timeout 3600`, the anchored
# idle check before each row (never `pgrep -f`), never SIGKILL. Then scripts/rung0b_read.py reads the fall shares
# and applies the pre-registered rule.
#
# Run ONLY after the staging smoke passed, from the landed tree (CODE_ROOT, default the main checkout) or the
# worktree's own eval script (CODE_ROOT=<worktree>): the main checkout's eval script puts its own checkout first on
# sys.path, so the eval script and PYTHONPATH must come from the same tree.
#
# Usage: rung0b.sh [ROW_ID ...]      (default: all 8 rows, seed-1 block then seed-2 block)
# Env:   CODE_ROOT, PYTHONPATH_EVAL (as run_arm.sh), X02_CKPT, X02_STAMP (the driver stamp whose env is carried)
set -u
R=/home/nickmagus/krabby/krabby-research
PY=/home/nickmagus/krabby/isaac_venv/bin/python
CODE_ROOT=${CODE_ROOT:-$R}
C=$(cd "$(dirname "$0")/.." && pwd)
MAN=$C/scenarios_rung0b.yaml
EVAL=$CODE_ROOT/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py
PYP=${PYTHONPATH_EVAL:-$CODE_ROOT/parkour/parkour_tasks:$CODE_ROOT/parkour}
X02_CKPT=${X02_CKPT:-$R/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-29_01-52-56_seed3_X02_s1_A15pB/model_2999.pt}
X02_STAMP=${X02_STAMP:-$R/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect/seed3_X02_s1_A15pB/side22_25__omni/seed001/2026-09-29_07-00-57}
OUT=$R/parkour/outputs/omni_plan/ppo_construction/rung0b
LOG=$R/parkour/outputs/omni_plan/ppo_construction/queue.log
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
mkdir -p "$OUT"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
drivers_idle() { while ps -eo args | grep -qE "^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py"; do sleep 30; done; }

[ -f "$X02_CKPT" ] || { echo "no X02 checkpoint $X02_CKPT" >&2; exit 2; }
[ -f "$X02_STAMP/run_meta.json" ] || { echo "no X02 stamp $X02_STAMP" >&2; exit 2; }
ROWS=("$@")
if [ ${#ROWS[@]} -eq 0 ]; then
  ROWS=()
  for s in 1 2; do for r in frac0_off frac0_on rsi_off rsi_on; do ROWS+=("rung0b_${r}_e${s}__omni"); done; done
fi
ENVS=$(python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(' '.join(f'{k}={v}' for k,v in sorted(m['env_vars'].items())))" "$X02_STAMP/run_meta.json")
say "rung0b start: ${ROWS[*]} (code root $CODE_ROOT, X02 $X02_CKPT)"
for row in "${ROWS[@]}"; do
  drivers_idle
  ( cd "$CODE_ROOT/parkour" && env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONPATH="$PYP" $ENVS \
      $CAP timeout 3600 "$PY" "$EVAL" --headless --manifest "$MAN" --scenario "$row" \
      --checkpoint "$X02_CKPT" --save-raw --no-plot --output-root "$OUT/$row" ) > "$OUT/$row.log" 2>&1
  rc=$?
  d=$(ls -d "$OUT/$row/$row"/seed*/*/ 2>/dev/null | tail -1)
  # A plain schedule row gets no complete.json (category-suite rows only): finished = exit 0 and every raw
  # episode npz present (rung0b_read.stamp_complete; fix 2026-10-01, SMOKES.md).
  if [ -z "$d" ] || [ "$rc" != 0 ] || ! "$PY" -c "import sys; sys.path.insert(0, sys.argv[1]); import rung0b_read as r, pathlib; sys.exit(0 if r.stamp_complete(pathlib.Path(sys.argv[2])) else 1)" "$C/scripts" "$d"; then
    say "STOP rung0b $row: no complete stamp (rc $rc, log $OUT/$row.log)"
    exit 1
  fi
  say "rung0b $row stamp ${d%/} (staging log lines: $(grep -c RSI_CAM_STAGE "$OUT/$row.log"))"
done
"$PY" "$C/scripts/rung0b_read.py" --root "$OUT" --json "$OUT/rung0b_read.json" | tee -a "$LOG"
say "DONE rung0b"

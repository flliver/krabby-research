#!/bin/bash
# DR stress sweep, step 3: the KNOB-OFF proof. Dumps the Student-Play env cfg (tests/integration/crab_hex_phase_cfg_dump.py,
# Kit, no simulation stepped) with EXACTLY a row's env (every parent KRABBY_* removed, then the env file's keys) from a
# reference checkout (main, before the knobs) and from this checkout, for p3b.env and p3b.field_light.env, then compares
# the two dumps byte for byte after replacing each checkout's root path with a token. Memory-capped, SIGTERM-only,
# waits while any Isaac python runs.
#
#   dump_check.sh REF_ROOT OUT_DIR [EXTRA_KEY=VALUE ...]     (extra keys go into THIS checkout's dumps only)
# Prints "DUMPCHECK <env> identical=<True|False>" per env file and "DUMPCHECK END all_identical=<...>".
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(cd "$HERE/../../../../../../.." && pwd)
REF=$1; OUT=$2; shift 2
PY=/home/nickmagus/krabby/isaac_venv/bin/python
TASK=Isaac-Crab-Hex-Omni-Student-Play-v0
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|stress_smoke|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+|bc_dump_cfgs|t4_dump_cfgs)\.py'
mkdir -p "$OUT"
dump() {  # root env_file out_json extra...
  local root=$1 src=$2 js=$3; shift 3
  local envs unset_args
  envs=$(grep -E '^KRABBY_[A-Z0-9_]+=' "$src" | tr '\n' ' ')
  unset_args=$(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')
  while ps -ww -eo args | grep -qE "$ISAAC_PAT"; do sleep 20; done
  ( cd "$root/parkour" && env $unset_args OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 \
      PYTHONPATH="$root/parkour/parkour_tasks:$root/parkour" $envs "$@" \
      $CAP timeout --signal=TERM 900 "$PY" -u "$root/tests/integration/crab_hex_phase_cfg_dump.py" --headless \
      --task "$TASK" --out "$js" ) > "${js%.json}.log" 2>&1
}
all=True
for e in p3b p3b.field_light; do
  src=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/F19/envs/$e.env
  dump "$REF" "$src" "$OUT/ref__$e.json"
  dump "$R" "$src" "$OUT/new__$e.json" "$@"
  same=$(python3 - "$OUT/ref__$e.json" "$OUT/new__$e.json" "$REF" "$R" <<'PY'
import sys
a, b, ra, rb = sys.argv[1:5]
try:
    ta = open(a).read().replace(ra, "<ROOT>")
    tb = open(b).read().replace(rb, "<ROOT>")
except FileNotFoundError as exc:
    print(f"False(missing {exc.filename})"); sys.exit(0)
print(ta == tb)
PY
)
  echo "DUMPCHECK $e identical=$same"
  [ "$same" = True ] || all=False
done
echo "DUMPCHECK END all_identical=$all"

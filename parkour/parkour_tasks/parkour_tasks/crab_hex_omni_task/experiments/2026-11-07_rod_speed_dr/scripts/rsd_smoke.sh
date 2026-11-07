#!/bin/bash
# Rod-speed DR campaign, step 4 / K2: the dose smoke, ONE Kit process per group (D0 off, D1 0.5:1.4, D2 0.5:1.0:300).
# The env is composed like tools/pipeline/gate_row.sh composes a row's: every KRABBY_* of the parent is removed, then the
# F19 p3b.distill.env keys (T*'s training env), then the group's KRABBY_DR_ROD_SPEED. Isaac-Crab-Hex-Omni-Teacher-v0,
# 256 envs, seed 7, ~600 env steps (rsd_smoke.py). Memory-capped `systemd-run --user --scope`, `timeout --signal=TERM`
# (SIGTERM only); waits while any Isaac python runs (the anchored `ps -ww -eo args` pattern, never `pgrep -f`).
#
#   rsd_smoke.sh OUT_DIR D0|D1|D2 [D0|D1|D2 ...]
# Per group: OUT_DIR/<g>.json (the result), OUT_DIR/<g>.log (the Kit log). Prints one
#   "RSD_SMOKE PASS|FAIL group=<g> rc=<rc> checks=<n> json=<path>" line (+ one "RSD_SMOKE   failed <check>: <detail>" line
# per failed check) and "RSD_SMOKE END pass=<n> fail=<n>"; exits 1 on any FAIL.
# A group PASSES iff the JSON is complete with all_ok, the Kit log has exactly the expected number of "[DR]" (0 / 1) and
# "[KNOB] KRABBY_DR_ROD_SPEED=" (0 / 1) lines, and the exit code is 0 (124 with a complete JSON = a teardown hang: PASS + note).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(cd "$HERE/../../../../../../.." && pwd)
PY=/home/nickmagus/krabby/isaac_venv/bin/python
BASE_ENV=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/F19/envs/p3b.distill.env
CAP=(systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0)
TMO=${RSD_SMOKE_TIMEOUT:-1500}
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|stress_smoke|rsd_smoke|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+|bc_dump_cfgs|t4_dump_cfgs)\.py'
[ $# -ge 2 ] || { echo "usage: rsd_smoke.sh OUT_DIR D0|D1|D2 ..." >&2; exit 2; }
OUT=$1; shift
mkdir -p "$OUT" || exit 2
OUT=$(cd "$OUT" && pwd)
[ -f "$BASE_ENV" ] || { echo "RSD_SMOKE FAIL setup -- no env file $BASE_ENV" >&2; exit 2; }
group_keys() {
  case "$1" in
    D0) echo "" ;;
    D1) echo "KRABBY_DR_ROD_SPEED=0.5:1.4" ;;
    D2) echo "KRABBY_DR_ROD_SPEED=0.5:1.0:300" ;;
    *) echo BAD ;;
  esac
}
verdict() {  # JSON -> "complete=<b> all_ok=<b> n=<checks>" and the failed checks, one per line
  "$PY" -B -I - "$1" <<'PYEOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as exc:  # noqa: BLE001
    print(f"STATUS unreadable {exc!r}"); sys.exit(0)
print(f"STATUS complete={r.get('complete') is True} all_ok={r.get('all_ok') is True} n={len(r.get('checks', {}))}")
for k, v in sorted(r.get("checks", {}).items()):
    if not v.get("ok"):
        print(f"FAILED {k}: {v.get('detail', '')[:400]}")
PYEOF
}
npass=0; nfail=0
for g in "$@"; do
  keys=$(group_keys "$g")
  if [ "$keys" = BAD ]; then echo "RSD_SMOKE FAIL group=$g rc=2 -- unknown group"; nfail=$((nfail + 1)); continue; fi
  mapfile -t envs < <(grep -E '^KRABBY_[A-Z0-9_]+=' "$BASE_ENV")
  [ -n "$keys" ] && envs+=("$keys")
  unset_args=(); while IFS= read -r k; do unset_args+=(-u "$k"); done < <(env | grep -oE '^KRABBY_[A-Z0-9_]+')
  js=$OUT/$g.json; lg=$OUT/$g.log
  while ps -ww -eo args | grep -qE "$ISAAC_PAT"; do sleep 20; done
  rm -f "$js"
  ( cd "$R/parkour" && env "${unset_args[@]}" OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 \
      PYTHONPATH="$R/parkour/parkour_tasks:$R/parkour" "${envs[@]}" \
      "${CAP[@]}" timeout --signal=TERM "$TMO" "$PY" -u "$HERE/rsd_smoke.py" --headless --group "$g" --out "$js" \
      ) > "$lg" 2>&1
  rc=$?
  want=0; [ "$g" != D0 ] && want=1
  n_dr=$(grep -c '\[DR\] rod_speed n=' "$lg"); n_knob=$(grep -c '\[KNOB\] KRABBY_DR_ROD_SPEED=' "$lg")
  st=$(verdict "$js")
  status=$(printf '%s\n' "$st" | grep '^STATUS' | head -1)
  problems=()
  case "$status" in "STATUS complete=True all_ok=True"*) ;; *) problems+=("json: ${status#STATUS }") ;; esac
  [ "$n_dr" -eq "$want" ] || problems+=("log has $n_dr [DR] lines, want $want")
  [ "$n_knob" -eq "$want" ] || problems+=("log has $n_knob [KNOB] KRABBY_DR_ROD_SPEED lines, want $want")
  note=""
  if [ "$rc" -eq 124 ] && [ "${#problems[@]}" -eq 0 ]; then note=" (rc=124: killed by the timeout at Kit teardown, after the complete JSON)"
  elif [ "$rc" -ne 0 ]; then problems+=("exit code $rc"); fi
  nchk=$(printf '%s\n' "$status" | sed -n 's/.* n=\([0-9]*\).*/\1/p')
  if [ "${#problems[@]}" -eq 0 ]; then
    echo "RSD_SMOKE PASS group=$g rc=$rc checks=${nchk:-?} json=$js$note"; npass=$((npass + 1))
  else
    echo "RSD_SMOKE FAIL group=$g rc=$rc checks=${nchk:-?} json=$js log=$lg -- ${problems[*]}"; nfail=$((nfail + 1))
    printf '%s\n' "$st" | grep '^FAILED' | sed 's/^FAILED /RSD_SMOKE   failed /'
  fi
done
echo "RSD_SMOKE END pass=$npass fail=$nfail"
[ "$nfail" -eq 0 ]

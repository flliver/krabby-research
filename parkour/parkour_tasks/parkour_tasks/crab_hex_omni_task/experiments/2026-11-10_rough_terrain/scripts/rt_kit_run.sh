#!/bin/bash
# Rough-terrain campaign: launch ONE Kit python (a step-5 smoke script) the way the repo launchers do -- a copy of
# 2026-10-28_site_teacher/scripts/kit_run.sh, changed only in ISAAC_PAT (+ rt_kit_*.py, this campaign's Kit entry
# points: rt_kit_train_smoke.py, rt_kit_eval_probe.py) and this header. gate_row.sh's env contract, distill.sh's shape.
#
#   kit_run.sh ENV_FILE SCRIPT [ARGS...]
#     ENV_FILE  ABSOLUTE KEY=VALUE file; its ^KRABBY_* lines are the WHOLE KRABBY_* environment of the child (every
#               KRABBY_* of the parent is removed, as gate_row.sh / distill.sh / run_phases._subenv do); values are
#               passed verbatim (no $REPO resolution: a $REPO value is refused)
#     SCRIPT    ABSOLUTE path of the python entry point (run as `$PY SCRIPT ARGS...`, the run_phases.train shape: no -u,
#               PYTHONUNBUFFERED=1 instead)
#   KIT_LEVEL_FILE  optional KEY=VALUE file of KRABBY_PAYLOAD_* keys applied after ENV_FILE (refused when ENV_FILE
#                   already sets one of them; gate_row.sh's TP_LEVEL_FILE rule)
#   KIT_TIMEOUT     seconds (default 3600; `timeout --signal=TERM`, SIGTERM only, never a follow-up KILL)
#   KIT_RUN_DRY=1   print the composed env (one KEY=VALUE per line) and the command, then exit 0 -- nothing waits,
#                   nothing launches
# The child: cwd <this checkout>/parkour (train.py's relative logs/rsl_rl root, run_phases' cwd), OMNI_KIT_ACCEPT_EULA,
# TERM=xterm, PYTHONUNBUFFERED=1, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=<this checkout>/parkour/parkour_tasks:
# <this checkout>/parkour, inside `systemd-run --user --scope` with MemoryMax=48G / MemorySwapMax=0. Before the
# launch it waits while any other Isaac python runs (anchored `ps -ww -eo args`, never `pgrep -f`) and refuses a Kit
# DerivedDataCache above ${RTQ_DDC_MAX_GB:-80} GB (distill.sh's guard; the queue's RTQ_DDC_MAX_GB, default 80). Prints "KITRUN END rc=..." as its last line; the rc is the
# child's (124 = the timeout), 2 = usage / refusal, 5 = DerivedDataCache too big.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)  # this checkout's root, from the script's own location
PY=/home/nickmagus/krabby/isaac_venv/bin/python
PYP=$R/parkour/parkour_tasks:$R/parkour
CAP="systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0"
DDC=/home/nickmagus/krabby/isaac_venv/lib/python3.11/site-packages/isaacsim/kit/cache/DerivedDataCache
DDC_MAX_GB=${RTQ_DDC_MAX_GB:-80}      # the queue's knob (rt_machinery.DDC_MAX_GB), default 80 like distill.sh
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(rt_kit_[A-Za-z0-9_]+|run_phases|run_candidates|eval_crab_hex_gait|site_rec|obs_dump|site_dr_smoke|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+|bc_dump_cfgs|t4_dump_cfgs)\.py'
NONKIT_PAT='(s[hn]_dagger\.py .*--(read|emit-env)([ =]|$))|(s[hn]_table_test\.py .*--(emit-facts|merge)([ =]|$))|(/s[hn]_(facts|kin|core|table_core|eval_common|read|bridge_read|observe_read|calib|common)\.py( |$))|(obs_dump\.py .*--compare)'

end() { echo "KITRUN END rc=$1 ${2:-}"; exit "$1"; }
[ $# -ge 2 ] || end 2 "(usage: kit_run.sh ENV_FILE SCRIPT [ARGS...])"
src=$1; script=$2; shift 2
for p in "$src" "$script" ${KIT_LEVEL_FILE:+"$KIT_LEVEL_FILE"}; do
  case "$p" in /*) ;; *) end 2 "(relative path refused: $p)" ;; esac
  case "$p" in *[[:space:]*?\[\]]*) end 2 "(path with whitespace / glob characters: $p)" ;; esac
  [ -f "$p" ] || end 2 "(no such file $p)"
done
for a in "$@"; do case "$a" in *[[:space:]]*) end 2 "(argument with whitespace: '$a')" ;; esac; done
case "$DDC_MAX_GB" in ''|*[!0-9.]*|.*|*.*.*) end 2 "(RTQ_DDC_MAX_GB must be a positive number, got '$DDC_MAX_GB')" ;; esac
DDC_MAX_INT=${DDC_MAX_GB%%.*}
TMO=${KIT_TIMEOUT:-3600}
case "$TMO" in ''|*[!0-9]*|0) end 2 "(KIT_TIMEOUT must be a positive integer)" ;; esac
envs=$(grep -E '^KRABBY_[A-Z0-9_]+=' "$src" | tr '\n' ' ')
for kv in $envs; do case "$kv" in *'$'*) end 2 "(env value with a \$ (unresolved \$REPO?): $kv)" ;; esac; done
level_envs=""
if [ -n "${KIT_LEVEL_FILE:-}" ]; then
  level_envs=$(grep -E '^KRABBY_PAYLOAD_[A-Z0-9_]+=' "$KIT_LEVEL_FILE" | tr '\n' ' ')
  for kv in $level_envs; do
    k=${kv%%=*}
    if printf '%s\n' $envs | grep -q "^$k="; then end 2 "(level key $k already set by $src)"; fi
  done
fi
unset_args=$(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ')
if [ "${KIT_RUN_DRY:-0}" = 1 ]; then
  printf '%s\n' $envs $level_envs
  echo "(cd $R/parkour && PYTHONPATH=$PYP $CAP timeout --signal=TERM $TMO $PY $script $*)"
  end 0 "(dry)"
fi
while ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; do sleep 20; done
DDC_GB=$(du -s --block-size=1G "$DDC" 2>/dev/null | cut -f1); DDC_GB=${DDC_GB:-0}
[ "$DDC_GB" -le "$DDC_MAX_INT" ] || end 5 "(Kit DerivedDataCache is ${DDC_GB} GB > ${DDC_MAX_GB}: quarantine it first)"
echo "KITRUN START $(date -Is) env_file=$src level_file=${KIT_LEVEL_FILE:-} script=$script args=$*"
( cd "$R/parkour" && env $unset_args OMNI_KIT_ACCEPT_EULA=yes TERM=xterm PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PYP" $envs $level_envs $CAP timeout --signal=TERM "$TMO" "$PY" "$script" "$@" ) < /dev/null 2>&1
rc=$?
end "$rc"

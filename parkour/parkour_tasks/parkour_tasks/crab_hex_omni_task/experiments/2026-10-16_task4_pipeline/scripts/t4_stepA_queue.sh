#!/bin/bash
# PLAN step 2 (Phase A at HEAD, no new code): the six 20-it stage smokes through the candidates driver, one at a
# time, then the A3 payload feasibility probe on S1 (zero-action settle + suite__omni es1 at 0/45/90/135/180 kg via
# KRABBY_DR_MASS=m:m). Read afterwards with scripts/t4_stepA_read.py (CHARTER rules pinned 2026-10-16).
#
#   t4_stepA_queue.sh [--no-smokes] [--no-a3]
# Lines for the watcher:  T4A LAUNCH|STAGE <chain> ...   T4A ROW <name> rc=<rc>   T4A STOP <why>   T4A END
# The queue never kills anything.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)  # this checkout's root, from the script's own location
EXP=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments
C=$EXP/2026-10-16_task4_pipeline
T=$C/preflight
L=$EXP/tools/launch_candidates.sh
ROW=$C/scripts/t4_eval_row.sh
CK=$R/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt
ENVSRC=$R/parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24
MO=$EXP/eval/scenarios_omni.yaml
TASK=Isaac-Crab-Hex-Omni-Walk-Play-v0
ISAAC_PAT='^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+)\.py'
smokes=1; a3=1
for a in "$@"; do
  case "$a" in --no-smokes) smokes=0 ;; --no-a3) a3=0 ;; *) echo "T4A STOP unknown arg $a"; exit 2 ;; esac
done
echo "T4A START $(date -Is) smokes=$smokes a3=$a3"
if [ "$smokes" = 1 ]; then
  for chain in PF_s1_s18 PF_c4_s18 PF_c5_s18 PF_c6_s18 PF_c6x_s18 PF_p2a_s18; do
    while ps -eo args | grep -qE "$ISAAC_PAT"; do sleep 30; done
    # T4_DRIVER_ARGS: extra run_candidates args, word-split on purpose
    out=$("$L" --campaign-dir "$T" --reference-chain "$chain" --seed 18 ${T4_DRIVER_ARGS:-} 2>&1)
    unit=$(printf '%s\n' "$out" | grep -E '^cands-' | head -n 1)
    [ -n "$unit" ] || { echo "T4A STOP $chain (launcher printed no unit: $(printf '%s' "$out" | tail -n 2 | tr '\n' ' '))"; exit 4; }
    echo "T4A LAUNCH $chain unit=$unit $(date -Is)"
    sleep 20
    while systemctl --user is-active -q "$unit"; do sleep 30; done
    echo "T4A STAGE $chain unit=$unit $(date -Is)"
  done
fi
if [ "$a3" = 1 ]; then
  for m in 0 45 90 135 180; do
    if [ "$m" = 0 ]; then extra=""; else extra="KRABBY_DR_MASS=$m:$m"; fi
    T4_EXTRA_ENV="$extra" "$ROW" "A3_settle_${m}kg" "$CK" "$ENVSRC" - "settle_${m}kg" 1 \
        --task "$TASK" --holds "0:10:stand" --episodes 32 --zero-actions > /dev/null 2>&1
    echo "T4A ROW A3_settle_${m}kg rc=$? $(date -Is)"
    T4_EXTRA_ENV="$extra" "$ROW" "A3_suite_${m}kg" "$CK" "$ENVSRC" "$MO" suite__omni 1 > /dev/null 2>&1
    echo "T4A ROW A3_suite_${m}kg rc=$? $(date -Is)"
  done
fi
echo "T4A END $(date -Is)"

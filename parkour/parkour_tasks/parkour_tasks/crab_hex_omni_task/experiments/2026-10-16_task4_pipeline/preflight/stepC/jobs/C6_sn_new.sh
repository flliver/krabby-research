# C6_sn_new [C6 stand] SN 100-it smoke beside the payload pilot (SN_WALKER_SPEC)
set -u
W=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/preflight/stepC/jobs/C6_sn_new.walker.json
export SN_WALKER_SPEC="$(/home/nickmagus/krabby/isaac_venv/bin/python -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$W" SN_WALKER_SPEC)"
export SN_WALKER_CK="$(/home/nickmagus/krabby/isaac_venv/bin/python -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$W" SN_WALKER_CK)"
export SN_WALKER_ST="$(/home/nickmagus/krabby/isaac_venv/bin/python -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$W" SN_WALKER_ST)"
export P2_REANCHOR="$(/home/nickmagus/krabby/isaac_venv/bin/python -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$W" P2_REANCHOR)"
SN_OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/task4/stepC/stand_network bash /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-11_stand_network/scripts/sn_dagger.sh --wait-idle T4C6_SN_new -- --mode dagger --arm mask --seed 17 --iterations 4000 --smoke --smoke-iters 100

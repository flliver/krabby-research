# C6_sh_new [C6 stand] SH 100-it smoke, warm start = the SN smoke head
set -u
W=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/preflight/stepC/jobs/C6_sh_new.walker.json
export SN_WALKER_SPEC="$(/home/nickmagus/krabby/isaac_venv/bin/python -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$W" SN_WALKER_SPEC)"
export SN_WALKER_CK="$(/home/nickmagus/krabby/isaac_venv/bin/python -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$W" SN_WALKER_CK)"
export SN_WALKER_ST="$(/home/nickmagus/krabby/isaac_venv/bin/python -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$W" SN_WALKER_ST)"
export P2_REANCHOR="$(/home/nickmagus/krabby/isaac_venv/bin/python -I -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$W" P2_REANCHOR)"
SH_OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/task4/stepC/stand_height bash /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-13_stand_height/scripts/sh_dagger.sh --wait-idle T4C6_SH_new_r2 -- --mode dagger --arm mask --seed 17 --iterations 5000 --deadband 0.25 --trim-branch --trim-min 0.0999 --smoke --smoke-iters 100 --warm-start /home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/task4/stepC/stand_network/runs/T4C6_SN_new/model_100.pt --warm-sha 2d3b2fec79cc501c26b3a4defdd4daccd55d4d81efaa532610f7c4e23d267a6d

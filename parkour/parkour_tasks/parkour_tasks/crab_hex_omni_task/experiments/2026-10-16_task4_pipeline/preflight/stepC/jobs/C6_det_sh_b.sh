# C6_det_sh_b [C6 stand] SH same-seed determinism (default warm start = the pinned A-mask s17)
set -u
SH_OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/task4/stepC/stand_height bash /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-13_stand_height/scripts/sh_dagger.sh --wait-idle T4C6_SH_det_b -- --mode dagger --arm mask --seed 17 --iterations 5000 --deadband 0.25 --trim-branch --trim-min 0.0999 --smoke --smoke-iters 100

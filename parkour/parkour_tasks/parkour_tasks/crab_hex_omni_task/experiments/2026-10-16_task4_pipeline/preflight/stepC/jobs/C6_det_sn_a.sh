# C6_det_sn_a [C6 stand] SN same-seed determinism (walker unset = the bundled S1, DS 17)
set -u
SN_OUT=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/task4/stepC/stand_network bash /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-11_stand_network/scripts/sn_dagger.sh --wait-idle T4C6_SN_det_a -- --mode dagger --arm mask --seed 17 --iterations 4000 --smoke --smoke-iters 100

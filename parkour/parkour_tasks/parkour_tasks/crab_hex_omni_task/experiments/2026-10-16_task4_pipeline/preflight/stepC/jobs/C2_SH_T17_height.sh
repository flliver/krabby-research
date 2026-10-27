# C2_SH_T17_height [C2 shrows] SH_T17 through the SH launcher (stand_net, deadband 0.25); aggregates compared
set -u
P2_REANCHOR=1 bash /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-13_stand_height/scripts/sh_eval_rows.sh --deadband 0.25 T4C2_SH_T17 stand_net manifest armed /home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/stand_height/runs/SH_T17/model_5000.pt 1 hwalkstop__omni@1 hraise__omni@1

# C7_dry [C7 dry] pipeline.py run --run dry (reduced rows by default; T4C_DRY=official for the official spec, ~12 GPU-h) with INJECT, auto-continue, SIGTERM-resume
set -u
/home/nickmagus/krabby/isaac_venv/bin/python -I /home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/scripts/t4_stepC_dry.py --mode "${T4C_DRY:-reduced}"

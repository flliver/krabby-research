# Crab Hex Flat-Walk Baseline: 2026-05-19_12-06-10

This folder snapshots the current best flat-walk baseline artifacts for the crab hexapod.

- Checkpoint: `model_4000.pt`
- USD: `crab_simple_2026-05-19_12-06-10.usda`

Key flat-walk settings from this baseline:

- `lin_vel_x = (0.25, 0.60)`
- `track_lin_vel_xy_exp = 1.0`
- `track_ang_vel_z_exp = 1.0`
- `reward_forward_progress_along_command = 0.50`
- `penalty_lin_vel_y = -3.0`
- `reward_feet_air_time_positive = 0.25`
- `feet_slide = 0.0`
- `reward_collision = 0.0`
- `KRABBY_HEX_SPAWN_Z = 1.05`

Use `../play_crab_hex_flat_walk_baseline.sh` to play this checkpoint with the bundled USD.

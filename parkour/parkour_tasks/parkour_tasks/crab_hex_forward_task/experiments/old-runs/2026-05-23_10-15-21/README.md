# Crab Hex Flat-Walk Baseline: 2026-05-23_10-15-21

This folder snapshots the flat-walk baseline trained in `logs/rsl_rl/crab_hex_flat_walk/2026-05-23_10-15-21/`.

**Tuning intent:** Finetune gait quality on top of the 2026-05-19 baseline — clearer alternating foot contacts, middle legs that step reliably, and stance tibiae held closer to perpendicular (spawn knee defaults) under load.

- Checkpoint: `model_6000.pt` (recommended play / stage-2 resume)
- USD: `crab_simple_2026-05-23_10-15-21.usda`

Key flat-walk settings (see task README Appendix C):

- `lin_vel_x = (0.30, 0.65)`
- `track_lin_vel_xy_exp = 1.25`
- `reward_forward_progress_along_command = 0.60` (`max_speed_scale = 1.75`)
- `reward_feet_air_time_positive = 0.40` @ **0.05** s
- `penalty_tibia_deviation_in_stance = -0.28`
- `penalty_foot_idle_when_forward = -0.12`
- `penalty_excess_feet_contact_forward = -0.20`
- joint action scale **0.24**
- ML/MR body–hip splay **±0.25**
- `KRABBY_HEX_SPAWN_Z = 1.05`

Play with the task README [Appendix C](../../../README.md#appendix-c--stage-1-flat-walk--2026-05-23-baseline) `play.py` block (`KRABBY_HEX_USD_PATH` = the bundled `crab_simple_2026-05-23_10-15-21.usda`, `KRABBY_HEX_SPAWN_Z=1.05`).

Plant: the pre-generator snapshot USD bundled in `../2026-05-23_10-15-21/` (`KRABBY_HEX_USD_PATH` to it); `KRABBY_HEX_SPAWN_Z=1.05` was the setting of record then (current default 1.085, main asset `assets/crab.usda` = A15+B).

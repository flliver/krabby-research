# Crab Hex Teacher 2b2 — lift-tuned teacher baseline: 2026-05-26_21-46-37

Stage 2b phase 2 (**lift-focused delta**): Appendix E `model_6198.pt` → **~106** PPO iters (`KRABBY_HEX_TEACHER_MODE=2b2`) → **`model_6300.pt`**.

Lift delta vs prior teacher-ready run: `reward_swing_vertical_vel` **0.8**, `penalty_swing_min_clearance` **−0.4**, `reward_recover_from_stall` **0.2**.

Log: `logs/rsl_rl/crab_hex_teacher/2026-05-26_21-46-37/`. USD: `../2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda`.

**Supersedes** [2026-05-26_11-30-18](../2026-05-26_11-30-18/) for distillation/play defaults.

Play: the task README [Appendix F](../../../README.md#appendix-f--stage-2b2-teacher-ready-baseline--2026-05-26) one-liner with the Appendix C USD + `model_6300.pt`.

`model_6400.pt` and `model_6500.pt` from the log are kept for reference; selected baseline remains `model_6300.pt` by play quality.

Plant: the pre-generator snapshot USD bundled in `../2026-05-23_10-15-21/` (`KRABBY_HEX_USD_PATH` to it); `KRABBY_HEX_SPAWN_Z=1.05` was the setting of record then (current default 1.085, main asset `assets/crab.usda` = A15+B).

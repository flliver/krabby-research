# Crab Hex Teacher 2b2 Baseline: 2026-05-26_11-30-18 (superseded)

**Superseded by** [2026-05-26_21-46-37](../2026-05-26_21-46-37/) — use that bundled `model_6300.pt` for play, distillation, and new work. This folder is kept for reference only.

Stage 2b phase 2 (2b2-v2 rewards): Appendix E `model_6198.pt` → **~102** PPO iters (`KRABBY_HEX_TEACHER_MODE=2b2`) → **`model_6300.pt`**.

Log: `logs/rsl_rl/crab_hex_teacher/2026-05-26_11-30-18/`. USD: `../2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda`.

Play (historical): the task README [Appendix F](../../../README.md#appendix-f--stage-2b2-teacher-ready-baseline--2026-05-26) one-liner with the Appendix C USD, substituting this `model_6300.pt` for the 2026-05-26_21-46-37 one. See Appendix F (superseded section).

**Do not use `6400+` from the same log** — play degrades (hole / fall) even when training continues.

Plant: the pre-generator snapshot USD bundled in `../2026-05-23_10-15-21/` (`KRABBY_HEX_USD_PATH` to it); `KRABBY_HEX_SPAWN_Z=1.05` was the setting of record then (current default 1.085, main asset `assets/crab.usda` = A15+B).

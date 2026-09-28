# Crab Hex Teacher Bridge Baseline: 2026-05-25_22-26-06

Stage 2a: Appendix C `model_6000.pt` → **100** bridge iters (`KRABBY_HEX_TEACHER_MODE=bridge`) → **`model_6099.pt`**.

Log: `logs/rsl_rl/crab_hex_teacher/2026-05-25_22-26-06/`. USD: `../2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda`.

Play: the task README [Appendix D](../../../README.md#appendix-d--stage-2a-teacher-bridge--2026-05-25-baseline) one-liner with the Appendix C USD + `model_6099.pt` — stable walk, some drift. Next curriculum step: Appendix E (`2b1`).

Plant: the pre-generator snapshot USD bundled in `../2026-05-23_10-15-21/` (`KRABBY_HEX_USD_PATH` to it); `KRABBY_HEX_SPAWN_Z=1.05` was the setting of record then (current default 1.085, main asset `assets/crab.usda` = A15+B).

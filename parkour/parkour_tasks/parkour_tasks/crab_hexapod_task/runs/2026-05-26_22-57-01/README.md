# Crab Hex Student — depth distillation baseline: 2026-05-26_22-57-01

Stage 3 student: distillation from [Appendix F](../2026-05-26_21-46-37/) teacher `model_6300.pt` → sweet-spot **`model_9800.pt`**.

**Log:** `logs/rsl_rl/crab_hex_student/2026-05-26_22-57-01/`.

**Why `9800`:** best combined metrics among late checkpoints — `ep_len` ~**758**, `goal_idx` ~**0.50**, `crab_failure` ~**28.7%**; later iters (e.g. **12000**, **12400**) were weaker on `goal_idx` / episode length. Play on 2b2-mixed student MDP matched (walking + hurdle crossing).

**Metrics @ `9800` (TensorBoard):** `crab_failure` ~**28.7%**; `ep_len` ~**758**; `goal_idx` ~**0.50**; `depth_actor_loss` ~**1.85**.

**Play @9800:** good forward walk on **50/50 flat/parkour** student terrain — use `play_crab_hex_student_baseline.sh`. Not tuned for 100% flat (`play_crab_hex_student_flat.sh` is diagnostic only).

**USD:** `runs/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda` (same as teacher bundles).

See task README [Appendix G](../../README.md#appendix-g--stage-3-student-distillation--2026-05-26).

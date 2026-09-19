# Phase pipeline campaign `smoke`

Plant **A15+B** (`/home/nickmagus/krabby/krabby-research/assets/variants/crab_simple__splay15_axis2p5in.usda`), seed 3, phases 2b. Driver: `sim_fine_tuning/tools/run_phases.py`; presets: `crab_hex_phases.py` (`KRABBY_PHASE` / `KRABBY_PLANT`).

| phase | task / mode | iterations | resume from | notes |
|---|---|---|---|---|
| 2b | `Isaac-Crab-Hex-Teacher-v0` + mode `2b` | 200 | 2a | phase 2 teacher-student: elements @10k, satellites -> eps |

## Phase 2b — Isaac-Crab-Hex-Teacher-v0 (mode 2b) — seed3 — 2026-09-07 13:30

- status **ok** | checkpoint `2026-09-07_13-24-28/model_10197.pt` | resume `model_9998.pt` | 200 its | wall 0.10 h
- smoke fail@2k 0.340 fail@end 0.340 coll@2k -0.036 ep_len@2k 1563.925
- exposure (non-RSI obstacle tiles): reach_edge 0.889 | reach_obst 0.839 | field_frac 0.497 (steps 710.620) | goals_passed 0.995
- obst_coverage[1..6]: 1:0.839 2:0.585 3:0.344 4:0.215 5:0.121 6:0.058
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.954 5:0.875 6:0.683
- failure share flat 0.171 (hazard 0.120/1k) | obst 0.500 (hazard 0.451/1k) | obst-RSI 0.537 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.342
- covariates: stand_frac_actual 0.162 | spread_frac_actual 0.000 | rsi_frac_actual 0.197 | terrain_levels 4.374 | how_far 3.826 | goal_idx 1.034 | ep len 1567.322 | prints 200
- stand time frac (logged) 0.162 | mean reward 51.599 | vloss 0.023 | mean ep len 1567.322
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_teacher
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)

## Phase 3a — Isaac-Crab-Hex-Student-v0 — seed3 — 2026-09-07 13:58

- status **ok** | checkpoint `2026-09-07_13-30-34/model_20195.pt` | resume `model_19996.pt` | 200 its | wall 0.47 h
- distillation: iterations logged 200 | depth_actor_loss first 10.757 -> last 3.767 | yaw_loss first 0.934 -> last 0.872
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_student
- check PASS: distillation algorithm (DistillationWithExtractor)
- check PASS: depth losses logged
- check PASS: iteration counter continued from the teacher head


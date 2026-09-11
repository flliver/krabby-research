# Sideways-formation campaign `2026-09-11_0414_sideways_formation`

Plant **A15+B** (`assets/crab.usda (main)`), seed 3, phases 1a, 2a, 2b, 2c. Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task/experiments/tools/run_phases.py`; presets: `crab_hex_side_phases.py` (`KRABBY_SIDE_PHASE` / `KRABBY_PLANT`).

| phase | task / mode | iterations | resume from | notes |
|---|---|---|---|---|
| 1a | `Isaac-Crab-Hex-Side-Walk-v0` | 5000 | scratch | sideways: phase 1 pure student: gait formation with walking slots and 40-s episodes |
| 2a | `Isaac-Crab-Hex-Side-Teacher-v0` + mode `2a` | 5000 | 1a | sideways: phase 2 teacher-student: elements @5k, apex/airtime/stride -> half |
| 2b | `Isaac-Crab-Hex-Side-Teacher-v0` + mode `2b` | 5000 | 2a | sideways: phase 2 teacher-student: elements @10k, satellites -> eps |
| 2c | `Isaac-Crab-Hex-Side-Teacher-v0` + mode `2c` | 5000 | 2b | sideways: phase 2 teacher-student: clock 1.0 -> 0.5 (policy of record) |

## Phase 1a — Isaac-Crab-Hex-Side-Walk-v0 — seed3 — 2026-09-11 13:37

- status **gate_1a** | checkpoint `2026-09-11_10-57-55/model_4999.pt` | resume `scratch` | 5000 its | wall 2.57 h
- smoke fail@2k 0.193 fail@end 0.236 coll@2k 0.000 ep_len@2k 1833.814
- exposure (non-RSI obstacle tiles): reach_edge 0.775 | reach_obst 0.637 | field_frac 0.429 (steps 725.080) | goals_passed 0.409
- obst_coverage[1..6]: 1:0.637 2:0.336 3:0.156 4:0.092 5:0.030 6:0.002
- RSI episodes: reach_edge 0.000 | reach_obst 0.000 | goals_passed 0.000 | coverage 1:0.000 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- failure share flat 0.109 (hazard 0.057/1k) | obst 0.574 (hazard 0.365/1k) | obst-RSI 0.000 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.235
- covariates: stand_frac_actual 0.149 | spread_frac_actual 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.973 | how_far 2.988 | goal_idx 1.017 | ep len 1845.434 | prints 5000
- stand time frac (logged) 0.149 | mean reward 69.657 | vloss 0.017 | mean ep len 1845.434
- slow canary (side manifest): tripod 0.355 | completion 0.930 | tracking 0.634 | falls 7/100 | pitch-fwd share 0.000
- step onset (side manifest, shallow 0.05-0.2): completion 0.770 | falls 23/100 | pitch-fwd share 0.000
- obstacle eval (recal2b2w 0.20-0.70): completion 0.780 | tripod 0.324 | falls 22/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_side_flat_walk
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: spawn yaw -pi/2 (params/env.yaml robot init_state rot)


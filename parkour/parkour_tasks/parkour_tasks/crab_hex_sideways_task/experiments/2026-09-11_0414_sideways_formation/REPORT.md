<!-- paths-note -->
> **Paths note:** this campaign lives at `parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task/experiments/2026-09-11_0414_sideways_formation/` and was created there on 2026-09-11 (it never lived under `sim_fine_tuning/`). Absolute paths below (`/home/.../parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval keep-set is under `evals/`.

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

## Phase 1a — gate_1a overridden — 2026-09-11 14:49

- `--continue`: model_4999.pt carried into 2a unretrained; the gate was revised to the walking-hold tripod (CHARTER: gate section + Log); the 1a record now reads `status ok` with `gate_1a_override`

## Phase 2a — Isaac-Crab-Hex-Side-Teacher-v0 (mode 2a) — seed3 — 2026-09-11 17:19

- status **ok** | checkpoint `2026-09-11_14-50-01/model_9998.pt` | resume `model_4999.pt` | 5000 its | wall 2.40 h
- smoke fail@2k 0.227 fail@end 0.178 coll@2k -0.006 ep_len@2k 1815.944
- exposure (non-RSI obstacle tiles): reach_edge 0.964 | reach_obst 0.931 | field_frac 0.658 (steps 1212.888) | goals_passed 1.017
- obst_coverage[1..6]: 1:0.931 2:0.810 3:0.552 4:0.320 5:0.154 6:0.082
- RSI episodes: reach_edge 0.000 | reach_obst 0.000 | goals_passed 0.000 | coverage 1:0.000 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- failure share flat 0.077 (hazard 0.040/1k) | obst 0.326 (hazard 0.184/1k) | obst-RSI 0.000 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.187
- covariates: stand_frac_actual 0.125 | spread_frac_actual 0.000 | rsi_frac_actual 0.000 | terrain_levels 5.737 | how_far 2.921 | goal_idx 1.173 | ep len 1868.346 | prints 5000
- stand time frac (logged) 0.125 | mean reward 68.239 | vloss 0.019 | mean ep len 1868.346
- slow canary (side manifest): walking-hold tripod 0.548 (gated; by hold: creep 0.548 / low 0.567 / stand 0.000; episode 0.384) | completion 0.880 | tracking 0.650 | falls 12/100 | pitch-fwd share 0.000
- step onset (side manifest, shallow 0.05-0.2): completion 0.820 | falls 18/100 | pitch-fwd share 0.000
- obstacle eval (recal2b2w 0.20-0.70): completion 0.940 | tripod 0.372 | falls 6/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_side_teacher
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: spawn yaw -pi/2 (params/env.yaml robot init_state rot)

## Phase 2b — Isaac-Crab-Hex-Side-Teacher-v0 (mode 2b) — seed3 — 2026-09-11 19:43

- status **ok** | checkpoint `2026-09-11_17-19-40/model_14997.pt` | resume `model_9998.pt` | 5000 its | wall 2.30 h
- smoke fail@2k 0.228 fail@end 0.212 coll@2k -0.011 ep_len@2k 1822.473
- exposure (non-RSI obstacle tiles): reach_edge 0.961 | reach_obst 0.936 | field_frac 0.663 (steps 1215.232) | goals_passed 1.017
- obst_coverage[1..6]: 1:0.936 2:0.789 3:0.587 4:0.349 5:0.187 6:0.095
- RSI episodes: reach_edge 0.000 | reach_obst 0.000 | goals_passed 0.000 | coverage 1:0.000 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- failure share flat 0.113 (hazard 0.060/1k) | obst 0.335 (hazard 0.192/1k) | obst-RSI 0.000 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.224
- covariates: stand_frac_actual 0.206 | spread_frac_actual 0.000 | rsi_frac_actual 0.000 | terrain_levels 6.168 | how_far 2.919 | goal_idx 1.092 | ep len 1785.899 | prints 5000
- stand time frac (logged) 0.206 | mean reward 52.531 | vloss 0.023 | mean ep len 1785.899
- slow canary (side manifest): walking-hold tripod 0.510 (gated; by hold: creep 0.582 / low 0.510 / stand 0.001; episode 0.378) | completion 0.910 | tracking 0.663 | falls 9/100 | pitch-fwd share 0.000
- step onset (side manifest, shallow 0.05-0.2): completion 0.820 | falls 18/100 | pitch-fwd share 0.000
- obstacle eval (recal2b2w 0.20-0.70): completion 0.870 | tripod 0.364 | falls 13/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_side_teacher
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: spawn yaw -pi/2 (params/env.yaml robot init_state rot)

## Phase 2c — Isaac-Crab-Hex-Side-Teacher-v0 (mode 2c) — seed3 — 2026-09-11 22:08

- status **ok** | checkpoint `2026-09-11_19-43-19/model_19996.pt` | resume `model_14997.pt` | 5000 its | wall 2.33 h
- smoke fail@2k 0.225 fail@end 0.302 coll@2k -0.010 ep_len@2k 1795.388
- exposure (non-RSI obstacle tiles): reach_edge 0.943 | reach_obst 0.900 | field_frac 0.641 (steps 1151.792) | goals_passed 1.109
- obst_coverage[1..6]: 1:0.900 2:0.759 3:0.534 4:0.303 5:0.178 6:0.079
- RSI episodes: reach_edge 0.000 | reach_obst 0.000 | goals_passed 0.000 | coverage 1:0.000 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- failure share flat 0.156 (hazard 0.084/1k) | obst 0.404 (hazard 0.242/1k) | obst-RSI 0.000 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.314
- covariates: stand_frac_actual 0.182 | spread_frac_actual 0.000 | rsi_frac_actual 0.000 | terrain_levels 5.780 | how_far 2.777 | goal_idx 1.101 | ep len 1737.818 | prints 5000
- stand time frac (logged) 0.182 | mean reward 42.854 | vloss 0.020 | mean ep len 1737.818
- slow canary (side manifest): walking-hold tripod 0.484 (gated; by hold: creep 0.567 / low 0.484 / stand 0.000; episode 0.364) | completion 0.840 | tracking 0.659 | falls 16/100 | pitch-fwd share 0.000
- step onset (side manifest, shallow 0.05-0.2): completion 0.630 | falls 37/100 | pitch-fwd share 0.027
- obstacle eval (recal2b2w 0.20-0.70): completion 0.820 | tripod 0.355 | falls 18/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_side_teacher
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: spawn yaw -pi/2 (params/env.yaml robot init_state rot)

## Phase 1a — Isaac-Crab-Hex-Side-Walk-v0 — seed2 — 2026-09-12 03:04

- status **ok** | checkpoint `2026-09-12_00-31-09/model_4999.pt` | resume `scratch` | 5000 its | wall 2.47 h
- smoke fail@2k 0.183 fail@end 0.210 coll@2k 0.000 ep_len@2k 1826.908
- exposure (non-RSI obstacle tiles): reach_edge 0.869 | reach_obst 0.801 | field_frac 0.538 (steps 928.061) | goals_passed 0.692
- obst_coverage[1..6]: 1:0.801 2:0.579 3:0.365 4:0.243 5:0.121 6:0.034
- RSI episodes: reach_edge 0.000 | reach_obst 0.000 | goals_passed 0.000 | coverage 1:0.000 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- failure share flat 0.084 (hazard 0.043/1k) | obst 0.514 (hazard 0.321/1k) | obst-RSI 0.000 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.195
- covariates: stand_frac_actual 0.193 | spread_frac_actual 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.027 | how_far 3.008 | goal_idx 0.986 | ep len 1788.632 | prints 5000
- stand time frac (logged) 0.193 | mean reward 71.112 | vloss 0.020 | mean ep len 1788.632
- slow canary (side manifest): walking-hold tripod 0.538 (gated; by hold: creep 0.538 / low 0.596 / stand 0.000; episode 0.381) | completion 0.790 | tracking 0.644 | falls 21/100 | pitch-fwd share 0.000
- step onset (side manifest, shallow 0.05-0.2): completion 0.600 | falls 40/100 | pitch-fwd share 0.000
- obstacle eval (recal2b2w 0.20-0.70): completion 0.760 | tripod 0.353 | falls 24/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_side_flat_walk
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: spawn yaw -pi/2 (params/env.yaml robot init_state rot)

## Phase 2a — Isaac-Crab-Hex-Side-Teacher-v0 (mode 2a) — seed2 — 2026-09-12 05:23

- status **ok** | checkpoint `2026-09-12_03-04-37/model_9998.pt` | resume `model_4999.pt` | 5000 its | wall 2.23 h
- smoke fail@2k 0.153 fail@end 0.076 coll@2k -0.004 ep_len@2k 1876.505
- exposure (non-RSI obstacle tiles): reach_edge 0.975 | reach_obst 0.954 | field_frac 0.696 (steps 1333.196) | goals_passed 1.527
- obst_coverage[1..6]: 1:0.954 2:0.881 3:0.744 4:0.545 5:0.357 6:0.217
- RSI episodes: reach_edge 0.000 | reach_obst 0.000 | goals_passed 0.000 | coverage 1:0.000 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- failure share flat 0.036 (hazard 0.018/1k) | obst 0.170 (hazard 0.091/1k) | obst-RSI 0.000 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.108
- covariates: stand_frac_actual 0.164 | spread_frac_actual 0.000 | rsi_frac_actual 0.000 | terrain_levels 6.303 | how_far 3.362 | goal_idx 1.372 | ep len 1870.687 | prints 5000
- stand time frac (logged) 0.164 | mean reward 75.561 | vloss 0.021 | mean ep len 1870.687
- slow canary (side manifest): walking-hold tripod 0.644 (gated; by hold: creep 0.644 / low 0.700 / stand 0.000; episode 0.453) | completion 0.970 | tracking 0.740 | falls 3/100 | pitch-fwd share 0.000
- step onset (side manifest, shallow 0.05-0.2): completion 0.880 | falls 12/100 | pitch-fwd share 0.000
- obstacle eval (recal2b2w 0.20-0.70): completion 0.970 | tripod 0.433 | falls 3/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_side_teacher
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: spawn yaw -pi/2 (params/env.yaml robot init_state rot)

## Phase 2b — Isaac-Crab-Hex-Side-Teacher-v0 (mode 2b) — seed2 — 2026-09-12 07:45

- status **ok** | checkpoint `2026-09-12_05-24-07/model_14997.pt` | resume `model_9998.pt` | 5000 its | wall 2.27 h
- smoke fail@2k 0.132 fail@end 0.225 coll@2k -0.004 ep_len@2k 1877.848
- exposure (non-RSI obstacle tiles): reach_edge 0.971 | reach_obst 0.941 | field_frac 0.691 (steps 1292.608) | goals_passed 1.648
- obst_coverage[1..6]: 1:0.941 2:0.857 3:0.705 4:0.488 5:0.347 6:0.224
- RSI episodes: reach_edge 0.000 | reach_obst 0.000 | goals_passed 0.000 | coverage 1:0.000 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- failure share flat 0.112 (hazard 0.059/1k) | obst 0.240 (hazard 0.133/1k) | obst-RSI 0.000 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.251
- covariates: stand_frac_actual 0.178 | spread_frac_actual 0.000 | rsi_frac_actual 0.000 | terrain_levels 5.747 | how_far 3.451 | goal_idx 1.410 | ep len 1737.456 | prints 5000
- stand time frac (logged) 0.178 | mean reward 56.954 | vloss 0.033 | mean ep len 1737.456
- slow canary (side manifest): walking-hold tripod 0.645 (gated; by hold: creep 0.645 / low 0.651 / stand 0.000; episode 0.441) | completion 0.900 | tracking 0.748 | falls 10/100 | pitch-fwd share 0.000
- step onset (side manifest, shallow 0.05-0.2): completion 0.840 | falls 16/100 | pitch-fwd share 0.000
- obstacle eval (recal2b2w 0.20-0.70): completion 0.840 | tripod 0.394 | falls 16/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_side_teacher
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: spawn yaw -pi/2 (params/env.yaml robot init_state rot)

## Phase 2c — Isaac-Crab-Hex-Side-Teacher-v0 (mode 2c) — seed2 — 2026-09-12 10:07

- status **ok** | checkpoint `2026-09-12_07-45-40/model_19996.pt` | resume `model_14997.pt` | 5000 its | wall 2.27 h
- smoke fail@2k 0.235 fail@end 0.222 coll@2k -0.005 ep_len@2k 1779.558
- exposure (non-RSI obstacle tiles): reach_edge 0.954 | reach_obst 0.918 | field_frac 0.659 (steps 1196.724) | goals_passed 1.526
- obst_coverage[1..6]: 1:0.918 2:0.810 3:0.633 4:0.451 5:0.302 6:0.174
- RSI episodes: reach_edge 0.000 | reach_obst 0.000 | goals_passed 0.000 | coverage 1:0.000 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- failure share flat 0.101 (hazard 0.053/1k) | obst 0.360 (hazard 0.210/1k) | obst-RSI 0.000 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.236
- covariates: stand_frac_actual 0.176 | spread_frac_actual 0.000 | rsi_frac_actual 0.000 | terrain_levels 6.177 | how_far 3.260 | goal_idx 1.367 | ep len 1753.362 | prints 5000
- stand time frac (logged) 0.176 | mean reward 47.946 | vloss 0.027 | mean ep len 1753.362
- slow canary (side manifest): walking-hold tripod 0.662 (gated; by hold: creep 0.662 / low 0.686 / stand 0.000; episode 0.456) | completion 0.900 | tracking 0.726 | falls 10/100 | pitch-fwd share 0.000
- step onset (side manifest, shallow 0.05-0.2): completion 0.700 | falls 30/100 | pitch-fwd share 0.000
- obstacle eval (recal2b2w 0.20-0.70): completion 0.800 | tripod 0.407 | falls 20/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_side_teacher
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: spawn yaw -pi/2 (params/env.yaml robot init_state rot)


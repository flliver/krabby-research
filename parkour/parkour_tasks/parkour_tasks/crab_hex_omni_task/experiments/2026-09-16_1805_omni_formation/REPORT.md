# Omni-formation campaign `2026-09-16_1805_omni_formation`

Plant **A15+B** (`assets/crab.usda (main)`), seed 3, phases 1a, 1b, 1c, 2a, 2b. Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_phases.py`; presets: `crab_hex_omni_phases.py` (`KRABBY_OMNI_PHASE` / `KRABBY_PLANT`).

| phase | task / mode | iterations | resume from | notes |
|---|---|---|---|---|
| 1a | `Isaac-Crab-Hex-Omni-Walk-v0` | 1000 | scratch | omni stage 1a: forward-axis walk from scratch, forward-bank RSI |
| 1b | `Isaac-Crab-Hex-Omni-Walk-v0` | 1000 | 1a | omni stage 1b: sideways-axis walk (body +y), sideways-bank RSI |
| 1c | `Isaac-Crab-Hex-Omni-Walk-v0` | 5000 | 1b | omni stage 1c: omnidirectional formation, four slots, merged bank |
| 2a | `Isaac-Crab-Hex-Omni-Teacher-v0` + mode `2a` | 5000 | 1c | omni phase 2 teacher: elements @5k, apex/airtime/stride -> half |
| 2b | `Isaac-Crab-Hex-Omni-Teacher-v0` + mode `2b` | 5000 | 2a | omni phase 2 teacher: elements @10k, satellites -> eps |

## Phase 1a — Isaac-Crab-Hex-Omni-Walk-v0 — seed3 — 2026-09-16 23:40

- status **gate_1a** | checkpoint `2026-09-16_23-11-00/model_999.pt` | resume `scratch` | 1000 its | wall 0.43 h
- smoke fail@2k 0.148 fail@end 0.148 coll@2k 0.000 ep_len@2k 1853.253
- exposure (ring): reach_edge 0.526 | reach_obst 0.359 | field_frac 0.302 (steps 594.526) | goals_passed 0.589
- terrain-promotion telemetry: track_ratio_mean 0.763 | heading_err_rms_mean 0.078 | block_steps_mean 77.894 | r_max_mean 6.339
- failure share flat 0.128 | obst 0.476 | campaign-wide 0.183
- slot fractions (logged): stand 0.230 | move 0.715 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.215 | terrain_levels 0.973 | mean ep len 1861.239 | prints 1000
- slow canary (omni manifest): walking-hold tripod 0.267 (gated; by hold: back 0.267 / diag 0.341 / fwd 0.452 / lat_l 0.376 / move_turn 0.268 / stand 0.170 / turn_l90 0.234 / turn_r90 0.254; episode 0.311) | completion 0.090 | tracking 0.376 | falls 91/100 | pitch-fwd share 0.582
- per-direction speed_ratio: fwd 0.756 / lat_l 0.124 / diag 0.273 / back 0.227
- facing by hold (final rad/settle s/yaw-progress rad/s; progress is the gated turning number, net = signed net rotation toward the target): back 0.411/2.620/-0.021 (net -0.016) | diag 0.228/5.250/-0.012 (net -0.004) | fwd 0.090/0.000/-0.007 (net -0.002) | lat_l 0.141/1.720/-0.007 (net -0.006) | move_turn 1.767/None/0.006 (net 0.006) | stand 0.036/0.000/-0.003 (net 0.002) | turn_l90 1.744/2.300/-0.001 (net -0.000) | turn_r90 0.723/2.350/0.007 (net 0.006)
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_omni_flat_walk
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: terrain-promotion telemetry logged (Metrics/base_parkour/track_ratio_mean)
- check PASS: yaw-progress reward logged (Episode_Reward/reward_yaw_progress)
- check PASS: yaw-progress telemetry logged (Metrics/base_velocity/yaw_progress)
- check PASS: omni parkour class (OmniParkourEvent) in params/env.yaml
- check PASS: omni reset func (reset_root_state_ring) in params/env.yaml
- check PASS: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check PASS: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- check PASS: dir_band matches stage 1a ((0.0, 0.0))
- check PASS: RSI source matches stage 1a (0)

## Phase 1a — gate_1a overridden — 2026-09-16 23:43

- `--continue`: model_999.pt carried into 1b unretrained (CHARTER: gate section + Log); the 1a record now reads `status ok` with `gate_1a_override`

## Phase 1b — Isaac-Crab-Hex-Omni-Walk-v0 — seed3 — 2026-09-17 00:13

- status **ok** | checkpoint `2026-09-16_23-44-00/model_1998.pt` | resume `model_999.pt` | 1000 its | wall 0.43 h
- smoke fail@2k 0.018 fail@end 0.018 coll@2k 0.000 ep_len@2k 1988.389
- exposure (ring): reach_edge 0.645 | reach_obst 0.479 | field_frac 0.167 (steps 334.537) | goals_passed 0.715
- terrain-promotion telemetry: track_ratio_mean 0.437 | heading_err_rms_mean 0.062 | block_steps_mean 106.119 | r_max_mean 2.662
- failure share flat 0.006 | obst 0.026 | campaign-wide 0.013
- slot fractions (logged): stand 0.238 | move 0.762 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.198 | terrain_levels 0.973 | mean ep len 1999.600 | prints 1000
- slow canary (omni manifest): walking-hold tripod 0.166 (gated; by hold: back 0.576 / diag 0.405 / fwd 0.660 / lat_l 0.589 / move_turn 0.166 / stand 0.412 / turn_l90 0.230 / turn_r90 0.361; episode 0.473) | completion 0.040 | tracking 0.252 | falls 96/100 | pitch-fwd share 0.573
- per-direction speed_ratio: fwd 0.162 / lat_l 0.341 / diag 0.304 / back 0.172
- facing by hold (final rad/settle s/yaw-progress rad/s; progress is the gated turning number, net = signed net rotation toward the target): back 0.094/0.000/-0.002 (net 0.002) | diag 0.077/0.000/-0.006 (net -0.000) | fwd 0.071/0.000/-0.003 (net 0.002) | lat_l 0.071/0.000/-0.001 (net 0.002) | move_turn 1.620/None/0.007 (net -0.005) | stand 0.055/0.000/-0.001 (net 0.005) | turn_l90 1.532/None/0.000 (net 0.002) | turn_r90 0.513/4.660/0.003 (net 0.004)
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_omni_flat_walk
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: terrain-promotion telemetry logged (Metrics/base_parkour/track_ratio_mean)
- check PASS: yaw-progress reward logged (Episode_Reward/reward_yaw_progress)
- check PASS: yaw-progress telemetry logged (Metrics/base_velocity/yaw_progress)
- check PASS: omni parkour class (OmniParkourEvent) in params/env.yaml
- check PASS: omni reset func (reset_root_state_ring) in params/env.yaml
- check PASS: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check PASS: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- check PASS: dir_band matches stage 1b ((1.5707963267948966, 1.5707963267948966))
- check PASS: RSI source matches stage 1b (1)

## Phase 1c — Isaac-Crab-Hex-Omni-Walk-v0 — seed3 — 2026-09-17 02:20

- status **gate_1c** | checkpoint `2026-09-17_00-13-40/model_6997.pt` | resume `model_1998.pt` | 5000 its | wall 1.93 h
- smoke fail@2k 0.030 fail@end 0.016 coll@2k 0.000 ep_len@2k 1961.969
- exposure (ring): reach_edge 0.018 | reach_obst 0.018 | field_frac 0.003 (steps 5.569) | goals_passed 0.019
- terrain-promotion telemetry: track_ratio_mean 0.241 | heading_err_rms_mean 0.099 | block_steps_mean 5.749 | r_max_mean 0.972
- failure share flat 0.023 | obst 0.025 | campaign-wide 0.008
- slot fractions (logged): stand 0.145 | move 0.537 | turn 0.058 | move_turn 0.147 | rsi_frac_actual 0.193 | terrain_levels 0.973 | mean ep len 1986.310 | prints 5000
- slow canary (omni manifest): walking-hold tripod 0.215 (gated; by hold: back 0.853 / diag 0.832 / fwd 0.315 / lat_l 0.850 / move_turn 0.215 / stand 0.000 / turn_l90 0.225 / turn_r90 0.051; episode 0.530) | completion 0.000 | tracking 0.180 | falls 100/100 | pitch-fwd share 0.340
- per-direction speed_ratio: fwd 0.159 / lat_l 0.028 / diag 0.147 / back 0.330
- facing by hold (final rad/settle s/yaw-progress rad/s; progress is the gated turning number, net = signed net rotation toward the target): back 0.021/0.000/None (net 0.002) | diag 0.026/0.000/-0.005 (net 0.001) | fwd 0.031/0.000/0.000 (net 0.002) | lat_l 0.019/0.000/-0.004 (net 0.003) | move_turn 1.878/None/-0.051 (net -0.051) | stand 0.030/0.000/0.008 (net 0.002) | turn_l90 1.536/None/0.005 (net 0.005) | turn_r90 0.884/0.000/0.020 (net 0.004)
- light ring onset (omni manifest): walking-hold tripod 0.115 (by hold: back 0.847 / diag 0.821 / fwd 0.115 / lat_l 0.832 / move_turn 0.159 / stand 0.000 / turn_l90 0.322 / turn_r90 0.087) | completion 0.010 | falls 99/100
- ring obstacle eval (0.20-0.70): completion 0.030 | walking-hold tripod 0.149 (episode 0.431) | falls 97/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_omni_flat_walk
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: terrain-promotion telemetry logged (Metrics/base_parkour/track_ratio_mean)
- check PASS: yaw-progress reward logged (Episode_Reward/reward_yaw_progress)
- check PASS: yaw-progress telemetry logged (Metrics/base_velocity/yaw_progress)
- check PASS: omni parkour class (OmniParkourEvent) in params/env.yaml
- check PASS: omni reset func (reset_root_state_ring) in params/env.yaml
- check PASS: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check PASS: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- check PASS: dir_band matches stage 1c (full range)
- check PASS: RSI source matches stage 1c (all)

## Phase 1c — Isaac-Crab-Hex-Omni-Walk-v0 — seed3 — 2026-09-18 04:37

- status **gate_1c** | checkpoint `2026-09-18_02-35-35/model_4999.pt` | resume `scratch (--from-scratch)` | 5000 its | wall 1.87 h
- smoke fail@2k 0.285 fail@end 0.333 coll@2k 0.000 ep_len@2k 1710.571
- training translation floor: track_ratio_mean (last 500 its) 0.568 (gate floor 0.6)
- exposure (ring): reach_edge 0.417 | reach_obst 0.335 | field_frac 0.114 (steps 219.379) | goals_passed 0.530
- terrain-promotion telemetry: track_ratio_mean 0.563 | heading_err_rms_mean 0.167 | block_steps_mean 62.175 | r_max_mean 1.920
- failure share flat 0.322 | obst 0.346 | campaign-wide 0.365
- slot fractions (logged): stand 0.107 | move 0.486 | turn 0.076 | move_turn 0.145 | rsi_frac_actual 0.177 | terrain_levels 0.973 | mean ep len 1656.878 | prints 5000
- slow canary (omni manifest): walking-hold tripod 0.489 (gated; by hold: back 0.579 / diag 0.489 / fwd 0.531 / lat_l 0.543 / move_turn 0.508 / stand 0.576 / turn_l90 0.542 / turn_r90 0.579; episode 0.536) | completion 0.600 | tracking 0.484 | falls 40/100 | pitch-fwd share 0.875
- per-direction speed_ratio: fwd 0.608 / lat_l 0.412 / diag 0.413 / back 0.504
- facing by hold (final rad/settle s/yaw-progress rad/s; progress is the gated turning number, net = signed net rotation toward the target): back 0.082/4.450/0.021 (net 0.030) | diag 0.219/6.080/-0.009 (net -0.006) | fwd 0.072/3.280/0.000 (net 0.008) | lat_l 0.132/5.700/-0.004 (net 0.000) | move_turn 1.135/5.920/0.039 (net 0.040) | stand 0.072/0.000/-0.002 (net 0.001) | turn_l90 0.488/11.180/0.092 (net 0.092) | turn_r90 0.089/10.290/0.098 (net 0.081)
- light ring onset (omni manifest): walking-hold tripod 0.472 (by hold: back 0.578 / diag 0.495 / fwd 0.520 / lat_l 0.560 / move_turn 0.472 / stand 0.587 / turn_l90 0.528 / turn_r90 0.587) | completion 0.560 | falls 44/100
- ring obstacle eval (0.20-0.70): completion 0.560 | walking-hold tripod 0.486 (episode 0.531) | falls 44/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_omni_flat_walk
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: terrain-promotion telemetry logged (Metrics/base_parkour/track_ratio_mean)
- check PASS: yaw-progress reward logged (Episode_Reward/reward_yaw_progress)
- check PASS: yaw-progress telemetry logged (Metrics/base_velocity/yaw_progress)
- check PASS: omni parkour class (OmniParkourEvent) in params/env.yaml
- check PASS: omni reset func (reset_root_state_ring) in params/env.yaml
- check PASS: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check PASS: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- check PASS: dir_band matches stage 1c (full range)
- check PASS: RSI source matches stage 1c (all)

## Phase 1c — gate_1c overridden — 2026-09-18 09:40

- `--continue`: model_4999.pt carried into 2a unretrained (CHARTER: gate section + Log); the 1c record now reads `status ok` with `gate_1c_override`

## Phase 2a — Isaac-Crab-Hex-Omni-Teacher-v0 (mode 2a) — seed3 — 2026-09-18 12:28

- status **gate_facing** | checkpoint `2026-09-18_09-41-10/model_9998.pt` | resume `model_4999.pt` | 5000 its | wall 2.40 h
- smoke fail@2k 0.569 fail@end 0.443 coll@2k -0.022 ep_len@2k 1401.485
- training translation floor: track_ratio_mean (last 500 its) 0.601 (gate floor 0.6)
- exposure (ring): reach_edge 0.353 | reach_obst 0.321 | field_frac 0.097 (steps 180.973) | goals_passed 0.457
- terrain-promotion telemetry: track_ratio_mean 0.601 | heading_err_rms_mean 0.308 | block_steps_mean 101.910 | r_max_mean 1.879
- failure share flat 0.458 | obst 0.503 | campaign-wide 0.474
- slot fractions (logged): stand 0.089 | move 0.360 | turn 0.127 | move_turn 0.151 | rsi_frac_actual 0.195 | terrain_levels 6.167 | mean ep len 1503.207 | prints 5000
- slow canary (omni manifest): walking-hold tripod None (gated; by hold: n/a; episode None) | completion None | tracking None | falls None/None | pitch-fwd share None
- per-direction speed_ratio: fwd None / lat_l None / diag None / back None
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_omni_teacher
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: terrain-promotion telemetry logged (Metrics/base_parkour/track_ratio_mean)
- check PASS: yaw-progress reward logged (Episode_Reward/reward_yaw_progress)
- check PASS: yaw-progress telemetry logged (Metrics/base_velocity/yaw_progress)
- check PASS: omni parkour class (OmniParkourEvent) in params/env.yaml
- check PASS: omni reset func (reset_root_state_ring) in params/env.yaml
- check PASS: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check PASS: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- check PASS: dir_band matches stage 2a (full range)
- check PASS: RSI source matches stage 2a (all)

## Phase 2a — gate_facing overridden — 2026-09-18 12:58

- `--continue`: model_9998.pt carried into 2b unretrained (CHARTER: gate section + Log); the 2a record now reads `status ok` with `gate_facing_override`

## Phase 2b — Isaac-Crab-Hex-Omni-Teacher-v0 (mode 2b) — seed3 — 2026-09-18 15:02

- status **ok** | checkpoint `2026-09-18_12-58-58/model_14997.pt` | resume `model_9998.pt` | 5000 its | wall 1.90 h
- smoke fail@2k 0.491 fail@end 0.477 coll@2k -0.023 ep_len@2k 1527.486
- training translation floor: track_ratio_mean (last 500 its) 0.645 (gate floor 0.6)
- exposure (ring): reach_edge 0.366 | reach_obst 0.353 | field_frac 0.112 (steps 209.849) | goals_passed 0.478
- terrain-promotion telemetry: track_ratio_mean 0.640 | heading_err_rms_mean 0.604 | block_steps_mean 116.870 | r_max_mean 1.944
- failure share flat 0.516 | obst 0.536 | campaign-wide 0.445
- slot fractions (logged): stand 0.094 | move 0.360 | turn 0.175 | move_turn 0.163 | rsi_frac_actual 0.205 | terrain_levels 6.237 | mean ep len 1556.589 | prints 5000
- slow canary (omni manifest): walking-hold tripod 0.377 (gated; by hold: back 0.498 / diag 0.377 / fwd 0.505 / lat_l 0.476 / move_turn 0.477 / stand 0.586 / turn_l90 0.660 / turn_r90 0.622; episode 0.516) | completion 0.500 | tracking 0.543 | falls 50/100 | pitch-fwd share 0.840
- per-direction speed_ratio: fwd 0.603 / lat_l 0.470 / diag 0.510 / back 0.603
- facing by hold (final rad/settle s/yaw-progress rad/s; progress is the gated turning number, net = signed net rotation toward the target): back 0.102/4.280/0.027 (net 0.043) | diag 0.263/3.420/-0.023 (net 0.019) | fwd 0.067/3.000/-0.000 (net 0.009) | lat_l 0.105/2.040/-0.007 (net 0.005) | move_turn 1.218/None/0.031 (net 0.031) | stand 0.053/0.000/-0.002 (net 0.001) | turn_l90 0.680/None/0.079 (net 0.079) | turn_r90 0.130/10.820/0.071 (net 0.063)
- light ring onset (omni manifest): walking-hold tripod 0.373 (by hold: back 0.533 / diag 0.373 / fwd 0.524 / lat_l 0.466 / move_turn 0.477 / stand 0.593 / turn_l90 0.657 / turn_r90 0.620) | completion 0.540 | falls 46/100
- ring obstacle eval (0.20-0.70): completion 0.540 | walking-hold tripod 0.372 (episode 0.508) | falls 46/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_omni_teacher
- check PASS: exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)
- check PASS: gait income telemetry logged (Episode_Reward/reward_clock_schedule)
- check PASS: mirror-symmetry loss active (Loss/symmetry or mirror)
- check PASS: terrain-promotion telemetry logged (Metrics/base_parkour/track_ratio_mean)
- check PASS: yaw-progress reward logged (Episode_Reward/reward_yaw_progress)
- check PASS: yaw-progress telemetry logged (Metrics/base_velocity/yaw_progress)
- check PASS: omni parkour class (OmniParkourEvent) in params/env.yaml
- check PASS: omni reset func (reset_root_state_ring) in params/env.yaml
- check PASS: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check PASS: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- check PASS: dir_band matches stage 2b (full range)
- check PASS: RSI source matches stage 2b (all)


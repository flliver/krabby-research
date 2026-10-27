# Candidate ladder `2026-09-19_facing_tracking_ladder`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 3, confirm seed 2, 8000 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `4c3f99c9a2c5f2799a9fb3ae425b70495df31cc97674543d4d4d1637974a4d11` · `candidates.yaml` sha256 `7303819ea5cf6895d82f9018c99058fa4bd70819aca31e5f443764116c839027`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | B0 | baseline | scratch_1c | the incumbent recipe at equal compute | — |
| 1 | C1 | curriculum | curriculum | `{"chain":"curriculum"}` | `—` |
| 2 | C1b | curriculum | curriculum | `{"ang_scale":{"*_turn":0.3}}` | `decision(C1).chosen == REJECT and (decision(C1).floor_stage is None or decision(C1).floor_stage >= c4)` |
| 3 | C1c | curriculum | scratch_c6 | `{"chain":"scratch_c6"}` | `decision(C1).chosen == ADOPT` |
| 4 | C2 | reward | effective | `{"KRABBY_YAW_PROGRESS_W":"3.0"}` | `—` |
| 5 | C3a | kinematics | effective | `{"KRABBY_OMNI_ANG_VEL_MAX":"0.21","KRABBY_OMNI_CLOCK_TURN_R":"1.72"}` | `—` |
| 6 | C3b | kinematics | effective | `{"KRABBY_OMNI_ANG_VEL_CLIP":"0.12","KRABBY_OMNI_ANG_VEL_MAX":"0.18","KRABBY_OMNI_CLOCK_TURN_R":"2.0"}` | `decision(C3a).chosen != REJECT or decision(C3a).pooled_confidence in (uncertain, disputed)` |
| 7 | C4 | reward | effective | `{"KRABBY_OMNI_FACING_TURN_W":"0.8"}` | `—` |
| 8 | C5 | observation | effective | `{"KRABBY_OMNI_FACING_OBS_ENCODING":"sincos"}` | `—` |
| 9 | C6 | reward | effective | `{"KRABBY_OMNI_TRACK_FACING_SIGMA":"1.0"}` | `—` |
| 10 | C7 | kinematics | effective | `{"ang_scale":{"*_turn":0.3}}` | `recipe.has_pool and decision(C1b).chosen != ADOPT` |
| 11 | C8 | economy | effective | `{"curriculum":{"pool_edit":{"renormalise":true,"set_share":{"stand":0.05}}},"scratch_1c":{"overlay":{"KRABBY_OMNI_SLOTS":"0.05:0.65:0.1:0.2"},"stages":"all"},"scratch_c6":{"pool_edit":{"renormalise":true,"set_share":{"stand":0.05}}}}` | `—` |
| 12 | C9 | economy | effective | `{"KRABBY_OMNI_PROGRESS_W":"1.5"}` | `—` |
| 13 | C10 | seeding | effective | `{"KRABBY_OMNI_RSI_TURN_SOURCE":"2"}` | `screen.yaw_progress_turn_min < 0.10` |
| 14 | C11 | economy | effective | `{"KRABBY_OMNI_ANG_TURN_ONLY":"1"}` | `screen.track_ratio_move_tail < 0.45` |
| 15 | C12 | task_definition | effective | `{"KRABBY_OMNI_TRAVEL_FRAME":"body"}` | `—` |
| — | M1 | milestone | — | promote ['2a', '2b'] after C3b | — |
| — | M2 | milestone | — | promote ['2a', '2b'] after C7 | — |
| — | M3 | milestone | — | promote ['2a', '2b'] after C12 | — |

### X01:1c — B0 s3 (baseline) — 2026-09-19T14:15:50

- recipe `43dce3047225` on chain `scratch_1c`, base `43dce3047225`, code_sha `4c3f99c9a2c5`
- overlay `{}`
- status **ok** | checkpoint `2026-09-19_11-12-21_seed3_X01_1c_A15pB/model_7999.pt` | resume `scratch (--from-scratch)` | 8000 its | wall 3.00 h
- smoke fail@2k 0.321 fail@end 0.448 coll@2k 0.000 ep_len@2k 1702.670
- training translation floor: track_ratio_mean (last 500 its) 0.637 (gate floor 0.6)
- exposure (ring): reach_edge 0.462 | reach_obst 0.338 | field_frac 0.133 (steps 249.667) | goals_passed 0.616
- terrain-promotion telemetry: track_ratio_mean 0.636 (move rows only 0.636) | heading_err_rms_mean 0.174 | block_steps_mean 71.605 | r_max_mean 2.205
- failure share flat 0.391 | obst 0.468 | campaign-wide 0.425
- slot fractions (logged): stand 0.145 | move 0.472 | turn 0.073 | move_turn 0.179 | rsi_frac_actual 0.207 | terrain_levels 0.973 | mean ep len 1666.261 | prints 8000
- slow canary (omni manifest): walking-hold tripod 0.384 (gated; by hold: back 0.604 / diag 0.384 / fwd 0.555 / lat_l 0.453 / move_turn 0.467 / stand 0.632 / turn_l90 0.598 / turn_r90 0.630; episode 0.545) | completion 0.560 | tracking 0.561 | falls 44/100 | pitch-fwd share 0.818
- per-direction speed_ratio: fwd 0.602 / lat_l 0.542 / diag 0.553 / back 0.611
- facing by hold (final rad/settle s/yaw-progress rad/s; progress is the gated turning number, net = signed net rotation toward the target): back 0.084/3.010/0.014 (net 0.021) | diag 0.193/3.420/-0.016 (net -0.002) | fwd 0.097/3.900/0.001 (net 0.009) | lat_l 0.091/2.600/0.000 (net 0.008) | move_turn 1.295/None/0.024 (net 0.024) | stand 0.089/2.650/-0.003 (net 0.002) | turn_l90 0.724/11.120/0.071 (net 0.072) | turn_r90 0.146/8.160/0.070 (net 0.062)
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
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.384 (need >= 0.4; by hold: back 0.604, diag 0.384, fwd 0.555, lat_l 0.453, move_turn 0.467, stand 0.632, turn_l90 0.598, turn_r90 0.630), completion 0.560 (need >= 0.7), yaw-progress rate turn_l90 0.071, turn_r90 0.070 rad/s (net turn_l90 0.072, turn_r90 0.062) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.542, diag 0.553, back 0.611, move_turn 0.441 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.637 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- floor PASS: dead_walker floor: track_ratio_mean 0.563 over its 800-1000 (need >= 0.350)

>>> ENTRY X01:1c

### experiment X01 — B0 s3 — ok — 2026-09-19T14:15:50

- recipe `43dce3047225` base `43dce3047225` chain `scratch_1c` seed 3
- stages ['X01:1c']
- screen: completion 0.560, tripod_dir 0.384, track tail 0.637
- replay: none

>>> ENTRY X01

### X02:1c — B0 s2 (baseline) — 2026-09-19T17:19:28

- recipe `43dce3047225` on chain `scratch_1c`, base `43dce3047225`, code_sha `4c3f99c9a2c5`
- overlay `{}`
- status **ok** | checkpoint `2026-09-19_14-16-00_seed2_X02_1c_A15pB/model_7999.pt` | resume `scratch (--from-scratch)` | 8000 its | wall 3.00 h
- smoke fail@2k 0.255 fail@end 0.249 coll@2k 0.000 ep_len@2k 1703.001
- training translation floor: track_ratio_mean (last 500 its) 0.615 (gate floor 0.6)
- exposure (ring): reach_edge 0.512 | reach_obst 0.543 | field_frac 0.169 (steps 328.835) | goals_passed 0.700
- terrain-promotion telemetry: track_ratio_mean 0.611 (move rows only 0.614) | heading_err_rms_mean 0.159 | block_steps_mean 115.304 | r_max_mean 2.227
- failure share flat 0.261 | obst 0.263 | campaign-wide 0.237
- slot fractions (logged): stand 0.182 | move 0.461 | turn 0.156 | move_turn 0.144 | rsi_frac_actual 0.199 | terrain_levels 1.027 | mean ep len 1795.719 | prints 8000
- slow canary (omni manifest): walking-hold tripod 0.470 (gated; by hold: back 0.561 / diag 0.472 / fwd 0.589 / lat_l 0.504 / move_turn 0.470 / stand 0.663 / turn_l90 0.605 / turn_r90 0.660; episode 0.562) | completion 0.680 | tracking 0.510 | falls 32/100 | pitch-fwd share 0.781
- per-direction speed_ratio: fwd 0.628 / lat_l 0.462 / diag 0.542 / back 0.583
- facing by hold (final rad/settle s/yaw-progress rad/s; progress is the gated turning number, net = signed net rotation toward the target): back 0.097/4.260/0.005 (net 0.021) | diag 0.177/4.600/-0.009 (net 0.012) | fwd 0.087/2.220/-0.002 (net 0.005) | lat_l 0.092/4.350/-0.000 (net 0.004) | move_turn 1.029/1.280/0.054 (net 0.054) | stand 0.074/0.000/-0.003 (net 0.003) | turn_l90 0.549/11.810/0.081 (net 0.081) | turn_r90 0.108/9.880/0.084 (net 0.074)
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
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.470 (need >= 0.4; by hold: back 0.561, diag 0.472, fwd 0.589, lat_l 0.504, move_turn 0.470, stand 0.663, turn_l90 0.605, turn_r90 0.660), completion 0.680 (need >= 0.7), yaw-progress rate turn_l90 0.081, turn_r90 0.084 rad/s (net turn_l90 0.081, turn_r90 0.074) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.462, diag 0.542, back 0.583, move_turn 0.353 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.615 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- floor PASS: dead_walker floor: track_ratio_mean 0.552 over its 800-1000 (need >= 0.422)

>>> ENTRY X02:1c

### experiment X02 — B0 s2 — ok — 2026-09-19T17:19:28

- recipe `43dce3047225` base `43dce3047225` chain `scratch_1c` seed 2
- stages ['X02:1c']
- screen: completion 0.680, tripod_dir 0.472, track tail 0.615
- replay: none

>>> ENTRY X02

### X03:c1 — C1 s3 (candidate) — 2026-09-19T17:47:28

- recipe `241c786f0905` on chain `curriculum`, base `43dce3047225`, code_sha `4c3f99c9a2c5`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,fwd:0.8', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': '0', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-19_17-19-39_seed3_X03_c1_A15pB/model_999.pt` | resume `scratch (--from-scratch)` | 1000 its | wall 0.40 h
- smoke fail@2k 0.501 fail@end 0.501 coll@2k 0.000 ep_len@2k 1571.309
- training translation floor: track_ratio_mean (last 500 its) 0.664 (gate floor 0.6)
- exposure (ring): reach_edge 0.347 | reach_obst 0.243 | field_frac 0.171 (steps 309.147) | goals_passed 0.377
- terrain-promotion telemetry: track_ratio_mean 0.720 (move rows only 0.720) | heading_err_rms_mean 0.079 | block_steps_mean 59.770 | r_max_mean 4.910
- failure share flat 0.413 | obst 0.731 | campaign-wide 0.448
- slot fractions (logged): stand 0.155 | move 0.606 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.210 | terrain_levels 0.973 | mean ep len 1616.915 | prints 1000
- slow canary (omni manifest): walking-hold tripod None (gated; by hold: n/a; episode None) | completion None | tracking None | falls None/None | pitch-fwd share None
- per-direction speed_ratio: fwd None / lat_l None / diag None / back None
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
- curriculum: completion 0.380, tripod_dir 0.434; survival stand 1.000, fwd 0.810, lat_l 0.780, lat_r 0.750, back 0.630, turn_l90 0.590, turn_r90 0.570, move_turn 0.450, move_turn_r 0.400, diag 0.380; ratio back 0.597, diag 0.471, fwd 0.622, lat_l 0.194, lat_r 0.216, move_turn 0.420, move_turn_r 0.574
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,fwd:0.8 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: survival through fwd on curriculum: 0.810 (need >= 0.3)

>>> ENTRY X03:c1

### X03:c2 — C1 s3 (candidate) — 2026-09-19T18:14:59

- recipe `241c786f0905` on chain `curriculum`, base `43dce3047225`, code_sha `4c3f99c9a2c5`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,fwd:0.3,side_l:0.2,side_r:0.2,back:0.1', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-19_17-47-39_seed3_X03_c2_A15pB/model_1998.pt` | resume `model_999.pt` | 1000 its | wall 0.40 h
- smoke fail@2k 0.403 fail@end 0.403 coll@2k 0.000 ep_len@2k 1683.680
- training translation floor: track_ratio_mean (last 500 its) 0.567 (gate floor 0.6)
- exposure (ring): reach_edge 0.588 | reach_obst 0.458 | field_frac 0.228 (steps 426.999) | goals_passed 0.706
- terrain-promotion telemetry: track_ratio_mean 0.594 (move rows only 0.594) | heading_err_rms_mean 0.100 | block_steps_mean 89.747 | r_max_mean 2.509
- failure share flat 0.366 | obst 0.336 | campaign-wide 0.327
- slot fractions (logged): stand 0.241 | move 0.641 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.214 | terrain_levels 0.973 | mean ep len 1747.189 | prints 1000
- slow canary (omni manifest): walking-hold tripod None (gated; by hold: n/a; episode None) | completion None | tracking None | falls None/None | pitch-fwd share None
- per-direction speed_ratio: fwd None / lat_l None / diag None / back None
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
- curriculum: completion 0.000, tripod_dir None; survival stand 0.980, fwd 0.710, lat_l 0.670, lat_r 0.620, back 0.360, turn_l90 0.270, turn_r90 0.240, move_turn 0.070, move_turn_r 0.000, diag 0.000; ratio back 0.532, fwd 0.676, lat_l 0.365, lat_r 0.368, move_turn 0.411, move_turn_r 0.601
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,fwd:0.3,side_l:0.2,side_r:0.2,back:0.1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c2: no incumbent trace yet -- this run seeds it (10 holds recorded)
- floor PASS: survival through back on curriculum: 0.360 (need >= 0.3)

>>> ENTRY X03:c2

### X03:c3 — C1 s3 (candidate) — 2026-09-19T18:38:59

- recipe `241c786f0905` on chain `curriculum`, base `43dce3047225`, code_sha `4c3f99c9a2c5`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-19_18-15-09_seed3_X03_c3_A15pB/model_2997.pt` | resume `model_1998.pt` | 1000 its | wall 0.40 h
- smoke fail@2k 0.231 fail@end 0.231 coll@2k 0.000 ep_len@2k 1816.952
- training translation floor: track_ratio_mean (last 500 its) 0.623 (gate floor 0.6)
- exposure (ring): reach_edge 0.540 | reach_obst 0.422 | field_frac 0.199 (steps 372.691) | goals_passed 0.712
- terrain-promotion telemetry: track_ratio_mean 0.622 (move rows only 0.622) | heading_err_rms_mean 0.149 | block_steps_mean 80.947 | r_max_mean 2.492
- failure share flat 0.250 | obst 0.329 | campaign-wide 0.176
- slot fractions (logged): stand 0.107 | move 0.668 | turn 0.157 | move_turn 0.000 | rsi_frac_actual 0.248 | terrain_levels 0.973 | mean ep len 1872.151 | prints 1000
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
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c3: no incumbent trace yet -- this run seeds it (0 holds recorded)
- floor PASS: no floor at this stage

>>> ENTRY X03:c3

### X04:c1 — C1b s3 (candidate) — 2026-09-19T19:47:00

- recipe `e6d6786d1394` on chain `curriculum`, base `43dce3047225`, code_sha `4c3f99c9a2c5`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,fwd:0.8', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': '0', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-19_19-19-10_seed3_X04_c1_A15pB/model_999.pt` | resume `scratch (--from-scratch)` | 1000 its | wall 0.40 h
- smoke fail@2k 0.501 fail@end 0.501 coll@2k 0.000 ep_len@2k 1571.309
- training translation floor: track_ratio_mean (last 500 its) 0.664 (gate floor 0.6)
- exposure (ring): reach_edge 0.347 | reach_obst 0.243 | field_frac 0.171 (steps 309.147) | goals_passed 0.377
- terrain-promotion telemetry: track_ratio_mean 0.720 (move rows only 0.720) | heading_err_rms_mean 0.079 | block_steps_mean 59.770 | r_max_mean 4.910
- failure share flat 0.413 | obst 0.731 | campaign-wide 0.448
- slot fractions (logged): stand 0.155 | move 0.606 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.210 | terrain_levels 0.973 | mean ep len 1616.915 | prints 1000
- slow canary (omni manifest): walking-hold tripod None (gated; by hold: n/a; episode None) | completion None | tracking None | falls None/None | pitch-fwd share None
- per-direction speed_ratio: fwd None / lat_l None / diag None / back None
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
- curriculum: completion 0.380, tripod_dir 0.434; survival stand 1.000, fwd 0.810, lat_l 0.780, lat_r 0.750, back 0.630, turn_l90 0.590, turn_r90 0.570, move_turn 0.450, move_turn_r 0.400, diag 0.380; ratio back 0.597, diag 0.471, fwd 0.622, lat_l 0.194, lat_r 0.216, move_turn 0.420, move_turn_r 0.574
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,fwd:0.8 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: survival through fwd on curriculum: 0.810 (need >= 0.3)

>>> ENTRY X04:c1


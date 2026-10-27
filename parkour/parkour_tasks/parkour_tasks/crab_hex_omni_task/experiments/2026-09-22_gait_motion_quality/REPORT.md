# Candidate ladder `2026-09-22_gait_motion_quality`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 3, confirm seed 2, 2000 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `8842c1a505d56cae0da241667e1ccb35923e3aff53d9e5165d11f73df984a241` · `candidates.yaml` sha256 `65c6ce996cf93d4d81329eeda2afded855f7b93ce7f1163705929b52d978160e`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | X0 | baseline | fork_c6 | the incumbent recipe at equal compute | — |
| 1 | M1 | exploration | effective | `{"KRABBY_ACTION_STD_MAX":"0.5","KRABBY_ACTION_STD_RESET":"0.4"}` | `—` |
| 2 | M2 | reward | effective | `{"KRABBY_POWER_W":"-0.00023"}` | `—` |
| 3 | M3 | reward | effective | `{"KRABBY_ACTION_RATE_SQ":"1","KRABBY_ACTION_RATE_W":"-0.14","KRABBY_ACTION_SMOOTH2_W":"-0.024","KRABBY_JOINT_VEL_W":"-0.049"}` | `—` |
| 4 | M4 | reward | effective | `{"KRABBY_FOOT_SLIP_W":"-0.27"}` | `—` |
| 5 | M5 | command | effective | `{"KRABBY_OMNI_RELAY_HYST":"1","KRABBY_OMNI_STAND_HOLD":"1"}` | `—` |
| 6 | M6 | stand | effective | `{"KRABBY_OMNI_STAND_PARK":"1"}` | `—` |
| 7 | M7 | action | effective | `{"KRABBY_ACTION_LPF_HZ":"4"}` | `—` |

### X01:c6ft — X0 s3 (baseline) — 2026-09-23T03:31:56

- recipe `979629245c00` on chain `fork_c6`, base `979629245c00`, code_sha `8842c1a505d5`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-23_02-38-36_seed3_X01_c6ft_A15pB/model_9995.pt` | resume `model_7996.pt` | 2000 its | wall 0.77 h
- smoke fail@2k 0.249 fail@end 0.249 coll@2k 0.000 ep_len@2k 1785.655
- training translation floor: track_ratio_mean (last 500 its) 0.620 (gate floor 0.6)
- exposure (ring): reach_edge 0.586 | reach_obst 0.485 | field_frac 0.214 (steps 408.410) | goals_passed 0.783
- terrain-promotion telemetry: track_ratio_mean 0.614 (move rows only 0.625) | heading_err_rms_mean 0.164 | block_steps_mean 101.240 | r_max_mean 2.405
- failure share flat 0.272 | obst 0.290 | campaign-wide 0.209
- slot fractions (logged): stand 0.146 | move 0.510 | turn 0.087 | move_turn 0.181 | rsi_frac_actual 0.202 | terrain_levels 0.973 | mean ep len 1829.904 | prints 2000
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
- suite trained categories: decisive stand, fwd, side, back, diag, turn, move_turn
- suite stand: n 100 surv 0.990 drift 0.089 [decisive]
- suite fwd: n 100 surv 0.950 ratio 0.724 tripod 0.599 [decisive]
- suite side: n 200 surv 0.975 ratio 0.422 tripod 0.482 (l: surv 0.980 ratio 0.428 tripod 0.487; r: surv 0.970 ratio 0.419 tripod 0.479) [decisive]
- suite back: n 100 surv 0.930 ratio 0.734 tripod 0.613 [decisive]
- suite diag: n 100 surv 0.860 ratio 0.533 tripod 0.412 [decisive]
- suite turn: n 200 surv 0.995 closed 0.610 progress 0.081 (l: surv 0.990 closed 0.605 progress 0.080; r: surv 1.000 closed 0.620 progress 0.081) [decisive]
- suite move_turn: n 200 surv 0.935 closed 0.411 progress 0.054 ratio 0.504 (l: surv 0.940 closed 0.436 progress 0.057 ratio 0.501; r: surv 0.930 closed 0.397 progress 0.051 ratio 0.510) [decisive]
- WATCH gate_1c: PASS -- formation gate passed — walking-hold tripod 0.412 (by hold: back 0.613, diag 0.412, fwd 0.599, lat_l 0.487, lat_r 0.479, move_turn 0.566, move_turn_r 0.559, settle 0.406, stand 0.583, turn_l90 0.621, turn_r90 0.622), completion 0.954, yaw-progress rate turn_l90 0.080, turn_r90 0.081 rad/s (net turn_l90 0.080, turn_r90 0.081), per-direction speed_ratio lat_l 0.428, diag 0.533, back 0.734, move_turn 0.501 on slow__omni, training track_ratio_mean 0.620 (need >= 0.6 over the last 500 its)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.990 (bound 1.013, n_env 100, min >= 0.79); fwd 0.950 (bound 1.001, n_env 100, min >= 0.72); side 0.975 (bound 1.011, n_env 100, min >= 0.77); back 0.930 (bound 0.989, n_env 100, min >= 0.72); diag 0.860 (bound 0.941, n_env 100, min >= 0.59); turn 0.995 (bound 1.011, n_env 100, min >= 0.75); move_turn 0.935 (bound 0.992, n_env 100, min >= 0.55)

>>> ENTRY X01:c6ft

### experiment X01 — X0 s3 — ok — 2026-09-23T03:31:56

- recipe `979629245c00` base `979629245c00` chain `fork_c6` seed 3
- stages ['X01:c6ft']
- screen: completion 0.954, tripod_dir 0.412, track tail 0.620
- replay: none

>>> ENTRY X01

### X02:c6ft — X0 s2 (baseline) — 2026-09-23T04:25:27

- recipe `979629245c00` on chain `fork_c6`, base `979629245c00`, code_sha `8842c1a505d5`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-23_03-32-07_seed2_X02_c6ft_A15pB/model_9995.pt` | resume `model_7996.pt` | 2000 its | wall 0.77 h
- smoke fail@2k 0.364 fail@end 0.364 coll@2k 0.000 ep_len@2k 1662.927
- training translation floor: track_ratio_mean (last 500 its) 0.615 (gate floor 0.6)
- exposure (ring): reach_edge 0.541 | reach_obst 0.465 | field_frac 0.186 (steps 348.724) | goals_passed 0.706
- terrain-promotion telemetry: track_ratio_mean 0.610 (move rows only 0.617) | heading_err_rms_mean 0.171 | block_steps_mean 97.900 | r_max_mean 2.265
- failure share flat 0.343 | obst 0.351 | campaign-wide 0.376
- slot fractions (logged): stand 0.170 | move 0.479 | turn 0.055 | move_turn 0.140 | rsi_frac_actual 0.183 | terrain_levels 1.027 | mean ep len 1582.366 | prints 2000
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
- suite trained categories: decisive stand, fwd, side, back, diag, turn, move_turn
- suite stand: n 100 surv 0.970 drift 0.091 [decisive]
- suite fwd: n 100 surv 0.830 ratio 0.749 tripod 0.557 [decisive]
- suite side: n 200 surv 0.965 ratio 0.408 tripod 0.497 (l: surv 0.950 ratio 0.407 tripod 0.498; r: surv 0.980 ratio 0.410 tripod 0.497) [decisive]
- suite back: n 100 surv 0.900 ratio 0.726 tripod 0.583 [decisive]
- suite diag: n 100 surv 0.630 ratio 0.528 tripod 0.366 [decisive]
- suite turn: n 200 surv 0.955 closed 0.508 progress 0.068 (l: surv 0.970 closed 0.508 progress 0.068; r: surv 0.940 closed 0.508 progress 0.066) [decisive]
- suite move_turn: n 200 surv 0.840 closed 0.142 progress 0.020 ratio 0.470 (l: surv 0.840 closed 0.135 progress 0.019 ratio 0.452; r: surv 0.840 closed 0.154 progress 0.020 ratio 0.491) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.366 (need >= 0.4; by hold: back 0.583, diag 0.366, fwd 0.557, lat_l 0.498, lat_r 0.497, move_turn 0.487, move_turn_r 0.502, settle 0.355, stand 0.544, turn_l90 0.561, turn_r90 0.548), completion 0.885 (need >= 0.7), yaw-progress rate turn_l90 0.068, turn_r90 0.066 rad/s (net turn_l90 0.068, turn_r90 0.066) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.407, diag 0.528, back 0.726, move_turn 0.452 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.615 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.970 (bound 1.010, n_env 100, min >= 0.79); fwd 0.830 (bound 0.917, n_env 100, min >= 0.72); side 0.965 (bound 1.008, n_env 100, min >= 0.77); back 0.900 (bound 0.970, n_env 100, min >= 0.72); diag 0.630 (bound 0.742, n_env 100, min >= 0.59); turn 0.955 (bound 1.003, n_env 100, min >= 0.75); move_turn 0.840 (bound 0.925, n_env 100, min >= 0.55)

>>> ENTRY X02:c6ft

### experiment X02 — X0 s2 — ok — 2026-09-23T04:25:27

- recipe `979629245c00` base `979629245c00` chain `fork_c6` seed 2
- stages ['X02:c6ft']
- screen: completion 0.885, tripod_dir 0.366, track tail 0.615
- replay: none

>>> ENTRY X02

### X03:c6ft — M1 s3 (candidate) — 2026-09-23T05:18:58

- recipe `1cbea4c2c35c` on chain `fork_c6`, base `979629245c00`, code_sha `8842c1a505d5`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_ACTION_STD_RESET': '0.4', 'KRABBY_ACTION_STD_MAX': '0.5'}`
- status **ok** | checkpoint `2026-09-23_04-25-38_seed3_X03_c6ft_A15pB/model_9995.pt` | resume `model_7996.pt` | 2000 its | wall 0.77 h
- smoke fail@2k 0.422 fail@end 0.422 coll@2k 0.000 ep_len@2k 1585.685
- training translation floor: track_ratio_mean (last 500 its) 0.611 (gate floor 0.6)
- exposure (ring): reach_edge 0.540 | reach_obst 0.445 | field_frac 0.197 (steps 371.978) | goals_passed 0.715
- terrain-promotion telemetry: track_ratio_mean 0.615 (move rows only 0.621) | heading_err_rms_mean 0.167 | block_steps_mean 89.383 | r_max_mean 2.278
- failure share flat 0.375 | obst 0.330 | campaign-wide 0.442
- slot fractions (logged): stand 0.115 | move 0.450 | turn 0.099 | move_turn 0.143 | rsi_frac_actual 0.196 | terrain_levels 0.973 | mean ep len 1579.760 | prints 2000
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
- suite trained categories: decisive stand, fwd, side, back, diag, turn, move_turn
- suite stand: n 100 surv 0.970 drift 0.083 [decisive]
- suite fwd: n 100 surv 0.930 ratio 0.695 tripod 0.637 [decisive]
- suite side: n 200 surv 0.965 ratio 0.430 tripod 0.474 (l: surv 0.950 ratio 0.430 tripod 0.475; r: surv 0.980 ratio 0.431 tripod 0.468) [decisive]
- suite back: n 100 surv 0.920 ratio 0.681 tripod 0.631 [decisive]
- suite diag: n 100 surv 0.770 ratio 0.541 tripod 0.392 [decisive]
- suite turn: n 200 surv 0.975 closed 0.510 progress 0.068 (l: surv 0.990 closed 0.495 progress 0.067; r: surv 0.960 closed 0.522 progress 0.069) [decisive]
- suite move_turn: n 200 surv 0.850 closed 0.301 progress 0.042 ratio 0.433 (l: surv 0.880 closed 0.286 progress 0.040 ratio 0.415; r: surv 0.820 closed 0.309 progress 0.044 ratio 0.446) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.392 (need >= 0.4; by hold: back 0.631, diag 0.392, fwd 0.637, lat_l 0.475, lat_r 0.468, move_turn 0.509, move_turn_r 0.508, settle 0.489, stand 0.601, turn_l90 0.567, turn_r90 0.543), completion 0.917 (need >= 0.7), yaw-progress rate turn_l90 0.067, turn_r90 0.069 rad/s (net turn_l90 0.067, turn_r90 0.069) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.430, diag 0.541, back 0.681, move_turn 0.415 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.611 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_ACTION_STD_MAX=0.5 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_STD_RESET=0.4 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.970 (bound 1.010, n_env 100, min >= 0.79); fwd 0.930 (bound 0.989, n_env 100, min >= 0.72); side 0.965 (bound 1.008, n_env 100, min >= 0.77); back 0.920 (bound 0.983, n_env 100, min >= 0.72); diag 0.770 (bound 0.868, n_env 100, min >= 0.59); turn 0.975 (bound 1.011, n_env 100, min >= 0.75); move_turn 0.850 (bound 0.933, n_env 100, min >= 0.55)

>>> ENTRY X03:c6ft

## D05 — M1 — REJECT (clear, decided) — 2026-09-23T05:18:58

- question: Is the saturated exploration noise the root of the extra motion (plan cause 1)? X08:c6's raw std sits at 2.0001-2.0004 on all 18 dimensions, at the 2.0 clamp whose gradient is zero, so the action-rate penalty is near a constant and the policy trains amid violent dither. Reset the std to 0.4 on the fork and clamp it at 0.5 (the literature's best is sigma 0.1-0.5). The reset sits BELOW the clamp on purpose: the step-2 smoke measured that a reset EQUAL to the clamp drifts straight back above it, where the clamp's gradient is zero again and the std is pinned at the ceiling -- the saturation the knob exists to undo.
- experiment `X03` recipe `1cbea4c2c35c` base `979629245c00` control `X01`
- why: WORSE on block Q, guard cat.back.slip_per_m, guard cat.move_turn.ratio
- block F (F): z -2.020 vs R 2.073 -> TIE (k=2, missing [])
  - cat.turn.facing_closed: 0.610 -> 0.510 (delta -0.100, R_eval 0.048, R_seed 0.102, z -2.066)
  - cat.turn.yaw_progress: 0.081 -> 0.068 (delta -0.013, R_eval 0.007, R_seed 0.013, z -1.975)
- block F_mt (F_mt): z -2.314 vs R 6.049 -> TIE (k=2, missing [])
  - cat.move_turn.facing_closed: 0.411 -> 0.301 (delta -0.111, R_eval 0.042, R_seed 0.269, z -2.651)
  - cat.move_turn.yaw_progress: 0.054 -> 0.042 (delta -0.012, R_eval 0.006, R_seed 0.034, z -1.977)
- block T (T): z -0.247 vs R 0.517 -> TIE (k=4, missing [])
  - cat.fwd.ratio: 0.724 -> 0.695 (delta -0.029, R_eval 0.041, R_seed 0.025, z -0.721)
  - cat.side.ratio: 0.422 -> 0.430 (delta 0.008, R_eval 0.017, R_seed 0.014, z 0.451)
  - cat.back.ratio: 0.734 -> 0.681 (delta -0.054, R_eval 0.054, R_seed 0.009, z -0.988)
  - cat.diag.ratio: 0.533 -> 0.541 (delta 0.007, R_eval 0.026, R_seed 0.006, z 0.271)
- block Q (Q): z -1.358 vs R 0.319 -> WORSE (k=41, missing [])
  - cat.stand.action_jitter: 0.131 -> 0.145 (lower is better, signed delta -0.015, R_eval 0.005, R_seed 0.002, z -3.125)
  - cat.stand.action_fft_freq: 1.921 -> 2.019 (lower is better, signed delta -0.098, R_eval 0.043, R_seed 0.001, z -2.259)
  - cat.stand.joint_speed_mean: 0.620 -> 0.609 (lower is better, signed delta 0.011, R_eval 0.018, R_seed 0.013, z 0.594)
  - cat.stand.rod_speed_mean: 0.020 -> 0.021 (lower is better, signed delta -0.002, R_eval 0.000, R_seed 0.000, z -7.348)
  - cat.stand.contact_switch_hz: 1.681 -> 1.660 (lower is better, signed delta 0.021, R_eval 0.101, R_seed 0.014, z 0.207)
  - cat.stand.foot_slip: 0.103 -> 0.107 (lower is better, signed delta -0.004, R_eval 0.004, R_seed 0.002, z -0.921)
  - cat.stand.wz_active_frac: 0.114 -> 0.072 (lower is better, signed delta 0.042, R_eval 0.076, R_seed 0.017, z 0.557)
  - cat.stand.clock_restarts: 3.000 -> 2.000 (lower is better, signed delta 1.000, R_eval 1.250, R_seed 0.000, z 0.800)
  - cat.stand.six_down_frac: 0.003 -> 0.003 (delta 0.000, R_eval 0.002, R_seed 0.000, z 0.000)
  - cat.stand.foot_disp_end: 0.262 -> 0.259 (lower is better, signed delta 0.004, R_eval 0.021, R_seed 0.016, z 0.175)
  - cat.stand.work_per_s: 363.747 -> 377.953 (lower is better, signed delta -14.206, R_eval 13.227, R_seed 6.164, z -1.074)
  - cat.fwd.action_jitter: 0.119 -> 0.133 (lower is better, signed delta -0.015, R_eval 0.004, R_seed 0.005, z -4.069)
  - cat.fwd.swings_per_m: 27.824 -> 28.896 (lower is better, signed delta -1.071, R_eval 2.694, R_seed 0.097, z -0.398)
  - cat.fwd.slip_per_m: 2.182 -> 2.308 (lower is better, signed delta -0.127, R_eval 0.210, R_seed 0.090, z -0.603)
  - cat.fwd.work_per_m: 1416.231 -> 1506.145 (lower is better, signed delta -89.913, R_eval 102.428, R_seed 29.355, z -0.878)
  - cat.fwd.cam_work_per_m: 1064.578 -> 1098.478 (lower is better, signed delta -33.899, R_eval 145.997, R_seed 117.106, z -0.232)
  - cat.side.action_jitter: 0.105 -> 0.113 (lower is better, signed delta -0.008, R_eval 0.002, R_seed 0.006, z -3.898)
  - cat.side.swings_per_m: 69.159 -> 69.523 (lower is better, signed delta -0.363, R_eval 3.202, R_seed 0.389, z -0.113)
  - cat.side.slip_per_m: 6.839 -> 6.741 (lower is better, signed delta 0.098, R_eval 0.204, R_seed 0.141, z 0.479)
  - cat.side.work_per_m: 3489.403 -> 3462.094 (lower is better, signed delta 27.309, R_eval 84.868, R_seed 44.029, z 0.322)
  - cat.side.cam_work_per_m: 1858.015 -> 1751.599 (lower is better, signed delta 106.416, R_eval 77.988, R_seed 50.792, z 1.365)
  - cat.back.action_jitter: 0.119 -> 0.134 (lower is better, signed delta -0.014, R_eval 0.003, R_seed 0.002, z -4.114)
  - cat.back.swings_per_m: 30.147 -> 33.398 (lower is better, signed delta -3.250, R_eval 3.122, R_seed 0.806, z -1.041)
  - cat.back.slip_per_m: 2.575 -> 2.782 (lower is better, signed delta -0.206, R_eval 0.156, R_seed 0.018, z -1.326)
  - cat.back.work_per_m: 1632.611 -> 1752.375 (lower is better, signed delta -119.764, R_eval 124.566, R_seed 7.862, z -0.961)
  - cat.back.cam_work_per_m: 1281.122 -> 1354.485 (lower is better, signed delta -73.363, R_eval 151.190, R_seed 11.675, z -0.485)
  - cat.diag.action_jitter: 0.113 -> 0.125 (lower is better, signed delta -0.012, R_eval 0.004, R_seed 0.005, z -2.717)
  - cat.diag.swings_per_m: 52.545 -> 53.675 (lower is better, signed delta -1.130, R_eval 4.559, R_seed 1.441, z -0.248)
  - cat.diag.slip_per_m: 4.632 -> 4.832 (lower is better, signed delta -0.200, R_eval 0.272, R_seed 0.083, z -0.735)
  - cat.diag.work_per_m: 2500.881 -> 2559.557 (lower is better, signed delta -58.676, R_eval 159.951, R_seed 9.175, z -0.367)
  - cat.diag.cam_work_per_m: 1343.761 -> 1294.671 (lower is better, signed delta 49.091, R_eval 92.169, R_seed 38.691, z 0.533)
  - cat.turn.action_jitter: 0.112 -> 0.128 (lower is better, signed delta -0.016, R_eval 0.003, R_seed 0.001, z -6.096)
  - cat.turn.swings_per_rad: 63.226 -> 78.695 (lower is better, signed delta -15.469, R_eval 5.993, R_seed 14.018, z -2.581)
  - cat.turn.slip_per_rad: 4.270 -> 4.821 (lower is better, signed delta -0.552, R_eval 0.445, R_seed 0.755, z -1.241)
  - cat.turn.work_per_rad: 2861.369 -> 3591.053 (lower is better, signed delta -729.684, R_eval 294.326, R_seed 622.644, z -2.479)
  - cat.turn.cam_work_per_rad: 1969.142 -> 2190.349 (lower is better, signed delta -221.207, R_eval 217.387, R_seed 287.000, z -1.018)
  - cat.move_turn.action_jitter: 0.122 -> 0.135 (lower is better, signed delta -0.012, R_eval 0.002, R_seed 0.005, z -6.362)
  - cat.move_turn.swings_per_rad: 101.424 -> 132.381 (lower is better, signed delta -30.957, R_eval 38.045, R_seed 115.407, z -0.814)
  - cat.move_turn.slip_per_rad: 7.013 -> 9.037 (lower is better, signed delta -2.024, R_eval 1.897, R_seed 7.793, z -1.067)
  - cat.move_turn.work_per_rad: 4646.008 -> 6027.635 (lower is better, signed delta -1381.627, R_eval 1291.796, R_seed 4673.903, z -1.070)
  - cat.move_turn.cam_work_per_rad: 3021.470 -> 4402.071 (lower is better, signed delta -1380.601, R_eval 1299.170, R_seed 4135.208, z -1.063)
- guard cat.stand.survival: 0.990 -> 0.970 (R 0.072) -> TIE
- guard cat.fwd.survival: 0.950 -> 0.930 (R 0.120) -> TIE
- guard cat.fwd.ratio: 0.724 -> 0.695 (R 0.049) -> TIE
- guard cat.fwd.slip_per_m: 2.182 -> 2.308 (R 0.248) -> TIE
- guard cat.side.survival: 0.975 -> 0.965 (R 0.051) -> TIE
- guard cat.side.ratio: 0.422 -> 0.430 (R 0.020) -> TIE
- guard cat.side.slip_per_m: 6.839 -> 6.741 (R 0.243) -> TIE
- guard cat.back.survival: 0.930 -> 0.920 (R 0.105) -> TIE
- guard cat.back.ratio: 0.734 -> 0.681 (R 0.063) -> TIE
- guard cat.back.slip_per_m: 2.575 -> 2.782 (R 0.202) -> WORSE
- guard cat.diag.survival: 0.860 -> 0.770 (R 0.230) -> TIE
- guard cat.diag.ratio: 0.533 -> 0.541 (R 0.031) -> TIE
- guard cat.diag.slip_per_m: 4.632 -> 4.832 (R 0.334) -> TIE
- guard cat.turn.survival: 0.995 -> 0.975 (R 0.055) -> TIE
- guard cat.move_turn.survival: 0.935 -> 0.850 (R 0.100) -> TIE
- guard cat.move_turn.ratio: 0.504 -> 0.433 (R 0.036) -> WORSE
- anchored drift: FAIL — drifted below the anchor on block Q, guard cat.back.slip_per_m, guard cat.move_turn.ratio

>>> ENTRY D05

### experiment X03 — M1 s3 — ok — 2026-09-23T05:18:58

- recipe `1cbea4c2c35c` base `979629245c00` chain `fork_c6` seed 3
- stages ['X03:c6ft']
- screen: completion 0.917, tripod_dir 0.392, track tail 0.611
- replay: none

>>> ENTRY X03


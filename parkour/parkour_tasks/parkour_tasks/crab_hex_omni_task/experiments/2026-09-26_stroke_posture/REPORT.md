# Candidate ladder `2026-09-26_stroke_posture`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 3, confirm seed 2, 4000 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `709161927627f6c00012abf143b3146ff6d51f81acdc709dbb0f04c014f24a30` · `candidates.yaml` sha256 `c2ba8955c202ee4fa269c8759bfe07f0441c15631f21bc4099b8fd625cfce5ac`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | B | baseline | c6y | the incumbent recipe at equal compute | — |
| 1 | S | None | effective | `{"c6y":{"KRABBY_OMNI_STROKE_W":"0.117"}}` | `—` |
| 2 | P | None | effective | `{"c6y":{"KRABBY_OMNI_CENTRING_W":"0.160"}}` | `—` |

### X01:c6y — B s3 (baseline) — 2026-09-26T04:46:50

- recipe `5abe9b1e0211` on chain `c6y`, base `5abe9b1e0211`, code_sha `709161927627`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1', 'KRABBY_OMNI_STROKE_W': '0.001', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1'}`
- status **ok** | checkpoint `2026-09-26_03-01-22_seed3_X01_c6y_A15pB/model_15994.pt` | resume `model_11995.pt` | 4000 its | wall 1.63 h
- smoke fail@2k 0.406 fail@end 0.431 coll@2k 0.000 ep_len@2k 1607.493
- training translation floor: track_ratio_mean (last 500 its) 0.700 (gate floor 0.6)
- exposure (ring): reach_edge 0.614 | reach_obst 0.472 | field_frac 0.230 (steps 422.839) | goals_passed 0.862
- terrain-promotion telemetry: track_ratio_mean 0.708 (move rows only 0.713) | heading_err_rms_mean 0.152 | block_steps_mean 100.863 | r_max_mean 2.497
- failure share flat 0.460 | obst 0.467 | campaign-wide 0.409
- slot fractions (logged): stand 0.124 | move 0.440 | turn 0.111 | move_turn 0.115 | rsi_frac_actual 0.198 | terrain_levels 0.973 | mean ep len 1609.939 | prints 4000
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
- suite stand: n 100 surv 0.990 drift 0.075 [decisive]
- suite fwd: n 100 surv 0.990 ratio 0.861 tripod 0.584 [decisive]
- suite side: n 200 surv 0.990 ratio 0.536 tripod 0.479 (l: surv 0.980 ratio 0.534 tripod 0.479; r: surv 1.000 ratio 0.539 tripod 0.478) [decisive]
- suite back: n 100 surv 0.980 ratio 0.872 tripod 0.613 [decisive]
- suite diag: n 100 surv 0.560 ratio 0.695 tripod 0.363 [decisive]
- suite turn: n 200 surv 0.995 closed 0.751 progress 0.100 (l: surv 1.000 closed 0.734 progress 0.098; r: surv 0.990 closed 0.782 progress 0.103) [decisive]
- suite move_turn: n 200 surv 0.585 closed 0.475 progress 0.074 ratio 0.587 (l: surv 0.660 closed 0.516 progress 0.076 ratio 0.589; r: surv 0.510 closed 0.445 progress 0.072 ratio 0.586) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.363 (need >= 0.4; by hold: back 0.613, diag 0.363, fwd 0.584, lat_l 0.479, lat_r 0.478, move_turn 0.479, move_turn_r 0.445, settle 0.670, stand 0.715, turn_l90 0.530, turn_r90 0.526), completion 0.865 (need >= 0.7), yaw-progress rate turn_l90 0.098, turn_r90 0.103 rad/s (net turn_l90 0.097, turn_r90 0.102) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.534, diag 0.695, back 0.872, move_turn 0.589 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.700 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.990 (bound 1.013, n_env 100, min >= 0.8); fwd 0.990 (bound 1.013, n_env 100, min >= 0.75); side 0.990 (bound 1.013, n_env 100, min >= 0.795); back 0.980 (bound 1.013, n_env 100, min >= 0.78); diag 0.560 (bound 0.675, n_env 100, min >= 0.67); turn 0.995 (bound 1.011, n_env 100, min >= 0.79); move_turn 0.585 (bound 0.700, n_env 100, min >= 0.165)

>>> ENTRY X01:c6y

### experiment X01 — B s3 — ok — 2026-09-26T04:46:50

- recipe `5abe9b1e0211` base `5abe9b1e0211` chain `c6y` seed 3
- stages ['X01:c6y']
- screen: completion 0.865, tripod_dir 0.363, track tail 0.700
- replay: none

>>> ENTRY X01

### X02:c6y — B s2 (baseline) — 2026-09-26T06:32:28

- recipe `5abe9b1e0211` on chain `c6y`, base `5abe9b1e0211`, code_sha `709161927627`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1', 'KRABBY_OMNI_STROKE_W': '0.001', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1'}`
- status **ok** | checkpoint `2026-09-26_04-47-00_seed2_X02_c6y_A15pB/model_15994.pt` | resume `model_11995.pt` | 4000 its | wall 1.63 h
- smoke fail@2k 0.476 fail@end 0.450 coll@2k 0.000 ep_len@2k 1565.892
- training translation floor: track_ratio_mean (last 500 its) 0.689 (gate floor 0.6)
- exposure (ring): reach_edge 0.604 | reach_obst 0.544 | field_frac 0.223 (steps 401.149) | goals_passed 0.845
- terrain-promotion telemetry: track_ratio_mean 0.691 (move rows only 0.695) | heading_err_rms_mean 0.146 | block_steps_mean 127.516 | r_max_mean 2.474
- failure share flat 0.452 | obst 0.467 | campaign-wide 0.461
- slot fractions (logged): stand 0.094 | move 0.465 | turn 0.073 | move_turn 0.179 | rsi_frac_actual 0.187 | terrain_levels 1.027 | mean ep len 1485.663 | prints 4000
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
- suite stand: n 100 surv 1.000 drift 0.071 [decisive]
- suite fwd: n 100 surv 0.970 ratio 0.793 tripod 0.578 [decisive]
- suite side: n 200 surv 0.995 ratio 0.520 tripod 0.479 (l: surv 1.000 ratio 0.526 tripod 0.486; r: surv 0.990 ratio 0.515 tripod 0.474) [decisive]
- suite back: n 100 surv 0.970 ratio 0.784 tripod 0.585 [decisive]
- suite diag: n 100 surv 0.640 ratio 0.632 tripod 0.341 [decisive]
- suite turn: n 200 surv 0.995 closed 0.665 progress 0.088 (l: surv 0.990 closed 0.665 progress 0.087; r: surv 1.000 closed 0.664 progress 0.088) [decisive]
- suite move_turn: n 200 surv 0.525 closed 0.368 progress 0.060 ratio 0.518 (l: surv 0.580 closed 0.405 progress 0.067 ratio 0.527; r: surv 0.470 closed 0.362 progress 0.059 ratio 0.504) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.341 (need >= 0.4; by hold: back 0.585, diag 0.341, fwd 0.578, lat_l 0.486, lat_r 0.474, move_turn 0.446, move_turn_r 0.439, settle 0.713, stand 0.720, turn_l90 0.502, turn_r90 0.481), completion 0.861 (need >= 0.7), yaw-progress rate turn_l90 0.087, turn_r90 0.088 rad/s (net turn_l90 0.087, turn_r90 0.086) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.526, diag 0.632, back 0.784, move_turn 0.527 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.689 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, min >= 0.8); fwd 0.970 (bound 1.010, n_env 100, min >= 0.75); side 0.995 (bound 1.011, n_env 100, min >= 0.795); back 0.970 (bound 1.010, n_env 100, min >= 0.78); diag 0.640 (bound 0.752, n_env 100, min >= 0.67); turn 0.995 (bound 1.011, n_env 100, min >= 0.79); move_turn 0.525 (bound 0.641, n_env 100, min >= 0.165)

>>> ENTRY X02:c6y

### experiment X02 — B s2 — ok — 2026-09-26T06:32:28

- recipe `5abe9b1e0211` base `5abe9b1e0211` chain `c6y` seed 2
- stages ['X02:c6y']
- screen: completion 0.861, tripod_dir 0.341, track tail 0.689
- replay: none

>>> ENTRY X02

### X03:c6y — S s3 (candidate) — 2026-09-26T08:18:05

- recipe `9041662b29d0` on chain `c6y`, base `5abe9b1e0211`, code_sha `709161927627`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1', 'KRABBY_OMNI_STROKE_W': '0.117', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1'}`
- status **ok** | checkpoint `2026-09-26_06-32-38_seed3_X03_c6y_A15pB/model_15994.pt` | resume `model_11995.pt` | 4000 its | wall 1.63 h
- smoke fail@2k 0.451 fail@end 0.435 coll@2k 0.000 ep_len@2k 1540.143
- training translation floor: track_ratio_mean (last 500 its) 0.674 (gate floor 0.6)
- exposure (ring): reach_edge 0.610 | reach_obst 0.484 | field_frac 0.217 (steps 397.954) | goals_passed 0.827
- terrain-promotion telemetry: track_ratio_mean 0.680 (move rows only 0.688) | heading_err_rms_mean 0.144 | block_steps_mean 100.653 | r_max_mean 2.532
- failure share flat 0.430 | obst 0.405 | campaign-wide 0.474
- slot fractions (logged): stand 0.081 | move 0.416 | turn 0.093 | move_turn 0.144 | rsi_frac_actual 0.187 | terrain_levels 0.973 | mean ep len 1466.571 | prints 4000
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
- suite stand: n 100 surv 1.000 drift 0.076 [decisive]
- suite fwd: n 100 surv 0.980 ratio 0.783 tripod 0.591 [decisive]
- suite side: n 200 surv 0.995 ratio 0.535 tripod 0.447 (l: surv 0.990 ratio 0.542 tripod 0.447; r: surv 1.000 ratio 0.529 tripod 0.446) [decisive]
- suite back: n 100 surv 0.970 ratio 0.802 tripod 0.596 [decisive]
- suite diag: n 100 surv 0.680 ratio 0.636 tripod 0.370 [decisive]
- suite turn: n 200 surv 0.995 closed 0.715 progress 0.095 (l: surv 1.000 closed 0.733 progress 0.097; r: surv 0.990 closed 0.706 progress 0.094) [decisive]
- suite move_turn: n 200 surv 0.200 closed 0.267 progress 0.059 ratio 0.525 (l: surv 0.200 closed 0.272 progress 0.062 ratio 0.539; r: surv 0.200 closed 0.267 progress 0.058 ratio 0.499) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.370 (need >= 0.4; by hold: back 0.596, diag 0.370, fwd 0.591, lat_l 0.447, lat_r 0.446, move_turn 0.378, move_turn_r 0.400, settle 0.716, stand 0.665, turn_l90 0.457, turn_r90 0.458), completion 0.801 (need >= 0.7), yaw-progress rate turn_l90 0.097, turn_r90 0.094 rad/s (net turn_l90 0.096, turn_r90 0.093) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.542, diag 0.636, back 0.802, move_turn 0.539 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.674 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.117 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, min >= 0.8); fwd 0.980 (bound 1.013, n_env 100, min >= 0.75); side 0.995 (bound 1.011, n_env 100, min >= 0.795); back 0.970 (bound 1.010, n_env 100, min >= 0.78); diag 0.680 (bound 0.789, n_env 100, min >= 0.67); turn 0.995 (bound 1.011, n_env 100, min >= 0.79); move_turn 0.200 (bound 0.293, n_env 100, min >= 0.165)

>>> ENTRY X03:c6y

## D05 — S — REJECT (clear, decided) — 2026-09-26T08:18:06

- question: On W, does a per-foot loaded stance-stroke income at w_S (the replaced stride term, alive and command-general; R4: fwd delivered income 10 % of tracking) for a further 4000 iterations change the stroke (the decision-9 manipulation check: the delivered dose landed and some stroke key moved beyond its MDE), keep its extra stride stroke-borne (side stroke_share not lower), and lengthen side/diag steps (LEN, mechanism evidence, not required) with no block or guard WORSE against B, the same 4000 iterations at eps?
- experiment `X03` recipe `9041662b29d0` base `5abe9b1e0211` control `X01`
- why: WORSE on block F_mt, guard cat.fwd.ratio, guard cat.turn.foot_slip, guard cat.move_turn.survival
- block F (F): z -0.754 vs R 1.810 -> TIE (k=2, missing [])
  - cat.turn.facing_closed: 0.751 -> 0.715 (delta -0.036, R_eval 0.050, R_seed 0.086, z -0.731)
  - cat.turn.yaw_progress: 0.100 -> 0.095 (delta -0.005, R_eval 0.007, R_seed 0.012, z -0.778)
- block F_mt (F_mt): z -3.829 vs R 2.451 -> WORSE (k=2, missing [])
  - cat.move_turn.facing_closed: 0.475 -> 0.267 (delta -0.208, R_eval 0.040, R_seed 0.107, z -5.163)
  - cat.move_turn.yaw_progress: 0.074 -> 0.059 (delta -0.015, R_eval 0.006, R_seed 0.013, z -2.494)
- block T (T): z -1.400 vs R 1.665 -> TIE (k=4, missing [])
  - cat.fwd.ratio: 0.861 -> 0.783 (delta -0.077, R_eval 0.041, R_seed 0.068, z -1.886)
  - cat.side.ratio: 0.536 -> 0.535 (delta -0.001, R_eval 0.018, R_seed 0.016, z -0.072)
  - cat.back.ratio: 0.872 -> 0.802 (delta -0.070, R_eval 0.053, R_seed 0.087, z -1.328)
  - cat.diag.ratio: 0.695 -> 0.636 (delta -0.059, R_eval 0.026, R_seed 0.063, z -2.315)
- block RIG (RIG): z -0.088 vs R 0.415 -> TIE (k=10, missing [])
  - cat.fwd.steps_per_m: 13.726 -> 15.273 (lower is better, signed delta -1.546, R_eval 1.635, R_seed 2.037, z -0.946)
  - cat.fwd.support_feet: 3.055 -> 3.101 (delta 0.046, R_eval 0.030, R_seed 0.042, z 1.528)
  - cat.side.steps_per_m: 27.208 -> 27.696 (lower is better, signed delta -0.489, R_eval 1.377, R_seed 1.611, z -0.355)
  - cat.side.support_feet: 3.583 -> 3.598 (delta 0.014, R_eval 0.028, R_seed 0.004, z 0.515)
  - cat.back.steps_per_m: 15.063 -> 16.953 (lower is better, signed delta -1.890, R_eval 2.032, R_seed 2.946, z -0.930)
  - cat.back.support_feet: 3.051 -> 3.107 (delta 0.056, R_eval 0.030, R_seed 0.058, z 1.887)
  - cat.diag.steps_per_m: 18.622 -> 21.143 (lower is better, signed delta -2.521, R_eval 1.623, R_seed 3.058, z -1.553)
  - cat.diag.support_feet: 3.334 -> 3.389 (delta 0.055, R_eval 0.074, R_seed 0.000, z 0.743)
  - cat.turn.steps_per_rad: 28.802 -> 33.011 (lower is better, signed delta -4.209, R_eval 3.898, R_seed 6.109, z -1.080)
  - cat.turn.support_feet: 3.175 -> 3.160 (delta -0.015, R_eval 0.022, R_seed 0.015, z -0.692)
- block LEN (LEN): z -0.954 vs R 1.527 -> TIE (k=2, missing [])
  - cat.side.steps_per_m: 27.208 -> 27.696 (lower is better, signed delta -0.489, R_eval 1.377, R_seed 1.611, z -0.355)
  - cat.diag.steps_per_m: 18.622 -> 21.143 (lower is better, signed delta -2.521, R_eval 1.623, R_seed 3.058, z -1.553)
- guard cat.stand.survival: 0.990 -> 1.000 (R 0.072) -> TIE
- guard cat.stand.steps_per_s: 2.667 -> 2.833 (R 0.312) -> TIE
- guard cat.fwd.survival: 0.990 -> 0.980 (R 0.110) -> TIE
- guard cat.fwd.ratio: 0.861 -> 0.783 (R 0.068) -> WORSE
- guard cat.fwd.foot_slip: 0.122 -> 0.122 (R 0.007) -> TIE
- guard cat.fwd.steps_per_m: 13.726 -> 15.273 (R 2.037) -> TIE
- guard cat.side.survival: 0.990 -> 0.995 (R 0.051) -> TIE
- guard cat.side.ratio: 0.536 -> 0.535 (R 0.021) -> TIE
- guard cat.side.foot_slip: 0.193 -> 0.198 (R 0.008) -> TIE
- guard cat.side.steps_per_m: 27.208 -> 27.696 (R 1.611) -> TIE
- guard cat.back.survival: 0.980 -> 0.970 (R 0.105) -> TIE
- guard cat.back.ratio: 0.872 -> 0.802 (R 0.087) -> TIE
- guard cat.back.foot_slip: 0.119 -> 0.119 (R 0.007) -> TIE
- guard cat.back.steps_per_m: 15.063 -> 16.953 (R 2.946) -> TIE
- guard cat.diag.survival: 0.560 -> 0.680 (R 0.160) -> TIE
- guard cat.diag.ratio: 0.695 -> 0.636 (R 0.063) -> TIE
- guard cat.diag.foot_slip: 0.189 -> 0.188 (R 0.010) -> TIE
- guard cat.diag.steps_per_m: 18.622 -> 21.143 (R 3.058) -> TIE
- guard cat.turn.survival: 0.995 -> 0.995 (R 0.058) -> TIE
- guard cat.turn.foot_slip: 0.103 -> 0.108 (R 0.004) -> WORSE
- guard cat.turn.steps_per_rad: 28.802 -> 33.011 (R 6.109) -> TIE
- guard cat.move_turn.survival: 0.585 -> 0.200 (R 0.114) -> WORSE
- guard cat.move_turn.ratio: 0.587 -> 0.525 (R 0.070) -> TIE
- anchored drift: FAIL — drifted below the anchor on block F_mt, guard cat.fwd.ratio, guard cat.turn.foot_slip, guard cat.move_turn.survival

>>> ENTRY D05

### experiment X03 — S s3 — ok — 2026-09-26T08:18:06

- recipe `9041662b29d0` base `5abe9b1e0211` chain `c6y` seed 3
- stages ['X03:c6y']
- screen: completion 0.801, tripod_dir 0.370, track tail 0.674
- replay: none

>>> ENTRY X03

### X04:c6y — P s3 (candidate) — 2026-09-26T10:09:02

- recipe `3341dbf981de` on chain `c6y`, base `5abe9b1e0211`, code_sha `709161927627`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1', 'KRABBY_OMNI_STROKE_W': '0.001', 'KRABBY_OMNI_CENTRING_W': '0.160', 'KRABBY_OMNI_REWARD_SPLIT': '1'}`
- status **ok** | checkpoint `2026-09-26_08-21-34_seed3_X04_c6y_A15pB/model_15994.pt` | resume `model_11995.pt` | 4000 its | wall 1.67 h
- smoke fail@2k 0.327 fail@end 0.399 coll@2k 0.000 ep_len@2k 1691.912
- training translation floor: track_ratio_mean (last 500 its) 0.680 (gate floor 0.6)
- exposure (ring): reach_edge 0.604 | reach_obst 0.485 | field_frac 0.210 (steps 392.218) | goals_passed 0.838
- terrain-promotion telemetry: track_ratio_mean 0.680 (move rows only 0.689) | heading_err_rms_mean 0.156 | block_steps_mean 107.562 | r_max_mean 2.524
- failure share flat 0.384 | obst 0.394 | campaign-wide 0.414
- slot fractions (logged): stand 0.138 | move 0.419 | turn 0.058 | move_turn 0.160 | rsi_frac_actual 0.215 | terrain_levels 0.973 | mean ep len 1583.978 | prints 4000
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
- suite stand: n 100 surv 1.000 drift 0.074 [decisive]
- suite fwd: n 100 surv 0.980 ratio 0.781 tripod 0.569 [decisive]
- suite side: n 200 surv 0.990 ratio 0.561 tripod 0.469 (l: surv 0.980 ratio 0.560 tripod 0.472; r: surv 1.000 ratio 0.562 tripod 0.467) [decisive]
- suite back: n 100 surv 0.980 ratio 0.776 tripod 0.616 [decisive]
- suite diag: n 100 surv 0.800 ratio 0.623 tripod 0.368 [decisive]
- suite turn: n 200 surv 0.995 closed 0.817 progress 0.108 (l: surv 0.990 closed 0.809 progress 0.109; r: surv 1.000 closed 0.822 progress 0.108) [decisive]
- suite move_turn: n 200 surv 0.275 closed 0.333 progress 0.063 ratio 0.520 (l: surv 0.360 closed 0.415 progress 0.067 ratio 0.520; r: surv 0.190 closed 0.278 progress 0.059 ratio 0.519) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.368 (need >= 0.4; by hold: back 0.616, diag 0.368, fwd 0.569, lat_l 0.472, lat_r 0.467, move_turn 0.419, move_turn_r 0.402, settle 0.733, stand 0.655, turn_l90 0.480, turn_r90 0.478), completion 0.828 (need >= 0.7), yaw-progress rate turn_l90 0.109, turn_r90 0.108 rad/s (net turn_l90 0.108, turn_r90 0.106) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.560, diag 0.623, back 0.776, move_turn 0.520 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.680 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.160 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, min >= 0.8); fwd 0.980 (bound 1.013, n_env 100, min >= 0.75); side 0.990 (bound 1.013, n_env 100, min >= 0.795); back 0.980 (bound 1.013, n_env 100, min >= 0.78); diag 0.800 (bound 0.893, n_env 100, min >= 0.67); turn 0.995 (bound 1.011, n_env 100, min >= 0.79); move_turn 0.275 (bound 0.379, n_env 100, min >= 0.165)

>>> ENTRY X04:c6y

## D06 — P — REJECT (clear, decided) — 2026-09-26T10:09:03

- question: On W, does a body-centring bonus at w_P (P2 prize parity with S) for a further 4000 iterations pull the body back over its six-foot centroid on side holds (body_offset down >= 0.05 m, the manipulation check) and does the recentred posture lengthen side/diag steps (LEN) with no block or guard WORSE against B? A POS+ reads "posture or side-row emphasis" (recorded, not controlled).
- experiment `X04` recipe `3341dbf981de` base `5abe9b1e0211` control `X01`
- why: WORSE on block F_mt, guard cat.fwd.ratio, guard cat.fwd.steps_per_m, guard cat.side.foot_slip, guard cat.back.ratio, guard cat.diag.ratio, guard cat.diag.steps_per_m, guard cat.move_turn.survival
- block F (F): z 1.295 vs R 1.810 -> TIE (k=2, missing [])
  - cat.turn.facing_closed: 0.751 -> 0.817 (delta 0.066, R_eval 0.050, R_seed 0.086, z 1.331)
  - cat.turn.yaw_progress: 0.100 -> 0.108 (delta 0.008, R_eval 0.007, R_seed 0.012, z 1.260)
- block F_mt (F_mt): z -2.625 vs R 2.451 -> WORSE (k=2, missing [])
  - cat.move_turn.facing_closed: 0.475 -> 0.333 (delta -0.142, R_eval 0.040, R_seed 0.107, z -3.523)
  - cat.move_turn.yaw_progress: 0.074 -> 0.063 (delta -0.010, R_eval 0.006, R_seed 0.013, z -1.727)
- block T (T): z -1.302 vs R 1.665 -> TIE (k=4, missing [])
  - cat.fwd.ratio: 0.861 -> 0.781 (delta -0.079, R_eval 0.041, R_seed 0.068, z -1.932)
  - cat.side.ratio: 0.536 -> 0.561 (delta 0.025, R_eval 0.018, R_seed 0.016, z 1.370)
  - cat.back.ratio: 0.872 -> 0.776 (delta -0.096, R_eval 0.053, R_seed 0.087, z -1.824)
  - cat.diag.ratio: 0.695 -> 0.623 (delta -0.072, R_eval 0.026, R_seed 0.063, z -2.821)
- block RIG (RIG): z -0.344 vs R 0.415 -> TIE (k=10, missing [])
  - cat.fwd.steps_per_m: 13.726 -> 15.817 (lower is better, signed delta -2.090, R_eval 1.635, R_seed 2.037, z -1.278)
  - cat.fwd.support_feet: 3.055 -> 3.032 (delta -0.023, R_eval 0.030, R_seed 0.042, z -0.778)
  - cat.side.steps_per_m: 27.208 -> 26.603 (lower is better, signed delta 0.605, R_eval 1.377, R_seed 1.611, z 0.439)
  - cat.side.support_feet: 3.583 -> 3.547 (delta -0.037, R_eval 0.028, R_seed 0.004, z -1.333)
  - cat.back.steps_per_m: 15.063 -> 17.286 (lower is better, signed delta -2.223, R_eval 2.032, R_seed 2.946, z -1.094)
  - cat.back.support_feet: 3.051 -> 3.076 (delta 0.025, R_eval 0.030, R_seed 0.058, z 0.845)
  - cat.diag.steps_per_m: 18.622 -> 21.707 (lower is better, signed delta -3.085, R_eval 1.623, R_seed 3.058, z -1.901)
  - cat.diag.support_feet: 3.334 -> 3.336 (delta 0.002, R_eval 0.074, R_seed 0.000, z 0.024)
  - cat.turn.steps_per_rad: 28.802 -> 28.272 (lower is better, signed delta 0.530, R_eval 3.898, R_seed 6.109, z 0.136)
  - cat.turn.support_feet: 3.175 -> 3.208 (delta 0.033, R_eval 0.022, R_seed 0.015, z 1.500)
- block LEN (LEN): z -0.731 vs R 1.527 -> TIE (k=2, missing [])
  - cat.side.steps_per_m: 27.208 -> 26.603 (lower is better, signed delta 0.605, R_eval 1.377, R_seed 1.611, z 0.439)
  - cat.diag.steps_per_m: 18.622 -> 21.707 (lower is better, signed delta -3.085, R_eval 1.623, R_seed 3.058, z -1.901)
- guard cat.stand.survival: 0.990 -> 1.000 (R 0.072) -> TIE
- guard cat.stand.steps_per_s: 2.667 -> 2.833 (R 0.312) -> TIE
- guard cat.fwd.survival: 0.990 -> 0.980 (R 0.110) -> TIE
- guard cat.fwd.ratio: 0.861 -> 0.781 (R 0.068) -> WORSE
- guard cat.fwd.foot_slip: 0.122 -> 0.118 (R 0.007) -> TIE
- guard cat.fwd.steps_per_m: 13.726 -> 15.817 (R 2.037) -> WORSE
- guard cat.side.survival: 0.990 -> 0.990 (R 0.051) -> TIE
- guard cat.side.ratio: 0.536 -> 0.561 (R 0.021) -> TIE
- guard cat.side.foot_slip: 0.193 -> 0.203 (R 0.008) -> WORSE
- guard cat.side.steps_per_m: 27.208 -> 26.603 (R 1.611) -> TIE
- guard cat.back.survival: 0.980 -> 0.980 (R 0.105) -> TIE
- guard cat.back.ratio: 0.872 -> 0.776 (R 0.087) -> WORSE
- guard cat.back.foot_slip: 0.119 -> 0.118 (R 0.007) -> TIE
- guard cat.back.steps_per_m: 15.063 -> 17.286 (R 2.946) -> TIE
- guard cat.diag.survival: 0.560 -> 0.800 (R 0.153) -> TIE
- guard cat.diag.ratio: 0.695 -> 0.623 (R 0.063) -> WORSE
- guard cat.diag.foot_slip: 0.189 -> 0.188 (R 0.010) -> TIE
- guard cat.diag.steps_per_m: 18.622 -> 21.707 (R 3.058) -> WORSE
- guard cat.turn.survival: 0.995 -> 0.995 (R 0.058) -> TIE
- guard cat.turn.foot_slip: 0.103 -> 0.102 (R 0.004) -> TIE
- guard cat.turn.steps_per_rad: 28.802 -> 28.272 (R 6.109) -> TIE
- guard cat.move_turn.survival: 0.585 -> 0.275 (R 0.115) -> WORSE
- guard cat.move_turn.ratio: 0.587 -> 0.520 (R 0.070) -> TIE
- anchored drift: FAIL — drifted below the anchor on block F_mt, guard cat.fwd.ratio, guard cat.fwd.steps_per_m, guard cat.side.foot_slip, guard cat.back.ratio, guard cat.diag.ratio, guard cat.diag.steps_per_m, guard cat.move_turn.survival

>>> ENTRY D06

### experiment X04 — P s3 — ok — 2026-09-26T10:09:03

- recipe `3341dbf981de` base `5abe9b1e0211` chain `c6y` seed 3
- stages ['X04:c6y']
- screen: completion 0.828, tripod_dir 0.368, track tail 0.680
- replay: none

>>> ENTRY X04


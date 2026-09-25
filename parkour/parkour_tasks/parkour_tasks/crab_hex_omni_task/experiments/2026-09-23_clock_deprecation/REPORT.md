# Candidate ladder `2026-09-23_clock_deprecation`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 3, confirm seed 2, 5000 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `8ce6da6156b31b7d85cb499394b517bc4bea6c83e91c4db56562f1a68d373a73` · `candidates.yaml` sha256 `303422311a5b606d6a8d67362879e45f41d1581e93e9b45be1d30ed225d3baa2`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | X0 | baseline | tail | the incumbent recipe at equal compute | — |
| 1 | A6000 | clock | effective | `{"c6":{"KRABBY_APEX_W":"0.001","KRABBY_CLOCK_W":"0.001","KRABBY_OMNI_CLOCK_HIDE":"1"}}` | `—` |
| 2 | A4500 | clock | effective | `{"c5":{"KRABBY_APEX_W":"0.001","KRABBY_CLOCK_W":"0.001","KRABBY_OMNI_CLOCK_HIDE":"1"},"c6":{"KRABBY_APEX_W":"0.001","KRABBY_CLOCK_W":"0.001","KRABBY_OMNI_CLOCK_HIDE":"1"}}` | `—` |
| 3 | A3000 | clock | effective | `{"c4":{"KRABBY_APEX_W":"0.001","KRABBY_CLOCK_W":"0.001","KRABBY_OMNI_CLOCK_HIDE":"1"},"c5":{"KRABBY_APEX_W":"0.001","KRABBY_CLOCK_W":"0.001","KRABBY_OMNI_CLOCK_HIDE":"1"},"c6":{"KRABBY_APEX_W":"0.001","KRABBY_CLOCK_W":"0.001","KRABBY_OMNI_CLOCK_HIDE":"1"}}` | `—` |

### X01:c4 — X0 s3 (baseline) — 2026-09-23T19:20:23

- recipe `f8f4ad5d955e` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.15:0.1', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-23_18-29-32_seed3_X01_c4_A15pB/model_4498.pt` | resume `model_2999.pt` | 1500 its | wall 0.70 h
- smoke fail@2k 0.212 fail@end 0.212 coll@2k 0.000 ep_len@2k 1830.111
- training translation floor: track_ratio_mean (last 500 its) 0.600 (gate floor 0.6)
- exposure (ring): reach_edge 0.481 | reach_obst 0.379 | field_frac 0.161 (steps 304.699) | goals_passed 0.608
- terrain-promotion telemetry: track_ratio_mean 0.594 (move rows only 0.599) | heading_err_rms_mean 0.153 | block_steps_mean 80.489 | r_max_mean 2.344
- failure share flat 0.226 | obst 0.319 | campaign-wide 0.191
- slot fractions (logged): stand 0.143 | move 0.531 | turn 0.153 | move_turn 0.115 | rsi_frac_actual 0.183 | terrain_levels 0.973 | mean ep len 1854.399 | prints 1500
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
- suite trained categories: decisive stand, fwd, side, back, turn, move_turn
- suite stand: n 100 surv 1.000 drift 0.097 [decisive]
- suite fwd: n 100 surv 0.870 ratio 0.698 tripod 0.577 [decisive]
- suite side: n 200 surv 0.940 ratio 0.472 tripod 0.463 (l: surv 0.920 ratio 0.473 tripod 0.475; r: surv 0.960 ratio 0.468 tripod 0.455) [decisive]
- suite back: n 100 surv 0.850 ratio 0.633 tripod 0.525 [decisive]
- suite diag: n 100 surv 0.620 ratio 0.587 tripod 0.418 [WATCH: untrained]
- suite turn: n 200 surv 0.975 closed 0.505 progress 0.067 (l: surv 0.980 closed 0.508 progress 0.067; r: surv 0.970 closed 0.497 progress 0.066) [decisive]
- suite move_turn: n 200 surv 0.795 closed 0.134 progress 0.018 ratio 0.473 (l: surv 0.740 closed 0.145 progress 0.022 ratio 0.476; r: surv 0.850 closed 0.120 progress 0.016 ratio 0.471) [decisive]
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.15:0.1 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, min >= 0.69); fwd 0.870 (bound 0.948, n_env 100, min >= 0.52); side 0.940 (bound 0.995, n_env 100, min >= 0.65); back 0.850 (bound 0.933, n_env 100, min >= 0.55); turn 0.975 (bound 1.011, n_env 100, min >= 0.62); move_turn 0.795 (bound 0.889, n_env 100, min >= 0.315)

>>> ENTRY X01:c4

### X01:c5 — X0 s3 (baseline) — 2026-09-23T20:08:53

- recipe `f8f4ad5d955e` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.1:0.15', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-23_19-20-33_seed3_X01_c5_A15pB/model_5997.pt` | resume `model_4498.pt` | 1500 its | wall 0.67 h
- smoke fail@2k 0.269 fail@end 0.269 coll@2k 0.000 ep_len@2k 1755.707
- training translation floor: track_ratio_mean (last 500 its) 0.595 (gate floor 0.6)
- exposure (ring): reach_edge 0.526 | reach_obst 0.444 | field_frac 0.180 (steps 341.677) | goals_passed 0.705
- terrain-promotion telemetry: track_ratio_mean 0.602 (move rows only 0.610) | heading_err_rms_mean 0.152 | block_steps_mean 90.829 | r_max_mean 2.315
- failure share flat 0.287 | obst 0.302 | campaign-wide 0.257
- slot fractions (logged): stand 0.135 | move 0.483 | turn 0.067 | move_turn 0.148 | rsi_frac_actual 0.190 | terrain_levels 0.973 | mean ep len 1834.750 | prints 1500
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
- suite stand: n 100 surv 1.000 drift 0.090 [decisive]
- suite fwd: n 100 surv 0.880 ratio 0.718 tripod 0.556 [decisive]
- suite side: n 200 surv 0.980 ratio 0.404 tripod 0.486 (l: surv 0.990 ratio 0.408 tripod 0.485; r: surv 0.970 ratio 0.401 tripod 0.487) [decisive]
- suite back: n 100 surv 0.870 ratio 0.706 tripod 0.542 [decisive]
- suite diag: n 100 surv 0.720 ratio 0.550 tripod 0.398 [decisive]
- suite turn: n 200 surv 0.990 closed 0.492 progress 0.066 (l: surv 1.000 closed 0.483 progress 0.064; r: surv 0.980 closed 0.524 progress 0.068) [decisive]
- suite move_turn: n 200 surv 0.870 closed 0.290 progress 0.038 ratio 0.497 (l: surv 0.850 closed 0.247 progress 0.035 ratio 0.491; r: surv 0.890 closed 0.328 progress 0.043 ratio 0.497) [decisive]
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.1:0.15 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c5: no incumbent trace yet -- this run seeds it (7 categories recorded)
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, prev >= 0.3); fwd 0.880 (bound 0.956, n_env 100, prev >= 0.3); side 0.980 (bound 1.013, n_env 100, prev >= 0.3); back 0.870 (bound 0.948, n_env 100, prev >= 0.3); diag 0.720 (bound 0.824, n_env 100, new >= 0.1); turn 0.990 (bound 1.013, n_env 100, prev >= 0.3); move_turn 0.870 (bound 0.948, n_env 100, prev >= 0.3)

>>> ENTRY X01:c5

### X01:c6 — X0 s3 (baseline) — 2026-09-23T21:02:25

- recipe `f8f4ad5d955e` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-23_20-09-04_seed3_X01_c6_A15pB/model_7996.pt` | resume `model_5997.pt` | 2000 its | wall 0.77 h
- smoke fail@2k 0.333 fail@end 0.333 coll@2k 0.000 ep_len@2k 1700.021
- training translation floor: track_ratio_mean (last 500 its) 0.620 (gate floor 0.6)
- exposure (ring): reach_edge 0.499 | reach_obst 0.395 | field_frac 0.161 (steps 298.495) | goals_passed 0.649
- terrain-promotion telemetry: track_ratio_mean 0.620 (move rows only 0.628) | heading_err_rms_mean 0.154 | block_steps_mean 81.022 | r_max_mean 2.289
- failure share flat 0.356 | obst 0.426 | campaign-wide 0.310
- slot fractions (logged): stand 0.137 | move 0.454 | turn 0.104 | move_turn 0.120 | rsi_frac_actual 0.205 | terrain_levels 0.973 | mean ep len 1723.970 | prints 2000
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
- suite stand: n 100 surv 0.990 drift 0.082 [decisive]
- suite fwd: n 100 surv 0.850 ratio 0.733 tripod 0.590 [decisive]
- suite side: n 200 surv 0.970 ratio 0.396 tripod 0.496 (l: surv 0.960 ratio 0.395 tripod 0.501; r: surv 0.980 ratio 0.401 tripod 0.488) [decisive]
- suite back: n 100 surv 0.880 ratio 0.746 tripod 0.586 [decisive]
- suite diag: n 100 surv 0.720 ratio 0.539 tripod 0.385 [decisive]
- suite turn: n 200 surv 0.985 closed 0.629 progress 0.082 (l: surv 0.980 closed 0.633 progress 0.083; r: surv 0.990 closed 0.627 progress 0.080) [decisive]
- suite move_turn: n 200 surv 0.745 closed 0.330 progress 0.048 ratio 0.498 (l: surv 0.690 closed 0.279 progress 0.043 ratio 0.480; r: surv 0.800 closed 0.375 progress 0.050 ratio 0.515) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.385 (need >= 0.4; by hold: back 0.586, diag 0.385, fwd 0.590, lat_l 0.501, lat_r 0.488, move_turn 0.494, move_turn_r 0.536, settle 0.518, stand 0.604, turn_l90 0.638, turn_r90 0.651), completion 0.884 (need >= 0.7), yaw-progress rate turn_l90 0.083, turn_r90 0.080 rad/s (net turn_l90 0.083, turn_r90 0.080) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.395, diag 0.539, back 0.746, move_turn 0.480 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.620 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c6: no incumbent trace yet -- this run seeds it (7 categories recorded)
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.990 (bound 1.013, n_env 100, min >= 0.79); fwd 0.850 (bound 0.933, n_env 100, min >= 0.72); side 0.970 (bound 1.010, n_env 100, min >= 0.77); back 0.880 (bound 0.956, n_env 100, min >= 0.72); diag 0.720 (bound 0.824, n_env 100, min >= 0.59); turn 0.985 (bound 1.013, n_env 100, min >= 0.75); move_turn 0.745 (bound 0.846, n_env 100, min >= 0.55)

>>> ENTRY X01:c6

### experiment X01 — X0 s3 — ok — 2026-09-23T21:02:25

- recipe `f8f4ad5d955e` base `f8f4ad5d955e` chain `tail` seed 3
- stages ['X01:c4', 'X01:c5', 'X01:c6']
- screen: completion 0.884, tripod_dir 0.385, track tail 0.620
- replay: none

>>> ENTRY X01

### X02:c4 — X0 s2 (baseline) — 2026-09-23T21:43:55

- recipe `f8f4ad5d955e` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.15:0.1', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-23_21-02-35_seed2_X02_c4_A15pB/model_4498.pt` | resume `model_2999.pt` | 1500 its | wall 0.57 h
- smoke fail@2k 0.242 fail@end 0.242 coll@2k 0.000 ep_len@2k 1803.793
- training translation floor: track_ratio_mean (last 500 its) 0.587 (gate floor 0.6)
- exposure (ring): reach_edge 0.481 | reach_obst 0.445 | field_frac 0.159 (steps 305.304) | goals_passed 0.618
- terrain-promotion telemetry: track_ratio_mean 0.572 (move rows only 0.579) | heading_err_rms_mean 0.152 | block_steps_mean 101.457 | r_max_mean 2.222
- failure share flat 0.270 | obst 0.284 | campaign-wide 0.256
- slot fractions (logged): stand 0.099 | move 0.509 | turn 0.159 | move_turn 0.084 | rsi_frac_actual 0.214 | terrain_levels 1.027 | mean ep len 1739.197 | prints 1500
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
- suite trained categories: decisive stand, fwd, side, back, turn, move_turn
- suite stand: n 100 surv 0.990 drift 0.100 [decisive]
- suite fwd: n 100 surv 0.880 ratio 0.643 tripod 0.518 [decisive]
- suite side: n 200 surv 0.935 ratio 0.444 tripod 0.447 (l: surv 0.960 ratio 0.444 tripod 0.448; r: surv 0.910 ratio 0.446 tripod 0.445) [decisive]
- suite back: n 100 surv 0.910 ratio 0.626 tripod 0.536 [decisive]
- suite diag: n 100 surv 0.710 ratio 0.532 tripod 0.393 [WATCH: untrained]
- suite turn: n 200 surv 0.955 closed 0.519 progress 0.069 (l: surv 0.950 closed 0.513 progress 0.070; r: surv 0.960 closed 0.522 progress 0.069) [decisive]
- suite move_turn: n 200 surv 0.845 closed 0.191 progress 0.027 ratio 0.413 (l: surv 0.800 closed 0.171 progress 0.027 ratio 0.406; r: surv 0.890 closed 0.203 progress 0.026 ratio 0.428) [decisive]
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.15:0.1 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.990 (bound 1.013, n_env 100, min >= 0.69); fwd 0.880 (bound 0.956, n_env 100, min >= 0.52); side 0.935 (bound 0.992, n_env 100, min >= 0.65); back 0.910 (bound 0.977, n_env 100, min >= 0.55); turn 0.955 (bound 1.003, n_env 100, min >= 0.62); move_turn 0.845 (bound 0.929, n_env 100, min >= 0.315)

>>> ENTRY X02:c4

### X02:c5 — X0 s2 (baseline) — 2026-09-23T22:25:26

- recipe `f8f4ad5d955e` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.1:0.15', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-23_21-44-06_seed2_X02_c5_A15pB/model_5997.pt` | resume `model_4498.pt` | 1500 its | wall 0.57 h
- smoke fail@2k 0.262 fail@end 0.262 coll@2k 0.000 ep_len@2k 1775.173
- training translation floor: track_ratio_mean (last 500 its) 0.593 (gate floor 0.6)
- exposure (ring): reach_edge 0.495 | reach_obst 0.453 | field_frac 0.159 (steps 302.954) | goals_passed 0.646
- terrain-promotion telemetry: track_ratio_mean 0.594 (move rows only 0.608) | heading_err_rms_mean 0.154 | block_steps_mean 106.024 | r_max_mean 2.294
- failure share flat 0.246 | obst 0.269 | campaign-wide 0.264
- slot fractions (logged): stand 0.123 | move 0.559 | turn 0.087 | move_turn 0.148 | rsi_frac_actual 0.208 | terrain_levels 1.027 | mean ep len 1802.036 | prints 1500
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
- suite stand: n 100 surv 0.990 drift 0.086 [decisive]
- suite fwd: n 100 surv 0.950 ratio 0.715 tripod 0.633 [decisive]
- suite side: n 200 surv 0.975 ratio 0.397 tripod 0.496 (l: surv 0.980 ratio 0.395 tripod 0.500; r: surv 0.970 ratio 0.402 tripod 0.491) [decisive]
- suite back: n 100 surv 0.990 ratio 0.707 tripod 0.626 [decisive]
- suite diag: n 100 surv 0.870 ratio 0.508 tripod 0.438 [decisive]
- suite turn: n 200 surv 0.925 closed 0.671 progress 0.089 (l: surv 0.930 closed 0.716 progress 0.091; r: surv 0.920 closed 0.660 progress 0.088) [decisive]
- suite move_turn: n 200 surv 0.685 closed 0.193 progress 0.031 ratio 0.311 (l: surv 0.650 closed 0.201 progress 0.032 ratio 0.321; r: surv 0.720 closed 0.161 progress 0.030 ratio 0.283) [decisive]
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.1:0.15 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c5 stand: survival 1.000 -> 0.990 held
- RETENTION c5 stand: drift_speed 0.090 -> 0.086 held
- RETENTION c5 fwd: survival 0.880 -> 0.950 held
- RETENTION c5 fwd: ratio 0.718 -> 0.715 held
- RETENTION c5 side: survival 0.980 -> 0.975 held
- RETENTION c5 side: ratio 0.404 -> 0.397 held
- RETENTION c5 back: survival 0.870 -> 0.990 held
- RETENTION c5 back: ratio 0.706 -> 0.707 held
- RETENTION c5 turn: survival 0.990 -> 0.925 held
- RETENTION c5 turn: facing_closed 0.492 -> 0.671 held
- RETENTION c5 move_turn: survival 0.870 -> 0.685 DROPPED 0.185 (>0.1)
- RETENTION c5 move_turn: facing_closed 0.290 -> 0.193 held
- RETENTION c5 move_turn: ratio 0.497 -> 0.311 DROPPED 37% (>20%)
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.990 (bound 1.013, n_env 100, prev >= 0.3); fwd 0.950 (bound 1.001, n_env 100, prev >= 0.3); side 0.975 (bound 1.011, n_env 100, prev >= 0.3); back 0.990 (bound 1.013, n_env 100, prev >= 0.3); diag 0.870 (bound 0.948, n_env 100, new >= 0.1); turn 0.925 (bound 0.986, n_env 100, prev >= 0.3); move_turn 0.685 (bound 0.793, n_env 100, prev >= 0.3)

>>> ENTRY X02:c5

### X02:c6 — X0 s2 (baseline) — 2026-09-23T23:18:27

- recipe `f8f4ad5d955e` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-23_22-25-37_seed2_X02_c6_A15pB/model_7996.pt` | resume `model_5997.pt` | 2000 its | wall 0.77 h
- smoke fail@2k 0.299 fail@end 0.299 coll@2k 0.000 ep_len@2k 1694.173
- training translation floor: track_ratio_mean (last 500 its) 0.594 (gate floor 0.6)
- exposure (ring): reach_edge 0.492 | reach_obst 0.479 | field_frac 0.163 (steps 307.982) | goals_passed 0.641
- terrain-promotion telemetry: track_ratio_mean 0.595 (move rows only 0.604) | heading_err_rms_mean 0.166 | block_steps_mean 101.977 | r_max_mean 2.297
- failure share flat 0.307 | obst 0.277 | campaign-wide 0.323
- slot fractions (logged): stand 0.178 | move 0.537 | turn 0.061 | move_turn 0.101 | rsi_frac_actual 0.198 | terrain_levels 1.027 | mean ep len 1712.698 | prints 2000
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
- suite stand: n 100 surv 0.990 drift 0.090 [decisive]
- suite fwd: n 100 surv 0.950 ratio 0.740 tripod 0.618 [decisive]
- suite side: n 200 surv 0.985 ratio 0.389 tripod 0.502 (l: surv 0.980 ratio 0.398 tripod 0.505; r: surv 0.990 ratio 0.381 tripod 0.499) [decisive]
- suite back: n 100 surv 0.920 ratio 0.696 tripod 0.622 [decisive]
- suite diag: n 100 surv 0.740 ratio 0.544 tripod 0.404 [decisive]
- suite turn: n 200 surv 0.985 closed 0.560 progress 0.074 (l: surv 1.000 closed 0.572 progress 0.076; r: surv 0.970 closed 0.534 progress 0.071) [decisive]
- suite move_turn: n 200 surv 0.980 closed 0.365 progress 0.048 ratio 0.514 (l: surv 0.970 closed 0.359 progress 0.047 ratio 0.506; r: surv 0.990 closed 0.377 progress 0.049 ratio 0.516) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.404 (need >= 0.4; by hold: back 0.622, diag 0.404, fwd 0.618, lat_l 0.505, lat_r 0.499, move_turn 0.584, move_turn_r 0.588, settle 0.547, stand 0.586, turn_l90 0.635, turn_r90 0.596), completion 0.950 (need >= 0.7), yaw-progress rate turn_l90 0.076, turn_r90 0.071 rad/s (net turn_l90 0.075, turn_r90 0.070) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.398, diag 0.544, back 0.696, move_turn 0.506 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.594 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c6 stand: survival 0.990 -> 0.990 held
- RETENTION c6 stand: drift_speed 0.082 -> 0.090 held
- RETENTION c6 fwd: survival 0.850 -> 0.950 held
- RETENTION c6 fwd: ratio 0.733 -> 0.740 held
- RETENTION c6 side: survival 0.970 -> 0.985 held
- RETENTION c6 side: ratio 0.396 -> 0.389 held
- RETENTION c6 back: survival 0.880 -> 0.920 held
- RETENTION c6 back: ratio 0.746 -> 0.696 held
- RETENTION c6 diag: survival 0.720 -> 0.740 held
- RETENTION c6 diag: ratio 0.539 -> 0.544 held
- RETENTION c6 turn: survival 0.985 -> 0.985 held
- RETENTION c6 turn: facing_closed 0.629 -> 0.560 held
- RETENTION c6 move_turn: survival 0.745 -> 0.980 held
- RETENTION c6 move_turn: facing_closed 0.330 -> 0.365 held
- RETENTION c6 move_turn: ratio 0.498 -> 0.514 held
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.990 (bound 1.013, n_env 100, min >= 0.79); fwd 0.950 (bound 1.001, n_env 100, min >= 0.72); side 0.985 (bound 1.013, n_env 100, min >= 0.77); back 0.920 (bound 0.983, n_env 100, min >= 0.72); diag 0.740 (bound 0.842, n_env 100, min >= 0.59); turn 0.985 (bound 1.013, n_env 100, min >= 0.75); move_turn 0.980 (bound 1.013, n_env 100, min >= 0.55)

>>> ENTRY X02:c6

### experiment X02 — X0 s2 — ok — 2026-09-23T23:18:27

- recipe `f8f4ad5d955e` base `f8f4ad5d955e` chain `tail` seed 2
- stages ['X02:c4', 'X02:c5', 'X02:c6']
- screen: completion 0.950, tripod_dir 0.404, track tail 0.594
- replay: none

>>> ENTRY X02

### X03:c4 — A6000 s3 (candidate) — 2026-09-24T08:47:01

- recipe `8dffe9930f3d` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.15:0.1', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-24_08-05-40_seed3_X03_c4_A15pB/model_4498.pt` | resume `model_2999.pt` | 1500 its | wall 0.57 h
- smoke fail@2k 0.280 fail@end 0.280 coll@2k 0.000 ep_len@2k 1747.022
- training translation floor: track_ratio_mean (last 500 its) 0.591 (gate floor 0.6)
- exposure (ring): reach_edge 0.467 | reach_obst 0.404 | field_frac 0.162 (steps 314.801) | goals_passed 0.598
- terrain-promotion telemetry: track_ratio_mean 0.588 (move rows only 0.594) | heading_err_rms_mean 0.155 | block_steps_mean 82.244 | r_max_mean 2.295
- failure share flat 0.262 | obst 0.258 | campaign-wide 0.263
- slot fractions (logged): stand 0.144 | move 0.564 | turn 0.115 | move_turn 0.075 | rsi_frac_actual 0.189 | terrain_levels 0.973 | mean ep len 1758.769 | prints 1500
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
- suite trained categories: decisive stand, fwd, side, back, turn, move_turn
- suite stand: n 100 surv 0.970 drift 0.091 [decisive]
- suite fwd: n 100 surv 0.820 ratio 0.636 tripod 0.523 [decisive]
- suite side: n 200 surv 0.940 ratio 0.436 tripod 0.456 (l: surv 0.910 ratio 0.437 tripod 0.466; r: surv 0.970 ratio 0.434 tripod 0.446) [decisive]
- suite back: n 100 surv 0.760 ratio 0.642 tripod 0.520 [decisive]
- suite diag: n 100 surv 0.670 ratio 0.553 tripod 0.398 [WATCH: untrained]
- suite turn: n 200 surv 0.950 closed 0.588 progress 0.080 (l: surv 0.950 closed 0.585 progress 0.080; r: surv 0.950 closed 0.596 progress 0.079) [decisive]
- suite move_turn: n 200 surv 0.800 closed 0.138 progress 0.021 ratio 0.438 (l: surv 0.750 closed 0.124 progress 0.020 ratio 0.449; r: surv 0.850 closed 0.151 progress 0.022 ratio 0.433) [decisive]
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.15:0.1 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.970 (bound 1.010, n_env 100, min >= 0.69); fwd 0.820 (bound 0.909, n_env 100, min >= 0.52); side 0.940 (bound 0.995, n_env 100, min >= 0.65); back 0.760 (bound 0.859, n_env 100, min >= 0.55); turn 0.950 (bound 1.001, n_env 100, min >= 0.62); move_turn 0.800 (bound 0.893, n_env 100, min >= 0.315)

>>> ENTRY X03:c4

### X03:c5 — A6000 s3 (candidate) — 2026-09-24T09:28:32

- recipe `8dffe9930f3d` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.1:0.15', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-24_08-47-11_seed3_X03_c5_A15pB/model_5997.pt` | resume `model_4498.pt` | 1500 its | wall 0.57 h
- smoke fail@2k 0.253 fail@end 0.253 coll@2k 0.000 ep_len@2k 1774.584
- training translation floor: track_ratio_mean (last 500 its) 0.576 (gate floor 0.6)
- exposure (ring): reach_edge 0.496 | reach_obst 0.400 | field_frac 0.165 (steps 313.281) | goals_passed 0.638
- terrain-promotion telemetry: track_ratio_mean 0.576 (move rows only 0.582) | heading_err_rms_mean 0.156 | block_steps_mean 83.150 | r_max_mean 2.241
- failure share flat 0.265 | obst 0.280 | campaign-wide 0.282
- slot fractions (logged): stand 0.098 | move 0.546 | turn 0.136 | move_turn 0.114 | rsi_frac_actual 0.188 | terrain_levels 0.973 | mean ep len 1778.332 | prints 1500
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
- suite fwd: n 100 surv 0.900 ratio 0.658 tripod 0.590 [decisive]
- suite side: n 200 surv 0.980 ratio 0.374 tripod 0.483 (l: surv 0.990 ratio 0.371 tripod 0.485; r: surv 0.970 ratio 0.377 tripod 0.481) [decisive]
- suite back: n 100 surv 0.960 ratio 0.663 tripod 0.606 [decisive]
- suite diag: n 100 surv 0.810 ratio 0.535 tripod 0.441 [decisive]
- suite turn: n 200 surv 0.955 closed 0.484 progress 0.065 (l: surv 0.940 closed 0.486 progress 0.066; r: surv 0.970 closed 0.480 progress 0.063) [decisive]
- suite move_turn: n 200 surv 0.870 closed 0.152 progress 0.021 ratio 0.373 (l: surv 0.850 closed 0.150 progress 0.021 ratio 0.373; r: surv 0.890 closed 0.152 progress 0.021 ratio 0.374) [decisive]
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.1:0.15 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c5 stand: survival 1.000 -> 0.970 held
- RETENTION c5 stand: drift_speed 0.090 -> 0.083 held
- RETENTION c5 fwd: survival 0.880 -> 0.900 held
- RETENTION c5 fwd: ratio 0.718 -> 0.658 held
- RETENTION c5 side: survival 0.980 -> 0.980 held
- RETENTION c5 side: ratio 0.404 -> 0.374 held
- RETENTION c5 back: survival 0.870 -> 0.960 held
- RETENTION c5 back: ratio 0.706 -> 0.663 held
- RETENTION c5 turn: survival 0.990 -> 0.955 held
- RETENTION c5 turn: facing_closed 0.492 -> 0.484 held
- RETENTION c5 move_turn: survival 0.870 -> 0.870 held
- RETENTION c5 move_turn: facing_closed 0.290 -> 0.152 DROPPED 0.138 absolute (>0.100 = max(0.1, R_eval))
- RETENTION c5 move_turn: ratio 0.497 -> 0.373 DROPPED 25% (>20%)
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.970 (bound 1.010, n_env 100, prev >= 0.3); fwd 0.900 (bound 0.970, n_env 100, prev >= 0.3); side 0.980 (bound 1.013, n_env 100, prev >= 0.3); back 0.960 (bound 1.006, n_env 100, prev >= 0.3); diag 0.810 (bound 0.901, n_env 100, new >= 0.1); turn 0.955 (bound 1.003, n_env 100, prev >= 0.3); move_turn 0.870 (bound 0.948, n_env 100, prev >= 0.3)

>>> ENTRY X03:c5

### X03:c6 — A6000 s3 (candidate) — 2026-09-24T10:25:33

- recipe `8dffe9930f3d` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-24_09-28-42_seed3_X03_c6_A15pB/model_7996.pt` | resume `model_5997.pt` | 2000 its | wall 0.83 h
- smoke fail@2k 0.380 fail@end 0.380 coll@2k 0.000 ep_len@2k 1602.799
- training translation floor: track_ratio_mean (last 500 its) 0.612 (gate floor 0.6)
- exposure (ring): reach_edge 0.466 | reach_obst 0.378 | field_frac 0.139 (steps 259.911) | goals_passed 0.596
- terrain-promotion telemetry: track_ratio_mean 0.627 (move rows only 0.631) | heading_err_rms_mean 0.167 | block_steps_mean 81.821 | r_max_mean 2.316
- failure share flat 0.344 | obst 0.381 | campaign-wide 0.351
- slot fractions (logged): stand 0.148 | move 0.483 | turn 0.072 | move_turn 0.135 | rsi_frac_actual 0.196 | terrain_levels 0.973 | mean ep len 1674.753 | prints 2000
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
- suite stand: n 100 surv 1.000 drift 0.084 [decisive]
- suite fwd: n 100 surv 0.990 ratio 0.694 tripod 0.642 [decisive]
- suite side: n 200 surv 0.995 ratio 0.435 tripod 0.461 (l: surv 1.000 ratio 0.421 tripod 0.470; r: surv 0.990 ratio 0.448 tripod 0.435) [decisive]
- suite back: n 100 surv 1.000 ratio 0.725 tripod 0.646 [decisive]
- suite diag: n 100 surv 0.870 ratio 0.551 tripod 0.436 [decisive]
- suite turn: n 200 surv 1.000 closed 0.704 progress 0.092 (l: surv 1.000 closed 0.705 progress 0.092; r: surv 1.000 closed 0.700 progress 0.092) [decisive]
- suite move_turn: n 200 surv 0.730 closed 0.330 progress 0.047 ratio 0.510 (l: surv 0.740 closed 0.306 progress 0.046 ratio 0.510; r: surv 0.720 closed 0.351 progress 0.050 ratio 0.509) [decisive]
- WATCH gate_1c: PASS -- formation gate passed — walking-hold tripod 0.435 (by hold: back 0.646, diag 0.436, fwd 0.642, lat_l 0.470, lat_r 0.435, move_turn 0.466, move_turn_r 0.489, settle 0.527, stand 0.692, turn_l90 0.559, turn_r90 0.533), completion 0.931, yaw-progress rate turn_l90 0.092, turn_r90 0.092 rad/s (net turn_l90 0.092, turn_r90 0.092), per-direction speed_ratio lat_l 0.421, diag 0.551, back 0.725, move_turn 0.510 on slow__omni, training track_ratio_mean 0.612 (need >= 0.6 over the last 500 its)
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c6 stand: survival 0.990 -> 1.000 held
- RETENTION c6 stand: drift_speed 0.082 -> 0.084 held
- RETENTION c6 fwd: survival 0.850 -> 0.990 held
- RETENTION c6 fwd: ratio 0.733 -> 0.694 held
- RETENTION c6 side: survival 0.970 -> 0.995 held
- RETENTION c6 side: ratio 0.396 -> 0.435 held
- RETENTION c6 back: survival 0.880 -> 1.000 held
- RETENTION c6 back: ratio 0.746 -> 0.725 held
- RETENTION c6 diag: survival 0.720 -> 0.870 held
- RETENTION c6 diag: ratio 0.539 -> 0.551 held
- RETENTION c6 turn: survival 0.985 -> 1.000 held
- RETENTION c6 turn: facing_closed 0.629 -> 0.704 held
- RETENTION c6 move_turn: survival 0.745 -> 0.730 held
- RETENTION c6 move_turn: facing_closed 0.330 -> 0.330 held
- RETENTION c6 move_turn: ratio 0.498 -> 0.510 held
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, min >= 0.79); fwd 0.990 (bound 1.013, n_env 100, min >= 0.72); side 0.995 (bound 1.011, n_env 100, min >= 0.77); back 1.000 (bound 1.000, n_env 100, min >= 0.72); diag 0.870 (bound 0.948, n_env 100, min >= 0.59); turn 1.000 (bound 1.000, n_env 100, min >= 0.75); move_turn 0.730 (bound 0.833, n_env 100, min >= 0.55)

>>> ENTRY X03:c6

## D05 — A6000 — REJECT (clear, decided) — 2026-09-24T10:25:33

- question: Does removing both clock rewards (to eps 0.001) and the clock observation (zeroed, width kept) from the c5 -> c6 boundary on -- 2000 clock-free iterations, c6's omni_turn arriving clock-free -- make the gait less rigid (RIG BETTER with both its steps and support parts >= 0: fewer steps per metre and per radian, more loaded feet) without losing facing, turning, speed, slip or survival against the clock-held twin X0? The pilot arm: a seed-3 RIG TIE parks the driver (S3p), because nothing yet shows that removal moves steps or support (n = 0).
- experiment `X03` recipe `8dffe9930f3d` base `f8f4ad5d955e` control `X01`
- why: WORSE on guard cat.side.foot_slip, guard cat.turn.foot_slip
- block F (F): z 1.537 vs R 1.280 -> BETTER (k=2, missing [])
  - cat.turn.facing_closed: 0.629 -> 0.704 (delta 0.075, R_eval 0.050, R_seed 0.068, z 1.508)
  - cat.turn.yaw_progress: 0.082 -> 0.092 (delta 0.010, R_eval 0.007, R_seed 0.008, z 1.565)
- block F_mt (F_mt): z -0.060 vs R 0.912 -> TIE (k=2, missing [])
  - cat.move_turn.facing_closed: 0.330 -> 0.330 (delta 0.000, R_eval 0.040, R_seed 0.035, z 0.008)
  - cat.move_turn.yaw_progress: 0.048 -> 0.047 (delta -0.001, R_eval 0.006, R_seed 0.000, z -0.128)
- block T (T): z 0.326 vs R 0.487 -> TIE (k=4, missing [])
  - cat.fwd.ratio: 0.733 -> 0.694 (delta -0.039, R_eval 0.041, R_seed 0.008, z -0.949)
  - cat.side.ratio: 0.396 -> 0.435 (delta 0.039, R_eval 0.018, R_seed 0.007, z 2.180)
  - cat.back.ratio: 0.746 -> 0.725 (delta -0.021, R_eval 0.053, R_seed 0.050, z -0.394)
  - cat.diag.ratio: 0.539 -> 0.551 (delta 0.012, R_eval 0.026, R_seed 0.005, z 0.465)
- block RIG (RIG): z 0.389 vs R 0.786 -> TIE (k=10, missing [])
  - cat.fwd.steps_per_m: 16.339 -> 16.905 (lower is better, signed delta -0.566, R_eval 1.635, R_seed 1.206, z -0.346)
  - cat.fwd.support_feet: 3.061 -> 3.087 (delta 0.027, R_eval 0.030, R_seed 0.052, z 0.897)
  - cat.side.steps_per_m: 35.112 -> 36.221 (lower is better, signed delta -1.109, R_eval 1.377, R_seed 0.383, z -0.806)
  - cat.side.support_feet: 3.528 -> 3.564 (delta 0.036, R_eval 0.028, R_seed 0.010, z 1.303)
  - cat.back.steps_per_m: 17.352 -> 18.166 (lower is better, signed delta -0.814, R_eval 2.032, R_seed 0.268, z -0.401)
  - cat.back.support_feet: 3.050 -> 3.091 (delta 0.041, R_eval 0.030, R_seed 0.034, z 1.380)
  - cat.diag.steps_per_m: 25.117 -> 24.697 (lower is better, signed delta 0.420, R_eval 1.623, R_seed 0.558, z 0.259)
  - cat.diag.support_feet: 3.340 -> 3.315 (delta -0.025, R_eval 0.074, R_seed 0.001, z -0.339)
  - cat.turn.steps_per_rad: 39.863 -> 31.407 (lower is better, signed delta 8.456, R_eval 3.898, R_seed 4.840, z 2.169)
  - cat.turn.support_feet: 3.187 -> 3.182 (delta -0.005, R_eval 0.022, R_seed 0.090, z -0.231)
- guard cat.stand.survival: 0.990 -> 1.000 (R 0.072) -> TIE
- guard cat.stand.steps_per_s: 3.500 -> 2.750 (R 0.333) -> TIE
- guard cat.fwd.survival: 0.850 -> 0.990 (R 0.110) -> TIE
- guard cat.fwd.ratio: 0.733 -> 0.694 (R 0.051) -> TIE
- guard cat.fwd.foot_slip: 0.147 -> 0.138 (R 0.008) -> TIE
- guard cat.fwd.steps_per_m: 16.339 -> 16.905 (R 1.933) -> TIE
- guard cat.side.survival: 0.970 -> 0.995 (R 0.051) -> TIE
- guard cat.side.ratio: 0.396 -> 0.435 (R 0.021) -> TIE
- guard cat.side.foot_slip: 0.173 -> 0.190 (R 0.005) -> WORSE
- guard cat.side.steps_per_m: 35.112 -> 36.221 (R 1.569) -> TIE
- guard cat.back.survival: 0.880 -> 1.000 (R 0.105) -> TIE
- guard cat.back.ratio: 0.746 -> 0.725 (R 0.062) -> TIE
- guard cat.back.foot_slip: 0.143 -> 0.135 (R 0.007) -> TIE
- guard cat.back.steps_per_m: 17.352 -> 18.166 (R 2.395) -> TIE
- guard cat.diag.survival: 0.720 -> 0.870 (R 0.140) -> TIE
- guard cat.diag.ratio: 0.539 -> 0.551 (R 0.029) -> TIE
- guard cat.diag.foot_slip: 0.186 -> 0.182 (R 0.010) -> TIE
- guard cat.diag.steps_per_m: 25.117 -> 24.697 (R 2.024) -> TIE
- guard cat.turn.survival: 0.985 -> 1.000 (R 0.058) -> TIE
- guard cat.turn.foot_slip: 0.115 -> 0.121 (R 0.004) -> WORSE
- guard cat.turn.steps_per_rad: 39.863 -> 31.407 (R 4.840) -> TIE
- guard cat.move_turn.survival: 0.745 -> 0.730 (R 0.235) -> TIE
- guard cat.move_turn.ratio: 0.498 -> 0.510 (R 0.037) -> TIE
- anchored drift: FAIL — drifted below the anchor on guard cat.side.foot_slip, guard cat.turn.foot_slip

>>> ENTRY D05

### experiment X03 — A6000 s3 — ok — 2026-09-24T10:25:33

- recipe `8dffe9930f3d` base `f8f4ad5d955e` chain `tail` seed 3
- stages ['X03:c4', 'X03:c5', 'X03:c6']
- screen: completion 0.931, tripod_dir 0.436, track tail 0.612
- replay: none

>>> ENTRY X03

### X04:c4 — A4500 s3 (candidate) — 2026-09-24T11:08:03

- recipe `689e53da26de` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.15:0.1', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-24_10-27-12_seed3_X04_c4_A15pB/model_4498.pt` | resume `model_2999.pt` | 1500 its | wall 0.57 h
- smoke fail@2k 0.212 fail@end 0.212 coll@2k 0.000 ep_len@2k 1830.111
- training translation floor: track_ratio_mean (last 500 its) 0.600 (gate floor 0.6)
- exposure (ring): reach_edge 0.481 | reach_obst 0.379 | field_frac 0.161 (steps 304.699) | goals_passed 0.608
- terrain-promotion telemetry: track_ratio_mean 0.594 (move rows only 0.599) | heading_err_rms_mean 0.153 | block_steps_mean 80.489 | r_max_mean 2.344
- failure share flat 0.226 | obst 0.319 | campaign-wide 0.191
- slot fractions (logged): stand 0.143 | move 0.531 | turn 0.153 | move_turn 0.115 | rsi_frac_actual 0.183 | terrain_levels 0.973 | mean ep len 1854.399 | prints 1500
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
- suite trained categories: decisive stand, fwd, side, back, turn, move_turn
- suite stand: n 100 surv 1.000 drift 0.097 [decisive]
- suite fwd: n 100 surv 0.870 ratio 0.698 tripod 0.577 [decisive]
- suite side: n 200 surv 0.940 ratio 0.472 tripod 0.463 (l: surv 0.920 ratio 0.473 tripod 0.475; r: surv 0.960 ratio 0.468 tripod 0.455) [decisive]
- suite back: n 100 surv 0.850 ratio 0.633 tripod 0.525 [decisive]
- suite diag: n 100 surv 0.620 ratio 0.587 tripod 0.418 [WATCH: untrained]
- suite turn: n 200 surv 0.975 closed 0.505 progress 0.067 (l: surv 0.980 closed 0.508 progress 0.067; r: surv 0.970 closed 0.497 progress 0.066) [decisive]
- suite move_turn: n 200 surv 0.795 closed 0.134 progress 0.018 ratio 0.473 (l: surv 0.740 closed 0.145 progress 0.022 ratio 0.476; r: surv 0.850 closed 0.120 progress 0.016 ratio 0.471) [decisive]
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.15:0.1 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, min >= 0.69); fwd 0.870 (bound 0.948, n_env 100, min >= 0.52); side 0.940 (bound 0.995, n_env 100, min >= 0.65); back 0.850 (bound 0.933, n_env 100, min >= 0.55); turn 0.975 (bound 1.011, n_env 100, min >= 0.62); move_turn 0.795 (bound 0.889, n_env 100, min >= 0.315)

>>> ENTRY X04:c4

### X04:c5 — A4500 s3 (candidate) — 2026-09-24T11:51:34

- recipe `689e53da26de` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.1:0.15', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-24_11-08-13_seed3_X04_c5_A15pB/model_5997.pt` | resume `model_4498.pt` | 1500 its | wall 0.60 h
- smoke fail@2k 0.362 fail@end 0.362 coll@2k 0.000 ep_len@2k 1582.459
- training translation floor: track_ratio_mean (last 500 its) 0.653 (gate floor 0.6)
- exposure (ring): reach_edge 0.446 | reach_obst 0.327 | field_frac 0.144 (steps 266.772) | goals_passed 0.592
- terrain-promotion telemetry: track_ratio_mean 0.651 (move rows only 0.652) | heading_err_rms_mean 0.150 | block_steps_mean 78.523 | r_max_mean 2.371
- failure share flat 0.387 | obst 0.451 | campaign-wide 0.393
- slot fractions (logged): stand 0.149 | move 0.449 | turn 0.108 | move_turn 0.132 | rsi_frac_actual 0.210 | terrain_levels 0.973 | mean ep len 1507.074 | prints 1500
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
- suite stand: n 100 surv 1.000 drift 0.092 [decisive]
- suite fwd: n 100 surv 0.970 ratio 0.714 tripod 0.619 [decisive]
- suite side: n 200 surv 0.995 ratio 0.477 tripod 0.425 (l: surv 0.990 ratio 0.478 tripod 0.431; r: surv 1.000 ratio 0.474 tripod 0.419) [decisive]
- suite back: n 100 surv 0.930 ratio 0.713 tripod 0.632 [decisive]
- suite diag: n 100 surv 0.800 ratio 0.605 tripod 0.417 [decisive]
- suite turn: n 200 surv 1.000 closed 0.569 progress 0.075 (l: surv 1.000 closed 0.560 progress 0.074; r: surv 1.000 closed 0.587 progress 0.076) [decisive]
- suite move_turn: n 200 surv 0.440 closed 0.320 progress 0.055 ratio 0.565 (l: surv 0.420 closed 0.317 progress 0.052 ratio 0.563; r: surv 0.460 closed 0.326 progress 0.059 ratio 0.566) [decisive]
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.1:0.15 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c5 stand: survival 1.000 -> 1.000 held
- RETENTION c5 stand: drift_speed 0.090 -> 0.092 held
- RETENTION c5 fwd: survival 0.880 -> 0.970 held
- RETENTION c5 fwd: ratio 0.718 -> 0.714 held
- RETENTION c5 side: survival 0.980 -> 0.995 held
- RETENTION c5 side: ratio 0.404 -> 0.477 held
- RETENTION c5 back: survival 0.870 -> 0.930 held
- RETENTION c5 back: ratio 0.706 -> 0.713 held
- RETENTION c5 turn: survival 0.990 -> 1.000 held
- RETENTION c5 turn: facing_closed 0.492 -> 0.569 held
- RETENTION c5 move_turn: survival 0.870 -> 0.440 DROPPED 0.430 (>0.1)
- RETENTION c5 move_turn: facing_closed 0.290 -> 0.320 held
- RETENTION c5 move_turn: ratio 0.497 -> 0.565 held
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, prev >= 0.3); fwd 0.970 (bound 1.010, n_env 100, prev >= 0.3); side 0.995 (bound 1.011, n_env 100, prev >= 0.3); back 0.930 (bound 0.989, n_env 100, prev >= 0.3); diag 0.800 (bound 0.893, n_env 100, new >= 0.1); turn 1.000 (bound 1.000, n_env 100, prev >= 0.3); move_turn 0.440 (bound 0.555, n_env 100, prev >= 0.3)

>>> ENTRY X04:c5

### X04:c6 — A4500 s3 (candidate) — 2026-09-24T12:46:35

- recipe `689e53da26de` on chain `tail`, base `f8f4ad5d955e`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-24_11-51-44_seed3_X04_c6_A15pB/model_7996.pt` | resume `model_5997.pt` | 2000 its | wall 0.80 h
- smoke fail@2k 0.335 fail@end 0.335 coll@2k 0.000 ep_len@2k 1665.690
- training translation floor: track_ratio_mean (last 500 its) 0.649 (gate floor 0.6)
- exposure (ring): reach_edge 0.550 | reach_obst 0.437 | field_frac 0.190 (steps 350.284) | goals_passed 0.736
- terrain-promotion telemetry: track_ratio_mean 0.644 (move rows only 0.650) | heading_err_rms_mean 0.159 | block_steps_mean 87.746 | r_max_mean 2.429
- failure share flat 0.395 | obst 0.434 | campaign-wide 0.384
- slot fractions (logged): stand 0.161 | move 0.442 | turn 0.049 | move_turn 0.227 | rsi_frac_actual 0.188 | terrain_levels 0.973 | mean ep len 1607.252 | prints 2000
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
- suite stand: n 100 surv 1.000 drift 0.083 [decisive]
- suite fwd: n 100 surv 0.990 ratio 0.716 tripod 0.650 [decisive]
- suite side: n 200 surv 1.000 ratio 0.501 tripod 0.445 (l: surv 1.000 ratio 0.500 tripod 0.455; r: surv 1.000 ratio 0.502 tripod 0.432) [decisive]
- suite back: n 100 surv 0.990 ratio 0.724 tripod 0.636 [decisive]
- suite diag: n 100 surv 0.820 ratio 0.590 tripod 0.423 [decisive]
- suite turn: n 200 surv 1.000 closed 0.726 progress 0.096 (l: surv 1.000 closed 0.725 progress 0.096; r: surv 1.000 closed 0.729 progress 0.097) [decisive]
- suite move_turn: n 200 surv 0.590 closed 0.450 progress 0.068 ratio 0.559 (l: surv 0.630 closed 0.475 progress 0.069 ratio 0.551; r: surv 0.550 closed 0.425 progress 0.067 ratio 0.561) [decisive]
- WATCH gate_1c: PASS -- formation gate passed — walking-hold tripod 0.423 (by hold: back 0.636, diag 0.423, fwd 0.650, lat_l 0.455, lat_r 0.432, move_turn 0.485, move_turn_r 0.489, settle 0.594, stand 0.711, turn_l90 0.520, turn_r90 0.508), completion 0.898, yaw-progress rate turn_l90 0.096, turn_r90 0.097 rad/s (net turn_l90 0.095, turn_r90 0.096), per-direction speed_ratio lat_l 0.500, diag 0.590, back 0.724, move_turn 0.551 on slow__omni, training track_ratio_mean 0.649 (need >= 0.6 over the last 500 its)
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c6 stand: survival 0.990 -> 1.000 held
- RETENTION c6 stand: drift_speed 0.082 -> 0.083 held
- RETENTION c6 fwd: survival 0.850 -> 0.990 held
- RETENTION c6 fwd: ratio 0.733 -> 0.716 held
- RETENTION c6 side: survival 0.970 -> 1.000 held
- RETENTION c6 side: ratio 0.396 -> 0.501 held
- RETENTION c6 back: survival 0.880 -> 0.990 held
- RETENTION c6 back: ratio 0.746 -> 0.724 held
- RETENTION c6 diag: survival 0.720 -> 0.820 held
- RETENTION c6 diag: ratio 0.539 -> 0.590 held
- RETENTION c6 turn: survival 0.985 -> 1.000 held
- RETENTION c6 turn: facing_closed 0.629 -> 0.726 held
- RETENTION c6 move_turn: survival 0.745 -> 0.590 DROPPED 0.155 (>0.1)
- RETENTION c6 move_turn: facing_closed 0.330 -> 0.450 held
- RETENTION c6 move_turn: ratio 0.498 -> 0.559 held
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, min >= 0.79); fwd 0.990 (bound 1.013, n_env 100, min >= 0.72); side 1.000 (bound 1.000, n_env 100, min >= 0.77); back 0.990 (bound 1.013, n_env 100, min >= 0.72); diag 0.820 (bound 0.909, n_env 100, min >= 0.59); turn 1.000 (bound 1.000, n_env 100, min >= 0.75); move_turn 0.590 (bound 0.704, n_env 100, min >= 0.55)

>>> ENTRY X04:c6

## D06 — A4500 — REJECT (clear, decided) — 2026-09-24T12:46:35

- question: The same removal from the c4 -> c5 boundary on: 3500 clock-free iterations, c5's omni and c6's omni_turn arriving clock-free. An alternative to A6000, judged against X0, never composed with it.
- experiment `X04` recipe `689e53da26de` base `f8f4ad5d955e` control `X01`
- why: WORSE on guard cat.side.foot_slip, guard cat.turn.foot_slip
- block F (F): z 2.067 vs R 1.280 -> BETTER (k=2, missing [])
  - cat.turn.facing_closed: 0.629 -> 0.726 (delta 0.098, R_eval 0.050, R_seed 0.068, z 1.963)
  - cat.turn.yaw_progress: 0.082 -> 0.096 (delta 0.014, R_eval 0.007, R_seed 0.008, z 2.171)
- block F_mt (F_mt): z 3.188 vs R 0.912 -> BETTER (k=2, missing [])
  - cat.move_turn.facing_closed: 0.330 -> 0.450 (delta 0.119, R_eval 0.040, R_seed 0.035, z 2.963)
  - cat.move_turn.yaw_progress: 0.048 -> 0.068 (delta 0.020, R_eval 0.006, R_seed 0.000, z 3.413)
- block T (T): z 1.757 vs R 0.487 -> BETTER (k=4, missing [])
  - cat.fwd.ratio: 0.733 -> 0.716 (delta -0.017, R_eval 0.041, R_seed 0.008, z -0.414)
  - cat.side.ratio: 0.396 -> 0.501 (delta 0.105, R_eval 0.018, R_seed 0.007, z 5.845)
  - cat.back.ratio: 0.746 -> 0.724 (delta -0.021, R_eval 0.053, R_seed 0.050, z -0.404)
  - cat.diag.ratio: 0.539 -> 0.590 (delta 0.051, R_eval 0.026, R_seed 0.005, z 2.000)
- block RIG (RIG): z 0.973 vs R 0.786 -> BETTER (k=10, missing [])
  - cat.fwd.steps_per_m: 16.339 -> 16.164 (lower is better, signed delta 0.175, R_eval 1.635, R_seed 1.206, z 0.107)
  - cat.fwd.support_feet: 3.061 -> 3.102 (delta 0.041, R_eval 0.030, R_seed 0.052, z 1.369)
  - cat.side.steps_per_m: 35.112 -> 31.145 (lower is better, signed delta 3.966, R_eval 1.377, R_seed 0.383, z 2.881)
  - cat.side.support_feet: 3.528 -> 3.541 (delta 0.013, R_eval 0.028, R_seed 0.010, z 0.454)
  - cat.back.steps_per_m: 17.352 -> 18.782 (lower is better, signed delta -1.431, R_eval 2.032, R_seed 0.268, z -0.704)
  - cat.back.support_feet: 3.050 -> 3.087 (delta 0.038, R_eval 0.030, R_seed 0.034, z 1.267)
  - cat.diag.steps_per_m: 25.117 -> 22.552 (lower is better, signed delta 2.565, R_eval 1.623, R_seed 0.558, z 1.580)
  - cat.diag.support_feet: 3.340 -> 3.331 (delta -0.009, R_eval 0.074, R_seed 0.001, z -0.123)
  - cat.turn.steps_per_rad: 39.863 -> 30.057 (lower is better, signed delta 9.807, R_eval 3.898, R_seed 4.840, z 2.516)
  - cat.turn.support_feet: 3.187 -> 3.195 (delta 0.008, R_eval 0.022, R_seed 0.090, z 0.385)
- guard cat.stand.survival: 0.990 -> 1.000 (R 0.072) -> TIE
- guard cat.stand.steps_per_s: 3.500 -> 2.750 (R 0.333) -> TIE
- guard cat.fwd.survival: 0.850 -> 0.990 (R 0.110) -> TIE
- guard cat.fwd.ratio: 0.733 -> 0.716 (R 0.051) -> TIE
- guard cat.fwd.foot_slip: 0.147 -> 0.145 (R 0.008) -> TIE
- guard cat.fwd.steps_per_m: 16.339 -> 16.164 (R 1.933) -> TIE
- guard cat.side.survival: 0.970 -> 1.000 (R 0.051) -> TIE
- guard cat.side.ratio: 0.396 -> 0.501 (R 0.021) -> TIE
- guard cat.side.foot_slip: 0.173 -> 0.203 (R 0.005) -> WORSE
- guard cat.side.steps_per_m: 35.112 -> 31.145 (R 1.569) -> TIE
- guard cat.back.survival: 0.880 -> 0.990 (R 0.105) -> TIE
- guard cat.back.ratio: 0.746 -> 0.724 (R 0.062) -> TIE
- guard cat.back.foot_slip: 0.143 -> 0.141 (R 0.007) -> TIE
- guard cat.back.steps_per_m: 17.352 -> 18.782 (R 2.395) -> TIE
- guard cat.diag.survival: 0.720 -> 0.820 (R 0.140) -> TIE
- guard cat.diag.ratio: 0.539 -> 0.590 (R 0.029) -> TIE
- guard cat.diag.foot_slip: 0.186 -> 0.187 (R 0.010) -> TIE
- guard cat.diag.steps_per_m: 25.117 -> 22.552 (R 2.024) -> TIE
- guard cat.turn.survival: 0.985 -> 1.000 (R 0.058) -> TIE
- guard cat.turn.foot_slip: 0.115 -> 0.123 (R 0.004) -> WORSE
- guard cat.turn.steps_per_rad: 39.863 -> 30.057 (R 4.840) -> TIE
- guard cat.move_turn.survival: 0.745 -> 0.590 (R 0.235) -> TIE
- guard cat.move_turn.ratio: 0.498 -> 0.559 (R 0.037) -> TIE
- anchored drift: FAIL — drifted below the anchor on guard cat.side.foot_slip, guard cat.turn.foot_slip

>>> ENTRY D06

### experiment X04 — A4500 s3 — ok — 2026-09-24T12:46:35

- recipe `689e53da26de` base `f8f4ad5d955e` chain `tail` seed 3
- stages ['X04:c4', 'X04:c5', 'X04:c6']
- screen: completion 0.898, tripod_dir 0.423, track tail 0.649
- replay: none

>>> ENTRY X04

### X05:t3 — ref:scratch_free s3 (reference) — 2026-09-25T00:40:34

- recipe `fe162a31149a` on chain `scratch_free`, base `fe162a31149a`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-24_23-21-43_seed3_X05_t3_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.20 h
- smoke fail@2k 0.848 fail@end 0.829 coll@2k 0.000 ep_len@2k 865.486
- training translation floor: track_ratio_mean (last 500 its) 0.676 (gate floor 0.6)
- exposure (ring): reach_edge 0.110 | reach_obst 0.101 | field_frac 0.032 (steps 57.885) | goals_passed 0.136
- terrain-promotion telemetry: track_ratio_mean 0.671 (move rows only 0.671) | heading_err_rms_mean 0.111 | block_steps_mean 15.908 | r_max_mean 1.230
- failure share flat 0.825 | obst 0.835 | campaign-wide 0.830
- slot fractions (logged): stand 0.060 | move 0.250 | turn 0.087 | move_turn 0.000 | rsi_frac_actual 0.172 | terrain_levels 0.973 | mean ep len 913.077 | prints 3000
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
- suite trained categories: decisive stand, fwd, side, back, turn
- suite stand: n 100 surv 0.690 drift 0.098 [decisive]
- suite fwd: n 100 surv 0.000 ratio 0.660 tripod 0.193 [decisive]
- suite side: n 200 surv 0.695 ratio 0.498 tripod 0.385 (l: surv 0.720 ratio 0.508 tripod 0.380; r: surv 0.670 ratio 0.487 tripod 0.402) [decisive]
- suite back: n 100 surv 0.000 ratio 0.642 tripod 0.194 [decisive]
- suite diag: n 100 surv 0.010 ratio 0.688 tripod 0.249 [WATCH: untrained]
- suite turn: n 200 surv 0.750 closed 0.062 progress 0.009 (l: surv 0.700 closed 0.071 progress 0.010; r: surv 0.800 closed 0.061 progress 0.008) [decisive]
- suite move_turn: n 200 surv 0.000 closed 0.018 progress 0.009 ratio 0.656 (l: surv 0.000 closed 0.016 progress 0.008 ratio 0.648; r: surv 0.000 closed 0.027 progress 0.012 ratio 0.662) [WATCH: untrained]
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor FAIL: category survival on suite (one-sided 99 % band): stand 0.690 (bound 0.798, n_env 100, new >= 0.1); fwd 0.000 (bound 0.000, n_env 100, new >= 0.1); side 0.695 (bound 0.802, n_env 100, new >= 0.1); back 0.000 (bound 0.000, n_env 100, new >= 0.1); turn 0.750 (bound 0.851, n_env 100, new >= 0.1) -- FAIL on ['fwd', 'back']

>>> ENTRY X05:t3

### X05:c4 — ref:scratch_free s3 (reference) — 2026-09-25T01:23:35

- recipe `fe162a31149a` on chain `scratch_free`, base `fe162a31149a`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.15:0.1', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-25_00-40-45_seed3_X05_c4_A15pB/model_4498.pt` | resume `model_2999.pt` | 1500 its | wall 0.60 h
- smoke fail@2k 0.877 fail@end 0.877 coll@2k 0.000 ep_len@2k 825.584
- training translation floor: track_ratio_mean (last 500 its) 0.635 (gate floor 0.6)
- exposure (ring): reach_edge 0.111 | reach_obst 0.087 | field_frac 0.031 (steps 54.169) | goals_passed 0.126
- terrain-promotion telemetry: track_ratio_mean 0.650 (move rows only 0.646) | heading_err_rms_mean 0.131 | block_steps_mean 17.516 | r_max_mean 1.293
- failure share flat 0.867 | obst 0.851 | campaign-wide 0.855
- slot fractions (logged): stand 0.069 | move 0.238 | turn 0.089 | move_turn 0.027 | rsi_frac_actual 0.213 | terrain_levels 0.973 | mean ep len 833.585 | prints 1500
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
- suite trained categories: decisive stand, fwd, side, back, turn, move_turn
- suite stand: n 100 surv 0.820 drift 0.087 [decisive]
- suite fwd: n 100 surv 0.050 ratio 0.554 tripod 0.256 [decisive]
- suite side: n 200 surv 0.725 ratio 0.555 tripod 0.400 (l: surv 0.670 ratio 0.559 tripod 0.401; r: surv 0.780 ratio 0.553 tripod 0.399) [decisive]
- suite back: n 100 surv 0.130 ratio 0.581 tripod 0.250 [decisive]
- suite diag: n 100 surv 0.030 ratio 0.645 tripod 0.273 [WATCH: untrained]
- suite turn: n 200 surv 0.815 closed 0.254 progress 0.034 (l: surv 0.810 closed 0.239 progress 0.031; r: surv 0.820 closed 0.257 progress 0.036) [decisive]
- suite move_turn: n 200 surv 0.000 closed 0.027 progress 0.010 ratio 0.619 (l: surv 0.000 closed 0.005 progress 0.004 ratio 0.616; r: surv 0.000 closed 0.057 progress 0.028 ratio 0.619) [decisive]
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.15:0.1 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c4: no incumbent trace yet -- this run seeds it (7 categories recorded)
- floor FAIL: category survival on suite (one-sided 99 % band): stand 0.820 (bound 0.909, n_env 100, prev >= 0.3); fwd 0.050 (bound 0.101, n_env 100, prev >= 0.3); side 0.725 (bound 0.829, n_env 100, prev >= 0.3); back 0.130 (bound 0.208, n_env 100, prev >= 0.3); turn 0.815 (bound 0.905, n_env 100, prev >= 0.3); move_turn 0.000 (bound 0.000, n_env 100, new >= 0.1) -- FAIL on ['fwd', 'back', 'move_turn']

>>> ENTRY X05:c4

### X05:c5 — ref:scratch_free s3 (reference) — 2026-09-25T02:06:36

- recipe `fe162a31149a` on chain `scratch_free`, base `fe162a31149a`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.1:0.15', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-25_01-23-45_seed3_X05_c5_A15pB/model_5997.pt` | resume `model_4498.pt` | 1500 its | wall 0.60 h
- smoke fail@2k 0.866 fail@end 0.866 coll@2k 0.000 ep_len@2k 968.047
- training translation floor: track_ratio_mean (last 500 its) 0.653 (gate floor 0.6)
- exposure (ring): reach_edge 0.160 | reach_obst 0.105 | field_frac 0.042 (steps 68.099) | goals_passed 0.197
- terrain-promotion telemetry: track_ratio_mean 0.645 (move rows only 0.643) | heading_err_rms_mean 0.132 | block_steps_mean 15.182 | r_max_mean 1.502
- failure share flat 0.861 | obst 0.892 | campaign-wide 0.850
- slot fractions (logged): stand 0.079 | move 0.256 | turn 0.045 | move_turn 0.068 | rsi_frac_actual 0.219 | terrain_levels 0.973 | mean ep len 913.153 | prints 1500
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
- suite stand: n 100 surv 0.790 drift 0.082 [decisive]
- suite fwd: n 100 surv 0.120 ratio 0.478 tripod 0.246 [decisive]
- suite side: n 200 surv 0.775 ratio 0.562 tripod 0.379 (l: surv 0.790 ratio 0.562 tripod 0.375; r: surv 0.760 ratio 0.563 tripod 0.382) [decisive]
- suite back: n 100 surv 0.360 ratio 0.485 tripod 0.291 [decisive]
- suite diag: n 100 surv 0.250 ratio 0.554 tripod 0.257 [decisive]
- suite turn: n 200 surv 0.765 closed 0.357 progress 0.047 (l: surv 0.790 closed 0.340 progress 0.046; r: surv 0.740 closed 0.371 progress 0.048) [decisive]
- suite move_turn: n 200 surv 0.120 closed 0.061 progress 0.021 ratio 0.532 (l: surv 0.110 closed 0.079 progress 0.025 ratio 0.541; r: surv 0.130 closed 0.041 progress 0.016 ratio 0.523) [decisive]
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.1:0.15 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c5: no incumbent trace yet -- this run seeds it (7 categories recorded)
- floor PASS: no floor at this stage

>>> ENTRY X05:c5

### X05:c6 — ref:scratch_free s3 (reference) — 2026-09-25T02:59:37

- recipe `fe162a31149a` on chain `scratch_free`, base `fe162a31149a`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-25_02-06-46_seed3_X05_c6_A15pB/model_7996.pt` | resume `model_5997.pt` | 2000 its | wall 0.77 h
- smoke fail@2k 0.819 fail@end 0.819 coll@2k 0.000 ep_len@2k 1118.819
- training translation floor: track_ratio_mean (last 500 its) 0.648 (gate floor 0.6)
- exposure (ring): reach_edge 0.180 | reach_obst 0.105 | field_frac 0.045 (steps 76.630) | goals_passed 0.223
- terrain-promotion telemetry: track_ratio_mean 0.637 (move rows only 0.630) | heading_err_rms_mean 0.155 | block_steps_mean 17.154 | r_max_mean 1.624
- failure share flat 0.825 | obst 0.861 | campaign-wide 0.791
- slot fractions (logged): stand 0.113 | move 0.317 | turn 0.058 | move_turn 0.113 | rsi_frac_actual 0.203 | terrain_levels 0.973 | mean ep len 1177.253 | prints 2000
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
- suite stand: n 100 surv 0.810 drift 0.083 [decisive]
- suite fwd: n 100 surv 0.460 ratio 0.466 tripod 0.320 [decisive]
- suite side: n 200 surv 0.865 ratio 0.514 tripod 0.413 (l: surv 0.870 ratio 0.496 tripod 0.429; r: surv 0.860 ratio 0.522 tripod 0.399) [decisive]
- suite back: n 100 surv 0.560 ratio 0.478 tripod 0.386 [decisive]
- suite diag: n 100 surv 0.300 ratio 0.548 tripod 0.299 [decisive]
- suite turn: n 200 surv 0.805 closed 0.508 progress 0.065 (l: surv 0.790 closed 0.478 progress 0.063; r: surv 0.820 closed 0.513 progress 0.067) [decisive]
- suite move_turn: n 200 surv 0.550 closed 0.185 progress 0.032 ratio 0.483 (l: surv 0.540 closed 0.214 progress 0.033 ratio 0.479; r: surv 0.560 closed 0.179 progress 0.032 ratio 0.485) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.299 (need >= 0.4; by hold: back 0.386, diag 0.299, fwd 0.320, lat_l 0.429, lat_r 0.399, move_turn 0.414, move_turn_r 0.424, settle 0.398, stand 0.636, turn_l90 0.594, turn_r90 0.597), completion 0.656 (need >= 0.7), yaw-progress rate turn_l90 0.063, turn_r90 0.067 rad/s (net turn_l90 0.063, turn_r90 0.067) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.496, diag 0.548, back 0.478, move_turn 0.479 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.648 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c6: no incumbent trace yet -- this run seeds it (7 categories recorded)
- floor PASS: no floor at this stage

>>> ENTRY X05:c6

## HALT X06 — dead_plant — 2026-09-25T04:41:56

- t3 died dead_plant twice

### X07:t3 — ref:scratch_free_s2 s2 (reference) — 2026-09-25T06:02:56

- recipe `34c94d4fe704` on chain `scratch_free`, base `34c94d4fe704`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-25_04-48-05_seed2_X07_t3_A15pB_c/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.13 h
- smoke fail@2k 0.898 fail@end 0.867 coll@2k 0.000 ep_len@2k 738.703
- training translation floor: track_ratio_mean (last 500 its) 0.627 (gate floor 0.6)
- exposure (ring): reach_edge 0.076 | reach_obst 0.065 | field_frac 0.013 (steps 24.093) | goals_passed 0.088
- terrain-promotion telemetry: track_ratio_mean 0.632 (move rows only 0.632) | heading_err_rms_mean 0.118 | block_steps_mean 10.345 | r_max_mean 1.064
- failure share flat 0.869 | obst 0.856 | campaign-wide 0.831
- slot fractions (logged): stand 0.077 | move 0.208 | turn 0.091 | move_turn 0.000 | rsi_frac_actual 0.179 | terrain_levels 1.027 | mean ep len 781.248 | prints 2900
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
- suite trained categories: decisive stand, fwd, side, back, turn
- suite stand: n 100 surv 0.930 drift 0.088 [decisive]
- suite fwd: n 100 surv 0.000 ratio 0.755 tripod 0.271 [decisive]
- suite side: n 200 surv 0.615 ratio 0.456 tripod 0.514 (l: surv 0.600 ratio 0.457 tripod 0.512; r: surv 0.630 ratio 0.456 tripod 0.521) [decisive]
- suite back: n 100 surv 0.000 ratio 0.717 tripod 0.234 [decisive]
- suite diag: n 100 surv 0.000 ratio 0.823 tripod 0.233 [WATCH: untrained]
- suite turn: n 200 surv 0.925 closed 0.067 progress 0.009 (l: surv 0.900 closed 0.042 progress 0.006; r: surv 0.950 closed 0.089 progress 0.011) [decisive]
- suite move_turn: n 200 surv 0.000 closed 0.059 progress 0.034 ratio 0.677 (l: surv 0.000 closed 0.040 progress 0.027 ratio 0.690; r: surv 0.000 closed 0.070 progress 0.046 ratio 0.672) [WATCH: untrained]
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor FAIL: category survival on suite (one-sided 99 % band): stand 0.930 (bound 0.989, n_env 100, new >= 0.1); fwd 0.000 (bound 0.000, n_env 100, new >= 0.1); side 0.615 (bound 0.728, n_env 100, new >= 0.1); back 0.000 (bound 0.000, n_env 100, new >= 0.1); turn 0.925 (bound 0.986, n_env 100, new >= 0.1) -- FAIL on ['fwd', 'back']

>>> ENTRY X07:t3

### X07:c4 — ref:scratch_free_s2 s2 (reference) — 2026-09-25T06:45:57

- recipe `34c94d4fe704` on chain `scratch_free`, base `34c94d4fe704`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.15:0.1', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-25_06-03-07_seed2_X07_c4_A15pB/model_4498.pt` | resume `model_2999.pt` | 1500 its | wall 0.60 h
- smoke fail@2k 0.902 fail@end 0.902 coll@2k 0.000 ep_len@2k 767.236
- training translation floor: track_ratio_mean (last 500 its) 0.621 (gate floor 0.6)
- exposure (ring): reach_edge 0.074 | reach_obst 0.060 | field_frac 0.014 (steps 24.300) | goals_passed 0.083
- terrain-promotion telemetry: track_ratio_mean 0.614 (move rows only 0.611) | heading_err_rms_mean 0.130 | block_steps_mean 9.334 | r_max_mean 1.066
- failure share flat 0.898 | obst 0.891 | campaign-wide 0.905
- slot fractions (logged): stand 0.062 | move 0.192 | turn 0.080 | move_turn 0.037 | rsi_frac_actual 0.231 | terrain_levels 1.027 | mean ep len 774.828 | prints 1500
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
- suite trained categories: decisive stand, fwd, side, back, turn, move_turn
- suite stand: n 100 surv 1.000 drift 0.083 [decisive]
- suite fwd: n 100 surv 0.000 ratio 0.646 tripod 0.250 [decisive]
- suite side: n 200 surv 0.890 ratio 0.439 tripod 0.526 (l: surv 0.870 ratio 0.442 tripod 0.526; r: surv 0.910 ratio 0.434 tripod 0.524) [decisive]
- suite back: n 100 surv 0.000 ratio 0.598 tripod 0.369 [decisive]
- suite diag: n 100 surv 0.000 ratio 0.701 tripod 0.274 [WATCH: untrained]
- suite turn: n 200 surv 0.960 closed 0.043 progress 0.005 (l: surv 0.940 closed 0.049 progress 0.007; r: surv 0.980 closed 0.041 progress 0.005) [decisive]
- suite move_turn: n 200 surv 0.000 closed 0.053 progress 0.024 ratio 0.525 (l: surv 0.000 closed 0.054 progress 0.026 ratio 0.548; r: surv 0.000 closed 0.049 progress 0.022 ratio 0.508) [decisive]
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.15:0.1 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c4 stand: survival 0.820 -> 1.000 held
- RETENTION c4 stand: drift_speed 0.087 -> 0.083 held
- RETENTION c4 fwd: survival 0.050 -> 0.000 held
- RETENTION c4 fwd: ratio 0.554 -> 0.646 held
- RETENTION c4 side: survival 0.725 -> 0.890 held
- RETENTION c4 side: ratio 0.555 -> 0.439 DROPPED 21% (>20%)
- RETENTION c4 back: survival 0.130 -> 0.000 DROPPED 0.130 (>0.1)
- RETENTION c4 back: ratio 0.581 -> 0.598 held
- RETENTION c4 turn: survival 0.815 -> 0.960 held
- RETENTION c4 turn: facing_closed 0.254 -> 0.043 DROPPED 0.211 absolute (>0.100 = max(0.1, R_eval))
- floor FAIL: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, prev >= 0.3); fwd 0.000 (bound 0.000, n_env 100, prev >= 0.3); side 0.890 (bound 0.963, n_env 100, prev >= 0.3); back 0.000 (bound 0.000, n_env 100, prev >= 0.3); turn 0.960 (bound 1.006, n_env 100, prev >= 0.3); move_turn 0.000 (bound 0.000, n_env 100, new >= 0.1) -- FAIL on ['fwd', 'back', 'move_turn']

>>> ENTRY X07:c4

### X07:c5 — ref:scratch_free_s2 s2 (reference) — 2026-09-25T07:28:58

- recipe `34c94d4fe704` on chain `scratch_free`, base `34c94d4fe704`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.1:0.15', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-25_06-46-07_seed2_X07_c5_A15pB/model_5997.pt` | resume `model_4498.pt` | 1500 its | wall 0.60 h
- smoke fail@2k 0.933 fail@end 0.933 coll@2k 0.000 ep_len@2k 676.413
- training translation floor: track_ratio_mean (last 500 its) 0.614 (gate floor 0.6)
- exposure (ring): reach_edge 0.067 | reach_obst 0.057 | field_frac 0.011 (steps 17.712) | goals_passed 0.078
- terrain-promotion telemetry: track_ratio_mean 0.623 (move rows only 0.622) | heading_err_rms_mean 0.125 | block_steps_mean 7.129 | r_max_mean 1.034
- failure share flat 0.935 | obst 0.910 | campaign-wide 0.923
- slot fractions (logged): stand 0.073 | move 0.202 | turn 0.044 | move_turn 0.049 | rsi_frac_actual 0.216 | terrain_levels 1.027 | mean ep len 712.585 | prints 1500
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
- suite stand: n 100 surv 0.980 drift 0.080 [decisive]
- suite fwd: n 100 surv 0.000 ratio 0.575 tripod 0.272 [decisive]
- suite side: n 200 surv 0.850 ratio 0.449 tripod 0.499 (l: surv 0.860 ratio 0.450 tripod 0.497; r: surv 0.840 ratio 0.443 tripod 0.505) [decisive]
- suite back: n 100 surv 0.000 ratio 0.599 tripod 0.338 [decisive]
- suite diag: n 100 surv 0.000 ratio 0.669 tripod 0.317 [decisive]
- suite turn: n 200 surv 0.915 closed 0.067 progress 0.009 (l: surv 0.930 closed 0.078 progress 0.009; r: surv 0.900 closed 0.064 progress 0.008) [decisive]
- suite move_turn: n 200 surv 0.000 closed 0.053 progress 0.025 ratio 0.561 (l: surv 0.000 closed 0.047 progress 0.022 ratio 0.567; r: surv 0.000 closed 0.058 progress 0.026 ratio 0.552) [decisive]
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.1:0.15 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c5 stand: survival 0.790 -> 0.980 held
- RETENTION c5 stand: drift_speed 0.082 -> 0.080 held
- RETENTION c5 fwd: survival 0.120 -> 0.000 DROPPED 0.120 (>0.1)
- RETENTION c5 fwd: ratio 0.478 -> 0.575 held
- RETENTION c5 side: survival 0.775 -> 0.850 held
- RETENTION c5 side: ratio 0.562 -> 0.449 DROPPED 20% (>20%)
- RETENTION c5 back: survival 0.360 -> 0.000 DROPPED 0.360 (>0.1)
- RETENTION c5 back: ratio 0.485 -> 0.599 held
- RETENTION c5 turn: survival 0.765 -> 0.915 held
- RETENTION c5 turn: facing_closed 0.357 -> 0.067 DROPPED 0.290 absolute (>0.100 = max(0.1, R_eval))
- RETENTION c5 move_turn: survival 0.120 -> 0.000 DROPPED 0.120 (>0.1)
- RETENTION c5 move_turn: facing_closed 0.061 -> 0.053 held
- RETENTION c5 move_turn: ratio 0.532 -> 0.561 held
- floor PASS: no floor at this stage

>>> ENTRY X07:c5

### X07:c6 — ref:scratch_free_s2 s2 (reference) — 2026-09-25T08:23:59

- recipe `34c94d4fe704` on chain `scratch_free`, base `34c94d4fe704`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1'}`
- status **ok** | checkpoint `2026-09-25_07-29-08_seed2_X07_c6_A15pB/model_7996.pt` | resume `model_5997.pt` | 2000 its | wall 0.80 h
- smoke fail@2k 0.942 fail@end 0.942 coll@2k 0.000 ep_len@2k 708.101
- training translation floor: track_ratio_mean (last 500 its) 0.616 (gate floor 0.6)
- exposure (ring): reach_edge 0.044 | reach_obst 0.036 | field_frac 0.008 (steps 13.835) | goals_passed 0.052
- terrain-promotion telemetry: track_ratio_mean 0.608 (move rows only 0.601) | heading_err_rms_mean 0.145 | block_steps_mean 7.087 | r_max_mean 1.019
- failure share flat 0.927 | obst 0.924 | campaign-wide 0.928
- slot fractions (logged): stand 0.085 | move 0.224 | turn 0.049 | move_turn 0.050 | rsi_frac_actual 0.201 | terrain_levels 1.027 | mean ep len 767.748 | prints 2000
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
- suite stand: n 100 surv 0.990 drift 0.083 [decisive]
- suite fwd: n 100 surv 0.000 ratio 0.503 tripod 0.282 [decisive]
- suite side: n 200 surv 0.875 ratio 0.419 tripod 0.450 (l: surv 0.900 ratio 0.408 tripod 0.456; r: surv 0.850 ratio 0.432 tripod 0.434) [decisive]
- suite back: n 100 surv 0.010 ratio 0.503 tripod 0.266 [decisive]
- suite diag: n 100 surv 0.000 ratio 0.609 tripod 0.238 [decisive]
- suite turn: n 200 surv 0.980 closed 0.076 progress 0.010 (l: surv 0.970 closed 0.069 progress 0.009; r: surv 0.990 closed 0.079 progress 0.010) [decisive]
- suite move_turn: n 200 surv 0.000 closed 0.059 progress 0.022 ratio 0.467 (l: surv 0.000 closed 0.043 progress 0.015 ratio 0.497; r: surv 0.000 closed 0.074 progress 0.026 ratio 0.442) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.238 (need >= 0.4; by hold: back 0.266, diag 0.238, fwd 0.282, lat_l 0.456, lat_r 0.434, move_turn 0.291, move_turn_r 0.414, settle 0.000, stand 0.694, turn_l90 0.677, turn_r90 0.654), completion 0.471 (need >= 0.7), yaw-progress rate turn_l90 0.009, turn_r90 0.010 rad/s (net turn_l90 0.009, turn_r90 0.010) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.408, diag 0.609, back 0.503, move_turn 0.497 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.616 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- RETENTION c6 stand: survival 0.810 -> 0.990 held
- RETENTION c6 stand: drift_speed 0.083 -> 0.083 held
- RETENTION c6 fwd: survival 0.460 -> 0.000 DROPPED 0.460 (>0.1)
- RETENTION c6 fwd: ratio 0.466 -> 0.503 held
- RETENTION c6 side: survival 0.865 -> 0.875 held
- RETENTION c6 side: ratio 0.514 -> 0.419 held
- RETENTION c6 back: survival 0.560 -> 0.010 DROPPED 0.550 (>0.1)
- RETENTION c6 back: ratio 0.478 -> 0.503 held
- RETENTION c6 diag: survival 0.300 -> 0.000 DROPPED 0.300 (>0.1)
- RETENTION c6 diag: ratio 0.548 -> 0.609 held
- RETENTION c6 turn: survival 0.805 -> 0.980 held
- RETENTION c6 turn: facing_closed 0.508 -> 0.076 DROPPED 0.431 absolute (>0.100 = max(0.1, R_eval))
- RETENTION c6 move_turn: survival 0.550 -> 0.000 DROPPED 0.550 (>0.1)
- RETENTION c6 move_turn: facing_closed 0.185 -> 0.059 DROPPED 0.126 absolute (>0.100 = max(0.1, R_eval))
- RETENTION c6 move_turn: ratio 0.483 -> 0.467 held
- floor PASS: no floor at this stage

>>> ENTRY X07:c6


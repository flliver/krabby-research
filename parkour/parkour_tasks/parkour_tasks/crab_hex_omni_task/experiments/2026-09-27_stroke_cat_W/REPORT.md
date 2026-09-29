# Candidate ladder `2026-09-27_stroke_cat_W`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 3, confirm seed 2, 4000 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `fb17d2ea389eb13c5719fccbf5cf2de4abc4bb2ead470b7fb4ab6b9683866ec0` · `candidates.yaml` sha256 `65c7369cab0a98fd41046b548a6cb315a521fe2127b1d747783bf49ccd2bd529`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | B | baseline | c6y | the incumbent recipe at equal compute | — |

### X01:c6y — ref:S_cat s3 (reference) — 2026-09-28T10:00:55

- recipe `cfae5b009b17` on chain `c6y_scat`, base `cfae5b009b17`, code_sha `fb17d2ea389e`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1', 'KRABBY_OMNI_STROKE_W': '0.126240', 'KRABBY_OMNI_STROKE_CAT_W': 'fwd:1.000000,back:1.000000,side:2.928327,diag:1.314741,turn:1.214263,move_turn:1.000000'}`
- status **ok** | checkpoint `2026-09-28_08-11-28_seed3_X01_c6y_A15pB/model_15994.pt` | resume `model_11995.pt` | 4000 its | wall 1.70 h
- smoke fail@2k 0.440 fail@end 0.432 coll@2k 0.000 ep_len@2k 1567.776
- training translation floor: track_ratio_mean (last 500 its) 0.684 (gate floor 0.6)
- exposure (ring): reach_edge 0.593 | reach_obst 0.496 | field_frac 0.212 (steps 388.554) | goals_passed 0.810
- terrain-promotion telemetry: track_ratio_mean 0.686 (move rows only 0.690) | heading_err_rms_mean 0.151 | block_steps_mean 110.652 | r_max_mean 2.340
- failure share flat 0.439 | obst 0.402 | campaign-wide 0.423
- slot fractions (logged): stand 0.143 | move 0.425 | turn 0.095 | move_turn 0.153 | rsi_frac_actual 0.200 | terrain_levels 0.973 | mean ep len 1580.162 | prints 4000
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
- suite fwd: n 100 surv 0.880 ratio 0.711 tripod 0.455 [decisive]
- suite side: n 200 surv 0.995 ratio 0.490 tripod 0.463 (l: surv 1.000 ratio 0.493 tripod 0.463; r: surv 0.990 ratio 0.484 tripod 0.463) [decisive]
- suite back: n 100 surv 0.940 ratio 0.752 tripod 0.534 [decisive]
- suite diag: n 100 surv 0.700 ratio 0.611 tripod 0.345 [decisive]
- suite turn: n 200 surv 0.990 closed 0.727 progress 0.096 (l: surv 1.000 closed 0.726 progress 0.096; r: surv 0.980 closed 0.729 progress 0.096) [decisive]
- suite move_turn: n 200 surv 0.735 closed 0.294 progress 0.043 ratio 0.478 (l: surv 0.700 closed 0.311 progress 0.049 ratio 0.484; r: surv 0.770 closed 0.271 progress 0.038 ratio 0.475) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.345 (need >= 0.4; by hold: back 0.534, diag 0.345, fwd 0.455, lat_l 0.463, lat_r 0.463, move_turn 0.425, move_turn_r 0.405, settle 0.357, stand 0.684, turn_l90 0.497, turn_r90 0.484), completion 0.896 (need >= 0.7), yaw-progress rate turn_l90 0.096, turn_r90 0.096 rad/s (net turn_l90 0.096, turn_r90 0.096) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.493, diag 0.611, back 0.752, move_turn 0.484 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.684 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
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
- overlay PASS: overlay KRABBY_OMNI_STROKE_CAT_W=fwd:1.000000,back:1.000000,side:2.928327,diag:1.314741,turn:1.214263,move_turn:1.000000 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.126240 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, min >= 0.8); fwd 0.880 (bound 0.956, n_env 100, min >= 0.75); side 0.995 (bound 1.011, n_env 100, min >= 0.795); back 0.940 (bound 0.995, n_env 100, min >= 0.78); diag 0.700 (bound 0.807, n_env 100, min >= 0.67); turn 0.990 (bound 1.013, n_env 100, min >= 0.79); move_turn 0.735 (bound 0.838, n_env 100, min >= 0.165)

>>> ENTRY X01:c6y

### X02:c6y — ref:S_cat s2 (reference) — 2026-09-28T16:25:26

- recipe `cfae5b009b17` on chain `c6y_scat`, base `cfae5b009b17`, code_sha `fb17d2ea389e`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1', 'KRABBY_OMNI_STROKE_W': '0.126240', 'KRABBY_OMNI_STROKE_CAT_W': 'fwd:1.000000,back:1.000000,side:2.928327,diag:1.314741,turn:1.214263,move_turn:1.000000'}`
- status **ok** | checkpoint `2026-09-28_14-37-59_seed2_X02_c6y_A15pB/model_15994.pt` | resume `model_11995.pt` | 4000 its | wall 1.67 h
- smoke fail@2k 0.401 fail@end 0.309 coll@2k 0.000 ep_len@2k 1620.881
- training translation floor: track_ratio_mean (last 500 its) 0.678 (gate floor 0.6)
- exposure (ring): reach_edge 0.645 | reach_obst 0.578 | field_frac 0.252 (steps 467.759) | goals_passed 0.884
- terrain-promotion telemetry: track_ratio_mean 0.675 (move rows only 0.682) | heading_err_rms_mean 0.147 | block_steps_mean 143.202 | r_max_mean 2.573
- failure share flat 0.296 | obst 0.359 | campaign-wide 0.308
- slot fractions (logged): stand 0.106 | move 0.553 | turn 0.081 | move_turn 0.148 | rsi_frac_actual 0.229 | terrain_levels 1.027 | mean ep len 1705.942 | prints 4000
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
- suite fwd: n 100 surv 0.960 ratio 0.773 tripod 0.608 [decisive]
- suite side: n 200 surv 1.000 ratio 0.545 tripod 0.455 (l: surv 1.000 ratio 0.547 tripod 0.448; r: surv 1.000 ratio 0.541 tripod 0.461) [decisive]
- suite back: n 100 surv 0.980 ratio 0.767 tripod 0.634 [decisive]
- suite diag: n 100 surv 0.840 ratio 0.651 tripod 0.421 [decisive]
- suite turn: n 200 surv 0.985 closed 0.614 progress 0.081 (l: surv 0.980 closed 0.617 progress 0.081; r: surv 0.990 closed 0.614 progress 0.080) [decisive]
- suite move_turn: n 200 surv 0.660 closed 0.519 progress 0.072 ratio 0.526 (l: surv 0.620 closed 0.471 progress 0.068 ratio 0.533; r: surv 0.700 closed 0.561 progress 0.077 ratio 0.517) [decisive]
- WATCH gate_1c: PASS -- formation gate passed — walking-hold tripod 0.421 (by hold: back 0.634, diag 0.421, fwd 0.608, lat_l 0.448, lat_r 0.461, move_turn 0.482, move_turn_r 0.484, settle 0.699, stand 0.709, turn_l90 0.489, turn_r90 0.490), completion 0.907, yaw-progress rate turn_l90 0.081, turn_r90 0.080 rad/s (net turn_l90 0.081, turn_r90 0.080), per-direction speed_ratio lat_l 0.547, diag 0.651, back 0.767, move_turn 0.533 on slow__omni, training track_ratio_mean 0.678 (need >= 0.6 over the last 500 its)
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
- overlay PASS: overlay KRABBY_OMNI_STROKE_CAT_W=fwd:1.000000,back:1.000000,side:2.928327,diag:1.314741,turn:1.214263,move_turn:1.000000 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.126240 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, min >= 0.8); fwd 0.960 (bound 1.006, n_env 100, min >= 0.75); side 1.000 (bound 1.000, n_env 100, min >= 0.795); back 0.980 (bound 1.013, n_env 100, min >= 0.78); diag 0.840 (bound 0.925, n_env 100, min >= 0.67); turn 0.985 (bound 1.013, n_env 100, min >= 0.79); move_turn 0.660 (bound 0.770, n_env 100, min >= 0.165)

>>> ENTRY X02:c6y


# Candidate ladder `2026-09-26_stroke_formation`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 3, confirm seed 2, 3000 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `709161927627f6c00012abf143b3146ff6d51f81acdc709dbb0f04c014f24a30` · `candidates.yaml` sha256 `61b96e80f5b749adc76ffacba4093140df8d97f4cb3bebe6b797ea004d416b37`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | B_f | baseline | t3f | the incumbent recipe at equal compute | — |
| 1 | S_f | None | effective | `{"t3":{"KRABBY_OMNI_STROKE_W":"0.119"}}` | `—` |
| 2 | P_f | None | effective | `{"t3":{"KRABBY_OMNI_CENTRING_W":"0.158"}}` | `—` |

### X01:t3 — B_f s3 (baseline) — 2026-09-26T19:04:11

- recipe `d695a100e1a7` on chain `t3f`, base `d695a100e1a7`, code_sha `709161927627`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_STROKE_W': '0.001', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1'}`
- status **ok** | checkpoint `2026-09-26_17-42-44_seed3_X01_t3_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.23 h
- smoke fail@2k 0.427 fail@end 0.431 coll@2k 0.000 ep_len@2k 1564.656
- training translation floor: track_ratio_mean (last 500 its) 0.570 (gate floor 0.6)
- exposure (ring): reach_edge 0.332 | reach_obst 0.228 | field_frac 0.096 (steps 185.574) | goals_passed 0.433
- terrain-promotion telemetry: track_ratio_mean 0.567 (move rows only 0.567) | heading_err_rms_mean 0.149 | block_steps_mean 40.813 | r_max_mean 1.848
- failure share flat 0.436 | obst 0.465 | campaign-wide 0.455
- slot fractions (logged): stand 0.145 | move 0.488 | turn 0.170 | move_turn 0.000 | rsi_frac_actual 0.202 | terrain_levels 0.973 | mean ep len 1429.361 | prints 3000
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
- suite stand: n 100 surv 0.930 drift 0.111 [decisive]
- suite fwd: n 100 surv 0.490 ratio 0.474 tripod 0.374 [decisive]
- suite side: n 200 surv 0.760 ratio 0.458 tripod 0.465 (l: surv 0.780 ratio 0.456 tripod 0.481; r: surv 0.740 ratio 0.465 tripod 0.447) [decisive]
- suite back: n 100 surv 0.510 ratio 0.533 tripod 0.417 [decisive]
- suite diag: n 100 surv 0.570 ratio 0.495 tripod 0.445 [WATCH: untrained]
- suite turn: n 200 surv 0.925 closed 0.467 progress 0.062 (l: surv 0.950 closed 0.472 progress 0.063; r: surv 0.900 closed 0.442 progress 0.062) [decisive]
- suite move_turn: n 200 surv 0.280 closed 0.035 progress 0.011 ratio 0.410 (l: surv 0.330 closed 0.027 progress 0.008 ratio 0.377; r: surv 0.230 closed 0.056 progress 0.011 ratio 0.454) [WATCH: untrained]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.345 (need >= 0.4; by hold: back 0.417, diag 0.445, fwd 0.374, lat_l 0.481, lat_r 0.447, move_turn 0.354, move_turn_r 0.345, settle 0.135, stand 0.549, turn_l90 0.548, turn_r90 0.554), completion 0.643 (need >= 0.7), yaw-progress rate turn_l90 0.063, turn_r90 0.062 rad/s (net turn_l90 0.063, turn_r90 0.062) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.456, diag 0.495, back 0.533, move_turn 0.377 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.570 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.930 (bound 0.989, n_env 100, new >= 0.1); fwd 0.490 (bound 0.606, n_env 100, new >= 0.1); side 0.760 (bound 0.859, n_env 100, new >= 0.1); back 0.510 (bound 0.626, n_env 100, new >= 0.1); turn 0.925 (bound 0.986, n_env 100, new >= 0.1)

>>> ENTRY X01:t3

### experiment X01 — B_f s3 — ok — 2026-09-26T19:04:11

- recipe `d695a100e1a7` base `d695a100e1a7` chain `t3f` seed 3
- stages ['X01:t3']
- screen: completion 0.643, tripod_dir 0.374, track tail 0.570
- replay: none

>>> ENTRY X01

### X02:t3 — B_f s2 (baseline) — 2026-09-27T00:15:29

- recipe `d695a100e1a7` on chain `t3f`, base `d695a100e1a7`, code_sha `709161927627`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_STROKE_W': '0.001', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1'}`
- status **ok** | checkpoint `2026-09-26_22-54-02_seed2_X02_t3_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.23 h
- smoke fail@2k 0.266 fail@end 0.166 coll@2k 0.000 ep_len@2k 1775.746
- training translation floor: track_ratio_mean (last 500 its) 0.607 (gate floor 0.6)
- exposure (ring): reach_edge 0.450 | reach_obst 0.471 | field_frac 0.133 (steps 260.545) | goals_passed 0.589
- terrain-promotion telemetry: track_ratio_mean 0.594 (move rows only 0.594) | heading_err_rms_mean 0.154 | block_steps_mean 94.832 | r_max_mean 2.145
- failure share flat 0.186 | obst 0.223 | campaign-wide 0.169
- slot fractions (logged): stand 0.137 | move 0.584 | turn 0.189 | move_turn 0.000 | rsi_frac_actual 0.188 | terrain_levels 1.027 | mean ep len 1815.779 | prints 3000
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
- suite stand: n 100 surv 0.960 drift 0.117 [decisive]
- suite fwd: n 100 surv 0.910 ratio 0.542 tripod 0.614 [decisive]
- suite side: n 200 surv 0.970 ratio 0.477 tripod 0.435 (l: surv 0.950 ratio 0.478 tripod 0.431; r: surv 0.990 ratio 0.472 tripod 0.437) [decisive]
- suite back: n 100 surv 0.970 ratio 0.544 tripod 0.611 [decisive]
- suite diag: n 100 surv 0.830 ratio 0.514 tripod 0.496 [WATCH: untrained]
- suite turn: n 200 surv 0.980 closed 0.412 progress 0.053 (l: surv 0.980 closed 0.409 progress 0.054; r: surv 0.980 closed 0.413 progress 0.052) [decisive]
- suite move_turn: n 200 surv 0.770 closed 0.093 progress 0.013 ratio 0.336 (l: surv 0.800 closed 0.136 progress 0.022 ratio 0.361; r: surv 0.740 closed 0.043 progress 0.005 ratio 0.291) [WATCH: untrained]
- WATCH gate_1c: PASS -- formation gate passed — walking-hold tripod 0.431 (by hold: back 0.611, diag 0.496, fwd 0.614, lat_l 0.431, lat_r 0.437, move_turn 0.551, move_turn_r 0.529, settle 0.398, stand 0.573, turn_l90 0.698, turn_r90 0.663), completion 0.910, yaw-progress rate turn_l90 0.054, turn_r90 0.052 rad/s (net turn_l90 0.054, turn_r90 0.052), per-direction speed_ratio lat_l 0.478, diag 0.514, back 0.544, move_turn 0.361 on slow__omni, training track_ratio_mean 0.607 (need >= 0.6 over the last 500 its)
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.960 (bound 1.006, n_env 100, new >= 0.1); fwd 0.910 (bound 0.977, n_env 100, new >= 0.1); side 0.970 (bound 1.010, n_env 100, new >= 0.1); back 0.970 (bound 1.010, n_env 100, new >= 0.1); turn 0.980 (bound 1.013, n_env 100, new >= 0.1)

>>> ENTRY X02:t3

### experiment X02 — B_f s2 — ok — 2026-09-27T00:15:29

- recipe `d695a100e1a7` base `d695a100e1a7` chain `t3f` seed 2
- stages ['X02:t3']
- screen: completion 0.910, tripod_dir 0.431, track tail 0.607
- replay: none

>>> ENTRY X02

### X03:t3 — S_f s3 (candidate) — 2026-09-27T01:35:10

- recipe `8cff5f3c42e8` on chain `t3f`, base `d695a100e1a7`, code_sha `709161927627`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_STROKE_W': '0.119', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1'}`
- status **ok** | checkpoint `2026-09-27_00-15-39_seed3_X03_t3_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.20 h
- smoke fail@2k 0.085 fail@end 0.112 coll@2k 0.000 ep_len@2k 1922.508
- training translation floor: track_ratio_mean (last 500 its) 0.599 (gate floor 0.6)
- exposure (ring): reach_edge 0.522 | reach_obst 0.414 | field_frac 0.162 (steps 316.267) | goals_passed 0.666
- terrain-promotion telemetry: track_ratio_mean 0.593 (move rows only 0.593) | heading_err_rms_mean 0.128 | block_steps_mean 87.904 | r_max_mean 2.298
- failure share flat 0.112 | obst 0.136 | campaign-wide 0.103
- slot fractions (logged): stand 0.160 | move 0.663 | turn 0.159 | move_turn 0.000 | rsi_frac_actual 0.202 | terrain_levels 0.973 | mean ep len 1914.863 | prints 3000
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
- suite stand: n 100 surv 1.000 drift 0.110 [decisive]
- suite fwd: n 100 surv 0.960 ratio 0.612 tripod 0.627 [decisive]
- suite side: n 200 surv 0.975 ratio 0.452 tripod 0.466 (l: surv 0.970 ratio 0.457 tripod 0.466; r: surv 0.980 ratio 0.448 tripod 0.466) [decisive]
- suite back: n 100 surv 0.940 ratio 0.628 tripod 0.631 [decisive]
- suite diag: n 100 surv 0.970 ratio 0.541 tripod 0.531 [WATCH: untrained]
- suite turn: n 200 surv 0.990 closed 0.460 progress 0.059 (l: surv 0.990 closed 0.472 progress 0.062; r: surv 0.990 closed 0.447 progress 0.057) [decisive]
- suite move_turn: n 200 surv 0.900 closed 0.061 progress 0.008 ratio 0.364 (l: surv 0.850 closed 0.050 progress 0.007 ratio 0.356; r: surv 0.950 closed 0.076 progress 0.010 ratio 0.366) [WATCH: untrained]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.466 (need >= 0.4; by hold: back 0.631, diag 0.531, fwd 0.627, lat_l 0.466, lat_r 0.466, move_turn 0.558, move_turn_r 0.590, settle 0.378, stand 0.605, turn_l90 0.708, turn_r90 0.723), completion 0.960 (need >= 0.7), yaw-progress rate turn_l90 0.062, turn_r90 0.057 rad/s (net turn_l90 0.062, turn_r90 0.057) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.457, diag 0.541, back 0.628, move_turn 0.356 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.599 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.119 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, new >= 0.1); fwd 0.960 (bound 1.006, n_env 100, new >= 0.1); side 0.975 (bound 1.011, n_env 100, new >= 0.1); back 0.940 (bound 0.995, n_env 100, new >= 0.1); turn 0.990 (bound 1.013, n_env 100, new >= 0.1)

>>> ENTRY X03:t3

## D05 — S_f — REJECT (clear, speculative) — 2026-09-27T01:35:10

- question: Armed from iteration 0 of formation (C1d's t3, from scratch, clock held), does the per-foot loaded stance-stroke income at w_S land (fwd delivered >= 0.5 x the replay prediction) and move the stroke -- side stance_stroke, side stroke_share or fwd stance_stroke beyond its band, pooled over seeds 3 and 2 against B_f -- and does it lengthen side strides (stride_along, stance_stroke) or shrink the lateral body lead as it forms?
- experiment `X03` recipe `8cff5f3c42e8` base `d695a100e1a7` control `X01`
- why: WORSE on guard cat.turn.foot_slip; pair_on always: a seed-2 pair is queued -- the entry stays speculative until it is pooled, and the pooled decision is final
- block F (F): z -0.327 vs R 1.256 -> TIE (k=2, missing [])
  - cat.turn.facing_closed: 0.467 -> 0.460 (delta -0.007, R_eval 0.050, R_seed 0.055, z -0.134)
  - cat.turn.yaw_progress: 0.062 -> 0.059 (delta -0.003, R_eval 0.007, R_seed 0.009, z -0.520)
- block T (T): z 1.619 vs R 0.985 -> BETTER (k=3, missing [])
  - cat.fwd.ratio: 0.474 -> 0.612 (delta 0.138, R_eval 0.041, R_seed 0.068, z 3.371)
  - cat.side.ratio: 0.458 -> 0.452 (delta -0.006, R_eval 0.018, R_seed 0.020, z -0.309)
  - cat.back.ratio: 0.533 -> 0.628 (delta 0.094, R_eval 0.053, R_seed 0.011, z 1.795)
- block LEN (LEN): z 1.382 vs R 1.551 -> TIE (k=1, missing [])
  - cat.side.steps_per_m: 32.390 -> 30.487 (lower is better, signed delta 1.903, R_eval 1.377, R_seed 2.136, z 1.382)
- guard cat.stand.survival: 0.930 -> 1.000 (R 0.072) -> TIE
- guard cat.stand.steps_per_s: 3.083 -> 3.167 (R 0.312) -> TIE
- guard cat.fwd.survival: 0.490 -> 0.960 (R 0.420) -> TIE
- guard cat.fwd.ratio: 0.474 -> 0.612 (R 0.068) -> TIE
- guard cat.fwd.foot_slip: 0.124 -> 0.144 (R 0.038) -> TIE
- guard cat.fwd.steps_per_m: 27.236 -> 19.212 (R 4.498) -> TIE
- guard cat.side.survival: 0.760 -> 0.975 (R 0.210) -> TIE
- guard cat.side.ratio: 0.458 -> 0.452 (R 0.021) -> TIE
- guard cat.side.foot_slip: 0.179 -> 0.197 (R 0.039) -> TIE
- guard cat.side.steps_per_m: 32.390 -> 30.487 (R 2.136) -> TIE
- guard cat.back.survival: 0.510 -> 0.940 (R 0.460) -> TIE
- guard cat.back.ratio: 0.533 -> 0.628 (R 0.062) -> TIE
- guard cat.back.foot_slip: 0.127 -> 0.145 (R 0.029) -> TIE
- guard cat.back.steps_per_m: 29.679 -> 20.362 (R 3.943) -> TIE
- guard cat.turn.survival: 0.925 -> 0.990 (R 0.058) -> TIE
- guard cat.turn.foot_slip: 0.106 -> 0.129 (R 0.019) -> WORSE
- guard cat.turn.steps_per_rad: 62.731 -> 53.393 (R 4.402) -> TIE
- anchored drift: FAIL — drifted below the anchor on guard cat.turn.foot_slip

>>> ENTRY D05

### experiment X03 — S_f s3 — ok — 2026-09-27T01:35:10

- recipe `8cff5f3c42e8` base `d695a100e1a7` chain `t3f` seed 3
- stages ['X03:t3']
- screen: completion 0.960, tripod_dir 0.466, track tail 0.599
- replay: none

>>> ENTRY X03

### X04:t3 — S_f s2 (seed2) — 2026-09-27T02:56:48

- recipe `8cff5f3c42e8` on chain `t3f`, base `d695a100e1a7`, code_sha `709161927627`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_STROKE_W': '0.119', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1'}`
- status **ok** | checkpoint `2026-09-27_01-35-20_seed2_X04_t3_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.23 h
- smoke fail@2k 0.275 fail@end 0.195 coll@2k 0.000 ep_len@2k 1749.260
- training translation floor: track_ratio_mean (last 500 its) 0.581 (gate floor 0.6)
- exposure (ring): reach_edge 0.466 | reach_obst 0.459 | field_frac 0.152 (steps 292.213) | goals_passed 0.592
- terrain-promotion telemetry: track_ratio_mean 0.583 (move rows only 0.583) | heading_err_rms_mean 0.153 | block_steps_mean 100.448 | r_max_mean 2.253
- failure share flat 0.170 | obst 0.232 | campaign-wide 0.178
- slot fractions (logged): stand 0.151 | move 0.632 | turn 0.154 | move_turn 0.000 | rsi_frac_actual 0.200 | terrain_levels 1.027 | mean ep len 1798.335 | prints 3000
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
- suite stand: n 100 surv 1.000 drift 0.118 [decisive]
- suite fwd: n 100 surv 0.950 ratio 0.605 tripod 0.572 [decisive]
- suite side: n 200 surv 0.985 ratio 0.457 tripod 0.436 (l: surv 0.990 ratio 0.467 tripod 0.434; r: surv 0.980 ratio 0.446 tripod 0.437) [decisive]
- suite back: n 100 surv 0.910 ratio 0.553 tripod 0.516 [decisive]
- suite diag: n 100 surv 0.880 ratio 0.521 tripod 0.483 [WATCH: untrained]
- suite turn: n 200 surv 1.000 closed 0.362 progress 0.046 (l: surv 1.000 closed 0.367 progress 0.049; r: surv 1.000 closed 0.356 progress 0.046) [decisive]
- suite move_turn: n 200 surv 0.730 closed 0.048 progress 0.007 ratio 0.317 (l: surv 0.720 closed 0.070 progress 0.010 ratio 0.286; r: surv 0.740 closed 0.031 progress 0.005 ratio 0.369) [WATCH: untrained]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.434 (need >= 0.4; by hold: back 0.516, diag 0.483, fwd 0.572, lat_l 0.434, lat_r 0.437, move_turn 0.460, move_turn_r 0.476, settle 0.292, stand 0.529, turn_l90 0.649, turn_r90 0.627), completion 0.917 (need >= 0.7), yaw-progress rate turn_l90 0.049, turn_r90 0.046 rad/s (net turn_l90 0.049, turn_r90 0.046) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.467, diag 0.521, back 0.553, move_turn 0.286 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: move_turn) on slow__omni, training track_ratio_mean 0.581 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.119 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, new >= 0.1); fwd 0.950 (bound 1.001, n_env 100, new >= 0.1); side 0.985 (bound 1.013, n_env 100, new >= 0.1); back 0.910 (bound 0.977, n_env 100, new >= 0.1); turn 1.000 (bound 1.000, n_env 100, new >= 0.1)

>>> ENTRY X04:t3

## D05 — S_f — REJECT (disputed, final) — 2026-09-27T02:56:48

- question: Armed from iteration 0 of formation (C1d's t3, from scratch, clock held), does the per-foot loaded stance-stroke income at w_S land (fwd delivered >= 0.5 x the replay prediction) and move the stroke -- side stance_stroke, side stroke_share or fwd stance_stroke beyond its band, pooled over seeds 3 and 2 against B_f -- and does it lengthen side strides (stride_along, stance_stroke) or shrink the lateral body lead as it forms?
- experiment `X03` recipe `8cff5f3c42e8` base `d695a100e1a7` control `X01`
- why: the two seeds disagree in sign(Σ z_b/R_b) (s3 +2.274, s2 -1.363) -> the incumbent stands
- block F (F): z -0.679 vs R 1.256 -> TIE (k=2, missing [])
  - cat.turn.facing_closed: 0.467 -> 0.460 (delta -0.028, R_eval 0.050, R_seed 0.055, z -0.570)
  - cat.turn.yaw_progress: 0.062 -> 0.059 (delta -0.005, R_eval 0.007, R_seed 0.009, z -0.788)
- block T (T): z 0.899 vs R 0.985 -> TIE (k=3, missing [])
  - cat.fwd.ratio: 0.474 -> 0.612 (delta 0.100, R_eval 0.041, R_seed 0.068, z 2.446)
  - cat.side.ratio: 0.458 -> 0.452 (delta -0.013, R_eval 0.018, R_seed 0.020, z -0.727)
  - cat.back.ratio: 0.533 -> 0.628 (delta 0.051, R_eval 0.053, R_seed 0.011, z 0.979)
- block LEN (LEN): z 0.130 vs R 1.551 -> TIE (k=1, missing [])
  - cat.side.steps_per_m: 32.390 -> 30.487 (lower is better, signed delta 0.179, R_eval 1.377, R_seed 2.136, z 0.130)
- guard cat.stand.survival: 0.930 -> 1.000 (R 0.051) -> TIE
- guard cat.stand.steps_per_s: 3.083 -> 3.167 (R 0.334) -> TIE
- guard cat.fwd.survival: 0.490 -> 0.960 (R 0.427) -> TIE
- guard cat.fwd.ratio: 0.474 -> 0.612 (R 0.077) -> TIE
- guard cat.fwd.foot_slip: 0.124 -> 0.144 (R 0.039) -> TIE
- guard cat.fwd.steps_per_m: 27.236 -> 19.212 (R 4.701) -> TIE
- guard cat.side.survival: 0.760 -> 0.975 (R 0.212) -> TIE
- guard cat.side.ratio: 0.458 -> 0.452 (R 0.025) -> TIE
- guard cat.side.foot_slip: 0.179 -> 0.197 (R 0.039) -> TIE
- guard cat.side.steps_per_m: 32.390 -> 30.487 (R 2.407) -> TIE
- guard cat.back.survival: 0.510 -> 0.940 (R 0.466) -> TIE
- guard cat.back.ratio: 0.533 -> 0.628 (R 0.045) -> TIE
- guard cat.back.foot_slip: 0.127 -> 0.145 (R 0.029) -> TIE
- guard cat.back.steps_per_m: 29.679 -> 20.362 (R 4.291) -> TIE
- guard cat.turn.survival: 0.925 -> 0.990 (R 0.068) -> TIE
- guard cat.turn.foot_slip: 0.106 -> 0.129 (R 0.019) -> TIE
- guard cat.turn.steps_per_rad: 62.731 -> 53.393 (R 3.738) -> TIE
- anchored drift: OK — no block or guard below the anchor

>>> ENTRY D05

### experiment X04 — S_f s2 — ok — 2026-09-27T02:56:48

- recipe `8cff5f3c42e8` base `d695a100e1a7` chain `t3f` seed 2
- stages ['X04:t3']
- screen: completion 0.917, tripod_dir 0.434, track tail 0.581
- replay: none

>>> ENTRY X04

### X05:t3 — P_f s3 (candidate) — 2026-09-27T04:16:29

- recipe `bc00f504d0ce` on chain `t3f`, base `d695a100e1a7`, code_sha `709161927627`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_STROKE_W': '0.001', 'KRABBY_OMNI_CENTRING_W': '0.158', 'KRABBY_OMNI_REWARD_SPLIT': '1'}`
- status **ok** | checkpoint `2026-09-27_02-56-58_seed3_X05_t3_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.20 h
- smoke fail@2k 0.341 fail@end 0.216 coll@2k 0.000 ep_len@2k 1666.259
- training translation floor: track_ratio_mean (last 500 its) 0.583 (gate floor 0.6)
- exposure (ring): reach_edge 0.462 | reach_obst 0.359 | field_frac 0.131 (steps 248.552) | goals_passed 0.580
- terrain-promotion telemetry: track_ratio_mean 0.587 (move rows only 0.587) | heading_err_rms_mean 0.159 | block_steps_mean 68.105 | r_max_mean 2.178
- failure share flat 0.257 | obst 0.317 | campaign-wide 0.174
- slot fractions (logged): stand 0.144 | move 0.588 | turn 0.197 | move_turn 0.000 | rsi_frac_actual 0.190 | terrain_levels 0.973 | mean ep len 1861.944 | prints 3000
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
- suite stand: n 100 surv 1.000 drift 0.104 [decisive]
- suite fwd: n 100 surv 0.950 ratio 0.625 tripod 0.570 [decisive]
- suite side: n 200 surv 0.975 ratio 0.451 tripod 0.501 (l: surv 0.970 ratio 0.444 tripod 0.503; r: surv 0.980 ratio 0.453 tripod 0.497) [decisive]
- suite back: n 100 surv 0.950 ratio 0.618 tripod 0.536 [decisive]
- suite diag: n 100 surv 0.860 ratio 0.569 tripod 0.504 [WATCH: untrained]
- suite turn: n 200 surv 1.000 closed 0.292 progress 0.039 (l: surv 1.000 closed 0.305 progress 0.040; r: surv 1.000 closed 0.288 progress 0.037) [decisive]
- suite move_turn: n 200 surv 0.920 closed 0.109 progress 0.015 ratio 0.532 (l: surv 0.950 closed 0.106 progress 0.014 ratio 0.537; r: surv 0.890 closed 0.119 progress 0.015 ratio 0.530) [WATCH: untrained]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.497 (need >= 0.4; by hold: back 0.536, diag 0.504, fwd 0.570, lat_l 0.503, lat_r 0.497, move_turn 0.591, move_turn_r 0.573, settle 0.400, stand 0.601, turn_l90 0.683, turn_r90 0.697), completion 0.955 (need >= 0.7), yaw-progress rate turn_l90 0.040, turn_r90 0.037 rad/s (net turn_l90 0.041, turn_r90 0.037) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.444, diag 0.569, back 0.618, move_turn 0.537 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.583 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.158 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, new >= 0.1); fwd 0.950 (bound 1.001, n_env 100, new >= 0.1); side 0.975 (bound 1.011, n_env 100, new >= 0.1); back 0.950 (bound 1.001, n_env 100, new >= 0.1); turn 1.000 (bound 1.000, n_env 100, new >= 0.1)

>>> ENTRY X05:t3

## D06 — P_f — REJECT (clear, speculative) — 2026-09-27T04:16:29

- question: Armed from iteration 0 of formation (C1d's t3, from scratch, clock held), does the side body-centring bonus at w_P keep the lateral posture from forming -- pooled side body_offset <= -0.05 m against B_f over seeds 3 and 2 -- and does the recentred posture lengthen side strides?
- experiment `X05` recipe `bc00f504d0ce` base `d695a100e1a7` control `X01`
- why: WORSE on block F, guard cat.turn.foot_slip, guard cat.turn.steps_per_rad; pair_on always: a seed-2 pair is queued -- the entry stays speculative until it is pooled, and the pooled decision is final
- block F (F): z -3.552 vs R 1.256 -> WORSE (k=2, missing [])
  - cat.turn.facing_closed: 0.467 -> 0.292 (delta -0.175, R_eval 0.050, R_seed 0.055, z -3.518)
  - cat.turn.yaw_progress: 0.062 -> 0.039 (delta -0.024, R_eval 0.007, R_seed 0.009, z -3.587)
- block T (T): z 1.631 vs R 0.985 -> BETTER (k=3, missing [])
  - cat.fwd.ratio: 0.474 -> 0.625 (delta 0.151, R_eval 0.041, R_seed 0.068, z 3.680)
  - cat.side.ratio: 0.458 -> 0.451 (delta -0.007, R_eval 0.018, R_seed 0.020, z -0.392)
  - cat.back.ratio: 0.533 -> 0.618 (delta 0.084, R_eval 0.053, R_seed 0.011, z 1.604)
- block LEN (LEN): z 0.806 vs R 1.551 -> TIE (k=1, missing [])
  - cat.side.steps_per_m: 32.390 -> 31.280 (lower is better, signed delta 1.110, R_eval 1.377, R_seed 2.136, z 0.806)
- guard cat.stand.survival: 0.930 -> 1.000 (R 0.072) -> TIE
- guard cat.stand.steps_per_s: 3.083 -> 3.250 (R 0.312) -> TIE
- guard cat.fwd.survival: 0.490 -> 0.950 (R 0.420) -> TIE
- guard cat.fwd.ratio: 0.474 -> 0.625 (R 0.068) -> TIE
- guard cat.fwd.foot_slip: 0.124 -> 0.150 (R 0.038) -> TIE
- guard cat.fwd.steps_per_m: 27.236 -> 19.765 (R 4.498) -> TIE
- guard cat.side.survival: 0.760 -> 0.975 (R 0.210) -> TIE
- guard cat.side.ratio: 0.458 -> 0.451 (R 0.021) -> TIE
- guard cat.side.foot_slip: 0.179 -> 0.201 (R 0.039) -> TIE
- guard cat.side.steps_per_m: 32.390 -> 31.280 (R 2.136) -> TIE
- guard cat.back.survival: 0.510 -> 0.950 (R 0.460) -> TIE
- guard cat.back.ratio: 0.533 -> 0.618 (R 0.062) -> TIE
- guard cat.back.foot_slip: 0.127 -> 0.152 (R 0.029) -> TIE
- guard cat.back.steps_per_m: 29.679 -> 22.313 (R 3.943) -> TIE
- guard cat.turn.survival: 0.925 -> 1.000 (R 0.058) -> TIE
- guard cat.turn.foot_slip: 0.106 -> 0.144 (R 0.019) -> WORSE
- guard cat.turn.steps_per_rad: 62.731 -> 82.691 (R 4.402) -> WORSE
- anchored drift: FAIL — drifted below the anchor on block F, guard cat.turn.foot_slip, guard cat.turn.steps_per_rad

>>> ENTRY D06

### experiment X05 — P_f s3 — ok — 2026-09-27T04:16:29

- recipe `bc00f504d0ce` base `d695a100e1a7` chain `t3f` seed 3
- stages ['X05:t3']
- screen: completion 0.955, tripod_dir 0.503, track tail 0.583
- replay: none

>>> ENTRY X05

### X06:t3 — P_f s2 (seed2) — 2026-09-27T05:38:07

- recipe `bc00f504d0ce` on chain `t3f`, base `d695a100e1a7`, code_sha `709161927627`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_STROKE_W': '0.001', 'KRABBY_OMNI_CENTRING_W': '0.158', 'KRABBY_OMNI_REWARD_SPLIT': '1'}`
- status **ok** | checkpoint `2026-09-27_04-16-39_seed2_X06_t3_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.23 h
- smoke fail@2k 0.284 fail@end 0.300 coll@2k 0.000 ep_len@2k 1718.183
- training translation floor: track_ratio_mean (last 500 its) 0.594 (gate floor 0.6)
- exposure (ring): reach_edge 0.470 | reach_obst 0.424 | field_frac 0.149 (steps 285.578) | goals_passed 0.588
- terrain-promotion telemetry: track_ratio_mean 0.596 (move rows only 0.596) | heading_err_rms_mean 0.132 | block_steps_mean 97.556 | r_max_mean 2.267
- failure share flat 0.260 | obst 0.276 | campaign-wide 0.319
- slot fractions (logged): stand 0.164 | move 0.552 | turn 0.212 | move_turn 0.000 | rsi_frac_actual 0.191 | terrain_levels 1.027 | mean ep len 1695.640 | prints 3000
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
- suite stand: n 100 surv 1.000 drift 0.098 [decisive]
- suite fwd: n 100 surv 0.710 ratio 0.651 tripod 0.496 [decisive]
- suite side: n 200 surv 0.980 ratio 0.438 tripod 0.514 (l: surv 0.990 ratio 0.442 tripod 0.511; r: surv 0.970 ratio 0.433 tripod 0.520) [decisive]
- suite back: n 100 surv 0.680 ratio 0.680 tripod 0.520 [decisive]
- suite diag: n 100 surv 0.510 ratio 0.598 tripod 0.403 [WATCH: untrained]
- suite turn: n 200 surv 0.900 closed 0.212 progress 0.028 (l: surv 0.910 closed 0.224 progress 0.029; r: surv 0.890 closed 0.210 progress 0.027) [decisive]
- suite move_turn: n 200 surv 0.055 closed 0.033 progress 0.009 ratio 0.421 (l: surv 0.060 closed 0.028 progress 0.009 ratio 0.421; r: surv 0.050 closed 0.036 progress 0.009 ratio 0.423) [WATCH: untrained]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.298 (need >= 0.4; by hold: back 0.520, diag 0.403, fwd 0.496, lat_l 0.511, lat_r 0.520, move_turn 0.303, move_turn_r 0.298, settle 0.061, stand 0.544, turn_l90 0.527, turn_r90 0.500), completion 0.676 (need >= 0.7), yaw-progress rate turn_l90 0.029, turn_r90 0.027 rad/s (net turn_l90 0.029, turn_r90 0.028) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.442, diag 0.598, back 0.680, move_turn 0.421 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.594 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.158 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 1.000 (bound 1.000, n_env 100, new >= 0.1); fwd 0.710 (bound 0.816, n_env 100, new >= 0.1); side 0.980 (bound 1.013, n_env 100, new >= 0.1); back 0.680 (bound 0.789, n_env 100, new >= 0.1); turn 0.900 (bound 0.970, n_env 100, new >= 0.1)

>>> ENTRY X06:t3

## D06 — P_f — REJECT (clear, final) — 2026-09-27T05:38:07

- question: Armed from iteration 0 of formation (C1d's t3, from scratch, clock held), does the side body-centring bonus at w_P keep the lateral posture from forming -- pooled side body_offset <= -0.05 m against B_f over seeds 3 and 2 -- and does the recentred posture lengthen side strides?
- experiment `X05` recipe `bc00f504d0ce` base `d695a100e1a7` control `X01`
- why: pooled WORSE on block F, guard cat.turn.steps_per_rad
- block F (F): z -3.720 vs R 1.256 -> WORSE (k=2, missing [])
  - cat.turn.facing_closed: 0.467 -> 0.292 (delta -0.187, R_eval 0.050, R_seed 0.055, z -3.765)
  - cat.turn.yaw_progress: 0.062 -> 0.039 (delta -0.024, R_eval 0.007, R_seed 0.009, z -3.674)
- block T (T): z 1.321 vs R 0.985 -> BETTER (k=3, missing [])
  - cat.fwd.ratio: 0.474 -> 0.625 (delta 0.130, R_eval 0.041, R_seed 0.068, z 3.165)
  - cat.side.ratio: 0.458 -> 0.451 (delta -0.023, R_eval 0.018, R_seed 0.020, z -1.294)
  - cat.back.ratio: 0.533 -> 0.618 (delta 0.110, R_eval 0.053, R_seed 0.011, z 2.093)
- block LEN (LEN): z 0.375 vs R 1.551 -> TIE (k=1, missing [])
  - cat.side.steps_per_m: 32.390 -> 31.280 (lower is better, signed delta 0.517, R_eval 1.377, R_seed 2.136, z 0.375)
- guard cat.stand.survival: 0.930 -> 1.000 (R 0.051) -> TIE
- guard cat.stand.steps_per_s: 3.083 -> 3.250 (R 0.334) -> TIE
- guard cat.fwd.survival: 0.490 -> 0.950 (R 0.427) -> TIE
- guard cat.fwd.ratio: 0.474 -> 0.625 (R 0.077) -> TIE
- guard cat.fwd.foot_slip: 0.124 -> 0.150 (R 0.039) -> TIE
- guard cat.fwd.steps_per_m: 27.236 -> 19.765 (R 4.701) -> TIE
- guard cat.side.survival: 0.760 -> 0.975 (R 0.212) -> TIE
- guard cat.side.ratio: 0.458 -> 0.451 (R 0.025) -> TIE
- guard cat.side.foot_slip: 0.179 -> 0.201 (R 0.039) -> TIE
- guard cat.side.steps_per_m: 32.390 -> 31.280 (R 2.407) -> TIE
- guard cat.back.survival: 0.510 -> 0.950 (R 0.466) -> TIE
- guard cat.back.ratio: 0.533 -> 0.618 (R 0.045) -> TIE
- guard cat.back.foot_slip: 0.127 -> 0.152 (R 0.029) -> TIE
- guard cat.back.steps_per_m: 29.679 -> 22.313 (R 4.291) -> TIE
- guard cat.turn.survival: 0.925 -> 1.000 (R 0.068) -> TIE
- guard cat.turn.foot_slip: 0.106 -> 0.144 (R 0.019) -> TIE
- guard cat.turn.steps_per_rad: 62.731 -> 82.691 (R 3.738) -> WORSE
- anchored drift: FAIL — drifted below the anchor on block F, guard cat.turn.steps_per_rad

>>> ENTRY D06

### experiment X06 — P_f s2 — ok — 2026-09-27T05:38:07

- recipe `bc00f504d0ce` base `d695a100e1a7` chain `t3f` seed 2
- stages ['X06:t3']
- screen: completion 0.676, tripod_dir 0.403, track tail 0.594
- replay: none

>>> ENTRY X06


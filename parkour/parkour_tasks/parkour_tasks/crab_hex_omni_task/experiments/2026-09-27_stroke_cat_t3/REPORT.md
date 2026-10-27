# Candidate ladder `2026-09-27_stroke_cat_t3`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 3, confirm seed 2, 3000 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `fb17d2ea389eb13c5719fccbf5cf2de4abc4bb2ead470b7fb4ab6b9683866ec0` · `candidates.yaml` sha256 `b5f9639a62ae4f12b936c04e1cfb2475739384441c640db5e4c80dbf704581d6`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | B_f | baseline | t3f | the incumbent recipe at equal compute | — |

### X01:t3 — ref:S_cat_f s3 (reference) — 2026-09-28T18:04:54

- recipe `87987a2fcfa1` on chain `t3f_scat`, base `87987a2fcfa1`, code_sha `fb17d2ea389e`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1', 'KRABBY_OMNI_STROKE_W': '0.112339', 'KRABBY_OMNI_STROKE_CAT_W': 'fwd:1.000000,back:1.000000,side:3.460881,diag:1.599056,turn:2.115884,move_turn:1.000000'}`
- status **ok** | checkpoint `2026-09-28_16-40-58_seed3_X01_t3_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.27 h
- smoke fail@2k 0.220 fail@end 0.235 coll@2k 0.000 ep_len@2k 1801.578
- training translation floor: track_ratio_mean (last 500 its) 0.619 (gate floor 0.6)
- exposure (ring): reach_edge 0.519 | reach_obst 0.421 | field_frac 0.177 (steps 341.748) | goals_passed 0.695
- terrain-promotion telemetry: track_ratio_mean 0.621 (move rows only 0.621) | heading_err_rms_mean 0.138 | block_steps_mean 99.940 | r_max_mean 2.368
- failure share flat 0.210 | obst 0.253 | campaign-wide 0.204
- slot fractions (logged): stand 0.125 | move 0.632 | turn 0.196 | move_turn 0.000 | rsi_frac_actual 0.198 | terrain_levels 0.973 | mean ep len 1837.246 | prints 3000
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
- suite stand: n 100 surv 0.990 drift 0.111 [decisive]
- suite fwd: n 100 surv 0.960 ratio 0.662 tripod 0.565 [decisive]
- suite side: n 200 surv 0.990 ratio 0.539 tripod 0.428 (l: surv 0.990 ratio 0.535 tripod 0.426; r: surv 0.990 ratio 0.542 tripod 0.434) [decisive]
- suite back: n 100 surv 0.980 ratio 0.664 tripod 0.582 [decisive]
- suite diag: n 100 surv 0.870 ratio 0.541 tripod 0.464 [WATCH: untrained]
- suite turn: n 200 surv 0.935 closed 0.716 progress 0.093 (l: surv 0.910 closed 0.699 progress 0.092; r: surv 0.960 closed 0.740 progress 0.095) [decisive]
- suite move_turn: n 200 surv 0.790 closed 0.172 progress 0.025 ratio 0.419 (l: surv 0.750 closed 0.163 progress 0.023 ratio 0.419; r: surv 0.830 closed 0.194 progress 0.029 ratio 0.420) [WATCH: untrained]
- WATCH gate_1c: PASS -- formation gate passed — walking-hold tripod 0.426 (by hold: back 0.582, diag 0.464, fwd 0.565, lat_l 0.426, lat_r 0.434, move_turn 0.511, move_turn_r 0.522, settle 0.414, stand 0.590, turn_l90 0.581, turn_r90 0.568), completion 0.922, yaw-progress rate turn_l90 0.092, turn_r90 0.095 rad/s (net turn_l90 0.092, turn_r90 0.096), per-direction speed_ratio lat_l 0.535, diag 0.541, back 0.664, move_turn 0.419 on slow__omni, training track_ratio_mean 0.619 (need >= 0.6 over the last 500 its)
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_CAT_W=fwd:1.000000,back:1.000000,side:3.460881,diag:1.599056,turn:2.115884,move_turn:1.000000 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.112339 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.990 (bound 1.013, n_env 100, new >= 0.1); fwd 0.960 (bound 1.006, n_env 100, new >= 0.1); side 0.990 (bound 1.013, n_env 100, new >= 0.1); back 0.980 (bound 1.013, n_env 100, new >= 0.1); turn 0.935 (bound 0.992, n_env 100, new >= 0.1)

>>> ENTRY X01:t3

### X02:t3 — ref:S_cat_f s2 (reference) — 2026-09-28T19:40:21

- recipe `87987a2fcfa1` on chain `t3f_scat`, base `87987a2fcfa1`, code_sha `fb17d2ea389e`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2', 'KRABBY_OMNI_SLOTS': '0.15:0.65:0.2:0', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_CENTRING_W': '0.001', 'KRABBY_OMNI_REWARD_SPLIT': '1', 'KRABBY_OMNI_STROKE_W': '0.112339', 'KRABBY_OMNI_STROKE_CAT_W': 'fwd:1.000000,back:1.000000,side:3.460881,diag:1.599056,turn:2.115884,move_turn:1.000000'}`
- status **ok** | checkpoint `2026-09-28_18-05-55_seed2_X02_t3_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.43 h
- smoke fail@2k 0.264 fail@end 0.200 coll@2k 0.000 ep_len@2k 1744.430
- training translation floor: track_ratio_mean (last 500 its) 0.594 (gate floor 0.6)
- exposure (ring): reach_edge 0.432 | reach_obst 0.381 | field_frac 0.128 (steps 246.606) | goals_passed 0.550
- terrain-promotion telemetry: track_ratio_mean 0.596 (move rows only 0.596) | heading_err_rms_mean 0.144 | block_steps_mean 79.535 | r_max_mean 2.189
- failure share flat 0.238 | obst 0.317 | campaign-wide 0.228
- slot fractions (logged): stand 0.111 | move 0.615 | turn 0.157 | move_turn 0.000 | rsi_frac_actual 0.217 | terrain_levels 1.027 | mean ep len 1763.209 | prints 3000
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
- suite stand: n 100 surv 0.990 drift 0.109 [decisive]
- suite fwd: n 100 surv 0.840 ratio 0.565 tripod 0.475 [decisive]
- suite side: n 200 surv 0.960 ratio 0.468 tripod 0.437 (l: surv 0.950 ratio 0.467 tripod 0.449; r: surv 0.970 ratio 0.471 tripod 0.437) [decisive]
- suite back: n 100 surv 0.860 ratio 0.580 tripod 0.502 [decisive]
- suite diag: n 100 surv 0.630 ratio 0.548 tripod 0.408 [WATCH: untrained]
- suite turn: n 200 surv 0.875 closed 0.394 progress 0.055 (l: surv 0.830 closed 0.354 progress 0.051; r: surv 0.920 closed 0.457 progress 0.060) [decisive]
- suite move_turn: n 200 surv 0.615 closed -0.058 progress -0.010 ratio 0.346 (l: surv 0.500 closed -0.006 progress -0.001 ratio 0.349; r: surv 0.730 closed -0.079 progress -0.011 ratio 0.344) [WATCH: untrained]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.385 (need >= 0.4; by hold: back 0.502, diag 0.408, fwd 0.475, lat_l 0.449, lat_r 0.437, move_turn 0.385, move_turn_r 0.417, settle 0.083, stand 0.575, turn_l90 0.555, turn_r90 0.574), completion 0.822 (need >= 0.7), yaw-progress rate turn_l90 0.051, turn_r90 0.060 rad/s (net turn_l90 0.052, turn_r90 0.061) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.467, diag 0.548, back 0.580, move_turn 0.349 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.594 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CENTRING_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.25,side_l:0.16,side_r:0.16,back:0.08,turn:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_REWARD_SPLIT=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.65:0.2:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_CAT_W=fwd:1.000000,back:1.000000,side:3.460881,diag:1.599056,turn:2.115884,move_turn:1.000000 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_STROKE_W=0.112339 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.990 (bound 1.013, n_env 100, new >= 0.1); fwd 0.840 (bound 0.925, n_env 100, new >= 0.1); side 0.960 (bound 1.006, n_env 100, new >= 0.1); back 0.860 (bound 0.941, n_env 100, new >= 0.1); turn 0.875 (bound 0.952, n_env 100, new >= 0.1)

>>> ENTRY X02:t3


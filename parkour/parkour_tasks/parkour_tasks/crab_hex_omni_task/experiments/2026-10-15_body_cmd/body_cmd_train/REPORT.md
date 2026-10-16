# Candidate ladder `body_cmd_train`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 18, confirm seed 16, 100 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `fe714b41490619fd31bf8698f5a5629b58445b37c552a33090c4f4f391175b75` · `candidates.yaml` sha256 `1f985c749b36a60936e0f23c71c46d7d48f8c9e9e27c36a2d33db91c6e3f4962`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | BC_SMK_1b_s18 | baseline | BC_SMK_1b_s18 | the incumbent recipe at equal compute | — |

### X01:fb6k — ref:BC_SMK_1b_s18 s18 (reference) — 2026-10-16T01:59:04

- recipe `8a5ae1f18bd6` on chain `BC_SMK_1b_s18`, base `8a5ae1f18bd6`, code_sha `fe714b414906`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,fwd:0.2,side_l:0.2,side_r:0.2,back:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1', 'KRABBY_OMNI_BODY_ENVELOPE': '0.05:0.168:0.21', 'KRABBY_OMNI_BODY_CAP': '0@0.21,0.22@0.105,0.3@0.065'}`
- status **ok** | checkpoint `2026-10-16_01-55-15_seed18_X01_fb6k_A15pB/model_99.pt` | resume `scratch (--from-scratch)` | 100 its | wall 0.07 h
- smoke fail@2k 0.274 fail@end 0.274 coll@2k 0.000 ep_len@2k 731.674
- training translation floor: track_ratio_mean (last 500 its) 0.200 (gate floor 0.6)
- exposure (ring): reach_edge 0.000 | reach_obst 0.000 | field_frac 0.000 (steps 0.000) | goals_passed 0.000
- terrain-promotion telemetry: track_ratio_mean 0.250 (move rows only 0.250) | heading_err_rms_mean 0.001 | block_steps_mean 0.000 | r_max_mean 0.444
- failure share flat 0.507 | obst 0.429 | campaign-wide 0.599
- slot fractions (logged): stand 0.088 | move 0.490 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.039 | mean ep len 1127.208 | prints 100
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
- check FAIL: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_CAP=0@0.21,0.22@0.105,0.3@0.065 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_ENVELOPE=0.05:0.168:0.21 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,fwd:0.2,side_l:0.2,side_r:0.2,back:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: overlay KRABBY_STD_PARAM=soft in params/agent.yaml
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_LOSS consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_RAMP consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND consumed
- overlay PASS: log: [KNOB] KRABBY_STD_PARAM consumed
- overlay PASS: log: [KNOB] KRABBY_INIT_STD consumed
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- overlay PASS: log: Mean action_bound printed (the bound loss ran)
- floor PASS: no floor at this stage

>>> ENTRY X01:fb6k

### X02:c4 — ref:BC_SMK_c4_s18 s18 (reference) — 2026-10-16T02:03:24

- recipe `264115dc92b1` on chain `BC_SMK_c4_s18`, base `264115dc92b1`, code_sha `fe714b414906`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.15:0.1', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1', 'KRABBY_OMNI_BODY_ENVELOPE': '0.05:0.168:0.21', 'KRABBY_OMNI_BODY_CAP': '0@0.21,0.22@0.105,0.3@0.065'}`
- status **ok** | checkpoint `2026-10-16_01-59-35_seed18_X02_c4_A15pB/model_99.pt` | resume `scratch (--from-scratch)` | 100 its | wall 0.07 h
- smoke fail@2k 0.390 fail@end 0.390 coll@2k 0.000 ep_len@2k 651.409
- training translation floor: track_ratio_mean (last 500 its) 0.229 (gate floor 0.6)
- exposure (ring): reach_edge 0.000 | reach_obst 0.000 | field_frac 0.000 (steps 0.000) | goals_passed 0.000
- terrain-promotion telemetry: track_ratio_mean 0.285 (move rows only 0.284) | heading_err_rms_mean 0.002 | block_steps_mean 0.000 | r_max_mean 0.489
- failure share flat 0.643 | obst 0.666 | campaign-wide 0.719
- slot fractions (logged): stand 0.080 | move 0.276 | turn 0.076 | move_turn 0.042 | rsi_frac_actual 0.000 | terrain_levels 1.039 | mean ep len 953.723 | prints 100
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
- check FAIL: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_CAP=0@0.21,0.22@0.105,0.3@0.065 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_ENVELOPE=0.05:0.168:0.21 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.15:0.1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: overlay KRABBY_STD_PARAM=soft in params/agent.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_LOSS consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_RAMP consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND consumed
- overlay PASS: log: [KNOB] KRABBY_STD_PARAM consumed
- overlay PASS: log: [KNOB] KRABBY_INIT_STD consumed
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- overlay PASS: log: Mean action_bound printed (the bound loss ran)
- floor PASS: no floor at this stage

>>> ENTRY X02:c4

### X03:p2a — ref:BC_SMK_p2a_s18 s18 (reference) — 2026-10-16T02:07:45

- recipe `9700636bf06b` on chain `BC_SMK_p2a_s18`, base `9700636bf06b`, code_sha `fe714b414906`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '-0.05', 'KRABBY_PPO_HAND_DIAG': '1', 'KRABBY_PHASEOUT': '', 'KRABBY_FLAT_TERRAIN_GEOM': 'field', 'KRABBY_FIELD_PAD_R': '1.7', 'KRABBY_FIELD_BLOCKS': '14:28', 'KRABBY_FIELD_KINDS': '0.4:0.4:0.2', 'KRABBY_FIELD_EDGES': 'wall', 'KRABBY_FLAT_TERRAIN_DIFF': '0.20:1.0', 'KRABBY_TERRAIN_PROMOTE': '0.7:0.4', 'KRABBY_OMNI_BODY_ENVELOPE': '0.05:0.168:0.21', 'KRABBY_OMNI_BODY_CAP': '0@0.21,0.22@0.105,0.3@0.065'}`
- status **ok** | checkpoint `2026-10-16_02-03-55_seed18_X03_p2a_A15pB/model_99.pt` | resume `scratch (--from-scratch)` | 100 its | wall 0.07 h
- smoke fail@2k 0.188 fail@end 0.188 coll@2k -0.010 ep_len@2k 756.288
- training translation floor: track_ratio_mean (last 500 its) 0.193 (gate floor 0.6)
- exposure (ring): reach_edge 0.000 | reach_obst 0.000 | field_frac 0.000 (steps 0.000) | goals_passed 0.000
- terrain-promotion telemetry: track_ratio_mean 0.249 (move rows only 0.250) | heading_err_rms_mean 0.002 | block_steps_mean 0.000 | r_max_mean 0.443
- failure share flat 0.428 | obst 0.404 | campaign-wide 0.517
- slot fractions (logged): stand 0.068 | move 0.291 | turn 0.047 | move_turn 0.116 | rsi_frac_actual 0.000 | terrain_levels 0.289 | mean ep len 1158.494 | prints 100
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
- check FAIL: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: field generator installed (omni_field_sparse/omni_field_dense sub-terrains, no omni_ring_*) in params/env.yaml
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=-0.05 in params/env.yaml
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_FIELD_BLOCKS=14:28 in params/env.yaml
- overlay PASS: overlay KRABBY_FIELD_EDGES=wall in params/env.yaml
- overlay PASS: overlay KRABBY_FIELD_KINDS=0.4:0.4:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_FIELD_PAD_R=1.7 in params/env.yaml
- overlay PASS: overlay KRABBY_FLAT_TERRAIN_DIFF=0.20:1.0 in params/env.yaml
- overlay PASS: overlay KRABBY_FLAT_TERRAIN_GEOM=field in params/env.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_CAP=0@0.21,0.22@0.105,0.3@0.065 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_ENVELOPE=0.05:0.168:0.21 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PHASEOUT= in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: overlay KRABBY_STD_PARAM=soft in params/agent.yaml
- overlay PASS: overlay KRABBY_TERRAIN_PROMOTE=0.7:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_LOSS consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_RAMP consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND consumed
- overlay PASS: log: [KNOB] KRABBY_STD_PARAM consumed
- overlay PASS: log: [KNOB] KRABBY_INIT_STD consumed
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- overlay PASS: log: Mean action_bound printed (the bound loss ran)
- floor PASS: no floor at this stage

>>> ENTRY X03:p2a

### X04:fb6k — ref:BC_fb6k_s18 s18 (reference) — 2026-10-16T05:23:44

- recipe `cc9550047f12` on chain `BC_fb6k_s18`, base `cc9550047f12`, code_sha `fe714b414906`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,fwd:0.2,side_l:0.2,side_r:0.2,back:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1', 'KRABBY_OMNI_BODY_ENVELOPE': '0.05:0.168:0.21', 'KRABBY_OMNI_BODY_CAP': '0@0.21,0.22@0.105,0.3@0.065'}`
- status **ok** | checkpoint `2026-10-16_02-58-20_seed18_X04_fb6k_A15pB/model_5999.pt` | resume `scratch (--from-scratch)` | 6000 its | wall 2.30 h
- smoke fail@2k 0.092 fail@end 0.105 coll@2k 0.000 ep_len@2k 1924.551
- training translation floor: track_ratio_mean (last 500 its) 0.699 (gate floor 0.6)
- exposure (ring): reach_edge 0.876 | reach_obst 0.732 | field_frac 0.389 (steps 767.216) | goals_passed 1.423
- terrain-promotion telemetry: track_ratio_mean 0.715 (move rows only 0.715) | heading_err_rms_mean 0.002 | block_steps_mean 190.372 | r_max_mean 3.402
- failure share flat 0.087 | obst 0.096 | campaign-wide 0.091
- slot fractions (logged): stand 0.198 | move 0.790 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.039 | mean ep len 1910.268 | prints 6000
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
- check FAIL: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- suite_body trained categories: decisive stand, fwd, side, back
- suite_body stand: n 100 surv 1.000 drift 0.032 [decisive]
- suite_body fwd: n 100 surv 1.000 ratio 0.980 tripod 0.477 [decisive]
- suite_body side: n 200 surv 1.000 ratio 0.830 tripod 0.734 (l: surv 1.000 ratio 0.785 tripod 0.759; r: surv 1.000 ratio 0.865 tripod 0.716) [decisive]
- suite_body back: n 100 surv 1.000 ratio 0.893 tripod 0.388 [decisive]
- suite_body diag: n 100 surv 1.000 ratio 0.829 tripod 0.554 [WATCH: untrained]
- suite_body turn: n 200 surv 1.000 closed None progress None (l: surv 1.000 closed None progress None; r: surv 1.000 closed None progress None) [WATCH: untrained]
- suite_body move_turn: n 200 surv 1.000 closed None progress None ratio 0.894 (l: surv 1.000 closed None progress None ratio 0.902; r: surv 1.000 closed None progress None ratio 0.889) [WATCH: untrained]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_CAP=0@0.21,0.22@0.105,0.3@0.065 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_ENVELOPE=0.05:0.168:0.21 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,fwd:0.2,side_l:0.2,side_r:0.2,back:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: overlay KRABBY_STD_PARAM=soft in params/agent.yaml
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_LOSS consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_RAMP consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND consumed
- overlay PASS: log: [KNOB] KRABBY_STD_PARAM consumed
- overlay PASS: log: [KNOB] KRABBY_INIT_STD consumed
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- overlay PASS: log: Mean action_bound printed (the bound loss ran)
- floor PASS: no floor at this stage

>>> ENTRY X04:fb6k

### X05:c4 — ref:BC_c4_s18 s18 (reference) — 2026-10-16T06:08:03

- recipe `590f868ea0b2` on chain `BC_c4_s18`, base `590f868ea0b2`, code_sha `fe714b414906`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.15:0.1', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1', 'KRABBY_OMNI_BODY_ENVELOPE': '0.05:0.168:0.21', 'KRABBY_OMNI_BODY_CAP': '0@0.21,0.22@0.105,0.3@0.065'}`
- status **ok** | checkpoint `2026-10-16_05-24-43_seed18_X05_c4_A15pB/model_7498.pt` | resume `model_5999.pt` | 1500 its | wall 0.60 h
- smoke fail@2k 0.137 fail@end 0.137 coll@2k 0.000 ep_len@2k 1908.561
- training translation floor: track_ratio_mean (last 500 its) 0.709 (gate floor 0.6)
- exposure (ring): reach_edge 0.853 | reach_obst 0.712 | field_frac 0.390 (steps 761.817) | goals_passed 1.329
- terrain-promotion telemetry: track_ratio_mean 0.718 (move rows only 0.723) | heading_err_rms_mean 0.002 | block_steps_mean 204.078 | r_max_mean 3.366
- failure share flat 0.078 | obst 0.086 | campaign-wide 0.080
- slot fractions (logged): stand 0.135 | move 0.599 | turn 0.157 | move_turn 0.106 | rsi_frac_actual 0.000 | terrain_levels 1.039 | mean ep len 1995.285 | prints 1500
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
- check FAIL: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- suite_body trained categories: decisive stand, fwd, side, back, turn, move_turn
- suite_body stand: n 100 surv 1.000 drift 0.032 [decisive]
- suite_body fwd: n 100 surv 1.000 ratio 0.823 tripod 0.428 [decisive]
- suite_body side: n 200 surv 1.000 ratio 0.714 tripod 0.746 (l: surv 1.000 ratio 0.608 tripod 0.804; r: surv 1.000 ratio 0.877 tripod 0.703) [decisive]
- suite_body back: n 100 surv 1.000 ratio 0.874 tripod 0.583 [decisive]
- suite_body diag: n 100 surv 0.920 ratio 0.737 tripod 0.448 [WATCH: untrained]
- suite_body turn: n 200 surv 1.000 closed None progress None (l: surv 1.000 closed None progress None; r: surv 1.000 closed None progress None) [decisive]
- suite_body move_turn: n 200 surv 1.000 closed None progress None ratio 0.783 (l: surv 1.000 closed None progress None ratio 0.787; r: surv 1.000 closed None progress None ratio 0.780) [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_CAP=0@0.21,0.22@0.105,0.3@0.065 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_ENVELOPE=0.05:0.168:0.21 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.15:0.1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: overlay KRABBY_STD_PARAM=soft in params/agent.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_LOSS consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_RAMP consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND consumed
- overlay PASS: log: [KNOB] KRABBY_STD_PARAM consumed
- overlay PASS: log: [KNOB] KRABBY_INIT_STD consumed
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- overlay PASS: log: Mean action_bound printed (the bound loss ran)
- floor PASS: no floor at this stage

>>> ENTRY X05:c4

### X06:c5 — ref:BC_c5_s18 s18 (reference) — 2026-10-16T06:52:26

- recipe `631b8f047671` on chain `BC_c5_s18`, base `631b8f047671`, code_sha `fe714b414906`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.1:0.15', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1', 'KRABBY_OMNI_BODY_ENVELOPE': '0.05:0.168:0.21', 'KRABBY_OMNI_BODY_CAP': '0@0.21,0.22@0.105,0.3@0.065'}`
- status **ok** | checkpoint `2026-10-16_06-09-05_seed18_X06_c5_A15pB/model_8997.pt` | resume `model_7498.pt` | 1500 its | wall 0.60 h
- smoke fail@2k 0.075 fail@end 0.075 coll@2k 0.000 ep_len@2k 1953.389
- training translation floor: track_ratio_mean (last 500 its) 0.735 (gate floor 0.6)
- exposure (ring): reach_edge 0.867 | reach_obst 0.731 | field_frac 0.415 (steps 805.782) | goals_passed 1.409
- terrain-promotion telemetry: track_ratio_mean 0.732 (move rows only 0.737) | heading_err_rms_mean 0.002 | block_steps_mean 198.001 | r_max_mean 3.446
- failure share flat 0.081 | obst 0.101 | campaign-wide 0.068
- slot fractions (logged): stand 0.152 | move 0.573 | turn 0.079 | move_turn 0.167 | rsi_frac_actual 0.000 | terrain_levels 1.039 | mean ep len 1947.766 | prints 1500
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
- check FAIL: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- suite_body trained categories: decisive stand, fwd, side, back, diag, turn, move_turn
- suite_body stand: n 100 surv 1.000 drift 0.034 [decisive]
- suite_body fwd: n 100 surv 1.000 ratio 0.889 tripod 0.589 [decisive]
- suite_body side: n 200 surv 1.000 ratio 0.728 tripod 0.766 (l: surv 1.000 ratio 0.751 tripod 0.759; r: surv 1.000 ratio 0.694 tripod 0.771) [decisive]
- suite_body back: n 100 surv 1.000 ratio 0.868 tripod 0.623 [decisive]
- suite_body diag: n 100 surv 1.000 ratio 0.780 tripod 0.700 [decisive]
- suite_body turn: n 200 surv 1.000 closed None progress None (l: surv 1.000 closed None progress None; r: surv 1.000 closed None progress None) [decisive]
- suite_body move_turn: n 200 surv 0.995 closed None progress None ratio 0.858 (l: surv 1.000 closed None progress None ratio 0.864; r: surv 0.990 closed None progress None ratio 0.854) [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_CAP=0@0.21,0.22@0.105,0.3@0.065 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_ENVELOPE=0.05:0.168:0.21 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.6:0.1:0.15 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: overlay KRABBY_STD_PARAM=soft in params/agent.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_LOSS consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_RAMP consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND consumed
- overlay PASS: log: [KNOB] KRABBY_STD_PARAM consumed
- overlay PASS: log: [KNOB] KRABBY_INIT_STD consumed
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- overlay PASS: log: Mean action_bound printed (the bound loss ran)
- floor PASS: no floor at this stage

>>> ENTRY X06:c5

### X07:c6 — ref:BC_c6_s18 s18 (reference) — 2026-10-16T07:48:48

- recipe `7fc7fef50dd7` on chain `BC_c6_s18`, base `7fc7fef50dd7`, code_sha `fe714b414906`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1', 'KRABBY_OMNI_BODY_ENVELOPE': '0.05:0.168:0.21', 'KRABBY_OMNI_BODY_CAP': '0@0.21,0.22@0.105,0.3@0.065'}`
- status **ok** | checkpoint `2026-10-16_06-53-27_seed18_X07_c6_A15pB/model_10996.pt` | resume `model_8997.pt` | 2000 its | wall 0.80 h
- smoke fail@2k 0.135 fail@end 0.135 coll@2k 0.000 ep_len@2k 1894.086
- training translation floor: track_ratio_mean (last 500 its) 0.755 (gate floor 0.6)
- exposure (ring): reach_edge 0.888 | reach_obst 0.735 | field_frac 0.422 (steps 820.376) | goals_passed 1.491
- terrain-promotion telemetry: track_ratio_mean 0.761 (move rows only 0.768) | heading_err_rms_mean 0.002 | block_steps_mean 205.332 | r_max_mean 3.498
- failure share flat 0.130 | obst 0.141 | campaign-wide 0.128
- slot fractions (logged): stand 0.095 | move 0.596 | turn 0.145 | move_turn 0.152 | rsi_frac_actual 0.000 | terrain_levels 1.039 | mean ep len 1914.750 | prints 2000
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
- check FAIL: omni command class (CrabHexOmniCommand) in params/env.yaml
- check PASS: omni obs func (CrabHexOmniParkourObservations) in params/env.yaml
- check PASS: ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- suite_body trained categories: decisive stand, fwd, side, back, diag, turn, move_turn
- suite_body stand: n 100 surv 1.000 drift 0.032 [decisive]
- suite_body fwd: n 100 surv 1.000 ratio 0.917 tripod 0.606 [decisive]
- suite_body side: n 200 surv 1.000 ratio 0.863 tripod 0.761 (l: surv 1.000 ratio 0.809 tripod 0.753; r: surv 1.000 ratio 0.934 tripod 0.769) [decisive]
- suite_body back: n 100 surv 1.000 ratio 0.890 tripod 0.568 [decisive]
- suite_body diag: n 100 surv 1.000 ratio 0.867 tripod 0.650 [decisive]
- suite_body turn: n 200 surv 1.000 closed None progress None (l: surv 1.000 closed None progress None; r: surv 1.000 closed None progress None) [decisive]
- suite_body move_turn: n 200 surv 1.000 closed None progress None ratio 0.871 (l: surv 1.000 closed None progress None ratio 0.867; r: surv 1.000 closed None progress None ratio 0.879) [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_CAP=0@0.21,0.22@0.105,0.3@0.065 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_BODY_ENVELOPE=0.05:0.168:0.21 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: overlay KRABBY_STD_PARAM=soft in params/agent.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_LOSS consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_RAMP consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND consumed
- overlay PASS: log: [KNOB] KRABBY_STD_PARAM consumed
- overlay PASS: log: [KNOB] KRABBY_INIT_STD consumed
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- overlay PASS: log: Mean action_bound printed (the bound loss ran)
- floor PASS: no floor at this stage

>>> ENTRY X07:c6


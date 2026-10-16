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


# Candidate ladder `preflight`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 18, confirm seed 16, 20 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `a85193431a87f09178ea65fd81bfa9df7eaeb9b19dc0dfc033485c74c7a6fce6` · `candidates.yaml` sha256 `69c761e38a140d840cdc4b2b2312e4f09b151a0a2cef02a38816d41d65e66f00`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | PF_s1_s18 | baseline | PF_s1_s18 | the incumbent recipe at equal compute | — |

### X01:s1 — ref:PF_s1_s18 s18 (reference) — 2026-10-16T17:32:15

- recipe `f529328e6f97` on chain `PF_s1_s18`, base `f529328e6f97`, code_sha `a85193431a87`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,fwd:0.2,side_l:0.2,side_r:0.2,back:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-16_17-30-26_seed18_X01_s1_A15pB/model_19.pt` | resume `scratch (--from-scratch)` | 20 its | wall 0.03 h
- smoke fail@2k 0.001 fail@end 0.001 coll@2k 0.000 ep_len@2k 124.781
- training translation floor: track_ratio_mean (last 500 its) 0.111 (gate floor 0.6)
- exposure (ring): reach_edge 0.000 | reach_obst 0.000 | field_frac 0.000 (steps 0.000) | goals_passed 0.000
- terrain-promotion telemetry: track_ratio_mean 0.111 (move rows only 0.111) | heading_err_rms_mean 0.018 | block_steps_mean 0.000 | r_max_mean 0.080
- failure share flat 0.002 | obst 0.000 | campaign-wide 0.001
- slot fractions (logged): stand 0.016 | move 0.098 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.030 | mean ep len 124.781 | prints 20
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
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

>>> ENTRY X01:s1

### X02:c4 — ref:PF_c4_s18 s18 (reference) — 2026-10-16T17:34:36

- recipe `5bd5603c0a0f` on chain `PF_c4_s18`, base `5bd5603c0a0f`, code_sha `a85193431a87`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.15:0.1', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-16_17-32-46_seed18_X02_c4_A15pB/model_6018.pt` | resume `model_5999.pt` | 20 its | wall 0.03 h
- smoke fail@2k 0.012 fail@end 0.012 coll@2k 0.000 ep_len@2k 128.572
- training translation floor: track_ratio_mean (last 500 its) 0.506 (gate floor 0.6)
- exposure (ring): reach_edge 0.000 | reach_obst 0.000 | field_frac 0.000 (steps 0.000) | goals_passed 0.000
- terrain-promotion telemetry: track_ratio_mean 0.506 (move rows only 0.448) | heading_err_rms_mean 0.083 | block_steps_mean 0.000 | r_max_mean 0.371
- failure share flat 0.059 | obst 0.026 | campaign-wide 0.012
- slot fractions (logged): stand 0.012 | move 0.062 | turn 0.017 | move_turn 0.025 | rsi_frac_actual 0.000 | terrain_levels 1.030 | mean ep len 128.572 | prints 20
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.22,side_l:0.14,side_r:0.14,back:0.1,turn:0.15,fwd_turn:0.05,side_l_turn:0.025,side_r_turn:0.025 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
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

### X03:c5 — ref:PF_c5_s18 s18 (reference) — 2026-10-16T17:37:56

- recipe `a74e98911805` on chain `PF_c5_s18`, base `a74e98911805`, code_sha `a85193431a87`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04', 'KRABBY_OMNI_SLOTS': '0.15:0.6:0.1:0.15', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-16_17-36-06_seed18_X03_c5_A15pB/model_7517.pt` | resume `model_7498.pt` | 20 its | wall 0.03 h
- smoke fail@2k 0.013 fail@end 0.013 coll@2k 0.000 ep_len@2k 127.188
- training translation floor: track_ratio_mean (last 500 its) 0.532 (gate floor 0.6)
- exposure (ring): reach_edge 0.009 | reach_obst 0.000 | field_frac 0.000 (steps 0.190) | goals_passed 0.009
- terrain-promotion telemetry: track_ratio_mean 0.532 (move rows only 0.439) | heading_err_rms_mean 0.083 | block_steps_mean 0.000 | r_max_mean 0.373
- failure share flat 0.073 | obst 0.000 | campaign-wide 0.013
- slot fractions (logged): stand 0.012 | move 0.062 | turn 0.014 | move_turn 0.028 | rsi_frac_actual 0.000 | terrain_levels 1.030 | mean ep len 127.188 | prints 20
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.15,side_l:0.1,side_r:0.1,back:0.1,omni:0.15,turn:0.1,fwd_turn:0.07,side_l_turn:0.04,side_r_turn:0.04 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
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

>>> ENTRY X03:c5

### X04:c6 — ref:PF_c6_s18 s18 (reference) — 2026-10-16T17:40:16

- recipe `7fb8100bc1d5` on chain `PF_c6_s18`, base `7fb8100bc1d5`, code_sha `a85193431a87`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-16_17-38-26_seed18_X04_c6_A15pB/model_9016.pt` | resume `model_8997.pt` | 20 its | wall 0.03 h
- smoke fail@2k 0.015 fail@end 0.015 coll@2k 0.000 ep_len@2k 130.236
- training translation floor: track_ratio_mean (last 500 its) 0.446 (gate floor 0.6)
- exposure (ring): reach_edge 0.008 | reach_obst 0.000 | field_frac 0.000 (steps 0.192) | goals_passed 0.008
- terrain-promotion telemetry: track_ratio_mean 0.446 (move rows only 0.438) | heading_err_rms_mean 0.096 | block_steps_mean 0.000 | r_max_mean 0.398
- failure share flat 0.062 | obst 0.048 | campaign-wide 0.015
- slot fractions (logged): stand 0.010 | move 0.061 | turn 0.007 | move_turn 0.039 | rsi_frac_actual 0.000 | terrain_levels 1.030 | mean ep len 130.236 | prints 20
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
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

>>> ENTRY X04:c6

### X05:c6x — ref:PF_c6x_s18 s18 (reference) — 2026-10-16T17:42:36

- recipe `78bb243ace7b` on chain `PF_c6x_s18`, base `78bb243ace7b`, code_sha `a85193431a87`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-16_17-40-46_seed18_X05_c6x_A15pB/model_11015.pt` | resume `model_10996.pt` | 20 its | wall 0.03 h
- smoke fail@2k 0.119 fail@end 0.119 coll@2k 0.000 ep_len@2k 158.157
- training translation floor: track_ratio_mean (last 500 its) 0.440 (gate floor 0.6)
- exposure (ring): reach_edge 0.000 | reach_obst 0.000 | field_frac 0.000 (steps 0.000) | goals_passed 0.000
- terrain-promotion telemetry: track_ratio_mean 0.440 (move rows only 0.427) | heading_err_rms_mean 0.089 | block_steps_mean 0.000 | r_max_mean 0.457
- failure share flat 0.311 | obst 0.252 | campaign-wide 0.119
- slot fractions (logged): stand 0.007 | move 0.069 | turn 0.006 | move_turn 0.028 | rsi_frac_actual 0.000 | terrain_levels 1.030 | mean ep len 158.157 | prints 20
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_APEX_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_CLOCK_W=0.001 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
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

>>> ENTRY X05:c6x

### X06:p2a — ref:PF_p2a_s18 s18 (reference) — 2026-10-16T17:44:56

- recipe `a23a1ee60ca7` on chain `PF_p2a_s18`, base `a23a1ee60ca7`, code_sha `a85193431a87`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_FACING': '0.2:1.2', 'KRABBY_YAW_W': '0.2', 'KRABBY_CLOCK_W': '0.001', 'KRABBY_APEX_W': '0.001', 'KRABBY_OMNI_CLOCK_HIDE': '1', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '-0.05', 'KRABBY_PPO_HAND_DIAG': '1', 'KRABBY_PHASEOUT': '', 'KRABBY_FLAT_TERRAIN_GEOM': 'field', 'KRABBY_FIELD_PAD_R': '1.7', 'KRABBY_FIELD_BLOCKS': '14:28', 'KRABBY_FIELD_KINDS': '0.4:0.4:0.2', 'KRABBY_FIELD_EDGES': 'wall', 'KRABBY_FLAT_TERRAIN_DIFF': '0.20:1.0', 'KRABBY_TERRAIN_PROMOTE': '0.7:0.4'}`
- status **ok** | checkpoint `2026-10-16_17-43-06_seed18_X06_p2a_A15pB/model_15014.pt` | resume `model_14995.pt` | 20 its | wall 0.03 h
- smoke fail@2k 0.017 fail@end 0.017 coll@2k -0.000 ep_len@2k 135.188
- training translation floor: track_ratio_mean (last 500 its) 0.489 (gate floor 0.6)
- exposure (ring): reach_edge 0.017 | reach_obst 0.055 | field_frac 0.003 (steps 1.097) | goals_passed 0.017
- terrain-promotion telemetry: track_ratio_mean 0.489 (move rows only 0.522) | heading_err_rms_mean 0.186 | block_steps_mean 4.995 | r_max_mean 0.427
- failure share flat 0.070 | obst 0.109 | campaign-wide 0.017
- slot fractions (logged): stand 0.021 | move 0.053 | turn 0.010 | move_turn 0.034 | rsi_frac_actual 0.000 | terrain_levels 1.067 | mean ep len 135.188 | prints 20
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
- overlay PASS: overlay KRABBY_OMNI_CLOCK_HIDE=1 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:1.2 in params/env.yaml
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

>>> ENTRY X06:p2a


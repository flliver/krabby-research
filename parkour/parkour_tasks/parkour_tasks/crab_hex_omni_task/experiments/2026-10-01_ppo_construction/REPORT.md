# Candidate ladder `2026-10-01_ppo_construction`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 11, confirm seed 16, 3000 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `f7166d558c5d8d9b00e9dd08d505cc27b4db61239c213f961673c67bd4ba2c19` · `candidates.yaml` sha256 `d2a935df3a5510a2b2bc1f29077dc8350e1de29534b80ec809e684d42f9a9210`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | C__C0 | baseline | C__C0 | the incumbent recipe at equal compute | — |

### X01:s1 — ref:C__C0_s11 s11 (reference) — 2026-10-01T11:40:32

- recipe `0bc0d4761923` on chain `C__C0`, base `0bc0d4761923`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25'}`
- status **ok** | checkpoint `2026-10-01_10-32-08_seed11_X01_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.007 fail@end 0.003 coll@2k 0.000 ep_len@2k 1993.210
- training translation floor: track_ratio_mean (last 500 its) 0.501 (gate floor 0.6)
- exposure (ring): reach_edge 0.935 | reach_obst 0.653 | field_frac 0.341 (steps 682.295) | goals_passed 1.041
- terrain-promotion telemetry: track_ratio_mean 0.500 (move rows only 0.500) | heading_err_rms_mean 0.070 | block_steps_mean 140.590 | r_max_mean 3.056
- failure share flat 0.002 | obst 0.014 | campaign-wide 0.004
- slot fractions (logged): stand 0.190 | move 0.810 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.977 | mean ep len 1991.932 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 1.000 drift 0.099 [decisive]
- side22_25 side: n 200 surv 1.000 ratio 0.369 tripod 0.554 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- floor PASS: no floor at this stage

>>> ENTRY X01:s1

### X03:s1 — ref:C__A_s11_r2 s11 (reference) — 2026-10-01T16:17:30

- recipe `4649f356c7c8` on chain `C__A`, base `4649f356c7c8`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025'}`
- status **ok** | checkpoint `2026-10-01_15-11-06_seed11_X03_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.07 h
- smoke fail@2k 0.012 fail@end 0.023 coll@2k 0.000 ep_len@2k 1988.254
- training translation floor: track_ratio_mean (last 500 its) 0.821 (gate floor 0.6)
- exposure (ring): reach_edge 0.989 | reach_obst 0.727 | field_frac 0.710 (steps 1416.426) | goals_passed 1.115
- terrain-promotion telemetry: track_ratio_mean 0.810 (move rows only 0.810) | heading_err_rms_mean 0.052 | block_steps_mean 148.189 | r_max_mean 7.258
- failure share flat 0.014 | obst 0.015 | campaign-wide 0.021
- slot fractions (logged): stand 0.211 | move 0.782 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.977 | mean ep len 1976.460 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 0.840 drift 0.052 [decisive]
- side22_25 side: n 200 surv 0.950 ratio 0.959 tripod 0.730 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: overlay KRABBY_STD_PARAM=soft in params/agent.yaml
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_LOSS consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_RAMP consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND consumed
- overlay PASS: log: [KNOB] KRABBY_STD_PARAM consumed
- overlay PASS: log: [KNOB] KRABBY_INIT_STD consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- overlay PASS: log: Mean action_bound printed (the bound loss ran)
- floor PASS: no floor at this stage

>>> ENTRY X03:s1

### X04:s1 — ref:C_lr__C0_s11 s11 (reference) — 2026-10-01T17:27:08

- recipe `4bc6b989406c` on chain `C_lr__C0`, base `4bc6b989406c`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-01_16-18-44_seed11_X04_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.140 fail@end 0.107 coll@2k 0.000 ep_len@2k 1884.042
- training translation floor: track_ratio_mean (last 500 its) 0.482 (gate floor 0.6)
- exposure (ring): reach_edge 0.338 | reach_obst 0.269 | field_frac 0.076 (steps 149.633) | goals_passed 0.376
- terrain-promotion telemetry: track_ratio_mean 0.481 (move rows only 0.481) | heading_err_rms_mean 0.094 | block_steps_mean 52.417 | r_max_mean 1.808
- failure share flat 0.104 | obst 0.176 | campaign-wide 0.107
- slot fractions (logged): stand 0.135 | move 0.785 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.977 | mean ep len 1920.295 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 0.990 drift 0.104 [decisive]
- side22_25 side: n 200 surv 0.900 ratio 0.417 tripod 0.377 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- floor PASS: no floor at this stage

>>> ENTRY X04:s1

### X05:s1 — ref:C_lr__A_s11 s11 (reference) — 2026-10-01T18:42:10

- recipe `a2fc34671b74` on chain `C_lr__A`, base `a2fc34671b74`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-01_17-35-46_seed11_X05_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.07 h
- smoke fail@2k 0.033 fail@end 0.061 coll@2k 0.000 ep_len@2k 1971.198
- training translation floor: track_ratio_mean (last 500 its) 0.602 (gate floor 0.6)
- exposure (ring): reach_edge 0.564 | reach_obst 0.429 | field_frac 0.174 (steps 344.682) | goals_passed 0.602
- terrain-promotion telemetry: track_ratio_mean 0.612 (move rows only 0.612) | heading_err_rms_mean 0.056 | block_steps_mean 85.624 | r_max_mean 2.496
- failure share flat 0.048 | obst 0.050 | campaign-wide 0.064
- slot fractions (logged): stand 0.214 | move 0.785 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.977 | mean ep len 1970.149 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.065 [decisive]
- side22_25 side: n 200 surv 0.950 ratio 0.404 tripod 0.392 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
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

>>> ENTRY X05:s1

### X06:s1 — ref:C_lr__B_s11 s11 (reference) — 2026-10-01T20:00:44

- recipe `5b2550065bea` on chain `C_lr__B`, base `5b2550065bea`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-01_18-52-19_seed11_X06_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.042 fail@end 0.034 coll@2k 0.000 ep_len@2k 1976.218
- training translation floor: track_ratio_mean (last 500 its) 0.642 (gate floor 0.6)
- exposure (ring): reach_edge 0.709 | reach_obst 0.488 | field_frac 0.250 (steps 496.661) | goals_passed 0.769
- terrain-promotion telemetry: track_ratio_mean 0.655 (move rows only 0.655) | heading_err_rms_mean 0.050 | block_steps_mean 106.981 | r_max_mean 3.138
- failure share flat 0.026 | obst 0.044 | campaign-wide 0.013
- slot fractions (logged): stand 0.240 | move 0.757 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.977 | mean ep len 1985.881 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.028 [decisive]
- side22_25 side: n 200 surv 0.995 ratio 0.977 tripod 0.720 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
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

>>> ENTRY X06:s1

### X07:s1 — ref:C_lr_rsi__A_s11 s11 (reference) — 2026-10-01T21:17:25

- recipe `21cc8f978d47` on chain `C_lr_rsi__A`, base `21cc8f978d47`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_RSI_FRAC': '0.2', 'KRABBY_RSI_BANK': '/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_RSI_CAM_STAGE': '0', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-01_20-09-01_seed11_X07_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.076 fail@end 0.079 coll@2k 0.000 ep_len@2k 1956.508
- training translation floor: track_ratio_mean (last 500 its) 0.634 (gate floor 0.6)
- exposure (ring): reach_edge 0.661 | reach_obst 0.495 | field_frac 0.241 (steps 472.237) | goals_passed 0.710
- terrain-promotion telemetry: track_ratio_mean 0.651 (move rows only 0.651) | heading_err_rms_mean 0.053 | block_steps_mean 101.607 | r_max_mean 3.308
- failure share flat 0.103 | obst 0.202 | campaign-wide 0.049
- slot fractions (logged): stand 0.181 | move 0.817 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.182 | terrain_levels 0.977 | mean ep len 1981.919 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.059 [decisive]
- side22_25 side: n 200 surv 0.985 ratio 0.420 tripod 0.484 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_BANK=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_CAM_STAGE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0.2 in params/env.yaml
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

>>> ENTRY X07:s1

### X08:s1 — ref:C_lr_rsi__B_s11 s11 (reference) — 2026-10-01T22:32:18

- recipe `2dd2d8996224` on chain `C_lr_rsi__B`, base `2dd2d8996224`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_RSI_FRAC': '0.2', 'KRABBY_RSI_BANK': '/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_RSI_CAM_STAGE': '0', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-01_21-25-53_seed11_X08_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.07 h
- smoke fail@2k 0.000 fail@end 0.065 coll@2k 0.000 ep_len@2k 2000.000
- training translation floor: track_ratio_mean (last 500 its) 0.662 (gate floor 0.6)
- exposure (ring): reach_edge 0.594 | reach_obst 0.433 | field_frac 0.203 (steps 401.998) | goals_passed 0.655
- terrain-promotion telemetry: track_ratio_mean 0.669 (move rows only 0.669) | heading_err_rms_mean 0.053 | block_steps_mean 108.161 | r_max_mean 3.281
- failure share flat 0.042 | obst 0.057 | campaign-wide 0.069
- slot fractions (logged): stand 0.164 | move 0.752 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.189 | terrain_levels 0.977 | mean ep len 1953.991 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.040 [decisive]
- side22_25 side: n 200 surv 0.940 ratio 1.024 tripod 0.680 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_BANK=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_CAM_STAGE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0.2 in params/env.yaml
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

>>> ENTRY X08:s1

### X09:s1 — ref:C_lr_rsi__C0_s11 s11 (reference) — 2026-10-01T23:48:56

- recipe `c7004a1787d0` on chain `C_lr_rsi__C0`, base `c7004a1787d0`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_RSI_FRAC': '0.2', 'KRABBY_RSI_BANK': '/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_RSI_CAM_STAGE': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-01_22-40-32_seed11_X09_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.264 fail@end 0.164 coll@2k 0.000 ep_len@2k 1760.825
- training translation floor: track_ratio_mean (last 500 its) 0.535 (gate floor 0.6)
- exposure (ring): reach_edge 0.399 | reach_obst 0.309 | field_frac 0.102 (steps 200.474) | goals_passed 0.437
- terrain-promotion telemetry: track_ratio_mean 0.530 (move rows only 0.530) | heading_err_rms_mean 0.075 | block_steps_mean 56.903 | r_max_mean 2.007
- failure share flat 0.149 | obst 0.152 | campaign-wide 0.123
- slot fractions (logged): stand 0.183 | move 0.769 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.182 | terrain_levels 0.977 | mean ep len 1863.551 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.091 [decisive]
- side22_25 side: n 200 surv 0.960 ratio 0.323 tripod 0.478 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_BANK=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_CAM_STAGE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0.2 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- floor PASS: no floor at this stage

>>> ENTRY X09:s1

### X10:s1 — ref:C_lr__A_s12 s12 (reference) — 2026-10-02T01:05:56

- recipe `f290d2362472` on chain `C_lr__A`, base `f290d2362472`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-01_23-57-31_seed12_X10_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.064 fail@end 0.088 coll@2k 0.000 ep_len@2k 1934.500
- training translation floor: track_ratio_mean (last 500 its) 0.645 (gate floor 0.6)
- exposure (ring): reach_edge 0.505 | reach_obst 0.397 | field_frac 0.165 (steps 321.073) | goals_passed 0.540
- terrain-promotion telemetry: track_ratio_mean 0.651 (move rows only 0.651) | heading_err_rms_mean 0.060 | block_steps_mean 87.999 | r_max_mean 2.517
- failure share flat 0.104 | obst 0.121 | campaign-wide 0.080
- slot fractions (logged): stand 0.169 | move 0.778 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.945 | mean ep len 1964.198 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.070 [decisive]
- side22_25 side: n 200 surv 0.890 ratio 0.756 tripod 0.584 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
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

>>> ENTRY X10:s1

### X11:s1 — ref:C_lr__C0_s12 s12 (reference) — 2026-10-02T02:20:55

- recipe `db4674ef897e` on chain `C_lr__C0`, base `db4674ef897e`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_01-14-31_seed12_X11_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.07 h
- smoke fail@2k 0.083 fail@end 0.066 coll@2k 0.000 ep_len@2k 1923.106
- training translation floor: track_ratio_mean (last 500 its) 0.554 (gate floor 0.6)
- exposure (ring): reach_edge 0.498 | reach_obst 0.374 | field_frac 0.129 (steps 257.072) | goals_passed 0.550
- terrain-promotion telemetry: track_ratio_mean 0.559 (move rows only 0.559) | heading_err_rms_mean 0.074 | block_steps_mean 73.731 | r_max_mean 2.097
- failure share flat 0.066 | obst 0.103 | campaign-wide 0.075
- slot fractions (logged): stand 0.204 | move 0.706 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.945 | mean ep len 1889.603 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 0.940 drift 0.104 [decisive]
- side22_25 side: n 200 surv 0.940 ratio 0.407 tripod 0.558 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- floor PASS: no floor at this stage

>>> ENTRY X11:s1

### X12:s1 — ref:C_lr_rsi__C0_s12 s12 (reference) — 2026-10-02T03:37:38

- recipe `8b1d69d1423c` on chain `C_lr_rsi__C0`, base `8b1d69d1423c`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_RSI_FRAC': '0.2', 'KRABBY_RSI_BANK': '/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_RSI_CAM_STAGE': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_02-29-14_seed12_X12_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.211 fail@end 0.110 coll@2k 0.000 ep_len@2k 1778.763
- training translation floor: track_ratio_mean (last 500 its) 0.493 (gate floor 0.6)
- exposure (ring): reach_edge 0.354 | reach_obst 0.272 | field_frac 0.082 (steps 162.732) | goals_passed 0.388
- terrain-promotion telemetry: track_ratio_mean 0.485 (move rows only 0.485) | heading_err_rms_mean 0.083 | block_steps_mean 50.844 | r_max_mean 1.749
- failure share flat 0.105 | obst 0.158 | campaign-wide 0.109
- slot fractions (logged): stand 0.152 | move 0.811 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.192 | terrain_levels 0.945 | mean ep len 1882.744 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.114 [decisive]
- side22_25 side: n 200 surv 0.985 ratio 0.354 tripod 0.530 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_BANK=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_CAM_STAGE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0.2 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- floor PASS: no floor at this stage

>>> ENTRY X12:s1

### X13:s1 — ref:C__A_s12 s12 (reference) — 2026-10-02T04:52:32

- recipe `1d05f1fd7be9` on chain `C__A`, base `1d05f1fd7be9`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025'}`
- status **ok** | checkpoint `2026-10-02_03-46-08_seed12_X13_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.07 h
- smoke fail@2k 0.006 fail@end 0.002 coll@2k 0.000 ep_len@2k 1993.144
- training translation floor: track_ratio_mean (last 500 its) 0.752 (gate floor 0.6)
- exposure (ring): reach_edge 0.972 | reach_obst 0.711 | field_frac 0.578 (steps 1154.322) | goals_passed 1.073
- terrain-promotion telemetry: track_ratio_mean 0.763 (move rows only 0.763) | heading_err_rms_mean 0.045 | block_steps_mean 182.334 | r_max_mean 6.256
- failure share flat 0.002 | obst 0.012 | campaign-wide 0.000
- slot fractions (logged): stand 0.200 | move 0.800 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.945 | mean ep len 2000.000 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 1.000 drift 0.035 [decisive]
- side22_25 side: n 200 surv 1.000 ratio 0.866 tripod 0.718 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: overlay KRABBY_STD_PARAM=soft in params/agent.yaml
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_LOSS consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_RAMP consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND consumed
- overlay PASS: log: [KNOB] KRABBY_STD_PARAM consumed
- overlay PASS: log: [KNOB] KRABBY_INIT_STD consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- overlay PASS: log: Mean action_bound printed (the bound loss ran)
- floor PASS: no floor at this stage

>>> ENTRY X13:s1

### X14:s1 — ref:C_lr__B_s12 s12 (reference) — 2026-10-02T06:00:10

- recipe `bbaab9f9b866` on chain `C_lr__B`, base `bbaab9f9b866`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_04-53-45_seed12_X14_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.07 h
- smoke fail@2k 0.022 fail@end 0.008 coll@2k 0.000 ep_len@2k 1988.891
- training translation floor: track_ratio_mean (last 500 its) 0.703 (gate floor 0.6)
- exposure (ring): reach_edge 0.805 | reach_obst 0.608 | field_frac 0.314 (steps 622.937) | goals_passed 0.889
- terrain-promotion telemetry: track_ratio_mean 0.704 (move rows only 0.704) | heading_err_rms_mean 0.046 | block_steps_mean 143.404 | r_max_mean 3.520
- failure share flat 0.001 | obst 0.050 | campaign-wide 0.017
- slot fractions (logged): stand 0.164 | move 0.810 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.945 | mean ep len 1989.028 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.042 [decisive]
- side22_25 side: n 200 surv 1.000 ratio 1.072 tripod 0.654 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
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

>>> ENTRY X14:s1

### X15:s1 — ref:C__C0_s12 s12 (reference) — 2026-10-02T07:14:50

- recipe `f287704e113c` on chain `C__C0`, base `f287704e113c`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25'}`
- status **ok** | checkpoint `2026-10-02_06-08-26_seed12_X15_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.07 h
- smoke fail@2k 0.051 fail@end 0.040 coll@2k 0.000 ep_len@2k 1939.353
- training translation floor: track_ratio_mean (last 500 its) 0.721 (gate floor 0.6)
- exposure (ring): reach_edge 0.983 | reach_obst 0.751 | field_frac 0.647 (steps 1285.639) | goals_passed 1.125
- terrain-promotion telemetry: track_ratio_mean 0.730 (move rows only 0.730) | heading_err_rms_mean 0.067 | block_steps_mean 169.754 | r_max_mean 6.324
- failure share flat 0.040 | obst 0.040 | campaign-wide 0.041
- slot fractions (logged): stand 0.200 | move 0.795 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.945 | mean ep len 1962.666 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 0.920 drift 0.112 [decisive]
- side22_25 side: n 200 surv 0.955 ratio 0.724 tripod 0.557 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- floor PASS: no floor at this stage

>>> ENTRY X15:s1

### X16:s1 — ref:C_lr_rsi__B_s12 s12 (reference) — 2026-10-02T08:28:26

- recipe `409ddd9daf56` on chain `C_lr_rsi__B`, base `409ddd9daf56`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_RSI_FRAC': '0.2', 'KRABBY_RSI_BANK': '/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_RSI_CAM_STAGE': '0', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_07-20-01_seed12_X16_s1_A15pB_e2/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.17 h
- smoke fail@2k 0.147 fail@end 0.034 coll@2k 0.000 ep_len@2k 1903.763
- training translation floor: track_ratio_mean (last 500 its) 0.634 (gate floor 0.6)
- exposure (ring): reach_edge 0.676 | reach_obst 0.453 | field_frac 0.237 (steps 472.049) | goals_passed 0.736
- terrain-promotion telemetry: track_ratio_mean 0.636 (move rows only 0.636) | heading_err_rms_mean 0.051 | block_steps_mean 112.832 | r_max_mean 3.210
- failure share flat 0.041 | obst 0.043 | campaign-wide 0.023
- slot fractions (logged): stand 0.234 | move 0.766 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.191 | terrain_levels 0.945 | mean ep len 1995.122 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.040 [decisive]
- side22_25 side: n 200 surv 1.000 ratio 0.989 tripod 0.586 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_BANK=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_CAM_STAGE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0.2 in params/env.yaml
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

>>> ENTRY X16:s1

### X17:s1 — ref:C_lr_rsi__A_s12 s12 (reference) — 2026-10-02T09:37:27

- recipe `8ed44cfda860` on chain `C_lr_rsi__A`, base `8ed44cfda860`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_RSI_FRAC': '0.2', 'KRABBY_RSI_BANK': '/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_RSI_CAM_STAGE': '0', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_08-29-02_seed12_X17_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.185 fail@end 0.140 coll@2k 0.000 ep_len@2k 1819.710
- training translation floor: track_ratio_mean (last 500 its) 0.537 (gate floor 0.6)
- exposure (ring): reach_edge 0.496 | reach_obst 0.350 | field_frac 0.137 (steps 269.665) | goals_passed 0.543
- terrain-promotion telemetry: track_ratio_mean 0.535 (move rows only 0.535) | heading_err_rms_mean 0.068 | block_steps_mean 73.138 | r_max_mean 2.040
- failure share flat 0.135 | obst 0.157 | campaign-wide 0.137
- slot fractions (logged): stand 0.217 | move 0.777 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.203 | terrain_levels 0.945 | mean ep len 1899.359 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 0.990 drift 0.054 [decisive]
- side22_25 side: n 200 surv 0.995 ratio 0.439 tripod 0.536 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_BANK=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_CAM_STAGE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0.2 in params/env.yaml
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

>>> ENTRY X17:s1

### X18:s1 — ref:C_lr_rsi__B_s13 s13 (reference) — 2026-10-02T11:01:31

- recipe `85d9e77f6884` on chain `C_lr_rsi__B`, base `85d9e77f6884`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_RSI_FRAC': '0.2', 'KRABBY_RSI_BANK': '/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_RSI_CAM_STAGE': '0', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_09-53-07_seed13_X18_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.063 fail@end 0.016 coll@2k 0.000 ep_len@2k 1925.178
- training translation floor: track_ratio_mean (last 500 its) 0.625 (gate floor 0.6)
- exposure (ring): reach_edge 0.687 | reach_obst 0.426 | field_frac 0.239 (steps 475.820) | goals_passed 0.751
- terrain-promotion telemetry: track_ratio_mean 0.651 (move rows only 0.651) | heading_err_rms_mean 0.057 | block_steps_mean 105.555 | r_max_mean 3.420
- failure share flat 0.009 | obst 0.024 | campaign-wide 0.025
- slot fractions (logged): stand 0.263 | move 0.729 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.232 | terrain_levels 1.031 | mean ep len 1983.066 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.040 [decisive]
- side22_25 side: n 200 surv 1.000 ratio 0.597 tripod 0.605 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_BANK=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_CAM_STAGE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0.2 in params/env.yaml
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

>>> ENTRY X18:s1

### X19:s1 — ref:C_lr_rsi__C0_s13 s13 (reference) — 2026-10-02T12:18:22

- recipe `0ba4bcb4c564` on chain `C_lr_rsi__C0`, base `0ba4bcb4c564`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_RSI_FRAC': '0.2', 'KRABBY_RSI_BANK': '/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_RSI_CAM_STAGE': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_11-09-58_seed13_X19_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.155 fail@end 0.170 coll@2k 0.000 ep_len@2k 1846.847
- training translation floor: track_ratio_mean (last 500 its) 0.579 (gate floor 0.6)
- exposure (ring): reach_edge 0.206 | reach_obst 0.176 | field_frac 0.034 (steps 68.661) | goals_passed 0.223
- terrain-promotion telemetry: track_ratio_mean 0.587 (move rows only 0.587) | heading_err_rms_mean 0.072 | block_steps_mean 30.248 | r_max_mean 1.599
- failure share flat 0.142 | obst 0.162 | campaign-wide 0.177
- slot fractions (logged): stand 0.184 | move 0.746 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.210 | terrain_levels 1.031 | mean ep len 1887.536 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 0.930 drift 0.101 [decisive]
- side22_25 side: n 200 surv 0.975 ratio 0.323 tripod 0.382 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_BANK=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_CAM_STAGE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0.2 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- floor PASS: no floor at this stage

>>> ENTRY X19:s1

### X20:s1 — ref:C_lr_rsi__A_s13 s13 (reference) — 2026-10-02T13:35:10

- recipe `82df41ed8d48` on chain `C_lr_rsi__A`, base `82df41ed8d48`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_RSI_FRAC': '0.2', 'KRABBY_RSI_BANK': '/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_RSI_CAM_STAGE': '0', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_12-26-46_seed13_X20_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.047 fail@end 0.073 coll@2k 0.000 ep_len@2k 1965.558
- training translation floor: track_ratio_mean (last 500 its) 0.667 (gate floor 0.6)
- exposure (ring): reach_edge 0.746 | reach_obst 0.482 | field_frac 0.273 (steps 544.411) | goals_passed 0.830
- terrain-promotion telemetry: track_ratio_mean 0.665 (move rows only 0.665) | heading_err_rms_mean 0.057 | block_steps_mean 108.180 | r_max_mean 2.968
- failure share flat 0.048 | obst 0.053 | campaign-wide 0.063
- slot fractions (logged): stand 0.205 | move 0.779 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.197 | terrain_levels 1.031 | mean ep len 1964.008 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.027 [decisive]
- side22_25 side: n 200 surv 0.980 ratio 0.450 tripod 0.599 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_BANK=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/rsi_bank_side_lr.npz in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_CAM_STAGE=0 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0.2 in params/env.yaml
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

>>> ENTRY X20:s1

### X21:s1 — ref:C_lr__B_s13 s13 (reference) — 2026-10-02T14:52:05

- recipe `0b03fd524290` on chain `C_lr__B`, base `0b03fd524290`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_13-43-41_seed13_X21_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.010 fail@end 0.042 coll@2k 0.000 ep_len@2k 1994.088
- training translation floor: track_ratio_mean (last 500 its) 0.622 (gate floor 0.6)
- exposure (ring): reach_edge 0.656 | reach_obst 0.450 | field_frac 0.233 (steps 465.522) | goals_passed 0.712
- terrain-promotion telemetry: track_ratio_mean 0.651 (move rows only 0.651) | heading_err_rms_mean 0.052 | block_steps_mean 105.272 | r_max_mean 3.082
- failure share flat 0.036 | obst 0.019 | campaign-wide 0.035
- slot fractions (logged): stand 0.222 | move 0.778 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.031 | mean ep len 1973.843 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.031 [decisive]
- side22_25 side: n 200 surv 0.995 ratio 0.953 tripod 0.733 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
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

>>> ENTRY X21:s1

### X22:s1 — ref:C_lr__A_s13 s13 (reference) — 2026-10-02T16:08:44

- recipe `beab6d91b18c` on chain `C_lr__A`, base `beab6d91b18c`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_15-00-20_seed13_X22_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.054 fail@end 0.003 coll@2k 0.000 ep_len@2k 1945.239
- training translation floor: track_ratio_mean (last 500 its) 0.557 (gate floor 0.6)
- exposure (ring): reach_edge 0.436 | reach_obst 0.288 | field_frac 0.115 (steps 229.129) | goals_passed 0.478
- terrain-promotion telemetry: track_ratio_mean 0.565 (move rows only 0.565) | heading_err_rms_mean 0.065 | block_steps_mean 63.833 | r_max_mean 2.064
- failure share flat 0.003 | obst 0.023 | campaign-wide 0.000
- slot fractions (logged): stand 0.206 | move 0.794 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.031 | mean ep len 2000.000 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.041 [decisive]
- side22_25 side: n 200 surv 1.000 ratio 0.339 tripod 0.602 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
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

>>> ENTRY X22:s1

### X23:s1 — ref:C__A_s13 s13 (reference) — 2026-10-02T17:23:40

- recipe `abbed03b2a3c` on chain `C__A`, base `abbed03b2a3c`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025'}`
- status **ok** | checkpoint `2026-10-02_16-17-16_seed13_X23_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.07 h
- smoke fail@2k 0.015 fail@end 0.007 coll@2k 0.000 ep_len@2k 1991.876
- training translation floor: track_ratio_mean (last 500 its) 0.828 (gate floor 0.6)
- exposure (ring): reach_edge 0.992 | reach_obst 0.616 | field_frac 0.714 (steps 1425.292) | goals_passed 1.092
- terrain-promotion telemetry: track_ratio_mean 0.845 (move rows only 0.845) | heading_err_rms_mean 0.050 | block_steps_mean 142.657 | r_max_mean 7.705
- failure share flat 0.012 | obst 0.014 | campaign-wide 0.008
- slot fractions (logged): stand 0.191 | move 0.802 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.031 | mean ep len 1986.089 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 1.000 drift 0.040 [decisive]
- side22_25 side: n 200 surv 0.975 ratio 1.017 tripod 0.714 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: overlay KRABBY_STD_PARAM=soft in params/agent.yaml
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_LOSS consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND_RAMP consumed
- overlay PASS: log: [KNOB] KRABBY_ACTION_BOUND consumed
- overlay PASS: log: [KNOB] KRABBY_STD_PARAM consumed
- overlay PASS: log: [KNOB] KRABBY_INIT_STD consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- overlay PASS: log: Mean action_bound printed (the bound loss ran)
- floor PASS: no floor at this stage

>>> ENTRY X23:s1

### X24:s1 — ref:C__C0_s13 s13 (reference) — 2026-10-02T18:31:18

- recipe `7c72f60afdee` on chain `C__C0`, base `7c72f60afdee`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25'}`
- status **ok** | checkpoint `2026-10-02_17-24-54_seed13_X24_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.07 h
- smoke fail@2k 0.028 fail@end 0.006 coll@2k 0.000 ep_len@2k 1976.173
- training translation floor: track_ratio_mean (last 500 its) 0.636 (gate floor 0.6)
- exposure (ring): reach_edge 0.973 | reach_obst 0.657 | field_frac 0.562 (steps 1122.105) | goals_passed 1.101
- terrain-promotion telemetry: track_ratio_mean 0.640 (move rows only 0.640) | heading_err_rms_mean 0.062 | block_steps_mean 173.640 | r_max_mean 5.059
- failure share flat 0.004 | obst 0.011 | campaign-wide 0.008
- slot fractions (logged): stand 0.207 | move 0.785 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.031 | mean ep len 1987.329 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 1.000 drift 0.098 [decisive]
- side22_25 side: n 200 surv 0.940 ratio 0.603 tripod 0.542 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- floor PASS: no floor at this stage

>>> ENTRY X24:s1

### X25:s1 — ref:C_lr__C0_s13 s13 (reference) — 2026-10-02T19:40:54

- recipe `554448894b49` on chain `C_lr__C0`, base `554448894b49`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_18-32-29_seed13_X25_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.086 fail@end 0.048 coll@2k 0.000 ep_len@2k 1941.423
- training translation floor: track_ratio_mean (last 500 its) 0.486 (gate floor 0.6)
- exposure (ring): reach_edge 0.290 | reach_obst 0.229 | field_frac 0.073 (steps 146.157) | goals_passed 0.328
- terrain-promotion telemetry: track_ratio_mean 0.481 (move rows only 0.481) | heading_err_rms_mean 0.089 | block_steps_mean 41.973 | r_max_mean 1.737
- failure share flat 0.074 | obst 0.071 | campaign-wide 0.038
- slot fractions (logged): stand 0.211 | move 0.757 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.031 | mean ep len 1965.256 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.070 [decisive]
- side22_25 side: n 200 surv 0.995 ratio 0.298 tripod 0.450 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- floor PASS: no floor at this stage

>>> ENTRY X25:s1

### X26:s1 — ref:C_lr__B_s14 s14 (reference) — 2026-10-02T20:56:03

- recipe `e28504223adc` on chain `C_lr__B`, base `e28504223adc`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_19-49-39_seed14_X26_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.07 h
- smoke fail@2k 0.016 fail@end 0.022 coll@2k 0.000 ep_len@2k 1987.540
- training translation floor: track_ratio_mean (last 500 its) 0.667 (gate floor 0.6)
- exposure (ring): reach_edge 0.648 | reach_obst 0.505 | field_frac 0.244 (steps 486.312) | goals_passed 0.686
- terrain-promotion telemetry: track_ratio_mean 0.673 (move rows only 0.673) | heading_err_rms_mean 0.050 | block_steps_mean 125.063 | r_max_mean 3.297
- failure share flat 0.038 | obst 0.044 | campaign-wide 0.017
- slot fractions (logged): stand 0.225 | move 0.745 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.047 | mean ep len 1984.958 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.037 [decisive]
- side22_25 side: n 200 surv 1.000 ratio 0.965 tripod 0.611 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
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

>>> ENTRY X26:s1

### X27:s1 — ref:C_lr__C0_s14 s14 (reference) — 2026-10-02T22:12:50

- recipe `194f6104cc80` on chain `C_lr__C0`, base `194f6104cc80`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_21-04-25_seed14_X27_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.113 fail@end 0.068 coll@2k 0.000 ep_len@2k 1911.065
- training translation floor: track_ratio_mean (last 500 its) 0.482 (gate floor 0.6)
- exposure (ring): reach_edge 0.277 | reach_obst 0.256 | field_frac 0.071 (steps 141.106) | goals_passed 0.299
- terrain-promotion telemetry: track_ratio_mean 0.491 (move rows only 0.491) | heading_err_rms_mean 0.081 | block_steps_mean 46.460 | r_max_mean 1.760
- failure share flat 0.059 | obst 0.106 | campaign-wide 0.062
- slot fractions (logged): stand 0.253 | move 0.734 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.047 | mean ep len 1941.964 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.117 [decisive]
- side22_25 side: n 200 surv 0.940 ratio 0.395 tripod 0.403 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- floor PASS: no floor at this stage

>>> ENTRY X27:s1

### X28:s1 — ref:C_lr__B_s15 s15 (reference) — 2026-10-02T23:29:54

- recipe `570684b5a983` on chain `C_lr__B`, base `570684b5a983`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_ACTION_BOUND_LOSS': '0.001', 'KRABBY_ACTION_BOUND_RAMP': '0.0001:0.001:500', 'KRABBY_ACTION_BOUND': '1.1', 'KRABBY_STD_PARAM': 'soft', 'KRABBY_INIT_STD': '0.8', 'KRABBY_ENTROPY_COEF': '0.0025', 'KRABBY_ACTION_RATE_W': '0', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_22-21-30_seed15_X28_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.093 fail@end 0.080 coll@2k 0.000 ep_len@2k 1945.408
- training translation floor: track_ratio_mean (last 500 its) 0.639 (gate floor 0.6)
- exposure (ring): reach_edge 0.710 | reach_obst 0.437 | field_frac 0.261 (steps 516.888) | goals_passed 0.773
- terrain-promotion telemetry: track_ratio_mean 0.657 (move rows only 0.657) | heading_err_rms_mean 0.050 | block_steps_mean 114.910 | r_max_mean 3.278
- failure share flat 0.088 | obst 0.089 | campaign-wide 0.060
- slot fractions (logged): stand 0.211 | move 0.781 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.035 | mean ep len 1969.204 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.047 [decisive]
- side22_25 side: n 200 surv 1.000 ratio 0.706 tripod 0.637 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_ACTION_BOUND=1.1 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_LOSS=0.001 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_BOUND_RAMP=0.0001:0.001:500 in params/agent.yaml
- overlay PASS: overlay KRABBY_ACTION_RATE_W=0 in params/env.yaml
- overlay PASS: overlay KRABBY_ENTROPY_COEF=0.0025 in params/agent.yaml
- overlay PASS: overlay KRABBY_INIT_STD=0.8 in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
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

>>> ENTRY X28:s1

### X29:s1 — ref:C_lr__C0_s15 s15 (reference) — 2026-10-03T00:46:45

- recipe `1f44748bcaea` on chain `C_lr__C0`, base `1f44748bcaea`, code_sha `f7166d558c5d`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25', 'KRABBY_PPO_HAND_DIAG': '1'}`
- status **ok** | checkpoint `2026-10-02_23-38-20_seed15_X29_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.099 fail@end 0.130 coll@2k 0.000 ep_len@2k 1925.420
- training translation floor: track_ratio_mean (last 500 its) 0.522 (gate floor 0.6)
- exposure (ring): reach_edge 0.413 | reach_obst 0.274 | field_frac 0.105 (steps 207.535) | goals_passed 0.467
- terrain-promotion telemetry: track_ratio_mean 0.529 (move rows only 0.529) | heading_err_rms_mean 0.075 | block_steps_mean 61.961 | r_max_mean 1.921
- failure share flat 0.118 | obst 0.144 | campaign-wide 0.129
- slot fractions (logged): stand 0.154 | move 0.816 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.035 | mean ep len 1915.299 | prints 3000
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
- check FAIL: omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.083 [decisive]
- side22_25 side: n 200 surv 0.980 ratio 0.306 tripod 0.471 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_PPO_HAND_DIAG=1 in params/agent.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- overlay PASS: log: [KNOB] KRABBY_PPO_HAND_DIAG consumed
- overlay PASS: log: no PPO-construction field dropped by the runner
- floor PASS: no floor at this stage

>>> ENTRY X29:s1


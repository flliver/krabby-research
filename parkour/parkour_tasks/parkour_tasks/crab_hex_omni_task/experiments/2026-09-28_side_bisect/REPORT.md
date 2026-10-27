# Candidate ladder `2026-09-28_side_bisect`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 3, confirm seed 2, 3000 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `07a0037ed5d9b6bcccfd7b7c21a722cc74647593e58e6c34d98eb05174fb24a3` · `candidates.yaml` sha256 `a9f3a62b47e9de3255aa9696b2b4d6c8649c8c1d9ad0031fc4e23acc2d737118`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | O | baseline | O | the incumbent recipe at equal compute | — |

### X01:s1 — ref:O s3 (reference) — 2026-09-29T01:52:39

- recipe `07e591f6b17f` on chain `O`, base `07e591f6b17f`, code_sha `07a0037ed5d9`
- overlay `{}`
- status **ok** | checkpoint `2026-09-29_00-40-18_seed3_X01_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.17 h
- smoke fail@2k 0.021 fail@end 0.019 coll@2k 0.000 ep_len@2k 1969.320
- training translation floor: track_ratio_mean (last 500 its) 0.562 (gate floor 0.6)
- exposure (ring): reach_edge 0.944 | reach_obst 0.588 | field_frac 0.421 (steps 840.840) | goals_passed 1.044
- terrain-promotion telemetry: track_ratio_mean 0.570 (move rows only 0.570) | heading_err_rms_mean 0.064 | block_steps_mean 155.474 | r_max_mean 3.528
- failure share flat 0.016 | obst 0.024 | campaign-wide 0.019
- slot fractions (logged): stand 0.126 | move 0.872 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.204 | terrain_levels 0.973 | mean ep len 1975.121 | prints 3000
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
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 1.000 drift 0.081 [decisive]
- side22_25 side: n 200 surv 0.960 ratio 0.387 tripod 0.513 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- floor PASS: no floor at this stage

>>> ENTRY X01:s1

### X02:s1 — ref:C s3 (reference) — 2026-09-29T03:03:17

- recipe `0cf93df3875c` on chain `C`, base `0cf93df3875c`, code_sha `07a0037ed5d9`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25'}`
- status **ok** | checkpoint `2026-09-29_01-52-56_seed3_X02_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.13 h
- smoke fail@2k 0.103 fail@end 0.073 coll@2k 0.000 ep_len@2k 1914.044
- training translation floor: track_ratio_mean (last 500 its) 0.794 (gate floor 0.6)
- exposure (ring): reach_edge 0.971 | reach_obst 0.620 | field_frac 0.659 (steps 1288.495) | goals_passed 1.108
- terrain-promotion telemetry: track_ratio_mean 0.797 (move rows only 0.797) | heading_err_rms_mean 0.073 | block_steps_mean 133.549 | r_max_mean 6.964
- failure share flat 0.082 | obst 0.128 | campaign-wide 0.073
- slot fractions (logged): stand 0.213 | move 0.753 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.973 | mean ep len 1940.702 | prints 3000
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
- side22_25 stand: n 100 surv 0.790 drift 0.146 [decisive]
- side22_25 side: n 200 surv 0.885 ratio 0.848 tripod 0.564 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- floor PASS: no floor at this stage

>>> ENTRY X02:s1

### X03:s1 — ref:R s3 (reference) — 2026-09-29T09:38:51

- recipe `fdab6c96a82c` on chain `R`, base `fdab6c96a82c`, code_sha `07a0037ed5d9`
- overlay `{'KRABBY_OMNI_CLOCK_COMBINE': 'sum', 'KRABBY_TRACK_SIGMA2': '0.1', 'KRABBY_TRACK_L1_W': '-1.0', 'KRABBY_YAW_PROGRESS_W': '0'}`
- status **ok** | checkpoint `2026-09-29_08-28-30_seed3_X03_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.13 h
- smoke fail@2k 0.114 fail@end 0.102 coll@2k 0.000 ep_len@2k 1880.776
- training translation floor: track_ratio_mean (last 500 its) 0.459 (gate floor 0.6)
- exposure (ring): reach_edge 0.752 | reach_obst 0.386 | field_frac 0.230 (steps 456.721) | goals_passed 0.819
- terrain-promotion telemetry: track_ratio_mean 0.462 (move rows only 0.462) | heading_err_rms_mean 0.052 | block_steps_mean 93.979 | r_max_mean 2.607
- failure share flat 0.096 | obst 0.145 | campaign-wide 0.103
- slot fractions (logged): stand 0.178 | move 0.781 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.199 | terrain_levels 0.973 | mean ep len 1848.731 | prints 3000
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
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 0.980 drift 0.070 [decisive]
- side22_25 side: n 200 surv 0.945 ratio 0.362 tripod 0.214 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CLOCK_COMBINE=sum in params/env.yaml
- overlay PASS: overlay KRABBY_TRACK_L1_W=-1.0 in params/env.yaml
- overlay PASS: overlay KRABBY_TRACK_SIGMA2=0.1 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_PROGRESS_W=0 in params/env.yaml
- floor PASS: no floor at this stage

>>> ENTRY X03:s1

### X04:s1 — ref:C_mo s3 (reference) — 2026-09-29T10:51:43

- recipe `5bda941458b0` on chain `C_mo`, base `5bda941458b0`, code_sha `07a0037ed5d9`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar'}`
- status **ok** | checkpoint `2026-09-29_09-39-22_seed3_X04_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.17 h
- smoke fail@2k 0.010 fail@end 0.018 coll@2k 0.000 ep_len@2k 1985.097
- training translation floor: track_ratio_mean (last 500 its) 0.771 (gate floor 0.6)
- exposure (ring): reach_edge 0.985 | reach_obst 0.645 | field_frac 0.594 (steps 1185.649) | goals_passed 1.105
- terrain-promotion telemetry: track_ratio_mean 0.774 (move rows only 0.774) | heading_err_rms_mean 0.064 | block_steps_mean 155.505 | r_max_mean 5.879
- failure share flat 0.011 | obst 0.018 | campaign-wide 0.012
- slot fractions (logged): stand 0.246 | move 0.754 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.203 | terrain_levels 0.973 | mean ep len 1994.449 | prints 3000
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
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 1.000 drift 0.097 [decisive]
- side22_25 side: n 200 surv 0.995 ratio 0.621 tripod 0.525 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- floor PASS: no floor at this stage

>>> ENTRY X04:s1

### X05:s1 — ref:C_band s3 (reference) — 2026-09-29T12:07:14

- recipe `5a29d7966a75` on chain `C_band`, base `5a29d7966a75`, code_sha `07a0037ed5d9`
- overlay `{'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25'}`
- status **ok** | checkpoint `2026-09-29_10-52-23_seed3_X05_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.20 h
- smoke fail@2k 0.203 fail@end 0.175 coll@2k 0.000 ep_len@2k 1834.075
- training translation floor: track_ratio_mean (last 500 its) 0.577 (gate floor 0.6)
- exposure (ring): reach_edge 0.876 | reach_obst 0.487 | field_frac 0.384 (steps 755.645) | goals_passed 0.987
- terrain-promotion telemetry: track_ratio_mean 0.578 (move rows only 0.578) | heading_err_rms_mean 0.070 | block_steps_mean 136.587 | r_max_mean 3.341
- failure share flat 0.203 | obst 0.225 | campaign-wide 0.154
- slot fractions (logged): stand 0.198 | move 0.732 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.200 | terrain_levels 0.973 | mean ep len 1831.142 | prints 3000
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
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 0.960 drift 0.092 [decisive]
- side22_25 side: n 200 surv 0.880 ratio 0.461 tripod 0.471 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- floor PASS: no floor at this stage

>>> ENTRY X05:s1

### X06:s1 — ref:C_rsi s3 (reference) — 2026-09-29T13:22:28

- recipe `f1d1152db439` on chain `C_rsi`, base `f1d1152db439`, code_sha `07a0037ed5d9`
- overlay `{'KRABBY_RSI_FRAC': '0'}`
- status **ok** | checkpoint `2026-09-29_12-09-37_seed3_X06_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.17 h
- smoke fail@2k 0.026 fail@end 0.003 coll@2k 0.000 ep_len@2k 1975.082
- training translation floor: track_ratio_mean (last 500 its) 0.532 (gate floor 0.6)
- exposure (ring): reach_edge 0.867 | reach_obst 0.476 | field_frac 0.294 (steps 587.239) | goals_passed 0.976
- terrain-promotion telemetry: track_ratio_mean 0.537 (move rows only 0.537) | heading_err_rms_mean 0.065 | block_steps_mean 125.967 | r_max_mean 2.857
- failure share flat 0.004 | obst 0.023 | campaign-wide 0.004
- slot fractions (logged): stand 0.248 | move 0.752 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 0.973 | mean ep len 2000.000 | prints 3000
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
- side22_25 stand: n 100 surv 1.000 drift 0.085 [decisive]
- side22_25 side: n 200 surv 0.965 ratio 0.340 tripod 0.393 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- floor PASS: no floor at this stage

>>> ENTRY X06:s1

### X07:s1 — ref:C_s2 s2 (reference) — 2026-09-29T14:35:29

- recipe `c82d3b2b0da8` on chain `C`, base `c82d3b2b0da8`, code_sha `07a0037ed5d9`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar', 'KRABBY_RSI_FRAC': '0', 'KRABBY_OMNI_SPEED_ELLIPSE': '0.35:0.35:0.25'}`
- status **ok** | checkpoint `2026-09-29_13-23-08_seed2_X07_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.17 h
- smoke fail@2k 0.038 fail@end 0.030 coll@2k 0.000 ep_len@2k 1966.255
- training translation floor: track_ratio_mean (last 500 its) 0.734 (gate floor 0.6)
- exposure (ring): reach_edge 0.984 | reach_obst 0.799 | field_frac 0.660 (steps 1313.068) | goals_passed 1.118
- terrain-promotion telemetry: track_ratio_mean 0.749 (move rows only 0.749) | heading_err_rms_mean 0.065 | block_steps_mean 188.733 | r_max_mean 6.425
- failure share flat 0.032 | obst 0.035 | campaign-wide 0.030
- slot fractions (logged): stand 0.192 | move 0.797 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.000 | terrain_levels 1.027 | mean ep len 1970.149 | prints 3000
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
- side22_25 stand: n 100 surv 0.880 drift 0.137 [decisive]
- side22_25 side: n 200 surv 0.880 ratio 0.792 tripod 0.578 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SPEED_ELLIPSE=0.35:0.35:0.25 in params/env.yaml
- overlay PASS: overlay KRABBY_RSI_FRAC=0 in params/env.yaml
- floor PASS: no floor at this stage

>>> ENTRY X07:s1

### X08:s1 — ref:C_mo_s2 s2 (reference) — 2026-09-29T15:48:30

- recipe `8fd9f081001c` on chain `C_mo`, base `8fd9f081001c`, code_sha `07a0037ed5d9`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar'}`
- status **ok** | checkpoint `2026-09-29_14-36-09_seed2_X08_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.17 h
- smoke fail@2k 0.124 fail@end 0.153 coll@2k 0.000 ep_len@2k 1862.027
- training translation floor: track_ratio_mean (last 500 its) 0.848 (gate floor 0.6)
- exposure (ring): reach_edge 0.957 | reach_obst 0.727 | field_frac 0.612 (steps 1197.162) | goals_passed 1.099
- terrain-promotion telemetry: track_ratio_mean 0.869 (move rows only 0.869) | heading_err_rms_mean 0.070 | block_steps_mean 181.037 | r_max_mean 6.301
- failure share flat 0.105 | obst 0.123 | campaign-wide 0.175
- slot fractions (logged): stand 0.196 | move 0.791 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.211 | terrain_levels 1.027 | mean ep len 1899.350 | prints 3000
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
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 0.960 drift 0.078 [decisive]
- side22_25 side: n 200 surv 0.955 ratio 0.852 tripod 0.585 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- floor PASS: no floor at this stage

>>> ENTRY X08:s1

### X09:s1 — ref:C_mo_s1 s1 (reference) — 2026-09-29T19:18:05

- recipe `95742856ac77` on chain `C_mo`, base `95742856ac77`, code_sha `07a0037ed5d9`
- overlay `{'KRABBY_OMNI_MIRROR_IMAGES': 'fa', 'KRABBY_OMNI_CMD_OBS': 'scalar'}`
- status **ok** | checkpoint `2026-09-29_18-09-45_seed1_X09_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.10 h
- smoke fail@2k 0.029 fail@end 0.005 coll@2k 0.000 ep_len@2k 1973.891
- training translation floor: track_ratio_mean (last 500 its) 0.858 (gate floor 0.6)
- exposure (ring): reach_edge 0.994 | reach_obst 0.676 | field_frac 0.661 (steps 1319.112) | goals_passed 1.124
- terrain-promotion telemetry: track_ratio_mean 0.853 (move rows only 0.853) | heading_err_rms_mean 0.059 | block_steps_mean 134.637 | r_max_mean 6.337
- failure share flat 0.013 | obst 0.014 | campaign-wide 0.011
- slot fractions (logged): stand 0.224 | move 0.776 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.196 | terrain_levels 1.020 | mean ep len 1995.880 | prints 3000
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
- side22_25 trained categories: decisive stand; watch side (one hand)
- side22_25 stand: n 100 surv 0.980 drift 0.109 [decisive]
- side22_25 side: n 200 surv 0.965 ratio 0.755 tripod 0.626 [WATCH: one hand]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_OBS=scalar in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- floor PASS: no floor at this stage

>>> ENTRY X09:s1

### X10:s1 — ref:LRfa s3 (reference) — 2026-09-29T20:29:13

- recipe `a1b143765879` on chain `LRfa`, base `a1b143765879`, code_sha `07a0037ed5d9`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0', 'KRABBY_OMNI_MIRROR_IMAGES': 'fa'}`
- status **ok** | checkpoint `2026-09-29_19-18-51_seed3_X10_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.13 h
- smoke fail@2k 0.073 fail@end 0.113 coll@2k 0.000 ep_len@2k 1951.538
- training translation floor: track_ratio_mean (last 500 its) 0.524 (gate floor 0.6)
- exposure (ring): reach_edge 0.215 | reach_obst 0.184 | field_frac 0.044 (steps 86.224) | goals_passed 0.231
- terrain-promotion telemetry: track_ratio_mean 0.524 (move rows only 0.524) | heading_err_rms_mean 0.076 | block_steps_mean 34.587 | r_max_mean 1.559
- failure share flat 0.079 | obst 0.105 | campaign-wide 0.114
- slot fractions (logged): stand 0.203 | move 0.753 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.213 | terrain_levels 0.973 | mean ep len 1916.957 | prints 3000
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
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 0.970 drift 0.096 [decisive]
- side22_25 side: n 200 surv 0.990 ratio 0.329 tripod 0.447 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_MIRROR_IMAGES=fa in params/agent.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- floor PASS: no floor at this stage

>>> ENTRY X10:s1

### X11:s1 — ref:LR4 s3 (reference) — 2026-09-29T21:44:03

- recipe `72dd0ba585f8` on chain `LR4`, base `72dd0ba585f8`, code_sha `07a0037ed5d9`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.2,side_l:0.4,side_r:0.4', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_SLOTS': '0.2:0.8:0:0'}`
- status **ok** | checkpoint `2026-09-29_20-33-42_seed3_X11_s1_A15pB/model_2999.pt` | resume `scratch (--from-scratch)` | 3000 its | wall 1.13 h
- smoke fail@2k 0.160 fail@end 0.107 coll@2k 0.000 ep_len@2k 1808.853
- training translation floor: track_ratio_mean (last 500 its) 0.592 (gate floor 0.6)
- exposure (ring): reach_edge 0.408 | reach_obst 0.206 | field_frac 0.094 (steps 184.673) | goals_passed 0.439
- terrain-promotion telemetry: track_ratio_mean 0.598 (move rows only 0.598) | heading_err_rms_mean 0.069 | block_steps_mean 45.104 | r_max_mean 1.975
- failure share flat 0.144 | obst 0.167 | campaign-wide 0.106
- slot fractions (logged): stand 0.123 | move 0.803 | turn 0.000 | move_turn 0.000 | rsi_frac_actual 0.206 | terrain_levels 0.973 | mean ep len 1907.641 | prints 3000
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
- side22_25 trained categories: decisive stand, side
- side22_25 stand: n 100 surv 1.000 drift 0.091 [decisive]
- side22_25 side: n 200 surv 0.870 ratio 0.386 tripod 0.390 [decisive]
- WATCH gate_1c: n/a (no slow row on this stage)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.2:0.8:0:0 in params/env.yaml
- floor PASS: no floor at this stage

>>> ENTRY X11:s1


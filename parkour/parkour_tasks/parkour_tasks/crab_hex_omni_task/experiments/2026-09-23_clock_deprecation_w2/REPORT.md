# Candidate ladder `2026-09-23_clock_deprecation_w2`

Plant **A15+B** (`assets/crab.usda (main)`), screen seed 3, confirm seed 2, 4000 iterations per screen.

Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is `ledger.json`, the user's answers go through `ledger_cli.py`).

`code_sha` `8ce6da6156b31b7d85cb499394b517bc4bea6c83e91c4db56562f1a68d373a73` · `candidates.yaml` sha256 `03d6bf57630959bedca8cbb2661dbf5ae59b0dd51b5b5f62db5fbe495083db9e`

| rank | id | kind | chain | change | conditional |
|---|---|---|---|---|---|
| 0 | X0x | baseline | c6x | the incumbent recipe at equal compute | — |
| 1 | W | clock | effective | `{"c6x":{"KRABBY_APEX_W":"0.001","KRABBY_CLOCK_W":"0.001","KRABBY_OMNI_CLOCK_HIDE":"1"}}` | `—` |

### X01:c6x — X0x s3 (baseline) — 2026-09-25T13:09:14

- recipe `b48e7efd9851` on chain `c6x`, base `b48e7efd9851`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-25_11-25-53_seed3_X01_c6x_A15pB/model_11995.pt` | resume `model_7996.pt` | 4000 its | wall 1.60 h
- smoke fail@2k 0.476 fail@end 0.429 coll@2k 0.000 ep_len@2k 1532.127
- training translation floor: track_ratio_mean (last 500 its) 0.628 (gate floor 0.6)
- exposure (ring): reach_edge 0.531 | reach_obst 0.417 | field_frac 0.181 (steps 339.807) | goals_passed 0.697
- terrain-promotion telemetry: track_ratio_mean 0.633 (move rows only 0.636) | heading_err_rms_mean 0.167 | block_steps_mean 88.317 | r_max_mean 2.312
- failure share flat 0.373 | obst 0.388 | campaign-wide 0.435
- slot fractions (logged): stand 0.148 | move 0.406 | turn 0.066 | move_turn 0.165 | rsi_frac_actual 0.196 | terrain_levels 0.973 | mean ep len 1586.799 | prints 4000
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
- suite stand: n 100 surv 0.990 drift 0.079 [decisive]
- suite fwd: n 100 surv 0.840 ratio 0.719 tripod 0.581 [decisive]
- suite side: n 200 surv 0.965 ratio 0.418 tripod 0.494 (l: surv 0.970 ratio 0.428 tripod 0.503; r: surv 0.960 ratio 0.408 tripod 0.493) [decisive]
- suite back: n 100 surv 0.870 ratio 0.712 tripod 0.616 [decisive]
- suite diag: n 100 surv 0.740 ratio 0.523 tripod 0.375 [decisive]
- suite turn: n 200 surv 0.970 closed 0.687 progress 0.089 (l: surv 0.950 closed 0.679 progress 0.089; r: surv 0.990 closed 0.703 progress 0.092) [decisive]
- suite move_turn: n 200 surv 0.865 closed 0.298 progress 0.042 ratio 0.521 (l: surv 0.870 closed 0.317 progress 0.045 ratio 0.528; r: surv 0.860 closed 0.288 progress 0.039 ratio 0.512) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.375 (need >= 0.4; by hold: back 0.616, diag 0.375, fwd 0.581, lat_l 0.503, lat_r 0.493, move_turn 0.545, move_turn_r 0.537, settle 0.518, stand 0.614, turn_l90 0.634, turn_r90 0.634), completion 0.904 (need >= 0.7), yaw-progress rate turn_l90 0.089, turn_r90 0.092 rad/s (net turn_l90 0.088, turn_r90 0.091) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.428, diag 0.523, back 0.712, move_turn 0.528 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.628 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.990 (bound 1.013, n_env 100, min >= 0.79); fwd 0.840 (bound 0.925, n_env 100, min >= 0.65); side 0.965 (bound 1.008, n_env 100, min >= 0.77); back 0.870 (bound 0.948, n_env 100, min >= 0.68); diag 0.740 (bound 0.842, n_env 100, min >= 0.52); turn 0.970 (bound 1.010, n_env 100, min >= 0.785); move_turn 0.865 (bound 0.944, n_env 100, min >= 0.545)

>>> ENTRY X01:c6x

### experiment X01 — X0x s3 — ok — 2026-09-25T13:09:15

- recipe `b48e7efd9851` base `b48e7efd9851` chain `c6x` seed 3
- stages ['X01:c6x']
- screen: completion 0.904, tripod_dir 0.375, track tail 0.628
- replay: none

>>> ENTRY X01

### X02:c6x — X0x s2 (baseline) — 2026-09-25T14:53:17

- recipe `b48e7efd9851` on chain `c6x`, base `b48e7efd9851`, code_sha `8ce6da6156b3`
- overlay `{'KRABBY_OMNI_CMD_POOL': 'stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07', 'KRABBY_OMNI_SLOTS': '0.15:0.55:0.1:0.2', 'KRABBY_OMNI_DIR_BAND': '', 'KRABBY_OMNI_RSI_SOURCE': 'all', 'KRABBY_OMNI_FACING': '0.2:0.6', 'KRABBY_YAW_W': '0.2'}`
- status **ok** | checkpoint `2026-09-25_13-09-25_seed2_X02_c6x_A15pB/model_11995.pt` | resume `model_7996.pt` | 4000 its | wall 1.60 h
- smoke fail@2k 0.420 fail@end 0.473 coll@2k 0.000 ep_len@2k 1600.442
- training translation floor: track_ratio_mean (last 500 its) 0.639 (gate floor 0.6)
- exposure (ring): reach_edge 0.499 | reach_obst 0.416 | field_frac 0.149 (steps 271.287) | goals_passed 0.645
- terrain-promotion telemetry: track_ratio_mean 0.645 (move rows only 0.649) | heading_err_rms_mean 0.165 | block_steps_mean 88.461 | r_max_mean 2.181
- failure share flat 0.465 | obst 0.475 | campaign-wide 0.433
- slot fractions (logged): stand 0.156 | move 0.425 | turn 0.073 | move_turn 0.112 | rsi_frac_actual 0.224 | terrain_levels 1.027 | mean ep len 1460.127 | prints 4000
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
- suite stand: n 100 surv 0.980 drift 0.085 [decisive]
- suite fwd: n 100 surv 0.800 ratio 0.768 tripod 0.568 [decisive]
- suite side: n 200 surv 0.950 ratio 0.415 tripod 0.479 (l: surv 0.940 ratio 0.431 tripod 0.482; r: surv 0.960 ratio 0.399 tripod 0.476) [decisive]
- suite back: n 100 surv 0.850 ratio 0.723 tripod 0.554 [decisive]
- suite diag: n 100 surv 0.580 ratio 0.595 tripod 0.359 [decisive]
- suite turn: n 200 surv 0.995 closed 0.584 progress 0.078 (l: surv 1.000 closed 0.566 progress 0.074; r: surv 0.990 closed 0.585 progress 0.079) [decisive]
- suite move_turn: n 200 surv 0.785 closed 0.303 progress 0.042 ratio 0.571 (l: surv 0.750 closed 0.287 progress 0.042 ratio 0.568; r: surv 0.820 closed 0.307 progress 0.044 ratio 0.578) [decisive]
- WATCH gate_1c: FAIL -- formation gate FAILED — walking-hold tripod 0.359 (need >= 0.4; by hold: back 0.554, diag 0.359, fwd 0.568, lat_l 0.482, lat_r 0.476, move_turn 0.510, move_turn_r 0.524, settle 0.530, stand 0.532, turn_l90 0.557, turn_r90 0.562), completion 0.866 (need >= 0.7), yaw-progress rate turn_l90 0.074, turn_r90 0.079 rad/s (net turn_l90 0.074, turn_r90 0.079) (need >= 0.05 rad/s on each), per-direction speed_ratio lat_l 0.431, diag 0.595, back 0.723, move_turn 0.568 (need >= 0.3 on each of lat_l, diag, back, move_turn; failing: none) on slow__omni, training track_ratio_mean 0.639 (need >= 0.6 over the last 500 its) (WATCH only: this gate never decides in the ladder)
- overlay PASS: overlay KRABBY_OMNI_CMD_POOL=stand:0.15,fwd:0.14,side_l:0.09,side_r:0.09,back:0.09,omni:0.14,turn:0.1,fwd_turn:0.06,side_l_turn:0.035,side_r_turn:0.035,omni_turn:0.07 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_DIR_BAND= in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_FACING=0.2:0.6 in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_RSI_SOURCE=all in params/env.yaml
- overlay PASS: overlay KRABBY_OMNI_SLOTS=0.15:0.55:0.1:0.2 in params/env.yaml
- overlay PASS: overlay KRABBY_YAW_W=0.2 in params/env.yaml
- floor PASS: category survival on suite (one-sided 99 % band): stand 0.980 (bound 1.013, n_env 100, min >= 0.79); fwd 0.800 (bound 0.893, n_env 100, min >= 0.65); side 0.950 (bound 1.001, n_env 100, min >= 0.77); back 0.850 (bound 0.933, n_env 100, min >= 0.68); diag 0.580 (bound 0.695, n_env 100, min >= 0.52); turn 0.995 (bound 1.011, n_env 100, min >= 0.785); move_turn 0.785 (bound 0.881, n_env 100, min >= 0.545)

>>> ENTRY X02:c6x

### experiment X02 — X0x s2 — ok — 2026-09-25T14:53:17

- recipe `b48e7efd9851` base `b48e7efd9851` chain `c6x` seed 2
- stages ['X02:c6x']
- screen: completion 0.866, tripod_dir 0.359, track tail 0.639
- replay: none

>>> ENTRY X02


## Close (2026-09-25): W baked as the omni phase-1 head of record

**User decision (2026-09-25):** accept W as the omni phase-1 head of record, overruling the driver's D05 floor
REJECT. The overrule is `ledger_cli overrule D05 --choose ADOPT` (uuid 74b5e271…); OVERRIDE-APPLIED 19:08, with no
squash and nothing queued. W's `arm_class` is UNCLASSED (user-overruled). The formation-seed-2 confirm (plan step
15, 7.65 GPU-h) is **waived** by the user. "Bake" means records plus a policy folder only: no `KRABBY_OMNI_PHASE`
presets (user choice), so code_sha does not move. The recipe of record keeps the clock weights at ε 0.001, as W
trained.

- **Head of record:** `crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt`
  - source `crab_hex_omni_flat_walk/2026-09-25_14-53-27_seed3_X03_c6x_A15pB/model_11995.pt`
  - sha256 `3a527683…`
  - The whole lineage (1_t3 → 2_c4 → 3_c5 → 4_c6 → 5_c6x_clockfree), each stage's evals, the observation contract
    and the follow-ups are in `crab_hex_omni_task/policy/manifest.yaml`.
- **Recipe of record:**
  - C1d t3 (0 → 3000, from scratch) → c4 (→ 4500) → c5 (→ 6000) → c6 (→ 8000), clock held;
  - then c6x (→ 12000): C1d's c6 overlay plus K (`KRABBY_CLOCK_W=0.001`, `KRABBY_APEX_W=0.001`,
    `KRABBY_OMNI_CLOCK_HIDE=1`).
- **Observation contract:** any play, eval or export of W must set `KRABBY_OMNI_CLOCK_HIDE=1`, plus the rest of the
  carried eval env (`policy/README.md`).
- **The full frozen-rule readout, ignoring the floor** (offline replay of the driver's own decision code;
  `parkour/outputs/omni_plan/clock_deprecation/w2_readout/rule.md`):
  - Still a clear REJECT, class BREAK.
  - WORSE guards: move_turn survival (0.365 vs 0.865), move_turn ratio (0.442 vs 0.521) and side foot_slip (0.185
    vs 0.173, 1.8× its R).
  - Blocks: F BETTER (z +2.02), T BETTER (+2.09), F_mt TIE, RIG TIE (+1.11 vs R 1.13; both halves ≥ 0).
  - Flags: the move_turn survival and ratio margins, and speed-driven.
- **Post-cut transient** (tfevents; `w2_readout/transient.md`):
  - episode length −27 % at it 164, back in band by 450;
  - crab_failure +0.34 at it 337, back in band by 534;
  - nothing degrades late beyond the X0x seed-pair null.
- **Diagnostics** (never gated; suite evals of mid-stage checkpoints, same carry as the driver; tables appended to
  WAVE2_METRICS.md):
  - W's move_turn survival goes 0.62 (+1000) → 0.255 (+2000) → 0.28 (+3000) → 0.365 (+4000). The twin X0x is
    s3 0.71 → 0.865 and s2 0.835 → 0.785 (+2000 → +4000). So move_turn was lost during clock-free training between
    +1000 and +2000; it was not an initial shock.
  - Turning improved steadily: turn facing_closed 0.475 → 0.547 → 0.780 → 0.787.
  - The light__omni terrain WATCH shows no large difference: completion X0x 0.38 vs W 0.34, falls 62 vs 66, median
    slip 0.37 vs 0.41.
- **Follow-up (user, 2026-09-25): move_turn, particularly "Nobody actually turns during move_turn".**
  - Every omni head, clock-held or not, reaches 0.02–0.07 rad/s against the 0.3 rad/s commanded. The heading
    error at the end of the hold is 1.1–1.5 rad.
  - W's failure is a forward nose-dive: the front pair lifts together, and penalty_base_pitch_forward_linear sits
    at weight 0.
  - Mechanism and levers: [MOVE_TURN_MECHANISM.md](MOVE_TURN_MECHANISM.md).
  - Recorded as `followups.move_turn` in the policy manifest.
- **Video:** `parkour/logs/videos/clock_w2_cut/w_cut_categories.mp4`, all 10 holds. The two move_turn holds fall at
  13.2 s and 8.9 s with 67° and 51° of heading unclosed; the in-place turns close to within 3–4°.

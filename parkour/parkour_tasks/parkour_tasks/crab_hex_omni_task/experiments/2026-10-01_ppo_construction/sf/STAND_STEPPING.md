# Stand stepping on the omni head of record (B s18 c6x)

2026-10-06. CPU-only re-analysis of existing eval npz, run configs and code; nothing was trained or evaluated.
**M** = measured, **I** = inference. Source keys are at the end.

## Answer

**No, the facing reward does not cause it.**
- During the head's stand holds the yaw command is exactly 0 on every row, and the heading error never exceeds 0.058 rad. The dead band is 0.133 rad.
- The facing reward pays 0.1977 of its 0.200/s maximum whether or not the feet move.
- The seed-18 ancestor trained with facing weight 0 (FB6k) already stepped at stand, at 2.33 steps/s. [R, H, TR-s1]

**What it is instead.** The head runs its walking rhythm at zero command: alternating tripods at 0.476 Hz, inside its own walking range of 0.36–0.48 Hz, with 8 cm lifts, and all six feet down only 0.2 % of the time.
- No reward term pays for this; a still stance would earn about +0.17/s more.
- Nothing at c6x pays much for stillness either, and the policy has never learned a separate stand mode.

## 1. Mechanisms, ranked by evidence

| # | mechanism | strength | key evidence (M) | source |
|---|---|---|---|---|
| 1 | **No stand mode: one learned rhythm for every command** | strong | Stand cadence 0.476 Hz vs fwd 0.409 Hz and diag 0.476 Hz. About 10 tripod swaps per 11 s. Planar stride per swing 0.031 m at stand vs 0.518 m on fwd. Starts in the prologue (3.0 swings/s after the spawn transient) and stays flat for 12 s. Continues after an in-place turn reaches the band: 3.5 swings/s before entry, then 3.0 and 3.11 after (200/200 turns). Every trained head steps (§3); the scripted E6 statue does not. | H summary.md:62-68; V-trip; V-tab; T3, LR-tb |
| 2 | **The c6x stage turned tapping into a visible march** | moderate | Before c6x, B heads planted one tripod (94.5 / 94.5 / 91.3 % of the hold at s1 / c4 / c6) and tapped with the other (median swaps 0 / 4 / 2 per 11 s). 12–20 % of s18 episodes swung less than 0.25/s, and clearance was 0.035 m at c6. At c6x they march with 0.080 m lifts and every episode steps. The c6 → c6x env.yaml diff is three keys: `clock_hide` → True, `reward_clock_schedule` and `reward_clock_swing_apex` 1.0 → 0.001. With the clock stopped, the schedule term pays stance quality on all six legs, so c6x removed the only term that paid for planted feet. | V-trip; V-ext E1; T1; `crab_hex_clock_reward.py:137,153` |
| 3 | **Stepping is cheap in B** (contributes, not the origin) | weak | action_rate is 0 at every B stage; at W's −0.3, the head's stand action rate would cost 0.333/s. C0 and W train at −0.3 and step at the same rate. B's bound loss cut saturation (\|a\|>1 on 26 % of stand steps vs 97 % for W) without stopping the stepping. B also flips action sign more often (7.8 vs 1.9 per dim per s). | replay.md; TR env.yaml; prec_sat.txt |

**I (inference).**
- A still stance earns more (§4), so the stepping is a basin the optimiser never left, not a reward it is chasing.
- Two things weaken the pull out of it:
  - Stand is only 13 % of c6x training rows (TensorBoard `stand_frac_actual`, code.md).
  - Exploration noise is large: σ is 1.10 on the cams and 0.67 on the rods (replay.md). That shrinks what stillness is worth during training.
- These data cannot tell which c6x change caused the march, the hidden clock or the lost stance income. Check D2 separates them.

## 2. Ruled out

| candidate | evidence (M) | source |
|---|---|---|
| Facing reward `reward_tracking_yaw` = 0.2·exp(−\|err\|) | No command gate and no contact input. Pays 0.1977 of 0.200/s. Inside the band it can never separate behaviours by more than 0.025/s. Heads trained at weight 0 step anyway: B FB6k at 2.33 and C0 FB6k at 3.25 steps/s. | `crab_hex_omni_rewards.py:263-282`; R; TR-s1 |
| Facing command / relay | wz_cmd = 0 on every head stand row. In C0 and W the relay fires on 4–21 % of stand rows. Their swings/s with the relay on vs off are 3.37/3.47, 3.35/3.41, 3.16/2.75 and 2.78/2.90, with no consistent sign. W steps at 3.23/s before its first engagement and 3.31/s after. | V-tab V2/V4; R |
| Facing observation (raw error, slots 6/7) | Swings/s are flat across \|err\| bins from 0 to 0.06 rad: 3.27, 3.05, 2.82, 2.96, 3.90. Per-episode correlation between error rms and swings/s is 0.07. | V-ext E3 |
| Gait clock | `clock_running` is 0 on every stand row, and phase is constant within each episode. Clock plus apex income at c6x is 0.0004/s. | R |
| Saturated policy mean (RL review `DOSSIER.md:496`) | B saturates on only 26 % of steps and still steps. | prec_sat.txt |
| Turning STAND_HOLD / PARK / OBS on at eval | None was ever trained. HOLD only zeroes wz, which is already 0. PARK's clock reward is weighted 0.001 at c6x. OBS changes the observation width. | `motion_knobs.py:40-46`; manifest.yaml:116-120 |

## 3. Stand stepping per head (12-s suite stand hold, 100 episodes)

| head (eval stamp) | role | facing w | action_rate w | steps/s [S] | swings ≥1 cm/s [T1] | clearance m [T1] | six-down [S] | tripod [S] | gait Hz [S] | stand rows wz≠0 [R] |
|---|---|---|---|---|---|---|---|---|---|---|
| B s18 FB6k (`seed18_X43_s1…/2026-10-04_16-15-43`) | ancestor | 0 | 0 | 2.333 | 2.29 | 0.064 | 0.000 | 0.000 | 1.550 | 0 % |
| B s18 t3 (`seed18_X50_t3…/2026-10-05_11-18-21`) | sibling (P1-B6) | 0.2 | 0 | 3.792 | 3.58 | 0.098 | 0.012 | 0.457 | 1.138 | 0 % |
| B s18 t3x (`seed18_X53_t3x…/2026-10-05_18-00-53`) | sibling | 0.2 | 0 | 3.250 | 3.04 | 0.092 | 0.003 | 0.236 | 1.215 | 0 % |
| B s18 c4 (`seed18_X57_c4…/2026-10-05_22-15-16`) | ancestor | 0.2 | 0 | 3.667 | 3.54 | 0.079 | 0.003 | 0.218 | 1.367 | 0 % |
| B s18 c6 (`seed18_X60_c6…/2026-10-06_06-16-50`) | ancestor | 0.2 | 0 | 3.250 | 3.33 | 0.035 | 0.009 | 0.042 | 1.639 | 0 % |
| **B s18 c6x = head** (H) | head | 0.2 | 0 | **3.167** | **3.17** | 0.080 | 0.002 | 0.794 | 0.476 | **0 %** |
| B s16 c6x (`seed16_X58_c6x…/2026-10-06_01-25-00`) | seed | 0.2 | 0 | 3.500 | 3.58 | 0.122 | 0.013 | 0.777 | 0.586 | 0 % |
| B s17 c6x (`seed17_X59_c6x…/2026-10-06_04-40-41`) | seed | 0.2 | 0 | 3.583 | 3.67 | 0.074 | 0.008 | 0.753 | 0.555 | 0 % |
| C0 s18 FB6k (`seed18_X44_s1…/2026-10-04_18-47-59`) | control | 0 | −0.3 | 3.250 | 3.25 | 0.095 | 0.005 | 0.584 | 0.522 | 20.7 % |
| C0 side-first s16 c6x (`seed16_X64_c6x…/2026-10-06_14-11-51`) | control | 0.2 | −0.3 | 2.667 | 2.83 | 0.165 | 0.002 | 0.716 | 0.433 | 12.8 % |
| W (W-stamp) | old head | 0.2 | −0.3 | 2.750 | 2.92 | 0.119 | 0.002 | 0.699 | 0.452 | 3.8 % |
| E6 scripted statue (E6-stamp) | still control | – | – | 0.000 | – | – | 0.000 (5.0 feet loaded) | 0.000 | – | 0 % |

Weights come from each run's `params/env.yaml`. "Six-down" is the share of time all six feet are down.

## 4. Reward economics: stepping vs still (c6x weights, head stand, 59,900 steps) [R]

| term (weight) | head stepping /s | ideal still /s | still − head |
|---|---|---|---|
| track_lin_vel_xy_exp (1.25) | 1.2196 | 1.2500 | +0.0304 |
| track_ang_vel_z_exp (1.0) | 0.9952 | 1.0000 | +0.0048 |
| **reward_tracking_yaw / facing (0.2)** | **0.1977** | **0.1977** | **0.0000** |
| penalty_tracking_error_l1 (−2.0) | −0.0541 | 0 | +0.0541 |
| reward_delta_torques (−1e-6) | −0.0822 | 0 (idealised) | +0.0822 |
| reward_lin_vel_z (−0.15) | −0.0027 | 0 | +0.0027 |
| penalty_tibia_deviation_in_stance (−0.28) | −0.0078 | −0.0148 | −0.0070 |
| reward_orientation (−0.7) | −0.0004 | −0.0004 | 0 |
| reward_clock_schedule, stopped-clock stance quality (0.001) | 0.0004 | 0.0010 | +0.0006 |
| apex, yaw_progress, air time, stride, excess-feet, foot_idle | 0 (gated off at zero command) | 0 | 0 |
| action_rate (0 in B; −0.3 in W and C0) | 0 (would be −0.333) | 0 | 0 |
| **total** | **2.2656** | **2.4335** | **+0.168** |
| same stand re-weighted at c4–c6 (clock schedule 1.0) | 2.6738 | 3.4325 | +0.759 (stance quality +0.591) |

- M: the c4 and c6 heads, trained under that +0.76/s pull, still stepped at 3.67 and 3.25 steps/s.
- M: the head's own stillest 10 % of 1-s windows earn 2.3038/s, on exactly 3 loaded feet.
- I: the "ideal still" column assumes zero torque change.

## 5. Precedent

| record | finding | source |
|---|---|---|
| X08:c6 (C1d head), 2026-09-22 | Same symptom: stand tripod 0.587 vs forward 0.576. | `2026-09-22_gait_motion_quality/CHARTER.md:11-13` |
| Facing-loop hypothesis | Raised for X08:c6, whose relay leaked (wz_active 0.105). | `lit-review-gait-motion-quality.md:25-33` |
| Clock-campaign stand attribution (A4500, X0 s3) | Falsified the facing loop. Steps/s for errors below 0.05 rad / 0.05–0.133 rad / above the band: A4500 2.68 / 2.68 / 3.03, X0 3.36 / 3.32 / 3.12. Steps/s before → after a relay onset: 2.73 → 2.82 and 3.27 → 3.15. Stepping cost 0.92–0.96/s vs the statue with action_rate −0.3, and they stepped anyway. Recorded verdict: "the policy's own tripod rhythm". | prec_policy_lens2/3.txt; prec_checker.txt; memory `clock-deprecation-campaign-active.md:193` |
| Only near-still stander (1c head, 2026-09-17) | 5.33 feet loaded, when the clock term paid 0.65–0.69/s on the stand hold. Old economy with many confounds. | `ANALYSIS_1c_failure_2026-09-18.md:89,106` |
| Stand knobs | HOLD, PARK and OBS were built in 2bb196c0 and never trained; motion-quality ladder rows M5/M6 parked before running. STAND_POSE −0.14 failed replay Gate 1 on stand (ρ −0.61). | `candidates.yaml:276-297`; `REPLAY_GATE.md:31` |

## 6. Confirming the cause (proposals; GPU only on your go)

Eval-only first. Training runs at about 44 its/min (c6x did 4,000 its from 02:24 to 03:54, TR-c6x mtimes). Eval costs are rough and were not timed.

| check | if the facing hypothesis holds | if the default rhythm holds | cost |
|---|---|---|---|
| **D1. Facing-blind stand eval**: run the head's stand wave with `KRABBY_OMNI_FACING_OBS_CLIP=1e-4`, which clips obs slots 6/7 to about 0 (the command path is already 0). It is an EVAL_CARRY key: set it at launch; the driver's carry check may need bypassing. | stepping drops | 3.0–3.3 swings/s, unchanged | ~10–20 GPU-min, no code |
| **D2. Split the c6x cut**: fork the s18 c6 head into a c6x run that hides the clock but keeps `KRABBY_CLOCK_W=1.0` (apex 0.001). Same seed, so it pairs with the real c6x. Read the stand only, since the clock weight also pays walking rows. | – | about 3 steps/s either way. A march (tripod ≈0.8, ≈0.48 Hz) blames the hidden clock; parked-tripod tapping (tripod ≤0.2, ≈1.4–1.6 Hz) blames the lost stance income. | ~45 GPU-min (2,000 its) + 2 suites |
| **D3. Offline replay gate** (CPU; required before F1/F2): score candidate stand-slot terms (stance quality, foot planar speed, airborne-foot count, action rate) on saved npz, E6 statue vs head, W and C0, using `experiments/tools/replay_motion_terms.py`. | – | Passes if the statue beats every stepping head and the term's within-head correlation with swings/s is negative. | 0 GPU |

## 7. Candidate fixes (proposals for you)

| fix | stage / cost | expected | side effects |
|---|---|---|---|
| **F1. Stand-slot stance income through c6x.** A stand-slot-gated copy of the stopped-clock stance-quality branch at weight 0.5–1.0, with running-clock income left at 0.001. Small code change. Gate on the stand slot, not on "clock stopped": the clock also stops for slow commands (≤0.2 m/s, `crab_hex_clock_reward.py:49-83`). | Paired c6 → c6x fork (~1.5 GPU-h), or a 2,000-it fine-tune of the head (~45 min) | Fewer, lower steps. A six-foot stand is unlikely: at weight 1.0 the c4 and c6 heads collected only 0.43–0.47 of this income. | None on walking rows; check stand → move onset (fwd/side survival, move_turn) |
| **F2. Stand-slot stillness penalty**: foot planar speed or airborne-foot count, optionally plus stand-only action rate, sized so stillness beats stepping by ≥0.5/s. | Same as F1; must pass D3 first (STAND_POSE failed it) | Direct pressure on the symptom | Risk of a stiff stand that is slow to start |
| **F3. STAND_OBS from scratch** | Full chain, ~15,000 its (~6 GPU-h per seed) | Lowest priority: cmd = (0, 0, 0) is already observed | Changes the observation width; every stage must be retrained |

- **Not expected to help:**
  - Facing weight 0: FB6k already had it and stepped.
  - STAND_HOLD.
  - STAND_PARK alone at c6x.
  - action_rate alone: W and C0 still step with it. Restoring it would still cut the jitter flagged under `action_jitter` in the manifest.
- **Decision for you:** F1 and F2 are stand-gated. On 2026-09-22 you rejected a stand-centric plan ("the stand is one symptom of a larger problem"; memory `gait-motion-quality-campaign.md:10-12`). The evidence here says general smoothness terms did not stop stand stepping in W or C0, but which way to go is your call.

## Corrections made in verification

- **Pre-c6x B heads tapped, not shuffled:** they parked one tripod and tapped the other.
- **Facing relay at FB6k:** the relay existed at FB6k even though the facing reward weight there was 0.
- **C0/W action rate:** C0 and W trained with action_rate −0.3.
- **W re-eval is one sample:** the W re-eval stamp is byte-identical to the original one.
- **No c5 eval stamp exists.**
- **Heading error:** 0.058 rad is the pooled maximum; 0.026 rad is the median of the per-episode maxima.
- **Lit-review error:** `lit-review-gait-motion-quality.md` §1.3 understates stopped-clock stance income. The code pays all six legs.
- **Precedent citations:** the precedent lens had no verifier; I spot-checked its repo citations and they matched.
- **Stepping in the video is inferred:** the sequence-video sanity JSON has no per-step trace. For the stand segment it records wz 0.0, a −0.95° heading change and 0.201 m drift.

## Source keys

- **GE** = `/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-10-01_ppo_construction/<tag>/suite__omni/seed001/<ts>`; its `summary.md` is [S].
- **H** = GE `seed18_X60_c6x_A15pB/…/2026-10-06_07-56-23`.
- **W-stamp** = `…/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29`.
- **E6-stamp** = `…/omni_phases/2026-09-23_clock_deprecation/step0_E6_statue/suite__omni/seed001/2026-09-23_21-59-41`.
- **TR** = `/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/<run>/params/env.yaml`:
  - TR-s1 = `2026-10-04_10-01-39_seed18_X43_s1_A15pB`
  - TR-c6x = `2026-10-06_02-24-21_seed18_X60_c6x_A15pB`
  - C0 runs: `2026-10-04_12-33-54_seed18_X44_s1`, `2026-10-06_08-41-48_seed16_X64_c6x`
  - W run: `2026-09-25_14-53-27_seed3_X03_c6x`
- **LR-tb** = `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/ppo_construction/lr/SF__B_rest_s18_tb/turn_band__omni/seed001/2026-10-06_08-41-42`.
- **Scratch files** (this directory):
  - R = `v_replay.json`
  - V-tab = `verify_tables.txt`
  - V-ext = `verify_extra.txt`
  - V-trip = `verif_tripod_sets.txt`
  - T1/T3 = `measure.md`
  - also `prec_*.txt`, `code.md`, `replay.md`
- **Code** is under `parkour/parkour_tasks/parkour_tasks/crab_hex_{omni,forward}_task/mdp/` and in `parkour_isaaclab/envs/mdp/rewards.py`.

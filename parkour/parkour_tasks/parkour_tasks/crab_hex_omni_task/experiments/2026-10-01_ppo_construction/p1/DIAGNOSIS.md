# P1-B6 t3 stop: diagnosis (opus·high, workflow wf_4c16e6f4-60b; CPU only)

## Turn failure mechanism

B's turning does not fail during training. It fails to transfer to a sustained, out-of-band in-place turn.

**What training shows.** The turn reward and progress are not starved for B. By its 5999:
- yaw_progress is B 0.023-0.028 vs C0 0.011-0.019;
- reward_yaw_progress is equal (about 0.004/episode in both arms);
- reward_tracking_yaw is B 0.176 vs C0 0.150-0.164;
- facing_err_abs is lower for B.

Sigma is healthy (0.78-0.81), and LR and KL match C0's.

**What the eval shows (deterministic, 90-degree hold).** B produces one yaw burst: 0.14-0.19 rad/s for about 1 s, worth 0.15-0.19 rad. It then steps in place, clock-locked at 3.3 Hz with normal swing apex, and gets almost no yaw per step: 145-486 steps/rad vs C0's 50-70. Net rotation over 4-12 s is -0.11/-0.03/+0.05 rad, against C0's +0.38/+0.61/+0.47. B is missing a repeatable in-place turning stroke (stroke plus return).

**Why the training metrics did not catch it.** In t3, turns only ask for 0.2-0.6 rad facing offsets (about 0.07-0.47 rad of rotation to reach the dead band), so a single burst partly satisfies them. The eval's 1.57 rad error keeps the facing observation pinned at its 0.6 clip for the first roughly 1 rad, so B stops while its input still says "turn".

**How C0 differs.** C0 turns with deterministic, saturated ±1 bang-bang actions (|mu| about 8, sat>1 at 97%, lock about 0.7). That gait does keep rotating for the whole hold.

**What this is not.** The bound is not what blocks B. C0's executed waveforms all lie inside [-1, 1], below the 1.1 hinge. On turn holds B's share of |mu| > 1.1 is only 0.01-0.08 for the cams, less than on its own fwd holds. B also sustains 0.03-0.04 rad/s while translating on move_turn.

**Two features the data shows but cannot resolve:**
- B drives its cams weakly in place (cam |mu| 0.29-0.40, executed SD about 0.4) compared with fwd and move_turn (|mu| 0.7-1.1) and with C0 (executed SD 0.98). Whether a cam-driven stroke is what continuous turning needs is not decidable here.
- The training metric is computed on stochastic rollouts and the eval uses the deterministic mean. With σ≈0.78 against |μ|≲1.1, those can differ, whereas for C0 they cannot.

## Side formation

Slower, not blocked.

**All six B hands rise from model_4500 to the head** (hand means of feet_ss at 0.22 and 0.25 m/s):

| Seed | L, 4500 → head | R, 4500 → head |
|---|---|---|
| s16 | 0.125 → 0.147 (+0.022) | 0.108 → 0.127 (+0.019) |
| s17 | 0.088 → 0.128 (+0.040) | 0.137 → 0.150 (+0.013) |
| s18 | 0.129 → 0.188 (+0.059) | 0.142 → 0.171 (+0.029) |

C0 is flat over the same span (-0.015 to +0.007 per hand).

**Exposure.** Measured hand_frac is 0.15-0.17 per hand on t3 vs 0.19-0.21 on FB6k. That gives about 740 hand-iterations at t3 4500 and about 990 at the t3 head, against about 600 at FB6k 3000 and about 900 at FB6k 4500, the second-hand-FAST point the task cites as "~900 at 0.2". So t3's head has reached FB6k's completion exposure.

**The slowdown is beyond dilution.**
- At more exposure (t3 4500, about 740), B's best hands are 0.125-0.142. FB6k B at less exposure (3000, about 600) already had first hands at 0.162-0.198.
- At about equal exposure (t3 head about 990 vs FB6k 4500 about 900), t3 first hands are 0.147/0.150/0.188 vs FB6k 0.210/0.232/0.212, and second hands are 0.127/0.128/0.171 vs 0.195/0.178/0.172.

**Rate.** B t3's rise of +0.012 to +0.059 per 1500 its is at or below FB6k B's second-hand rise of about +0.05-0.07 between its 3000 and 4500. Extrapolating linearly, t3 B would need roughly 1500-4500 more iterations for the weaker hands, but that is not measured.

**Per-hand gradient telemetry explains none of it.** hand_grad_cos is about 0 (±0.07) and hand_adv_mean_l/r is within ±0.1 with no persistent sign, in B t3 as in FB6k and C0.

**Training aggregates confirm a slower, not stalled, run.** track_ratio_mean at 5500-5999 is B t3 0.67-0.69, still rising slowly, vs FB6k B 0.74-0.77.

## Findings

- **[strong] In training, B's turning improves and matches or beats C0. The t3 failure appears only in the eval's 90-degree in-place hold.**
  Source: tfevents in each run_dir, windowed means from scratchpad win.py. Metrics/base_velocity/yaw_progress (signed wz*sign(cmd_wz) over rows with cmd_wz != 0) for B s16/s17/s18 goes from 0.002/0.004/0.005 at its 100-500 to 0.023/0.024/0.028 at its 5500-5999, and was still rising after 4000. C0 goes from -0.0001 to 0.019/0.011/0.016. Episode_Reward/reward_yaw_progress at 5500-5999 is B 0.0041/0.0047/0.0042 vs C0 0.0045/0.0040/0.0038. reward_tracking_yaw (YAW_W 0.2) is B 0.176/0.176/0.177 vs C0 0.163/0.150/0.164. facing_err_abs is B 0.080/0.081/0.072 vs C0 0.085/0.094/0.091. B run_dirs: crab_hex_omni_flat_walk/2026-10-04_19-02-50_seed16_X46_t3_A15pB, 2026-10-05_00-04-34_seed17_X48_t3_A15pB, 2026-10-05_05-04-17_seed18_X50_t3_A15pB. C0 run_dirs: 2026-10-04_16-38-18_seed16_X45, 2026-10-04_21-31-34_seed17_X47, 2026-10-05_02-33-17_seed18_X49. reward_facing_turn and reward_one_direction_spin are 0 in every run, so those terms are inactive.
- **[strong] On the eval turn holds, B makes one fast yaw burst of about 0.15-0.19 rad in the first 1.25 s and then stops turning for the remaining 10 s. C0 keeps rotating at 0.04-0.08 rad/s for the whole 12 s hold.**
  Source: suite__omni raw npz, turn_l90 + turn_r90, 200 episodes per run, scratchpad turn_tc.py. Each figure below is for s16 / s17 / s18.
- B yaw rate toward the target (rad/s), 0.25-1.0 s: 0.14-0.19 in all three seeds. 1.5-2 s: 0.009 / 0.019 / 0.006. 2-12 s: -0.020..-0.013 / -0.015..-0.003 / -0.002..+0.008.
- B |heading error|: 1.57 at the start, about 1.38 at 2 s, then 1.51 / 1.43 / 1.32 at 12 s.
- C0 yaw rate toward the target: 0.075-0.119 peak at 0.25-1 s, then 0.04-0.08 sustained over 2-12 s. |heading error| falls to 0.90-1.09 by 10-12 s.
- Net rotation over 4-12 s (scratchpad twist.py): B -0.11 / -0.03 / +0.05 rad vs C0 +0.38 / +0.61 / +0.47 rad.
- B steps in place without yawing. Suite summaries give step_rate_hz 3.25-3.33 (C0 3.54-3.67), swing_apex 0.095-0.130 (C0 0.118-0.128) and gait_hz 0.5 clock-locked, but steps_per_rad 486 / 145 / 148 vs C0 70 / 50 / 58.
- Suite stamps: gait_eval/omni_phases/2026-10-01_ppo_construction/seed16_X46_t3_A15pB/suite__omni/seed001/2026-10-05_01-16-54, seed17_X48/.../2026-10-05_06-18-38, seed18_X50/.../2026-10-05_11-18-21; C0 seed16_X45/.../2026-10-04_22-48-22, seed17_X47/.../2026-10-05_03-47-38, seed18_X49/.../2026-10-05_08-47-21.
- **[moderate] The eval turn is outside what t3 trained. Training facing targets are 0.2-0.6 rad (about 0.07-0.47 rad of rotation needed to reach the 0.133 rad dead band). The eval asks for 1.57 rad, with the facing observation pinned at its 0.6 clip for the first roughly 1 rad. A one-stroke burst of 0.15-0.19 rad would satisfy a meaningful share of training turns.**
  candidates_v6.yaml lines 1765-1769 set KRABBY_OMNI_FACING '0.2:0.6' and YAW_W 0.2. The suite's env_vars set KRABBY_OMNI_FACING_OBS_CLIP 0.6, ANG_VEL_MAX 0.3 and ANG_VEL_CLIP 0.2, and suite.holds turn_l90 has face 1.5708 with hold_s 12 (run_meta.json). crab_hex_omni_phases.py (comment at the FACING_OBS_CLIP entry, around lines 173-185) records that a 1c-band head already fell on the 90-degree canary in 2026-09. B stops turning while its observed facing error is still clipped at 0.6 (|he| about 1.38 > 0.6). So whatever ends the burst is not a change in the observed error: it could be proprioception or the history encoder, but the data cannot say which.
- **[strong] C0 turns with deterministic, saturated bang-bang actions, not with large random ones. Its noise is irrelevant because |mu| is far above 1 + 2 sigma.**
  Turn holds (scratchpad rawstats.py on raw npz): C0 median-like mean |mu| is cam 7.9-8.3, femur 10.0-11.1, tibia 7.2-8.8. The share with |mu| > 1 is cam 0.97-0.98 and rod 0.94-0.96. The executed (clipped) per-joint SD is 0.97-0.98 cam and 0.85-0.99 rod, i.e. ±1 square waves. Training PPO/lock_cam / lock_rod (|mu| > 1 + 2 sigma) at 5500-5999 is 0.69/0.73/0.75 and 0.76/0.74/0.74, with effective_std pinned at 2.000 from about its 3000. The deterministic eval sustains 0.04-0.08 rad/s, and C0's rollouts are near-deterministic too. B's turn-hold actions are smaller: cam |mu| 0.29-0.40, executed cam SD 0.35-0.43, joint_speed_mean 0.42-0.46 vs C0's 0.74-0.76 (summary.md).
- **[strong] Exploration does not collapse. B's sigma is flat at the 0.8 init and is slightly higher with turning in the pool than on FB6k. C0's sigma hits the 2.0 clamp.**
  PPO/effective_std at 5500-5999: B t3 0.786/0.806/0.782; FB6k B 0.758/0.746/0.770; C0 t3 and FB6k 2.000, reached by about its 3000-3500. Loss/entropy: B t3 20.96/21.39/20.92; FB6k B 20.31/20.11/20.64; C0 38.02, the σ=2 maximum. PPO/kl_mean is 0.0115-0.0121 in every arm. PPO/lr_applied at 5500-5999 is B t3 2.04-2.06e-5, FB6k B 1.90-1.92e-5, C0 t3 1.86-1.96e-5, so there is no LR confound. PPO/exec_change (per-update executed-mean change) is B t3 0.0123-0.0128 vs C0 0.0023-0.0025: B keeps updating and C0 is frozen.
- **[moderate] The 1.1 bound does not make in-place turning unreachable. Every executed action C0 uses lies in [-1, 1], below the 1.1 hinge. B itself rotates faster than C0 in the first second, and sustains rotation on move_turn, without pressing on the bound.**
  On the turn holds, B's share of |mu| > 1.1 is cam 0.01/0.07/0.01 (turn_l90) and rod 0.25/0.28/0.32. That is below B's own fwd holds (cam 0.24/0.28/0.18, rod 0.35/0.51/0.40), so the policy is not at the bound when it stops turning. B's 0.25-1 s burst is 0.14-0.19 rad/s vs C0's 0.05-0.12 rad/s. On move_turn, B sustains +0.03-0.04 rad/s from 2 to 10 s (s16/s17), with cam |mu| 1.02-1.12 vs 0.29-0.40 on in-place turn (rawstats.py). Loss/action_bound at 5500-5999 is a little higher with turning in the pool (B t3 4.99/6.10/5.30 vs FB6k 4.62/4.72/4.65), a weak sign of somewhat more pressure on the hinge in training rollouts.
- **[strong] B's side formation is slower, not blocked. All six B t3 hands rise from model_4500 to the head while C0 is flat. Per unit of side exposure, t3 B is behind FB6k B, so the slowdown is more than dilution of exposure.**
  Hands are means of lat_022 and lat_025 feet_ss (p1/gate/P1B6__*_t3_s*{,_m4500}_lr.json).
- B 4500 → head, L/R: s16 0.125/0.108 → 0.147/0.127 (+0.022/+0.019); s17 0.088/0.137 → 0.128/0.150 (+0.040/+0.013); s18 0.129/0.142 → 0.188/0.171 (+0.059/+0.029).
- C0 4500 → head moves between -0.015 and +0.007 per hand.
- Exposure: PPO/hand_frac_l and hand_frac_r (rows with observed vy beyond ±0.05) are 0.15-0.17 on t3 vs 0.19-0.21 on FB6k. That gives about 740 hand-iterations at t3 4500, about 990 at the t3 head, about 600 at FB6k 3000 and about 900 at FB6k 4500.
- FB6k B at 3000, i.e. less exposure (fb6k/gate/FB6k__B_s*_m3000_lr.json): first hands 0.198/0.162/0.181, second hands 0.132/0.149/0.117.
- t3 B at 4500, i.e. more exposure: first hands 0.125/0.137/0.142, second hands 0.108/0.088/0.129.
- t3 B head vs FB6k B 4500, best hands: 0.147/0.150/0.188 vs 0.210/0.232/0.212.
- Training track_ratio_mean at 5500-5999 is B t3 0.670/0.669/0.689 vs FB6k B 0.768/0.750/0.738.
- **[moderate] The per-hand gradient diagnostics are uninformative. They show no hand-vs-hand conflict signal in any arm, so they cannot explain the slowdown.**
  PPO/hand_grad_cos window means after its 100 are inside ±0.07 in every run (B t3, C0 t3, FB6k B/C0 and INC0) with no persistent sign; only the first 0-100 window reaches -0.13 to -0.18 in some runs. PPO/hand_adv_mean_l and hand_adv_mean_r window means after its 100 are within ±0.1 with no consistent sign by hand or arm. Telemetry is every 25 its (240 points per run).
- **[weak] The aggregate training telemetry shows no direct sign of turn-side competition: side tracking and turn progress rise together in B. Per-category training tracking is not logged, so this cannot rule interference out.**
  B s16/s17/s18 from its 4000 to 5999: track_ratio_mean 0.626→0.670, 0.655→0.669, 0.667→0.689; yaw_progress 0.021→0.023, 0.022→0.024, 0.026→0.028. There is no window where one rises while the other falls in a systematic way. In the eval, side hands rise from 4500 to the head while in-place turning shows only the burst. The only per-category trajectory evidence is the 4500/head lr evals; no suite turn eval exists at model_4500.
- **[strong] With action_rate at 0, B's executed action rate is about twice C0's in every arm, whether or not turning is in the pool. Its eval jitter is highest on the turn holds, at 1.2-1.6 times C0's.**
  PPO/exec_action_rate at 5500-5999 is B t3 3.04/3.03/3.01 vs C0 t3 1.45/1.43/1.49, and FB6k B 2.96/2.90/3.02 vs FB6k C0 1.22/1.32/1.28. So the gap does not depend on turning. INC0 (action_rate 0, unbounded) is 3.71-3.85. In summary.md, turn action_jitter is B 0.197/0.213/0.199 vs C0 0.147/0.136/0.168, and side is B 0.176/0.204/0.163 vs C0 0.130/0.112/0.138. On turn holds after 1 s, per-joint |Δa_exec| per step is B 0.14-0.18 vs C0 0.08-0.10 (turn_tc.py). B's turn jitter is above its own fwd (0.165/0.129/0.191) and move_turn (0.132/0.127/0.145). The mean ratio to C0 is 1.35× on turn and 1.43× on side, just under the §12.5 smoothness flag's 1.5×.

## Open questions

- Does B's stochastic policy keep rotating on a sustained 90-degree turn? Training yaw_progress is computed on stochastic rollouts and the eval on the deterministic mean. With σ≈0.78 against |μ|≲1.1 these can differ, while for C0 they cannot. Deciding it needs a stochastic-action eval or an in-band (0.2-0.6 rad) turn eval of the B t3 heads, which is GPU work and not done here.
- Is the 0.15-0.19 rad burst a stance twist that runs out of joint range, or a single stroke whose return phase undoes it? Loaded feet (about 3.1-3.3) and joint shift (0.57-0.61 vs C0 0.51-0.55) do not separate the two. A per-leg stance/swing yaw-displacement analysis of the raw npz could.
- What ends B's burst while the observed facing error is still pinned at its 0.6 clip: the history encoder, proprioception, or the clock phase? The telemetry cannot attribute it.
- The per-exposure slowdown of side formation cannot be attributed to turning specifically. The t3 pool also changes fwd (.25 vs .2), back (.08 vs .2) and stand (.15 vs .2), and adds FACING plus YAW_W 0.2, against FB. No FB-plus-turn-only or t3-minus-turn arm exists to isolate it.
- There is no per-category tracking in the training tfevents (only pooled track_ratio_mean and turn-row yaw_progress), and no suite turn eval at model_4500. Whether turn and side strokes compete for the same action structure over training cannot be decided from existing data.
- Does a cam-driven stroke matter for in-place turning? B uses weak cam drive in place (cam |mu| 0.29-0.40) but strong cam drive on fwd and move_turn (0.68-1.12), and C0 drives its cams at ±1 everywhere. The function of the cam shaft in yaw generation is not established here.
- Would more t3 iterations close side formation (all six hands rising) without fixing in-place turning? The burst-only turn looks qualitative and not obviously exposure-limited, but that is unmeasured.

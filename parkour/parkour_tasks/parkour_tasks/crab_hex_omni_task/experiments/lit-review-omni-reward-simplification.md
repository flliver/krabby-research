# Literature review: reward-term simplification for the omni task — mapped to the Krabby campaigns

*2026-09-23. Written for `crab_hex_omni_task` under the following conditions: the incumbent X08:c6 (C1d turn-first curriculum, `model_7996.pt`) is fixed, the motion-quality ladder is parked behind the M1 saturation finding, and a separate session owns clock deprecation. This review builds on `lit-review-omnidirectional-gait-formation.md`, `lit-review-gait-motion-quality.md` and the forward task's `lit-review-hexapod-reward-stability.md`, `lit-review-continuous-spin.md` and `lit-review-plasticity-cliff.md`, and does not repeat them. Every citation was verified by opening the source: the paper's full text, or the named file in the official repository. Where verification corrected a sweep claim, the corrected wording is used. Claims that failed verification, or that are our own inference, are listed in the appendix rather than softened. Repo figures are cited to their ledgers; they are not literature. An independent final check aligned this review with the adversarially revised candidate catalogue ([2026-09-23_reward_simplification/CANDIDATES.md](2026-09-23_reward_simplification/CANDIDATES.md)); where the two documents differed, the catalogue's revised wording is used.*

## 1. The problems this review targets

Restated here so the document stands alone.

1. **Overlap.** The omni reward registers 37 names. 17 carry weight on 1a/1b and 25 on 2b/2c. Planar velocity is priced by four terms and yaw by three. Income on X08:c6, taken from TensorBoard `Episode_Reward/<term>` (per-episode sum divided by the maximum episode seconds, mean of the last 10 % of iterations), with share = |term| / Σ|terms|:

| cluster | terms (income/s, share) |
|---|---|
| translation | `track_lin_vel_xy_exp` +0.676 (22.5 %), `penalty_tracking_error_l1` −0.233 (7.7 %), `reward_forward_progress_along_command` +0.055 (1.8 %), `penalty_lin_vel_y` (omni `penalty_cross_command_vel_l2`) −0.014 (0.5 %) |
| yaw / facing | `track_ang_vel_z_exp` +0.721 (24.0 %), `reward_tracking_yaw` (`reward_facing_exp`) +0.150 (5.0 %), `reward_yaw_progress` +0.004 (0.1 %) |
| clock (owned elsewhere) | `reward_clock_schedule` +0.311 (10.3 %), `reward_clock_swing_apex` +0.133 (4.4 %) |
| foot / contact | `penalty_foot_idle_when_forward` −0.236 (7.9 %), `penalty_tibia_deviation_in_stance` −0.042 (1.4 %), `penalty_excess_feet_contact_forward` −0.017 (0.6 %), `reward_feet_air_time_positive` +0.0006, `reward_stride_length` 0.0000 |
| posture | `reward_orientation` −0.003 (0.1 %), `reward_lin_vel_z` −0.0015 |
| smoothness | `reward_action_rate` −0.342 (11.4 %), `reward_delta_torques` −0.069 (2.3 %) |

The teacher runs (2a `2026-09-18_09-41-10`, 2b `2026-09-18_12-58-58`) add collision −0.018/−0.020 and stumble −0.015. `reward_feet_edge` is 0 on every iteration. On 2b, `reward_obstacle_clearance` pays +0.118 (4.2 %), `reward_tracking_goal_vel` +0.098 (3.5 %) and `reward_foot_clearance` +0.063 (2.2 %).

2. **Dead income.** Air time and stride pay about 0 because the contact-sensor first-contact and first-air helpers stop firing about 2 s into an episode (`2026-09-22_gait_motion_quality/REPLAY_GATE.md`). Twelve names are at weight 0 in every omni phase. Four levers have never been trained.
3. **Provenance.** A git weight trace shows that 9 of the 18 active flat-stage weights were set on the May-2026 plant (about 104 kg, position actions) and three more in era A. None of these 12 has been ablated on any plant since the 2026-08-22 collision fix. Per-term A15+B evidence amounts to the walker-vs-creeper replay (`2026-09-16_1805_omni_formation/replay_reward_economy.md`) and the C2 dose-response (`2026-09-19_facing_tracking_ladder/REPORT.md`, D10).
4. **Structure.** `ParkourRewardManager.compute` clips the summed reward at zero on every step (`parkour_isaaclab/managers/parkour_reward_manager.py:38`), so every share in the table is pre-clip. The action-rate term prices the clipped sample of a policy whose mean is saturated (median |μ| 8.1; `ANALYSIS_M1_saturation.md`).
5. **Corrections to the brief.** `penalty_foot_idle_when_forward` penalises a foot held *airborne* for more than 90 steps (`rewards.py:951–987`), not one held in contact. The progress cap is an absolute 1.05 m/s clamp (`rewards.py:923`), not 1.05 × the command.

**The short answer.** The reference stacks price translation and yaw rate with one kernel each and use about nine non-zero terms in total [1, 3, 9]. The omni overlaps are alternatives that the literature uses one at a time, here stacked together (§2–§3). About a third of the stack can be removed with no change in behaviour (§4–§8). A further group each earns under 1 % of income and has hexapod precedent for removal (§6). The largest penalties (action rate, foot idle, L1) are fix-then-decide, because each is tangled with a defect. The zero clip sits upstream of every penalty decision (§9). And "not worse" needs a declared margin; a TIE is not enough (§10).

## 2. Linear velocity tracking

**Key papers**
- One Gaussian on the full xy error: legged_gym [1, 2], Isaac Lab [3], Rudin et al. 2021 [8], CaT [9], Kim et al. [10]. Hexapods: HeLoM [11], Liu et al. [12].
- A projection plus an orthogonal penalty: Lee et al. [13], RMA [14], Qu et al. [15]. Capped progress: Extreme Parkour [16, 17].
- Merged or bounded kernels: Isaac Lab Spot [6, 7], van Marum et al. [20], humanoid-gym [21], Hwangbo et al. [19], Ha et al. [22]. Temporary shaping: Rudin et al. 2022 [18].

**What they show.** The standard term is exp(−‖v* − v‖²/σ²) with σ² 0.25 for ±1 m/s commands [1, 3, 8]. CaT's entire reward is that kernel plus half a yaw kernel [9]. Kim et al. train seven robots with three reward terms: command tracking, torque and smoothness [10]. Hexapod controllers also use a single kernel [11, 12]. A separate across-direction penalty appears only where the task term is a projection that cannot see perpendicular error [13–15]; none of the full-vector recipes reviewed here adds one. Progress terms are capped at the commanded or achievable speed: Extreme Parkour uses min(⟨v, d̂⟩, v_cmd)/v_cmd [17], and Lee et al. cap at 0.6 m/s, "the maximum speed reachable" [13]. Such shaping is removed once the task is learned; Rudin et al. drop r_bias when task reward reaches 50 % of its maximum [18]. Hwangbo et al. scale every cost by a rising curriculum factor except the velocity objective [19].

Exponential and linear pieces can share one term. Spot tracks with exp(−‖e‖/std) [6, 7], and humanoid-gym's `track_vel_hard` is a narrow exponential minus a linear penalty [21]. Hwangbo et al. chose a bounded kernel because a Euclidean cost made falling more rewarding early in training [19], and bounded kernels are named common practice [22]. Counterpoint: humanoid-gym stacks three velocity terms for a slow robot, one of them banded on the speed/command ratio [21]. Overlap is therefore not unusual, only unnecessary for terms that depend on the same error vector.

**What it means here.**
- `track_lin_vel_xy_exp`: keep. Scale-matched to this plant's command range, the omni σ² 0.04 is near stock (0.25 × 0.35² ≈ 0.031). At a 0.25 m/s command it cut standstill income from 0.835 to about 0.21 of the maximum (Revision 4).
- `penalty_tracking_error_l1`: merge (**track-single-kernel**). At a 0.25 m/s command the exp+L1 pair pays −0.24 at standstill, +1.25 on target and +0.06 at a 0.2 m/s error (v ≈ 0.45 m/s); at v = 0.5 m/s (error 0.25) it pays −0.24, the same as standstill. It is the only restoring slope at large error, so fold it into one bounded kernel fitted offline rather than delete it. All of its causal evidence is old-plant (task1_velocity C1; gait_formation A5/C3/C8).
- `reward_forward_progress_along_command`: replace (**progress-relative-cap**). The cap never binds against 0.35/0.25 m/s commands. Income at perfect tracking is 0.126/s for a 0.21 m/s lateral command against 0.21/s for 0.35 m/s forward. The clamp at zero makes retreat free. The per-command form already exists (`crab_hex_omni_rewards.py:210–245`), and phasing it out later has precedent [18].
- `penalty_lin_vel_y`: remove (**cross-penalty-remove**). At a 0.25 m/s command with 0.1 m/s cross error, the exp term loses 0.28/s, the L1 0.20/s and this term 0.03/s. The replay charges the walker more than the creeper on every moving hold (lateral −0.108 vs −0.039).
- `track_lin_vel_xy_exp_facing_coupled`: remove (**dormant-lever-retire**). It has never been trained, and it couples only one of four velocity terms. Walk These Ways composes multiplicatively on the whole task reward instead [23].

## 3. Yaw rate and facing

**Key papers**
- Heading, then a P-controller, then a rate kernel: legged_gym [1, 2], Isaac Lab [3, 4], CaT [9], Kim et al. [10].
- Facing-only precedents: Extreme Parkour [17], Isaac Lab navigation [24], van Marum et al. [20], ANYmal Parkour [28], Rudin et al. 2022 [18].
- Shaping theory: Ng et al. [25], Grześ [26], Jeon et al. [27]. Hexapods: Yang, Gao, Li [29], Liu et al. [12], Jiang et al. [30].

**What they show.** The reference stacks represent a heading target once. A P-controller turns heading error into a rate command, and the reward is a rate kernel at half the linear weight, with no dead band [1–4]. A facing reward appears only where no rate command exists (Extreme Parkour's `tracking_yaw` is its only yaw term [17]) or on a high-level policy that sits above a rate tracker (Isaac Lab navigation [24]). No verified stack pairs a facing reward with a rate tracker on the same policy. van Marum et al. choose the opposite single term: yaw orientation against a rotating setpoint, with no rate kernel [20]. ANYmal Parkour pays heading only in the final second [28]. Only potential-difference shaping preserves the optimal policy [25], and that invariance breaks when the terminal potential is non-zero [26]. PBRS gives little speed-up but is robust to scaling [27]. Hexapod controllers keep only the rate kernel [12, 29]. Jiang et al.'s heading term holds drift and was not ablated [30]. Yang et al.'s simplified reward held only with a CPG prior [29].

**What it means here.** The omni P-relay (stiffness 1.5, cap 0.3 rad/s, dead band 0.133 rad) already does the navigation layer's job, so the facing target reaches the reward three times.
- `track_ang_vel_z_exp`: keep (w 1.0, std 0.5). It gives the steepest signal on turn rows (a settle bonus of about +0.30/s). It is also the only yaw damping on the roughly 70 % of c6 rows that have no turn slot. By estimate, about three quarters of its 24 % share is paid on zero-command rows. The literature's fix for that is **continuous-heading-servo** [4], not a lower weight, since turning is the weakest category.
- `reward_yaw_progress`: phase out (**yaw-progress-phaseout**; removed outright if **one-yaw-kernel** passes). It earns 0.1 % of income but is the only yaw term with a positive dose-response. At w 3 (C2/D10), turn facing_closed rose from 0.419 to 0.748 while move_turn survival fell from 0.58 to 0.12. The term is only approximately potential-based: it is capped, dead-banded, written in γ = 1 form inside γ < 1 PPO, and pays nothing at target resamples [25, 26]. That is one untested explanation of the trade.
- `reward_tracking_yaw`: remove (**one-yaw-kernel**; fallback **turn-slot-facing**). By estimate, about 0.13/s of its 0.150/s falls on non-turn rows. The replay pays walker, mid and creeper heads alike, 0.172–0.195/s per hold (`replay_C2.md`).
- `reward_facing_turn`: adopt it at 0.2 with its shipped gate (turn + move_turn slots) as the replacement for `reward_tracking_yaw` (**turn-slot-facing**). A turn-only gate is a variant, needed only if a later move_turn-feasibility decision requires it; narrowing would cut the move_turn price the candidate protects. `track_ang_vel_z_exp_turn_gated`: remove (**dormant-lever-retire**). Because it is gated on cmd_wz ≠ 0, a body parked just outside the dead band earns about 0.85/s while a finished turn earns 0.

## 4. Gait clock and legacy gait-pattern terms (coordination points only)

**Key papers**
- Clock rewards: Siekmann et al. [31], Walk These Ways [23], Unitree G1 [32].
- Removal after formation: Li et al. [33]. Limits of a fixed clock: HRP-5P [34], WTW §5 [23].
- State-based gait and clearance terms: Spot [6, 7], G1 [32]. Gating: Isaac Lab [5].

**What they show.** Clock schedule rewards are standard [23, 31, 32], but they are either small (G1 0.18 against tracking 1.0 [32]) or multiplied into the task reward (r_task·exp(0.02 r_aux) [23]). A periodic reward needs a clock observation; the reverse does not hold [31]. Li et al. drop their hand-built gait rewards after formation and the hexapod keeps its tripod, in simulation [33]. A fixed-period clock restricted HRP-5P to a rigid stepping pattern; letting the policy modulate the clock raised mean episode length on 7 cm uneven terrain from 6.98 s to 8.49 s [34]. Spot scores pair synchrony and clearance from measured air/contact state, with no phase variable [6, 7]. Isaac Lab's feet-air-time gate uses only the planar command, which switches turning in place off [5].

**What it means here.** These points are handed to the clock session; this review does not duplicate its work.
- `reward_clock_schedule`: coordinate (**clock-hold-twin**). It has never been ablated in omni, and neither omni 2c nor its 2c_clockheld control was run. The forward phase-out it would copy (pre-A15+B) passed only in round 3, and it trailed its full-income control on tripod (0.563 vs 0.615) while beating it on completion (0.79 vs 0.73).
- `reward_clock_swing_apex`: **apex-term-retire**. It never raised lift (forward J1 clearance 0.045 vs 0.044). It pays 0.0001 by teacher 2b, and it pays the creeper more than the walker on lateral, diagonal and back holds.
- `_cmd_gate` / `_clock_cmd_speed`: **gait-gate-decouple**. Five non-clock terms are gated through the clock cadence. Under the omni sampler, a named "locomotion commanded" predicate would behave identically; it matches Spot's full-command gate [6].
- `reward_tripod_schedule`, `reward_one_direction_spin`, `penalty_cam_contact_schedule`, `reward_cam_phase_lock`, `penalty_motor_direction_reversal`: **zero-weight-purge**. All of their evidence is era A. **clockless-tripod-prior** is their dense successor. Its trigger is observed tripod loss after clock removal; it has no timer-fix precondition, because `last_air_time`/`last_contact_time` latch on every transition.

## 5. Foot and contact shaping

**Key papers**
- Air time: legged_gym [1, 2], Isaac Lab [3, 5], Rudin 2021 [8]. Ablations: van Marum [20], Chane-Sane et al. [39].
- The contact-timing bug: Isaac Lab PR #7574 [38].
- State-based timing and constraints: Spot [6, 7], CaT [9], Kim et al. [10]. Hexapod leg idling: Yu et al. [40].

**What they show.** Air-time shaping is small and gated on the command (Isaac Lab 0.125; legged_gym 1.0 × dt) [1, 3]. Removing timing pressure with nothing in its place produces tapping or over-frequent stepping [20, 39]. Isaac Lab fixed a float32 tolerance bug, merged 2026-09-07, that makes `compute_first_contact` and `compute_first_air` miss most transitions once the sensor clock ages (352 of 500 in the reproducer) [38]. The local install still has `abs_tol=1.0e-8`. Spot's mode-time reward reads the current air and contact times, caps both over-long swings and over-long stances, and has a stance branch at zero command [6]. CaT turns air time and contact count into flat-terrain constraints [9]. Yu et al.'s simulated hexapod lifts its front legs at speed [40]; that is the failure foot idle guards against. None of the configs opened has a stride term [1, 3, 7].

**What it means here.**
- `reward_feet_air_time_positive`, `reward_stride_length`: remove now (**dead-swing-income**). They are the only reward callers of the broken helpers (`rewards.py:868, 1390–1391`). Removal changes no behaviour. The risk of leaving them is that a sensor fix would switch them on at weights the incumbent never trained with.
- `penalty_foot_idle_when_forward`: fix, then decide (**foot-idle-audit**; successor **mode-time-after-clock**). It costs −0.74/s early in t3 and −0.236/s at c6. It bills airborne time accrued while its gate was closed, and on the forward hold it charged the walker −1.747/s against the creeper's −0.441.
- `penalty_excess_feet_contact_forward`: keep. It is the only regulariser whose sign separates the creeper correctly (−0.179 vs −0.006 on the forward hold).
- `penalty_tibia_deviation_in_stance`: replace (**tibia-pose-retire**). It is gated on foot contact, which no standard pipeline does [3, 5, 7]. Its rationale predates the linkage default (0.310 rad), and its cost grew from −0.014 to −0.042 across the chain.
- `feet_slide` and `reward_stance_support_feet_when_forward`: **zero-weight-purge**.

## 6. Base posture and stability

**Key papers:** legged_gym [1], Isaac Lab [3, 41], Rudin 2021 [8], Spot [6], Kim et al. [10], Lee et al. 2020 [13], CaT [9], Lee et al. 2023 [42], Qu et al. [15], psu-LiBR [43], Yang, Gao, Li [29], Extreme Parkour [17], Walk These Ways [23].

**What they show.** An orientation penalty is off by default in legged_gym and Isaac Lab and is switched on (−5.0) only in flat quadruped variants [1, 3, 41]. Rudin 2021's reward table has no orientation row [8]. Off-command base motion is priced as one merged quantity [6, 10, 13]. CaT makes orientation a style constraint on flat terrain only [9], while constrained-RL work keeps base-motion damping as a style reward [42]. Hexapod designs carry little or none of this: Qu et al. have no attitude or vertical-velocity term [15], LiBR sets v_z to −1e-8 [43], and Yang et al. dropped both under a CPG [29]. Extreme Parkour masks orientation off its flat terrain class, a choice made for climbing [17]. Where posture matters, WTW commands it rather than penalising it [23].

**What it means here.**
- `reward_orientation` and `reward_lin_vel_z`: drop together (**posture-penalty-drop**). They earn 0.1 % and about 0.05 %. The replay separates the basins by 0.01/s or less, against NET gaps of about 1–2/s. Era-A doses up to 3× moved pitch by 0.000 rad. Orientation is also zeroed on every tile not named `parkour_flat` (`rewards.py:395`), which is 20 % of envs on 1a–1c and 50 % on 2a–2c. The 0.5 rad `crab_failure` termination does the bounding: it ends 36 % of X08:c6 episodes. If the drop loses, the fallback is to drop `lin_vel_z` alone. Removing orientation's tile mask is a separate conditional candidate (**orientation-unmask**), because it adds penalty mass on 20–50 % of envs and needs its own teacher screen. Vertical velocity is where the literature splits (quadrupeds −2.0 [1, 3]; hexapods drop it [15, 43]), so re-check it once the clock is gone [29].
- `reward_ang_vel_xy`, `penalty_base_pitch_forward_linear` and the dead roll/pitch 1.5 rad branches of `terminate_episode_no_goals`: **zero-weight-purge**. `penalty_stand_pose_l1`: **dormant-lever-retire** (Gate 1 ρ −0.61).

## 7. Smoothness, effort and energy

**Key papers:** legged_gym [1, 2], Isaac Lab [3], Extreme Parkour [17], Spot [6, 7], Seyde et al. [44], Andrychowicz et al. [45], rl_games [46], IsaacGymEnvs [47], CAPS [48], LCP [49], Fu et al. [35], Kim et al. [10], RMA [14], Qu et al. [15].

**What they show.** The default is a squared action rate at −0.01 plus tiny torque and acceleration terms [1, 3]. Krabby inherited Extreme Parkour's unsquared action-rate norm (−0.1 there) and its `delta_torques` (−1e-7), then raised them 3× and 10× on a narrower clip [17]. Continuous-control agents prefer the action bounds [44], and tanh squashing with a small initial std does better than clipping [45]. The standard fix is a loss-level bound on μ: rl_games `bound_loss` with a soft bound of 1.1 [46, 47]. Action penalties reduce bang-bang control but cost task performance, and the plant dynamics already low-pass the input [44]. CAPS and LCP move smoothness into the policy loss [48, 49]. A single energy term drives torque, delta-torque, slip and action penalties down as a by-product [35]. Kim et al. keep one smoothness term and one torque term [10]. RMA ramps its penalties in over training [14]. The one hexapod stack uses action rate at −0.5 on the squared form [15].

**What it means here.**
- `reward_action_rate`: fix, then decide (**rate-follows-bounds**; loss-side alternative **caps-smoothness-loss**). The term prices the clipped, delayed sample. At σ 2 the expected per-dimension |Δa| is 0.95 at |μ| 0 and 0.00 at |μ| 9 (Monte-Carlo run for this review). Saturating μ is therefore the cheapest way to cut the cost, which makes the term a plausible co-cause of the bang-bang mean. The stock form fails replay Gate 2 (14.5–18.3 % share). Decide after M1b. There is no hexapod precedent for removing it [15].
- `reward_delta_torques`: remove (**delta-torque-drop**). It has never been tested alone. It prices all 24 joints, including the Body_Hip emulator. 51 % of Σdτ² falls on the 22 % of steps where a foot crosses the load threshold, so it acts mainly as a tax on touchdown and lift-off.
- `penalty_mechanical_power`: **rod-power-after-clock**. This is the Fu et al. single-regulariser idea [35], but it must be scoped to the rods (24–31 % of Σ|τq̇| is on the emulator joint) and replayed first. `reward_dof_error`: **zero-weight-purge**.

## 8. Terrain and obstacle teacher terms (2a–2c)

**Key papers:** Zhuang et al. [50], Extreme Parkour [16, 17], ANYmal Parkour [28], Rudin 2022 [18], legged_gym [1, 2], Isaac Lab [3], SoloParkour [51], Lee et al. 2020 [13], Spot [6, 7], Kim, Lee, Park [52], Ng et al. [25].

**What they show.** Parkour policies learn to cross obstacles from task progress, a terrain curriculum and generic regularisers [16, 18, 28, 50]. Robot Parkour Learning drops its one terrain-specific term at fine-tuning [50]. Collision is a small count over non-foot links, and body contact is usually a termination [1, 3, 28]. The stumble term uses thresholds k = 2–5 or is off by default [2, 17, 28]. SoloParkour turns both into constraints [51]. Clearance is measured against the terrain under the foot [13, 52] or as a bounded target-height kernel [6, 7]. Indicator-product bonuses are non-potential shaping [25].

**What it means here.**
- `reward_feet_edge`: remove (**zero-weight-purge**). At horizontal scale 0.08 the edge mask needs a 0.12 m riser. The tallest ring block is 0.072 m, and the term paid 0 on all 10,000 teacher iterations.
- `reward_obstacle_clearance` and `reward_tracking_goal_vel`: drop (**ring-bonus-drop**). The bonus indicator is on for 51 % of move-slot steps of the 1c head, which never trained with it. Its share of pay earned over blocks (0.198) equals the share of steps spent over blocks (0.206). Together the two terms add 0.216/s, about a 34 % top-up in all-env means (0.216 against 0.585 + 0.047). The per-ring-tile top-up depends on the flat-tile share, which has not been reported.
- `reward_foot_clearance` and `penalty_swing_min_clearance`: **swing-height-rebase**. The root-relative ground reference makes crouching pay (per-episode ρ −0.30 to −0.42 root-relative, against +0.46 to +0.48 terrain-relative). An earlier 1c-vs-2b apex comparison used the wrong control and a different metric. On a like-for-like reading, lift fell at 2a (1c 0.057/0.084 m vs 2a 0.017/0.060 m), and apex income collapsed inside 2b under a constant 0.5 weight (0.0564 → 0.00014/s), with `foot_clearance` a prime suspect.
- `reward_feet_stumble` and `reward_collision`: **contact-telemetry-audit**. Stumble income is flat across both teacher windows and is priced a second time inside the obstacle bonus. Collision fires on femur contact at 90–120 N, as often on flat tiles as on ring tiles. The repo reads hip/femur contact as leg-leg interference, a hardware constraint, which **contact-telemetry-audit** tests before any change; the default is to keep collision at −2.0.

## 9. Cross-cutting structure

**Key papers:** legged_gym [1, 2], Isaac Lab [53], Reda et al. [54], CaT [9], Kim et al. [10], Lee et al. 2023 [42], Knox et al. [55], Booth et al. [56], Eureka [57], Hwangbo [19], Isaac Lab curricula [58], MultiCriticAL [59], Kim, Lee, Park [52], Andrychowicz et al. [45], van Marum [20], Fu et al. [35].

**What they show.** legged_gym clips negative totals at zero, which its config comment says "avoids early termination problems" [1]; upstream Isaac Lab dropped that clip [53]. The size of the survival bonus decides between a falling-forward minimum and a standing-still minimum [54]. Redundant reward attributes should be removed, and shaping should be reported separately from the true reward [55]. Trial-and-error rewards overfit the algorithm, so a performance drop when one component is removed "may merely reflect" that overfitting [56]. Components that stay near-constant through training are discarded [57], and zero-weight terms are skipped at no compute cost [2, 53]. Hwangbo et al. use one curriculum factor for all costs [19], and Isaac Lab schedules weights in config [58]. Separate critics prevent interference between conflicting tasks [52, 59]. Value normalisation strongly affects performance [45]. Tracking-only rewards produced hopping; the practical floor is tracking plus a small regulariser set [20, 35].

**What it means here.** At the end of X08:c6 the pre-clip return is 43.7 against 52.0 post-clip; at the start of t3 it is −27.0 against +6.4. Every penalty weight is therefore conditional, and every income share and replay gate is pre-clip. That is why **survival-floor-audit** comes first. Survival is bought implicitly by kernels that pay a standing robot, on top of a +1.09/s net economy, so removing positive income (the clock, the ring bonus) shrinks it [54]. Seven effective values disagree between the class body and the presets (six uniformly in every phase; `tracking_yaw` only by phase), and two variables set one parameter (**config-truth-align**). Constraint routes (**cat-style-constraints**, with **soft-pitch-constraint** as its conditional sibling [9]; N-P3O [10, 42]) and grouped critics [59] are structural options, not first steps.

## 10. Deciding "not worse"

**Key papers:** Lakens [60], Everson-Stewart & Emerson [61], Patterson et al. [62], Agarwal et al. [63], Woods & Lewis [64], Ash & Adams [65]; for motion priors, Li et al. [33], Liu et al. [12], AMP [36, 66–69].

A non-significant difference does not show equivalence. Non-inferiority needs a margin fixed in advance, which may be the smallest effect the available budget can detect [60]. Adopting changes one after another, each against the previous adopted control, lets small losses accumulate ("bio-creep") [61]. Five runs are weak evidence, paired seeds are stronger, and ablating components out of an agent tuned with them is a named error [62, 63]. Group screening moves same-sign factors together and splits a group only when it fails [64]. A warm-started network is not equivalent to one trained from scratch [65].

Applied here: the anchored drift test (z ≥ −R, with R_F 3.19 for scratch runs) would pass a removal that loses about 0.14 turn facing_closed (0.618 → about 0.47). The proposal is a declared margin δ_b = 1.5 R_b, passing iff z_b ≥ −0.5 R_b, always measured against a fixed anchor (X0 for forks). Fork seed terms are small on T and Q (R_T 0.517, R_Q 0.319, against 1.131 for scratch R_T) but large on F_mt and diagonal survival (X0 s2 vs s3 differ by 0.27 and 0.23). A fork only answers whether the gait is maintained. Terms whose income peaks during formation, such as foot idle, also need a from-scratch C1d test.

**Own-gait AMP: decline for now.** Two hexapod papers show a style prior replacing hand-built gait terms [12, 33], but both come from one group, and multi-behaviour datasets invite mode collapse [67–69]. The port to rsl_rl is substantial [66]. The eval npz already holds about 1,046–1,191 s of steady frames per hold, but every category is bang-bang (mean |action| 8.1–10.0) and diagonal tripod is 0.396, so a prior trained on this data would learn those defects.

## 11. Literature gaps

- No hexapod or omnidirectional source compares energy against action rate as the only regulariser [15, 35].
- No source pairs a facing reward with a rate tracker on one policy, and none tracks an absolute world facing target through a dead-banded rate command [1–4, 17, 20]. We found no hexapod precedent for keeping three yaw representations.
- The practice of keeping a clock observation after its reward is gone has a repo precedent but no literature test [31].
- The hexapod motion-prior evidence comes from one group, and its self-imitation step is simulation-only [12, 33].
- We found no precedent for any of these: an absolute progress cap far above the command, an event-based crossing-credit tripod reward, a tile-masked second tracking copy, a dense slot-gated facing term, or an r_turn-scaled gait gate.
- The loss-level smoothness evidence comes from drones and humanoids, not hexapods [48, 49].

## 12. Ranked recommendations

The ranking weighs literature evidence, fit to the diagnosis and cost. Every new or re-derived term passes the offline replay gate on saved healthy and degenerate npz before any screen, and every removal is judged against the declared margin in §10.

1. **Cleanup with no change in behaviour: zero-weight-purge, dead-swing-income, dormant-lever-retire, config-truth-align** [2, 38, 53, 57]. Decided by cfg-dump diffs and unit tests; no training arm, and no smoke run (runs are not bit-reproducible, so a smoke cannot test a no-op). Evidence: air time pays 0.0006/s, stride 0, `feet_edge` 0 on every teacher iteration; 12 names sit at 0.0 in all six dumps.
2. **survival-floor-audit** [1, 9, 53, 54]. Count the clipped steps per command slot, then keep, remove or replace the clip as a deliberate choice. Evidence: pre-clip 43.7 vs post-clip 52.0 at the end of c6, and −27.0 vs +6.4 at the start of t3.
3. **The three individual fork screens cross-penalty-remove, posture-penalty-drop and tibia-pose-retire, or minimal-reward-arm if attribution is not needed** [9, 10, 15, 43, 64]. The catalogue screens them one at a time; minimal-reward-arm bundles them (with six other removals) and is cheaper but loses attribution. A grouped screen in the Woods & Lewis sense is not a catalogue candidate. Evidence: income shares of 0.5 %, 0.1 %, 0.05 % and 1.4 %, and the cross penalty taxes the walker more than the creeper.
4. **progress-relative-cap (parity offline now; promoted arm after the clock and ring-bonus-drop), then track-single-kernel** [6, 17, 19, 21]. Evidence: L1 carries 7.7 %, and the 1.05 m/s cap never binds against the omni speed limits.
5. **one-yaw-kernel part A (alternative: turn-slot-facing), then part B, with yaw-progress-phaseout as B's conditional follow-up and continuous-heading-servo after** [1, 4, 17, 25, 26]. Evidence: in C2, facing rose from 0.419 to 0.748 while move_turn survival fell from 0.58 to 0.12. About 0.13 of the 0.150/s facing income is paid on non-turn rows.
6. **ring-bonus-drop, swing-height-rebase, contact-telemetry-audit** [13, 17, 25, 50]. Evidence: the bonus indicator is on for 51 % of move steps even for a head never trained with it; over-block pay 0.198 vs over-block steps 0.206; crouch correlation ρ −0.30 to −0.42.
7. **rate-follows-bounds and delta-torque-drop, after M1b; caps-smoothness-loss as the loss-side alternative** [10, 44, 46, 48]. Evidence: median |μ| 8.1, Gate 2 failure, and 51 % of Σdτ² on 22 % of steps.
8. **foot-idle-audit** [6, 40]. Evidence: −0.236/s at c6; the walker pays −1.747/s against the creeper's −0.441 on the forward hold.
9. **Clock coordination: clock-hold-twin, apex-term-retire, gait-gate-decouple** [6, 23, 31, 33, 34]. Evidence: omni 2c has never been run, and apex pays 0.0001 by 2b.
10. **Before the clock decision, needing a working swing term: minimal-reward-arm as a diagnostic (about 12 GPU-h: a 0.9 GPU-h kill fork plus 3 × 3.6 GPU-h for s3, s2 and the seed-2 incumbent). After the clock goes: mode-time-after-clock, rod-power-after-clock, clockless-tripod-prior. Last, and only if the plain removals fail: cat-style-constraints (with soft-pitch-constraint)** [6, 9, 10, 20, 35]. Evidence: the only clock-free gait income, air time, currently pays 0.

**Where the campaign and the literature disagree.** Quadruped stacks keep vertical-velocity shaping strong [1, 3] while hexapod stacks drop it [15, 43]. PBRS is reported to be robust to scaling [27], yet C2 collapsed at 3×; the implemented term is not exact PBRS. The literature phases cost terms *in* over training [14, 19], whereas the repo phases gait income *out*.

## 13. References

*arXiv references are given by their abs/ page. For [14, 16, 20, 22, 28, 36, 40, 51, 57, 67, 69], verification opened the html/ or pdf/ rendering of the same arXiv ID.*

1. Rudin, Hoeller, Reist, Hutter. legged_gym `legged_robot_config.py`, master. https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot_config.py — reward scales, tracking_sigma 0.25, only_positive_rewards, heading_command, command ranges.
2. Same authors. legged_gym `legged_robot.py`, master. https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot.py — tracking kernels, heading P-control, air-time counter, stumble, stand_still, reward clip, zero-scale pop, terrain curriculum.
3. Isaac Lab. `locomotion/velocity/velocity_env_cfg.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py — RewardsCfg, CommandsCfg, TerminationsCfg.
4. Isaac Lab. `envs/mdp/commands/velocity_command.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py — heading P-control with no dead band.
5. Isaac Lab. `locomotion/velocity/mdp/rewards.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py — feet_air_time (planar gate), feet_slide, stand_still_joint_deviation_l1.
6. Isaac Lab. Spot `mdp/rewards.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py — abs-exp velocity, GaitReward, air_time_reward, foot_clearance, base_motion, action_smoothness.
7. Isaac Lab. Spot `flat_env_cfg.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py — weights.
8. Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes Using Massively Parallel Deep RL.* CoRL 2021. https://arxiv.org/abs/2109.11978 — App. A.3 Table 2.
9. Chane-Sane, Leziart, Flayols, Stasse, Souères, Mansard. *CaT: Constraints as Terminations.* IROS 2024. https://arxiv.org/abs/2403.18765 — reward; Table I; p_max ramp; r ≥ 0.
10. Kim, Oh, Lee, Choi, Ji, Jung, Youm, Hwangbo. *Not Only Rewards But Also Constraints.* IEEE T-RO 2024. https://arxiv.org/abs/2308.12517 — Table I, Sec. V, Table IV.
11. Yang et al. *HeLoM.* arXiv:2509.23651, 2025. https://www.arxiv.org/pdf/2509.23651v1 — Table III.
12. Liu, Wu, Li, Qi, Xue, Gao. *Learning Natural and Robust Hexapod Locomotion … via Motion Priors.* arXiv:2511.03167, 2025. https://arxiv.org/html/2511.03167 — Table I; Sec. III-A; Sec. V-A.
13. Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning Quadrupedal Locomotion over Challenging Terrain.* Science Robotics 2020. https://arxiv.org/abs/2010.11251 — Suppl. S4.
14. Kumar, Fu, Pathak, Malik. *RMA.* RSS 2021. https://arxiv.org/abs/2107.04034 — reward list; curriculum.
15. Qu, Li, Zakhor, Yu, Zhang. *Versatile Locomotion Skills for Hexapod Robots.* arXiv:2412.10628, 2024. https://arxiv.org/html/2412.10628 — Table I.
16. Cheng, Shi, Agarwal, Pathak. *Extreme Parkour with Legged Robots.* ICRA 2024. https://arxiv.org/abs/2309.14341 — reward section.
17. Cheng et al. extreme-parkour `legged_robot.py`, `legged_robot_config.py`, `utils/terrain.py`, main. https://raw.githubusercontent.com/chengxuxin/extreme-parkour/main/legged_gym/legged_gym/envs/base/legged_robot.py — goal-velocity and yaw terms, unsquared action rate, delta_torques, edge, stumble, class-17 masks.
18. Rudin, Hoeller, Bjelonic, Hutter. *Advanced Skills by Learning Locomotion and Local Navigation End-to-End.* IROS 2022. https://arxiv.org/abs/2209.12827 — r_bias removal; r_stall; Sec. V.
19. Hwangbo, Lee, Dosovitskiy, Bellicoso, Tsounis, Koltun, Hutter. *Learning agile and dynamic motor skills for legged robots.* Science Robotics 2019. https://arxiv.org/abs/1901.08652 — Suppl. S3 kernel; curriculum k_c.
20. van Marum, Shrestha, Duan, Dugar, Dao, Fern. *Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking.* IROS 2024. https://arxiv.org/abs/2404.19173 — Table I; Sec. IV-B; Sec. V-C.
21. Gu et al. (RobotEra). humanoid-gym `humanoid_env.py` / `humanoid_config.py`, main. https://raw.githubusercontent.com/roboterax/humanoid-gym/main/humanoid/envs/custom/humanoid_env.py — track_vel_hard, low_speed, scales.
22. Ha, Lee, van de Panne, Xie, Yu, Khadiv. *Learning-based legged locomotion; state of the art and future perspectives.* IJRR 2025. https://arxiv.org/abs/2406.01152 — Sec. 3.3.1.
23. Margolis, Agrawal. *Walk These Ways.* CoRL 2022. https://arxiv.org/abs/2212.03238 — Table 1; Sec. 3.1; Sec. 5.
24. Isaac Lab. `navigation/config/anymal_c/navigation_env_cfg.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/anymal_c/navigation_env_cfg.py — heading error −0.2 above a pretrained velocity policy.
25. Ng, Harada, Russell. *Policy invariance under reward transformations.* ICML 1999. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf — abstract; Sec. 1.
26. Grześ. *Reward Shaping in Episodic Reinforcement Learning.* AAMAS 2017. https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf — Φ(s_N) = 0.
27. Jeon, Heim, Khazoom, Kim. *Benchmarking Potential Based Rewards for Learning Humanoid Locomotion.* ICRA 2023. https://arxiv.org/abs/2307.10142 — abstract.
28. Hoeller, Rudin, Sako, Hutter. *ANYmal Parkour.* Science Robotics 2024. https://arxiv.org/abs/2306.14874 — Tables S2, S6; locomotion reward table.
29. Yang, Gao, Li. *Terrain-adaptive CPGs with RL for Hexapod Locomotion.* arXiv:2310.07744, 2023. https://arxiv.org/abs/2310.07744 — Table I; Reward 2.
30. Jiang, Lin, Li, Shuai. *Bio-Inspired CPG Modulation via Proprioceptive DRL….* Biomimetics 11(8):570, 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13509619/ — reward table; Table 4.
31. Siekmann, Godse, Fern, Hurst. *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition.* ICRA 2021. https://arxiv.org/abs/2011.01387 — Sec. III–V.
32. Unitree. unitree_rl_gym G1 `g1_env.py` / `g1_config.py`, main. https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_env.py — contact term, feet_swing_height.
33. Li, Wu, Liu, Guo, Xue. *Experience-Learning Inspired Two-Step Reward Method….* arXiv:2401.12389, 2024. https://arxiv.org/abs/2401.12389 — Table I; Sec. III; Sec. V-A/B.
34. *Robust Humanoid Walking on Compliant and Uneven Terrain with Deep RL* (HRP-5P). arXiv:2504.13619, 2025. https://arxiv.org/abs/2504.13619 — Sec. V; Table V.
35. Fu, Kumar, Malik, Pathak. *Minimizing Energy Consumption Leads to the Emergence of Gaits.* CoRL 2021. https://arxiv.org/abs/2111.01674 — Sec. 2.4; Figs. 7–8; Sec. 4.2–4.3.
36. Escontrela, Peng, Yu, Zhang, Iscen, Goldberg, Abbeel. *Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions.* IROS 2022. https://arxiv.org/abs/2203.15103.
37. Bellegarda, Ijspeert. *CPG-RL.* RA-L 2022. https://arxiv.org/abs/2211.00458 — Sec. III-C.
38. Isaac Lab PR #7574 (merged 2026-09-07). https://github.com/isaac-sim/IsaacLab/pull/7574 — float32 first-contact tolerance.
39. Chane-Sane, Roux, Stasse, Mansard. *Reinforcement Learning from Wild Animal Videos.* arXiv:2412.04273, 2024. https://arxiv.org/html/2412.04273v1 — air-time ablation.
40. Yu, Turk, Liu. *Learning Symmetric and Low-energy Locomotion.* ACM TOG 2018. https://arxiv.org/abs/1801.08093 — hexapod results.
41. Isaac Lab. ANYmal-D / ANYmal-C `flat_env_cfg.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py — flat_orientation_l2 −5.0.
42. Lee, Schroth, Klemm, Bjelonic, Reske, Hutter. *Evaluation of Constrained RL Algorithms for Legged Locomotion.* arXiv:2309.15430, 2023. https://arxiv.org/abs/2309.15430 — Table II.
43. Penn State LiBR. IsaacLab-LiBRHexapod `hexapod/flat_env_cfg.py`, main. https://raw.githubusercontent.com/psu-LiBR/IsaacLab-LiBRHexapod/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/flat_env_cfg.py.
44. Seyde et al. *Is Bang-Bang Control All You Need?* NeurIPS 2021. https://arxiv.org/abs/2111.02552 — Sec. 5.3, 5.5.
45. Andrychowicz et al. *What Matters In On-Policy RL?* ICLR 2021. https://arxiv.org/abs/2006.05990 — Sec. 3.2–3.3.
46. rl_games `algos_torch/a2c_continuous.py`, master. https://raw.githubusercontent.com/Denys88/rl_games/master/rl_games/algos_torch/a2c_continuous.py — bound_loss.
47. NVIDIA. IsaacGymEnvs `cfg/train/AnymalPPO.yaml` (also AntPPO, AnymalTerrainPPO). https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/train/AnymalPPO.yaml — bounds_loss_coef.
48. Mysore, Mabsout, Mancuso, Saenko. *CAPS.* ICRA 2021. https://arxiv.org/abs/2012.06644.
49. Chen et al. *Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies.* arXiv:2410.11825, 2024. https://arxiv.org/abs/2410.11825.
50. Zhuang et al. *Robot Parkour Learning.* CoRL 2023. https://arxiv.org/abs/2309.05665 — Sec. 3.1, 4.1.
51. Chane-Sane, Amigo, Flayols, Righetti, Mansard. *SoloParkour.* CoRL 2024. https://arxiv.org/abs/2409.13678 — reward; Table 2.
52. Kim, Lee, Park. *Barrier-Based Style Rewards.* ICRA 2025. https://arxiv.org/html/2409.15780v4 — clearance barrier; two critics.
53. Isaac Lab. `managers/reward_manager.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/managers/reward_manager.py — no clip; zero-weight skip.
54. Reda, Tao, van de Panne. *Learning to Locomote.* MIG 2020. https://arxiv.org/abs/2010.04304 — survival bonus.
55. Knox, Allievi, Banzhaf, Schmitt, Stone. *Reward (Mis)design for Autonomous Driving.* AIJ 2023. https://arxiv.org/abs/2104.13906 — Table 1.
56. Booth, Knox, Shah, Niekum, Stone, Allievi. *The Perils of Trial-and-Error Reward Design.* AAAI 2023. https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/booth2023perils.pdf.
57. Ma et al. *Eureka.* ICLR 2024. https://arxiv.org/abs/2310.12931 — Sec. 3.3.
58. Isaac Lab. `envs/mdp/curriculums.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/curriculums.py — modify_reward_weight.
59. Mysore, Cheng, Zhao, Saenko, Wu. *Multi-Critic Actor Learning.* ICLR 2022. https://sidmysore.github.io/publication/2022-01-28-MultiCriticAL-1.
60. Lakens. *Equivalence Tests.* SPPS 2017. https://doi.org/10.1177/1948550617697177 (opened as PMC5502906).
61. Everson-Stewart, Emerson. *Bio-creep in non-inferiority clinical trials.* Stat. Med. 2010. https://doi.org/10.1002/sim.4053 (abstract via Europe PMC).
62. Patterson, Neumann, White, White. *Empirical Design in RL.* JMLR 25, 2024. https://jmlr.org/papers/v25/23-0183.html — Sec. 2.6, 4.4, 7.
63. Agarwal et al. *Deep RL at the Edge of the Statistical Precipice.* NeurIPS 2021. https://arxiv.org/abs/2108.13264.
64. Woods, Lewis. *Design of Experiments for Screening.* 2017. https://arxiv.org/abs/1510.05248 — Sec. 3.1–3.2.
65. Ash, Adams. *On Warm-Starting Neural Network Training.* NeurIPS 2020. https://arxiv.org/abs/1910.08475.
66. Escontrela. AMP_for_hardware (`amp_ppo.py`, `a1_amp_config.py`). https://github.com/escontra/AMP_for_hardware.
67. Vollenweider et al. *Multiple Adversarial Motion Priors.* ICRA 2023. https://arxiv.org/abs/2203.14912 — Sec. II.
68. Huang, Xie, Li. *Conditional Adversarial Motion Priors.* arXiv:2509.21810, 2025. https://arxiv.org/html/2509.21810 — Sec. III-C.
69. Zhang, Li, Miki, Hutter. *Motion Priors Reimagined.* CoRL 2025. https://arxiv.org/abs/2505.16084 — Sec. 1, 6.

## Appendix: unverified or dropped items

- **humanoid-gym heading mode.** The config values are correct [21], but `ang_vel_yaw` [−0.3, 0.3] has no effect in heading mode, because the base P-controller issues up to ±1 rad/s. The humanoid-gym stack is therefore not evidence about small yaw ranges. That claim is dropped; the `track_vel_hard` and `low_speed` facts are kept.
- **Li et al. venue.** The IEEE Xplore listing (document 10802685) returned an empty page, so the conference name is unverified and the paper is cited as arXiv.
- **Jeon et al. authors.** The sweep listed "Jeon, Jiang, Kulić". The catalogue's verification resolved arXiv 2307.10142 to Jeon, Heim, Khazoom and Kim, which is the author list used in [27].
- **Dropped:** a Yang et al. walk → trot → fly-trot claim and a Qu et al. hexapod gait-emergence claim, for which no verified reference was found.
- **Values as rendered:** Liu et al.'s kernel is printed as exp(‖·‖/0.15), so the sign and square are our reading. Qu et al.'s weights (1², −1¹, −5⁻¹) were read from the HTML rendering.
- **Our inferences, not source statements:** the fused-lasso analogy for the unsquared action-rate norm (Tibshirani et al., JRSS-B 67(1), 2005, https://web.stanford.edu/group/SOL/papers/fused-lasso-JRSSB.pdf, opened; the transfer to actions is ours); the Monte-Carlo |Δa| figures (computed for this review); ring risers meshed as ramps (not checked in sim); a slip penalty fighting turning in place; the tibia term acting as a saturation penalty; the six-foot stumble indicator firing more often than on a quadruped; the reference yaw kernel being "sized for" ±1 rad/s; non-exact PBRS as the mechanism behind C2.
- **Repo evidence without URLs:** the walker/creeper replay reproduces the logs only to about 15 % and overestimates foot idle about 2×. The pre-clip vs post-clip returns are estimates because the two sides average differently. The provenance comes from a git weight trace.
- **Conflicting figure:** the share of envs where orientation is masked was reported both as 50 % (the tile mix) and as 20 % on 1a–1c (`KRABBY_FLAT_TERRAIN_FLAT_FRAC` 0.8). We used the preset value.
- **Opened and correct, but omitted for length:** Tan et al. 2018, Peng et al. 2020, Schilling et al. 2020, dm_control quadruped, Hybrid Reward Architecture, Colas et al. 2018, IsaacGymEnvs `ant.py`.
# Literature review: omnidirectional gait formation — command spaces, turning, and the stand-still optimum on a crab-like hexapod

*2026-09-18. Prepared for the `crab_hex_omni_task` campaign after the seed-3 `1a → 1b → 1c` chain
converged to a step-in-place policy, Revision 4 re-priced the reward economy, and the from-scratch
1c window (`seed3_1c_004`, `model_4999.pt`) walked in every direction but missed the formation gate
(completion 0.60; training `track_ratio_mean` 0.568 against a 0.6 floor). Companion to the forward
task's `lit-review-hexapod-reward-stability.md` (its §8 states the shaping theorem reused here).
Every citation was opened at the time of writing; where verification corrected a sweep claim the
corrected wording is used, and items that failed verification are listed in the appendix rather
than softened.*

## 1. Question and scope

Plant A15+B: sprawled lateral legs, tripod gait under a phase clock, PPO in Isaac Lab / RSL-RL at
256 envs. Command = a WORLD travel direction with a speed under a direction-scaled ellipse
(0.35 fore-aft / 0.25 lateral) plus an ABSOLUTE world facing, steered by a P-controller
(stiffness 1.5, dead band 0.133 rad) into a yaw-rate command capped at 0.3 rad/s; slots
stand / move / turn / move+turn. Campaign evidence to explain (`ANALYSIS_1c_failure_2026-09-18.md`,
`REPORT.md`, README Revisions 1–4): no head turned until `reward_yaw_progress` =
clip(wz·sign(cmd_wz), ±0.3) and turn-gated gait shaping were added; the staged chain lost
translation at every handoff (training track ratio 0.747 → 0.433 → 0.245 while mean reward rose
61 → 93; 1c income 34 % clock+apex, 30 % linear kernel, 28 % angular kernel, 1.2 %
motion-requiring; a zero-velocity policy was predicted to earn `track_lin_vel_xy_exp` 0.835,
logged 0.826); Revision 4 (σ² 0.1 → 0.04, L1 −1 → −2, angular std 0.15 → 0.5, clock combine
sum → product) plus a from-scratch window gave speed ratios 0.41–0.61 in all directions and
0.092 / 0.098 rad/s on the turn holds (`turn_l90` settling at 0.49 rad; offline replay margin
+16.6 %, +37.5 % with the angular term gated to turn rows); and the facing-error observation had
to be clipped at the 0.6-rad band or a 1.57-rad target put the policy out of distribution
(100/100 falls → 39 % completion).

## 2. Command spaces and heading control

**The reference convention.** Both stacks the task descends from command a BODY-frame
(vx, vy, wz) triplet resampled every 10 s over ±1 m/s and ±1 rad/s [1, 2, 3, 5], and both implement
"heading mode" as exactly the omni mechanism: a P-controller on wrap_to_pi(ψ* − ψ) saturated into
the yaw-rate slot. legged_gym hard-codes gain 0.5 and a ±1 rad/s clip (`legged_robot.py:330`) [1];
Isaac Lab exposes `heading_control_stiffness` (default 1.0, 0.5 in the locomotion task) and clips
to the configured `ang_vel_z` range [3, 4, 5] — the form `ang_vel_max = 0.3` instantiates. Neither
observes the heading target: legged_gym feeds `commands[:, :3] * commands_scale` (slot 3
excluded; scales 2.0 / 2.0 / 0.25), Isaac Lab returns the 3-vector `vel_command_b` [1, 2, 3, 5]. The
stock policy sees only a bounded, post-saturation yaw-rate image (§6). Verification corrected the
sweep: legged_gym's default IS heading mode (`heading_command = True`) [2]; the hexapod RL papers
sample the rate directly instead (HeLoM ±1 rad/s [39]; Liu et al. [29]); Cassie samples ±π/6 [13].

**World travel plus absolute facing has no single precedent — its halves do.** Extreme Parkour
tracks velocity in the WORLD frame, r = min(⟨v, d̂_w⟩, v_cmd), because base-frame tracking let the
robot game the reward by turning around obstacles — a rationale for the frame, not for
decoupling; its yaw target is the bearing of the same goal vector [8, 9]. Isaac Lab's
`UniformPose2dCommand` decouples only in its non-`simple_heading` branch (world heading sampled
independently of the position target); `simple_heading` re-couples to the bearing or its π-flip,
whichever is nearer [7]. Rudin et al. command a base-frame position plus remaining time, no
heading [10]; ANYmal Parkour adds a yaw target reached by the end of the trajectory, scored
TERMINALLY (weight 5 vs position 10), with a direction-cosine term (+1) and a stall indicator
(−1 when ‖v_b‖ < 0.2) as the only dense motion terms; without the heading command success drops
98.2 → 89.5 % on the terrain needing repeated on-the-spot turns [11]. Yaw-angle [12], joint direction-plus-rotational-speed
[14] and constant-rate [15] command variants exist. Isaac Lab's navigation rewards are
absolute-error only, 1 − tanh(‖des_pos_b‖/std) and |heading_b| [16] — the pay-for-being-there
family the campaign showed cannot teach turning here. Biology backs the decoupling: the largest
comparative crab dataset (50 species) classifies bouts by travel direction relative to the
pre-movement body axis and never measures reorientation [25] — a crab redirects without turning,
and there is no biological turn-rate anchor for `ang_vel_max`.

## 3. Turning in sprawled hexapods

**Mechanism: a left/right differential.** RHex turns while running by perturbing the leg-profile
parameters of contralateral legs in opposite senses with the cycle time identical for all legs,
and in place by contralateral counter-rotation with the tripods still synchronised [17]; the
CPG-RL hexapod scales left/right coxa targets by (1±δt), δt ∈ [−0.25, 0.25] [18]; stick insects
shorten the inside-leg period (front 0.83 vs 1.21 s) [19]. The model-based turn rotates every
stance foot about one centre, θ_max per cycle follows from the leg-length inequality, and even
that controller undershoots its commanded radius (Rs = a·Rref + b, a = 0.68 at duty 1/2) [22].

**Magnitudes.** RHex: 0.19 rad/s at 0.39 m/s while running (radius ≈ 2 m), 0.7 rad/s in place, top
speed reduced while turning [17]. SPINDER (18-DoF CPG, tripod 1.7 Hz): 18.2 ± 1.4 °/s = 0.32 rad/s
in place, sagittal 0.14 / lateral 0.08 m/s [20] (re-read from the Nature body text for this review; appendix). The budget relation
behind the plan's estimate is explicit in OpenSHC, v_body = l_s·f_s/β with stride bounded by the
walkspace and frequency by joint-velocity limits [23]; a turn sweeps each stance foot through
r_foot·θ, so ω_max ≈ v_foot,max / r_foot — at r_foot ≈ 1.2 m and a 0.18–0.22 m/s foot-speed ceiling,
0.15–0.18 rad/s, the plan's number. The 0.3 rad/s cap is therefore ~2× above the plant, as the
0.35 m/s fore-aft cap is 1.6–1.9× above the measured 0.19–0.22 m/s.

**Twist coupling.** Model-based omnidirectional controllers ADD translation and rotation foot
displacements (SPINDER p* = p + Δr_torso·s_swing + Δr_rot [20]; Yue et al.'s ΔL + ΔR [14]), so a
simultaneous command demands up to |v| + r·|ω| of outer-foot speed. The omni clock uses
max(‖cmd_xy‖, 1.2·|wz|), and the `move_turn` hold (0.22 m/s under a 0.3 rad/s cap) asks for
≈ 0.58 m/s — 2.6–3.2× the foot-speed ceiling, infeasible before any policy question. No published
controller drives cadence from a twist norm [20, 23].

**Anisotropy.** SPINDER's lateral/sagittal ratio is 0.57 [20]; Chen et al.'s three-joint hexapod
walks sideways 75 % faster at 40 % lower cost of transport [21]; decapod joints articulate in
perpendicular planes and sideways stride is longer because neighbouring legs do not obstruct
[24]. The sign is morphology-specific: the inherited 0.35 / 0.25 ellipse should be measured. The
from-scratch 0.09–0.10 rad/s is about half the kinematic ceiling; a 90° turn at 0.15 rad/s needs
~10 s, so the 0.49-rad residual on `turn_l90` reads as rate-limited.

## 4. Reward economies and the stand-still optimum

**The kernel.** exp(−‖cmd − v‖²/σ²) is the literal reference form: Rudin's
φ(x) = exp(−‖x‖²/0.25) [26], `tracking_sigma = 0.25` [1, 2], Isaac Lab `std = sqrt(0.25)` [5].
Its standstill income exp(−‖cmd‖²/σ²) is 1.8 % of max at 1 m/s but 78 % at 0.25 m/s (54 % at
σ² 0.1, 21 % at 0.04). legged_gym hides the exploit by commanding up to 1 m/s; a 0.21–0.35 m/s
plant cannot, so σ² 0.04 is a plant-bound departure (published hexapod practice
stops at 0.15 [29]). Margolis et al. document the same
pathology from the command side: with commands sampled wider than the policy can achieve,
learning fails and the heat-maps show a robot "jittering in place" (Margolis et al. [40]) — a
rival cause to the reward-economy story, since both omni caps sit above the plant.

**Zero-at-standstill forms vs the rest.** Only two cited designs remove the optimum by
construction: Extreme Parkour's clipped world-frame progress min(⟨v, d̂⟩, v_cmd)/v_cmd, the only
positive translation term in its reward file [8, 9], and Qu et al.'s clip(v_x, ±0.4) [28]. Lee
et al.'s projection form exp(−2(v_pr − 0.6)²) / 1.0 / 0.0 zeroes income only under a zero command
and still pays 0.49 at v = 0 under a move command [27]; the Miki et al. form pays MORE at
standstill than the Rudin kernel (appendix). Hwangbo et al. name the optimum — heavy motion
penalties make standing a good local minimum — and anneal every cost except the objective terms,
k_c ← k_c^0.997 from k_0 = 0.3 [34]; Reda et al. show the survival bonus selects between falling and
standing minima [35]. Rudin et al.'s position-target
formulation needed r_stall = −1 (‖ẋ‖ < 0.1 m/s while > 0.5 m from target) plus a temporary
direction-cosine bonus [10], and ANYmal Parkour keeps the stall indicator [11]. Kim et al. report that
6–14-term additive rewards do not transfer across robots without re-tuning [36] — the 36-term
omni stack inherits that fragility.

**Gait income.** Two families. Pure-cost shaping: Siekmann's phase coefficients in {−1, 0}
multiply force/speed costs, standing entered through a smooth sigmoid on the swing ratio [30];
Walk These Ways' phase terms at −0.08 under r_task·exp(0.02·r_aux), so zero task reward implies
zero total [31]. Positive-but-command-gated: legged_gym `feet_air_time` at +1.0 behind ‖cmd‖ > 0.1,
commands under 0.2 snapped to zero [1]; Isaac Lab air time +0.125 behind the same gate,
`stand_still_joint_deviation_l1` at 0.06, 2 % standing envs [3, 5, 32]. The second family — where
the omni `_cmd_gate` lives — is fully farmable by stepping in place under a move command, which is
the 1c ledger. Lee et al. [27] and van Marum et al. [33] treat stand as its own regime (gait generator
stopped after 0.5 s; contact and air-time terms held constant). Revision 4's
product clock moves toward the first family; the replay's remaining leak — the even angular
kernel paying a stander ~0.98/s — has the same shape.

**Yaw shaping.** `reward_yaw_progress` is potential-based shaping on Φ = −|wrap(ψ* − ψ)| with
γ = 1: outside the dead band sign(cmd_wz) = sign(e), so wz·sign(cmd_wz) = dΦ/dt, the form Ng et al.
prove necessary and sufficient for policy invariance; a full revolution nets zero [37]. Their
bicycle counter-example (progress paid, retreat unpunished, tiny circles) is the shape of the
clamp(min=0) translation twin. Jeon et al. find PBRS buys weight-robustness, not speed [38]: the
smoke's jump from no rotation to 0.086 rad/s was a signed gradient replacing even functions, so
yaw_progress at 0.3 % of income is a matter of its ±0.3 clip and dead band (reachable 0.078/s),
not its weight. The angular kernel now matches reference width (std 0.5 = √0.25) but keeps weight
1.0 where legged_gym, Isaac Lab and HeLoM pay angular at half the linear weight [2, 5, 39].

## 5. Curricula, priors, forgetting, symmetry

Wołczyk et al. name the handoff mechanism: a fine-tuned RL model degrades on the downstream state
subspace not visited early in fine-tuning, commonly and often catastrophically; remedies are EWC,
kickstarting, episodic memory or BC on a pre-training buffer [42]. Gai et al. reproduce it on
quadruped task sequences [43], Dohare et al. show plasticity loss under long continual training
(supervised, 89 % → 77 %) [44]; Rudin et al. loop max-level robots back to random terrain levels [26].
Joint alternatives: DeepMimic's max-over-clips objective with a one-hot selector [45], Multi-AMP's
per-style discriminators [46], specialists distilled into one policy [47]. Command curricula are
competence-gated (legged_gym widens lin_vel_x when episodic tracking exceeds 80 % of max — a
per-axis band ramp [1]; Margolis / WTW gate a joint (vx, wz) grid at 0.5 m / 0.5 rad because
independent draws create infeasible pairs at high speed [40, 31]), iteration-scheduled
(OmniMimic [41]), or absent (Li et al.: 8-s resampling, first command forward [13]).

Priors seed only what they cover: DeepMimic without RSI never completes the backflip [45]; the
hexapod prior set is 8.6 s of trajectory-optimised forward, backward, lateral, steering and
combined motion [29]; OmniMimic augments narrow demonstrations by time reversal, dynamics
completion and sagittal reflection because they under-supervise backward, lateral and turning
[41] (unrefereed). AMP swaps 13 hand-designed style terms for 4.5 s of mocap [48]. The omni RSI bank (forward + sideways frames, |wz| ≈ 0.003 rad/s)
contains no turn or backward state.

Symmetry: Mittal et al. use the same 4-element group (identity, reflect-x, reflect-y, rot-180)
and find augmentation converges faster than a mirror loss [50]; Abdolhosseini et al. find no
consistent speed effect, and that a symmetric policy cannot leave a neutral pose — worst under
architectural enforcement, with non-neutral resets as the workaround [51]. Su et al. show the
PPO-side case for augmentation assumes an approximately symmetric policy, secured only by
zero-mean initialisation — unavailable when resuming a single-axis head — and that augmentation
alone still favoured one leg [52]. Normalisation statistics must include mirrored states [51]; rsl_rl's
normaliser never sees the augmented images [53] — moot here (`empirical_normalization = False`),
which is also why nothing compresses an out-of-band value.

## 6. Observation encoding and train/eval shift

The reference stacks never observe a raw heading error (§2) and treat clipping as a declared
pipeline stage — Isaac Lab `ObservationTermCfg.clip` after noise, before scale [6]; legged_gym's
global clip at 100 [1] — so Revision 3's clip is in-convention. But it leaves channels 6/7 as an
affine image of yaw-rate slot 5 over most of the band and blind beyond 0.6 rad, which is the
burst-then-stall signature. Bounded alternatives with precedent: a wrapped base-frame scalar paired
with a base-frame position [7]; a base-frame target plus remaining time [10]; a local-frame unit
direction plus speed (AMP's target-heading task) [49] — a (cos, sin) pair keeps magnitude to 180°
and sign-flips cleanly under both mirrors. Ciebielski et al. document the failure genus: a biped
trained only forward/backward could not track a dense direction grid at 0.5 m/s, survivability
shrank outside the training box, and redundant goal information worsened out-of-distribution
behaviour [54] — the omni observation carries the facing error twice plus its derived rate plus
history. The eval-side genus is open upstream: Isaac Lab issue #3836 is a play script silently
dropping the observation normaliser [55] — the driver dropping `KRABBY_OMNI_FACING_OBS_CLIP` is the same defect class.

## 7. Evaluation protocols

There is no standard omnidirectional command-following benchmark and yaw is the least-measured
axis: Gait in Eight drops the yaw target as common sim-to-real practice and rates yaw poor in its
omnidirectional row (x 0.25 / y 0.15 m/s against U(−0.5, 0.5)) [59]; van Marum et al. note prior
work mostly reports video [33]. Reusable elements: per-direction error under a stressor, 1,000
robots (MOVE [56]); a command-grid sweep with per-cell RMS over 5 trials and a command-area scalar
[40] (WTW's heat-maps are mean task reward, not error [31]); turning scored by angular error AND
lateral drift from a 2-ft circle at 0.5 rad/s for 1, 5 and 30 s [33]; 10-minute random-command
runs, five seeds, paired t-test [57]; one scalar with time and completion penalties [58]. Isaac
Lab's `error_vel_xy` / `error_vel_yaw` are episode-accumulated env means that pool stand with move
rows [3]; Margolis evaluate outside the trained range on tracking error, never as a survival gate
[40].

## 8. Implications for the omni task (ranked)

1. **Size the caps to the plant before the next window** (§3; [17, 20, 23, 40]). Both
   `ang_vel_max` 0.3 and the 0.35 fore-aft cap sit above the measured envelope. Run a from-scratch per-direction
   ceiling probe on BOTH axes [21, 24], set the ellipse from it, screen `ang_vel_max` 0.15–0.2, and
   lengthen 90° holds to ≥ 10 s or score rate on short holds and accuracy on long ones [33].
2. **Make the move+turn slot feasible** (§3; [14, 20]). Drive the clock from ‖cmd_xy‖ + r_turn·|wz|
   and scale the ellipse down when a turn is co-commanded.
3. **Promote the zero-at-standstill progress term** (§4; [8, 9, 28, 10, 11]). Raise
   `reward_forward_progress_along_command` toward the primary tracking role with a
   min(·, cmd_speed) cap, make it signed [37], and add a stall indicator (−1 when ‖v_b‖ < 0.1–0.2
   under a move command); keep the exponential bounded [34].
4. **Gate gait and angular income on achieved motion, not the command** (§4; [30, 31, 33]). Arm
   `KRABBY_OMNI_ANG_TURN_ONLY` (replay +16.6 → +37.5 %) or halve the angular weight to 0.5 [2, 5];
   make stand a regime that pays for NOT stepping [27, 33] or cut the stand share toward 0.02.
5. **Prefer the from-scratch window; if resuming, measure the handoff** (§5; [42, 52]). Log KL
   and action std over the first 200 iterations after each resume, or ramp the mirror images
   1 → 2 → 4; gate every stage on per-direction retention [26].
6. **Seed turning and backward** (§5; [45, 29, 41]). Harvest turn and backward frames from the
   yaw-smoke and from-scratch heads into the RSI bank; an AMP prior on the current bank inherits
   the same coverage gap.
7. **Encode facing as (cos, sin) or drop the raw channels** (§6; [1, 3, 7, 49, 54]). Observe only
   the saturated yaw-rate command as both reference stacks do, or repurpose channels 6/7 as a unit
   vector; keep the clip and the zero-diff pytest meanwhile [55].
8. **Evaluation** (§7): a polar (direction × speed × |Δψ|) grid with a command-area scalar,
   per-direction `speed_ratio` with absolute m/s error, planar drift on pure-turn holds, n / IQR
   with per-seed repetition; never a survival gate on out-of-band commands [40, 54].

Where campaign and literature disagree: reference kernel defaults assume ±1 m/s commands; the
dense yaw-progress term has no ANYmal-lineage precedent but is theoretically sound [37]; the
recommended 4-image augmentation [50] assumes a from-scratch start [52].

## 9. References

1. Rudin, Hoeller, Reist, Hutter (ETH RSL). legged_gym `legged_robot.py`, master, opened 2026-09-18. https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot.py — L330 heading→yaw-rate clip; L351 small-command zeroing; L872–880 tracking kernels; L886 air-time gate; L900–902 stand_still; L99–100 obs clip; only_positive_rewards in compute_reward.
2. Same authors. legged_gym `legged_robot_config.py`, master. https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot_config.py — commands class (num_commands 4, resampling 10 s, heading_command True, ranges ±1), rewards.scales, tracking_sigma 0.25, normalization scales, clip_observations 100.
3. Isaac Lab. `envs/mdp/commands/velocity_command.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py — L143–163 heading P-control clipped to ranges.ang_vel_z, standing-env zeroing; command property returns the 3-vector; `_update_metrics` accumulation.
4. Isaac Lab. `envs/mdp/commands/commands_cfg.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/commands/commands_cfg.py — `UniformVelocityCommandCfg` L33–82 (heading_command, heading_control_stiffness default 1.0, Ranges).
5. Isaac Lab. `locomotion/velocity/velocity_env_cfg.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py — L94–105 base_velocity cfg (stiffness 0.5, rel_standing_envs 0.02); RewardsCfg track_lin/ang (1.0 / 0.5, std √0.25); feet_air_time 0.125; height_scan clip.
6. Isaac Lab. `managers/manager_term_cfg.py` L172–182 (ObservationTermCfg.clip / scale) and docs. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/managers/manager_term_cfg.py ; https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html
7. Isaac Lab. `envs/mdp/commands/pose_2d_command.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/commands/pose_2d_command.py — heading_command_b = wrap_to_pi(heading_command_w − heading_w); simple_heading flip branch; uniform-heading else branch.
8. Cheng, Shi, Agarwal, Pathak. *Extreme Parkour with Legged Robots.* ICRA 2024; arXiv:2309.14341. https://arxiv.org/abs/2309.14341 — Sec. 3.1 Eq. (1)–(2) world-frame tracking and its anti-gaming rationale; Sec. 4.3 NoInner ablation; Eq. (4) r_stylized.
9. chengxuxin/extreme-parkour `legged_robot.py`, main. https://raw.githubusercontent.com/chengxuxin/extreme-parkour/main/legged_gym/legged_gym/envs/base/legged_robot.py — `_reward_tracking_goal_vel`, `_reward_tracking_yaw`, `_update_goals` target_yaw.
10. Rudin, Hoeller, Bjelonic, Hutter. *Advanced Skills by Learning Locomotion and Local Navigation End-to-End.* IROS 2022; arXiv:2209.12827. https://arxiv.org/abs/2209.12827 — Eq. (1) terminal r_task (T 6 s, T_r 1 s); Eq. (3) r_bias; Eq. (4) r_stall; base-frame target + remaining time; +80 % / +35 % vs velocity tracking; discussion on unused kinematic symmetry.
11. Hoeller, Rudin, Sako, Hutter. *ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots.* arXiv:2306.14874 (Science Robotics 2024). https://arxiv.org/abs/2306.14874 — heading target; Table S2 locomotion rewards; Table S5/S6 heading ablation; S3 symmetry augmentation trick.
12. Li, Peng, Abbeel, Levine, Berseth, Sreenath. *RL for Versatile, Dynamic, and Robust Bipedal Locomotion Control.* IJRR (online 2024), doi 10.1177/02783649241285161; arXiv:2401.16889. https://arxiv.org/abs/2401.16889 — c_walk with a desired yaw angle.
13. Li, Cheng, Peng, Abbeel, Levine, Berseth, Sreenath. *RL for Robust Parameterized Locomotion Control of Bipedal Robots.* ICRA 2021; arXiv:2103.14295. https://arxiv.org/abs/2103.14295 — 4-D command sampled every 8 s, yaw ±π/6, first command forward/zero yaw.
14. Yue, Jiang, Zhang, Zhang. *Research on Omnidirectional Gait Switching and Attitude Control in Hexapod Robots.* Biomimetics 9(12):729, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11672882/ (doi 10.3390/biomimetics9120729) — fuzzy inputs D ∈ [−180°, 180°], R ∈ ±30°/s; ΔL + ΔR vector composition.
15. Xu, Raj, Xiao, Stone. *Dexterous Legged Locomotion in Confined 3D Spaces with RL.* arXiv:2403.03848. https://arxiv.org/html/2403.03848v1 — constant 1.57 rad/s yaw command toward the target.
16. Isaac Lab. `manager_based/navigation/mdp/rewards.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/mdp/rewards.py — position_command_error_tanh, heading_command_error_abs.
17. Saranlı. *Dynamic Locomotion with a Hexapod Robot.* PhD dissertation, University of Michigan, 2002. https://user.ceng.metu.edu.tr/~saranli/documents/saranli_thesis_2up.pdf — Ch. 2: 0.19 rad/s at 0.39 m/s, ~2 m radius, 0.7 rad/s in place; §2.3.2 differential-perturbation turning.
18. Jiang, Lin, Li, Shuai. *Bio-Inspired CPG Modulation via Proprioceptive DRL for Adaptive Hexapod Locomotion.* Biomimetics 11(8):570, 2026 (doi 10.3390/biomimetics11080570). https://pmc.ncbi.nlm.nih.gov/articles/PMC13509619/ — coxa scaling (1 ± δt), δt ∈ [−0.25, 0.25]; heading penalty −2.0, yaw-rate tracking 0.2.
19. Gruhn, Rosenbaum, Bockemühl, Büschges. *Body side-specific control of motor activity during turning in a walking animal.* eLife 5:e13799, 2016. https://elifesciences.org/articles/13799 — inside/outside step periods.
20. Maoudj, Saputra, Brahmi, Nouioua. *SPINDER: an open-source 18-DoF hexapod robot with hierarchical CPG control and analytic IK.* Scientific Reports, 2026. https://www.nature.com/articles/s41598-026-56856-0 (PMC13490506) — Tables 4–5 (0.14 / 0.12 / 0.08 m/s; 18.2 ± 1.4 °/s; 1.7 Hz); Eq. (36)–(37) stance compensation; Eq. (29) v_cmd-scheduled CPG weights.
21. Chen, Grezmak, Graf, Daltorio. *Sideways crab-walking is faster and more efficient than forward walking for a hexapod robot.* Bioinspir. Biomim. 17(4):046001, 2022. https://iopscience.iop.org/article/10.1088/1748-3190/ac6847 — +75 % speed, −40 % COT.
22. Zhu, Guo, Liu, Zhu, Zhao, Jin. *Turning and Radius Deviation Correction for a Hexapod Walking Robot Based on an Ant-Inspired Sensory Strategy.* Sensors 17(12):2710, 2017. https://doi.org/10.3390/s17122710 (PMC5751548) — rotation-about-R stance transform, θ_max constraint, Rs = a·Rref + b table.
23. Tam, Talbot, Steindl, Elfes, Kottege. *OpenSHC: A Versatile Multilegged Robot Controller.* IROS 2020; arXiv:2006.04424. https://arxiv.org/abs/2006.04424 — v_body = l_s·f_s/β; walkspace-limited stride; joint-velocity-limited frequency.
24. Vidal-Gadea, Rinehart, Belanger. *Skeletal adaptations for forwards and sideways walking in three species of decapod crustaceans.* Arthropod Struct. Dev. 37:95–108, 2008. https://doi.org/10.1016/j.asd.2007.06.002 — §4.10; Table 3 joint ROM.
25. Taniguchi et al. *Evolution of sideways locomotion in crabs.* eLife reviewed preprint RP110015, 2026. https://elifesciences.org/reviewed-preprints/110015 (PMC13521562) — bout direction bins, FSI, 50 species.
26. Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes Using Massively Parallel DRL.* CoRL 2021, PMLR 164; arXiv:2109.11978. https://arxiv.org/abs/2109.11978 — Table 2 φ(x) = exp(−‖x‖²/0.25); terrain-curriculum loop-back.
27. Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning quadrupedal locomotion over challenging terrain.* Science Robotics 5(47):eabc5986, 2020; arXiv:2010.11251. https://arxiv.org/abs/2010.11251 — S4 projection reward; FTG f0 → 0 after 0.5 s zero command.
28. Qu, Li, Zakhor, Yu, Zhang. *Versatile Locomotion Skills for Hexapod Robots.* arXiv:2412.10628, 2024. https://arxiv.org/abs/2412.10628 — Table I clip(v_x, ±0.4) as sole positive term.
29. Liu, Wu, Li, Qi, Xue, Gao. *Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion Priors.* arXiv:2511.03167, 2025. https://arxiv.org/abs/2511.03167 — Table I exponentials at 0.15; §IV.B direct yaw-velocity command; 8.6-s prior dataset.
30. Siekmann, Godse, Fern, Hurst. *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition.* ICRA 2021; arXiv:2011.01387. https://arxiv.org/abs/2011.01387 — Sec. IV coefficients; standing sigmoid on r_swing.
31. Margolis, Agrawal. *Walk These Ways.* CoRL 2022, PMLR 205; arXiv:2212.03238. https://arxiv.org/abs/2212.03238 — r_task·exp(c_aux r_aux), Table 1 weights; Table 4 heat-maps; Table 7 curriculum thresholds 0.8 / 0.7.
32. Isaac Lab. `locomotion/velocity/mdp/rewards.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py — feet_air_time gate > 0.1; stand_still_joint_deviation_l1 threshold 0.06.
33. van Marum, Shrestha, Duan, Dugar, Dao, Fern. *Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking.* IROS 2024; arXiv:2404.19173. https://arxiv.org/abs/2404.19173 — Sec. IV-B stand-command constants; benchmark: rotation error + drift from a 2-ft circle at 0.5 rad/s for 1/5/30 s.
34. Hwangbo et al. *Learning agile and dynamic motor skills for legged robots.* Science Robotics 4(26):eaau5872, 2019; arXiv:1901.08652. https://arxiv.org/abs/1901.08652 — standing local minimum; k_c curriculum (k_0 0.3, k_d 0.997); bounded logistic kernel.
35. Reda, Tao, van de Panne. *Learning to Locomote: Understanding How Environment Design Matters for DRL.* MIG 2020; arXiv:2010.04304. https://arxiv.org/abs/2010.04304 — Sec. 9 survival bonus, Fig. 9.
36. Kim et al. *Not Only Rewards But Also Constraints.* IEEE T-RO 2024; arXiv:2308.12517. https://arxiv.org/abs/2308.12517 — reward-term counts 6–14; Raibo → ANYmal C transfer failure.
37. Ng, Harada, Russell. *Policy invariance under reward transformations.* ICML 1999, pp. 278–287. https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf — Theorem 1; bicycle example.
38. Jeon, Heim, Khazoom, Kim. *Benchmarking Potential Based Rewards for Learning Humanoid Locomotion.* ICRA 2024; arXiv:2307.10142. https://arxiv.org/abs/2307.10142 — marginal speed-up, scaling robustness, γ = 1 in practice.
39. Yang et al. *HeLoM: Hierarchical Learning for Whole-Body Loco-Manipulation by a Hexapod Robot.* arXiv:2509.23651, 2025. https://arxiv.org/abs/2509.23651 — commands [0, 0.5] × [−0.3, 0.3] m/s × [−1, 1] rad/s; Table III kernels at 0.25, weights 2dt / 1dt.
40. Margolis, Yang, Paigwar, Chen, Agrawal. *Rapid Locomotion via RL.* RSS 2022 (IJRR 43(4), 2024); arXiv:2205.02824. https://www.roboticsproceedings.org/rss18/p022.pdf ; https://arxiv.org/abs/2205.02824 — Sec. III-D grid curriculum, Fig. 3 no-curriculum jitter; command-area evaluation.
41. Wu et al. (10 authors). *OmniMimic: Dynamics-completed Motion Augmentation for Multi-style Omnidirectional Quadruped Locomotion.* arXiv:2609.20566, 2026-09-17 (preprint). https://arxiv.org/abs/2609.20566 — augmentation motivation; iteration-scheduled command widening; mirror sign table.
42. Wołczyk et al. *Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem.* ICML 2024; arXiv:2402.02868. https://arxiv.org/abs/2402.02868 — abstract; retention methods.
43. Gai, Lyu, Zhang, Wang. *Continual RL for Quadruped Robot Locomotion.* Entropy 26(1):93, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11154561/
44. Dohare, Hernandez-Garcia, Rahman, Mahmood, Sutton. *Maintaining Plasticity in Deep Continual Learning.* arXiv:2306.13812 (Nature 632:768–774, 2024, as *Loss of plasticity in deep continual learning*, with Lan). https://arxiv.org/abs/2306.13812
45. Peng, Abbeel, Levine, van de Panne. *DeepMimic.* ACM TOG 37(4), 2018; arXiv:1804.02717. https://arxiv.org/abs/1804.02717 — RSI section and backflip ablation; multi-clip max objective.
46. Vollenweider, Bjelonic, Klemm, Rudin, Lee, Hutter. *Advanced Skills through Multiple Adversarial Motion Priors in RL.* ICRA 2023; arXiv:2203.14912. https://arxiv.org/abs/2203.14912
47. Zhuang et al. *Robot Parkour Learning.* CoRL 2023; arXiv:2309.05665. https://arxiv.org/abs/2309.05665
48. Escontrela, Peng, Yu, Zhang, Iscen, Goldberg, Abbeel. *Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions.* IROS 2022; arXiv:2203.15103. https://arxiv.org/abs/2203.15103
49. Peng, Ma, Abbeel, Levine, Kanazawa. *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control.* ACM TOG 40(4), 2021; arXiv:2104.02180. https://arxiv.org/pdf/2104.02180 — target-heading goal g_t = (d̃*, v*).
50. Mittal, Rudin, Klemm, Allshire, Hutter. *Symmetry Considerations for Learning Task Symmetric Robot Policies.* ICRA 2024; arXiv:2403.04359. https://arxiv.org/abs/2403.04359 — per-task symmetry groups; augmentation vs mirror loss.
51. Abdolhosseini, Ling, Xie, Peng, van de Panne. *On Learning Symmetric Locomotion.* MIG 2019, doi 10.1145/3359566.3360070. https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/2019-MIG-symmetry.pdf — §4.5 normalisation; §7 learning speed; §8 neutral-pose hazard.
52. Su et al. *Leveraging Symmetry in RL-based Legged Locomotion Control.* IROS 2024; arXiv:2403.17320. https://arxiv.org/abs/2403.17320 — Sec. IV-A off-policy caveat and zero-mean init; Sec. V one-leg reliance.
53. rsl_rl `modules/normalization.py` (EmpiricalNormalization), main = installed rsl-rl-lib 5.0.1. https://raw.githubusercontent.com/leggedrobotics/rsl_rl/main/rsl_rl/modules/normalization.py — lifetime cumulative update, buffers in state_dict.
54. Ciebielski, Burgio, Khadiv. *Contact-conditioned learning of multi-gait locomotion policies.* arXiv:2408.00776, 2024. https://arxiv.org/abs/2408.00776 — Fig. 4/5 out-of-distribution direction grid and survivability.
55. isaac-sim/IsaacLab issue #3836, *Incorrect handling of rsl-rl observation normalizer.* https://github.com/isaac-sim/IsaacLab/issues/3836
56. Li, Luo, Wu, Zhu. *MOVE: Multi-skill Omnidirectional Legged Locomotion with Limited View in 3D Environments.* arXiv:2412.03353, 2024. https://arxiv.org/abs/2412.03353 — four-direction tracking error, 1,000 robots.
57. Nahrendra, Yu, Myung. *DreamWaQ.* ICRA 2023; arXiv:2301.10602. https://arxiv.org/abs/2301.10602 — 10-min random-command protocol, five seeds, paired t-test.
58. Caluwaerts et al. *Barkour: Benchmarking Animal-level Agility with Quadruped Robots.* arXiv:2305.14654, 2023. https://arxiv.org/abs/2305.14654 — R_agility formula.
59. Bohlinger, Kinzel, Palenicek, Antczak, Peters. *Gait in Eight: Efficient On-Robot Learning for Omnidirectional Quadruped Locomotion.* arXiv:2503.08375, 2025. https://arxiv.org/abs/2503.08375 — yaw omitted; Table II omnidirectional row.

## Appendix: unverified or dropped items

- **Miki et al., Science Robotics 7(62):eabk2822, 2022 (arXiv:2201.08117).** Exists and was opened;
  DROPPED as support for the stand-still claim. Its command-following reward
  (1.0 if v_des·v > |v_des|, else exp(−(v_des·v − |v_des|)²); exp(−|v|²) at zero command) pays
  exp(−|v_des|²) at v = 0 — 0.94 at a 0.25 m/s command, 0.37 at 1 m/s — MORE standstill income
  than the Rudin kernel. It mitigates by projection, saturation and a zero-command branch only.
- **SPINDER in-place yaw 50.4 ± 2.3 °/s (gap sweep).** Contradicted by this review's own read of
  the Nature body text (18.2(1.4) °/s) and by the C7 verifier's read of the PMC inline images;
  the number lives only in images, and text-stripping fetches have confabulated it twice
  (36.4 and 50.4). The gap sweep's ω ≈ v/r_foot consistency arithmetic built on 50.4 is dropped;
  at 0.32 rad/s SPINDER sits well below that bound, consistent with its stated hip-yaw-range limit.
- **Isaac Lab `pose_2d_command.py` `error_heading` as a resample-window average (sweep C12).**
  The metric is ASSIGNED per step, |wrap_to_pi(heading_command_w − heading_w)|, not accumulated;
  only `velocity_command.py`'s `error_vel_xy` / `error_vel_yaw` accumulate (over the whole episode,
  normalised by one window's step count). The pooled-metric point in §7 rests on [3] alone.
- **Universal claims from the sweep that verification rejected, not used here:** that deployed
  systems avoid the stand-still optimum structurally (only the clipped-progress forms do); that
  gait shaping is never farmable in the reference designs (legged_gym and Isaac Lab air-time
  income is positive and command-gated); that published command curricula are joint grids rather
  than band ramps (legged_gym is a per-axis ramp; Li et al. have none; OmniMimic is
  iteration-scheduled); that symmetry enforcement should not be expected to speed learning
  (Mittal et al. report faster convergence); that the omni angular kernel is 11× sharper than
  every default (true of the class-body literal only; every preset has used std 0.5 since
  Revision 4); and that Cassie's ±π/6 is a third of legged_gym's range (it is half).
- **Metadata corrections carried into the list:** Jiang et al. is 2026, not 2025; Dohare et al.'s
  arXiv title is *Maintaining Plasticity…* and the Nature version *Loss of plasticity…* (Lan
  added); Li et al. IJRR is online-first 2024, volume year 2025; Multi-AMP's ICRA 2023 venue was
  confirmed through citing works, not IEEE Xplore; Mittal et al.'s IEEE DOI page returned an empty
  body behind the paywall (arXiv used); Extreme Parkour's ICRA 2024 venue is not in the arXiv
  comments field; nature.com 303-redirects command-line fetches for SPINDER but renders in a
  browser, and PMC rate-limited this client during the check. Liu et al.'s Table I prints the
  exponent without a minus sign and as an unsquared L2 norm, so its 0.15 is not on the same
  footing as the σ² values in §4.
- **Unrefereed sources used with that caveat:** OmniMimic [41] (arXiv, 2026-09-17); Taniguchi
  et al. [25] (eLife reviewed preprint); Liu et al. [29], Qu et al. [28], HeLoM [39], MOVE [56],
  Gait in Eight [59], Ciebielski et al. [54], Xu et al. [15] (arXiv preprints).

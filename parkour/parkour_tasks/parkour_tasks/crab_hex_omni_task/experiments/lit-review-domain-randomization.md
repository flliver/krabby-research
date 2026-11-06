# Literature review: domain randomization for sim-to-real transfer -- mapped to the Krabby campaigns

*2026-11-06. Prepared after the Task 4 pipeline close (F19 = omni policy of record,
`crab_hex_omni_task/experiments/2026-10-16_task4_pipeline/REPORT_baseline_vs_final.md`, 2026-10-24; PR #10
merged 2026-10-27) and the SEN0581 camera and site-teacher campaigns (closed 2026-10-28 and 2026-10-31), with
no hardware yet. The effective DR inventory it argues against was verified against the per-run
`params/env.yaml` dumps of every bundled stage (scratchpad `dr_inventory_F19.md`). Builds on
`lit-review-rl-algorithm-and-ppo-hyperparameters.md` (batch, σ, KL; not repeated), `lit-review-stand-network.md`
and `crab_hex_forward_task/experiments/lit-review-hexapod-reward-stability.md`. Every citation was opened by
the review's verification pass and checked against the paper's abstract or full text, or the raw source file,
at time of writing; where verification corrected a sweep claim the corrected wording is used, and items that
failed verification or are only inferred are in the appendix rather than softened.*

## 1. The problems this review targets

Restated from the inventory and the Task 4 ledgers so this document stands alone:

1. **The plant.** 230 kg (body 158.8 kg + 6 × 11.9 kg legs), 18 DOF: six cam-shaft velocity servos (20 N·m,
   8 rad/s; action ±1 → ±π rad/s) through passive Whitworth hip-yaw linkages tracked by an ideal k2000/d40 PD;
   hip pitch and knee are non-backdrivable lead screws (SLA08 2000 N @ 28 mm/s; YH8 500 N @ 33 mm/s) modelled
   as stiff implicit PD (k16000/d600, k6000/d200) under a fixed rod slew cap (joint rates 0.23–0.47 and
   0.43–2.3 rad/s). Actuator friction 0, armature None, no load-dependent speed, no backlash, nothing fitted
   to data (`crab_hex_dimensions.py:162-183`). Max commanded speed ≈ 0.35 m/s; 50 Hz policy on a 5 ms step.
2. **What is randomized.** Robot-body friction U(0.6, 2.0) at startup against a fixed ground of 1.0 with
   "average" combine (effective foot μ ≈ 0.8–1.5); body mass add U(−0.5, +1.5) kg (−0.3..+0.9 %) and CoM
   ±1 cm, phase 2+ only; root xy velocity *replaced* by U(±0.5) m/s every 8 s on global time; deck payload
   U(0, 180) kg (25 % zero) with CoM xy ±0.10 m in phase 3; joint reset ×U(0.95, 1.05); heightfield roughness
   fixed at 2 cm. All copied from the Extreme Parkour A1/Go2 config (`go2/parkour_mdp_cfg.py:268-322`,
   `crab_hex_env_cfg.py:576-590`).
3. **What is not.** `enable_corruption=false`, noise null in every group (IMU, encoders, foot contact, the
   current-sense proxy, the ground-truth body xy velocity at `observations.py:254`, depth); action delay fixed
   at one step (20 ms) and depth latency at 100 ms; no Kp/Kd, strength, joint friction, damping, armature,
   rod-speed/force or leg-mass DR; no depth noise or holes; camera-pose DR disabled (world-frame bug);
   restitution, ground friction, gravity and compliance fixed. The teacher's 41-dim privileged latent is
   mass(1) + CoM(3) + friction(1) + Kp(18) + Kd(18) (`crab_hex_rl_cfg.py:10`), of which only friction varies.
4. **Campaign evidence.** p3b trains payload to 180 kg; eval max_stable 135 kg, first unstable 180 kg. The
   inherited Go2 comment on a commented-out ±2.5 % Kp/Kd block says "you will get a bad result"
   (`parkour_mdp_cfg.py:285`); no Krabby test exists. Student camera use was "not detectable" on the SEN0581
   rig (T* barely scan-responsive, 2026-10-28). Measured F19 stage costs: phase 1 ≈ 7.2 GPU-h, p2a 4.4, p3a
   3.4, p3b 3.8, student 10.2, sel 3.7, stand nets 6.6; full pipeline 47.3; one eval row ≈ 0.11 GPU-h.

**The short answer.** F19's DR is not adequate, and the deficit is specific. It lacks the three terms every
transferred legged pipeline reviewed carries (per-channel observation noise, some latency variation, some
actuator-parameter variation) and carries two Go2 terms that do not scale to the plant (a velocity-replacement
push worth ≈ 115–205 N·s and a friction floor of ≈ 0.8). Its payload range is already the widest in relative
terms in the literature and partly infeasible. Above all of that sits a modelling, not a randomization,
problem: the lead-screw/linkage stand-in belongs to the class the heavy-robot literature shows degrading or
failing on hardware, and DR does not substitute for identifying it (§2). Nearly all of the quantitative
evidence comes from torque- or position-PD controlled backdrivable quadrupeds, bipeds and humanoids of
12–105 kg; the rate-limited, self-locking, linkage-driven case has no direct precedent, so the ranges below are
scaled by the plant's own physics and marked as inferences where they are.

## 2. Actuator fidelity before randomization

**Key papers:** Hwangbo et al. 2019 [2]; Tan et al. 2018 [1]; Lee et al. 2025 (hydraulic, > 300 kg) [3];
Xie et al. 2021 [4]; Masuda & Takahashi 2023 [5]; Shin et al. 2023 [6]; Maslennikov et al. 2025 [8]; Hong et
al. 2026 (S3N) [64]; Bjelonic et al. (PACE) [10]; Spinelli et al. 2024 [62]; MathWorks [7]; Isaac Lab
actuators [9].

Ideal and hand-tuned analytical ANYmal actuator models "could not make a single step without falling"; the
actuator net cut torque error from 3.55 to 0.740 N·m, and accurate analytical models exist for direct drives
"but not for more complex actuators such as servomotors, SEAs, and hydraulic cylinders" [2]. On Minitaur, "if
the model discrepancy is too big, even a robust controller trained with random perturbations cannot overcome
the reality gap", and a linear torque–current model made the robot "sink to its feet" until saturation was
modelled [1]. The closest mass analogue, a > 300 kg hydraulic quadruped with "inherent slow control response",
transferred with a physics-structured model (position-error gain, torque lag, velocity damping, flow-limit
term) fitted on 20 s of logs: RMSE 7.45 vs 36.0/33.3/27.4 for MLP/LSTM/GRU nets whose RL runs did not
converge; its stiff high-gain PD stand-in walked stably but "restricted the control's ability to lift the legs
sufficiently" over obstacles [3]. On Laikago, gain choice decided transfer: "Policies without velocity
feedback or with kp = 160 all fail the sim-to-real tests", even with DR [4]. Analogues for Krabby's
transmission, none identical: directional gear efficiency on a high-ratio servo humanoid (DR policy 0/3,
re-identified model with a brake-torque efficiency term 3/3) [5]; a motor torque–speed envelope (Hound
4.5 m/s without it, 6.5 m/s with; the no-envelope policy fell at 5 m/s) [6]; a generic lead-screw example
whose speed falls 26 → 19 mm/s from no load to its rated 1000 N, where "the no-load and rated-load speed
information depends on the unknown friction levels" [7]; serialising closed-chain linkages "significantly
impairs sim-to-real transfer" [8]; and S3N, which restores actuator inertia and damping reflected through a
parallel linkage (joint-position RMSE −80.9 %, torque −82.1 %, sim-to-real gap 17.3 → 9.9 %) while stating
that DR "does not explicitly restore the missing coupling and inertia structure" [64]. For velocity-commanded
heavy hydraulics, a 40 t handler's boom is a first-order lag plus delay on the velocity reference with
controller parameters resampled per episode; only its dead-zone-ridden slew motor gets a learned model [62].
PACE treats torque/velocity saturation as "a known and enforced nonlinearity that is applied
deterministically", identifies 7.5 ms global delay on ANYmal and TYTAN, and counts MINIMAL (lead-screw lever)
and TYTAN (ball-screw knee) among its 13 transferred robots, both with high-bandwidth drives [10]. Isaac Lab
ships `DCMotor` (four-quadrant torque–speed clip), `RemotizedPDActuator` (angle-dependent limits through a
transmission table) and `DelayedPDActuator` [9].

**What it means here.** Krabby's rods are the stiff PD stand-in that was limited in [3], with a load-blind
slew cap where [6, 7] show the envelope is what breaks on hardware; the Whitworth tracker is the serialised
linkage of [8, 64]. No DR width fixes a wrong envelope. Pre-hardware: replace the constant slew with a
force-dependent speed line (no-load speed, stall force; `DCMotor` is the primitive) and add
configuration-dependent reflected armature/damping on hip yaw [64, 9]; randomize *around* those (§3). At
bring-up: 20–60 s bench excitation per actuator type (sine, square, chirp and noise rod commands under load;
cam chirps through the linkage), fitting speed line, directional efficiency, delay and compliance [10, 3],
then a first-walk proprio + current log for distribution matching [17]. Caveat: every transfer here is a
backdrivable or hydraulic drive; whether a self-locking screw needs a learned residual at all is untested.

## 3. Actuator-parameter DR: strength, speed, gains, friction, backlash, offsets

**Key papers:** Tan [1]; RMA [18]; Extreme Parkour config [19]; Walk-These-Ways [13]; Robot Parkour [24];
Humanoid-Gym [22]; Fey et al. [14]; No More Blind Spots [23]; Siekmann et al. [20]; Singh et al. 2023 [21];
Berkeley Humanoid [11]; SPI-Active [12]; Loco-Loco-RL [60]; Taylor et al. (Sony) [61]; ERFI [45]; HeLoM
[63]; motion-prior hexapod [75]; Qu 2025 [77].

Motor-strength scaling is the most common actuator term, usually ±10–20 % (Tan 80–120 % [1]; RMA 0.90–1.10,
test 0.88–1.22 [18]; Extreme Parkour 0.8–1.2 [19]; WTW 0.9–1.1 [13]; Robot Parkour 0.9–1.1 [24]; Humanoid-Gym
95–105 % Gaussian [22]; Fey, on a Unitree B2 + Z1 arm, stall torque 0.9–1.1 [14]; torque efficiency 0.9–1.0,
"particularly impactful for transfer to extreme terrains" [23]). It is not universal: Cassie [20] and HRP-5P
[21] skip it. Gains are randomized too (RMA Kp 50–60, Kd 0.4–0.8 [18]; Fey PD stiffness ×0.9–1.1, damping
×0.5–1.5 [14]; hexapods ×0.8–1.2 [75] and ×0.9–1.1 [63]), so the Go2 comment covers one ±2.5 % block and
nothing more. Joint damping ×0.3–4.0 and joint mass ×0.5–1.5 with encoder offset ±0.05 rad on Cassie [20];
motor friction 0–0.05 N·m [1]. The identification-first recipes keep these narrow: joint friction ×0.9–1.1 and
armature ×1.0–1.05 around measured values, calling strength/gain DR a "lazy approach" because "extensive
domain randomization slows down training and results in conservative policies" [11]; SPI-Active's "Heavy DR"
(ranges ≈ doubled) was worse than vanilla on 4 of 5 tasks (1.75/1.40/1.06/1.10; better only on attitude,
0.85) [12]. A *speed* limit is almost never randomized: the only instance is joint max velocity ×U(0.5, 1.5)
on an OP3, unablated [60]; Tan bounds speed indirectly through battery voltage 14.0–16.8 V [1], and a hexapod's
failure rate rose when supply sagged from 12 to ≈ 10 V [77]. Backlash: a dead zone around the target, width
U(0, max) with max ramped linearly to 5° "to gently introduce backlash", on a 1.5 kg servo humanoid [61].
Current feedback: the 105 kg HRP-5P trained with a counter-torque τ_pd − k_bemf·q̇, k_bemf ~ U[5, 40]
resampled every 100 ms, static friction U(2, 8) N·m and viscous U(0.5, 5), feeding measured current as an
observation; a separately trained policy with that input zeroed self-collided on hardware, so feedback is
"vital" when tracking is degraded; feedback *without* the randomized imperfection was never tested, and no
noise was put on the current channel [21]. ERFI's episodic per-joint torque offset "emulate[s] a shift in the
robot's mass, inertia, impedance gains and contact Jacobian" and gave on average 53 % better success than
per-step injection under a 22–65 kg base-mass sweep (sim); base forces above 5 N / 3 N·m caused pronking [45].

**What it means here.** For Krabby "strength" is the rod force limit (2000/500 N), the rod speed (28/33 mm/s,
load- and voltage-dependent) and the cam effort/velocity limits (20 N·m, 8 rad/s). The convention-consistent
addition is a per-env scale ×0.85–1.15 on each, applied to the no-load speed and stall force of §2's speed
line, plus per-joint rod/encoder offsets (±0.02–0.05 rad [13, 20]), screw friction/efficiency, and a
first-order time constant on each velocity response [62]. A dead zone on the rod target and linkage output,
ramped from 0 [61], covers lost motion on reversal of a self-locking nut (inference). This is also what makes
the current-sense proxy informative: today it is a deterministic function of clean torque and rod speed,
carrying nothing the joint state does not; on hardware it will reflect friction, efficiency and speed sag
[21]. Blanket Kp/Kd scaling on the implicit PD has no hardware counterpart on a position-mode screw and should
stay off; the hedge for a statically indeterminate six-leg stance is a rod compliance term [4, 75]. An ERFI
torque offset would be absorbed by the stiff PD and slew limiter, so the analogue is an episodic rod-position
bias or force/speed scale (inference). All ranges are extrapolated from 12–105 kg backdrivable robots and
should narrow to identification uncertainty once §2's bench data exist [10, 11].

## 4. Observation noise, the velocity channel and the current proxy

**Key papers:** Isaac Lab velocity cfg [25]; legged_gym [26]; Rudin et al. [27]; Ji et al. 2022 [28];
Agarwal et al. [29]; Berkeley Humanoid [11]; PACE [10]; Lee et al. 2020 [31]; Peng et al. 2018 [30];
extreme-parkour code [19]; NVM [32]; Xie [4]; motion-prior hexapod [75].

Every reference framework and pipeline turns per-channel noise on, the minimal-DR recipes included. Isaac
Lab's velocity task (F19's framework) ships `enable_corruption=True` with Unoise ±0.1 m/s lin vel, ±0.2 rad/s
ang vel, ±0.05 gravity, ±0.01 rad joint pos, ±1.5 rad/s joint vel, ±0.1 height scan [25]; legged_gym the same
scales [26]; Rudin et al. base the amount on "real data measured on the robot" [27]; Ji et al. size it to "the
true distribution of the measurements" (numerical differentiation plus logging jitter): joint pos ±0.05 rad,
joint vel ±0.5 rad/s, orientation ±0.03, ang vel ±0.1 rad/s [28]; Agarwal uses Gaussian σ 0.01 rad joint
angle, 0.02 orientation, and 0.05 on *normalised* joint and angular velocity (≈ 1 rad/s and 0.2 rad/s
physical) [29]; Berkeley Humanoid 8 terms [11]; PACE keeps i.i.d. noise on "all but commands" after dropping
every dynamics term [10]; Lee 2020 trains the teacher with additive noise and deploys the student with each
quantity "replaced with readings from the proprioceptive sensors" [31]. The only hardware ablation is a
Fetch-arm puck task: no observation noise dropped success 0.89 → 0.25 (12 trials), fixed timestep 0.29,
fixed mass 0.64, fixed friction 0.48 [30]. On velocity: the parkour lineage F19's observation code descends
from puts base_lin_vel in `priv_explicit` and regresses it from proprioception [19]; Ji finds "linear velocity
estimation plays the most important role" [28]; NVM drops it at distillation because it is "typically only
available in instrumented environments" [32]; Agarwal's policy carries none [29]; Lee replaces it with an
IMU + kinematics estimator [31]; removing it outright is not supported: Laikago pacing policies without
velocity feedback "cannot control the lateral velocity and often fall sideways", even with DR [4]; the 25.5 kg
hexapod adds "it is difficult to obtain accurate linear velocity on real robots" [75].

**What it means here.** F19 is the only pipeline here with every channel clean, teacher and student alike,
and this is the cheapest gap to close. Priorities: (1) noise, bias and slow drift on the current-sense proxy,
which on a non-backdrivable screw is friction-dominated and lagged, and which today is an unrealistically
perfect payload cue (§6); (2) estimator-grade noise, bias and drift on the xy-velocity slot, or moving it into
`priv_explicit` with the lineage's concurrent estimator head [19, 28], since on hardware the slot gets nothing
or a drifting leg-odometry estimate from non-backdrivable legs plus IMU; (3) Isaac Lab defaults for joint
position and IMU, with joint-velocity noise *rescaled*: ±1.5 rad/s would swamp joint rates of 0.2–2.3 rad/s, so
size it from the planned encoder/derivative pipeline [28]. A pre-hardware audit should list every student
input, its hardware source and whether sim corrupts it accordingly [31].

## 5. Latency

**Key papers:** Tan [1]; Xie [4]; Robot Parkour [24]; MMDR [33]; Agarwal [29]; Extreme Parkour [34]; Isaac
Lab `DelayedPDActuator` [9]; Booster Gym [35]; Humanoid-Gym [22]; ANYmal Parkour [36]; No More Blind Spots
[23].

Most transfers simulate action and/or observation latency over a band that covers and often exceeds the
measured value; the exceptions pad to a constant or have negligible latency. Tan measured it with a one-step
PWM spike (3 ms microcontroller, 15–19 ms TX2) and randomized 0–40 ms plus a 3–20 ms control step; without it,
feedback policies "oscillate, diverge", and "without either" the actuator model or latency simulation "the
learned controllers do not work on the real robot" [1]. Xie found latency the *one* dimension where DR clearly
helped (tolerance 17 → 32 ms from 0–20 ms randomization, sim and robot), yet their default policies
transferred with none because the loop was under 4 ms [4]. Robot Parkour: proprio latency [37.5, 47.5] ms
resampled every 2 s, depth [0.20, 0.26] s every 5 s [24]; MMDR randomizes both modalities out of sync and
raised real moving distance 444.7 → 859.9 and 505.7 → 973.0 on two of four courses (≈ 1.4× on the others)
[33]; Agarwal measured 100 ± 20 ms capture and 10 ± 10 ms transport and trains [80, 120] ms update /
[10, 30] ms latency over a curriculum [29]; Booster Gym measured 9–12 ms and trains 0–20 ms [35]; Humanoid-Gym
0–10 ms [22]; No More Blind Spots measured < 20 ms and trains 20–100 ms [23]; ANYmal Parkour trained
perception with zero delay and reports the asynchronous deployment "does not seem to be affected" [36].
Extreme Parkour pads the loop to a constant 0.08 s depth / 0.016 s proprio latency [34]. `DelayedPDActuator`
samples a per-env command lag in [min_delay, max_delay] physics steps at every reset [9].

**What it means here.** F19's fixed 20 ms action and 100 ms depth delays are defensible only if the runtime
pads to those values; the drive controllers, cam servos and bus do not exist, so nothing is measured.
Randomize action delay over 0–2 policy steps (0–40 ms at 5 ms granularity, per env, resampled per episode)
and depth latency narrowly around 100 ms (≈ 80–140 ms), proprio and depth independent [24, 33]; plan the
spike test for bring-up [1]. Because Krabby moves at ≤ 0.35 m/s the geometric cost of depth latency is
centimetre-level ("0.2 s ... 0.2 m" at 1 m/s [24]), so keep the depth band narrow (Xie's over-widening
warning [4]) and put the weight on action latency interacting with the slow rods.

## 6. Inertial, payload, push and friction ranges scaled to the plant

**Key papers:** Hwangbo [2]; Tan [1]; Peng 2020 [37]; Xie [4]; Bellegarda [38]; Miki [39]; Chang 2025 [40];
Fey [14]; Isaac Lab cfgs [25, 46]; legged_gym [26]; extreme-parkour [19]; RMA [18]; Lee 2020 [31]; SimOpt
[41]; DrEureka [42]; MULE [43]; Ma et al. 2022 [44]; ERFI [45]; Rudin [27]; WTW [13]; Humanoid-Gym [22];
Hirose et al. [79]; Garcia et al. [80, 81]; Haldane et al. [82]; Oke et al. [83]; HeLoM [63]; hexapod peers
[74, 75].

*Mass.* Published inertial DR is a fraction of the robot's own link masses and covers legs and inertia:
Hwangbo ±15 % on all links, CoM and joint positions ±2 cm, because CAD carries "up to about 20 % error" [2];
Tan, Peng and Xie mass ×0.8–1.2, inertia ×0.5–1.5 [1, 37, 4]; Bellegarda every body part ±20 % with inertia
following, plus 0–5 kg added [38]; Chang link ×0.8–1.2, leg CoM ±1.5 cm [40]; Fey link ×0.5–1.5, CoM ±2 cm
[14]; HeLoM link and inertia ×0.9–1.1, CoM x U(−0.2, 0.1) m [63]. Framework defaults use absolute kg: Isaac
Lab U(−5, 5) kg on the base (CoM ±5 cm only from v2.2.0) [25]; legged_gym ±1 kg base, ±5 kg ANYmal-C with an
actuator net [26]; Extreme Parkour 0–3 kg and CoM ±0.2 m on an A1, fed to the teacher as privileged latents
[19]. Xie, cited for its ranges, found mass DR did not improve robustness and made policies conservative [4].
*Payload.* F19's 0–180 kg (≈ 78 % of robot mass; eval max_stable 135 kg ≈ 59 %) exceeds every quadruped
*training* range (RMA 0–6 kg on 12 kg [18]; Bellegarda 0–5 kg at p = 0.8 [38]; Lee none, yet carried
10 kg = 22.7 % [31]), though RMA carried 100 % on hardware [18]. The payload literature adds: bounds from
feasibility (wide distributions "include scenarios with infeasible solutions ... or lead to exceedingly
conservative policies" [41]; RAPP sets per-parameter bounds by one-at-a-time rollouts [42]; MULE repeats the
conservatism warning [43]); anisotropic footprint-scale CoM (Bellegarda ±[0.15, 0.05, 0.05] m [38]);
in-episode variation and load dynamics (RMA resample p = 0.004/step [18]; MULE every 4 s [43]; Chang a sliding
cube with friction U(0.001, 0.2) and initial velocity ≤ 0.5 m/s [40]; Ma an acceleration-proportional wrench
[44]); and adaptation through latents or estimators (RMA's extrinsics tracked a thrown-on 5 kg [18]; Lee's
student decoder recovers an unknown 10 kg load [31]; Chang's proprio-only load estimator [40]). *Pushes.*
legged_gym's push "emulates an impulse by setting a randomized base velocity", so the impulse scales with
mass [26]; Tan's force pushes are 130–220 N for 0.06 s (≈ 8–13 N·s) [1]; ERFI trained with *no* base force
(> 5 N / 3 N·m pronked) and its 0–150 N × 3 s was an evaluation [45]; Isaac Lab's stock push *adds* ±0.5 m/s on
per-env 10–15 s intervals and its Go2 rough config sets `push_robot = None` [46]; Rudin ±1 m/s every 10 s [27].
Hirose's NESM is a mass-invariant yardstick: a disturbance is a horizontal CoG velocity V = √(2E/m), and
tipping needs E/mg to exceed the support-polygon barrier height [79]; static margins are inexact once inertial
forces matter [80, 81]. *Friction.* Lower bounds sit well below F19's effective ≈ 0.8: RMA 0.05–4.5 [18]; Lee
slippery hills N(0.3, 0.1) clipped above 0.1 [31]; WTW 0.40–1.00 (+ restitution 0–1, gravity ±1 m/s²) [13];
Tan, legged_gym, Rudin, Robot Parkour 0.5–1.25 or 0.5–1.0 [1, 26, 27, 24]; Humanoid-Gym 0.1–2.0 [22]; hexapods
0.4–1.2 [63], 0.5–1.25 [74], 0.1–2.5 [75]; Miki "occasionally" sets foot friction low "to introduce slippage"
[39]. Dynamic similarity (velocity ∝ √L, time ∝ √(L/g), force and impulse ∝ m; friction dimensionless) is the
scaling used for design [82, 83].

**What it means here.** (i) Body-mass DR of −0.3..+0.9 % is effectively zero; legs (31 % of mass) and
inertia are untouched. Literature-consistent values: body ±5–10 % (≈ ±8–16 kg) with CoM ±3–5 cm, per-leg-link
×0.85–1.15 with inertia following, from phase 1 (ramped, §9), written into the privileged latent, which today
encodes mass, CoM, friction, Kp and Kd of which only friction moves. (ii) Do not widen the payload: 180 kg is
past the measured feasibility edge (135 stable, 180 unstable), so a RAPP sweep on T* should set the upper
bound; make the CoM anisotropic toward the 0.711 × 1.219 m deck (e.g. ±0.3 m long axis, ±0.15 m short;
inference), add a base-CoM CAD error and a mid-episode step change or acceleration wrench [38, 43, 44].
(iii) The push is the clearest unscaled copy: replacing velocity by 0.5 m/s on 230–410 kg is ≈ 115–205 N·s
per axis at rest (more if moving against it), 15–27× the momentum of the same setting on a 15 kg Go2 and
10–25× Tan's pushes, synchronized across envs, above the 0.35 m/s command ceiling, against hips slewing at
0.23–0.47 rad/s; in NESM terms 0.5 m/s is a 12.7 mm barrier on any mass [79]. Specify pushes in N·s sized to
what the rods can resist (sweep, §10), additive, on per-env intervals [46], and add multi-second body
force/torque disturbances through the inert `base_external_force_torque` term [39], below ERFI's pronking
threshold scaled to this mass [45]. (iv) Lower the effective friction floor to ≈ 0.3–0.4 by randomizing the
ground material or the combine mode, with occasional slippery episodes [39]; keep the high tail (rubber feet
gripped harder than sim in [74]). Compare on the *effective* coefficient; per-shape vs combined was not
verified per paper. Caveat: Froude scaling breaks for Krabby, whose recovery time is set by rod slew, not body
time scale [82]; push and latency tolerance must come from the sweep, not from similarity.

## 7. Ground side: compliance and the contact solver

**Key papers:** Singh et al. 2025 (HRP-5P compliant terrain) [65]; MILD [66]; Lee et al. soft contact [68];
Kim & Lee 2021 [69]; Le Lidec et al. [70]; sim2sim report [71]; Liu et al. 2025 [72]; Isaac Lab material cfg
[73]; PACE [10]; Mouret et al. [78]; Dante II [84].

On the 105 kg HRP-5P a policy trained on rigid uneven ground "failed to walk more than a few steps on the
compliant terrain" (5/5) and one fixed compliance value produced "redundant (and dangerous)" ankle twists;
the fix was a per-*foot* soft-contact time constant U(0.02, 0.4) resampled about every 0.5 s, with joint
damping 0.2–5 and static friction 2–8 N, inferred from proprioception and motor current only [65]. MILD's
baselines with spring/cone/rigid contact "adopt overly conservative gaits characterized by insufficient foot
clearance during swing phases" on foam, hitting joint limits; its 154-element foot with μ 0.2–1.3 and
penetration stiffness 8–1400 N/m randomized walked seven terrains at 1.2 m/s without failure [66].
Spring-damper soft contact in a standard loop needed a 5000 Hz step, a hard-then-soft curriculum (iterations
4000–7000), and degraded only at low stiffness with high damping [68]; sinking tiles up to 5 cm needed
end-effector history in the observation [69]. Solver choice matters for redundant contact: compliance removes
"the jamming internal forces in the under-determined cases" and PGS-type solvers saturate at mild accuracy
[70]; an ANYmal-C policy moved Isaac Lab → MuJoCo kept timing but its load distribution differed "quite
significantly", peak foot forces higher in Isaac "sometimes by 100s of Newtons", MuJoCo needing Kp 40 → 100
(course report) [71]; on a B1, PhysX gave yaw oscillation and MuJoCo foot drift, and impratio 1 → 100 was
"traction tuning rather than intrinsic robustness" [72]. Counterweight: the closest-mass systematic study
(52–53 kg) found the dominant failure was actuator modelling ("URDF-only fails (caught by crane)"),
identified in the air, and deferred contact refinement [10]; evolutionary hexapod work blamed contact and slip
(1294 mm sim vs 411 mm real) [78]; Dante II (≈ 770 kg) "stepped on some particularly soft ash, lost its
balance, and fell over" (press account) [84]. Isaac Lab exposes per-material
`compliant_contact_stiffness/damping` (enabled when stiffness > 0), restitution and combine modes [73]; no
paper reports which PhysX contact-offset or iteration settings had to change to match real foot forces.

**What it means here.** F19 trains on fully rigid contact (compliance 0, restitution 0, TGS 20/6 iterations,
max_depenetration 1.0 m/s) with 2 cm roughness. A tripod or six-foot stance is statically indeterminate, so
per-foot load, hence joint torque, hence the current-sense proxy, is partly a solver artefact that becomes a
policy input. Rank ground-side DR below actuator fidelity [10], but add a per-env, per-foot compliant-contact
randomization on the six footpads (the HRP-5P placement ports directly to Isaac Lab's per-body materials) in
phase 2+, after a 200 Hz stability smoke (PhysX's implicit model, not an explicit spring [68]), with a foot
sinkage/clearance read in eval [66]; and make contact settings part of the pre-hardware sweep (§10). Slow
non-backdrivable knees cannot correct a sinking foot quickly, so the case is stronger than for HRP-5P
(inference).

## 8. Depth-student augmentation

**Key papers:** Robot Parkour [24]; MMDR [33]; NVM [32]; extreme-parkour config [19]; Parkour in the Wild
[47]; Now You See That [48]; Agarwal [29]; Miki [39]; ANYmal Parkour [36].

Most, not all, vision students corrupt rendered depth and clean real depth; some randomize camera pose. Robot
Parkour: clipping, Gaussian noise (near σ 0.02, far σ 0.08 beyond 2 m), full-block artefacts p = 0.008 and
half-block sparks p = 0.02 in sim; clipping, hole filling, spatial and temporal smoothing on the real image;
camera position N(0.27, 0.01)/N(0.0075, 0.0025)/N(0.033, 0.0005) m, pitch U(0, 5°), FOV U(85, 88°) [24]. MMDR
sets 3–30 pixels per frame to max depth and fills real holes with 10 m [33]; NVM 1 % of pixels [32]; Extreme
Parkour pitch ±5° but `dis_noise = 0` [19]; Agarwal renders from a calibrated fixed camera and only cleans the
real image [29]; Parkour in the Wild (ANYmal D): clip, edge-pixel dropout/shuffle, temporally consistent Perlin
holes, a 1–5 column blind strip, Gaussian blur, per-camera delay [47]. Now You See That's simulated ablation:
98.9 % → 90.4 without stereo-hole fusion, 92.1 without depth-dependent noise, 94.5 without calibration
(scale, intrinsics, extrinsics) uncertainty, 96.2 without Perlin noise [48]. Map-based ETH: per-point,
per-foot and per-episode height noise in a 60/30/10 nominal/offset/blackout mix ramped linearly [39]; the
elevation map shifted up to 7.5 cm, and perturb/blob/patch/pose augmentation of the perception network "key to
make the pipeline robust to noise and drift" [36].

**What it means here.** The student sees clean rendered depth with a 2 m clip and no camera-pose DR (frame
bug). Fix the pose event and re-enable a few degrees pitch/roll, ≈ 1 cm position and small FOV jitter (mounts
on a 230 kg chassis under a 0–180 kg deck sag; the SEN0581 rig rotates and edge-mounts its cameras); add
range-scaled noise, dropout-to-max, edge dropout and Perlin holes with a versioned real-side preprocessing
chain; add a Miki-style blackout regime that doubles as a graceful-fallback test. All student-only
(≈ 10.2 GPU-h). Where the campaign disagrees: camera use was "not detectable" on the SEN0581 rig and T* is
barely scan-responsive, so until scan use is demonstrated this buys robustness to a signal the policy may not
read; [48] is a simulated benchmark.

## 9. Over-randomization, curricula and the hexapod peers

**Key papers:** Tan [1]; Xie [4]; RMA [18]; Peng 2020 [37]; Miki [39]; DORAEMON [49]; OpenAI [50]; DeXtreme
[51]; Fu et al. [52]; Chen et al. [53]; Peng 2018 [30]; Mehta [54]; TransCurriculum [55]; Kawawa-Beaudan [74];
motion-prior hexapod [75]; Qu [76, 77]; HeLoM [63]; Mouret [78].

"Randomization is not a free meal, we should use it only when necessary" [1]; "Superfluous dynamics
randomization harms performance in measurable ways" (trot 1.1 → 0.9 m/s, sim and robot) and DR "often offers
negligible actual improvements in robustness" once gains and velocity feedback are right [4]; RMA's DR-only
baseline "learns a very conservative policy" (62.4 vs 73.5 %, sim) [18]; on dynamic skills robust policies
"are prone to falling" while adaptive ones succeed, though no-DR policies failed outright [37]. Ramps and
gates: RMA linearly raises mass/friction/strength difficulty and resamples in-episode [18]; Miki multiplies DR
magnitude by c_{k+1} = c_k^d, d = 0.98 [39]; DORAEMON widens entropy only while success stays high
(manipulation) [49]; OpenAI's ADR starts from one calibrated environment, and larger fixed entropy lengthened
training, hypothesised "infeasible altogether" [50]; DeXtreme's ADR beat manual DR in the mean (27.8 ± 19.0 vs
14.8 ± 5.4, medians ≈ equal) at the cost of 40 % of envs [51]; uniform DR gives "suboptimal, high-variance
policies" [54]; TransCurriculum gates friction/payload DR on competence (transfer loss 27 → 18 %, Go1
preprint) [55]. A teacher can supply supervision "impossible for the student network to predict" (ROA 0.31 →
2e-4) [52]; DR without memory is bounded worse [53, 30]. Every hexapod peer with hardware results carries
broader actuator/inertial DR and observation noise than F19: SpiderPi friction 0.5–1.25, pushes every 8 s,
uniform noise on every state element, and a foot-shuffling gait that did not transfer because "rubber-tipped
feet grip the ground more strongly" [74]; HeLoM friction 0.4–1.2, base mass −2..+3 kg, CoM up to ±0.2 m,
inertia and Kp/Kd ×0.9–1.1, then per-actuator nets from short hardware logs (ang-vel error 0.146 → 0.105 rad/s;
pure actuator nets jittered) [63]; the 25.5 kg hexapod stiffness/damping ×0.8–1.2, link mass ×0.9–1.1, payload
0–5 kg at ±0.15 m, friction 0.1–2.5, strength 0.8–1.2 [75]; Qu randomizes only terrain geometry [76].

**What it means here.** F19's present conservatism cost is small because its DR is narrow; each addition
should be justified by a specific expected mismatch, centred on a datasheet or measured value, and enter on a
Miki ramp or success gate that the existing terrain/payload curricula already approximate. The hand-staged
schedule (DR off in phase 1, payload 90 → 180) is the right shape; full ADR has no shipped legged precedent.
Prefer parameters the student can infer from proprioception (payload, rod lag, friction) [18, 52], check the
student's imitation error per DR bin before widening teacher-visible DR, and verify the history window spans
multi-second lead-screw transients. Krabby's statically stable regime at 0.35 m/s should be more forgiving of
model error than Go2-class dynamic gaits [78], so copy *less*, not more, of the Go2 ranges.

## 10. Evaluating DR adequacy before hardware

**Key papers:** Tan [1]; Xie [4]; DrEureka [42]; Humanoid-Gym [22]; PolySim [56]; RoboGauge [57]; Kadian et
al. 2020 [59]; Muratore et al. [58]; Hirose [79]; Mouret et al. [78].

Three checks, each necessary, none sufficient. (a) Single-parameter stress sweeps on a frozen policy: Tan
samples 10 values per parameter and records return [1]; Xie raises mass, P gain, latency, push and slope one at
a time, records the largest perturbation handled, and "advocate[s] identifying important, i.e.,
high-sensitivity, sim-to-real bottlenecks using simulations" [4]; RAPP runs the same sweep to *set* DR bounds
[42]. (b) Cross-simulator transfer: Humanoid-Gym's Isaac Gym → MuJoCo check for researchers "lacking physical
robots", on a hardware-calibrated MuJoCo [22]; PolySim's IsaacSim-only G1 at 0.036 success in MuJoCo vs 0.500
IsaacGym-only and 0.564 multi-simulator, because "Policies trained in a particular simulator inevitably
inherit its bias" [56]; RoboGauge's MuJoCo proprio metrics with payload/friction/latency/noise DR tracked
real Go2 ground truth better than the Isaac Gym training env (0.0558 vs 0.0883) [57]. (c) Real-data metrics
once rollouts exist: the Sim-vs-Real Correlation Coefficient rose from 0.18 to 0.844 after simulator tuning
(navigation) [59]. Muratore et al.: a noise width is itself a hyperparameter, "if the noise level is too high,
the learning procedure will fail" [58]. A mass-invariant margin exists for pushes and payload offsets: the
minimum NESM/DNESM along a rollout [79]; the transferability approach penalises behaviours whose score is
sensitive to simulator perturbation [78].

**What it means here.** Protocol: a held-out sweep grid on T* and the ONNX student over the terms F19 does not
randomize (rod speed/force ±, cam limits, Kp/Kd, leg mass, inertia, ground friction and compliance, contact
iterations, action/depth latency jitter, obs noise, current-proxy bias, depth holes, payload beyond 135 kg,
push impulse in N·s), one at a time beyond the plausible band, reporting the tolerated range and the minimum
stability margin per term; add DR only where F19 is fragile inside the plausible uncertainty [4], and take
the feasible range as the DR bound [42], which sizes the push and payload terms of §6. Rows run through the
existing gait_eval/obstacle gates (≈ 0.11 GPU-h each). A MuJoCo re-implementation of the lead-screw and
Whitworth actuation is the optional second check; a pass is necessary only, because none of these validates
the actuator model (§2).

## 11. Gap table: F19 terms against typical practice

| Term | F19 (effective) | Typical practice (source) | Verdict; plant-scaled range |
|---|---|---|---|
| Actuator model (rods, cams, linkage) | stiff implicit PD, constant slew, ideal tracker, unfitted | identified analytic or learned model; stiff PD limited on obstacles [2, 3, 10, 64] | **investigate first**: force-dependent speed line, reflected armature/damping, delay; bench ID at bring-up |
| Rod force/speed, cam limits | fixed 2000/500 N, 28/33 mm/s, 20 N·m, 8 rad/s | strength ×0.8–1.2 [1, 18, 19, 24]; speed rarely [60, 1] | **add** ×0.85–1.15 per env on no-load speed, stall force, cam limits (inference from the strength convention) |
| Kp/Kd | none ("bad result" comment) | ×0.9–1.1 common [18, 14, 63, 75] | **keep off** on the implicit PD; **add** a rod compliance term instead (inference) |
| Joint friction/damping/armature | 0 / 0 / None | friction ×0.9–1.1, armature ×1.0–1.05 around identified values [11]; damping ×0.3–4 [20] | **add** nominal values first (§2), then ±10–15 % |
| Backlash/offsets | none | dead zone 0 → 5° ramped [61]; offsets ±0.02–0.05 rad [13, 20] | **add** rod/linkage dead zone and per-joint offset |
| Action delay | fixed 20 ms | 0–20/40 ms randomized, or padded constant [1, 4, 35, 34] | **widen** to 0–40 ms per env |
| Depth latency | fixed 100 ms | [200, 260] ms, [20, 100] ms randomized [24, 23] | **widen narrowly** ≈ 80–140 ms, independent of proprio |
| Observation noise | none | on in every pipeline [25, 26, 27, 28, 29, 11, 10] | **add**; joint-vel scaled to 0.2–2.3 rad/s rates |
| Current-sense proxy | deterministic | measured current with a randomized plant behind it [21] | **add** noise/bias/drift and a randomized plant |
| xy body velocity | ground truth, teacher and student | estimated or privileged-only [19, 28, 31] | **investigate**: estimator head or matched corruption |
| Body mass/CoM | −0.3..+0.9 %, ±1 cm, phase 2+ | ±15–20 % links, CoM ±2–5 cm [2, 38, 40] | **widen** body ±5–10 %, CoM ±3–5 cm, from phase 1 (ramped) |
| Leg mass/inertia | none | links ±15–20 % with inertia [2, 38] | **add** ×0.85–1.15 per leg link |
| Payload mass | U(0, 180) kg, 78 % | 0–50 % training [18, 38] | **narrow** upper bound to the feasibility sweep (≈ 135 kg measured) |
| Payload CoM | ±0.10 m isotropic | ±[0.15, 0.05, 0.05] m [38] | **widen anisotropically** toward the deck (±0.3/±0.15 m; inference); add mid-episode change |
| Push | replace ±0.5 m/s / 8 s global (≈ 115–205 N·s) | add ±0.5–1 m/s per env 10–15 s [46, 27]; forces 8–13 N·s [1]; none [45] | **remove and replace** with additive, per-env, N·s-specified pushes sized by sweep/NESM [79] |
| Body force/torque | inert (0, 0) | multi-second wrenches [39]; > 5 N / 3 N·m pronked on 27 kg [45] | **add** small, after the sweep |
| Robot friction | U(0.6, 2.0) → effective 0.8–1.5 | lower bounds 0.05–0.5 [18, 31, 13, 27] | **widen** effective floor to ≈ 0.3–0.4 via ground/combine; keep the high tail |
| Ground compliance/restitution | rigid, 0 | per-foot compliance U(0.02, 0.4) [65]; restitution 0–1 [13] | **add** per-foot compliant stiffness/damping in phase 2+ after a 200 Hz smoke; restitution optional |
| Terrain roughness | fixed 2 cm | curricula [39] | **keep**; check that it forces clearance [74] |
| Depth image | clean | noise, holes, artefacts, blur [24, 47, 48] | **add** (student only), after the scan-use check |
| Camera pose/FOV | disabled (bug) | ±5°, ≈ 1 cm, FOV ±1.5° [24, 19] | **add** after the frame fix |
| Gravity | fixed | ±1 m/s² [13] | **keep** (optional, not core) |
| Joint reset ×0.95–1.05 | on | standard [25] | **keep** |
| DR schedule | hand-staged, off in phase 1 | ramps/gates [39, 49]; no shipped legged ADR | **keep the shape**; ramp new terms; no ADR |

## 12. Literature gaps

- No sim-to-real study of a rate-limited, self-locking lead-screw or velocity-servo-through-linkage legged
  robot; the identification-first evidence [3, 10, 64] is from hydraulic and high-bandwidth electric drives.
- No ablation of randomizing an actuator *speed* limit as distinct from torque [60, 1]; no hardware test of
  current feedback without the randomized imperfection [21]; no published noise model for a current channel.
- No legged-locomotion hardware ablation of observation noise; the one ablation is a manipulator with 12–28
  trials per row [30].
- No paper randomizes ground stiffness as a DR parameter inside Isaac Lab, nor reports which PhysX contact
  settings had to change to match real foot forces [73, 71, 72]; compliant-contact RL evidence is a workshop
  paper and sim-only studies [68, 69].
- No ablation of mass-DR width against robot size; fractional ranges are convention [2, 38]; Froude scaling
  is not validated for rate-limited actuators [82].
- No shipped legged policy uses full ADR; the ADR evidence is manipulation [49, 50, 51].
- Hexapod peers are small (≤ 25.5 kg) with position servos or SEAs [63, 74, 75]; the heavy classical walkers
  are documented here only through a press account [84].
- Whether the depth student should see wider DR than the teacher (DAgger under DR) is not addressed by any
  reviewed source.
- Load-bearing numbers from single 2025–2026 preprints or non-peer-reviewed reports: [48, 56, 57, 60, 64, 66,
  71, 72, 75].

## 13. Ranked recommendations

Ranked by expected sim-to-real value per GPU-hour. Cost basis is F19's measured stage table: student relaunch
10.2 GPU-h; p3b 3.8; p2a → p3b 11.5; full pipeline 47.3; one eval row ≈ 0.11
(`REPORT_baseline_vs_final.md`). Knobs are this repo's (`crab_hex_env_cfg.py`, `payload_knobs.py`,
`omni_events.py`, `parkour_actions.py`, `crab_hex_dimensions.py`, `crab_hex_omni_observations.py`).

0. **Audit, 0 GPU-h** (§4, §9; [31, 52]). List every student input with its planned hardware source and sim
   corruption; confirm whether hardware will supply the xy-velocity slot (`observations.py:254`); record that
   the 41-dim privileged latent varies only in friction, so RMA-style adaptation currently has nothing to
   encode; pre-register gates (walls/field, tracking, max_stable, gait_eval, minimum NESM along rollouts).
1. **Pre-hardware stress sweep on frozen T* and the ONNX student, ≈ 7–13 GPU-h** (§10; [4, 42, 1, 79]).
   ≈ 12 terms × 5 levels × 2 policies ≈ 120 rows at 0.11 GPU-h, trimmable to ≈ 60. Outputs the tolerated band
   per term and the feasibility bounds for payload (campaign: 135 stable / 180 unstable) and push impulse.
   Highest value per hour because it decides which of rungs 3–5 to run and sizes them; the literature's claim
   that most DR buys nothing [4] is tested here, not assumed.
2. **One student relaunch bundling observation noise, current-proxy noise/bias/drift, velocity-slot
   corruption, depth-latency jitter 80–140 ms, depth corruption and the camera-pose fix, ≈ 10.2 GPU-h** (§4,
   §5, §8; [25, 28, 21, 24, 33, 47]). Teacher unchanged. Gate: no regression on the Task 4 thresholds. The depth
   terms are expected to read inert until scan use is demonstrated (campaign: "not detectable"), which is
   itself the read.
3. **One p2a → student rebuild bundling the actuator and inertial terms, ≈ 25 GPU-h** (§2, §3, §6, §9;
   [1, 18, 2, 38, 61, 62, 9, 39]). Force-dependent rod speed line via a `DCMotor`-style envelope; ×0.85–1.15
   scales on no-load speed, stall force and cam limits; first-order lag time constant and 0–40 ms per-env action
   delay; rod/linkage dead zone ramped from 0; per-joint offsets; body ±5–10 %, leg links ×0.85–1.15 with
   inertia, CoM ±3–5 cm, all written into the privileged latent; Miki ramp c_{k+1} = c_k^0.98. Disagreement:
   the identification-first recipes [10, 11] would fit rather than randomize, which is impossible before
   hardware; rung 1 decides which terms earn their width.
4. **Rescale the inherited Go2 terms in the same rebuild, 0 extra GPU-h** (§6; [26, 46, 45, 31, 39]). Push →
   additive, per-env 10–15 s intervals, specified in N·s from rung 1 and the NESM barrier; small multi-second
   body wrenches; effective friction floor to 0.3–0.4 via ground material or combine mode plus occasional
   slippery episodes; payload upper bound from rung 1 (narrow, not widen), anisotropic CoM, mid-episode step
   change. Disagreement: the payload literature says F19's range is already generous; the campaign's 135 kg
   ceiling agrees.
5. **Ground compliance, ≈ 0.5 GPU-h smoke, then folded into the next rebuild** (§7; [65, 68, 73]). Per-foot
   `compliant_contact_stiffness/damping` at 200 Hz; add only if the smoke is stable and rung 1 shows contact
   sensitivity; eval gains foot sinkage/clearance and per-foot load spread.
6. **Bench identification plan and first-walk log, hardware time only** (§2; [10, 3, 17, 21]). 20–60 s
   chirp/square/sine/noise excitation per actuator type under load, cam chirps through the linkage; fit speed
   line, efficiency, delay, compliance; narrow rung-3 ranges to the fit uncertainty; a < 5 min proprio +
   current log for distribution matching.
7. **Not recommended now**: ADR/DORAEMON (no legged precedent, 40 % env tax [51]); blanket Kp/Kd DR on the
   implicit PD; wider payload mass; a MuJoCo re-implementation before rungs 1–4 (high dev cost, second-order
   check [56, 22]).

Rungs 1–5 total ≈ 43–49 GPU-h, about one F19 pipeline.

**Where the literature disagrees with the campaign.** (i) The literature ranks observation noise and latency
as near-universal; the campaign has never tested either, so rung 2's expected value is high but its read is
one student relaunch. (ii) The literature treats strength DR as standard; the identification-first line says
fit instead; with no hardware the only option is modest DR around datasheet values, revisited after rung 6.
(iii) On pushes the literature and the NESM arithmetic agree the current term is 15–27× too strong; the
campaign has no survival measurement, hence rung 1. (iv) On depth augmentation the literature's effect sizes
are large [48] but the campaign found no detectable camera use, so it rides along in rung 2 rather than
leading. (v) On Kp/Kd the Go2 comment and the literature disagree in general, but for a position-mode screw
both point to compliance rather than gain scaling.

## 14. References

1. Tan, Zhang, Coumans, Iscen, Bai, Hafner, Bohez, Vanhoucke. *Sim-to-Real: Learning Agile Locomotion For Quadruped Robots.* RSS 2018. https://arxiv.org/abs/1804.10332 — Table I ranges; Sec. V-A actuator/latency; Sec. VI-B "not a free meal", "without either of them".
2. Hwangbo, Lee, Dosovitskiy, Bellicoso, Tsounis, Koltun, Hutter. *Learning agile and dynamic motor skills for legged robots.* Science Robotics 4(26) eaau5872, 2019. https://arxiv.org/abs/1901.08652 — "could not make a single step"; 3.55 → 0.740 N·m; ±15 % masses, ±2 cm; coupled-actuator caveat.
3. Lee, Kim, Kim, Park, Lee, Cho, Hwangbo. *Learning Quadrupedal Locomotion for a Heavy Hydraulic Robot Using an Actuator Model.* IEEE RA-L 10(12), 2025. https://arxiv.org/abs/2601.11143 — > 300 kg; 20 s fit; Table III RMSE 7.45 vs 36.0/33.3/27.4; stiff PD "restricted".
4. Xie, Da, van de Panne, Babich, Garg. *Dynamics Randomization Revisited: A Case Study for Quadrupedal Locomotion.* ICRA 2021, DOI 10.1109/ICRA48506.2021.9560837. https://arxiv.org/abs/2011.02404 — Table III kp = 160 / no-velocity failures; latency 17 → 32 ms; 0.9 vs 1.1 m/s; "superfluous"; Sec. IV-B sweep protocol.
5. Masuda, Takahashi. *Sim-to-Real Transfer of Compliant Bipedal Locomotion on Torque Sensor-Less Gear-Driven Humanoid.* Humanoids 2023. https://arxiv.org/abs/2204.03897 — directional efficiency as brake torque; DR 0/3 vs re-identified 3/3.
6. Shin, Song, Ji, Park. *Actuator-Constrained Reinforcement Learning for High-Speed Quadrupedal Locomotion.* arXiv:2312.17507, 2023. https://arxiv.org/abs/2312.17507 — Eq. 2 operating region; 4.5 vs 6.5 m/s; fell at 5 m/s.
7. MathWorks. *Linear Electric Actuator* (Simscape Electrical example). https://www.mathworks.com/help/sps/ug/linear-electric-actuator-motor-model.html — 26 mm/s no-load, 19 mm/s at 1000 N; "unknown friction levels".
8. Maslennikov et al. *Robust RL Control for Bipedal Locomotion with Closed Kinematic Chains.* arXiv:2507.10164, 2025. https://arxiv.org/abs/2507.10164 — serial simplification "significantly impairs sim-to-real transfer"; Table III.
9. Isaac Lab. `isaaclab/actuators/actuator_pd.py`, `actuator_pd_cfg.py` (v2.1.0 / main). https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/actuators/actuator_pd.py — DelayedPDActuator per-env lag at reset; RemotizedPDActuator lookup table; DCMotor four-quadrant curve.
10. Bjelonic, Tischhauser, Hutter. *Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots (PACE).* IJRR 2026; arXiv:2509.06342. https://arxiv.org/abs/2509.06342 — 4n+1 params by CMA-ES on 20–60 s in-air chirps; 13 robots "without randomization of dynamic parameters"; saturation enforced; 7.5 ms delay; MINIMAL/TYTAN; "URDF-only fails (caught by crane)".
11. Liao, Zhang, Huang, Huang, Li, Sreenath. *Berkeley Humanoid: A Research Platform for Learning-based Control.* arXiv:2407.21781, 2024. https://arxiv.org/abs/2407.21781 — "lazy approach"; "extensive domain randomization ... conservative"; Table 6 ranges and 8 noise terms.
12. Sobanbabu, He, He, Yang, Shi. *Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning (SPI-Active).* CoRL 2025; arXiv:2505.14266. https://arxiv.org/abs/2505.14266 — Heavy DR Table 1(b) 1.75/1.40/1.06/0.85/1.10.
13. Margolis, Agrawal. *Walk These Ways.* CoRL 2022. https://arxiv.org/abs/2212.03238 — actuator net + constant 20 ms delay; Table 6: strength 90–110 %, friction 0.40–1.00, restitution 0–1, gravity ±1, calibration ±0.02 rad.
14. Fey, Margolis, Peticco, Agrawal. *Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation.* arXiv:2502.10894, 2025. https://arxiv.org/abs/2502.10894 — Unitree B2 + Z1; UAN for the harmonic-drive arm; Table II leg stiffness ×0.9–1.1, damping ×0.5–1.5, stall ×0.9–1.1, link mass ×0.5–1.5, friction 0.5–4.0.
15. Cha, Cha, Shin, Kim, Park. *Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection.* arXiv:2504.06585 (preprint). https://arxiv.org/abs/2504.06585 — state-dependent torque perturbations.
16. He et al. *ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills.* RSS 2025; arXiv:2502.01143. https://arxiv.org/abs/2502.01143 — delta-action model.
17. Dao, Fern. *Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via Proprioceptive Distribution Matching.* arXiv:2604.11090, 2026. https://arxiv.org/abs/2604.11090 — no mocap; "less than five minutes of hardware data".
18. Kumar, Fu, Pathak, Malik. *RMA: Rapid Motor Adaptation for Legged Robots.* RSS 2021. https://arxiv.org/abs/2107.04034 — Table I (friction 0.05–4.5, Kp/Kd, payload 0–6 kg, strength 0.90–1.10, resample 0.004); 12 kg carried; Robust 62.4 vs 73.5 %.
19. Cheng, Shi, Agarwal, Pathak. extreme-parkour `legged_robot_config.py`, `legged_robot.py`, `rsl_rl/algorithms/ppo.py`. https://raw.githubusercontent.com/chengxuxin/extreme-parkour/main/legged_gym/legged_gym/envs/base/legged_robot_config.py — friction [0.6, 2], mass [0, 3], CoM ±0.2, push 0.5/8 s, strength 0.8–1.2, add_noise False, dis_noise 0, depth angle ±5; priv_explicit = base_lin_vel; estimator.
20. Siekmann, Godse, Fern, Hurst. *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition.* ICRA 2021. https://arxiv.org/abs/2011.01387 — Table I damping ×0.3–4.0, joint mass ×0.5–1.5, friction 0.35–1.1, encoder offset ±0.05 rad; no strength term.
21. Singh, Xie, Gergondet, Kanehiro. *Learning Bipedal Walking for Humanoids with Current Feedback.* IEEE Access 11, 2023, DOI 10.1109/ACCESS.2023.3301175. https://arxiv.org/abs/2303.03724 — k_bemf U[5, 40] every 100 ms; friction U(2, 8), U(0.5, 5); Policies A/B self-collide; "vital".
22. Gu, Wang, Chen. *Humanoid-Gym.* arXiv:2404.05695, 2024. https://arxiv.org/abs/2404.05695 — Table III delay 0–10 ms, strength 95–105 %, friction 0.1–2.0; Isaac Gym → MuJoCo sim2sim "lacking physical robots".
23. Gadde, Dugar, Malik, Fern. *No More Blind Spots.* arXiv:2508.11929, 2025. https://arxiv.org/abs/2508.11929 — Table II torque efficiency 0.9–1.0, torque delay 0.5–3 ms, map delay 20–100 ms; measured < 20 ms.
24. Zhuang, Fu, Wang, Atkeson, Schwertfeger, Finn, Zhao. *Robot Parkour Learning* + `a1_field_distill_config.py`. CoRL 2023. https://arxiv.org/abs/2309.05665 — Table 9 latencies, camera pose, FOV, strength 0.9–1.1, friction 0.5–1.0; depth noise/artefact config; "0.2 s ... 0.2 m".
25. Isaac Lab `velocity_env_cfg.py` and `config/go2/rough_env_cfg.py` (v2.1.0). https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py — Unoise scales; enable_corruption True; add_base_mass (−5, 5); Go2 (−1, 3), push None; base_com from v2.2.0.
26. Rudin et al. legged_gym `legged_robot_config.py`, `legged_robot.py`, `anymal_c_rough_config.py`. https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py — noise scales; friction [0.5, 1.25]; push 1 m/s / 15 s "emulates an impulse"; mass ±1 / ±5 kg; actuator net.
27. Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes.* CoRL 2021. https://arxiv.org/abs/2109.11978 — Sec. 3.3 friction U[0.5, 1.25], noise "based on real data", push ±1 m/s / 10 s; Table 4.
28. Ji, Mun, Kim, Hwangbo. *Concurrent Training of a Control Policy and a State Estimator.* RA-L 7(2), 2022, DOI 10.1109/LRA.2022.3151396. https://arxiv.org/abs/2202.05481 — Sec. II-C noise ±0.05 rad, ±0.5 rad/s, ±0.03, ±0.1 rad/s; Sec. III-B velocity "most important".
29. Agarwal, Kumar, Malik, Pathak. *Legged Locomotion in Challenging Terrains using Egocentric Vision.* CoRL 2022. https://arxiv.org/abs/2211.07638 — Table 2 noise σ on normalised obs; no linear velocity; camera 100 ± 20 ms, 10 ± 10 ms; Table 3 ranges; fixed calibrated camera.
30. Peng, Andrychowicz, Zaremba, Abbeel. *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization.* ICRA 2018. https://arxiv.org/abs/1710.06537 — Table III 0.89/0.29/0.25/0.64/0.48; LSTM generalises better.
31. Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning Quadrupedal Locomotion over Challenging Terrain.* Science Robotics 5(47), 2020, DOI 10.1126/scirobotics.abc5986. https://arxiv.org/abs/2010.11251 — student inputs "replaced with readings"; estimator; 10 kg = 22.7 % unsimulated; Table S2 friction N(0.3, 0.1); force decoder.
32. Yang, Yang, Wang. *Neural Volumetric Memory for Visual Locomotion Control.* CVPR 2023. https://arxiv.org/abs/2304.01201 — "only available in instrumented environments"; 1 % pixels to max.
33. Imai et al. *Vision-Guided Quadrupedal Locomotion in the Wild with Multi-Modal Delay Randomization.* IROS 2022. https://arxiv.org/abs/2109.14549 — Table V 444.7/859.9, 505.7/973.0; 3–30 pixels to max; no camera-pose DR.
34. Cheng, Shi, Agarwal, Pathak. *Extreme Parkour with Legged Robots.* ICRA 2024. https://arxiv.org/abs/2309.14341 — Sec. 4.1 constant 0.08 s / 0.016 s latency.
35. Wang, Chen, Han, Wu, Zhao. *Booster Gym.* arXiv:2506.15132, 2025. https://arxiv.org/abs/2506.15132 — 9–12 ms measured; 0–20 ms trained.
36. Hoeller, Rudin, Sako, Hutter. *ANYmal Parkour.* Science Robotics 9(88), 2024, DOI 10.1126/scirobotics.adi7566. https://arxiv.org/abs/2306.14874 — 250 → 25 ms; zero-delay training "does not seem to be affected"; map shift 7.5 cm; "key to make the pipeline robust".
37. Peng, Coumans, Zhang, Lee, Tan, Levine. *Learning Agile Robotic Locomotion Skills by Imitating Animals.* RSS 2020. https://arxiv.org/abs/2004.00784 — mass 0.8–1.2, inertia 0.5–1.5; robust "prone to falling"; no-DR fails.
38. Bellegarda, Chen, Liu, Nguyen. *Robust High-speed Running for Quadruped Robots via Deep RL.* IROS 2022. https://arxiv.org/abs/2103.06484 — each body part ±20 %; added mass 0–5 kg at ±[0.15, 0.05, 0.05] m, p = 0.8; 42 % on hardware.
39. Miki, Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning robust perceptive locomotion for quadrupedal robots in the wild.* Science Robotics 7(62), 2022. https://arxiv.org/abs/2201.08117 — body and leg masses; wrenches; occasional low friction; c_{k+1} = c_k^0.98; 60/30/10 height-sample mix.
40. Chang, Nai, Chen, Yang. *Beyond Robustness: Learning Unknown Dynamic Load Adaptation.* ICRA 2025. https://arxiv.org/abs/2507.07825 — Table III link ×0.8–1.2, base CoM ±5 cm, leg CoM ±1.5 cm, sliding load U(0.001, 8) kg; load estimator.
41. Chebotar et al. *Closing the Sim-to-Real Loop (SimOpt).* ICRA 2019. https://arxiv.org/abs/1810.05687 — "infeasible solutions ... exceedingly conservative policies".
42. Ma et al. *DrEureka.* RSS 2024. https://arxiv.org/abs/2406.01967 — RAPP one-at-a-time feasibility bounds; defaults "may significantly hamper policy learning".
43. Kurva, Kolathaya. *MULE.* arXiv:2505.00488, 2025 (preprint). https://arxiv.org/abs/2505.00488 — "overly conservative" (cited); masses resampled every 4 s from [0, 2.5] kg.
44. Ma, Farshidian, Miki, Lee, Hutter. *Combining Learning-based Locomotion Policy with Model-based Manipulation.* RA-L 2022. https://arxiv.org/abs/2201.03871 — acceleration-proportional wrench.
45. Campanaro, Gangapurwala, Merkt, Havoutis. *Learning and Deploying Robust Locomotion Policies with Minimal Dynamics Randomization (ERFI).* L4DC 2024, PMLR 242:578–590. https://proceedings.mlr.press/v242/campanaro24a.html — RAO emulates mass/inertia/gain shifts; 53 % over RFI; > 5 N / 3 N·m pronking; 0–150 N × 3 s evaluation only.
46. Isaac Lab `envs/mdp/events.py` (push_by_setting_velocity), `manager_term_cfg.py` (is_global_time). https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/events.py — `vel_w +=`; per-env intervals by default.
47. Rudin, He, Aurand, Hutter. *Parkour in the Wild.* arXiv:2505.11164, 2025. https://arxiv.org/abs/2505.11164 — five-step depth degradation; per-camera delay; ANYmal D.
48. Sun et al. *Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels.* arXiv:2602.06382, 2026. https://arxiv.org/abs/2602.06382 — Table V 98.9/90.4/92.1/94.5/96.2 (RDT-Bench, simulated).
49. Tiboni et al. *Domain Randomization via Entropy Maximization (DORAEMON).* ICLR 2024. https://arxiv.org/abs/2311.01885 — success-gated entropy growth; manipulation.
50. OpenAI (Akkaya et al.). *Solving Rubik's Cube with a Robot Hand.* arXiv:1910.07113, 2019. https://arxiv.org/abs/1910.07113 — single calibrated start; Sec. 8.2 training time vs entropy, "infeasible altogether" hypothesised.
51. Handa et al. *DeXtreme.* ICRA 2023. https://arxiv.org/abs/2210.13702 — 40 % envs for evaluation; Table 9 27.8 ± 19.0 vs 14.8 ± 5.4.
52. Fu, Cheng, Pathak. *Deep Whole-Body Control.* CoRL 2022. https://arxiv.org/abs/2210.10044 — realizability gap; ROA 0.31 → 2e-4; DR "just stands in place".
53. Chen, Hu, Jin, Li, Wang. *Understanding Domain Randomization for Sim-to-real Transfer.* ICLR 2022. https://arxiv.org/abs/2110.03239 — memory matters (theory).
54. Mehta, Diaz, Golemo, Pal, Paull. *Active Domain Randomization.* CoRL 2019, PMLR 100. https://proceedings.mlr.press/v100/mehta20a.html — uniform DR "suboptimal, high-variance".
55. Mishra, Raj, Xiao, Manocha. *TransCurriculum.* arXiv:2603.14156, 2026 (preprint). https://arxiv.org/abs/2603.14156 — Table IV 27/23/18 %.
56. Lei et al. *PolySim.* arXiv:2510.01708v3, 2025. https://arxiv.org/abs/2510.01708 — Table I 0.036 / 0.500 / 0.564; "inherit its bias".
57. Wu et al. *Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion (RoboGauge).* arXiv:2602.00678, 2026. https://arxiv.org/abs/2602.00678 — Table III 0.0558 vs 0.0883.
58. Muratore, Ramos, Turk, Yu, Gienger, Peters. *Robot Learning From Randomized Simulations: A Review.* Front. Robot. AI 2022. https://doi.org/10.3389/frobt.2022.799893 — Sec. 3.3 noise level too low "pointless", too high "will fail".
59. Kadian, Truong, Gokaslan, Clegg, Wijmans, Lee, Savva, Chernova, Batra. *Sim2Real Predictivity.* RA-L 2020. https://arxiv.org/abs/1912.06321 — SRCC_Succ 0.18 → 0.844 after simulator tuning.
60. Dowdy, Reizian, Chagas Vaz. *Loco-Loco-RL.* arXiv:2609.19041, 2026. https://arxiv.org/abs/2609.19041 — Table III joint max velocity ×(0.5, 1.5); OP3; no ablation.
61. Taylor et al. (Sony). *Learning Bipedal Robot Locomotion from Human Movement.* ICRA 2021. https://arxiv.org/abs/2105.12277 — dead-zone backlash 0 → 5° ramped; 1.5 kg; failure rises without DR.
62. Spinelli, Egli, Nubert, Hutter et al. *RL Control for Autonomous Hydraulic Material Handling Machines with Underactuated Tools.* IROS 2024. https://arxiv.org/abs/2410.05093 — Eq. 3 first-order lag + delay; parameters resampled per episode; > 40 t.
63. Yang et al. *HeLoM: Hierarchical Learning for Whole-Body Loco-Manipulation by a Hexapod Robot.* arXiv:2509.23651, 2025. https://arxiv.org/abs/2509.23651 — Table IV DR; 12 actuator nets; 0.105 vs 0.146 rad/s; pure ACNet jitter.
64. Hong, Kim, Lee, Kim, Oh. *Bridging the Sim-to-Real Gap in Parallel-Link Leg Mechanisms via Simulator-Side Dynamics Normalization (S3N).* arXiv:2608.01697, 2026. https://arxiv.org/abs/2608.01697 — −80.9 % / −82.1 % RMSE; 17.3 → 9.9 %; DR "does not explicitly restore".
65. Singh, Morisawa, Benallegue, Xie, Kanehiro. *Robust Humanoid Walking on Compliant and Uneven Terrain with Deep RL.* Humanoids 2024; arXiv:2504.13619. https://arxiv.org/html/2504.13619 — per-foot time constant (0.02, 0.4); 5/5 failure on rigid-trained; "redundant (and dangerous)".
66. Luo et al. *MILD: Tractable Terrain Modeling for Learning Improved Bipedal Locomotion on Deformable Surfaces.* RA-L 11(2); arXiv:2608.19955. https://arxiv.org/html/2608.19955 — Table II ranges; "insufficient foot clearance"; 1.2 m/s on seven terrains.
67. KAIST. *RaiBo press release* (EurekAlert 977777), 2023. https://www.eurekalert.org/news-releases/977777 — rigid-ground controller "rapidly decreases"; recurrent terrain inference (press account; paper not opened, see appendix).
68. Lee, Kim, Park, Park, Park. *Soft Contact Model for Robust Locomotion of Legged Robots.* ICRA 2025 workshop. https://iscicra25.github.io/papers/2025-Lee-4_Soft_Contact_Model_for_Robus.pdf — kmax 1.5–2.5× kmin; 5000 Hz; soft ratio ramped 4000–7000.
69. Kim, Lee. *Quadruped Locomotion on Non-Rigid Terrain using RL.* arXiv:2107.02955, 2021. https://arxiv.org/abs/2107.02955v1 — sink up to 5 cm; end-effector history "crucial".
70. Le Lidec, Jallet, Montaut, Laptev, Schmid, Carpentier. *Contact Models in Robotics: a Comparative Analysis.* IEEE T-RO; arXiv:2304.06372v3. https://arxiv.org/html/2304.06372v3 — jamming internal forces; PGS saturates.
71. University of Colorado Boulder course report. *Towards Effective Sim-to-Real Policy Transfer via Sim-to-Sim Transfer on Quadruped Robots* (non-peer-reviewed). https://lorinachey.github.io/assets/pdf/Sim2SimTransferQuadrupedRobots-1.pdf — load distribution "quite significantly"; "100s of Newtons"; Kp 40 → 100.
72. Liu, Liu, Liang, Kanoulas. *A Framework for Deploying Learning-based Quadruped Loco-Manipulation.* arXiv:2512.18938, 2025. https://arxiv.org/html/2512.18938 — PhysX yaw oscillation vs MuJoCo drift; impratio "traction tuning".
73. Isaac Lab docs. `isaaclab_physx.sim.spawners` PhysxRigidBodyMaterialCfg. https://isaac-sim.github.io/IsaacLab/main/source/api/lab_physx/isaaclab_physx.sim.spawners.html — compliant_contact_stiffness enabled when > 0; combine-mode priority; restitution default 0.
74. Kawawa-Beaudan. *Learning to Walk: Legged Hexapod Locomotion from Simulation to the Real World.* UC Berkeley EECS-2022-146. https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-146.pdf — noise on all state elements; friction 0.5–1.25; pushes every 8 s; "rubber-tipped feet grip"; 4 substeps.
75. (authors as listed). *Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion Priors.* arXiv:2511.03167, 2025. https://arxiv.org/html/2511.03167v1 — Table II ranges; 25.5 kg; "difficult to obtain accurate linear velocity".
76. Qu, Li, Zakhor, Yu, Zhang. *Versatile Locomotion Skills for Hexapod Robots.* arXiv:2412.10628, 2024. https://arxiv.org/html/2412.10628v1 — terrain-only randomization.
77. Qu. *Locomotion Skills for Reconfigurable Hexapod Robots.* UC Berkeley EECS-2025-150. https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-150.pdf — failure rate rises at ≈ 10 V vs 12 V.
78. Mouret, Koos, Doncieux. *Crossing the Reality Gap: a Short Introduction to the Transferability Approach.* arXiv:1307.1870, 2013. https://arxiv.org/abs/1307.1870 — 1294 vs 411 mm; slippage; static gaits transfer better.
79. Hirose, Lazarenko, Endo. *Directional Normalized Energy Stability Margin.* IFToMM Asian MMS 2012. https://t2r2.star.titech.ac.jp/rrws/file/CTT100645736/ATD100000413/ — NESM = ESM/mg; V = √(2E/m).
80. Garcia, Gonzalez de Santos. *An improved energy stability margin for walking machines subject to dynamic effects.* Robotica 23(1), 2005, DOI 10.1017/S0263574704000487 — "only exact stability margin when robot dynamics ... exist".
81. Garcia, Estremera, Gonzalez de Santos. *A comparative study of stability margins for walking machines.* Robotica 20(6), 2002, DOI 10.1017/S0263574702004502 — wrong criterion "may prevent the task from succeeding".
82. Haldane, Peterson, Garcia Bermudez, Fearing. *Animal-inspired Design and Aerodynamic Stabilization of a Hexapedal Millirobot.* ICRA 2013. https://people.eecs.berkeley.edu/~ronf/PAPERS/dhaldane-ICRA13.pdf — Table I dynamic scaling; Froude number.
83. Oke et al. *Allometric Scaling Laws for Bipedal Robots.* arXiv:2603.22560, 2026. https://arxiv.org/abs/2603.22560 — velocity ∝ L^0.5; τ ∝ mL.
84. Zimmer. *Robot on the Volcano.* Discover, 1995 (author's archive). https://carlzimmer.com/?p=9069 — Dante II "stepped on some particularly soft ash ... fell over".

## Appendix: unverified or dropped items

- **"F19's lead-screw model sits in the class the literature shows failing" / "transferred only with a
  physics-structured model".** Softened (§2): Lee et al.'s stiff PD baseline transferred and walked stably,
  limited on obstacles; Shin's no-envelope policy ran 4.5 m/s before falling; F19 has a delay and slew
  limiter, so it is not the ideal infinite-bandwidth model of [2]. Masuda, Shin and Maslennikov are analogues
  of Krabby's transmission, not instances of it [5, 6, 8].
- **Fey et al. [14] robot identity.** One verifier quotes the paper ("a Unitree B2 quadruped with a modified
  Unitree Z1 Pro arm"), another states "Spot" without a quote; the quoted version is used. Its
  stiffness/damping scales are PD-gain DR, not passive joint damping.
- **"Every paper kept strength scaling" and "every pipeline corrupts depth and randomizes camera pose".**
  Corrected: Cassie [20] and HRP-5P [21] have no strength term; Agarwal [29] renders from a fixed calibrated
  camera; MMDR, NVM and Parkour in the Wild have no camera-pose DR; Extreme Parkour's public depth noise is 0.
- **"Latency is simulated in every successful transfer".** Corrected: Xie's default Laikago policies (< 4 ms
  loop) and ANYmal Parkour's zero-delay perception transferred without it [4, 36]. Xie's "compromised
  performance" warning concerns a full-DR pace policy, not the latency-only one. MMDR "roughly doubled" holds
  on two of four courses [33].
- **RMA CoM "±0.15" as a footprint-scale offset.** Table I labels it centimetres; dropped from the CoM
  precedents (§6). Chang's ±5 cm is base-CoM, not payload placement [40]. Extreme Parkour's ±0.2 m is base CoM
  on an A1 whose ≈ 12 kg mass is general knowledge, not from the repo [19].
- **ERFI "0–150 N for 3 s" as a training push.** Evaluation only; ERFI trained with zero base force [45].
  Its 53 % figure is RAO/ERFI-50 vs RFI in a one-parameter-at-a-time simulation sweep.
- **Singh et al. 2023 "feedback only helps when the imperfection is randomized".** Inference; the paper never
  tested feedback without the counter-torque, and its ablation retrained a policy with the input zeroed [21].
- **Muratore et al. "It remains unclear whether this approach scales to sim-to-real scenarios".** Dropped:
  the sentence is about Mozifian et al.'s sim-return-driven parameter adaptation, not sim-to-sim validation
  [58]. "SOB" from the same review is not cited for lack of a locator.
- **Peng 2018 as a "highest-value" ranking for legged robots.** It is a Fetch-arm task with 12–28 trials per
  row; kept as suggestive only [30].
- **Isaac Lab v2.1.0 `base_com` ±5 cm.** Exists from v2.2.0 [25]. Rudin's paper Table 4 lists lin-vel noise
  ±0.01 m/s where the code uses 0.1 [27]. Minor locator slips (Tan Sec. V-A not IV-B/III-A; Ji Sec. II-C not
  II-E) are corrected in the text.
- **Peng 2020 adaptive 0.827.** After real-world latent adaptation (0.395 before) [37]. OpenAI's "infeasible
  altogether" is a stated hypothesis [50]. DeXtreme's ADR advantage is in the mean with medians ≈ equal [51].
- **Now You See That [48], PolySim [56], RoboGauge [57].** Simulated-benchmark or preprint numbers; PolySim's
  IsaacGym-only baseline already reaches 0.500, so the multi-simulator gain over the best single simulator is
  6.4 points; its real deployment used multi-simulator plus parameter DR.
- **Choi et al., *Learning quadrupedal locomotion on deformable terrain*, Science Robotics 8(74), 2023, DOI
  10.1126/scirobotics.ade2256.** Not opened (HTTP 403); known only through the KAIST press release [67]; its
  ranges are not cited.
- **Koos, Mouret, Doncieux, IEEE TEVC 17(1), 2013 (DOI 10.1109/TEVC.2012.2185849); Alexander & Jayes, J.
  Zool. 1983; Bares & Wettergreen, IJRR 18(7), 1999; Krotkov, Simmons, Whittaker, IAF 92-0735 (NTRS
  19920074535); Cully et al., Nature 2015.** Not opened or abstract-only with no DR content; dropped.
 
- **Loco-Loco-RL OP3 mass (≈ 3.5 kg) and HeLoM robot mass.** From memory / not stated in the papers; not used.
- **Berkeley friction "0.2–1.25" and the reviewer's footpad ground-pressure estimate (≈ 0.2–0.33 MPa).**
  Unreferenced or own arithmetic; not used.
- **Froude-scaled latency (≈ 35 ms) and push (≈ 0.87 m/s) for L ≈ 0.9 m.** Reviewer arithmetic on an assumed
  leg length; stated as not applicable to rate-limited actuators (§6) and not used for ranges.
- **Berkeley Humanoid "ICRA 2025" and Humanoid-Gym "ICRA 2024 workshop".** Venues not confirmed on arXiv;
  cited as arXiv.

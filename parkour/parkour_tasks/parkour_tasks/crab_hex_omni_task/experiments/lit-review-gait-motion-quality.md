# Literature review: suppressing excess leg motion and planting all six feet at zero command — mapped to the Krabby campaigns

*2026-09-22. Prepared for the `crab_hex_omni_task` after the facing/tracking ladder
(`experiments/2026-09-19_facing_tracking_ladder`, revision 3) ended with C1d as the incumbent
(X08:c6 `model_7996.pt`; D12 rejected by the driver, adopted by user overrule; C1e rejected, D13) and
the user reported "a lot of extra motion in the gait" with the stand category stepping instead of
planting. Builds on `crab_hex_forward_task/experiments/lit-review-hexapod-reward-stability.md`
(clock contact-schedule rewards; "no holdable state") and
`lit-review-omnidirectional-gait-formation.md` §4 / Rec. 4 (standstill income of the exp kernels;
Lee et al.'s 0.5-s stop, van Marum's stand constants, Isaac Lab's 2 % standing envs), which it does
not repeat. Every citation was opened by the review's verification pass and four load-bearing ones
were re-opened at writing time ([2, 19, 21, 28]); where verification corrected a sweep claim the
corrected wording is used, and items that failed verification or are only inferred are in the
appendix rather than softened.*

## 1. The problems this review targets

Restated from the campaign ledgers so this document stands alone:

1. **The stander steps in place.** On the per-category suite (2 s settle at the pinned yaw, then a
   12-s hold, 100 envs; `CHARTER.md:1032, 1077`) the incumbent X08:c6 scores stand survival 0.99,
   base drift 0.083 m/s, achieved speed 0.003 m/s, but a stand **tripod score of 0.587** against
   0.576 on the forward hold (`REPORT.md` X08:c6 block, lines 1020–1027). A tripod score is a
   stepping score; every ladder head shows it (stand 0.527–0.663, `REPORT.md:42–656`).
2. **No stand signal, and the facing loop keeps the clock alive.** The observation carries
   cmd = (0, 0, 0), the facing error and clock sin/cos, no stand bit. The facing controller runs in
   the stand slot, wz = clip(1.5·err, ±0.3) zeroed when |wz| ≤ 0.2 (`crab_hex_omni_command_math.py:
   213–258`; band 0.1333 rad); clock speed = max(‖cmd‖, 1.2·|wz|) with the stop at 0.2, so drift past
   0.133 rad issues a ≥ 0.2 rad/s turn command, restarts the clock and re-arms swing, apex and
   yaw-progress income; at a true stop the phase freezes where it was. Stand-hold facing rms 0.082
   rad, max 0.175 rad straddles the band.
3. **The economy pays a stepping stander.** `track_lin_vel_xy_exp` (σ² 0.04, w 1.25) and the L1
   term (−2.0) ignore leg motion at v = 0; `track_ang_vel_z_exp` (std 0.5, w 1.0) is near free;
   `clock_schedule` (product mode) returns stance quality alone at a stopped clock,
   loaded·exp(−(v_foot/0.1)²) (`crab_hex_clock_reward.py:113–127`), 0.96 at 0.02 m/s and only on the
   legs the frozen phase labels stance. Regularisers: `reward_action_rate` −0.3 (unsquared first
   difference), `dof_acc` −2.5e-7, `delta_torques` −1e-6. Absent: stand-gated joint deviation,
   joint velocity, jerk, torque/power, foot slip (`feet_slide` w 0).
4. **Extra motion everywhere, on a plant no reference stack has.** Linear actuators with slew
   0.028/0.033 m/s and rod clamps, velocity-commanded cams, 1-step delay, `clip_actions = 1.0`, no
   action filter. The sweep's repo probe found the PPO std parked at `ACTION_STD_MAX = 2.0`
   (`crab_actor_critic_with_encoder.py:10–27`; `init_noise_std` 1.5, `entropy_coef` 0.01): 2.0 by
   iteration 2872 of X01:1c and in every head since (appendix; not re-read here).
5. **C8** (stand share 0.15 → 0.05, `candidates.yaml` row 11) is untested; `drift_speed` is the only
   stand WATCH metric.

**The short answer.** Every stack that stands still makes stand a regime keyed on the sampled
command, not a frozen clock: the schedule goes all-stance, the clock reads zero, a stand flag is
observed, the heading loop is off, and posture/stillness/slip costs are scaled 5–20× inside that
regime (§2–§4). Smoothness rewards and loss-level regularisers reduce chatter, not the regime (§5).
Two plant-side causes no prior review examined — σ = 2.0 under a ±1 clip and absolute targets behind
a rate limiter — are cheap to test (§6). Nothing published scores stand at foot level (§8).

## 2. Stand as an explicit regime: schedule, clock and flag

**Key papers:** Margolis & Agrawal 2022 [1]; Portela et al. 2024 [2]; Isaac Lab Spot [3]; Siekmann
et al. 2021 [4]; Singh et al. 2022 [5]; Kim, Lee, Park 2025 [19]; Booster Gym [20]; Peng, Bao, Zhou
2025 [21]; Singh et al. 2025 [22]; Grandia et al. 2024 [23]; HugWBC [24]; OpenSHC / hexapod_ros /
Phoenix code [15–17]; Walknet [8]; Ijspeert et al. 2007 [25]; CPG-RL [26]; Sapkal et al. 2024 [27].

Clock-driven controllers step in place at zero command by construction: Walk These Ways advances
its clock from the commanded frequency with no velocity gate (it zeroes ‖v_cmd‖ ≤ 0.2 but keeps the
clock running) and shows gaits "in place" [1]; Humanoid-Gym and unitree_rl_gym take phase from
episode time [6, 7]. The fixes change the schedule. Portela et al.: "when the magnitude of the
commanded base velocity falls below 0.1, the desired contact schedule for each foot is set to zero,
ensuring all feet remain grounded", with stance foot-velocity income at weight 4.0 [2]. Spot returns
0 from `GaitReward` and switches air time to a bounded stance income
clip(contact_time − air_time, ±mode_time) when cmd == 0 and body speed ≤ 0.5 m/s [3]. Siekmann et al.
define standing as swing ratio ≈ 0, where ω ≈ 0 removes the force, speed and tracking costs and a
standing cost on action difference and foot symmetry is added [4]; Singh et al. expand double
support to the whole cycle [5]; HugWBC pins φ_i = 0.25 below φ_stance = 0.5 [24].

Recent stacks add the flag. Kim et al. observe "a stand-mode indicator" at ‖v_cmd‖ < 0.2; in stand
mode "barrier rewards for gait and foot clearance are adjusted to keep all legs in contact, and the
cyclic functions are set to zero" [19]. Booster Gym zeroes command AND `gait_frequency` for still
envs and multiplies the clock observation and swing masks by (gait_frequency > 1e-8) [20]. Peng et
al. feed a five-way one-hot gait ID and route rewards by it [21]; Singh et al. a three-way one-hot
[22]; Disney runs a perpetual standing policy beside a periodic one [23]. Entering stand is a
transition with a terminal state: Singh et al. switch "only when the clock signal is the
double-support region" [22]; Disney delays "until the start of the next double support phase" [23];
Peng et al. enter walk-to-stand below 0.1 m/s and Stand only after 1.5 s of low speed and double
support [21]. Classical engines agree: OpenSHC goes MOVING → STOPPING, has every leg attempt one
step to its default tip position, declares STOPPED when all legs are at phase, then FORCE_STOPs
each at phase 0 [15]; hexapod_ros runs one extra cycle "to allow all legs to set down" [16];
Phoenix walks until every leg is home [17]. Biology matches: duty factor rises tripod → tetrapod →
wave as speed falls [8]; salamander oscillators go sub-threshold below d_low [25]; fly brake neurons
let a leg "finish the swing phase in an unaltered manner, but freeze as soon as it lands" [27].

**What it means here.** Krabby is the un-fixed baseline: a frozen mid-cycle phase with up to three
legs labelled swing, no flag, income re-armed by drift. The precedented change is a stand slot that
advances to the next touchdown, parks all six legs at a stance phase, zeroes the clock observation,
carries a latched stand bit, and pays stance quality on all six legs with zero swing/apex/
yaw-progress income while parked [2, 3, 19, 20, 23]. Product-mode `clock_schedule` already has the
stance half to reuse.

## 3. Heading hold during stand and the inherited dead band

**Key papers:** Isaac Lab command / cfg / rewards [9, 10, 13]; Spot cfg [3]; Lee et al. 2020 [11];
legged_gym [12]; Rudin et al. 2022 [18]; OpenSHC paper + config [14]; Spot SDK [69].

No reference generator lets heading hold inject yaw during stand. Isaac Lab computes the heading
yaw rate, clip(stiffness·wrap(ψ* − ψ), ang_vel_z range), no dead band, then overwrites all three
components with 0 for standing envs (`rel_standing_envs` 0.02, stiffness 0.5 in the base config)
[9, 10]. Spot turns heading control off and keys stand on cmd exactly 0 AND body speed ≤ 0.5 m/s [3].
Lee et al. stop the gait generator after 0.5 s of zero command and restart it only on a direction
command or base speed above 0.3 m/s, "for the disturbance rejection" [11]. legged_gym keeps its
P-loop (gain 0.5, ±1, no dead band) on every env but gates stand_still and air-time income on
‖cmd_xy‖ alone, so a heading-correcting zero-linear env earns no stepping income [12, 13]. Krabby's
per-step relay (P → saturate → zero below the clip) is inherited, not validated (appendix).
Classical engines shape the command that decides whether to walk: OpenSHC rate-limits desired
velocities by max acceleration [15]; hexapod_ros filters cmd_vel (α 0.05) and walks only above
1 mm or 0.25° per cycle [16]; Phoenix zeroes travel below a dead zone [17]. Rudin et al. pay the goal
as 1/(1+d²) only in a final window "long enough to force the policy to stop at the target in a
stable configuration" [18]. A regime keyed on a state the policy controls invites gaming; published
gates key on the command with at most a disturbance-sized escape [3, 11]. Small yaw over planted
feet is classically a body-pose problem: OpenSHC poses the body in 6 DoF within `max_rotation`
limits [14]; Spot's stand sets body yaw in the footprint frame (0.4 rad example) [69]; Walk These
Ways has no body-yaw command [1]. No RL precedent for yaw-by-posing was found.

**What it means here.** Stand facing max 0.175 rad exceeds the 0.133-rad band, so the relay fires
inside the hold. Cheapest fix: wz = 0 and clock speed 0 for `SLOT_STAND` after the facing controller
runs, re-anchoring at the next resample [9, 3]. If heading hold stays: drop the 1.2·|wz| clock term
on stand rows and use hysteresis (engage ≈ 0.25, release ≈ 0.05 rad) or a slew-limited wz [15, 16];
any body-speed escape must sit well above drift plus noise excursions [3, 11].

## 4. Stand-gated posture, stillness and slip terms

**Key papers:** Spot `joint_position_penalty` [3]; Siekmann et al. [4]; Li et al. 2024 [28]; Peng et
al. [21]; Hwangbo et al. 2019 [31]; Isaac Lab `feet_slide` [13]; Humanoid-Gym [6]; RMA [32]; Booster
Gym [20]; legged_gym [12]; OpenSHC [14]; Walknet [8]; Ng et al. 1999 [29]; Jeon et al. [30].

Stand-specific terms are strengthened inside the regime and pull toward a nominal pose: Spot
multiplies its joint-deviation penalty (plain L2, w −0.7) by `stand_still_scale = 5` [3]; Siekmann's
standing cost 1 − exp(−(err_sym + 20·q_action_diff)) acts only where ω ≈ 0 [4]; Li et al. switch the
reference to a stacked standing pose and "the weights of smoothing terms, such as motor velocity
and change of action, are increased" [28]; Peng et al. route Contact 2.5, Base motion 2.5 and
Stillness 2.0 to the Stand ID [21]. Classical stand is the default foot position, "the position of
the foot with zero body velocity" [14], held under negative-feedback position control [8].
Potential-based shaping preserves the optimum [29] with marginal convergence gains [30] — our
inference: shaping alone cannot remove stepping in place.

Slip is a contact-gated, gait-independent foot-velocity term whose norm matters near zero: Hwangbo
k_c·2.0·dt·‖v_ft‖ [31]; Isaac Lab `feet_slide` and Spot `foot_slip` the unsquared planar norm,
contact = max force over history > 1 N, w −0.5 [13, 3]; Humanoid-Gym sqrt(speed), F_z > 5 N [6];
Walk These Ways squared, −8e-4 paper / −0.04 code [1]; RMA −‖diag(g)v_f‖² at 0.8 [32]; Booster
1_stance‖v‖² at −0.1 [20]. Unsquared and sqrt forms keep a marginal penalty for tiny shuffles;
squared forms and exp(−(v/σ)²) income are flat at v = 0. legged_gym's stumble and contact-force
terms target scuffing, not stillness [12].

**What it means here.** Krabby has no stand-conditioned term. A stand-slot-gated deviation toward a
planted six-leg pose (rods inside clamps) at 5–20× the walking weight, stand-scaled action-rate and
joint-velocity weights [3, 21, 28], and an unsquared planar slip penalty on all six loaded legs
regardless of phase (gate on the ≈ 50 N `TOUCH_REF_N`, not a contact bit) target drift 0.083 m/s
directly; the stance kernel's 0.96 at 0.02 m/s is why the current term cannot.

## 5. Motion regularisers: reward terms, curricula, loss-level smoothness, filters

**Key papers:** Walk These Ways [1]; HIMLoco/DreamWaQ [33]; Kim et al. T-RO 2024 [34]; Disney [23];
Solo12 [35]; Booster Gym [20]; Fu et al. 2021 [36]; RMA [32]; Hwangbo [31]; legged_gym / Isaac Lab
[12, 37]; CAPS [38]; LCP [39]; Shin et al. [40]; Christmann et al. 2024 [41]; Peng et al. 2020 [42];
gSDE [43]; L2C2-v2 [44]; MELA [49]; BRUCE [50]; Isaac Lab EMA action [51].

Reference pipelines carry a fixed set Krabby lacks: squared first difference, second difference
‖a_t − 2a_{t−1} + a_{t−2}‖², joint velocity/acceleration, torque or power. Walk These Ways uses both
smoothing orders (−2e-3 paper / −0.1 code) plus joint velocity under r_task·exp(c_aux r_aux) [1];
HIMLoco/DreamWaQ action rate −0.01, smoothness −0.01, joint acc −2.5e-7, power −2e-5 [33]; Kim et
al. call the first+second difference "crucial for preventing motor vibration" [34]; Disney leg
action rate 1.5, acceleration 0.45 [23]; Solo12 both orders "without vibrations and jitter" [35];
Booster power max(τq̇, 0), joint vel/acc, action rate −1.0 [20]. Fu et al. show one energy term
−τᵀq̇ removes "high-frequency tapping" and minimises the other penalties "as a byproduct" [36].
Forms differ (legged_gym/Isaac Lab squared, action_rate −0.01; Spot unsquared, −1.0) [12, 37, 3].
Strong costs early create a stand-still attractor, so Hwangbo (k0 0.3) and RMA (0.03) anneal
k ← k^0.997 and Solo12 ramps linearly [31, 32, 35]; Isaac Lab ships `modify_reward_weight` [37].

Loss-level regularisers smooth the state-to-action map without a reward the kernels can outbid:
CAPS temporal + spatial terms (≈ 80 % quadrotor power cut) [38]; LCP λ_gp·E‖∇_s log π‖² at 0.002
(0.001 jittery, 0.01 sluggish), humanoid action jitter 3.21 vs 5.74 (smoothness reward), 7.86
(filter), 42.19 (none); filtered policies "tend to exhibit lower task returns" [39]; spectral
normalisation matches it with less memory [40]. Christmann et al. score an FFT weighted-mean
frequency, found L2C2 / LipsNet+L2C2 the best real-quadruped trade-off, and reject filters because
a filter "has memory" [41]. Filters stay common: Butterworth on PD targets [42]; MELA 5 Hz sim /
3 Hz real, which "only limits the frequency ... but not the magnitude", plus an actor-loss term
‖μ(s_t) − q‖² [49]; Li et al. 4 Hz, filter and smoothing rewards "complement each other" [28]; BRUCE
second-order plus dead band [50]; Isaac Lab EMA action (α 0.95 in-hand; state unobserved) [51].
Per-step Gaussian exploration "leads to jerky motion patterns" [43]; L2C2-v2 (2026) reports smoother
Go2 walking under sudden commands [44].

**What it means here.** A second-difference term is free from the 8-step action history and taxes
the reversals of a leg bobbing in place; a power term (rod force × rod velocity, not the cams) taxes
stepping while the kernels pay full [36]. The slew limiter is a rate clamp, not a low-pass: sign
flips pass at full rate, so dither becomes a rod sawtooth; a first-order filter before the linkage
with the filtered target observed, paired with a moderate squared action-rate, is the recipe of
[28, 49], and a "rod lead" cost |λ_cmd − λ_rod| mirrors Isaac Lab's `applied_torque_limits` [37].
LCP/L2C2 add no lag on the 1-step delay [39, 41]; a coherent 1–2 Hz step-in-place is low-frequency,
so none fixes the regime (inference). Iteration ramps must survive the counter-resetting resume path.

## 6. Action space, clipping and exploration noise

**Key papers:** Peng & van de Panne 2017 [52]; Aljalbout et al. 2024 [53]; Fujita & Maeda 2018 [54];
Chou et al. 2017 [55]; rsl_rl [56]; Isaac Lab PPO cfgs [57]; Seyde et al. 2021 [58]; Andrychowicz et
al. 2021 [59]; Sabatini, Li, Hutter 2026 [60]; Raffin [61]; Hollenstein et al. 2024 [62]; Lattice
[63]; FPO++ [64].

Action space sets smoothness by construction: PD targets were best and most robust [52]; one-step
delta spaces bound the target change by c·dt, joint-velocity spaces have the fewest jerk violations
and nominal joint position the most, multi-step integrators "consistently perform the worst" [53].
Tight clips bias a Gaussian policy (Beta is "bias-free" [55]; CAPG unbiased [54]); reference stacks
clip at 100 and 10 [12, 1]. The noise itself is a candidate cause: rsl_rl, legged_gym and Isaac Lab
start at σ 1.0 with no clamp and entropy 0.005–0.01 [56, 12, 57]; Andrychowicz et al. find
performance "very sensitive to the initial action standard deviation", 0.5 best, no benefit from
entropy, tanh better than clipping [59]; Li et al. fix σ = 0.1·I [28]; Sabatini et al. show σ0 = 1
puts mass at the bounds while 0.15 converges faster [60]; a trained A1 policy "uses less than 5 % of
the action space" [61]; Seyde et al. document the boundary preference [58]. At σ = 2 with clip ±1,
62 % of samples saturate, a mean of 0.5 executes as 0.19, and noise alone gives a step-to-step rms
≈ 1.2 on a range of 2 (0.14 at σ 0.1) — recomputed here. A forward-pass clamp passes no gradient to
a parameter beyond the bound. Correlated noise (gSDE, coloured β ≈ 0.5, Lattice) trades performance
for smoothness but is unvalidated for PPO legged locomotion and adds low-frequency heading
excursions at stand [43, 62, 63]. Evaluation is the deterministic mean everywhere and the
stochastic/deterministic gap "can be drastic" [56, 43, 64], so 0.587 is a property of the mean.

**What it means here.** In training the stand regime is never quiet: near-bang-bang actuation
produces the drift that re-arms the relay and the clock, and turns the −0.3 action-rate cost into a
near-constant with little gradient on the mean. Confirm `policy.std` on X08:c6, then screen a lower
cap or fixed/annealed σ 0.3–0.5, entropy ≤ 0.005, or a gradient-passing clamp, one variable per run.
`clip_actions = 1.0` is load-bearing for the student, so a Beta/tanh head needs its own screen. A
one-step delta action scaled to the rod slew makes action = 0 mean "hold" [53]; the cam channels
already integrate any nonzero mean into leg drift.

## 7. Training stages and reward aggregation

**Key papers:** Li et al. [28]; Peng et al. [21]; stand fractions [10, 3, 20]; Kim et al. multi-critic
[19]; Escontrela et al. 2022 [45]; Liu et al. 2025 [46].

Standing is trained either as a late sub-stage — Li et al. command stand after a random walking
span to episode end, later for a random mid-episode span, and the joint policy steps to recover
from pushes where stand-only policies fall [28]; Peng et al. add Standing/Walk-to-Stand as Phase 2
[21] — or from scratch as a regime (Isaac Lab 2 %, Spot 10 %, Booster 0.1; Kim's stand mode is in the
main recipe) [10, 3, 20, 19]. No ablation of stand fraction against stand quality was found. Kim et
al. keep sharp style penalties from swamping tracking with separate critics and two equally
weighted standardised advantages, R_standard = r_pos·exp(0.2·r_neg) [19]. Without a style reward a
policy "learns to move by vibrating its legs at high speeds with large torques"; AMP removes this at
COT 0.93–1.12 vs 1.37–1.54 hand-designed [45]. The one hexapod AMP controller has no stand or slip
term and warns quadruped rewards "typically fail to develop the tripod gait" [46].

**What it means here.** A stand-stabilisation stage forked from X08:c6 (accepting the priv_reg
reset on resume) fits [28]; the 10-s resample in 40-s episodes already exercises both transitions.
C8 must be gated on stand quality. A second advantage group is the documented way to add
stand/smoothness terms without re-tuning tracking, but touches rsl_rl [19].

## 8. Measuring stand

**Key papers:** Rytz et al. 2025 [47]; Roditis et al. 2016 [48]; Christmann et al. [41]; van Marum et
al. 2024 [65]; LCP metrics [39]; Panchal et al. 2026 [66]; Yu et al. [67]; Crowley et al. 2025 [68];
Jiang et al. 2026 [70].

Canonical stand evaluations score the base only: Rytz's ν_zero = 1 if ‖v_{x,y,ω}‖ < 0.2 during a zero
command gates their curriculum (Tr_zero > 0.55) [47]; van Marum et al. score impulse trials without
a fall (79–214 N, 200–500 ms) and set the stand contact reward to a constant 1, "giving no preference
for foot contact" [65]; Isaac Lab logs no standing split [9]. Joint-level metrics exist for walking
only: action and DoF-position jitter, mean DoF velocity, energy over 1000 envs × 10 s [39];
per-body-group action rate and DoF acceleration [66]; the FFT weighted-mean frequency [41]. Yu et
al. give a foot template: s_f = 1 − (1/8)Σ|p_f,i − p̂_f,i|/d̂ against nominal stance and
s_r = 1 − T/T̂ [67]; the static margin is the COM projection's distance to the support-polygon edge
[48]. Crowley et al. switch into standing at swing apex [68]. The 2026 hexapod DRL-CPG paper
evaluates no stand regime [70].

**What it means here.** Krabby's stand passes every published bar (drift 0.083 < 0.2; survival 0.99)
while stepping in place; foot-level metrics are a new measurement.

## 9. Literature gaps

- No ablation of standing-env fraction against stand quality [10, 3, 20].
- No foot-level stand metric anywhere; the 2026 hexapod DRL work skips stand [65, 47, 70].
- No RL precedent for absorbing heading error by body posing, none pairing a dead band with a
  stand regime [14, 69].
- No source tunes a body-speed re-entry threshold against exploration noise [3, 11].
- No ablation of an action filter against an action-rate reward; none observes filter state [51, 49].
- No loss-level smoothness evidence on slew-limited linear-actuator plants [39, 41].
- No hexapod-specific stand recipe [46].

## 10. Ranked recommendations

0. **Before any training change: stand-hold metrics and diagnostics** (§8; [67, 39, 41, 47]). Per
   hold: fraction of time with all six feet loaded (reuse `TOUCH_REF_N` ≈ 50 N), lift-off /
   contact-switch events per second, settle time to six-down over the 2-s window, end-of-hold foot
   displacement from the settle footholds [67], mean |q̇| and rod speed, FFT mean frequency of
   actions [41], min static margin [48], fraction of stand steps with wz ≠ 0 and clock restarts per
   hold; run the hold with the mean and with actions sampled at the training σ [64]. These gate
   every item below and C8. **Every new reward term passes the offline reward-replay gate first**
   (replayed on saved healthy and degenerate npz; healthy earns a meaningful share, stepping
   standers ≈ 0) before any training screen.
1. **Silence the facing loop and clock in the stand slot** (§3; [9, 3, 11]). wz = 0 and clock speed
   0 for `SLOT_STAND` after `heading_rate_command`; re-anchor at the next resample. Targets the
   0.175-rad max crossing the 0.133-rad band. Cost: command-term edit and a unit test. Alternative:
   hysteresis 0.25 / 0.05 rad and no 1.2·|wz| term on stand rows [15, 16].
2. **Stand as schedule plus latched bit** (§2; [2, 19, 20, 23]). Advance to the next touchdown,
   park six legs at stance phase, zero clock sin/cos, observe a stand bit; pay stance quality on all
   six and zero swing/apex/yaw-progress income while parked; re-seed tripod offsets on exit.
   Targets 0.587 ≈ 0.576. Cost: clock-reward and observation edits, fine-tune from X08:c6.
3. **Confirm and cap the exploration std** (§6; [59, 60, 28, 56]). Read `policy.std` on X08:c6; if
   2.0, screen `ACTION_STD_MAX` 0.5 or fixed/annealed σ 0.3–0.5, entropy ≤ 0.005, gradient-passing
   clamp, one variable per run. Prerequisite for any smoothing term to bite.
4. **Stand-gated posture, stillness and slip** (§4; [3, 28, 21, 13]). Deviation toward a planted
   pose at ×5–20 in stand, stand-scaled action-rate and joint-velocity weights, unsquared planar slip
   on all six loaded feet (≈ Spot −0.5 walking, higher at stand). Targets drift 0.083 m/s.
5. **General smoothness** (§5; [1, 34, 36, 39]). Second-difference and power terms, ramped or
   fine-tuned; then LCP λ_gp ≈ 0.002 or L2C2 in the rsl_rl loss. Chatter reduction, not a stand fix.
6. **Actuator-side parameterisation** (§5–§6; [53, 49, 28, 51]). First-order filter before the rod
   slew with the filtered target observed, rod-lead cost, one-step delta action scaled to the slew;
   each its own screen because `clip_actions = 1.0` is load-bearing.
7. **C8** only after item 0, scored on stand quality; multi-critic [19] and AMP [45, 46] as fallbacks.

Where campaign and literature disagree: published gates key on the command with a body-speed escape
[3, 11], whereas Krabby's re-entry is driven by its own heading; Spot's 0.5 m/s escape would exceed
this plant's walking speed (0.19–0.22 m/s) and must be re-sized; van Marum's constant contact reward
deliberately permits recovery steps at stand [65], the opposite of the user's request, which argues
for a bounded stance income [3] rather than a hard all-six-down constraint.

## 11. References

1. Margolis, Agrawal. *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior.* CoRL 2022; arXiv:2212.03238. https://arxiv.org/abs/2212.03238 — Fig. 2 gaits "in place"; Table 1 smoothing/slip/joint-velocity weights; behaviour vector without body yaw. Code: `go1_gym/envs/base/legged_robot.py` (L833 clock without gate; L820 ‖v‖ ≤ 0.2 zeroing), `corl_rewards.py` (`_reward_feet_slip`, `_reward_action_smoothness_2`), `scripts/train.py` (weights −0.1/−0.1/−1e-4/−0.04; clip_actions 10). https://github.com/Improbable-AI/walk-these-ways
2. Portela, Margolis, Ji, Agrawal. *Learning Force Control for Legged Manipulation.* ICRA 2024; arXiv:2405.01402. https://arxiv.org/html/2405.01402v2 — Sec. IV-B contact schedule set to zero below 0.1; Table I stance 4.0 / swing 0.9.
3. Isaac Lab. `locomotion/velocity/config/spot/mdp/rewards.py` and `config/spot/flat_env_cfg.py`, main. https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py — `air_time_reward`, `GaitReward`, `joint_position_penalty`, `foot_slip_penalty` gated on logical_or(cmd > 0, body_vel > 0.5); cfg weights 5.0 / 10.0 / −0.7 (stand_still_scale 5) / −0.5; heading off, rel_standing_envs 0.1.
4. Siekmann, Godse, Fern, Hurst. *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition.* ICRA 2021; arXiv:2011.01387. https://arxiv.org/abs/2011.01387 — standing region, ω sigmoid, standing cost 0.1(ω−1).
5. Singh, Benallegue, Morisawa, Cisneros, Kanehiro. *Learning Bipedal Walking On Planned Footsteps For Humanoid Robots.* Humanoids 2022; arXiv:2207.12644. https://arxiv.org/abs/2207.12644 — Sec. IV-C quiet standing.
6. Gu, Wang, Chen. Humanoid-Gym `humanoid/envs/custom/humanoid_env.py` (arXiv:2404.05695). https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_env.py — `_get_phase`, `_reward_feet_contact_number`, `_reward_foot_slip`.
7. Unitree Robotics. unitree_rl_gym `legged_gym/envs/g1/g1_env.py`. https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_env.py — phase from episode time; `_reward_contact`.
8. Schilling, Hoinville, Schmitz, Cruse. *Walknet, a bio-inspired controller for hexapod walking.* Biol. Cybern. 107:397–419, 2013. https://doi.org/10.1007/s00422-013-0563-5 (full text PMC3755227) — gait continuum; standing vs walking modes.
9. Isaac Lab. `envs/mdp/commands/velocity_command.py`, main. https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py — heading P-control then standing-env zeroing; metrics.
10. Isaac Lab. `locomotion/velocity/velocity_env_cfg.py`, main. https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py — L96–103 command cfg.
11. Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning Quadrupedal Locomotion over Challenging Terrain.* Science Robotics 5(47), 2020; arXiv:2010.11251. https://arxiv.org/abs/2010.11251 — FTG state machine 0.5 s / 1.25 Hz / 0.3 m/s; stop reward.
12. Rudin, Hoeller, Reist, Hutter. legged_gym `legged_robot.py`, `legged_robot_config.py`, master. https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py — L330 heading loop; L351 zeroing; `_reward_stand_still`, `_reward_feet_air_time`, `_reward_stumble`, limit terms; scales action_rate −0.01, dof_acc −2.5e-7, torques −1e-5, stand_still 0; init_noise_std 1.0, entropy 0.01, clip_actions 100.
13. Isaac Lab. `locomotion/velocity/mdp/rewards.py`, main. https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py — `stand_still_joint_deviation_l1` (0.06), `feet_air_time` (> 0.1), `feet_slide` (1 N).
14. Tam, Talbot, Steindl, Elfes, Kottege. *OpenSHC: A Versatile Multilegged Robot Controller.* IEEE Access 8, 2020; arXiv:2006.04424. https://arxiv.org/abs/2006.04424 — default foot position; pose controller; walkspace. Config: https://github.com/csiro-robotics/syropod_highlevel_controller/blob/master/config/readme.md — max_rotation, max_rotation_velocity 0.01.
15. CSIRO Robotics. `syropod_highlevel_controller/src/walk_controller.cpp`, master. https://github.com/csiro-robotics/syropod_highlevel_controller/blob/master/src/walk_controller.cpp — acceleration limits L507–526; MOVING/STOPPING/STOPPED, FORCE_STOP, setPhase(0.0).
16. Ochs. `hexapod_ros/hexapod_controller/src/gait.cpp`, master. https://github.com/KevinOchs/hexapod_ros/blob/master/hexapod_controller/src/gait.cpp — α 0.05 filter; 1 mm / 0.25° travel; extra set-down cycle.
17. Lynxmotion. `BotBoarduino_CH3R_PS2.ino` (Phoenix V2.0, Janssen/Eckhardt). https://github.com/Lynxmotion/3DOF-4DOF-Hex/blob/master/BotBoarduino/3%20DoF%20Ready/BotBoarduino_CH3R_PS2/BotBoarduino_CH3R_PS2.ino — cTravelDeadZone; return-to-home; fContinueWalking.
18. Rudin, Hoeller, Bjelonic, Hutter. *Advanced Skills by Learning Locomotion and Local Navigation End-to-End.* IROS 2022; arXiv:2209.12827. https://arxiv.org/abs/2209.12827 — Eq. (1); T_r stop condition.
19. Kim, Lee, Park. *A Learning Framework for Diverse Legged Robot Locomotion Using Barrier-Based Style Rewards.* ICRA 2025; arXiv:2409.15780v4. https://arxiv.org/html/2409.15780v4 — stand-mode indicator (< 0.2), zeroed cyclic functions; multi-critic; R_standard.
20. Wang, Chen, Han, Wu, Zhao. *Booster Gym.* arXiv:2506.15132, 2025. https://arxiv.org/abs/2506.15132 ; `envs/t1.py`, `envs/T1.yaml` https://github.com/BoosterRobotics/booster_gym/blob/main/envs/t1.py — still_envs zero cmd + gait_frequency; clock obs gate; Table II weights; still_proportion 0.1.
21. Peng, Bao, Zhou. *Gait-Conditioned RL with Multi-Phase Curriculum for Humanoid Locomotion.* arXiv:2505.20619v3, 2025. https://arxiv.org/html/2505.20619v3 — one-hot gait ID; W2S < 0.1 m/s, Stand after 1.5 s; standing rows 2.5/2.5/2.0.
22. Singh, Morisawa, Benallegue, Xie, Kanehiro. *Robust Humanoid Walking on Compliant and Uneven Terrain with DRL.* arXiv:2504.13619, 2025. https://arxiv.org/html/2504.13619 — 3-D one-hot; Standing switch only in double support.
23. Grandia, Knoop, Hopkins, Wiedebach, Bishop, Pickles, Müller, Bächer. *Design and Control of a Bipedal Robotic Character.* RSS 2024; arXiv:2501.05204. https://arxiv.org/html/2501.05204 — perpetual standing + periodic walking policies; switch at next double support; action rate 1.5 / acc 0.45.
24. Xue, Dong, Liu, Zhang, Pang. *HugWBC.* arXiv:2502.03206v2. https://arxiv.org/html/2502.03206v2 — φ_i = 0.25, φ_stance = 0.5 for standing.
25. Ijspeert, Crespi, Ryczko, Cabelguen. *From Swimming to Walking with a Salamander Robot Driven by a Spinal Cord Model.* Science 315:1416–1420, 2007. https://faculty.washington.edu/minster/bio_inspired_robotics_2022/papers/Ijspeert_crespi_Ryczko_cabelguen_swimming_walking_salamander_science07.pdf — sub-threshold phase, d_low/d_high.
26. Bellegarda, Ijspeert. *CPG-RL.* IEEE RA-L 2022; arXiv:2211.00458. https://arxiv.org/html/2211.00458 — ω ∈ [0, 4.5] Hz; Eq. 8.
27. Sapkal, Mancini, Sthanu Kumar, … Bidaye. *Neural circuit mechanisms underlying context-specific halting in Drosophila.* Nature 634, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11446846/ — swing finishes, freeze on landing.
28. Li, Peng, Abbeel, Levine, Berseth, Sreenath. *RL for Versatile, Dynamic, and Robust Bipedal Locomotion Control.* IJRR 2024; arXiv:2401.16889. https://arxiv.org/abs/2401.16889 — App. A action filter (4 Hz) ablation; App. B standing sub-stage, raised smoothing weights; fixed std 0.1·I; Fig. 11 push recovery.
29. Ng, Harada, Russell. *Policy Invariance under Reward Transformations.* ICML 1999. https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf — Theorem 1.
30. Jeon, Heim, Khazoom, Kim. *Benchmarking Potential Based Rewards for Learning Humanoid Locomotion.* arXiv:2307.10142. https://arxiv.org/abs/2307.10142 — marginal convergence benefit, scaling robustness.
31. Hwangbo et al. *Learning agile and dynamic motor skills for legged robots.* Science Robotics 2019; arXiv:1901.08652. https://arxiv.org/abs/1901.08652 — slip cost 2.0·dt; k0 0.3, k_d 0.997; standing under high penalties.
32. Kumar, Fu, Pathak, Malik. *RMA.* RSS 2021; arXiv:2107.04034. https://arxiv.org/abs/2107.04034 — foot slip 0.8; work/smoothness/joint-speed; k0 0.03.
33. Long, Wang, Li, Gao, Cao, Pang. *Hybrid Internal Model (HIMLoco).* arXiv:2312.11460. https://arxiv.org/abs/2312.11460 — Table 5 (same values in DreamWaQ, arXiv:2301.10602 Table I).
34. Kim, Oh, Lee, Choi, Ji, Jung, Youm, Hwangbo. *Not Only Rewards But Also Constraints.* IEEE T-RO 2024; arXiv:2308.12517. https://arxiv.org/abs/2308.12517 — Table I action smoothness; motor vibration.
35. Aractingi, Léziart, Flayols, Perez, Silander, Souères. *Controlling the Solo12 Quadruped Robot with DRL.* arXiv:2309.16683. https://arxiv.org/abs/2309.16683 — first/second differences; linear k_c ramp.
36. Fu, Kumar, Malik, Pathak. *Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots.* CoRL 2021; arXiv:2111.01674. https://arxiv.org/abs/2111.01674 — Eq. 5; Fig. 7–8; Sec. 4.4.
37. Isaac Lab. `isaaclab/envs/mdp/rewards.py` and `envs/mdp/curriculums.py`, main. https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/rewards.py ; https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/curriculums.py — squared terms; `applied_torque_limits`; `joint_vel_limits`; `modify_reward_weight`.
38. Mysore, Mabsout, Mancuso, Saenko. *Regularizing Action Policies for Smooth Control (CAPS).* ICRA 2021; arXiv:2012.06644. https://arxiv.org/abs/2012.06644
39. Chen, He, Wang, Liao, Ze, Li, Sastry, Wu, Sreenath, Gupta, Peng. *Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies.* arXiv:2410.11825. https://arxiv.org/html/2410.11825 — λ_gp 0.002; Table I jitter; metrics protocol.
40. Shin, Cha, Kim, Cha, Park. *Spectral Normalization for Lipschitz-Constrained Policies.* arXiv:2504.08246. https://arxiv.org/abs/2504.08246
41. Christmann, Luo, Mandala, Chen. *Benchmarking Smoothness and Reducing High-Frequency Oscillations in Continuous Control Policies.* IROS 2024; arXiv:2410.16632. https://arxiv.org/html/2410.16632 — Sm metric; L2C2 trade-off; filter memory.
42. Peng, Coumans, Zhang, Lee, Tan, Levine. *Learning Agile Robotic Locomotion Skills by Imitating Animals.* RSS 2020; arXiv:2004.00784. https://arxiv.org/abs/2004.00784 — Butterworth on PD targets.
43. Raffin, Kober, Stulp. *Smooth Exploration for Robotic RL (gSDE).* CoRL 2021, PMLR 164. https://proceedings.mlr.press/v164/raffin22a.html — jerky per-step noise; continuity cost; deterministic eval.
44. Kobayashi, Yamanaka. *Redesigning Regularization for Effective Policy Smoothing (L2C2-v2).* arXiv:2606.13169, 2026 (preprint). https://arxiv.org/html/2606.13169
45. Escontrela, Peng, Yu, Zhang, Iscen, Goldberg, Abbeel. *Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions.* IROS 2022; arXiv:2203.15103. https://arxiv.org/abs/2203.15103 — vibration baseline; COT.
46. Liu, Wu, Li, Qi, Xue, Gao. *Learning Natural and Robust Hexapod Locomotion … via Motion Priors.* arXiv:2511.03167, 2025. https://arxiv.org/html/2511.03167 — TO dataset; Table I; tripod warning.
47. Rytz, Ly, Havoutis. *Sampling Strategies for Robust Universal Quadrupedal Locomotion Policies.* arXiv:2510.07094. https://arxiv.org/html/2510.07094 — Eq. 7–8.
48. Roditis, Nitsos, Porichis, Chatzakos, Bertos, Lika, Papadopoulos. *Maintaining static stability and continuous motion in rough terrain hexapod locomotion without terrain mapping.* MED 2016. https://nereus.mech.ntua.gr/pdf_ps/MED16-3.pdf — Sec. II McGhee–Frank criterion, SSM.
49. Yang, Yuan, Zhu, Yu, Li. *Multi-expert learning of adaptive legged locomotion (MELA).* Science Robotics 2020; arXiv:2012.05810. https://arxiv.org/abs/2012.05810 — Butterworth 5/3 Hz; smoothing loss ‖μ(s_t) − q‖².
50. Tanaka, Zhu, Wang, et al. *Mechanical Intelligence-Aware Curriculum RL for Humanoids with Parallel Actuation.* arXiv:2507.00273. https://arxiv.org/abs/2507.00273 — second-order Butterworth + dead band.
51. Isaac Lab. `envs/mdp/actions/joint_actions_to_limits.py` (EMAJointPositionToLimitsAction) and `manipulation/inhand/inhand_env_cfg.py`, main. https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions_to_limits.py — α default 1.0, in-hand 0.95; filter state unobserved.
52. Peng, van de Panne. *Learning Locomotion Skills Using DeepRL: Does the Choice of Action Space Matter?* SCA 2017; arXiv:1611.01055. https://arxiv.org/abs/1611.01055
53. Aljalbout, Frank, Karl, van der Smagt. *On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer.* IEEE RA-L 2024; arXiv:2312.03673. https://arxiv.org/abs/2312.03673 — c·dt bound; clipping mass at borders; violation ranking.
54. Fujita, Maeda. *Clipped Action Policy Gradient.* ICML 2018; arXiv:1802.07564. https://arxiv.org/abs/1802.07564
55. Chou, Maturana, Scherer. *Improving Stochastic Policy Gradients … using the Beta Distribution.* ICML 2017, PMLR 70. https://proceedings.mlr.press/v70/chou17a.html
56. Robotic Systems Lab. rsl_rl `modules/actor_critic.py`, v2.3.0. https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v2.3.0/rsl_rl/modules/actor_critic.py — init_noise_std 1.0, no clamp; `act_inference` returns the mean.
57. Isaac Lab. ANYmal-D `agents/rsl_rl_ppo_cfg.py` and `isaaclab_rl/rsl_rl/rl_cfg.py`, main. https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py — init_std 1.0, entropy 0.005; noise_std_type 'scalar'.
58. Seyde et al. *Is Bang-Bang Control All You Need?* NeurIPS 2021; arXiv:2111.02552. https://arxiv.org/abs/2111.02552
59. Andrychowicz et al. *What Matters In On-Policy Reinforcement Learning?* ICLR 2021; arXiv:2006.05990. https://arxiv.org/abs/2006.05990 — Sec. 3.2 initial std 0.5; regularisers; tanh vs clip.
60. Sabatini, Li, Hutter. *Bridging the Gap: Enabling Soft Actor Critic for High Performance Legged Locomotion.* arXiv:2605.24975, 2026. https://arxiv.org/abs/2605.24975 — σ0 0.15 vs 1.0; clamped log-std.
61. Raffin. *Getting SAC to Work on a Massive Parallel Simulator (Part I).* Blog 2025. https://araffin.github.io/post/sac-massive-sim/ — "less than 5 % of the action space".
62. Hollenstein, Martius, Piater. *Colored Noise in PPO.* AAAI 2024; arXiv:2312.11091. https://arxiv.org/abs/2312.11091 — β 0.5 best; more correlation with more envs.
63. Chiappa, Marin Vargas, Huang, Mathis. *Latent Exploration for RL (Lattice).* NeurIPS 2023; arXiv:2305.20065. https://arxiv.org/abs/2305.20065
64. Yi et al. *Flow Policy Gradients for Robot Control (FPO++).* arXiv:2602.02481, 2026. https://arxiv.org/abs/2602.02481 — zero-sampling eval; stochastic/deterministic gap; pronk observation.
65. van Marum, Shrestha, Duan, Dugar, Dao, Fern. *Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking.* IROS 2024; arXiv:2404.19173. https://arxiv.org/html/2404.19173 — stand c = 0; impulse protocol; constant contact reward.
66. Panchal, Kleyko, Artan, Loutfi. *Smoothness as a Constraint for Stable Humanoid Locomotion.* arXiv:2609.24552, 2026. https://arxiv.org/html/2609.24552 — Sec. IV-C metrics.
67. Yu, Yang, McGreavy, Triantafyllidis, Bellegarda, Shafiee, Ijspeert, Li. *Identifying Important Sensory Feedback for Learning Locomotion Skills.* Nature Machine Intelligence (arXiv:2306.17101). https://arxiv.org/pdf/2306.17101 — Eq. 11–12 recovery speed, foot placement.
68. Crowley, Dao, Duan, Green, Hurst, Fern. *Optimizing Bipedal Locomotion for The 100m Dash.* arXiv:2508.03070. https://arxiv.org/pdf/2508.03070 — Sec. C "Taking the Last Step".
69. Boston Dynamics. Spot SDK `bosdyn.client.robot_command` docs. https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_command.html — footprint_R_body; stance_command reach limit; yaw 0.4 example.
70. Jiang, Lin, Li, Shuai. *Bio-Inspired CPG Modulation via Proprioceptive DRL for Adaptive Hexapod Locomotion.* Biomimetics 11(8):570, 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13509619/ — metrics; no stand evaluation.

## Appendix: unverified or dropped items

- **Everitt, Hutter, Kumar, Krakovna, *Reward Tampering…*, Synthese 2021 (arXiv:1908.04734).** Exists;
  DROPPED as support for calling a state-keyed stand gate "RF-input tampering" — their Sec. 4 reserves
  that term for corrupting the observation-to-state relationship and calls visiting over-rewarded
  states gaming. §3 uses "gaming" and cites the gates only.
- **Extreme Parkour `legged_robot.py` and CAI23sbP `Isaaclab_Parkour` `uniform_parkour_command.py`.**
  Both opened; values correct (gain 0.8, ±1, `ang_vel_clip` 0.4). The sweep's history was wrong: the
  per-step relay is already in Extreme Parkour (L559 zeroes below the clip right after the P-control);
  the port kept that order, and Krabby's `heading_rate_command` docstring confirms it. Unnumbered.
- **CaT (Chane-Sane et al., IROS 2024, arXiv:2403.18765).** Exists; the sweep called torque, joint
  velocity, acceleration and action rate hard constraints — they are soft (Table I); `c_still` is a
  flat-terrain style constraint with p_max 0.05 → 0.25 and needs ε_still. A termination-as-cost
  form of Rec. 4 remains an option, unnumbered.
- **OpenSHC as precedent for absorbing heading error by posing.** Never states it; §3 marks it as our
  inference.
- **Kim, Choi, Choi, Yoon, Choi, *DRL Control of a Hexapod Robot*, Actuators 15(1):33, 2026.** Not
  opened (HTTP 403).
- **Inferences, not source statements:** "all-legs-down is the limit of the gait family" (Walknet
  reports tripod persisting at low speed under load); "a stopped CPG-RL clock plants a foot only in
  the stance half"; "shaping alone cannot remove stepping in place"; "step-in-place is too
  low-frequency for loss-level smoothing"; Spot's 0.5 m/s as a push-recovery choice (no reason in
  source; Lee et al. say so for 0.3 m/s).
- **Figures re-attributed:** Christmann's 26.8 % / 2.8 % are simulated Gymnasium averages, not
  real-robot; Solo12 ramps k_c linearly, not k^0.997; Walk These Ways joint-velocity −2e-5 paper vs
  −1e-4 code; AMP hand-designed COT 1.37–1.54 (Table II) vs 1.37–1.65 (text); Drosophila third author
  D. Sthanu Kumar; Phoenix return-home also confirmed in KurtE/Phantom_Phoenix `_Phoenix_Code.h`.
- **Repo probe, re-read 2026-09-22 (main session):** `model_7996.pt` (X08:c6) `std` = 2.0001–2.0004 on all 18 dims,
  just above the `.clamp(0.05, 2.0)` in `crab_actor_critic_with_encoder.py:27` (zero gradient past the bound). The
  earlier std trajectory (1.5 → 1.846 at it 1000 → 2.0 at it 2872 of X01:1c; 2.0 in the forward, sideways and 3a heads)
  is the sweep's TensorBoard read and was not re-checked.
- **Not in the ladder record:** stand facing rms 0.082 / max 0.175 rad come from the X08:c6 suite
  stand-hold readout supplied with the brief; `REPORT.md` prints only survival and drift for stand.
- **Unrefereed sources used with that caveat:** [21, 22, 24, 40, 44, 46, 47, 50, 60, 64, 66, 68]
  (arXiv), [61] (blog), [70] (MDPI 2026).

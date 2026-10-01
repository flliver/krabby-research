# Literature review: PPO, its hyper-parameters and the action/step-size construction -- mapped to the Krabby campaigns

*2026-09-30. Prepared after the PPO evidence dossier
(`/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/dossier/DOSSIER.md`,
compiled 2026-09-30 from the 437-run / 1.09M-iteration / ~528 GPU-h census, sixteen verified claims C01–C16)
and while the omni side bisect (`crab_hex_omni_task/experiments/2026-09-28_side_bisect`, X20 still training)
is live. Builds on `lit-review-gait-motion-quality.md` §6 (clipping bias, Beta/CAPG, init std 0.5, gSDE and
coloured noise at survey level) and `crab_hex_forward_task/experiments/lit-review-plasticity-cliff.md`
(weight-norm growth, critic reset falsified on the v2 base), which it does not repeat. Every citation was
opened by the review's verification pass and checked against the paper's abstract or full text, or the raw
source file, at time of writing; where verification corrected a sweep claim the corrected wording is used,
and items that failed verification or are only inferred are in the appendix rather than softened.*

## 1. The problems this review targets

Restated from the dossier so this document stands alone (section numbers are the dossier's):

1. **The executed controller is bang-bang.** Recorded pre-clip deterministic means on the omni head of
   record W: 97.0 % of components have |μ| > 1 (median 11.2, p90 22.0); all 303 evaluated checkpoints at
   999+ iterations are 91–100 % saturated; 71 % by iteration 199 (§3.3, D1). Signature: 2.23 sign flips per
   joint-second at stand, stand stepping, the rigid 3-feet pattern, and an `action_rate` term whose expected
   cost is 0.95 at |μ| = 0 and 0.00 at |μ| = 9, i.e. it pays for saturation (§4.3). The only σ experiment,
   M1 (cap 0.5, fine-tune of saturated X08:c6, no bound on μ), was REJECTED with jitter z −2.7 to −6.4; it
   does not test lower σ from scratch or with a bounded mean (§4.3, C10).
2. **σ is pinned in a one-way trap.** `std.clamp(0.05, 2.0)` in `crab_actor_critic_with_encoder.py:50`
   passes zero gradient above 2.0 from every loss term; σ climbs from `init_noise_std` 1.5 to the cap in 59/59
   fresh runs of ≥ 3500 iterations (median first pin 2444) and none of 303 runs that touched 1.99 ever left
   (§3.2, D2). `entropy_coef` 0.01 in 436/437 runs; no entropy-0 control exists.
3. **The step-size controller is the real learning rate.** The rsl_rl per-minibatch adaptive-KL rule (÷1.5
   if KL > 0.02, ×1.5 if KL < 0.005, floor 1e-5) runs 80 times per iteration on 384-sample minibatches. The
   configured 3e-4 is erased within iteration 0 in 240/254 fresh runs; the logged LR sits at the floor in
   62.9 % of census iterations; late heads live at 1e-5 to 3.4e-5. On W only 11–16 % of the Gaussian KL the
   controller reacts to changes any executed action, against 70–79 % during formation (X19 at 300, σ 1.67)
   (§3.1, D3, C04). KL, clip fraction and explained variance are not logged (§3.7).
4. **Batch geometry was never rescaled.** 256 envs × 24 steps = 6,144 samples per iteration, 5 epochs × 16
   minibatches of 384, inherited from a 6,144-env Go2 config (147,456 per iteration). Per-iteration wall time
   is latency-bound: 4× the envs costs +5 %; a run uses 61–69 % of the GPU at 66 W and 4.4 GB; no run above
   256 envs exists (§1, §3.5, D4, C09). γ 0.99 / λ 0.95 give a 2.0 s horizon and a 0.34 s GAE horizon inside a
   0.48 s rollout on a 1.2–2.8 s gait (§1).
5. **Seed variance is the size of the effects.** Turning blocks are seed-dominated (R_seed/R_eval 3.19/0.97
   and 4.12/0.94); decided effects cleared the seed term by only 1.05–4.3×; tripod formed in 1/4 plain-clock
   seeds and 0/7 seeds alternated in the basin search; 2 of 4 same-seed pairs diverge from iteration 20 via a
   nondeterministic history-latent kernel (§4.1, D18).
6. **Degenerate basins.** The forward unison glide out-earns the alternating target under the same reward
   (28.69 vs 26.15 at 3k); every other degenerate family earns less, so its persistence is basin selection,
   which no exploration or optimizer change was ever asked to move (§4.4(a)).
7. **Two-direction side walking.** Every network trained on both lateral hands is SLOW or DEAD
   (0.028–0.103 m/s); +y-only heads are FAST (0.184–0.217 m/s) and the exact M_y canonicaliser makes them fast
   both ways with no retraining; a fresh network distils the two-hand policy to 98.9–99.0 % sign agreement, so
   capacity is ruled out; PPO settings were byte-identical across all 20 arms, so "not a hyper-parameter
   problem" is untested (§4.4(c), C12).
8. **Value side and stage boundaries.** Clipped value loss, one Adam group for actor + critic + std at the
   KL-set LR, no observation or return normalisation, and a per-step reward clipped at ≥ 0 (13.6 % of return
   forgiven on W; teacher stages log value loss ≤ 0.0003). A flat critic priced ~0.5/step against
   ~0.0015/step on the bridge (failure 0.00 → 0.90); a critic reset fixed that hand-off and hurt B1
   (0.601 → 0.450) (§3.4, §4.6, D5, D6b).
9. **Symmetry.** Mirror LOSS only, coefficient 0.5 on raw unclipped means with a detached target; augmentation
   never enabled. Large seed-1 gains (tripod 0.573 vs 0.032 at 3k) that did not replicate (E1 seeds
   0.541 / 0.0 / 0.026 / 0.0; both 20k velocity arms at 0.0) (§4.5).
10. **Constraints.** Only rsl-rl-lib 5.0.1 installed; one 16 GB RTX 5080 usually shared; 1,149-dim obs
    (4.6 KB/transition); every checkpoint consumer needs the estimator and history-encoder modules; the ±1
    clip is load-bearing at deploy (teacher fell 100/100 without it); the student regresses raw teacher means
    (§6, D15).

**The short answer.** The literature does not indict PPO as a family; it indicts the construction Krabby
runs it in. No reference legged stack pairs a binding ±1 clip with an unbounded mean and a clamped σ ≥ the
bound (§2); the resulting saturation is a documented clipped-Gaussian pathology in which lowering σ alone
cannot restore an interior mean (§2–§3), the per-minibatch KL rule is being run 64–96× below the minibatch it
was tuned in (§6), and γ/λ and the batch were inherited, not chosen (§7). The cheapest fixes are inside PPO
and cost tens of GPU-hours: a mean bound or bounded head co-varied with a gradient-passing σ, a fixed or
per-epoch step-size rule, more envs per process, and value-side hygiene. Where the literature disagrees with
the dossier's reading is on what those fixes buy: action geometry and smoothness change reliably, task return
and learning speed only inconsistently, and near-bang-bang control may be close to what this objective asks
for (§4). Off-policy and model-based families are slower or more seed-variable in the closest analogues and
should wait (§10); the two-direction side failure has no direct precedent, and its best-supported first test
is canonical training (§8); no PPO-setting verdict should rest on fewer than 3 seeds (§11).

## 2. The clipped-Gaussian pathology and what the reference stacks do (question a, part 1)

**Key papers:** Chou et al. 2017 [1]; Hsu et al. 2020 [2]; He et al. 2026 [3]; Geiß et al. 2024 [4];
legged_gym [5]; extreme-parkour [6]; Isaac Lab velocity cfgs and PR #2019 [7]; PR #984 [8]; Raffin 2025 [9];
IsaacGymEnvs configs [10]; Isaac Lab Ant rl_games cfg [11]; rl_games [12]; VideoMimic [13].

Env-side clipping replicates the reward at the bound: the "replicated" padding "results in artificially
higher rewards outside the bounds and therefore bias the estimated policy distribution toward the boundary",
and "as σ grows, bias also increases" [1] (TRPO/ACER experiments; the argument is generic). By analogy with
Hsu et al.'s toy region with no reward signal, once the executed action is flat in μ "the mean and the
standard deviation of the Gaussian policy fluctuate randomly due to noise" [2]. Entropy on the latent
(unclipped) Gaussian has zero gradient on the mean and a constant upward push on σ [3], so `entropy_coef` 0.01
pushes σ toward the cap and the clamp then zeroes every gradient above it. The most direct measurement is He
et al. (Aug 2026 preprint, SB3 PPO, 12 envs, 3 seeds, 80-muscle MyoLeg): a clipped Gaussian executed 89.07 %
of actions within 5 % of a bound, 82.12 % of state-conditioned means lay outside the interval, and with zero
variance 83.83 % were still near a bound, so the saturation sits in the mean. In a matched three-seed
comparison of tanh policies (latent interface widened to ±10), near-bound occupancy was 71.42 % with latent
entropy, 29.76 % with no entropy and 18.83 % with entropy on the executed action, whose Jacobian "pulls the
mean inward"; direct mean penalties centred the mean as well or better; and "if P10 is already large, reducing
variance alone cannot produce an interior mean" [3]. An independent on-policy study (GAIL with TRPO, 92-MTU
humanoid) saw the same drift, "the predicted mean arbitrarily far out-of-bounds ... while not reducing or even
increasing its standard deviation", no gain from target entropy, and gains from mean-regularising objectives
(about +200 cumulative reward on running for the flipped-uniform KL); bounded squashed-Gaussian and Beta heads
there "did not pay dividends" [4].

No reference legged PPO stack operates where Krabby does. legged_gym clips at ±100 in raw policy units with
action_scale 0.5, init std 1.0, entropy 0.01 and 4 minibatches of 24,576 [5]. Extreme-parkour (Krabby's parent)
clips at 1.2 with action_scale 0.25 for its default `a1` task, so the effective clip is ±4.8 policy units, with
an unclamped `nn.Parameter` std of 1.0, entropy 0.01, an optional `tanh_encoder_output` on the mean (False by
default) and 4 minibatches of 36,864 [6]. Isaac Lab velocity tasks use JointPosition scale 0.5 (0.25 for A1),
`clip_actions` None, init std 1.0, entropy 0.005 (ANYmal C/D) to 0.01 (A1/Go2/H1/Cassie/Digit), 0.008 (G1),
0.0025 (Spot), 4 minibatches of 24,576; the wrapper's optional clip (PR #2019, after the unmerged #984) was
added to break a last_action → critic-error → larger-action feedback loop, not to bound the policy [7, 8]. A
trained Isaac Lab A1 PPO policy "uses less than 5 % of the action space" of its ±100 box, with under 0.1 % of
actions outside [−5, 5] [9]. The one family that does clip at ±1 (rl_games in IsaacGymEnvs and Isaac Lab)
makes the clip binding on samples (std init 1, about a third truncated) but pairs it with a quadratic bound
loss that keeps the mean inside ±1.1, an unclamped log-std and entropy 0: AnymalPPO `bounds_loss_coef` 0.001,
`kl_threshold` 0.008, minibatch 32,768; Ant/Humanoid/ShadowHand/Franka 1e-4; AnymalTerrainPPO turns both the
clip and the bound loss off (entropy 0.001) [10, 11, 12].

**What it means here.** Krabby's velocity-target cams make ±1 the max-speed command, so its clip is binding
by construction, with σ 1.5–2.0 against it and nothing bounding the mean; every setting that makes the
reference stacks safe (a non-binding clip, a bound loss, an unclamped std, low entropy, 16k–37k minibatches)
is absent. The 97 % |μ| > 1 share means P10 is at least ~97 % at any small margin, so by [3] lowering σ alone
could not have produced an interior mean: M1's rejection is the predicted outcome, not evidence against lower
σ. A cheap audit before any head change: check whether the last-action and history observations carry the
raw (|μ| ~ 10–30) or the clipped action, the loop PR #2019 was written to break [7].

## 3. Remedies: mean bound, tanh, Beta, categorical (question a, part 2)

**Key papers:** rl_games bound loss [12]; IsaacGymEnvs [10]; VideoMimic [13]; He et al. [3]; Geiß et al. [4];
Andrychowicz et al. 2021 [14]; SAC [15]; Brax [16]; MuJoCo Playground [17]; rsl_rl BetaDistribution / PR #202
[18]; Chou [1]; Hsu [2]; DHAL [19]; CAPG [20]; 37 details [21]; Tang & Agrawal 2020 [22]; Zhu et al. 2024 [23];
RN-D 2026 [24]; GPO 2026 [25].

*Mean bound.* rl_games defines `bound_loss(μ) = Σ_dims clamp_min(μ − 1.1, 0)² + clamp_max(μ + 1.1, 0)²`, added
as `+ b_loss·bounds_loss_coef` next to an env clamp to [−1, 1] (`clip_actions` default True) [12]. VideoMimic
(G1, Isaac Gym PPO) keeps a linear actor output, clips actions "to prevent the policy from learning bang-bang
style control earlier in training", adds an L1 bounds loss (1/D)Σ|clip(μ_j) − μ_j| at 0.0005 with an action
limit of 8.0 and init std 0.8, and reports no ablation [13]. He et al.'s pre-tanh L2 and tanh² penalties centre
the mean as well as executed-action entropy (Dog-Stand P10 1.60 % / 0.09 % vs 1.91 %; E|μ| 0.475 vs 0.489)
but returned clearly less (about 465–484 vs 594), and the authors warn that matched mean geometry need not
mean matched variance or return [3]. Calibration (reviewer arithmetic): at W's median |μ| 11 over 18 dims the
rl_games hinge is (11 − 1.1)² × 18 ≈ 1,760, i.e. ≈ 0.18 of loss at coefficient 1e-4, large next to a median
|surrogate| of 0.008, which argues for from-scratch or ramped-coefficient runs rather than a fine-tune of a
saturated head. No legged-robot PPO source reports an ablation effect size for the bound loss; the only
rough-terrain rl_games config disables it [10].

*tanh with the log-determinant.* The log|tanh'| term "does not affect policy losses" and "does not affect the
KL divergences" because the two terms cancel on the same sample; it matters only in the entropy, where it
"penalizies the policy for taking extreme actions which prevents tanh saturation" and bounds the entropy [14];
SAC gives the change-of-variables form [15]. He et al. confirm that "replacing clipping by a smooth tanh map
does not remove the high-variance regime" under latent entropy, while executed-action entropy pulls the mean
inward, and that "more interior action geometry is therefore not the same as higher task return" [3]. tanh was
slightly better than clipping overall in Andrychowicz et al. (+30 % on HalfCheetah), attributed to "the
decreased magnitude of initial actions" [14]. Brax PPO defaults to `tanh_normal` with softplus std + 0.001 and
a single-sample tanh log-det in the entropy, and MuJoCo Playground's Go1 joystick configs keep that default at
8,192 envs, `entropy_cost` 1e-2, unroll 20, 32 minibatches, discount 0.97 and a 512-256-128 policy [16, 17].
Tang & Agrawal's "Gaussian+tanh" (tanh on the mean only) scored the same as the plain Gaussian on Humanoid-v1
(4007 ± 698 vs 3905 ± 502) and better on rllab Humanoid (5863 vs 2522) [22].

*Beta.* Upstream rsl-rl-lib 5.3.0 (PR #202, merged 2026-05-07) adds `BetaDistribution` (α, β = softplus + 1,
unimodal; samples rescaled to `action_range` (−1, 1); deterministic output the rescaled mean; Beta–Beta KL on
the bounded action); it is absent in the installed 5.0.1, ships with a request for "feedback on the
performance", and the only user result on the PR is a single-seed spacecraft-thruster task (final position
error 0.0040 vs 0.0178 m) [18]. Beta roughly halves TRPO's Humanoid-v1 time-to-return [1] and doubles PPO's
Humanoid-v2 return, with a uniform init that escapes the local optimum a N(0, 1) init falls into [2]. Legged
task-level evidence: in DHAL (Go1 skateboarding, 4 training seeds, 5 hardware trials per scenario) the
Beta-only ablation failed every real-world scenario the full method passed 60–100 %, and Gaussian policies
could not learn the task [19]; but under PPO defaults Beta was the weakest head on Humanoid in two MuJoCo
studies (2680 ± 2493 and 791 ± 36, the former needing LR 3e-5) [22, 23], and Geiß found no dividend [4].

*Categorical and ordinal heads.* Factorised categorical heads with 7–15 evenly spaced bins per dimension
give large gains only at Humanoid scale: PPO, 10M steps, 5 seeds, Humanoid-v1 Gaussian 3905 ± 502 vs discrete
5119 ± 151 vs ordinal 6018 ± 403; K = 2 collapses, K ~ 50 degrades slightly; wall time 116–120 % of Gaussian at
K = 5–11 [22]. Unimodal (Poisson-ordinal) heads at K = 9–15 reach 7072 ± 409 on Humanoid-v3 vs Gaussian
4886 ± 226 and plain categorical 5443 ± 280 [23]. RN-D pairs a K = 41 categorical actor (entropy coefficient 0)
with a residual + LayerNorm backbone and reports 1.3–1.9× fewer interaction steps than a Gaussian MLP over 5
seeds on MuJoCo and ManiSkill at equal throughput (873.8 vs 854.4 SPS); its loss-swap ablation shows the gain
comes from the categorical likelihood, and "discretization alone can also be beneficial" [24]. None of the
three has legged-robot tasks.

*Growing bound and CAPG.* GPO (Isaac Gym; quadruped plus an 18-DoF hexapod; Go2 hardware) uses a
time-growing tanh bound ã = β_t·tanh(a/β_t) with the Jacobian included; hexapod velocity-tracking error
0.00 ± 0.05 vs PPO −0.10 ± 0.05 m/s, base height 0.02 vs −0.12 m, hardware push/terrain success 100 % vs
0–40 %; the seed count is not reported and the PPO baseline is described as hard clipping [25]. CAPG, the
unbiased clipped-action estimator, corrects only the log-prob of clipped samples, gave significant AUC gains
on 3/10 MuJoCo tasks for PPO (7/10 for TRPO) and, by inference, adds no restoring force on |μ| [20, 21].

**What it means here.** Ranked by evidence and cost: (1) an rl_games-style bound loss or L2 on raw μ,
loss-only, touching neither the log-prob, KL, estimator, distiller nor the deploy clip; (2) tanh on the mean
with entropy measured on the executed action or a pre-tanh L2 (tanh alone does not stop drift); (3) a Beta
head ported from rsl_rl 5.3.0 (bounded mean makes the deploy clip a no-op and the KL behavioural, §6);
(4) an ordinal/unimodal categorical head at K = 11 as the first non-Gaussian alternative, since it puts
log-prob, KL and entropy exactly on the executed distribution and its gains grow with action dimension;
(5) GPO's growing bound; (6) CAPG last. Expect bounded heads to change action geometry reliably and task
return inconsistently [3, 4, 22], so pre-register task and smoothness gates separately.

## 4. Bang-bang, the plant and the action interface (question a, part 3)

**Key papers:** Seyde et al. 2021 [26]; Liberzon [27]; Peng & van de Panne 2017 [28]; Aljalbout et al. 2024
[29]; Bogdanovic et al. 2020 [30]; CPG-RL [31]; Lee et al. 2024 [32]; in-repo velocity-action record [33];
LCP [34]; Guldner & Utkin [35]; Caponigro et al. [36]; Wang et al. 2024 [37]; `parkour_actions.py` [38].

For deterministic control-affine dynamics with |a| ≤ 1, Pontryagin's principle gives bang-bang control with
no action cost except on singular arcs, bang-off-bang with an L1 cost, and generally non-bang-bang optima
with a quadratic cost [26]; time-optimal steering of normal linear systems with a hypercube control set takes
only vertex values [27]. These results do not imply that Krabby's velocity-tracking objective has a
bang-bang optimum (steady tracking can be a singular arc; Seyde's Cartpole policies use near-zero actions
during stabilisation), but they make near-bang-bang an expected outcome rather than an automatic defect.
Empirically, Bernoulli bang-bang heads perform on par with Gaussians under PPO, SAC, MPO and DreamerV2 on
DMC (4 seeds); converged MPO Gaussians "consistently resort to extremal actions" on Finger, Walker and
Quadruped; quadratic and difference penalties give smoother, lower-magnitude means with robustness "only
slightly increased" and lower nominal return on Quadruped, and the intro notes the "negative impact of action
penalties on exploration with Gaussian policies"; the authors suggest dynamics acting "as a low-pass filter on
the bang-bang input" explains why Bernoulli policies are as robust under sensor disturbances [26]. LCP's
humanoid ablation quantifies the smoothing trade: no smoothing gave the highest return (28.87) at jitter
42.19; a smoothness reward 26.56 / 5.74; LCP 26.03 / 3.21; a low-pass action filter the worst return
(24.98) and most variable jitter (7.86 ± 3.00), attributed to damped exploration [34].

The classical control literature treats chattering as the structural result of discontinuous control on
bounded-input integrator chains, not of the interface: fast switching excites neglected actuator dynamics,
the standard remedy is a boundary layer that replaces sign(s) by a saturation function [35]; the Fuller
phenomenon (infinitely many switches in finite time) is regularised by a total-variation penalty on the
control whose quasi-optimal solutions converge to the optimum [36]; and chattering is proven for
time-optimal chain-of-integrator problems with state constraints [37]. Krabby's cam channel is such a chain
(velocity command → accel-limited velocity at 2π rad/s² → shaft angle → hip angle) and its rods are position
targets through a stroke clamp and a 0.028/0.033 m/s slew every substep [38]; the cam acceleration limit
(0.126 rad/s per 20 ms step, ~4 % of the range) is exactly the low-pass stage that lets sign switching encode
any average velocity as a PWM code.

Does the interface itself matter? The one head-to-head legged study found PD position targets learned faster
than joint-velocity targets on all 7 motions (AUC 0.70 vs 0.63 biped walk, 0.93 vs 0.81 raptor run,
0.87 vs 0.75 dog rear-up), with final NCR tied on easy motions and favouring PD on hard ones; its velocity
interface is a damping servo, and the tasks are planar imitation [28]. In a 13-action-space manipulation study,
velocity-type spaces gave the best sample efficiency, transfer and fewest jerk violations, multi-step
integrators that integrate on the previous target "consistently perform the worst", and clipping integrated
targets puts "more probability mass on the borders" [29]; a fixed-gain position interface still produced
high-frequency force oscillations, with smoothness coming from a reward term [30]. CPG-RL's oscillator
frequency/amplitude action gave similar returns to joint-PD offsets but joint-PD "result in unnatural gaits
that overfit the simulator dynamics" with "high frequency small-step gaits" [31]; a wheeled-legged Science
Robotics controller sends position commands to leg joints and velocity commands to its continuously rotating
wheels in one action vector, with a Beta head for its bounded high-level action [32]. In-repo, the 2026-08-13
switch to velocity cams was hardware-motivated (a continuously rotating motor); the from-scratch velocity arm
"oscillates the shafts at ~1.4 Hz full speed instead of continuous rotation", and a cam lower clip of 0.1 was
tried only in 1k-iteration single-seed screens [33].

**What it means here.** Krabby's rod channels already are a position interface with a modelled rate limit,
and they saturate as badly as the cams (clipped rod-mean share 0.59–0.73 at iteration 300, 0.94–0.96 at 2999;
once saturated the slew reduces to rod_len += v_max·dt·sign(a), the same sign-times-max-rate integrator as a
saturated velocity command) [38, 143 §4.4(c)]. So the interface is not the lever; the distribution and clip
are (§2–§3), and the velocity cam is the correct analogue of a CPG frequency command for a continuously
rotating shaft [31, 32]. What a bounded head restores is a meaningful KL, entropy and σ gradient, not smooth
actions per se; a tanh head is likely to converge near ±1 anyway [26]. Smoothness and wear must therefore be
explicit, scored gates (sign flips per joint-second, energy, FFT mean frequency as in
`lit-review-gait-motion-quality.md` §8) bought with an explicit cost on the *executed* action: a
total-variation / L1 cost on the executed difference [36], not the present squared `action_rate` on clipped
actions, which is near-constant under saturation; a low-pass filter is the least attractive smoother [34].
For the non-backdrivable lead-screw rods and geared cams the chattering literature's harm (wear, excited
unmodelled dynamics) is a sim-to-real argument for removing bang-bang independent of any task-metric gain
[35].

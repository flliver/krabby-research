# Literature review: PPO, its hyper-parameters and the action/step-size construction -- mapped to the Krabby campaigns

*2026-09-30. Prepared after the PPO evidence dossier
(`crab_hex_omni_task/experiments/2026-09-30_rl_algorithm_review/DOSSIER.md`,
compiled 2026-09-30 from the 437-run / 1.09M-iteration / ~528 GPU-h census, claims C01–C16 verified) and
while the omni side bisect (`crab_hex_omni_task/experiments/2026-09-28_side_bisect`, X20 still training) is
live. Builds on `lit-review-gait-motion-quality.md` §6 (clipping bias, Beta/CAPG, init std 0.5, gSDE and
coloured noise at survey level) and `crab_hex_forward_task/experiments/lit-review-plasticity-cliff.md`
(weight-norm growth; critic reset falsified on the v2 base), which it does not repeat. Every citation was
opened by the review's verification pass and checked against the paper's abstract or full text, or the raw
source file, at time of writing; where verification corrected a sweep claim the corrected wording is used, and
items that failed verification or are only inferred are in the appendix rather than softened.*

## 1. The problems this review targets

Restated from the dossier [143] (its section numbers) so this document stands alone:

1. **Bang-bang control.** On the omni head of record W, 97.0 % of pre-clip mean components have |μ| > 1
   (median 11.2); all 303 evaluated checkpoints at 999+ iterations are 91–100 % saturated (§3.3): 2.23 sign
   flips per joint-second at stand, stand stepping, an `action_rate` cost of 0.95 at |μ| = 0 vs 0.00 at |μ| = 9.
   The only σ experiment, M1 (cap 0.5 on a fine-tune of saturated X08:c6, no bound on μ), was REJECTED with
   jitter z −2.7 to −6.4 (§4.3).
2. **σ pinned.** `std.clamp(0.05, 2.0)` passes zero gradient above 2.0; σ climbs from `init_noise_std` 1.5 to
   the cap in 59/59 fresh runs of ≥ 3500 iterations and never leaves (§3.2); `entropy_coef` 0.01, no entropy-0
   control.
3. **The KL controller is the real learning rate.** The per-minibatch rule (÷1.5 if KL > 0.02, ×1.5 if
   KL < 0.005, floor 1e-5) runs 80× per iteration on 384-sample minibatches; the configured 3e-4 is erased
   within iteration 0 in 240/254 fresh runs; the LR is at the floor in 62.9 % of iterations; on W only 11–16 %
   of the KL it reacts to changes any executed action, vs 70–79 % during formation (§3.1). KL, clip fraction
   and explained variance are not logged.
4. **Inherited geometry.** 256 envs × 24 steps = 6,144 samples in 16 minibatches of 384, from a 6,144-env Go2
   config (147,456). Wall time is latency-bound (4× envs = +5 %; GPU 61–69 % at 66 W, 4.4 GB); no run above
   256 envs (§3.5). γ 0.99 / λ 0.95: a 0.34 s GAE horizon inside a 0.48 s rollout on a 1.2–2.8 s gait.
5. **Seed variance ≈ effect size.** Turning blocks R_seed/R_eval 3.19/0.97 and 4.12/0.94; tripod formed in 1/4
   plain-clock seeds, 0/7 alternated in the basin search; 2 of 4 same-seed pairs diverge (§4.1, D18).
6. **Basins.** The forward unison glide out-earns the target under the same reward (28.69 vs 26.15 at 3k);
   every other degenerate family earns less (§4.4(a)).
7. **Two-direction side walking.** Two-hand networks are SLOW or DEAD (0.028–0.103 m/s); +y-only heads are
   FAST (0.184–0.217 m/s) and the exact M_y canonicaliser makes them fast both ways; a fresh net distils the
   two-hand policy to 99 % sign agreement; PPO settings were byte-identical across arms (§4.4(c)).
8. **Value side.** Clipped value loss; one Adam group for actor + critic + std; no obs or return normalisation;
   reward clipped at ≥ 0 (13.6 % of return forgiven on W); a flat critic priced ~0.5/step against ~0.0015/step
   on the bridge (§4.6).
9. **Symmetry.** Mirror LOSS only, 0.5 on raw means; augmentation never enabled; seed-1 gains (0.573 vs 0.032
   at 3k) that did not replicate (E1 seeds 0.541/0.0/0.026/0.0) (§4.5).
10. **Constraints.** rsl-rl-lib 5.0.1 only; one shared 16 GB GPU; 1,149-dim obs; every consumer needs the
    estimator/history-encoder modules; the ±1 clip is load-bearing; the student regresses raw means (§6).

**The short answer.** The literature indicts the construction, not PPO. No reference legged stack pairs a
binding ±1 clip with an unbounded mean and a clamped σ ≥ the bound (§2); the saturation is a documented
clipped-Gaussian pathology in which lowering σ alone cannot restore an interior mean (§2–§3); the
per-minibatch KL rule runs 64–96× below the minibatch it was tuned in (§6); batch and γ/λ were inherited (§7).
The fixes are inside PPO and cost tens of GPU-hours. Where the literature disagrees with the dossier is on
what they buy: geometry and smoothness change reliably, task return only inconsistently, and near-bang-bang
may be close to what this objective asks for (§4). Off-policy and model-based families are slower or more
seed-variable in the closest analogues (§10); the two-direction failure has no precedent and its
best-supported test is canonical training (§8); no verdict should rest on fewer than 3 seeds (§11).

## 2. The clipped-Gaussian pathology and the reference stacks (question a, part 1)

**Key papers:** Chou et al. 2017 [1]; Hsu et al. 2020 [2]; He et al. 2026 [3]; Geiß et al. 2024 [4];
legged_gym [5]; extreme-parkour [6]; Isaac Lab cfgs, PR #2019 [7]; PR #984 [8]; Raffin [9]; IsaacGymEnvs [10];
Isaac Lab Ant rl_games cfg [11]; rl_games [12]; VideoMimic [13].

Env-side clipping replicates the reward at the bound and biases the policy toward it, more so "as σ grows"
[1]; once the executed action is flat in μ, mean and std "fluctuate randomly due to noise" (by analogy with
Hsu's no-signal region) [2]. Entropy on the latent Gaussian has zero gradient on the mean and a constant upward
push on σ [3], so `entropy_coef` 0.01 drives σ to the cap, where the clamp zeroes every gradient. Measured (SB3
PPO, 3 seeds, 80-muscle MyoLeg): a clipped Gaussian executed 89.07 % of actions within 5 % of a bound, 82.12 %
of state-conditioned means lay outside the interval, and with zero variance 83.83 % were still near a bound;
among matched tanh policies, near-bound occupancy was 71.42 % with latent entropy, 29.76 % with none and
18.83 % with entropy on the executed action, whose Jacobian "pulls the mean inward"; "if P10 is already large,
reducing variance alone cannot produce an interior mean" [3]. A GAIL/TRPO musculoskeletal study saw the same
drift with std "not reducing or even increasing", no gain from target entropy, gains from mean-regularising
objectives (~+200 for a flipped-uniform KL on running), and bounded heads that "did not pay dividends" [4].

No reference stack operates here. legged_gym: clip ±100, action_scale 0.5, init std 1.0, entropy 0.01,
4 × 24,576 [5]. Extreme-parkour (Krabby's parent): clip 1.2 with action_scale 0.25 for its default `a1` task
(±4.8 policy units), unclamped std 1.0, entropy 0.01, 4 × 36,864 [6]. Isaac Lab velocity tasks: `clip_actions`
None, init std 1.0, entropy 0.0025–0.01, 4 × 24,576; the wrapper clip (PR #2019, after the unmerged #984) exists
to break a last_action → critic-error → larger-action loop [7, 8]; a trained A1 policy "uses less than 5 % of
the action space" of its ±100 box [9]. rl_games clips at ±1 with std 1 (binding on samples) but bounds the
mean with a quadratic loss inside ±1.1 (AnymalPPO `bounds_loss_coef` 0.001, `kl_threshold` 0.008, minibatch
32,768, entropy 0; Ant 1e-4); AnymalTerrainPPO turns both off [10, 11, 12].

**What it means here.** Krabby's velocity-target cams make ±1 the max-speed command, so its clip binds by
construction with σ 1.5–2.0 against it and no mean bound; every safeguard the reference stacks use is absent.
With 97 % of means outside ±1, P10 ≥ 97 %, so by [3] M1's rejection is the predicted outcome, not evidence
against lower σ. Cheap audit first: do the last-action and history observations carry the raw (|μ| ~ 10–30) or
the clipped action, the loop PR #2019 was written to break [7]?

## 3. Remedies: mean bound, tanh, Beta, categorical (question a, part 2)

**Key papers:** rl_games [12]; IsaacGymEnvs [10]; VideoMimic [13]; He et al. [3]; Geiß [4]; Andrychowicz et
al. 2021 [14]; SAC [15]; Brax [16]; MuJoCo Playground [17]; rsl_rl BetaDistribution / PR #202 [18]; Chou [1];
Hsu [2]; DHAL [19]; CAPG [20]; 37 details [21]; Tang & Agrawal 2020 [22]; Zhu et al. 2024 [23]; RN-D 2026 [24];
GPO 2026 [25].

*Mean bound.* rl_games adds `Σ clamp_min(μ − 1.1, 0)² + clamp_max(μ + 1.1, 0)²` × `bounds_loss_coef` beside
an env clamp to [−1, 1] [12]. VideoMimic (G1, Isaac Gym PPO) clips "to prevent the policy from learning
bang-bang style control earlier in training" and adds an L1 bounds loss (1/D)Σ|clip(μ_j) − μ_j| at 0.0005
(limit 8.0, init std 0.8), no ablation [13]. He et al.'s pre-tanh L2 and tanh² penalties centre the mean as
well as executed-action entropy (P10 1.60 % / 0.09 % vs 1.91 %) but return less (~465–484 vs 594) [3].
Calibration (arithmetic): at W's median |μ| 11 over 18 dims the hinge is (11 − 1.1)² × 18 ≈ 1,760, ≈ 0.18 of
loss at 1e-4 against a median |surrogate| 0.008, so run from scratch or ramp the coefficient. No legged PPO
ablation exists; the only rough-terrain rl_games config disables the loss [10].

*tanh with the log-determinant.* The log|tanh'| term affects neither policy losses nor KL; it acts only in
the entropy, where it "prevents tanh saturation" [14, 15]. "Replacing clipping by a smooth tanh map does not
remove the high-variance regime" under latent entropy, and "more interior action geometry is therefore not
the same as higher task return" [3]. tanh beat clipping slightly (+30 % HalfCheetah) via smaller initial
actions [14]; tanh on the mean alone tied the Gaussian on Humanoid-v1 (4007 ± 698 vs 3905 ± 502) [22]. Brax
PPO defaults to `tanh_normal` (softplus std, log-det in the entropy); MuJoCo Playground's Go1 joystick keeps it
at 8,192 envs, `entropy_cost` 1e-2 [16, 17].

*Beta.* rsl-rl-lib 5.3.0 (PR #202, 2026-05-07) adds `BetaDistribution` (softplus + 1; rescaled to (−1, 1);
Beta–Beta KL on the executed action), absent from 5.0.1, with only a single-seed thruster result on the PR
[18]. Beta halves TRPO's Humanoid-v1 time-to-return [1] and doubles PPO's Humanoid-v2 return [2]; in DHAL (Go1
skateboarding; 4 seeds; 5 hardware trials per scenario) the Beta-only ablation failed every scenario the full
method passed 60–100 % [19]; but under PPO defaults Beta was the weakest Humanoid head in two studies
(2680 ± 2493; 791 ± 36) [22, 23] and gave no dividend in [4].

*Categorical heads.* 7–15 bins gain only at Humanoid scale: PPO, 5 seeds, Humanoid-v1 Gaussian 3905 ± 502 vs
ordinal 6018 ± 403; wall time 116–120 % at K = 5–11 [22]; unimodal heads 7072 ± 409 vs 4886 ± 226 on
Humanoid-v3 [23]. RN-D (K = 41, entropy 0, residual + LayerNorm) needs 1.3–1.9× fewer steps than a Gaussian MLP
over 5 seeds at equal throughput, the gain traced to the categorical likelihood [24]. No legged tasks.

*Growing bound; CAPG.* GPO (quadruped and an 18-DoF hexapod, Go2 hardware) grows a tanh bound ã = β_t
tanh(a/β_t) with the Jacobian: hexapod tracking error 0.00 ± 0.05 vs PPO −0.10 ± 0.05 m/s, hardware success
100 % vs 0–40 %, seeds unreported [25]. CAPG corrects only clipped samples' log-prob (3/10 MuJoCo tasks for
PPO), no restoring force on |μ| (inference) [20, 21].

**What it means here.** Rank: (1) bound loss or L2 on raw μ, loss-only, touching neither log-prob, KL,
estimator, distiller nor deploy clip; (2) tanh mean with executed-action entropy or a pre-tanh L2; (3) a Beta
head ported from 5.3.0 (deploy clip a no-op, KL behavioural, §6); (4) an ordinal head at K = 11; (5) GPO's
growing bound; (6) CAPG. Expect geometry to move reliably and return inconsistently [3, 4, 22]: pre-register
task and smoothness gates separately.

## 4. Bang-bang, the plant and the action interface (question a, part 3)

**Key papers:** Seyde et al. 2021 [26]; Liberzon [27]; Peng & van de Panne 2017 [28]; Aljalbout et al. 2024
[29]; Bogdanovic et al. [30]; CPG-RL [31]; Lee et al. 2024 [32]; in-repo record [33]; LCP [34]; Guldner & Utkin
[35]; Caponigro et al. [36]; Wang et al. 2024 [37]; `parkour_actions.py` [38]; dossier [143].

For control-affine dynamics with |a| ≤ 1, Pontryagin gives bang-bang control with no action cost (except on
singular arcs), bang-off-bang with L1, non-bang-bang with a quadratic cost [26]; time-optimal control of normal
linear systems takes hypercube vertices [27]. Near-bang-bang is therefore expected, not a defect. Bernoulli
heads match Gaussians under PPO, SAC, MPO and DreamerV2 (4 seeds); converged MPO Gaussians "consistently
resort to extremal actions"; penalties give smoother means with robustness "only slightly increased" and lower
return on Quadruped; dynamics act "as a low-pass filter on the bang-bang input" [26]. LCP's humanoid ablation:
no smoothing 28.87 return / 42.19 jitter; smoothness reward 26.56 / 5.74; LCP 26.03 / 3.21; a low-pass filter
the worst return (24.98) and most variable jitter (7.86 ± 3.00) [34]. Classical control treats chattering as
structural to discontinuous control on bounded-input integrator chains, fixed by a boundary layer [35] or a
total-variation penalty [36], and proven for constrained chain-of-integrator problems [37]. Krabby's cam
channel is such a chain (velocity command → accel-limited velocity → shaft angle → hip), its rods position
targets through a stroke clamp and a 0.028/0.033 m/s slew [38]; the 0.126 rad/s per-step cam limit is the
low-pass stage that lets switching encode any average velocity.

The interface: PD targets learned faster than (damping-servo) velocity targets on 7 planar imitation motions
(AUC 0.70 vs 0.63 walk, 0.93 vs 0.81 raptor run) [28]; in manipulation, velocity-type spaces transferred best,
multi-step integrators "consistently perform the worst", and clipping integrated targets puts "more
probability mass on the borders" [29]; a fixed-gain position interface still oscillated [30]; joint-PD offsets
"result in unnatural gaits that overfit the simulator dynamics" where oscillator actions did not [31]; a
wheeled-legged controller mixes leg position and wheel velocity commands with a Beta head [32]; in-repo the
from-scratch velocity arm "oscillates the shafts at ~1.4 Hz full speed instead of continuous rotation" [33].

**What it means here.** Krabby's rods already are a rate-limited position interface and saturate like the
cams (clipped rod-mean share 0.59–0.73 at iteration 300, 0.94–0.96 at 2999; once saturated the slew reduces to
sign(a) × v_max·dt) [38, 143]: the interface is not the lever, the distribution and clip are, and the velocity
cam is the right analogue of a CPG frequency command [31, 32]. A bounded head restores a meaningful KL,
entropy and σ gradient, not smooth actions; a tanh head will likely converge near ±1 [26]. Smoothness must be
an explicit gate (sign flips per joint-second, energy, FFT mean frequency; `lit-review-gait-motion-quality.md`
§8) bought with a cost on the *executed* action, a total-variation/L1 difference [36] rather than the squared
`action_rate` on clipped actions, which is near-constant under saturation; a low-pass filter is the least
attractive smoother [34]. For non-backdrivable rods the chattering literature's harm (wear, excited
unmodelled dynamics) is a sim-to-real argument independent of task metrics [35].

## 5. σ parameterisation: init, clamp, entropy (question b)

**Key papers:** Andrychowicz [14]; Schulman et al. 2017 [39]; 37 details [21]; Sabatini, Li, Hutter 2026 [40];
Hsu [2]; rl_games MJLAB.md [41]; VideoMimic [13]; CleanRL SAC [42]; Brax [16]; rsl_rl PRs #66/#67/#112/#190/#201
[43]; PPO-CMA [44]; REPI [45].

Over ~250k MuJoCo agents, performance was "very sensitive to the initial action standard deviation with 0.5
performing best on all environments except Hopper"; a smaller last policy layer alone raised Humanoid return
66 %; "no evidence that any of the investigated regularizers helps significantly" except on HalfCheetah [14].
The PPO paper uses no entropy bonus on MuJoCo and anneals its Roboschool log-std −0.7 → −1.6 (std ≈ 0.5 → 0.2)
[39]; std-1 / `ent_coef` 0 is CleanRL's convention [21]. Legged references start at σ 1.0 (VideoMimic 0.8) with
entropy 0.0025–0.01; the ETH seven-robot PPO baseline uses σ0 1.0, entropy 0.005 [40, 13]. For SAC with a tanh
head, σ0 = 1 "places significant mass near the action bounds" while 0.15 "leads to faster and more stable
convergence" [40]. Counter-caveat: a Gaussian init concentrated near a suboptimal peak pulls the gradient
there, so a narrower init can worsen basin selection [2]. rl_games' maintainers report "a positive entropy
bonus on a global `fixed_sigma` can drive a sigma runaway over 1B+ frame runs" and use entropy 0 [41].

Hard σ clamps are common but loose: CleanRL SAC maps log-std through tanh into [−5, 2] [42]; Brax uses
softplus + 0.001 [16]; Andrychowicz recommend softplus with a negative offset, find the min std to "matter
little, if it is not set too large" and per-state std "no clear benefits" [14]; upstream rsl_rl 5.2.0 added
`std_range` as a hard clamp defaulting to (1e-6, 1e6) plus `learn_std=False` ("usefull for distillation or
finetuning"), removed the `0.0*mean + std` construction extreme-parkour still uses (NaNs), traced the "std ≥ 0"
crash to float32 underflow, and ships state-dependent σ opt-in [43]; SB3, Spinning Up and skrl clamp log-std at
≤ 2 (appendix). What lacks precedent is a clamp tight enough to bind, 33 % above the init. The opposite failure
exists: PPO "can prematurely shrink the exploration variance" [44]; weighted-MLE updates "always shrink
variance, causing premature convergence", fixed by decoupled KL bounds [45].

**What it means here.** `scalar` vs `log` is not the lever; softening `ACTION_STD_MAX` is (log_std = lo +
½(hi − lo)(tanh(x) + 1), hi = ln 2, or softplus + offset), or a fixed/annealed σ via `learn_std=False`. With init
σ 1.5 half the samples are clipped even at μ = 0 (D16), plausibly feeding the early lottery (formation decided
by iteration ~20–200; RSI inert at σ 1.5). Arms: init σ 0.5–1.0 with a small last-layer init [14], entropy 0 or
0.0025–0.005 [14, 13, 41], scored on formation rate across seeds; target entropy has no support [4, 14]. Without
cap and bonus σ may fall too fast [44, 45]: keep a soft floor near 0.05 and watch Hsu's trap [2].

## 6. Step-size control and what the KL measures (question c)

**Key papers:** Rudin et al. 2021 [46]; rsl_rl ppo.py [47]; extreme-parkour [6]; rl_games CONFIG_PARAMS.md
[48]; MJLAB.md [41]; rl_games PRs #381/#383 [49]; rsl_rl issue #221 [50]; Schulman [39]; Spinning Up [51]; SB3
[52]; 37 details [21]; Andrychowicz [14]; Engstrom et al. [53]; Hilton et al. [54]; KL facts [55]; He [3];
rsl_rl Beta [18].

Krabby's rule is Rudin's Algorithm 1 unmodified (Table 3 minibatch 24,576) [46], identical in rsl_rl master
and extreme-parkour with one Adam group [47, 6]. Rudin et al. say minibatches "much larger than what is
usually considered best practice" stabilise learning [46]; Krabby's are 64–96× smaller and checked 80× per
iteration instead of 20. rl_games' code says per-minibatch stepping "needs reliable KL estimates (large
minibatches)" and its docs advise per-mini-epoch stepping when minibatches are small [48]. Its 3-seed
WujiHand study (manipulation, obs normalisation): the legacy pre-step KL was "mostly drift accumulated over
the pass", braking to the floor for ~800 iterations; fixed 1e-4 beat it on two seeds and tied one
(19.6/18.9/18.9 vs 18.8/18.2/18.9); a per-mini-epoch step-KL scheduler recalibrated to 0.002 scored highest
(20.1/19.1/19.5) with a 95 % interval over fixed spanning −0.1 to +1.0; the "ratchets to min_lr" failure is
per-minibatch stepping on the step-KL signal, one seed [41, 48, 49]. One rsl_rl user reports an adapted LR of
~2.25e-5 and the resume-reset hazard [50].

The PPO paper's adaptive KL adapts a penalty coefficient once per update and lost to clipping (0.74 vs 0.82)
[39]. Mainstream practice fixes or anneals the LR ("0.0003 is a safe default") with optional target-KL early
stopping that resets each iteration (Spinning Up 1.5 × 0.01 per gradient step, π/V optimizers 3e-4/1e-3; SB3
per minibatch; CleanRL per epoch) [14, 51, 52, 21]; the trust region "is heavily dependent on the method with
which the clipped PPO objective is optimized" [53]. PPO is not batch-size invariant; below the critical batch,
Adam compensates a c× smaller batch with α/√c, mapping extreme-parkour's 2e-4 at 36,864 to ~2e-5 at 384
(arithmetic, outside the paper's conditions); clip fraction < 1 % says grow the batch, > 20 % with multiple
epochs says LR too high [54].

What the KL measures decides the remedy. KL is invariant under invertible maps and the log tanh' terms cancel
(App. B.8 choice C63), so a tanh head's Gaussian KL already equals its executed KL [55, 14, 3]. Clip is
many-to-one, so by the data-processing inequality the executed KL is at most the Gaussian KL, consistent with
the 11–16 % share [55]. A Beta head puts the KL on the executed action [18]; a mean bound makes the Gaussian KL
behavioural only when σ is also small, since at σ 2.0 most samples are clipped even with |μ| ≤ 1. rl_games'
`kl_schedule_source=optimizer_step` measures the step and needs a recalibrated target (0.002 → 0.003) [48].

**What it means here.** The controller, not clip 0.2, sets policy movement, so changing it is a first-order
change; the rl_games data say the gain is modest. Cheapest switches: (i) fixed LR 1e-4–3e-4 with per-epoch
target-KL early stop at 0.01–0.02; (ii) per-epoch stepping on a recalibrated step-KL; (iii) a larger batch
(§7). Log KL, clip fraction and executed-action change. Order: a fixed LR on the current head removes the
throttle but the KL stays non-behavioural; with a bounded head *and* small σ, `desired_kl` 0.01 regains its
meaning, so head and σ arms come first.

## 7. Batch geometry, env scaling, γ/λ and time limits (question d)

**Key papers:** Rudin [46]; Isaac Gym [56]; Isaac Lab benchmarks [57]; Isaac Lab paper [58]; Mayor et al. [59];
Beukman et al. [60]; Andrychowicz [14]; SAPG [61]; ANYmal-D cfg [7]; Humanoid-Gym [62]; Lee et al. 2020 [63];
Yang et al. [64]; Seo et al. [65]; Pardo et al. [66].

Rudin et al. varied 128–16,384 robots at batches of 49k/98k/197k: 2,048–4,096 robots at ~100k–200k "provides
the best trade-off", fewer robots (longer rollouts) give "a slow decrease in performance", and PPO "struggles
when we provide fewer than 25 consecutive steps, corresponding to 0.5 s"; ~9 GB at 4,096 with rendering [46].
Isaac Gym held experience fixed: 256 × 512 → 4,096 × 32 cut Humanoid time-to-reward ~10×, beyond which short
horizons gave "sub-optimal gaits" [56]. Isaac Lab: 82k env-steps/s with training for Rough-G1 at 4,096 envs on
one RTX 4090 in 6.1 GB [57]; single-core CPU speed is a residual bottleneck [58]. At fixed data more envs beat
longer rollouts [59]; PPO plateaus when "the outer step size is too large relative to the update noise level",
"tuning the learning rate cannot counteract a poor outer step size", remedy more minibatches at fixed
minibatch size and LR [60] (in tension with Rudin's large minibatches); transitions per iteration "influences
the performance quite significantly" [14]; SAPG's saturation is a ">>10k envs" regime, manipulation only [61].

γ 0.99 at 50 Hz has a ~69-step (1.4 s) half-life and a 17-step (0.34 s) GAE horizon inside the 0.48 s rollout
(arithmetic). ANYmal-D uses 0.99/0.95 at 24 steps [7]; Humanoid-Gym (100 Hz) 0.994/0.9 with 60-step rollouts
[62]; Lee's TRPO teacher 0.995 [63]; Yang et al. (withdrawn preprint) match the half-life to one foot step,
~0.95 at 25 Hz [64]; Seo et al. found γ 0.97 "helpful for simple velocity tracking" [65]. γ is "one of the most
important hyperparameters", tuned per environment from 0.99, with λ 0.9 [14]. Time-out bootstrapping gave Rudin
~10–20 % higher reward [46]; skipping it aliases states [66]; Andrychowicz saw no effect, "probably" because of
1,000-step limits [14].

**What it means here.** Krabby's batch is 8× below Rudin's smallest and its rollout sits at their threshold;
the 384-sample minibatch is what makes the KL noisy. Prescription: 1,024–4,096 envs × 24 steps (batch 25k–98k,
`num_mini_batches` 4, LR bounds re-checked), or 256 × 96–192 steps if VRAM forbids, never shorter rollouts [46,
56]. Reference VRAM at 4,096 envs (6–9 GB) fits 16 GB for RL phases, not the VRAM-bound student; throughput
must be measured on this plant. Fix the `total_terminates` tag (falls bootstrapped as time-outs; D9) before
any batch/γ arm, then test γ 0.995 (half-life ~2.8 s) with λ 0.95–0.97; no source ablates γ for slow
multi-legged gaits.

## 8. Direction-flipped skills in one network (question e)

**Key papers:** Mittal et al. 2024 [67]; Su et al. 2024 [68]; MS-PPO [69]; Ding et al. 2025 [70];
Abdolhosseini et al. 2019 [71]; Kaba et al. [72]; Puny et al. [73]; SWAP [74]; SE-Policy [75]; Reflex [76];
MoE-Loco [77]; PCGrad [78]; CAGrad [79]; Kurin et al. [80]; Rudin et al. 2025 [81]; MDP homomorphic networks
[82]; LEGO [83].

The documented failure of a shared policy on a symmetric task is favouring one mode, not a slow compromise in
both: without augmentation ANYmal-Climb policies "usually learn to climb only in one direction, always using
the same leg first" [67]; Su et al. name "under-exploration of multiple modalities with symmetric states"
[68]. The closest analogue of canonical training is Su's door-pushing result (3 seeds): a hard-equivariant net
trained on one side beat the same net trained on both (65.65 ± 23.16 vs 59.92 ± 17.39 %), PPOaug likewise
(54.39 ± 32.56 vs 50.24 ± 36.52), attributed to domain randomisation breaking symmetric dynamics; vanilla PPO on
both sides reached 61.18 ± 7.56 with a symmetry index of 12.25 vs ~1–4, i.e. imbalance, not failure to learn
[68]. MS-PPO trains Go2/G1 on one lateral side and its symmetry-structured policy tracks the mirrored side
(trot RMSE-O 0.537 ± 0.1 vs MLP 1.115 ± 0.07, 3 seeds), zero-shot to hardware, with no joint arm [69]. The one
canonical-training analogue remaps a quadruped's clock "in time-reversal style when the commanded velocity is
negative"; removing it "mainly increased errors at negative speeds" (no seed count) [70]. Abdolhosseini's PHASE
mirrors the second half-cycle; BASE "fails to learn the Stepper task in all five runs" while every symmetry
method learns it, but PHASE fixes the period and learns slowly [71].

Canonicalisation and frame averaging make any backbone exactly equivariant, shown only in supervised
settings [72, 73]. Hard equivariance gave Su's best returns with the *smaller* seed spread, but PPO-EMLP
"succeeds in only one of three seeds" under faulted locomotion and "fails to converge" with the full command
range under gait rewards [69]; exact symmetry can trap a policy in a neutral pose, "most severe when enforcing
symmetry through network architecture" [71]. SWAP alone states how a history buffer is mirrored: the reflection
"pointwise to each action in sequences" over a 5-step history [74]; SE-Policy trains a G1 jointly on symmetric
commands, no seed count [75]; Reflex adds a critic-invariance term and a decaying weight to the mirror loss (5
seeds, Gym/DMC) [76]. MoE-Loco found negative gradient cosine between quadrupedal and bipedal tasks in an MLP
(−0.132) that an MoE turned positive (+0.278), nothing for mirror directions [77]; gradient surgery [78, 79] is
matched by summed losses [80]. RL from scratch across terrains "collapses to a single mode", fixed by
multi-expert DAgger plus RL fine-tuning [81]. MDP-homomorphic and local-canonicalisation networks have no
locomotion evidence [82, 83].

**What it means here.** Krabby's slow-in-both-hands mode has no precedent; the only evidence that points one
way concerns enforcing the symmetry exactly. Order: (1) a training-time canonicaliser (M_y on obs including the
10-step history and estimator inputs, actions and commands; deterministic tie-break at v_y ≥ 0; history
mirrored per frame [74]; handle mid-episode command flips; keep RSI and random poses, since an M_y-invariant
command plus a symmetric pose forces a symmetric action, the unison glide being that attractor [71]); (2) log
the +y/−y gradient cosine on shared actor layers over iterations 0–600 as the ROOT_CAUSE cancellation
diagnostic [77]; (3) fallback: distil per-direction specialists, then RL-fine-tune [81]. Hard equivariance, MoE
and gradient surgery rank below: each changes every checkpoint consumer for mean gains with high seed variance.

## 9. Symmetry method: loss vs augmentation vs equivariance (question h)

**Key papers:** Mittal [67]; Su [68]; Isaac Lab symmetry cfgs [84]; rsl_rl 5.0.1 ppo.py [47]; Abdolhosseini
[71]; SGMA [85]; Yu, Turk, Liu 2018 [86].

In Krabby's lineage augmentation is the shipped opt-in and the loss is disfavoured: Isaac Lab defaults both
off (`mirror_loss_coeff` 0.0) and ships `WithSymmetry` variants for ANYmal-B/C/D with augmentation over four
images; no config enables the loss [84]. Mittal et al. (3 seeds): augmentation "obtains the highest return and
fastest convergence"; the loss "comes at the cost of performance and slower convergence" and raising its weight
"hinders learning as the gradients ... compete"; augmentation "struggles when initialized weights are high"
(CartPole), a Gaussian and its mirror being close only "for small means and large enough standard deviation"
[67]. Su et al. dropped the loss citing that work [68]. The installed 5.0.1 implements Mittal's Eq. 6 [47].
Abdolhosseini found the off-policy concern "not necessarily a critical issue in practice", but symmetry
methods have "no consistent and predictable impact" on learning speed and "cannot be reliably ranked", LOSS
most consistent at enforcing symmetry and DUP least [71]. SGMA re-runs the RNN on the mirrored sequence; naive
augmentation of a memoryless policy under partial observability hurt even on trained tasks [85]. Yu et al.'s
original loss (hexapod among the bodies) used w = 4 on a SUM over 20,000 samples, not comparable to 0.5 on an
MSE mean [86].

**What it means here.** Augmentation is a config-level change and the cheapest untested symmetry lever, but
saturated heads (|μ| ~ 10) are the regime Mittal's high-init experiment warns about, so start from scratch or
pair with a bounded mean and a small loss, and mirror the full 1,149-dim obs including history and estimator
inputs [85]. Krabby's record (3k gain vanishing at 20k; 1/4 seeds; coefficient only {0, 0.5, 1.0}) fits a
quality lever with seed-dependent learning effects: 3–5 seeds. A loss on raw means (median 0.026 × 0.5 vs
|surrogate| 0.008) spends gradient on differences that change no behaviour (D8); computing it on clip(μ) or
tanh(μ) is a one-line change that also bears on §8.

## 10. Other families: off-policy, model-based, PPO variants (question f)

**Key papers:** Sabatini, Li, Hutter [40]; FastTD3 [87]; FastTD3 repo [88]; Seo et al. [65]; Raffin Parts I/II
[9, 89]; PQL [90]; DreamerV3 [91]; TD-MPC2 [92]; CrossQ [93]; BRO [94]; SimBa [95]; PQN [96]; FPO++ [97]; FPO
[98]; SAPG [61]; EPO [99]; SPO [100].

*Off-policy.* The closest analogue, RSL-RL-SAC (Isaac Lab + RSL-RL; ANYmal, A1, Go1, Go2, H1, G1; 8,192 × 24;
5M replay; RTX 3090; PPO rewards reused; no hexapod or hardware), matched PPO only after σ0 = 0.15 on a tanh
head bounded by joint limits, timeout-aware critic targets and 5-step returns, and still reports "a significant
wall-clock time gap persists, and we argue it is structurally difficult to eliminate" and "higher variability
across random seeds compared to PPO" [40]. FastTD3 solves HumanoidBench in under 3 h on one A100, and the
FastSAC/FastTD3 recipe trains G1/T1 sim-to-real in ~15 min on one RTX 4090 where "PPO struggles with such
strong perturbations": 4,096 envs, batches 8k–32k, obs normalisation, a C51 critic, a tanh actor, < 10 reward
terms [87, 88, 65]. Raffin's SAC on Isaac Lab quadrupeds (1,024 envs) needed bounds from a trained PPO agent's
2.5–97.5th percentiles plus CMA-ES tuning, leaving "a slight performance gap" [9, 89]. Reward and algorithm are
coupled: under one reward PPO and FastTD3 gave "notably different gaits" [87]. PQL (5 seeds) shows off-policy
learning less sensitive to env count only on Shadow Hand [90]. Replay at 1,149 dims: FastTD3 keeps obs and
next_obs as float32 on the GPU, ~9.2 KB per transition, so a 5M buffer is ~46 GB and N = 10,240 at 256 envs
~24 GB; N ~ 1,024 (~2.4 GB) fits; proprio-only storage would cut this ~13× but no paper has done it for an
RMA-style actor (arithmetic) [88]. Symmetry appears only as augmentation [65, 40].

*Model-based and sample-efficiency methods.* DreamerV3's continuous-control wins over a fixed-HP Acme PPO are
at 500K steps with 16 env instances and one A100 per run [91]; TD-MPC2 runs Dog/Humanoid 14M steps and finds
SAC and DreamerV3 "prone to numerical instabilities" [92]; on HumanoidBench "even state-of-the-art RL
algorithms failed to solve many tasks after 48 hours" [87]. CrossQ (UTD 1, single env) and BRO ("within 1
million environment steps") target sample efficiency [93, 94]; PQN is discrete-action [96]. The one
model-based legged sim-to-real result (appendix; the sweep reversed it) matched PPO's real tracking at ~40×
fewer samples and ~5.5× the wall-clock. The transferable piece is SimBa's block (running-stats obs
normalisation + residual MLP + LayerNorm), which improved PPO at 1,024 envs on Craftax [95].

*On-policy variants.* FPO++ is the only variant with legged results at Krabby's env count: Go2, Spot, H1, G1
at 2⁸–2¹² envs, 5 seeds, "almost always" higher returns than Gaussian PPO "with reduced variance between random
seeds", "improved robustness to very small batch sizes", and on Spot at 4,096 envs Gaussian PPO "had a tendency
to learn more symmetric 'pronk' gaits" where FPO++ gave "more consistent 'trot' gaits"; costs: G1 23 vs 19
min, up to 3× on tracking, a new head, a baseline of rsl_rl defaults plus a clip sweep [97, 98]. SAPG splits
24,576 envs among 6 policies (manipulation) [61]; EPO's ANYmal 74.0 ± 6.65 vs PPO 67.7 ± 1.71 is within one EPO
SD at ~24 GB [99]; SPO's quadratic ratio penalty gains only with deep nets on MuJoCo/Atari [100].

**What it means here.** An off-policy arm would change env count, obs normalisation, reward, head and
algorithm at once; every transferable piece (bounded head, small σ0, time-out bootstrapping, obs normalisation,
γ 0.97) is testable inside PPO first, and on Krabby's limiting axes, wall-clock on a shared GPU and seed
variance, the closest analogue is worse [40]. Model-based methods address a budget Krabby does not have (18M
samples ≈ 1.1 GPU-h). FPO++ is a plausible later head arm: a symmetric synchronised gait preferred by Gaussian
PPO under the same reward is the closest published analogue of the unison glide, though its gain over a
*well-parameterised* Gaussian is not established [97].

## 11. Variance, evaluation design and seeds on one GPU (question g)

**Key papers:** Henderson et al. [101]; Colas et al. [102, 103]; Agarwal et al. [104]; Patterson et al. [105];
Eimer et al. [106]; Bjorck et al. [107]; Jordan et al. [108]; Isaac Lab reproducibility [109]; Nagarajan et al.
[110]; ARS [111]; DexPBT [112]; PBRL [113]; Isaac Lab PBT [114]; MAP-Elites [115]; PPGA [116]; DOMiNO [117]; Walk
These Ways [118]; PMTG [119]; Yang & Gao [120]; Schilling et al. [121]; DecQN [122]; Zhou & Vanschoren [123];
GPU-sharing reports [124, 125, 126, 127]; NVIDIA MPS [128]; GMI-DRL [129]; IsaacGymEnvs PBT [130]; Isaac Lab
Ray [131]; torch.func ensembling [132]; dossier [143].

*Seeds.* Two groups of 5 same-config TRPO runs differed at p = 0.0016; averaging under 5 trials "can be
potentially misleading" [101]. With ε = |Δμ|/σ_pool, power 0.8 needs N ≈ 5–10 for ε = 2, ~20 for ε = 1, ~100 for
ε = 0.5; Welch tests at N < 5 inflate false positives; bootstrap is unreliable below N ≈ 10–50 [102, 103]. IQM
and stratified bootstrap are for aggregates across tasks; single-task CIs at small N are "less useful" [104].
"In almost all cases 5 runs is insufficient to make strong claims"; guard against maximisation bias; equalise
tuning budgets [105]. "Top-performing seeds can stem from unstable configurations with low median
performance"; tune and test on separate seeds [106]. Andrychowicz's 3 seeds per cell works only as marginals
over ~250k agents [14]. Seed variance "mostly arises early in training as a result of poor outlier runs", one
cause "numerical instability which leads to saturating nonlinearities", cut by normalising penultimate
features [107]; define an algorithm by its hyper-parameter distribution [108]. Same-seed reruns are not a
paired control on GPU physics: runtime parameter changes reorder work and LSB changes "may lead to divergent
execution" [109]; single nondeterminism sources "can substantially impact the performance" [110].

*Basins, populations, structure.* ARS trained MuJoCo locomotion ~70 % of the time over 100 seeds, failed
seeds converging to hopping, backwards or swirling gaits [111]. DexPBT (8 agents, one 16 GB V100 each) beat
the best of 8 independent PPO seeds at equal compute, starting PBT after 2e8 steps to avoid a population
"filled by copies of one particularly lucky seed" [112]; PBRL beat PPO/SAC/DDPG on Isaac Gym tasks including
Anymal Terrain with 4–16 agents in one process on one RTX 4090, envs partitioned among agents [113]; Isaac
Lab's PBT is rl_games-only, one process per policy [114]. QD and diversity methods (MAP-Elites' ~13,000
hexapod gaits; PPGA; DOMiNO; Walk These Ways' command-selected family) are research options, not screens
[115, 116, 117, 118]. Hexapod results favour structure: PPO modulating CPG parameters beat PPO outputting foot
positions (4 runs) [120]; six decentralised per-leg controllers beat a centralised net over 15 seeds (657 vs
547, p = 0.011) with half the seed SD [121]; PMTG lets a 77-parameter linear policy and PPO learn Minitaur
locomotion where both fall short without it [119]. DecQN (per-actuator 2–3-bin Q-learning, 10 seeds) has no
σ/entropy/KL controller, its Mini Cheetah result qualitative [122]; the one hexapod PPO/SAC/ES comparison
(36 / 43 / 64 %) is confounded [123].

*Seeds on one 16 GB GPU.* Krabby's overlap test (two 256-env runs at 2,783 steps/s each, 97 % of one solo)
matches CUDA time-slicing, so N plain processes cost a serial queue's wall time and VRAM caps ~3 teacher
processes [143]. Documented multi-process failures are rendering- or Jacobian-specific (illegal memory access
in `get_jacobians`; two data-generation jobs per GPU; processes hanging on shader compilation, Kit "is not
designed to run like this") [124, 125, 126, 127], none headless PhysX-only RL. MPS targets exactly the
latency-bound case [128] and beat direct sharing for Isaac Gym PPO on A100s (utilisation "consistently under
50 %") [129], but is unverified with Isaac Lab or GeForce, couples faults across clients, and signal
termination "will lead to undefined behavior" [128], which conflicts with Krabby's kill/timeout job control.
NVIDIA tooling allows 4 agents per GPU and a Ray split with no published cost [130, 131]. The most defensible
route is in-process: seeds as env slices of one 768–1,280-env process with per-slice policies (PBRL's design;
`torch.func.stack_module_state` + `vmap`) [113, 132].

**What it means here.** Protocol: 3 seeds to kill or advance an arm on a large-effect metric (saturation share,
σ behaviour, side-speed class 0.16–0.22 vs 0.03–0.10 m/s); 5 when formation rate decides (reviewer
Fisher-exact: 5/5 vs 0/5 p = 0.008, 4/5 vs 0/5 p = 0.048, 3/3 vs 0/3 p = 0.10) [103, 105]; screen on seeds
{1, 2, 3}, confirm on fresh seeds, never re-use incumbent-ledger s2/s3 as both [106]; report per-seed values,
formation rate and spread as primary outcomes (a bounded head or feature normalisation may shrink the lottery
[107, 97]); equal budgets for the never-tuned baseline [105, 108]; every run an independent draw (D18). After
the in-PPO screens a seed-and-select at formation is the cheap PBT analogue [112]; a 4–6 member population
costs each member 4–6× wall time under time-slicing, so build the env-slice runner first. Structural priors
(an oscillator action space where lateral direction is a phase-rate sign flip; per-leg weight sharing with
left/right permutation) have stronger hexapod evidence than any algorithm swap [120, 121].

## 12. The value side (question i)

**Key papers:** Andrychowicz [14]; Engstrom [53]; 37 details [21]; PPG [133]; Spinning Up [51];
extreme-parkour [6]; PopArt [134]; MAPPO [135]; multi-task PopArt [136]; return-based scaling [137]; legged_gym
[5]; Isaac Lab RewardManager [138], ANYmal-D cfg [7]; Parkour in the Wild [81].

Value-loss clipping "hurts the performance regardless of the clipping threshold" [14]; Engstrom's ablation
names reward normalisation, Adam annealing and initialisation, not value clipping [53]; CleanRL kept it for
fidelity [21]. It is the legged_gym/extreme-parkour default, and the `max(unclipped, clipped)` form
(`ppo_with_extractor.py:420–421`) has zero gradient once V_new moves > 0.2 past V_old, so the critic re-prices a
state by at most ~0.2 per sample per iteration (~7 % of the ~2.9 value scale): a rate limit that binds when a
stage boundary shifts returns by tens of units (reviewer reading; untested as the collapse cause). One Adam
group means the critic trains at 1e-5–3.4e-5 whenever the policy is throttled; the PPO paper shares no
parameters, Spinning Up uses vf_lr 1e-3 > pi_lr 3e-4, PPG lets the value function be "more aggressively
optimized with a higher level of sample reuse", separate networks were better on four of five tasks [39, 51,
133, 14]. PopArt normalises targets online [134]; MAPPO's running-std value normalisation "never hurts training
and often improves the final performance" [135]; Andrychowicz found it crucial on HalfCheetah/Humanoid but
harmful on Walker2d [14]; multi-task PopArt and return-based scaling exist because updates "depend linearly on
the scale of returns" [136, 137]. Input normalisation is "crucial for good performance on all environments
apart from Hopper" [14]; Isaac Lab ships it off [7]. The "≥ 0" clip is a legged_gym heuristic ("avoids early
termination problems") with the termination penalty added *after* the clip; Isaac Lab applies no clip; Krabby
clips the whole sum with no post-clip term [5, 138]. RL fine-tuning of a distilled policy "tends to be
unstable"; the fix is "efficient pre-training of the critic network" with the policy frozen, plus a reduced
initial std [81].

**What it means here.** Chain #4 reset the critic and cleared Adam with no warm-up, and 2b1 degraded. Cheap,
deploy-invisible arms: a separate critic Adam at fixed 3e-4–1e-3 outside the KL rule;
`use_clipped_value_loss=False`; running value normalisation as a 3-seed A/B (test, do not bake); a
frozen-policy critic warm-up at every hand-off; explained variance logged. Reward-clip-off with a small
post-clip termination cost plus value normalisation is a reward-design lever that also unconfounds critic
diagnostics. Observation normalisation changes the checkpoint contract, so bundle it with a from-scratch head
arm. Per-command-category value normalisation is an untested probe for the omni side income [136, 137].

## 13. Distillation from bounded or saturated teachers (question j)

**Key papers:** extreme-parkour [6]; rsl_rl Distillation [139]; RMA [140]; Lee et al. [63]; Miki et al.
[141]; Parkour in the Wild [81]; Czarnecki et al. [142]; Seyde [26].

Extreme Parkour distils on the teacher's `act_inference` output, the unclipped mean, with an L2 norm (plus a
yaw term) and no clipping of the target [6]; rsl_rl's Distillation regresses `teacher(obs).detach()` with
mse/huber [139]. RMA's phase 2 regresses the latent only [140]; Lee et al. regress action and latent under
DAgger [63]; Miki et al. add a 0.5-weighted reconstruction loss [141]; Rudin et al. add action noise during
distillation [81]. Loss variants "can often drastically change the performance" [142]; a Bernoulli BC student
matches a Gaussian teacher on all tasks but Humanoid [26]. No source compares executed vs pre-squash targets
or handles saturated-mean teachers.

**What it means here.** The student regresses means with median |μ| 11–36 that execute as ±1, so the L2 loss
spends weight on magnitudes that change nothing; it has not bitten (0.79/0.64 vs 0.81/0.66) but will with a
bounded teacher. Inference: for a tanh or Beta teacher regress the pre-squash parameter or a distribution KL;
at minimum clip(μ) or Huber targets now; latent regressions are unaffected; the deploy clip stays.

## 14. Literature gaps

- No legged PPO ablation of the rl_games or VideoMimic bound loss; the only rough-terrain rl_games config
  disables it [10, 12, 13].
- Bounded-head task evidence on legged robots is two 2025–2026 studies with weak seed statistics [19, 25];
  rsl_rl's Beta has no locomotion result [18]; categorical heads have no robot evidence [22, 23, 24].
- No stack uses a binding σ clamp; no on-policy legged study varies init std or entropy 0 vs 0.01 [14, 40, 43].
- The per-minibatch KL rule at small minibatches rests on one 3-seed manipulation study whose scheduler
  differences span zero [41, 48]; no locomotion LR trajectories are published.
- No γ/λ or rollout-length ablation for slow multi-legged gaits [46]; no Isaac Lab VRAM at 256–1,024 envs
  [57, 58].
- No RL test of canonical vs joint two-direction training on a periodic gait [68, 69, 70]; none reports a
  slow-in-both-hands mode; only SWAP states how history is mirrored [74].
- No comparison of mirror loss vs augmentation on saturated or bounded means [67, 71].
- No off-policy, model-based or PPO-variant result on a hexapod or a slew-limited actuator in parallel sim; no
  replay design for RMA-style history obs [40, 87, 90].
- No comparison of executed vs pre-squash distillation targets [6, 139, 142].
- No MPS or multi-process PhysX-only RL measurement on Isaac Lab or GeForce; no published co-location slowdown
  [124, 128, 129, 130, 131].
- Load-bearing numbers from single 2026 preprints without replication: [3, 24, 25, 40, 60, 97].

## 15. Ranked recommendations

> **Superseded for planning (2026-09-30):** the test ladder below was challenged by five critiques and two independent checks; the revised ladder, budget and corrections to this section (action_rate is the unsquared L2 norm; there is no deploy clip; the escalation trigger is inverted; γ cut; step size and batch made conditional) are in `2026-09-30_rl_algorithm_review/REPORT.md` §3 and §5-§6. The literature findings above stand.

**Algorithm family: keep PPO now; an off-policy arm later under stated conditions; model-based not at all.**
Neither the census nor the literature indicts PPO as a family; both indict the construction. The closest
off-policy analogue is slower and more seed-variable than PPO and needed the same fixes first [40]; replay at
1,149 dims costs 2.4 GB per 1,024 transitions per env [88]. An off-policy arm (RSL-RL-SAC's released extension,
or FastTD3 with a proprio-only replay) becomes worth its porting cost (estimator/history stubs, mirror
augmentation, terminal labels, checkpoint contract; dossier §6) only if, after rungs 2–5, (a) formation rate
or side-speed class is still seed-dominated under a bounded head with behavioural KL, and (b) rung 1 shows the
learner share (~20 % of wall time) can absorb a UTD ≥ 1 without starving the simulator. Head alternatives
inside PPO come first [22, 23, 24, 25, 97].

Cost basis: 1.1–1.2 GPU-h per 3,000-iteration from-scratch arm at 256 envs; an early kill-read at ~1,000
iterations ~0.4 GPU-h (dossier §6). Seeds follow §11: 3 for large-effect reads, 5 where formation rate decides.
Knobs are this repo's (`crab_actor_critic_with_encoder.py`, `ppo_with_extractor.py`, `rsl_rl_ppo_cfg.py`,
`vecenv_wrapper.py`, `parkour_reward_manager.py`, `train.py --num_envs`).

0. **Telemetry and defect fixes, 0 GPU-h** (§6, §7, §11; [54, 3, 46, 66]). Log per-minibatch and
   per-iteration KL, clip fraction, mean within-iteration LR, explained variance, |μ| > 1 share, executed-vs-raw
   change per update, sign flips per joint-second (dossier §3.7); compute He et al.'s P10/P01 from the existing
   eval npz; split the `total_terminates` falls out of the `time_out` tag (D9); make unknown `algorithm` cfg keys
   raise (D14).
1. **Throughput and co-location probe, ~0.5 GPU-h** (§7, §11; [46, 57, 58, 128, 129]). One process at
   `--num_envs` 512/1024/2048/4096 for ~200 iterations (collection time, VRAM, physics stability); one
   2–3-process headless smoke with and without MPS. Dossier: +5 % for 4× envs to 256, 61–69 % utilisation,
   4.4 GB, 2,783 steps/s each when overlapped. If 1,024–4,096 envs are near-free, every arm below gains 4–16×
   samples at ~1.2–1.5 GPU-h.
2. **Bounded mean × gradient-passing σ, from scratch, 2 × 2 factorial, 3 seeds, ≈ 13–14 GPU-h** (§2, §3, §5;
   [12, 13, 3, 14, 40, 43]). Head: {unbounded Gaussian; rl_games bound loss on raw μ, soft bound 1.1,
   `bound_loss_coef` 1e-4 ramped to 1e-3, or tanh mean with executed-action entropy}. σ: {`init_noise_std` 1.5 +
   clamp 2.0 + `entropy_coef` 0.01; vs `init_noise_std` 0.5–1.0, log-std through tanh into [ln 0.05, ln 2] or
   softplus + offset, `entropy_coef` 0 (0.0025–0.005 fallback), last-layer init 0.01×}. Dossier: 97 % |μ| > 1,
   σ pinned 59/59, M1 as a fine-tune, half the samples clipped at μ = 0. Gates: saturation share and P10, sign
   flips, formation rate over seeds, tracking and side-speed class; smoothness scored separately (§4). The two
   knobs must be co-varied: lowering σ alone cannot produce an interior mean [3], and a bound alone at σ 2.0
   leaves the clip binding and the KL non-behavioural (§6). Interactions: mirror loss on the bounded mean [67];
   distil bounded teachers on the executed action or pre-squash parameter (§13); the deploy clip stays.
3. **Step-size rule on the winning head, 2 arms × 3 seeds ≈ 7 GPU-h** (§6; [46, 48, 41, 54, 51]). {fixed LR
   1e-4 and 3e-4, linear anneal, per-epoch target-KL early stop at 0.01–0.02} vs {per-epoch step-KL, target
   recalibrated from logged KL} vs the incumbent. Dossier: LR erased within iteration 0 in 94 % of fresh runs,
   62.9 % floor residency, 11–16 % behavioural KL on W. Expect a modest gain [41]; after rung 2, because the KL
   is behavioural only with a bounded head and small σ.
4. **Batch geometry, 2 arms × 3 seeds ≈ 7–9 GPU-h** (§7; [46, 56, 60, 54]). `--num_envs` 1,024–4,096 ×
   `num_steps_per_env` 24, `num_mini_batches` 4, `desired_kl` 0.01, LR bounds re-checked; or 256 × 96–192 steps
   if VRAM forbids; never fewer than 24 steps. Dossier: batch 8× below Rudin's smallest, minibatch 384, GPU at
   66 W. Changes the KL estimate's noise, so re-read rung 3's telemetry.
5. **Value-side hygiene, ≈ 7 GPU-h plus ~0.5 per hand-off** (§12; [14, 51, 133, 135, 5, 81]). Separate critic
   Adam at fixed 3e-4–1e-3; `use_clipped_value_loss=False`; running value normalisation as its own 3-seed A/B;
   frozen-policy critic warm-up (~200 iterations) at every hand-off; reward clip off with a small post-clip
   termination cost. Dossier: one Adam group, bridge critic 0.5 vs 0.0015/step, 13.6 % of return forgiven.
6. **Symmetry: augmentation vs loss on bounded means, 5 seeds ≈ 6 GPU-h** (§9; [67, 84, 47, 71, 85]).
   `use_data_augmentation=True`, `mirror_loss_coeff` 0 or small, from scratch on the rung-2 head, mirror function
   over the full 1,149-dim obs. Dossier: 0.573 vs 0.032 at 3k seed 1; E1 0.541/0.0/0.026/0.0.
7. **Canonical training for the two-hand failure, 3 seeds + confirm ≈ 4–8 GPU-h** (§8; [68, 69, 70, 74, 81]).
   M_y inside rollout and update gated on sign(v_y cmd), history mirrored per frame, deterministic tie-break;
   log the +y/−y gradient cosine over iterations 0–600 [77]. Dossier: 0.028–0.103 vs 0.184–0.217 m/s;
   canonicaliser fast both hands at deploy; 99 % sign-agreement distillation. Fallback: specialists distilled then
   RL-fine-tuned [81].
8. **γ 0.995 / λ 0.95–0.97, 3 seeds ≈ 3.6 GPU-h** (§7; [14, 62, 63, 65]), after rung 0's time-out fix. Dossier:
   0.34 s GAE horizon inside a 0.48 s rollout on a 1.2–2.8 s gait.
9. **Later head alternatives** (§3, §10; [22, 23, 24, 25, 97]): ordinal head at K = 11 with entropy 0, GPO's
   growing bound, FPO++; only if rung 2 moves geometry but not task metrics. Then the off-policy arm under the
   conditions above.

Rungs 0–8 total ≈ 50–60 GPU-h, about a tenth of the ~528 GPU-h spent under the never-varied configuration.

**Where the literature disagrees with the dossier.** (i) The dossier calls the parameterisation "the wrong
shape for the actuator interface" and expects task metrics to move; the literature says near-bang-bang may be
close to this objective's optimum [26, 27] and that bounded heads change geometry reliably but return
inconsistently [3, 4, 22], the legged exceptions being statistically weak [19, 25]; the head restores the
learning signal, and smoothness must be bought with an executed-action cost [36, 34]. (ii) The dossier treats
velocity cams as fixed; the literature gives position targets a modest learning-speed prior [28] but favours
velocity-type spaces for transfer and continuous rotation [29, 31, 32], and the rate-limited rods saturate
like the cams [38]. (iii) Hsu's failure mode 3 says a narrow init can worsen basin selection [2], a risk the
dossier does not carry, so rung 2 scores formation rate. (iv) On minibatches, Beukman et al. want more at fixed
size and Rudin very large ones [60, 46]; both agree with the dossier that the lever is more envs, but the
literature adds that rollouts under ~25 steps hurt gait discovery [46, 56]. (v) rl_games' only controlled data
show fixed-vs-adaptive differences spanning zero [41], so the step-size rule matters mainly through what the
KL measures, not as the first-order throttle the dossier reads. (vi) On the mirror loss Krabby's lineage
favours augmentation [67, 84] while Abdolhosseini rank LOSS most consistent with no learning-speed effect for
any method [71]: the literature is split, hence 5 seeds at rung 6.

**Interactions that fix the order.** Head ↔ σ (rung 2, co-varied). Head ↔ KL controller (behavioural only with
a bounded head and small σ; rung 3 after 2). Batch ↔ KL controller (rung 4 changes the estimate's noise). Head
↔ mirror loss (loss on bounded means; augmentation only on small-mean heads; rungs 2, 6). Head ↔ distillation
target (§13). Obs normalisation ↔ checkpoint contract (rung 5, bundled with a from-scratch head arm). Value
normalisation ↔ curriculum stages (test per stage). γ ↔ time-out tag (rung 0 before 8). Canonicaliser ↔
history/estimator inputs and mid-episode command flips (rung 7).

## 16. References

1. Chou, Maturana, Scherer. *Improving Stochastic Policy Gradients in Continuous Control with Deep RL using the Beta Distribution.* ICML 2017, PMLR 70. https://proceedings.mlr.press/v70/chou17a.html — Sec. 3.2 boundary bias grows with σ; Sec. 5 TRPO+Beta "about 2x faster" on Humanoid-v1.
2. Hsu, Mendler-Dünner, Hardt. *Revisiting Design Choices in Proximal Policy Optimization.* arXiv:2009.10897, 2020. https://arxiv.org/abs/2009.10897 — Sec. 3.1 failure mode 1 (toy, shrinking std); failure mode 3 narrow-init trap; Sec. 4.2 Beta as stand-alone fix; 2x Humanoid-v2.
3. He, Zhou, Wang, Xue, Fei. *Where Entropy Is Measured Matters: Policy Geometry in Bounded Continuous-Control PPO.* arXiv:2608.24488, Aug 2026 (preprint). https://arxiv.org/abs/2608.24488 — abstract percentages; Sec. 5.3/5.7 Table 3; Sec. 6.5 "reducing variance alone cannot produce an interior mean".
4. Geiß, Al-Hafez, Seyfarth, Peters, Tateo. *Exciting Action: Investigating Efficient Exploration for Learning Musculoskeletal Humanoid Locomotion.* Humanoids 2024; arXiv:2407.11658. https://arxiv.org/abs/2407.11658 — Sec. IV bang-bang drift; mean-regularising objectives; +200 flipped-uniform KL; bounded heads "did not pay dividends".
5. Rudin et al. legged_gym `legged_robot_config.py`, `legged_robot.py`, master. https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py — clip_actions 100; action_scale 0.5; init_noise_std 1.0; entropy 0.01; 4 minibatches; `only_positive_rewards` clip then termination reward.
6. Cheng, Shi, Agarwal, Pathak. extreme-parkour (ICRA 2024 code): `legged_robot_config.py`, `a1` task cfg, `rsl_rl/algorithms/ppo.py`, `actor_critic.py`, `on_policy_runner.py`. https://github.com/chengxuxin/extreme-parkour/blob/main/rsl_rl/rsl_rl/algorithms/ppo.py — clip_actions 1.2 / action_scale 0.25 (±4.8); std as unclamped Parameter; per-minibatch KL rule; one Adam; clipped value loss; depth-student L2 on unclipped `act_inference`.
7. Isaac Lab. `velocity_env_cfg.py`, `anymal_d/agents/rsl_rl_ppo_cfg.py`, `isaaclab_rl/rsl_rl/rl_cfg.py`; PR #2019 "Adds action clipping to rsl-rl wrapper" (merged 2025-03-13). https://github.com/isaac-sim/IsaacLab/pull/2019 — scale 0.5; init_std 1.0; entropy 0.005; clip_actions None; obs_normalization False; 4 × 24,576; γ 0.99 / λ 0.95.
8. Isaac Lab PR #984 "Clips actions to large limits before applying them to the environment" (closed unmerged 2025-03-05). https://github.com/isaac-sim/IsaacLab/pull/984 — last_action → critic-error → larger-action loop.
9. Raffin. *Getting SAC to Work on a Massive Parallel Simulator (Part I).* Blog, Feb 2025 (ICLR 2026 blogpost track). https://araffin.github.io/post/sac-massive-sim/ — Box(−100, 100); "less than 5 % of the action space"; SAC slower, inconsistent, slightly below PPO.
10. NVIDIA IsaacGymEnvs train configs (AnymalPPO, AnymalTerrainPPO, AntPPO, HumanoidPPO, ShadowHandPPO, FrankaCabinetPPO). https://github.com/isaac-sim/IsaacGymEnvs/tree/main/isaacgymenvs/cfg/train — bounds_loss_coef 0.001 / 1e-4 / 0; entropy 0; kl_threshold 0.008; minibatch 32,768.
11. Isaac Lab Ant `rl_games_ppo_cfg.yaml`. https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/agents/rl_games_ppo_cfg.yaml — clip_actions 1.0; bounds_loss_coef 1e-4; entropy 0.
12. Makoviichuk, Makoviychuk. rl_games `algos_torch/a2c_continuous.py`, `a2c_common.py`. https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/a2c_continuous.py — `bound_loss` (soft_bound 1.1); loss assembly; clip_actions default True.
13. Allshire, Choi, Zhang, McAllister et al. *Visual Imitation Enables Contextual Humanoid Control (VideoMimic).* CoRL 2025; arXiv:2505.03729 v5. https://arxiv.org/abs/2505.03729 — App. B: clip against early bang-bang; L1 bounds loss 0.0005; limit 8.0; init std 0.8; entropy 0.0025; desired_kl 0.02.
14. Andrychowicz et al. *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study.* ICLR 2021; arXiv:2006.05990. https://arxiv.org/abs/2006.05990 — Sec. 3.2 init std 0.5, +66 % Humanoid; Sec. 3.3 input/value normalisation; Sec. 3.4 value clipping hurts, λ 0.9; Sec. 3.5 batch; Sec. 3.6 γ; Sec. 3.7 LR; Sec. 3.8 regularisers; App. B.8 choice C63 / fn 27 tanh log-det.
15. Haarnoja, Zhou, Abbeel, Levine. *Soft Actor-Critic.* ICML 2018; arXiv:1801.01290. https://arxiv.org/abs/1801.01290 — App. C tanh change of variables.
16. Google Brax `training/agents/ppo/networks.py`, `training/distribution.py`. https://github.com/google/brax/blob/main/brax/training/agents/ppo/networks.py — distribution_type default 'tanh_normal'; softplus + min_std 0.001; entropy with single-sample log-det.
17. MuJoCo Playground `config/locomotion_params.py`. https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/config/locomotion_params.py — 8192 envs; entropy_cost 1e-2; unroll 20; 32 minibatches; discount 0.97; Go1 joystick (512, 256, 128).
18. leggedrobotics/rsl_rl `modules/distribution.py` (BetaDistribution) and PR #202 "Adds a Beta distribution" (merged 2026-05-07, v5.3.0); PR #81 (closed). https://github.com/leggedrobotics/rsl_rl/pull/202 — softplus + 1; action_range (−1, 1); Beta–Beta KL; single-seed thruster result on the PR.
19. Liu, Teng, Liu, Zhang, Ghaffari. *Discrete-Time Hybrid Automata Learning: Legged Locomotion Meets Skateboarding (DHAL).* arXiv:2503.01842, 2025. https://arxiv.org/abs/2503.01842 — Sec. IV-C/VI Gaussian bias; Table III Ours-wo-Beta fails every scenario, 5 trials; 4 training seeds.
20. Fujita, Maeda. *Clipped Action Policy Gradient.* ICML 2018, PMLR 80. https://proceedings.mlr.press/v80/fujita18a.html — Sec. 5: 3/10 (PPO), 7/10 (TRPO) at p < 0.025.
21. Huang, Dossa, Raffin, Kanervisto, Wang. *The 37 Implementation Details of Proximal Policy Optimization.* ICLR 2022 Blog Track. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ — unclipped sample stored; log std init 0; ent_coef 0; value clipping kept for fidelity; anneal_lr; target_kl per epoch.
22. Tang, Agrawal. *Discretizing Continuous Action Space for On-Policy Optimization.* AAAI 2020; arXiv:1901.10500. https://arxiv.org/abs/1901.10500 — Table 2 Humanoid-v1; App. A Gaussian+tanh; App. B 7 ≤ K ≤ 15; Table 3 wall time.
23. Zhu, Wang, Zhu, Chen, Zhao. *Discretizing Continuous Action Space with Unimodal Probability Distributions for On-Policy RL.* IEEE TNNLS 2024; arXiv:2408.00309. https://arxiv.org/abs/2408.00309 — Table II Humanoid-v3; K 9–15; τ 2.5.
24. Bian, Feng, Wang, Li, Gao, Shi. *RN-D: Discretized Categorical Actors for On-Policy Reinforcement Learning.* arXiv:2601.23075 v2, 2026. https://arxiv.org/abs/2601.23075 — 1.3–1.9× fewer steps, 5 seeds; entropy 0; K = 41; 873.8 vs 854.4 SPS; loss-swap ablation.
25. Liao, Li, Yang, Chang, Fan, Wang, Shi, Cao, Wu, Sartoretti. *GPO: Growing Policy Optimization for Legged Robot Locomotion and Whole-Body Control.* arXiv:2601.20668, 2026. https://arxiv.org/html/2601.20668 — ã = β_t tanh(a/β_t) with Jacobian; Table II hexapod tracking; hardware 100 % vs 0–40 %; seeds unreported.
26. Seyde, Gilitschenski, Schwarting, Stellato, Riedmiller, Wulfmeier, Rus. *Is Bang-Bang Control All You Need?* NeurIPS 2021; arXiv:2111.02552. https://arxiv.org/abs/2111.02552 — Sec. 3 Pontryagin cases; Fig. 2 Bernoulli on par (4 seeds); Sec. 5 MPO extremal actions, BC student, penalties, low-pass remark.
27. Liberzon. *Calculus of Variations and Optimal Control Theory: A Concise Introduction* (lecture-notes draft, 2011; Princeton UP 2012). http://liberzon.csl.illinois.edu/teaching/cvoc.pdf — Sec. 4.4.2 bang-bang principle for normal linear systems.
28. Peng, van de Panne. *Learning Locomotion Skills Using DeepRL: Does the Choice of Action Space Matter?* SCA 2017; arXiv:1611.01055. https://arxiv.org/abs/1611.01055 — Sec. 5 / Table 4 AUC PD > Vel 7/7; Vel = damping servo; query-rate paragraph.
29. Aljalbout, Frank, Karl, van der Smagt. *On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer.* IEEE RA-L 2024; arXiv:2312.03673. https://doi.org/10.1109/LRA.2024.3398428 — velocity spaces best; multi-step integrators worst; clipping mass at borders.
30. Bogdanovic, Khadiv, Righetti. *Learning Variable Impedance Control for Contact Sensitive Tasks.* IEEE RA-L / IROS 2020; arXiv:1907.07500. https://arxiv.org/abs/1907.07500 — Sec. V fixed-gain oscillations; Sec. VI tracking term.
31. Bellegarda, Ijspeert. *CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion.* IEEE RA-L 2022; arXiv:2211.00458. https://arxiv.org/abs/2211.00458 — Sec. IV: similar returns; joint-PD gaits overfit the simulator; best tracking for CPG-RL.
32. Lee, Bjelonic, Reske, Wellhausen, Miki, Hutter. *Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots.* Science Robotics 9(89), 2024; arXiv:2405.01792. https://arxiv.org/abs/2405.01792 — 12 position + 4 wheel-velocity actions; Beta for the bounded high-level action.
33. Krabby project. *Geometry alignment + camshaft velocity actions*, RESULTS.md / CHANGELOG.md, 2026-08-13. `/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_1900_geometry_velocity_actions/RESULTS.md` — from-scratch velocity arm oscillates shafts at ~1.4 Hz; supersedes Arm B position window.
34. Chen, He, Wang, Liao, Ze, Li, Sastry, Wu, Sreenath, Gupta, Peng. *Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies.* IROS 2025; arXiv:2410.11825. https://arxiv.org/abs/2410.11825 — Table I(a) return/jitter for none, smoothness reward, LCP, low-pass filter.
35. Guldner, Utkin. *The chattering problem in sliding mode systems.* MTNS 2000. https://mathweb.ucsd.edu/~helton/MTNSHISTORY/CONTENTS/2000PERPIGNAN/CDROM/articles/SI14_4.pdf — neglected fast dynamics excited by switching; boundary-layer saturation function.
36. Caponigro, Ghezzi, Piccoli, Trélat. *Regularization of chattering phenomena via bounded variation controls.* IEEE TAC 2018; arXiv:1303.5796. https://arxiv.org/abs/1303.5796 — Fuller phenomenon; total-variation penalty; quasi-optimal convergence.
37. Wang, Hu, Li, Lin, Lin, He. *Chattering Phenomena in Time-Optimal Control for High-Order Chain-of-Integrator Systems with Full State Constraints.* arXiv:2403.17675, 2024. https://arxiv.org/abs/2403.17675 — chattering proven for constrained chain-of-integrator problems.
38. Krabby project. `parkour_actions.py` (`CrabHexDelayedJointPositionAction`), `crab_hex_dimensions.py`. `/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp/parkour_actions.py` — L227–236 rod clamp and slew; L292–298 cam accel limit; CAM_ACCEL_LIMIT_RAD_S2 = π/0.5.
39. Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347 — Sec. 4 adaptive-KL penalty (0.74) vs clipping (0.82); Sec. 6.1 no entropy bonus, no shared parameters; Table 4 Roboschool log-std LinearAnneal(−0.7, −1.6).
40. Sabatini, Li, Hutter. *Bridging the Gap: Enabling Soft Actor Critic for High Performance Legged Locomotion.* arXiv:2605.24975, May 2026. https://arxiv.org/abs/2605.24975 — Sec. 3.3 σ0 0.15 vs 1.0; Table C10 PPO baseline; Sec. 5 wall-clock gap, seed variability; 8192 × 24; replay 5e6; RTX 3090.
41. Denys88/rl_games `docs/MJLAB.md`. https://github.com/Denys88/rl_games/blob/master/docs/MJLAB.md — WujiHand scheduler table (fixed 19.6/18.9/18.9; legacy 18.8/18.2/18.9; step-KL 20.1/19.1/19.5); sigma-runaway note; entropy 0.
42. CleanRL `sac_continuous_action.py`. https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py — tanh log-std into [−5, 2].
43. leggedrobotics/rsl_rl PRs #66 (std gradient), #67 (log-std), #112 (state-dependent std), #190 (closed; underflow), #201 (std_range / learn_std, v5.2.0). https://github.com/leggedrobotics/rsl_rl/pull/201 — hard clamp default (1e-6, 1e6); `learn_std` "usefull for distillation or finetuning".
44. Hämäläinen, Babadi, Ma, Lehtinen. *PPO-CMA: Proximal Policy Optimization with Covariance Matrix Adaptation.* IEEE MLSP 2020; arXiv:1810.02541. https://arxiv.org/abs/1810.02541 — PPO "can prematurely shrink the exploration variance".
45. Abdolmaleki, Springenberg, Degrave, Bohez, Tassa, Belov, Heess, Riedmiller. *Relative Entropy Regularized Policy Iteration.* arXiv:1812.02256, 2018. https://arxiv.org/abs/1812.02256 — Sec. 4.2.1 decoupled mean/covariance KL bounds; experiments: MLE "always shrinks variance".
46. Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning.* CoRL 2021, PMLR 164; arXiv:2109.11978. https://arxiv.org/abs/2109.11978 — Algorithm 1; Table 3; Sec. 2.2 large minibatches; Fig. 4 robots × batch; < 25 steps; A6000 scaling; App. A.2 time-out bootstrap +10–20 %.
47. leggedrobotics/rsl_rl `algorithms/ppo.py` (master; installed 5.0.1 inspected locally). https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py — per-minibatch KL rule; one optimizer; augmentation branch `.repeat(num_aug, 1)`; mirror-loss MSE (5.0.1 lines 249–252, 338–344).
48. Denys88/rl_games `docs/CONFIG_PARAMS.md`. https://github.com/Denys88/rl_games/blob/master/docs/CONFIG_PARAMS.md — schedule_type; kl_schedule_source optimizer_step; "needs reliable KL estimates"; recalibrate to 0.002.
49. Denys88/rl_games PR #381 (exact KL, step-KL scheduler, merged 2026-09-22) and PR #383 (mixed precision, 2026-09-29). https://github.com/Denys88/rl_games/pull/381 — "brakes to its floor for ~800 iterations"; bf16 rounding KL.
50. leggedrobotics/rsl_rl issue #221 "PPO self.learning_rate not synced from optimizer after load_state_dict()" (closed 2026-07-15). https://github.com/leggedrobotics/rsl_rl/issues/221 — adapted LR "e.g. to 2.25e-5"; resume reset.
51. Achiam. *Spinning Up: Proximal Policy Optimization* (docs and `ppo.py`). https://spinningup.openai.com/en/latest/algorithms/ppo.html — pi_lr 3e-4, vf_lr 1e-3, separate optimizers; early stop at 1.5 × target_kl per gradient step.
52. Raffin et al. Stable-Baselines3 PPO docs and `ppo.py`. https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html — target_kl default None, checked per minibatch.
53. Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph, Madry. *Implementation Matters in Deep Policy Gradients.* ICLR 2020; arXiv:2005.12729. https://arxiv.org/abs/2005.12729 — trust region depends on the optimiser; Fig. 1 ablation.
54. Hilton, Cobbe, Schulman. *Batch size-invariance for policy optimization.* NeurIPS 2022; arXiv:2110.00641. https://arxiv.org/abs/2110.00641 — Sec. 3.2 α/√c; Sec. 6 LR most important; clip-fraction guidance.
55. Wikipedia. *Kullback–Leibler divergence* (invariance under parameter transformations); *f-divergence* (data-processing inequality). https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence ; https://en.wikipedia.org/wiki/F-divergence
56. Makoviychuk et al. *Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning.* NeurIPS 2021 D&B; arXiv:2108.10470. https://arxiv.org/abs/2108.10470 — Sec. 5 horizon scaling; Sec. 5.2 Humanoid 10⁴ → 10³ s; sub-optimal gaits at short horizons.
57. Isaac Lab. *Performance Benchmarks.* https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html — Rough-G1 4096 envs, RTX 4090: 82k FPS train; 6.1 GB VRAM.
58. Mittal, Roth, Tigue, Richard et al. *Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning.* arXiv:2511.04831, 2025. https://arxiv.org/abs/2511.04831 — Sec. 4.1 single-core CPU bottleneck; 5090 throughput at small env counts.
59. Mayor, Obando-Ceron, Courville, Castro. *The Impact of On-Policy Parallelized Data Collection on Deep RL Networks.* arXiv:2506.03404, 2025. https://arxiv.org/abs/2506.03404 — N_envs over N_RO; feature rank; dormant neurons.
60. Beukman, Khetarpal, Zheng, Dabney, Foerster, Dennis, Lyle. *Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments.* arXiv:2603.06009 v2, 2026. https://arxiv.org/abs/2603.06009 — outer step size vs update noise; add minibatches at fixed size and LR.
61. Singla, Agarwal, Pathak. *SAPG: Split and Aggregate Policy Gradients.* ICML 2024 (oral); arXiv:2407.20230. https://arxiv.org/abs/2407.20230 — 24,576 envs, M = 6; manipulation only.
62. RobotEra Humanoid-Gym `humanoid_config.py`. https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_config.py — 100 Hz; γ 0.994; λ 0.9; 60 steps; 2 epochs; entropy 0.001.
63. Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning quadrupedal locomotion over challenging terrain.* Science Robotics 5(47), 2020; arXiv:2010.11251. https://arxiv.org/abs/2010.11251 — Table S7 γ 0.995 (TRPO teacher); Eq. 1 action + latent DAgger; S9 ablation.
64. Yang, Yuan, Heng, Komura, Li. *Learning natural locomotion behaviors for humanoid robots using human knowledge.* arXiv:2005.10195 v1, 2020 (WITHDRAWN on arXiv; v1 PDF only). https://arxiv.org/pdf/2005.10195v1 — γ^13 = 0.5 → 0.95 at 25 Hz.
65. Seo, Sferrazza, Chen, Shi, Duan, Abbeel. *Learning Sim-to-Real Humanoid Locomotion in 15 Minutes.* arXiv:2512.01996, 2025. https://arxiv.org/abs/2512.01996 — γ 0.97 for velocity tracking; batch 8K; LayerNorm; averaged Q; C51; < 10 reward terms; symmetry augmentation.
66. Pardo, Tavakoli, Levdik, Kormushev. *Time Limits in Reinforcement Learning.* ICML 2018; arXiv:1712.00378. https://arxiv.org/abs/1712.00378 — partial-episode bootstrapping; state aliasing.
67. Mittal, Rudin, Klemm, Allshire, Hutter. *Symmetry Considerations for Learning Task Symmetric Robot Policies.* ICRA 2024; arXiv:2403.04359. https://arxiv.org/abs/2403.04359 — Sec. IV-C aug vs loss (3 seeds); Sec. IV-D high-init; Sec. IV-F2 one-direction climbing; Eq. 6 ratio.
68. Su, Huang, Ordoñez-Apraez, Li, Li, Liao, Turrisi, Pontil, Semini, Wu, Sreenath. *Leveraging Symmetry in RL-based Legged Locomotion Control.* IROS 2024; arXiv:2403.17320; DOI 10.1109/IROS58592.2024.10802439. https://arxiv.org/abs/2403.17320 — Sec. VI-B4 / Table I one side vs both (3 seeds); PPO 61.18 ± 7.56 both sides; symmetry index; loss opted out citing prior work.
69. Wei, Chen, Xie, Katz, Gan, Gan. *Beyond Topology: A Morphological Symmetry Graph Representation for Locomotion Policy Learning (MS-PPO).* arXiv:2512.00727 v2, 2026. https://arxiv.org/abs/2512.00727 — Sec. 4.1 PPO-EMLP 1/3 seeds; App. D.2.1 walk-to-one-side; trot RMSE-O 0.537 vs 1.115.
70. Ding, Chen, Katz, Gan. *Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy Enables Free Gait Transitions at Varying Speeds.* arXiv:2510.10455, 2025. https://arxiv.org/html/2510.10455v1 — Sec. III-D3 Eq. 9 time-reversal phase remap; Sec. IV-C ablation.
71. Abdolhosseini, Ling, Xie, Peng, van de Panne. *On Learning Symmetric Locomotion.* MIG 2019. https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/2019-MIG-symmetry.pdf — Sec. 4 DUP/LOSS/PHASE/NET; Sec. 7.1–7.2 rankings, BASE 0/5 on Stepper; Sec. 8 neutral-pose trap.
72. Kaba, Mondal, Zhang, Bengio, Ravanbakhsh. *Equivariance with Learned Canonicalization Functions.* ICML 2023; arXiv:2211.06489. https://arxiv.org/abs/2211.06489 — plug-in canonicalisation, universality; supervised only.
73. Puny, Atzmon, Ben-Hamu, Misra, Grover, Smith, Lipman. *Frame Averaging for Invariant and Equivariant Network Design.* ICLR 2022; arXiv:2110.03336. https://arxiv.org/abs/2110.03336 — exact equivariance preserving backbone expressivity; supervised only.
74. Lan, Wang, Li, Jiang, Fu, Su, Wong, Jin, Wang. *SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour.* arXiv:2606.19928, 2026. https://arxiv.org/html/2606.19928 — 5-step history mirrored pointwise per element; one-sided terrain transfer.
75. Nie, Zhang, Jin, Cao, Lin, Yang, Gao. *Coordinated Humanoid Robot Locomotion with Symmetry Equivariant RL Policy (SE-Policy).* AAAI 2026; arXiv:2508.01247. https://arxiv.org/html/2508.01247 — joint symmetric command sampling; equivariant actor / invariant critic; no seed count.
76. Zhen, Zhang, Wang, Yu. *Reflex: RL with Reflection Symmetry Exploitation in State-Based Continuous Control.* arXiv:2605.23415, 2026. https://arxiv.org/html/2605.23415 — policy + critic symmetry regularisers, w_t = w0(1 − t/T); 5 seeds; RAD baselines only.
77. Huang, Zhu, Du, Zhao. *MoE-Loco: Mixture of Experts for Multitask Locomotion.* IROS 2025; arXiv:2503.08564. https://arxiv.org/abs/2503.08564 — Table III gradient cosine −0.132 (MLP) vs 0.278 (MoE), quadruped–biped pairs only.
78. Yu, Kumar, Gupta, Levine, Hausman, Finn. *Gradient Surgery for Multi-Task Learning (PCGrad).* NeurIPS 2020; arXiv:2001.06782. https://arxiv.org/abs/2001.06782
79. Liu, Liu, Jin, Stone, Liu. *Conflict-Averse Gradient Descent for Multi-task Learning (CAGrad).* NeurIPS 2021; arXiv:2110.14048. https://arxiv.org/abs/2110.14048
80. Kurin, De Palma, Kostrikov, Whiteson, Kumar. *In Defense of the Unitary Scalarization for Deep Multi-Task Learning.* NeurIPS 2022; arXiv:2201.04122. https://arxiv.org/abs/2201.04122 — summed losses + regularisation match multi-task optimisers.
81. Rudin, He, Aurand, Hutter. *Parkour in the Wild: Learning a General and Extensible Agile Locomotion Policy Using Multi-expert Distillation and RL Fine-tuning.* arXiv:2505.11164, 2025 (IJRR 2026, doi 10.1177/02783649261455067). https://arxiv.org/abs/2505.11164 — Sec. 2.2 DAgger loss + action noise; Sec. 2.3 single-mode collapse; fine-tuning instability, critic pre-training with frozen policy, reduced std.
82. van der Pol, Worrall, van Hoof, Oliehoek, Welling. *MDP Homomorphic Networks.* NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/2be5f9c2e3620eb73c2972d7552b6cb5-Abstract.html — CartPole, grid world, Pong only.
83. Wang, Zhong, Chang, Allen-Blanchette. *Local-Canonicalization Equivariant GNNs for Swarm Robot Control (LEGO).* IROS 2026; arXiv:2509.14431. https://arxiv.org/abs/2509.14431 — MARL/drones, no locomotion.
84. Isaac Lab ANYmal-D `rsl_rl_ppo_cfg.py` (`AnymalD{Flat,Rough}PPORunnerWithSymmetryCfg`), `isaaclab_rl/rsl_rl/symmetry_cfg.py`, `mdp/symmetry/anymal.py` (main; local clone b0542fe2d). https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py — augmentation opt-in over 4 images; mirror_loss_coeff 0.0; no config uses the loss.
85. Bao, Li, As, Krause, Hutter. *Symmetry-Guided Memory Augmentation for Efficient Locomotion Learning (SGMA).* arXiv:2502.01521, 2025. https://arxiv.org/abs/2502.01521 — Sec. IV RNN hidden-state augmentation; Sec. V naive feedforward augmentation hurts under partial observability.
86. Yu, Turk, Liu. *Learning Symmetric and Low-Energy Locomotion.* ACM TOG (SIGGRAPH) 2018; arXiv:1801.08093. https://arxiv.org/abs/1801.08093 — Eq. 6–7 sum over 20,000 samples, w = 4; hexapod in Fig. 1c/6.
87. Seo, Sferrazza, Geng, Nauman, Yin, Abbeel. *FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control.* arXiv:2505.22642, 2025. https://arxiv.org/abs/2505.22642 — < 3 h on one A100; batch 32,768; GPU buffer; Sec. 3 reward/algorithm gait coupling; 48-hour HumanoidBench baselines.
88. Seo et al. FastTD3 repository (README, `hyperparams.py`, `fast_td3.py`, `fast_td3_utils.py`). https://github.com/younggyoseo/FastTD3 — A100 80 GB defaults; IsaacLab 4096 envs, buffer 1024×10; nn.Tanh actor; float32 obs/next_obs buffers.
89. Raffin. *Getting SAC to Work on a Massive Parallel Simulator: Tuning for Speed (Part II).* Blog, Jul 2025. https://araffin.github.io/post/tune-sac-isaac-sim/ — 1024 envs; 2.5–97.5th-percentile bounds; Optuna CMA-ES; γ 0.983; "slight performance gap".
90. Li, Chen, Hong, Ajay, Agrawal. *Parallel Q-Learning: Scaling Off-policy RL under Massively Parallel Simulation.* ICML 2023, PMLR 202; arXiv:2307.12983. https://arxiv.org/abs/2307.12983 — Sec. 4.4.1 env-count sensitivity; 5 seeds; three processes.
91. Hafner, Pasukonis, Ba, Lillicrap. *Mastering diverse control tasks through world models (DreamerV3).* Nature 640:647–653, 2025; arXiv:2301.04104. https://arxiv.org/abs/2301.04104 — Proprio Control 500K steps; 16 env instances; one A100; fixed-HP Acme PPO.
92. Hansen, Su, Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024; arXiv:2310.16828. https://arxiv.org/abs/2310.16828 — 14M steps; UTD 1 vs 512; Dog instabilities; 33 GPU-days for the 317M multitask model.
93. Bhatt, Palenicek, Belousov, Argus, Amiranashvili, Brox, Peters. *CrossQ.* ICLR 2024; arXiv:1902.05605. https://arxiv.org/abs/1902.05605 — UTD 1; single env; 5M steps.
94. Nauman, Ostaszewski, Jankowski, Miłoś, Cygan. *Bigger, Regularized, Optimistic (BRO).* NeurIPS 2024; arXiv:2405.16158. https://arxiv.org/abs/2405.16158 — within 1M steps; memory limitation.
95. Lee, Hwang, Kim, Kim, Tai et al. *SimBa: Simplicity Bias for Scaling Up Parameters in Deep RL.* ICLR 2025; arXiv:2410.09754. https://arxiv.org/abs/2410.09754 — PPO + SimBa at 1024 envs on Craftax (whole block).
96. Gallici, Fellows, Ellis, Pou et al. *Simplifying Deep Temporal Difference Learning (PQN).* ICLR 2025; arXiv:2407.04811. https://arxiv.org/abs/2407.04811 — App. A.3 deterministic policy for discrete actions.
97. Yi, Choi, Singh, Huang, Truong, Sferrazza, Ma, Duan, Abbeel, Shi, Liu, Kanazawa. *Flow Policy Gradients for Robot Control (FPO++).* arXiv:2602.02481, 2026. https://arxiv.org/abs/2602.02481 — Fig. 8 (2⁸–2¹² envs, 5 seeds); Fig. 9 pronk vs trot at 4096 envs; 19 vs 23 min; App. C.1 baseline.
98. McAllister, Ge, Yi, Kim, Weber, Choi, Feng, Kanazawa. *Flow Matching Policy Gradients (FPO).* ICLR 2026; arXiv:2507.21053. https://arxiv.org/abs/2507.21053
99. Wang, Su, Gupta, Pathak. *Evolutionary Policy Optimization.* arXiv:2503.19037, 2025 (preprint). https://arxiv.org/abs/2503.19037 — Table 1 ANYmal / A1 parkour; ~24 GB; 24,576 envs; no sim-to-real.
100. Xie, Zhang, Yang, Hutter, Xu. *Simple Policy Optimization.* arXiv:2401.16025 v9, 2025. https://arxiv.org/abs/2401.16025 — objective; Table 1 ratio deviation 548.060 vs 0.190; 3-layer results below PPO on Ant/HalfCheetah.
101. Henderson, Islam, Bachman, Pineau, Precup, Meger. *Deep Reinforcement Learning that Matters.* AAAI 2018; arXiv:1709.06560. https://arxiv.org/abs/1709.06560 — Fig. 5 t = −9.0916, p = 0.0016.
102. Colas, Sigaud, Oudeyer. *How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments.* arXiv:1806.08295, 2018. https://arxiv.org/abs/1806.08295 — β = 0.51 at N = 5; bootstrap < 10–20.
103. Colas, Sigaud, Oudeyer. *A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms.* arXiv:1904.06979, 2019. https://arxiv.org/abs/1904.06979 — Sec. 4.5 N vs ε; Welch at N < 5; bootstrap < 50; bimodal inflation.
104. Agarwal, Schwarzer, Castro, Courville, Bellemare. *Deep RL at the Edge of the Statistical Precipice.* NeurIPS 2021; arXiv:2108.13264. https://arxiv.org/abs/2108.13264 — IQM; stratified bootstrap; N = 10 across-task CIs.
105. Patterson, Neumann, White, White. *Empirical Design in Reinforcement Learning.* JMLR 25(318), 2024. https://www.jmlr.org/papers/v25/23-0183.html — Sec. 2 summary item 5; Sec. 3.2 maximisation bias.
106. Eimer, Lindauer, Raileanu. *Hyperparameters in RL and How To Tune Them.* ICML 2023, PMLR 202; arXiv:2306.01324. https://arxiv.org/abs/2306.01324 — unstable top seeds; 3 × 5 tuning / 10 test seeds; 1/12 budget.
107. Bjorck, Gomes, Weinberger. *Is High Variance Unavoidable in RL? A Case Study in Continuous Control.* ICLR 2022; arXiv:2110.11222. https://arxiv.org/abs/2110.11222 — early outlier runs; saturating nonlinearities; penultimate normalisation (DrQ-v2).
108. Jordan, Chandak, Cohen, Zhang, Thomas. *Evaluating the Performance of RL Algorithms.* ICML 2020; arXiv:2006.16958. https://arxiv.org/abs/2006.16958
109. Isaac Lab. *Reproducibility and Determinism.* https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html — identical under fixed hardware/version; runtime writes reorder; LSB divergence.
110. Nagarajan, Warnell, Stone. *Deterministic Implementations for Reproducibility in Deep RL.* arXiv:1809.05676, 2018. https://arxiv.org/abs/1809.05676
111. Mania, Guy, Recht. *Simple random search provides a competitive approach to RL (ARS).* NeurIPS 2018; arXiv:1803.07055. https://arxiv.org/abs/1803.07055 — Sec. 4.5 / Fig. 3, 100 seeds, 70 % (Walker2d 20 %).
112. Petrenko, Allshire, State, Handa, Makoviychuk. *DexPBT.* RSS 2023; arXiv:2305.12127. https://arxiv.org/abs/2305.12127 — Sec. IV one V100 16 GB per agent, 8192 envs; PBT after 2e8 steps; smaller updates late.
113. Shahid, Narang, Petrone, Ferrentino, Handa, Fox, Pavone, Roveda. *Benchmarking Population-Based RL across Robotic Tasks with GPU-Accelerated Simulation.* IEEE CASE 2025; arXiv:2404.03336; code README. https://arxiv.org/abs/2404.03336 ; https://github.com/Asad-Shahid/PBRL — one RTX 4090; 1024 envs per agent; "num_envs must divide num_agents"; off-policy capped at 8.
114. Isaac Lab. *Population Based Training.* https://isaac-sim.github.io/IsaacLab/main/source/features/population_based_training.html — rl_games only; one process per policy; 6+ workers; mutation keys.
115. Cully, Clune, Tarapore, Mouret. *Robots that can adapt like animals.* Nature 521, 2015 (doi:10.1038/nature14422); arXiv:1407.3501. https://arxiv.org/abs/1407.3501 — ~13,000 gaits; 6-D duty-factor map; 40M evaluations; < 2 min adaptation.
116. Batra, Tjanaka, Fontaine, Petrenko, Nikolaidis, Sukhatme. *PPGA.* ICLR 2024; arXiv:2305.13795. https://arxiv.org/abs/2305.13795 — 4× best reward on humanoid.
117. Zahavy et al. *Discovering Policies with DOMiNO.* ICLR 2023; arXiv:2205.13521. https://arxiv.org/abs/2205.13521
118. Margolis, Agrawal. *Walk These Ways.* CoRL 2022, PMLR 205; arXiv:2212.03238. https://arxiv.org/abs/2212.03238 — command-conditioned family (MoB).
119. Iscen, Caluwaerts, Tan, Zhang, Coumans, Sindhwani, Vanhoucke. *Policies Modulating Trajectory Generators (PMTG).* CoRL 2018; arXiv:1910.02812. https://arxiv.org/abs/1910.02812 — Sec. 4.3: 77-parameter linear policy; without PMTG both fall short of optimal reward.
120. Yang, Gao. *Terrain-adaptive Central Pattern Generators with RL for Hexapod Locomotion.* arXiv:2310.07744, 2023. https://arxiv.org/abs/2310.07744 — Sec. IV, Fig. 4 (4 runs): CPG modulation beats foot-position output.
121. Schilling, Konen, Ohl, Korthals. *Decentralized Deep RL for a Distributed and Adaptive Locomotion Controller of a Hexapod Robot.* IROS 2020; arXiv:2005.11164. https://arxiv.org/abs/2005.11164 — 15 seeds; 657.11 (SD 68.07) vs 546.70 (SD 131.23), p = .011; Gazebo/Bullet PhantomX.
122. Seyde, Werner, Schwarting, Gilitschenski, Riedmiller, Rus, Wulfmeier. *Solving Continuous Control via Q-learning (DecQN).* ICLR 2023; arXiv:2210.12566. https://arxiv.org/abs/2210.12566 — 10 seeds; Mini Cheetah qualitative (Fig. 8).
123. Zhou, Vanschoren. *Open-Ended Learning Strategies for Learning Complex Locomotion Skills.* 5th Workshop on Meta-Learning at NeurIPS 2021; arXiv:2206.06796. https://arxiv.org/abs/2206.06796 — Table 1 PPO 36 %, SAC 43 %, ePOET 64 %; different curricula; no seed repeats.
124. NVIDIA Developer Forums thread 268134 (Sep–Nov 2023). https://forums.developer.nvidia.com/t/multiple-isaac-sim-containers-on-one-gpu-fails-with-cuda-illegal-memory-access-in-omni-physx-tensors-plugin/268134 — illegal memory access in `get_jacobians`; unresolved.
125. isaac-sim/IsaacLab issue #4240 (Dec 2025). https://github.com/isaac-sim/IsaacLab/issues/4240 — "only 2 jobs could be ran even I have plenty of VRAM left"; no maintainer answer.
126. NVIDIA Developer Forums thread 331011 (Apr–May 2025, Isaac Sim 4.5). https://forums.developer.nvidia.com/t/trouble-running-multiple-isaac-sim-processes/331011 — 3rd/4th processes hang on shader compilation; Kit "is not designed to run like this".
127. isaac-sim/IsaacLab discussion #1241 / issue #877 (2024). https://github.com/isaac-sim/IsaacLab/discussions/1241 — 80 s → ~500 s per item under multiprocessing; Ray suggested.
128. NVIDIA. *Multi-Process Service* (Release 615). https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf — Sec. 1.1.2.1 when to use; Linux only; fault containment; signal termination undefined.
129. Wang, Feng, Wang, Geng, Li, Ding. *GMI-DRL: Empowering Multi-GPU Deep RL with GPU Spatial Multiplexing / Adaptive-Grained Parallelism.* arXiv:2206.08482 (2022); USENIX ATC 2025. https://arxiv.org/abs/2206.08482 ; https://www.usenix.org/system/files/atc25-wang-yuke.pdf — Isaac Gym PPO utilisation "under 50 %"; MPS/MIG beat direct sharing on A100.
130. NVIDIA IsaacGymEnvs `docs/pbt.md`. https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/pbt.md — `--experiments_per_gpu=4 --num_gpus=1`.
131. Isaac Lab. *Ray Job Dispatch and Tuning.* https://isaac-sim.github.io/IsaacLab/main/source/features/ray.html — GPU split across workers; "[WARNING]: Splitting node into more than one worker".
132. PyTorch. *Model Ensembling* (`torch.func.stack_module_state` + `vmap`). https://docs.pytorch.org/tutorials/intermediate/ensembling.html — 1.33 ms looped vs 645.64 µs vectorised.
133. Cobbe, Hilton, Klimov, Schulman. *Phasic Policy Gradient.* ICML 2021; arXiv:2009.04416. https://arxiv.org/abs/2009.04416 — separate networks; value function "more aggressively optimized with a higher level of sample reuse" (Procgen).
134. van Hasselt, Guez, Hessel, Mnih, Silver. *Learning values across many orders of magnitude (PopArt).* NeurIPS 2016; arXiv:1602.07714. https://arxiv.org/abs/1602.07714
135. Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu. *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO).* NeurIPS 2022 D&B; arXiv:2103.01955. https://arxiv.org/abs/2103.01955 — Sec. 5.1 value normalisation "never hurts".
136. Hessel, Soyer, Espeholt, Czarnecki, Schmitt, van Hasselt. *Multi-task Deep RL with PopArt.* AAAI 2019; arXiv:1809.04474. https://arxiv.org/abs/1809.04474 — updates depend linearly on return scale.
137. Schaul, Ostrovski, Kemaev, Borsa. *Return-based Scaling: Yet Another Normalisation Trick for Deep RL.* arXiv:2105.05347, 2021. https://arxiv.org/abs/2105.05347
138. Isaac Lab `isaaclab/managers/reward_manager.py`. https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/managers/reward_manager.py — `compute()` accumulates weighted dt-scaled terms, no clip.
139. Schwarke, Mittal, Rudin, Hoeller, Hutter. rsl_rl `algorithms/distillation.py` (library paper arXiv:2509.10771). https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/distillation.py — raw `teacher(obs).detach()` target; mse/huber.
140. Kumar, Fu, Pathak, Malik. *RMA: Rapid Motor Adaptation for Legged Robots.* RSS 2021; arXiv:2107.04034. https://arxiv.org/abs/2107.04034 — phase 2 MSE on the latent only.
141. Miki, Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning robust perceptive locomotion for quadrupedal robots in the wild.* Science Robotics 2022; arXiv:2201.08117. https://arxiv.org/abs/2201.08117 — Lbc + 0.5 Lre.
142. Czarnecki, Pascanu, Osindero, Jayakumar, Swirszcz, Jaderberg. *Distilling Policy Distillation.* AISTATS 2019; arXiv:1902.02186. https://arxiv.org/abs/1902.02186
143. Krabby PPO review. *Evidence dossier (final): is PPO, or its hyper-parameters, limiting Krabby?* 2026-09-30. `crab_hex_omni_task/experiments/2026-09-30_rl_algorithm_review/DOSSIER.md` — §0 summary; §1 effective setup; §3 telemetry; §4.1–4.7 campaign evidence; §6 constraints; §10 reading.

## Appendix: unverified or dropped items

- **Robotic World Model (Li, Krause, Hutter; arXiv:2501.10100 v5).** Exists and was opened. DROPPED from
  §10 as cited because the sweep reversed the finding: Table 1 gives RWM + MBPO-PPO a real tracking reward of
  0.90 ± 0.04 against PPO's 0.90 ± 0.03 with 6M vs 250M transitions, but 55 vs 10 min wall-clock (50 min model
  pretraining + 5 min policy); the "still falls short of ... well-tuned model-free RL" quote is from its
  general Limitations section. Read correctly it strengthens the wall-clock case against model-based RL here.
- **Salimans et al., *Evolution Strategies as a Scalable Alternative to RL* (arXiv:1703.03864).** Exists;
  DROPPED from §11 because the sweep's "3–10x TRPO's data" is the Atari-vs-A3C figure; against TRPO on MuJoCo
  the paper says "no more than 10x" and up to 3x better on simple tasks. Its Humanoid gaits "never observed
  with TRPO" still support the claim that different optimisers reach different basins.
- **Smith, Kostrikov, Levine, *A Walk in the Park* (arXiv:2208.07860; RSS 2023).** Exists; DROPPED from §10
  as support for "all parallel off-policy actors bound the action inside the policy": it is a single real
  robot, and its constraint is a narrow joint-target window [p − o, p + o] (o = 0.2/0.4/0.4 rad), Sec. IV-A
  not III-A. Its "the simulated agent cannot make any progress in the unconstrained action space" remains a
  data point for bounding.
- **Cully et al. [115] as evidence that "controlled switching, not interior actions" spans the tripod
  family.** DROPPED (§4): the controller is a Gaussian-smoothed square-wave *position* CPG with interior
  amplitudes, smoothed "to remove sharp transitions"; it says nothing about bang-bang optimality. Cited only
  for MAP-Elites (§11).
- **Extreme-parkour effective clip "±2.4".** Corrected to ±4.8: the default `a1` task overrides
  action_scale to 0.25 (§2, [6]).
- **Isaac Lab entropy "0.005 in every velocity task".** Corrected: 0.005 is ANYmal C/D only; others use
  0.0025–0.01 (§2, [7]).
- **"In every reference stack the clip is non-binding relative to σ".** Corrected: false for rl_games (±1
  clip, std 1, about a third of samples truncated), which instead bounds the mean (§2, [10–12]).
- **DHAL "no seeds, confounded with multi-critic".** Corrected: 4 training seeds, 5 hardware trials per
  scenario, and the ablation is Beta-only inside the multi-critic framework (§3, [19]).
- **Su et al. "hard equivariance has large seed spread (65.7 ± 23.2 vs 54.4 ± 32.6)".** Corrected: those are
  success rates, PPOeqic's spread is the smaller, and Su attribute the one-side advantage to domain
  randomisation breaking symmetric dynamics (§8, [68]).
- **rl_games "ratchets the rate to min_lr" attributed to the legacy controller.** Corrected: the sentence
  concerns per-minibatch stepping on the step-KL signal, one seed; the legacy floor there is 5e-5 and part
  of its drift is observation-normaliser movement, which Krabby lacks (§6, [48, 41]).
- **SAPG "PPO saturation ceiling at ~24k envs".** Corrected: 24,576 is the experiment scale; the paper gives
  no ceiling (§7, [61]).
- **PPO paper "log-std init 0 for MuJoCo".** Corrected: the paper states no MuJoCo std init; Table 4
  Roboschool anneals log-std −0.7 → −1.6; std-1 is CleanRL's convention (§5, [39, 21]).
- **Hsu et al. as support for init std 0.5.** Reclassified as a counter-caveat (narrow init worsens basin
  selection) (§5, [2]).
- **He et al. occupancy 71/30/19 "for the clipped Gaussian".** Corrected: three matched tanh policies with
  the latent interface widened to ±10 (§2, [3]).
- **Geiß "+200" attributed to mean regularisation generally.** Corrected: the flipped-uniform KL objective on
  running, under GAIL with TRPO (§2, [4]).
- **Replay arithmetic "N ~ 1,024 feasible only via proprio storage".** Corrected: 256 × 1,024 × 9.2 KB ≈
  2.4 GB fits at full obs; proprio-only storage instead allows N ~ 10k in ~1.9 GB (§10, [88]).
- **VideoMimic "clip 8.0 non-binding relative to σ".** Reviewer inference (8.0 = 10 σ), not a paper
  statement (§3, [13]).
- **CAPG "adds no restoring force on |μ|".** Inference from the estimator's form; the paper does not discuss
  mean drift (§3, [20]).
- **Hard log-std clamps in SB3 SAC (`policies.py`, `th.clamp(log_std, −20, 2)`), Spinning Up SAC and skrl
  (`clip_log_std=True`, max 2).** Checked by the verification pass but carried without numbered entries; they
  establish that hard clamps are common and loose, not that a binding clamp has precedent (§5).
- **EqR (Mondal, Jain, Siddiqi, Ravanbakhsh, ICML 2022).** Located by search only, not opened; locomotion
  content unverified (§8 gap).
- **Kim et al. 2026 hexapod DRL (Actuators 15(1):33)** and other items not opened by the earlier
  gait-motion-quality review are not re-listed here.
- **Inferences, not source statements:** "lowering σ alone cannot undo Krabby's saturation" (applies He et
  al.'s P10 criterion to the 97 % figure); the (11 − 1.1)² × 18 bound-loss calibration; the α/√c mapping
  2e-4 → ~2e-5; the 69-step half-life and 17-step GAE horizon; the value-clip rate limit (~0.2 per sample per
  iteration) and its role in stage collapses; "tanh alone does not stop drift; the log-det entropy term
  does" as applied to Krabby; the cam channel as a chain of integrators with a PWM code; the Fisher-exact
  seed thresholds; "a bounded head may shrink the lottery itself"; the ordering of rungs 6–9.
- **In-repo numbers not re-measured here:** the clipped rod-mean shares (0.59–0.73 / 0.94–0.96), the
  two-process overlap (2,783 steps/s each), 4.4 GB per process, 61–69 % utilisation, and every census figure
  are quoted from the dossier [143] with its own verification tags; the dossier marks the total_terminates
  co-fire rate, the rollout/update KL mismatch (~3 % of desired_kl) and the X19 saturation trajectory as
  UNRESOLVED or indicative.
- **Unrefereed sources used with that caveat:** [3, 24, 25, 40, 60, 69, 70, 74, 76, 97, 99, 100] (2025–2026
  arXiv preprints), [9, 89] (blog), [41, 48, 49] (maintainer docs), [124–127] (forum/issue threads), [64]
  (withdrawn preprint), [123] (workshop).

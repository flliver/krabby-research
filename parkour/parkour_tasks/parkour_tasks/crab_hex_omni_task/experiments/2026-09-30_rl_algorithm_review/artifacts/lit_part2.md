
## 5. σ parameterisation: init, clamp, entropy (question b)

**Key papers:** Andrychowicz et al. [14]; Schulman et al. 2017 [39]; 37 details [21]; Sabatini, Li, Hutter
2026 [40]; Hsu [2]; rl_games MJLAB.md [41]; VideoMimic [13]; CleanRL SAC [42]; Brax [16]; rsl_rl PRs #66,
#67, #112, #190, #201 [43]; PPO-CMA [44]; REPI [45].

Over ~250k MuJoCo agents, performance was "very sensitive to the initial action standard deviation with 0.5
performing best on all environments except Hopper"; shrinking the last policy layer alone raised Humanoid
return by 66 %; and there was "no evidence that any of the investigated regularizers helps significantly"
(entropy/KL penalties, Lagrange constraints) except on HalfCheetah, attributed to PPO's own trust region plus
careful initialisation [14]. The PPO paper uses no entropy bonus on MuJoCo and states no MuJoCo std init; its
Roboschool humanoid setting anneals log-std from −0.7 to −1.6 (std ≈ 0.5 → 0.2) [39]; the std-1, `ent_coef` 0
convention is CleanRL's [21]. Legged PPO references start at σ 1.0 (VideoMimic 0.8) with entropy 0.0025–0.01;
the ETH seven-robot PPO baseline uses σ0 1.0 and entropy 0.005 (0.008 G1, 0.01 H1) at 8,192 envs × 24 steps,
4 minibatches, adaptive KL 0.01 [40, 13]. In the same paper, for SAC with a tanh head, σ0 = 1 "places
significant mass near the action bounds, resulting in near-uniform exploration" while σ0 = 0.15 "leads to
faster and more stable convergence" [40]. Counter-caveat: Hsu et al.'s failure mode 3 shows a Gaussian init
concentrated near a suboptimal peak pulling the aggregate gradient there, cured by a uniformly initialised
Beta, so a narrower init can worsen basin selection [2]. rl_games' maintainers report that "a positive entropy
bonus on a global `fixed_sigma` can drive a sigma runaway over 1B+ frame runs" (long-horizon manipulation);
their recipes use entropy 0, though global std and entropy 0 were changed together [41].

Hard clamps on σ are common but loose. CleanRL SAC maps log-std through tanh into [−5, 2] (gradient-passing)
[42]; Brax uses softplus + 0.001 [16]; Andrychowicz recommend softplus with a negative input offset, find
softplus and exp "perform very similarly" (exp sometimes NaN unless clipped), the minimum std to "matter
little, if it is not set too large", and per-state std to show "no clear benefits" (C59) [14]. Upstream
rsl_rl 5.2.0 (PR #201) added `std_range` as a hard torch clamp with a non-binding default (1e-6, 1e6) plus
`learn_std=False` for a constant σ ("useful for distillation or finetuning"); PR #67's `log` noise_std_type is
justified only by positivity; PR #66 removed the `0.0*mean + std` construction extreme-parkour still uses
because it leaked gradients into the mean and produced NaNs; PR #190 shows the "std ≥ 0" crash is float32
underflow under both parameterisations; state-dependent σ is available in 5.0.1 as the opt-in
`HeteroscedasticGaussianDistribution` (PR #112) [43]. The verification pass also found hard log-std clamps at
[−20, 2] in SB3 and Spinning Up SAC and in skrl (appendix): what lacks precedent is a clamp tight enough to
bind, 33 % above the init. The opposite failure is documented too: PPO "can prematurely shrink the exploration
variance" [44]; weighted-MLE updates "always shrink variance, causing premature convergence" on toy functions,
fixed by decoupled mean and covariance updates with separate KL bounds [45]; state-independent variance
"avoids the variance to shrink quickly" [2].

**What it means here.** `scalar` vs `log` is not the lever; the lever is removing or softening
`ACTION_STD_MAX` so entropy and surrogate gradients act again (log_std = lo + ½(hi − lo)(tanh(x) + 1) with
hi = ln 2, or softplus + offset), or fixing/annealing σ with an upstream-supported `learn_std=False`. With
actions clipped at ±1 and init σ 1.5, half of the samples are clipped even at μ = 0 (D16), which plausibly
feeds the early basin lottery (formation decided by iteration ~20–200; RSI inert at init σ 1.5). Best-supported
arms: init σ 0.5–1.0 with a small last-layer init [14], entropy 0 or 0.0025–0.005 [14, 13, 41], judged on
formation rate across seeds; target entropy has no support [4, 14]. Once the cap and the 0.01 bonus go, expect
σ to fall, possibly too fast [44, 45], so keep a soft floor near 0.05 and watch for Hsu's narrow-init trap [2].
Rank state-dependent σ last (adds a head, changes the loader contract, does not address the pin).

## 6. Step-size control: the per-minibatch KL rule and what the KL measures (question c)

**Key papers:** Rudin et al. 2021 [46]; rsl_rl ppo.py [47]; extreme-parkour ppo.py [6]; rl_games
CONFIG_PARAMS.md [48]; MJLAB.md [41]; rl_games PRs #381/#383 [49]; rsl_rl issue #221 [50]; Schulman et al.
[39]; Spinning Up [51]; SB3 [52]; 37 details [21]; Andrychowicz [14]; Engstrom et al. 2020 [53]; Hilton, Cobbe,
Schulman 2022 [54]; KL invariance and the data-processing inequality [55]; He et al. [3]; rsl_rl Beta [18].

Krabby's controller is the unmodified Rudin rule (Algorithm 1: α ← max(1e-5, α/1.5) if KL > 2kl*,
min(1e-2, 1.5α) if KL < 0.5kl*, kl* 0.01; Table 3 minibatch 24,576, 5 epochs) [46], identical in rsl_rl
master and extreme-parkour, both with one Adam group [47, 6], and verified in `ppo_with_extractor.py:377–404`.
Rudin et al. say minibatches "much larger than what is usually considered best practice" stabilise learning
[46]; Krabby runs it 64–96× smaller (384 vs 24,576–36,864) and checks 80 times per iteration instead of 20.
rl_games, the other user of the mechanism, documents in code that per-minibatch stepping "needs reliable KL
estimates (large minibatches)" and advises per-mini-epoch stepping when minibatches are small because noisy
estimates "oscillate between the band edges" (guidance, not a measurement) [48]. Its 3-seed WujiHand study
(hand manipulation, observation normalisation, band 5e-5–2e-4) found the legacy pre-step reference KL
"mostly drift accumulated over the pass plus observation-normaliser movement", so the controller braked to
its floor for ~800 iterations; a fixed 1e-4 beat it on two seeds and tied one (19.6/18.9/18.9 vs
18.8/18.2/18.9), and a per-mini-epoch step-KL scheduler recalibrated to 0.002 scored highest
(20.1/19.1/19.5) with a 95 % interval on its gain over fixed spanning −0.1 to +1.0; the "ratchets to min_lr"
failure applies to per-minibatch stepping on the heavy-tailed step-KL signal (one seed) [41, 48, 49]. One
rsl_rl user reports an adapted LR of ~2.25e-5 and the resume LR-reset hazard, since fixed in `load()` [50].

The PPO paper's adaptive KL is a different mechanism (a penalty coefficient halved or doubled once per
update, LR untouched) and lost to clipping (0.74 vs 0.82) [39]. Mainstream practice fixes or linearly anneals
the LR ("0.0003 is a safe default"; linear decay +15 % on Ant, "of secondary importance") with optional
target-KL early stopping that resets each iteration: Spinning Up checks 1.5 × 0.01 per policy gradient step
with separate π and V optimizers (3e-4 / 1e-3); SB3 checks per minibatch, default None; CleanRL per epoch,
default None, `anneal_lr` True [14, 51, 52, 21]. Engstrom et al. show the trust region PPO enforces "is
heavily dependent on the method with which the clipped PPO objective is optimized, rather than on the
objective itself" [53]. Hilton et al. show PPO is not batch-size invariant; below the critical batch size,
Adam compensates for a c-times smaller batch with α/√c, which maps extreme-parkour's 2e-4 at 36,864 to
~2e-5 at 384 (reviewer arithmetic, outside the paper's 1-epoch/EWMA conditions); "by far the most important
hyperparameter to tune is the learning rate"; a clip fraction "much less than 1 %" says grow the iteration
batch, more than 20 % with multiple epochs says the LR is too high [54].

What the KL measures decides which remedies make it behavioural. KL is invariant under an invertible change
of variables and the log tanh' terms cancel (Andrychowicz, appendix bullet C63, footnote 27), so a tanh head's
pre-squash Gaussian KL already equals the executed-action KL: tanh alone removes no "phantom" KL, it can only
change the incentive to saturate [55, 14, 3]. Clip(·, ±1) is many-to-one, so by the data-processing inequality
the KL after clipping is at most the Gaussian KL, consistent with Krabby's 11–16 % behavioural share on
saturated heads [55]. A Beta head with affine rescaling puts the KL on the executed action [18]. A mean bound
makes clip near-identity, and the Gaussian KL approximately behavioural, only when σ is also small relative
to the distance of μ from the bound; at σ 2.0 most samples are still clipped with |μ| ≤ 1. rl_games'
`kl_schedule_source=optimizer_step` (KL(post‖pre) on the same minibatch) measures step size rather than
executed change and needs a recalibrated target (0.002 → 0.003 after take-off) [48].

**What it means here.** The controller explains dossier item 3 mechanistically, and in Krabby the effective
policy movement is set by it, not by clip 0.2, so changing it is a first-order algorithmic change. The rl_games
data also say the expected gain from replacing it is modest, not transformational. Cheapest switches: (i) a
fixed LR 1e-4–3e-4 (linear anneal optional) with per-epoch target-KL early stopping at 0.01–0.02 as a guard;
(ii) per-epoch stepping on a recalibrated step-KL; (iii) a larger iteration batch so the estimate is reliable
(§7). Log KL, clip fraction and executed-action change per update, all absent today. Order matters: on the
current head a fixed LR removes the throttle but the Gaussian KL stays non-behavioural; with a bounded head
*and* small σ the KL becomes behavioural and `desired_kl` 0.01 regains its meaning, which is why the head and
σ arms come first.

## 7. Batch geometry, env scaling, γ/λ and time limits (question d)

**Key papers:** Rudin et al. [46]; Isaac Gym [56]; Isaac Lab benchmarks [57]; Isaac Lab paper [58]; Mayor et
al. 2025 [59]; Beukman et al. 2026 [60]; Andrychowicz [14]; SAPG [61]; Isaac Lab ANYmal-D cfg [7]; Humanoid-Gym
[62]; Lee et al. 2020 [63]; Yang et al. 2020 [64]; Seo et al. 2025 [65]; Pardo et al. 2018 [66].

Rudin et al. varied 128–16,384 robots at fixed batches of 49k/98k/197k over 1,500 updates: 2,048–4,096 robots
at ~100k–200k "provides the best trade-off", "below a certain threshold, we see a slow decrease in performance
with fewer robots" (128 robots means 384-step rollouts), and PPO "struggles when we provide fewer than 25
consecutive steps, corresponding to 0.5 s"; scaling is nearly linear to ~4,000 robots on an RTX A6000, and
4,096 robots need ~9 GB with rendering [46]. Isaac Gym held experience fixed: 256 envs × 512 steps → 4,096 × 32
cut Humanoid time-to-reward "by an order of magnitude from 10⁴ seconds (~2.7 hours) to 10³ seconds", beyond
which short horizons raised training time and gave "sub-optimal gaits", though doubled horizons let
8,192–16,384 envs walk [56]. Isaac Lab reports 82k env-steps/s (step + inference + train) for Rough-G1 at
4,096 envs on one RTX 4090 in 6.1 GB VRAM [57], names single-core CPU speed as a residual bottleneck, and
finds a GeForce 5090 system "achieves higher throughput at smaller environment counts" thanks to its CPU
[58]. At a fixed data budget, scaling N_envs beats scaling rollout length and raises feature rank while
lowering dormant neurons (PPO/PQN; Atari, Isaac Gym, Procgen) [59]. Beukman et al. show PPO plateaus when
"the outer step size is too large relative to the update noise level", that "tuning the learning rate cannot
counteract a poor outer step size", and recommend adding minibatches at fixed minibatch size and LR, raising
minibatch size only when hardware utilisation binds [60], in tension with Rudin's preference for very large
minibatches. Andrychowicz: transitions per iteration "influences the performance quite significantly"; more
parallel envs with shorter chunks can hurt sample complexity; recompute advantages once per data pass [14].
SAPG places PPO's batch saturation qualitatively in the ">>10k envs" regime on manipulation only [61].

γ 0.99 at 50 Hz has a ~69-step (1.4 s) reward half-life and a 100-step horizon; the GAE horizon
1/(1 − 0.99 × 0.95) ≈ 17 steps (0.34 s) sits inside the 0.48 s rollout (reviewer arithmetic). Other stacks:
Isaac Lab ANYmal-D 0.99/0.95 at 24 steps [7]; Humanoid-Gym (100 Hz) 0.994/0.9 with 60-step rollouts, 2 epochs,
entropy 0.001 [62]; Lee et al.'s ANYmal TRPO teacher 0.995 [63]; Yang et al. (withdrawn preprint) set γ so the
half-life equals one foot step, 0.5 s at 25 Hz → ~0.95 [64]; Seo et al. found γ 0.97 "helpful for simple
velocity tracking" with FastSAC/FastTD3 on humanoids [65]. Andrychowicz call γ "one of the most important
hyperparameters", to be tuned per environment from 0.99, with λ 0.9 and no PPO-style value clipping [14].
Time-out bootstrapping matters when each env contributes few steps: Rudin report ~10–20 % higher total reward
and lower critic loss from bootstrapping the critic on time-outs [46]; Pardo et al. show that not
bootstrapping partial episodes causes state aliasing [66]; Andrychowicz saw no effect, "probably" because of
their 1,000-step limits [14].

**What it means here.** Krabby's 6,144-sample batch is 8× below the smallest Rudin tested, its 24-step
rollout sits at their too-short threshold, and its 384-sample minibatch is what makes the KL estimate noisy
(§6). The primary sources prescribe a from-scratch arm at 1,024–4,096 envs × 24 steps (batch 25k–98k,
`num_mini_batches` 4, LR bounds re-checked, telemetry compared) or 256 × 96–192 steps if VRAM forbids; not
shorter rollouts [46, 56]. VRAM at 4,096 envs in references (6–9 GB) fits 16 GB for RL phases, not the
VRAM-bound depth student; throughput must be measured on the 5080 with this 230 kg plant and height scan,
since the +5 % figure is in-repo and stops at 256. Fix the `total_terminates` tag (roll/pitch > 1.5 rad and
low height bootstrapped as time-outs, co-fire rate unmeasured; dossier D9) before any batch/γ arm, else value
targets differ from every reference the arms are compared to. Then test γ 0.995 (half-life ~2.8 s, one gait
period) with λ 0.95–0.97 as a seed-replicated arm; no source ablates γ for slow multi-legged gaits.

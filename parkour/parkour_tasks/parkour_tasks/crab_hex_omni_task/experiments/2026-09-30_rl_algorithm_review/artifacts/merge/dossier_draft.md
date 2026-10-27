# Evidence dossier (draft): is PPO, or its hyper-parameters, limiting Krabby? Experiments and code half

Draft for adversarial verification and for the literature stage. Compiled 2026-09-30, read-only.

It merges ten evidence miners (impl_audit, telemetry, fwd-aug-early, fwd-aug-mid, fwd-aug-late, fwd-sep-side, omni-early, omni-late, prior-reviews/memory/git, infra) and adds spot checks done by the merger.

Path shorthand used below:
- `R` = /home/nickmagus/krabby/krabby-research
- `P` = R/parkour
- `T` = P/parkour_tasks/parkour_tasks
- `S` = /tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review
- "PPO file" = P/scripts/rsl_rl/modules/ppo_with_extractor.py

**Merger spot checks (this pass; CPU only; nothing written to the repo):**
- **PPO file.** Lines 377-404 confirm the KL formula and the per-minibatch rule: /1.5 when KL > 0.02, x1.5 when KL < 0.005, clamped to [1e-5, 1e-2]. Lines 186-213 confirm the estimator substitution in `act()`. Lines 428-456 confirm the loss sum and the mirror loss, whose target is `.detach()`ed.
- **Std clamp.** P/scripts/rsl_rl/modules/crab_actor_critic_with_encoder.py:10-11 sets ACTION_STD_MIN/MAX = 0.05/2.0. Line 50 applies `std.clamp(...)` in the forward pass.
- **Action clip.** P/scripts/rsl_rl/vecenv_wrapper.py:135-136 clamps raw actions to +-clip_actions before `env.step`.
- **Reward clip.** P/parkour_isaaclab/managers/parkour_reward_manager.py:53 applies `torch.clip(self._reward_buf, min=0.)`.
- **RL cfg.** T/crab_hex_forward_task/config/crab_hex/agents/crab_hex_rl_cfg.py:
  - lines 12-20: warning that num_prop 75 does not match the live 1149-dim obs, and that the "scan" slice is not the height scan;
  - line 42: `empirical_normalization = False`;
  - line 47: `resume_alg_state = True`;
  - line 59: `num_prop = 75`.
- **Student cfg.** In the student runner cfg (forward rsl_rl_ppo_cfg.py:177-193), the `algorithm` block says lr 2e-4 and 4 minibatches. The distiller does not use them. It builds `depth_actor_optimizer` at `depth_encoder_cfg["learning_rate"]` = 1e-3 and takes one step per rollout (P/scripts/rsl_rl/modules/distillation_with_extractor.py:50-66).
- **Resume fix.** eabe7324 (2026-09-22) is on feat/crab-hex-omni-task, feat/omni-side-bisect, feat/omni-stroke-posture and claude/stoic-ptolemy-71e304. It is not on main.
- **Installed RL libraries.** `pip list` in isaac_venv shows only rsl-rl-lib 5.0.1.
- **Collection time by env count.** From S/telemetry/runs.csv, median per-iteration times at 64 envs vs 256 envs:

  | experiment | collection, 64 envs | collection, 256 envs | learn time |
  |---|---|---|---|
  | omni | 0.933 s (n=15) | 1.038 s (n=108) | 0.30 s at both |
  | flat | 1.335 s (n=9) | 1.276 s (n=221) | 0.30 s at both |
  | teacher | 1.172 s (n=5) | 1.464 s (n=57) | 0.25-0.30 s |

- **LR floor by configured LR.** From S/telemetry/runs.csv, median per-run share of iterations at the floor:

  | configured LR | runs | median share at floor |
  |---|---|---|
  | 3e-4 | 382 | 0.56 |
  | 1e-4 | 22 | 0.79 |
  | 3e-5 | 29 | 0.997 |
  | 2e-4 | 4 | 1.0 |

  88 of 204 fresh PPO runs reach the std cap, at a median iteration of 2435.
- **ppo_sim design.** S/impl-audit/ppo_sim.py:55-104 is not a closed-loop LR simulation. It runs one update at a FIXED learning rate and counts the controller's votes (ups minus downs). Its advantages are synthetic, and the loss is actor-only: surrogate plus entropy, with no value, mirror or priv_reg terms.

---

## 0. Summary for the recommendation stage

1. **PPO's hyper-parameters were never tested; they were not refuted.** Across 437 PPO runs, 1,091,016 iterations, 6.70B env steps and about 528 GPU-h (462 PPO + 66 student), these were identical in every run: desired_kl, schedule, epochs, minibatches, clip, gamma, lambda, grad norm, steps/env, obs normalisation, std parameterisation and action distribution. entropy_coef differs only in one 20-iteration smoke run. The LR value was set per stage but overwritten within iteration 0. The only sigma experiment (M1) was a fork of a saturated head (Section 5).
2. **PPO as built here runs in a degenerate regime. This is a property of this parameterisation, not of PPO as a family.**
   - The action distribution is a Gaussian clipped at +-1, sampled with sigma 1.5-2.0.
   - The sigma parameter sits behind a forward clamp that passes no gradient once sigma is above 2.0, and a 0.01 entropy bonus pushes it into that clamp. sigma rises monotonically in every from-scratch run and never comes down: 1 of 437 runs ever dipped more than 0.01.
   - The policy mean saturates. On W, 96.6% of mean components have |mu| > 1 (median |mu| 10.5), so the executed controller is sign(mu), i.e. bang-bang.
   - The per-minibatch adaptive-KL controller then spends its budget on mean motion beyond the clip. About 11-16% of the measured KL is behavioural on W, 27-30% on X19 at iteration 2999 and about 50% before saturation.
   - The logged end-of-iteration LR is at the 1e-5 floor in 62.9% of iterations and at or below 3.4e-5 in 95.4%.
3. **The documented outcome failures fall into four classes, and only one points squarely at the optimizer parameterisation.**
   - (a) Reward-preferred degenerate basins (glide, creeper, fall-and-spin). These are reward/task problems: a stronger optimizer would find them too.
   - (b) Stochastic early basin entry. This was fixed by structure (gait clock, RSI from a good bank, mirror loss), not by any optimizer change. It is exploration-related, and an algorithm fix is untested.
   - (c) Slow two-direction omni side walking. This is shared-network interference, fixed at deploy by an exact M_y canonicaliser. Optimizer telemetry is the same in fast and slow runs.
   - (d) Motion quality (chatter, stand stepping, the rigid 3-feet pattern). The ledgers attribute this to the saturated mean, which is the parameterisation above.
4. **Every reward and curriculum verdict in the ledgers was obtained under the throttled regime in item 2.** Several "nulls" may therefore be conditional on it. This is most notable for the fine-tunes, which ran with the LR at the floor for 76-98% of iterations and sigma at 2.0.
5. **The cheapest, best-evidenced next steps are inside PPO.** Change these together and from scratch, with 3 or more seeds:
   - the action distribution or bounds: tanh-squashed or Beta, or a bound loss on mu;
   - the sigma parameterisation: gradient-passing or fixed/annealed, with entropy reduced or targeted;
   - the KL/LR controller: fixed LR, a KL computed on executed actions, or a larger desired_kl;
   - the batch geometry: fewer minibatches, and more envs if throughput allows.

   Swapping the algorithm alone is unlikely to fix classes (a)-(c). Off-policy methods (SAC/TD3 family) remain plausible on sample-efficiency grounds, but they must fit in 16 GB next to Isaac and must port the mirror loss and the actor contract (Section 6).

---

## 1. Effective training setup

"DISAGREE" marks rows where the config files and the real runs disagree.

| Knob | Config says | Real runs (effective) | Source | Flag |
|---|---|---|---|---|
| num_envs (RL) | 6144 | 256 in all 397 non-smoke PPO runs; 64 in 40 smoke runs | cfg: T/crab_hex_forward_task/config/crab_hex/crab_hex_env_cfg.py:430 and T/crab_hex_omni_task/config/crab_hex/crab_hex_omni_env_cfg.py:277. Runs: CLI in T/crab_hex_forward_task/experiments/tools/run_phases.py:52 and T/crab_hex_omni_task/experiments/tools/run_phases.py:71. Census: S/impl-audit/cfg_census.json, S/telemetry/runs.csv | DISAGREE |
| num_envs (student) | 192 (student cfg) | 192 for 3a; 256 for older August students (the symmetric-lineage student OOMed at 256: 14.0-14.1 GB of 15.5) | crab_hex_student_cfg.py:26; T/crab_hex_forward_task/experiments/2026-08-13_0035_mirror_symmetry/RESULTS.md | |
| Rollout | 24 steps/env (0.48 s) | 6,144 samples/iteration, 24x smaller than the Go2 parkour base (147,456) and 16x smaller than Isaac Lab Go2 (98,304) | crab_hex_rl_cfg.py:40; T/extreme_parkour_task/config/go2/parkour_teacher_cfg.py:34 | inherited, not rescaled |
| Epochs x minibatches | 5 x 16 | 80 Adam steps/iteration on 384-sample minibatches. The Go2 parkour base uses minibatches of 9,216 and Isaac Lab Go2 4 x 24,576. One randperm is reused across all epochs (rsl_rl/storage/rollout_storage.py:228) | forward rsl_rl_ppo_cfg.py:51-52, :106-107 | minibatch 24x / 64x smaller |
| learning_rate (configured) | 3e-4 (1a/2a-2c/side/omni); 3e-5 (bridge/2b1); 1e-4 (2b2); 2e-4 (legacy full/teacher base) | Overridden during iteration 0 (next row). The stage-to-stage differences never existed in practice (median share at floor 0.56 / 0.997 / 0.79 / 1.0) | forward rsl_rl_ppo_cfg.py:53, :74-89, :108 | DISAGREE |
| LR (logged, end of iteration) | n/a | At the 1e-5 floor in 62.9% of 1,091,016 iterations; <= 3.4e-5 in 95.4%; iteration-weighted mean 1.69e-5; > 1e-4 in 0.84%. At floor at iteration 0 in 396/437 runs (338/379 among runs of 50+ iterations). The within-iteration value is a sawtooth and is not logged (Section 3.1) | TB Loss/learning_rate; S/telemetry/runs.csv, analysis.txt; S/impl-audit/tb_census.json | |
| Schedule | adaptive, desired_kl 0.01 | same in 437/437. Checked on every minibatch: /1.5 if KL > 0.02, x1.5 if KL < 0.005, clamped to [1e-5, 1e-2] | PPO file :377-404 | |
| KL definition | Diagonal Gaussian KL(old \|\| current) on the UNCLIPPED mu and sigma, summed over 18 dims, minibatch mean. Measured before each step, so it is the cumulative drift since the rollout | n/a | PPO file :379-386 | 18 dims makes it 1.5x stricter per dim than a 12-dim Go2 |
| clip_param | 0.2 (log-ratio clamped at +-20) | same | PPO file :406-413 | |
| Value loss | clipped at +-0.2 around the stored V; coefficient 1.0 | Returns are unnormalised (V on W states 2.85 +- 0.45) | PPO file :415-423; S/impl-audit/misc_probe.py | |
| entropy_coef | 0.01 | 0.01 in 436/437 runs; 0.005 in one 20-iteration smoke | cfg; runs.csv | |
| gamma / lambda | 0.99 / 0.95 | Horizon 100 steps = 2.0 s. GAE horizon about 16.8 steps = 0.34 s. Gait cycle about 1.2-2.8 s. The rollout covers 0.2-0.4 of a cycle | cfg; T/crab_hex_forward_task/mdp/crab_hex_dimensions.py:195 | |
| max_grad_norm | 1.0, a single global norm over actor + critic + std | same; the estimator and history encoder are clipped separately | PPO file :372, :478, :596-598 | |
| Optimizer | Adam (0.9, 0.999, eps 1e-8), one param group | The critic runs at the actor-KL-driven LR | PPO file :113 | |
| init_noise_std | 1.5 (1a/2a-2c/side/omni); 0.65 (legacy teacher, student) | 0.65 never took effect: resumed stages load sigma of about 2.0. 200/379 runs start at the clamp | forward rsl_rl_ppo_cfg.py:31, :75, :118; TB Policy/mean_noise_std | DISAGREE |
| Std parameterisation | scalar per-dim `nn.Parameter` with a forward clamp to [0.05, 2.0]. Upstream rsl_rl and Isaac Lab have no clamp | sigma >= 1.99 in 70.8% of iterations; the raw parameter on W is 2.00009-2.00042 on all 18 dims | crab_actor_critic_with_encoder.py:1, :10-11, :50 | krabby-specific |
| clip_actions | 1.0. The student cfg was None until 2026-09-08. The flat-walk cfg was 2.0 until afbc0a46 (2026-05-19, no rationale given) | 1.0 in all PPO runs. Storage keeps the unclipped sample and its log-prob | vecenv_wrapper.py:135-136; PPO file :199-208 | |
| Action path | Cams: x pi rad/s velocity target, acceleration limit 2 pi rad/s^2. Rods: x0.25 rad offset converted to rod length, slew 0.028 m/s (hip) / 0.033 m/s (knee). 1-step delay | same | params/env.yaml; parkour_actions.py:315-357; crab_hex_dimensions.py:162-196 | |
| Obs normalisation | `empirical_normalization = False` | False in 450/450. Obs column 206 is a constant 158.76 (body mass) | crab_hex_rl_cfg.py:42; S/impl-audit/misc_probe.py | |
| Advantage normalisation | whole batch, once per iteration | same | PPO file :20-27, :272-273 | |
| Reward | Step reward x dt. The total is clipped at >= 0 per step (the legged_gym "only positive rewards" convention) | 13.6% of return forgiven on W, about 16% late in X08:c6; pre-clip -27.0 vs post-clip +6.4 at the start of t3 | parkour_reward_manager.py:53; T/crab_hex_omni_task/experiments/2026-09-23_reward_simplification/CANDIDATES_DETAIL.md:259-290 | never varied |
| Control rate / episode | sim dt 0.005 x decimation 4 = 50 Hz. Episodes 20 s (legacy forward); 40 s (A15+B, side, omni); 10 s in some arms | The hardware loop target is 100 Hz (R/docs/RUNTIME_ARCHITECTURE.md:256-260). Possible mismatch, unconfirmed | params/env.yaml | flag |
| Time-out bootstrap | reward += gamma * V(s_t) on truncation | 'total_terminates' is tagged time_out=True but also contains roll/pitch > 1.5 rad and height < -0.25 m (usually preceded by crab_failure). Co-fire rate unmeasured | PPO file :238-242; T/crab_hex_omni_task/mdp/crab_hex_omni_rewards.py:624-647, :750-752; parkour_mdp_cfg.py:1367-1369 | latent bug |
| Symmetry | Mirror LOSS only (`use_data_augmentation = False`), coefficient 0.5. MSE on raw `act_inference` means with a detached target. Images: forward L/R 2, side fore-aft 2, omni 4 (2 with `fa`) | on in 378/437 runs (off in bridge/2b1/2b2/full and in era-A before 2026-08-13) | forward rsl_rl_ppo_cfg.py:122-145; omni rsl_rl_ppo_cfg.py:76-89; PPO file :432-456 | lit-review-hexapod-reward-stability.md:124-126 says "data augmentation plus mirror loss"; the code wins |
| priv_reg | schedule [0, 0.1, 2000, 3000] in counter units (counter = its + its/20). The coefficient is 0 until about iteration 1905 and 0.1 from about 4762 | Reset at every resume before eabe7324. Run-end coefficient median 0.038. A 3000-iteration run ends at 0.0383 | PPO file :529-541 | |
| DAgger / history encoder | `dagger_update_freq` 20; a separate Adam at the initial LR (3e-4), never adapted | The latent is effectively constant (\|latent\| 0.011 on W; ablation KL 9e-5) | PPO file :122-124, :566-606 | vestigial |
| Estimator | 75->128->64->9 ELU, Adam 1e-4 fixed, MSE on obs[207:216]; substituted into the rollout and at deploy | 8 of 9 targets are constants; friction R^2 about 0.2 | PPO file :118-121, :186-201 | vestigial |
| Network slicing | num_prop 75, num_scan 132, priv_explicit 9, priv_latent 41, hist 10 (= 1007) | The live obs is 1149 = 89 proprio + 108 heights + 9 + 53 + 10x89. The "scan" slice is not the height scan, and base velocity, last actions and contacts enter via the tanh scan encoder | crab_hex_rl_cfg.py:12-20, :59-60; run_meta obs_dim_warning | DISAGREE |
| Networks | Actor 136->512->256->128->18 ELU. Critic 1149->512->256->128->1 | about 0.27M actor and 0.75M critic parameters | forward rsl_rl_ppo_cfg.py:30-43 | |
| max_iterations | 5000 per phase window; 3000 in omni screens; 4000 for c6x; 20000 as the flat-walk default | chained by resume | params/agent.yaml; crab_hex_phases.py | |
| Student 3a | The `algorithm` block lists lr 2e-4, 4 minibatches, 5 epochs | Actual: depth_actor + depth_encoder under Adam at a fixed 1e-3, ONE step per rollout of 23,040 samples (192 envs x 120 steps). Loss = mean L2 norm of (raw teacher mean minus student) plus a yaw term. Grad clipping covers depth_actor only | forward rsl_rl_ppo_cfg.py:177-193; distillation_with_extractor.py:50-66; runner :377-509 | DISAGREE |
| resume_alg_state | True since eabe7324 (2026-09-22) | The forward and sideways heads of record predate the fix. After the fix, only 15/33 resumed runs start at priv_reg 0.1 (unexplained) | crab_hex_rl_cfg.py:47; runs.csv | |
| Seeds | n/a | era-A: seed 1 in all 86 runs. flat_walk: seed 1 (51), 2 (95), 3 (83), 4 (1). omni: mostly 3, then 2 | params/agent.yaml | |
| Throughput | n/a | Median 4,110 env-steps/s (p25 3,650, p75 4,640). About 1.23 s collection + 0.30 s learning per iteration. Old plant (era-A) about 5.8k | TB Perf/total_fps, Perf/collection time, Perf/learning_time | |

---

## 2. Deviations from reference PPO, ranked by severity

Mostly from impl_audit, with additions from the other miners. The reference points are rsl_rl, legged_gym, Isaac Lab and Schulman et al.

| # | Severity | Deviation | Evidence | Likely effect | Cheap test |
|---|---|---|---|---|---|
| D1 | critical | **Clipped Gaussian.** The Gaussian is sampled unclipped and its unclipped log-prob and KL are optimised, while the env executes clip(a, +-1). With sigma >= 1 and an entropy bonus, nothing pulls mu back inside the bound. Reference legged_gym effectively does not clip (clip_actions 100); Isaac Lab Go2 uses None. | vecenv_wrapper.py:135-136; PPO file :199-208, :344. X19 share of \|mu\| > 1 by iteration: 0 at 0, 8.5% at 100, 52% at 300, 73.5% at 500, 87% at 1000, 93% at 2999. W: 96.6%, median \|mu\| 10.5, p90 20.6. P(sample inside +-1) = 3.9%. The censored/Gaussian Fisher ratio for mu is < 0.05 on 74% of dims. Reconstruction check: corr 0.9967, sign agreement 0.983 (S/impl-audit/sat_traj.py, recon.py, mismatch_kl.py). | Bang-bang control with no credit assignment on saturated dims; chatter (2.23 sign flips per dim per s at stand); the first gait found gets locked in; produces the "phantom" KL of D3. | Squashed tanh with the log-det term, a Beta policy, or an rl_games-style bound loss (coefficient 1e-4 to 1e-3, soft bound 1.1). Track the share of \|mu\| > 1 and the sign-flip rate. |
| D2 | critical | **Absorbing std clamp.** std.clamp(0.05, 2.0) in the forward pass gives zero gradient above 2.0, and the entropy bonus (gradient coef/sigma per dim) is the dominant net force on sigma. | crab_actor_critic_with_encoder.py:10-11, :50. sigma >= 1.99 in 70.8% of iterations; 303/437 runs end there; 1/437 runs ever dipped more than 0.01. X19 sigma 1.50 -> 2.0002 over 3000 iterations. M1 re-pinned at its 0.5 cap. | Exploration can never shrink downstream. Feeds D1 (mu must exceed sigma to act reliably). KL scales as 1/sigma^2. | A soft-bounded log-std, a straight-through clamp, entropy 0 or a target entropy. |
| D3 | major | **KL controller on phantom motion.** The adaptive-KL LR rule runs per minibatch on 384-sample minibatches and on the unclipped distribution, so it mostly measures gradient noise and mean motion beyond the clip. | PPO file :377-404. Census numbers in Section 3.1. The fixed-LR CPU re-enactment on W (S/impl-audit/ppo_sim.json) gives end-of-iteration KL (noise / signal advantages) and net controller votes:<br>- 1e-5: KL 0.010 / 0.013, votes +5 / +11;<br>- 3e-5: KL 0.018 / 0.017, votes -13 / -2;<br>- 1e-4: KL 0.051 / 0.074, votes -72 / -78.<br>Censored (behavioural) share of KL: 0.11-0.16 on W; 0.27-0.30 on X19 at iteration 2999; 0.49-0.52 at iteration 300. | Behavioural change per iteration is about 0.001-0.002 nats, 5-10x below what desired_kl intends. The configured LR is never tested. The critic is throttled too (D5). | Fixed LR 1e-4 (no KL check); 4 minibatches; KL on the executed (censored or tanh) distribution; desired_kl 0.02-0.03. |
| D4 | major | **Batch not rescaled.** The batch shrank 24x (6144 -> 256 envs) while num_mini_batches stayed at 16. | forward rsl_rl_ppo_cfg.py:51-52 vs T/extreme_parkour_task/config/go2/parkour_teacher_cfg.py:34; Isaac Lab Go2 rsl_rl_ppo_cfg.py:25-38. | Noisy steps and a lower LR equilibrium. In the simulation, 4 minibatches move the equilibrium LR about 3-10x higher, but the executed-action change per iteration rises only about 1.2-1.6x (0.0026 at 16 mb / 3e-5 vs 0.0032-0.0042 at 4 mb / 1e-4). | num_mini_batches 4, or more envs (Section 6), or longer rollouts. |
| D5 | major | **Critic shares the actor's LR.** Actor, critic and std are in one Adam group, so the critic learns at the LR set by the actor's phantom KL. | PPO file :113, :403-404. Loss/value_function run median 0.0078. No explained-variance telemetry exists. | Unmeasured; possibly stale advantages. The telemetry miner reads the value loss as healthy: RMSE about 3-4% of the value scale (Section 3.4). | Log explained variance; a separate critic optimizer. |
| D6 | major (for porting), minor (for behaviour) | **RMA machinery vestigial.** The obs layout mismatch (num_prop 75 vs 89) leaves the estimator, DAgger history encoder and priv_reg carrying no information. | Latent ablation on W: mean \|dmu\| 0.005, KL 9e-5, sign agreement 99.99%. Estimator: 8 of 9 outputs constant; friction R^2 about 0.2. Setting obs[75:89] to its mean changes mu by 1.41 mean-abs (S/impl-audit/misc_probe.py). | The RMA and teacher-student framing does not hold as built. An algorithm swap need not port this machinery, but the deploy path substitutes the estimator. | A unit test asserting the obs layout matches the slice sizes. |
| D6b | medium (untested) | **Reward clipped at >= 0.** The per-step total reward is clipped at zero (inherited legged_gym/extreme-parkour convention; not a PPO deviation as such). | parkour_reward_manager.py:53. Pre-clip vs post-clip return: 43.7 vs 52.0. Blamed for muting feet_slide inside the drag basin (T/crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/RESULTS.md:242-251). Teacher stages log mean reward of about 0.00-0.08. | Biased value targets and missing penalty gradient in bad states. Would carry over to any off-policy method. | A no-clip or offset-reward arm. |
| D7 | minor | **Behaviour and target policies differ.** act() uses obs with the estimator substitution (and hist_encoding on iteration % 20 == 0). update() recomputes on the true obs with hist=False. | PPO file :186-213 vs :343-350. KL(act vs update) on W: mean 3.2e-4, p99 2.7e-3. Ratio outside [0.8, 1.2] before any update: 0.04% of samples. | About 3% of desired_kl today; not the cause of the LR floor. Becomes first-order if D6 is fixed. | Store obs_est and the hist flag in the transition. |
| D8 | minor | **Mirror loss on raw means.** MSE on raw unclipped means (\|mu\| about 10) with a detached target, so mirrored images train the phantom-direction rows through shared weights. | PPO file :432-456. Loss/symmetry median 0.026 (x0.5 = 0.013) vs \|surrogate\| median 0.008. T/crab_hex_omni_task/experiments/2026-09-28_side_bisect/ANALYSIS.md:124-125. | Spends gradient and KL budget on differences that do not change behaviour. Implicated in the phantom -y rows that make +y-only runs slow (Section 4.4). Not included in the CPU simulation, so the real LR equilibrium is at or below the simulated one. | Compute the loss on clip(mu) or tanh(mu). |
| D9 | minor | **Failure cutoffs bootstrapped as time-outs.** Failure cutoffs tagged time_out get bootstrapped with gamma * V. | Rewards and MDP files cited in Section 1. | Rare; size unmeasured. It matters more for off-policy methods. | Split out a non-time-out termination term. |
| D10 | historical | **Resume reset.** Before eabe7324, every resume restarted the counter (priv_reg ramp), the adaptive LR and the auxiliary Adam states. | 200/200 pre-fix resumed runs start with priv_reg_coef 0. Resumed processes account for 56% of iterations (513k of 918k), at a mean coefficient of 0.022 vs 0.097 in continuous runs (memory ppo-counter-reset-on-resume.md). | Confounds pre-2026-09-22 chain comparisons. The memory note calling the fix "unmerged" is stale. | Already fixed. |
| D11 | minor | **Auxiliary optimizers at fixed LRs.** The history-encoder Adam runs at 3e-4 and the estimator at 1e-4. | PPO file :118-124, :566-606. | None measurable. | n/a |
| D12 | minor | **One minibatch partition for all epochs.** | rsl_rl/storage/rollout_storage.py:228-255 | Inherited from upstream. | Reshuffle every epoch. |
| D13 | minor | **Large unnormalised constant input.** The body mass (158.76) enters the critic and scan-encoder inputs unnormalised. | Critic first-layer \|w * 158.76\| is 2.36 of a 2.76 mean pre-activation. | Conditioning; a large input shift if mass randomisation is ever enabled. | Turn on empirical_normalization. |
| D14 | minor, but a hazard for experiments | **Unknown algorithm cfg keys are silently dropped** (a warning only). | P/scripts/rsl_rl/modules/on_policy_runner_with_extractor.py:131-141 | A new knob (for example bound_loss_coef) silently does nothing unless `PPOWithExtractor.__init__` also takes it. | Make unknown keys raise. |
| D15 | minor | **Distillation targets are raw means.** The student regresses raw saturated teacher means with an L2-norm loss. | distillation_with_extractor.py:56-66 | Student capacity is spent on differences that do not change behaviour. Student ≈ teacher anyway (Section 4.7). | Distil on clipped actions. |
| D16 | minor | **init_noise_std 1.5 exceeds the clip.** | forward rsl_rl_ppo_cfg.py:75, :118 | Half of the samples are clipped even at mu = 0; pushes early training toward D1. | init 0.5-1.0. |
| D17 | cosmetic | **Validation setting is a no-op.** `Normal.set_default_validate_args = False` replaces the method instead of calling it, so validation stays on. | actor_critic_with_encoder.py:180 | A NaN crashes the run. | n/a |

---

## 3. Optimizer-level pathologies measured in telemetry

### 3.1 Learning rate / KL controller

**Census.** Source: S/telemetry/runs.csv and analysis.txt; S/impl-audit/tb_census.json; TB `Loss/learning_rate` in all 437 PPO runs under P/logs/rsl_rl/*/.
- Logged LR at the 1e-5 floor in 62.9% of iterations; at or below 3.4e-5 in 95.4%; above 1e-4 in 0.84%.
- Iteration-weighted mean 1.69e-5, about 18x below the nominal 3e-4.
- Excursions off the floor: 205,937. Length median 1 iteration, p99 9, max 42.
- 16 runs never leave the floor after iteration 0: the Aug-26 bridge runs, the Aug-5/6 teacher runs, and M1.

**By regime.**
- From-scratch 3000-iteration omni/side arms: median 2.25e-5, 31-41% at the floor, 3-4% of iterations >= 1e-4, maximum 3.8e-4 to 5.8e-4 (omni-late tb_arms.json; omni-early X08:t3).
- Fine-tunes from W (c6y B/S/P/S_cat): median 1.0e-5, 76-78% at the floor.
- Teacher stages: 95% at the floor overall.
- Era-A short fine-tunes: 94-98% at the floor.
- Late lineage windows: 94.2% (2026-08-31_10-46-19, iterations 15k-20k) and 98.9% (2026-09-02_00-42-53, iterations 25k-30k).

**The controller tightens as training matures.**
- Share of iterations off the floor, first to last tenth of a run: 0.446 -> 0.308.
- Across omni chain stages: c4 about 0.59, c5 about 0.54, c6 about 0.45, c6x about 0.37, c6y about 0.23.
- Off-floor share is 57% of iterations while sigma is below the cap and 29% at the cap.
- The forward A15+B lineage floor share rises by window: w1 0.55, w2 0.70, w3 0.78, w4 0.90, w5 0.97.

**The logging hides the real step size.**
- `Loss/learning_rate` is the value after the last of 80 minibatch updates (runner :543).
- Consecutive-iteration jumps of x38.4, x57.7 and x86.5 (1.5^9 to 1.5^11) show the LR climbing from 1e-5 to 3.8e-4-8.6e-4 within one iteration.
- The KL is cumulative drift since the rollout. Early minibatches in each iteration therefore see KL near 0 and vote up; late ones vote down. The end-of-iteration value is biased toward the bottom of the sawtooth.
- Mean within-iteration LR, KL, clip fraction and explained variance are not logged anywhere.
- The CPU re-enactment puts the fixed-LR balance point at about 1.5-2.5e-5 for W (16 mb). For X19 at iteration 300 the balance point depends on the advantage structure: above 1e-4 with noise advantages, between 1e-5 and 3e-5 with synthetic signal advantages.
- Conclusion: "effective LR about 1e-5-3e-5" is well supported for saturated or late heads and only approximate (within a factor of about 2-5) for early formation.

**Mechanism** (impl_audit, CPU, W's own Adam state; medium confidence):
- At the configured 3e-4, per-minibatch KL is 0.2-0.5 with 97-99% of minibatches above 0.02. Nine divisions reach the floor within iteration 0.
- At 1e-5, noise advantages give about the same KL as informative ones: 0.0101 vs 0.0128.
- Only 11-16% of the measured KL on W is behavioural (censored at +-1).
- Executed-action change per iteration is about 0.002.
- With sigma pinned at 2.0, KL = sum(dmu^2)/8, so a KL of 0.02 allows 0.094 RMS change per dim on means of about 10.
- Ruled out as causes: the act/update mismatch (KL 8e-5 to 4e-4), priv_reg (acts on a dead latent), nan_to_num.
- Not simulated: the mirror loss (a deterministic pull, which would lower the equilibrium) and the value-loss share of the global grad-norm clip.

**Reframing for the literature stage.** The binding throttle is a per-iteration KL budget of about 0.01-0.02 nats measured on an unclipped Gaussian whose mean is saturated. Changing the minibatch count mostly moves where the LR number sits: per-iteration behavioural change rose only about 1.2-1.6x in the simulation. Fixes that make the KL behavioural (a bounded distribution, a KL on executed actions) or a fixed LR address the throttle directly.

### 3.2 sigma pinning

Sources: telemetry section (2); impl_audit census; TB `Policy/mean_noise_std` and `Loss/entropy`.
- 303/437 runs end at sigma >= 1.99, and 70.8% of all iterations are at sigma >= 1.99. impl_audit's stricter "above clamp" count is 67.7% of iterations, with 289/379 runs ending there.
- Fresh-run sigma (p50) by iteration: 1.56 at 100, 1.65 at 500, 1.76 at 1000, 1.95 at 2000, 2.00 at 3000.
- 88/204 fresh runs reach the cap, at a median first iteration of 2435 (p10 1792, p90 3103; impl_audit: median 2750, 79 runs, stricter threshold).
- Within-run drawdown p90 is 0.0013. Only 1/437 runs ever dipped more than 0.01 below its start.
- Entropy goes from 32.84 to 38.02, which equals 18 x (0.5 ln(2 pi e) + ln sigma) at sigma 1.5 -> 2.0, so the entropy term sees the clamped sigma.
- `Policy/mean_noise_std` logs the raw parameter (runner :544-551).
- Correlation (fwd-aug-late F2, n=11): runs that failed to form reached sigma 1.99 earlier (iterations 1720-2669) than runs that formed (2986-5116). This is correlational only.

### 3.3 Mean saturation

- W (omni head of record): 96.6% of mean components have \|mu\| > 1; median 10.5; p90 20.6.
- ANALYSIS_M1: X08:c6 0.960 (median 8.1), X01 0.964 (9.1), M1 0.965 (9.5); every one of 18 dims is 94-98% saturated (T/crab_hex_omni_task/experiments/2026-09-22_gait_motion_quality/ANALYSIS_M1_saturation.md:31-61).
- Side-bisect heads: 91-98% (median \|mu\| 6.1-7.4) in fast and slow heads alike (side_bisect/ANALYSIS.md:96).
- X19 trajectory: majority-saturated by iteration 300 (52%), 93% at 2999. P(sampled sign differs from sign mu) < 1% for 57% of (state, dim) pairs at iteration 2999.
- The infra miner could not re-verify this (no .pt reads). impl_audit did verify it, by CPU reconstruction validated against recorded pre-clip actions. impl_audit's figures stand.

### 3.4 Value / critic

- `Loss/value_function`, median of the last 100 iterations: flat 0.012 (p10 0.0041, p90 0.0235); omni 0.0058.
- RMSE relative to r_step/(1-gamma): flat 0.038, omni 0.029, teacher 0.154.
- Value loss rises about 2.4x in fresh runs as returns grow. The maximum in any run is 0.463.
- No non-finite scalars in any events file, and 0 "non-finite before sanitize" warnings in 546 stdout logs.
- Teacher stages under net-negative economics log reward of about 0 and value loss <= 0.0003. With the reward clip at 0, the critic gets almost no signal.
- Explained variance is not logged, so critic quality is unknown beyond these magnitudes.

### 3.5 Throughput

- Median 4,110 env-steps/s at 256 envs. About 80% of wall time is collection (1.23 s); the update takes about 0.30 s.
- Collection time is almost the same at 64 and 256 envs (Section 0 spot check), and learn time is 0.30 s at both. Per-iteration wall time is therefore dominated by per-step and per-minibatch latency, not by env count, up to 256 envs.
- 64-env runs achieve about 1,000-1,250 fps vs 3,900-4,600 at 256.
- The memory note (two parallel 256-env runs about 2,700 fps each, aggregate about 5,400) shows that two processes contend. It does not show that a single process at 512-4096 envs would not scale.
- No single-run probe above 256 envs exists anywhere.
- Era-A (old 104 kg plant) ran at about 5.8k fps; the 2026-08-20 rebuild (230 kg) runs slower.

### 3.6 Late collapse and anomalies

**Late collapse** (n=1): flat_walk 2026-08-04_11-55-56, no mirror loss, 13,159 iterations.
- LR 0% off the floor from iteration 7k; sigma pinned from 4596.
- Median surrogate: +0.056 over iterations 9-11k and +0.171 over 11-13k.
- Reward 45.4 -> 19.7; error_vel_xy 0.155 -> 0.70; terrain level flat.

**Degenerate fall-and-spin run** (fwd-aug-mid F9): screen_r4_synthesis 2026-08-14_15-24-00.
- Loss/symmetry grows to 26,081 (unbounded mu divergence).
- LR > 1e-4 in 37% of iterations.

**Sanity checks:**
- The surrogate exceeds 0.05 in only 0.31% of all iterations.
- The only ignored cfg key is `share_cnn_encoders`.

### 3.7 Telemetry gaps
None of the following are logged:
- approx-KL
- clip fraction
- mean within-iteration LR
- explained variance
- per-dim mu saturation
- executed-vs-raw action statistics

---

## 4. Campaign evidence by theme

Tags:
- **[ALG/HP]**: implicates the algorithm or its hyper-parameters.
- **[REWARD/TASK]**: implicates reward or task design.
- **[STRUCTURE]**: implicates representation, priors or architecture.
- **[CONFOUNDED]**: the evidence cannot separate these.
- **[DESIGN]**: a constraint on how any comparison must be run.

### 4.1 Seed variance vs effect sizes [DESIGN; secondarily ALG/HP-or-STRUCTURE, because basin lottery is an exploration symptom]

**Basin rates.**
- Era-A: of 7 from-scratch runs on the baked reward family, 1 reached the alternating gait (H: tripod 0.335 at 3k, 0.401 at 20k), 1 was partial (v4 0.255) and 5 scored 0.0 (T/crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/RESULTS.md:369-394).
- Seed search, seeds 2-8 (3k): 0/7 alternating, 4 gait families, duty_A 0.126-0.633, reward 20.6-31.7 (2026-08-12_1550_seed_basin_search/RESULTS.md).
- Plain clock: 1/4 seeds formed (s1 0.541/0.99; s2 0.0/0.12; s3 0.026/0.25; s4 0.0/0.29) (2026-08-22_1200_gait_formation_v2/CHANGELOG.md:21-29).
- Spin lottery: 0/7 walking spinners.

**Same schedule, different seeds.**
- gated_lineage: tripod 0.436 vs 0.606 (2026-08-26_2200_gated_lineage/REPORT.md:497-525).
- morph P2 at 5k: seed 3 vs seed 2 obstacle completion 0.67 vs 0.29 (B) and 0.68 vs 0.33 (A15+B). The ranking of B vs A15+B flips between seeds at 10k (2026-09-04_1105_morph_x_exposure/REPORT.md:268-320).
- Sideways 1a completion 0.93 vs 0.79 (T/crab_hex_sideways_task/experiments/2026-09-11_0414_sideways_formation/REPORT.md).

**Omni seed-to-eval noise ratios** (R_seed/R_eval):

| Decision block | F | F_mt | T | Other |
|---|---|---|---|---|
| facing ladder D00b | 3.19/0.97 | 4.12/0.94 | 1.13/0.54 | |
| MQ D00bs | 2.07/0.96 | 6.05/0.91 | | Q 0.319/0.252 |
| stroke D00bs | 1.81/0.97 | 2.45/0.91 | 1.67/0.49 | |

Sources: T/crab_hex_omni_task/experiments/2026-09-19_facing_tracking_ladder/CHANGELOG.md:47; 2026-09-22_gait_motion_quality/REPORT.md; 2026-09-26_stroke_posture/CHANGELOG.md:13.

**Omni seed pairs.**
- B_f t3 (identical recipe), s3 vs s2: fwd/back survival 0.49/0.51 vs 0.91/0.97 (2026-09-26_stroke_formation/GATE_t3.md:22-49).
- C3a pooled sign: s3 +1.196 vs s2 -3.531. The pooled decision flipped.
- S_cat F_mt: z -4.82 (s3) vs +2.85 (s2).
- The M1 effect on turn facing (0.610 -> 0.510) equals the X0 seed gap (0.610 vs 0.508).

**Eval noise.**
- 100-episode canary: +-0.10.
- Harness resolution: +-0.04 completion.
- An env write after Kit start-up flips one head between 0.79 and 0.75.
- Single-seed MDE: turn facing about 0.145, move_turn about 0.18.

**Reproducibility is inconsistent.**
- An exact reproduction exists: A15+B lineage w0 matches morph P2 stage 1, seed 3 (2026-09-06_2130_a15b_lineage/CHANGELOG.md:25; TB floor share 0.423 and first sigma at cap at iteration 2629 in both).
- An identical-config pair diverged after about 20 iterations: X09:t3 vs X08:t3, fwd survival 0.73 vs 0.90 (2026-09-19_facing_tracking_ladder/PLAN_AMENDMENTS.md:316-323).
- Two forward pairs that differ only in max_iterations diverged by iteration 20-500:
  - C2 3k screen vs C1+C2 20k: reward at 3k 27.1 vs 13.4; tripod 0.573 vs 0.0 (2026-08-15_1530_task1_velocity/CHANGELOG.md:57-61).
  - screen_w0.3 vs fromscratch_w0.3_20k: 27.1 vs 17.1.
- The cause is not identified.

**Large effects that clear the noise:**
- side speed class: 0.16-0.22 vs 0.06-0.10 m/s (seed sd about 0.015);
- clock+RSI formation;
- golden vs wide plants on obstacles;
- clip_actions None vs 1.0 (100/100 falls vs 0.70);
- mirror loss 0.401 -> 0.517 (n=1).

**Reading.** Most PPO-relevant verdicts were single-seed. Any algorithm or hyper-parameter screen needs 3 or more seeds per arm, and should score the formation rate and across-seed spread, not only means.

### 4.2 Plasticity, fine-tuning and training length [CONFOUNDED]

**Fine-tune nulls.**
- Era-A: 27 fine-tunes stayed at tripod 0.383-0.425. With the same config, a fine-tune from v3 scored 0.0 while a from-scratch run scored 0.401 (2026-08-09_0920_short_runs/RESULTS.md:155; 2026-08-09_1526_gait_tuned/RESULTS.md:22-41).
- Lean/stride fine-tunes moved pitch by 0.000 at up to 5x dose (2026-08-12_1017_lean_reduction/RESULTS.md).
- Omni S, P and S_cat fine-tunes of W moved no stride beyond the seed band (2026-09-26_stroke_posture/CLOSE.md:40-53; 2026-09-27_stroke_cat_W/CLOSE.md:17-42).

**Optimizer state during those fine-tunes.**
- LR at the floor in 94-98% of iterations (era-A short runs), about 50% (tripod fine-tunes, max 5.1e-5), and 76-78% (W forks).
- sigma at 2.0 for 100% of iterations.
- priv_reg at 0 before the fix.
- No fine-tune control with a working LR or a reset sigma was ever run. The ledgers' "basin lock-in" doctrine is untested against this confound.

**Counter-evidence (fine-tunes can be plastic).**
- W absorbed the removal of the clock observation and reward within about 450 iterations: episode length -27% at iteration 164, back by 450. It then beat its clocked twin (2026-09-23_clock_deprecation_w2/REPORT.md:147-165).
- A plain 2000-iteration fork moved move_turn facing from 0.235 to 0.411.

**Consolidation reduces modifiability.**
- 3k / 8k / 20k base: 2b2 clearance learning alive / partial / frozen (0.08-0.15 / 0.064 / 0.013).
- Actor weight norm 26.3 -> 34.6; critic 30.7 -> 46.7; C7 critic 57.1. No dormant units (T/crab_hex_forward_task/experiments/lit-review-plasticity-cliff.md:8-38).

**Late degradation, confounded with curriculum.**
- Forward A15+B obstacle completion by lineage checkpoint: 0.68 (5k), 0.66 (20k), 0.29 (25k), 0.48 (30k).
- Golden plant: 0.60 -> 0.27.
- The full-income control also decayed, to 0.19.
- Sideways 2c step onset: pooled -0.17 [-0.24, -0.10] on 2 seeds.
- No clock-held control was ever run.
- The least-shaped heads are often best: P2 10k no-ramp heads scored 0.73/0.95 vs 0.56 for the ramped lineage head (2026-09-06_2130_a15b_lineage/REPORT.md:101-126).

**Not converged at 12k (omni).** Fwd ratio 0.733 -> 0.785 over 4000 more iterations. Checkpoints of one run swing by 0.2 in survival (policy/README.md:140-147; WAVE2_METRICS.md:264-280).

**Reward curves.** Fresh runs reach 90% of final reward at median iteration 1344-1798 (omni 3000-iteration runs: 1777, IQR 1621-1985). The last third adds 5-15%. Resumed stages reach 90% within 85-147 iterations.

**Warm-start penalty after action-semantics change.** Fine-tune reward 11.4 vs 21.0 from scratch. The fine-tune's value loss peaked at 0.426 (2026-08-13_1900_geometry_velocity_actions/RESULTS.md).

### 4.3 Exploration and action bounds [ALG/HP: strongest optimizer-side evidence, about parameterisation rather than PPO as a family]

- Sections 3.2 and 3.3 give sigma pinning and mean saturation.
- **Chatter.** 2.23 sign flips per dim per second at stand and 1.83 while moving. Stand contact switching is 1.65 Hz, the same as walking. The ledgers attribute stand stepping and the rigid 3-feet pattern to the saturated mean, not to the clock or the reward (memory clock-deprecation-campaign-active.md:84-89; lets-explore-options plan :181-183).
- **action_rate penalty rewards saturation.** Monte Carlo at sigma 2: expected \|delta a\| per dim is 0.95 at \|mu\| = 0 and 0.00 at \|mu\| = 9 (lit-review-omni-reward-simplification.md:13-18).
- **M1 (sigma reset 0.4, cap 0.5, 2000-iteration fork of saturated X08:c6): REJECT.**
  - Q z -1.358 vs R 0.319; jitter +10-15%; move_turn ratio 0.504 -> 0.433; tracking income 0.61 vs 0.69.
  - sigma re-pinned at 0.5; median \|mu\| 9.1 -> 9.5.
  - The LR sat at the floor in 100% of 2000 iterations. The same-fork controls X01/X02 were off the floor in 41.5% / 43.0%.
  - Because KL scales as 1/sigma^2, cutting sigma also cut the step size. M1 is not a clean test of low sigma.
  - The bound-loss plan (M1b) was audited and never run (ANALYSIS_M1_saturation.md:6-30, 63-68; plans/there-is-a-lot-compressed-unicorn.md).
- **Action authority.** Scale 0.4 broke from-scratch formation (tripod 0.199, completion 0.25). Scale 0.3 failed too (0.022). Resuming a formed gait at 0.4 kept the gait but completion was 0.27-0.44. The ledger reads this as exploration violence proportional to authority (gait_formation_v2/CHANGELOG.md:38-47). A lower-sigma, higher-authority arm was never run.
- **Deploy dependence on the clip.** The unclipped student env made the teacher itself fall in 100/100 episodes. With the clip, teacher 0.70 and student 0.79 (2026-09-07_1330_phase_pipeline/CHANGELOG.md:16-20).
- **Era-A leg snaps.** Up to 195 mm of foot travel in 20 ms; 57% of MR air-time events last a single step (2026-08-08_0950_stride_length_on/CHANGELOG.md:48-56). Consistent with bang-bang control, but saturation was not measured on those checkpoints.
- **Stochastic vs deterministic eval on the same head.** Achieved vx 0.145 vs 0.151 (20k head), 0.169 vs 0.160 (30k head). Consistent with saturation (2026-09-03_1156_obstacle_exposure/REPORT.md:11-56).
- **Caveat.** Saturation is equal in fast and slow side heads, so it does not select the side-speed class (side_bisect/ANALYSIS.md:93-96, 116-119, 308).

### 4.4 Multi-command interference and gait attractors [mixed]

**(a) Reward-preferred degenerate basins [REWARD/TASK].**
- v5 unison glide (tripod 0.0) earned `Train/mean_reward` 28.69 at iteration 2999 vs 26.15 for alternating H. The two reward functions differ by a term that paid 0.0025 in total over the run.
- Glide seeds 6 and 7 earned 31.7 and 29.5 at 3k vs 27.7 for the symmetric reference.
- Omni 1c creeper: stepping in place earns about 99% of available income. The replayed creeper nets 2.23/s vs 0.96/s for the walking 1a head. Training reward rose 61 -> 93 while track ratio fell 0.747 -> 0.245 (T/crab_hex_omni_task/experiments/2026-09-16_1805_omni_formation/ANALYSIS_1c_failure_2026-09-18.md:8-12, 37-54).
- A fall-and-spin reward hack (2026-08-13_2000_onedir_spin/CHANGELOG.md).
- OS11: PPO delivers the added-reward dose as predicted (r 0.67-1.50) but buys cheaper behaviour: taps +0.24/s, lean +0.010-0.036 m, stride per unit speed down.
- Reading: a stronger optimizer would find these at least as readily.

**(b) Basin entry fixed by structure [STRUCTURE].**
- Clock plus RSI 0.2 from a high-quality 2330-state bank flipped the failing seed 2 to tripod 0.496/0.537. A low-quality bank failed (tripod 0.0).
- Never-clocked Z failed formation on 2/2 seeds (c6 survival 0.00-0.46).
- The reversal-penalty dose-response moves the basin only at doses that collapse income (w -1.0: completion 0.0).
- Reading: exploration/structure. No algorithm-side exploration change was ever tried.

**(c) Omni two-direction side walking [STRUCTURE; an optimisation trap inside a shared network].**
- Whenever one network produces both lateral directions, runs are SLOW or DEAD. This covers real -y sampling, and also the phantom -y rows created by the 4-image detached mirror loss in +y-only runs.
  - SLOW/DEAD feet_ss 0.061-0.103 m/s across O, R, C_band, C_rsi, LRfa, LR4, LR4_dwell, C_lr, C_lr_10, C_lr_rsi and W.
  - +y-only runs with `fa` and scalar command obs are FAST: 0.162-0.215 m/s (C 2/2 seeds; C_mo 2/3 seeds, the third short by 0.002). The sideways-task pin is 0.196-0.218.
- The deploy-time M_y canonicaliser makes all 4 fast heads FAST in both directions: 0.182-0.216 m/s, \|L-R\| <= 0.006 (side_bisect/ANALYSIS.md:10-21, 50-64, 318-344; GATE.md:128-141, 280-309; gate/canon_lr.json).
- The fast gait pays more under W's reward (+0.396/s in steady state), yet training returns are nearly equal (O 67.0 vs C_mo 66.1-73.1). So this failure is NOT reward-preferred.
- The gait is committed by iteration 300-600. An early read of track ratio at iterations 800-1000 separates fast from slow perfectly (0.654-0.765 vs 0.473-0.571).
- Optimizer telemetry overlaps between classes (sigma about 1.62-1.70 at iteration 600 in both; saturation 91-98% in both; LR/entropy/surrogate/priv_reg alike).
- Fast-gait RSI in both directions (C_lr_rsi) is still SLOW. RSI 0.33 raised the training failure share to 0.306 (C_lr_rsi33 class pending).
- Untested sub-mechanisms named by the ledger:
  - 1a: weak direction input. The raw +-0.22-0.35 command slots next to +-1 clock slots; no obs normalisation.
  - 1b: "direction x phase hard to represent under this PPO".
  - 5c: a no-mirror control.

**(d) Command additions and staged curricula [REWARD/TASK + CONFOUNDED].**
- Turning / yaw income added at round 0 cost the clock (0.77x / 0.78x). Both passed one round later. speed_band never passed in 4 rounds (gated_lineage/REPORT.md:10-53).
- The staged omni chain lost translation: fwd ratio 0.76 -> 0.16 at the 1a -> 1b switch.
- C1/C1b curricula dropped turn facing 0.419 -> 0.185 / 0.171.
- C1d (turn-first) was adopted by user overrule. C1e was REJECTED: diag survival 0.79 -> 0.49 (facing_tracking_ladder/REPORT.md).
- Retention methods (EWC, kickstarting, BC regularisation) were never tried.

**(e) move_turn.** Every head reaches 0.02-0.07 rad/s against 0.3 commanded. The ledger computes that the outer foot would need about 0.58 m/s, which is kinematically infeasible (facing_tracking_ladder/CHARTER.md:41-53). [REWARD/TASK]

### 4.5 Symmetry [STRUCTURE via the algorithm: the one algorithm-side lever with large effects; mostly n=1]

**Effects.**
- Position era, 20k, n=1: coefficient 0.5 raised tripod from 0.401 to 0.517, with duty balanced in 10/10 episodes. The gait survived 2200 mirror-free teacher iterations (2b2 0.5255) (2026-08-13_0035_mirror_symmetry/RESULTS.md, CHANGELOG.md).
- Velocity era, 3k:

  | mirror coefficient | outcome |
  |---|---|
  | 0 | tripod 0.032, slip 3.6%, tippy 5.7% |
  | 0.5 | tripod 0.573, slip 11.3%, tippy 23.8% |
  | 1.0 | tracking broken |

- Velocity era, 20k: coefficient 0 beat 0.5 on tracking (PASS vs FAIL) and slip (4.1% vs 16.7%); tripod 0 in both (2026-08-15_1530_task1_velocity/CHANGELOG.md:93-137).
- New plant: E5 (mirror off) scored tripod 0.257 / completion 0.20 vs 0.541 / 0.99, n=1 each, within the 1/4 basin rate.

**Current use.**
- Mirror loss magnitude: median 0.026 (x0.5 = 0.013) vs \|surrogate\| 0.008. It never decays: first-100 median 0.024, last-100 median 0.026. Omni 4-image about 0.047-0.059; `fa` about 0.028-0.035.
- Mirror type makes no difference once -y is genuinely sampled (LRfa vs LR4 both SLOW).
- The coefficient was only tested at {0, 0.5, 1.0}, on the older plant.
- Never run: data augmentation, hard-equivariant networks, and a mirror phase-out.

### 4.6 Curriculum, resume and critic [implementation (fixed) + critic scaling; CONFOUNDED]

- **Resume reset.** Before eabe7324, each resume restarted priv_reg, the LR and the auxiliary optimizer states. Heads with an open latent gap (about 0.035 vs 0.0004) turn at facing 0.17-0.25, while closed-gap heads reach 0.29-0.75. This is collinear with the curriculum; the separating control was never run (facing_tracking_ladder/PLAN_AMENDMENTS.md:262-296).
- **Post-fix re-run.** X0 vs pre-fix X08:c6: side ratio 0.396/0.389 vs 0.430; F_mt improved 0.235 -> 0.330/0.365. Not a clean A/B.
- **Teacher hand-off.**
  - Chain #3: bridge failure 0.00 -> 0.90. The flat critic priced about 0.5/step against about 0.0015/step on the bridge.
  - Chain #4, with critic reset and Adam clear: bridge gated at failure 0.25, but 2b1 degraded 0.58 -> 0.91 at the next boundary.
  - Chain #5 (full goal/yaw dose): failure 0.00 -> 0.96 (2026-08-26_0200_teacher_handoff/CHANGELOG.md:15-20).
- **Critic reset elsewhere.**
  - Phased_flat B1 (flat -> flat): negative (tripod 0.601 -> 0.450).
  - On the 2b2 v2 base: falsified (clearance 0.010 vs 0.013) (phased_flat/CHANGELOG.md:38-46; lit-review-plasticity-cliff.md:107-117).
- **Order effects.** Elements that were catastrophic at round 0 passed one round later: terrain50 clock 0.52x -> 1.03x (gated_lineage/REPORT.md).
- **Never tested:** value or return normalisation (PopArt), value-clip off, a separate critic LR, gamma 0.995, removing the reward clip. Section 3.4 has the value-loss magnitudes.

### 4.7 Distillation [not a bottleneck]

- Student 3a vs its 2c teacher (completion/falls): flat 0.79/21 vs 0.81/19; step 0.71/29 vs 0.68/32; obstacles 0.64/36 vs 0.66/34. 3b (+5k iterations on the hard band) gave no change. Single seed (2026-09-07_1330_phase_pipeline/REPORT.md:47-57, 76-88).
- The only failure was the clip plumbing bug.
- The old-plant student was weaker than its teacher: tripod 0.161 vs 0.237 (phased_flat/CHANGELOG.md:283-314).
- Distillation compute: 66.2 GPU-h over 12 runs; about 6.7 s per iteration at 192 envs (607-676 fps).
- Distillation of per-direction specialists is the ledger's named fallback for omni side walking (about 150-200 lines, never run).

### 4.8 Attribution scoreboard

| Failure class | Example (numbers) | Ledger attribution | Does optimizer telemetry separate good from bad? | Merged reading |
|---|---|---|---|---|
| Degenerate reward-preferred gait | glide 28.69 vs tripod 26.15; creeper 2.23/s vs 0.96/s | reward economy | n/a (both under the same regime) | REWARD/TASK |
| Early basin lottery | 1/4 plain-clock seeds form; 0/7 seeds alternating | GPU nondeterminism, then structure (clock/RSI) | weak correlation: earlier sigma pin in failing runs (n=11) | STRUCTURE + possibly exploration (ALG/HP untested) |
| Two-direction side slowness | 0.06-0.10 vs 0.16-0.22 m/s | shared-network interference | no | STRUCTURE (optimisation trap inside a shared network; algorithmic fix untested) |
| Chatter, stand stepping, 3-feet pattern | 96% saturation; 2.23 flips/s | saturated bang-bang mean | yes (all heads saturated) | ALG/HP (parameterisation) |
| Fine-tune inertia | 27 era-A fine-tunes 0.383-0.425; W forks stride flat | basin lock-in | fine-tunes at LR floor 76-98%, sigma 2.0 | CONFOUNDED |
| Late-window regressions | obstacles 0.66 -> 0.29 | curriculum / clock anneal | LR floor share rises with window | CONFOUNDED |
| Stage-boundary collapse | bridge failure 0.00 -> 0.90 | critic scale mismatch | n/a | critic/return scaling (value-side HP) |

---

## 5. Hyper-parameters and algorithm knobs

### 5.1 Ever varied

| Knob | Values | Outcome | Source |
|---|---|---|---|
| mirror_loss_coeff | 0 / 0.5 / 1.0 | 0.5 formed the tripod (0.517 vs 0.401; 0.573 vs 0.032; 0.541 vs 0.257); 1.0 broke tracking; at 20k in the velocity era 0 beat 0.5 on tracking and slip. All n=1 per arm. 0.5 baked since 2026-08-13 | mirror_symmetry CHANGELOG; task1_velocity CHANGELOG:93-137; gait_formation_v2 CHANGELOG:23 |
| Mirror images | 4-image vs `fa` | +y-only with fa and scalar obs: FAST; two-direction: SLOW with either | side_bisect GATE.md:128-141, 210-223 |
| sigma cap / reset (KRABBY_ACTION_STD_MAX / _RESET) | 2.0 vs 0.5 cap with 0.4 reset (M1) | REJECT, confounded by an LR at the floor 100% of the time | ANALYSIS_M1_saturation.md; TB X01/X02/X03_c6ft |
| entropy_coef | 0.01; 0.005 in one 20-iteration smoke | effectively never varied | runs.csv |
| learning_rate (configured) | 3e-4 / 2e-4 / 1e-4 / 3e-5 by mode | no effect: overridden by the controller | runs.csv; rsl_rl_ppo_cfg.py:53, :74-89 |
| init_noise_std | 1.5 vs 0.65 | 0.65 never effective, because resumes load sigma 2.0 | TB Policy/mean_noise_std |
| clip_actions | 2.0 -> 1.0 (flat walk, afbc0a46, no rationale); student None -> 1.0 | None made the teacher fall 100/100; 1.0 is load-bearing | git afbc0a46; phase_pipeline CHANGELOG:16-20 |
| Action scale (KRABBY_ACTION_SCALE) | 0.25 / 0.3 / 0.4 | 0.3 and 0.4 fail formation from scratch | gait_formation_v2 CHANGELOG:38-47 |
| Cam lower clip (KRABBY_CAM_CLIP_LO) | none / 0.1 | dead | gait_formation CHANGELOG:92-102 |
| Critic reset + Adam clear at a stage boundary | on / off | mixed: positive at one teacher bridge; negative in B1; falsified on v2 | teacher_handoff CHANGELOG:17-19; phased_flat CHANGELOG:38-46 |
| resume_alg_state | reset vs carried (eabe7324) | no clean A/B | memory ppo-counter-reset-on-resume.md |
| RSI fraction / bank | 0 / 0.15 probe / 0.2 good bank / 0.33 | a good bank closes the formation lottery; does not fix the two-direction side gait | gait_formation_v2 CHANGELOG:25-31; side_bisect CHARTER log |
| Training length / fine-tune vs from scratch | 1k-30k; 2000- to 6000-iteration fine-tunes | Sections 4.2 and 4.4 | many |
| Seeds | 1-8 | Section 4.1 | many |
| num_envs | 256 (RL); 192/256 (student); 64 (smokes) | never varied for RL training | runs.csv |
| Episode length / command resample | 20 / 40 / 70 s; resample 6 / 10 / 40 s | 40 s adopted; 70 s and one-command episodes harmful | morph CHANGELOG:108-123; obstacle_exposure CHANGELOG:209-220; side_bisect |
| Tracking kernel sigma^2 (a reward width, not the policy sigma) | 0.02 / 0.1 / 0.25 | 0.1 baked | gait_formation CHANGELOG:67, 83, 106 |
| Command obs scalar vs visible; command pool | side_bisect arms | Section 4.4(c) | side_bisect GATE |
| Smoothness knobs M2/M3/M7; CAPS; bound loss (M1b) | prepared or planned | never run | MQ REPORT:9-18; reward_simplification CANDIDATES |

### 5.2 Never varied in any of the 437 PPO runs

- desired_kl (0.01)
- schedule (adaptive, per minibatch)
- LR floor (1e-5)
- num_learning_epochs (5)
- num_mini_batches (16)
- clip_param (0.2)
- gamma (0.99)
- lambda (0.95)
- value_loss_coef (1.0)
- clipped value loss (on)
- max_grad_norm (1.0)
- num_steps_per_env (24)
- entropy_coef (0.01, except one smoke)
- num_envs for RL (256)
- empirical_normalization (off)
- noise_std_type / std parameterisation (scalar + clamp)
- action distribution family (Gaussian + clip)
- Adam hyper-parameters
- network sizes and activation
- critic LR separation (shared)
- advantage normalisation mode
- reward >=0 clip
- time-out labelling
- mirror as augmentation vs loss
- algorithm family (no SAC/TD3/other ever run)

---

## 6. Constraints on alternatives

**Hardware.** One RTX 5080 (16,303 MiB); Ryzen 9 9950X3D (32 threads); 59 GiB RAM. Another session is usually training. Runs are serial: two parallel 256-env runs split throughput to about 2,700 fps each.

**Budget.**
- About 528 GPU-h were used over the roughly 57 days of retained logs (2026-08-04 to 2026-09-30).
- Arm costs:

  | arm | cost |
  |---|---|
  | 3000-iteration from-scratch arm | about 1.1-1.2 GPU-h (18.4M samples) |
  | 5000-iteration window | 1.8-2.6 GPU-h |
  | W lineage (12k iterations) | about 5.4 GPU-h per seed |
  | 3 arms x 3 seeds x 3000 iterations | about 10-11 GPU-h |

- The early side read at iteration about 1000 costs about 0.4 GPU-h per arm.

**Env count.**
- 256 envs is a CLI choice made from one 2026-08-12 measurement.
- Collection appears latency-bound at 256 or fewer envs (Section 3.5), so 512-4096 envs in one process may add samples almost for free. This is untested.
- Physics scaling of the 230 kg plant with height scan beyond 256 envs is unknown.
- VRAM: two 256-env teachers fit in about 10 GB. The depth student OOMed at 256.

**Learner share.**
- The learner uses about 20% of wall time.
- An off-policy learner with a high update-to-data ratio would compete with the simulator for the same GPU.

**Replay memory.**
- Full observations are 1149 float32 values, about 4.6 KB each.
- 1M transitions take 4.6 GB, or 9.2 GB with next_obs. That is too large next to Isaac on 16 GB.
- Viable options:
  - store the 89-dim proprio (356 B) and rebuild the 10-step history;
  - use fp16;
  - use a smaller buffer.
- Runs so far used 18M-123M samples; the W lineage used 74M.

**Libraries.**
- isaac_venv has only rsl-rl-lib 5.0.1 (PPO, Distillation). Its own PPO cannot take this policy: it requires MLPModel actor/critic and TensorDict obs groups (PPO file :45-49) and has no std clamp or bound loss.
- tensordict 0.13.0, torch 2.7.0+cu128 and gymnasium 1.2.1 are present.
- IsaacLab's isaaclab_rl 0.5.2 ships wrappers and pip extras for skrl (>=2.0.0), rl_games and sb3. None of these packages is installed, and neither are torchrl, jax, sample_factory or tianshou.
- Any switch needs a pip install plus porting of the RMA-style actor.

**Runner and checkpoint contract.**
- The runner dispatches on the `class_name` string, accepting only PPOWithExtractor and DistillationWithExtractor (on_policy_runner_with_extractor.py:63-68).
- The save/load dict has fixed keys: model_state_dict (actor.*, critic.*, std), estimator_state_dict, optimizer_state_dict, iter, alg_state, depth_* (:682-740).
- Consumers of this contract:
  - play.py:126, :145
  - evaluation.py:126
  - eval_crab_hex_gait.py (reads policy.std for --sample-actions, :730-744; canonicaliser hook :567-576)
  - bundle_policy.py
  - policy_summary.py
  - the policy/ manifests
- The phase chains fork via --resume / --checkpoint.
- Unknown algorithm keys are dropped silently (D14).

**Telemetry and CLI contract.**
- TB tags parsed by the experiment tools (side_bisect/scripts/analysis/tb.py, replay_*.py, c7_design):
  - Loss/learning_rate, Loss/entropy, Loss/symmetry, Loss/priv_reg(_coef), Loss/surrogate, Loss/value_function
  - Policy/mean_noise_std
  - Train/*, Episode/*, Metrics/*
- AGENT_DUMP_KEYS in run_candidates.py.
- train.py flags (--task --num_envs --resume --checkpoint --max_iterations --run_name --seed), called by run_phases.py and run_candidates.py.

**Deploy path.**
- The exporter serialises only the actor (plus normaliser, estimator and depth encoder in the deploy variant). There is no algorithm state in it (P/scripts/rsl_rl/exporter.py:112-150).
- A same-shaped actor trained by any algorithm exports identically.
- Must keep:
  - the obs slicing (75/132/9/41/10 as built);
  - estimator substitution with hist_encoding=True in eval_crab_hex_gait.py:1022-1025, play_stream_video.py:203-204 and evaluation.py (or simplify all of these consistently);
  - action dimension 18;
  - executed-action semantics (clip at +-1 is load-bearing; a tanh or Beta head must preserve the executed range end to end).
- The 50 Hz policy vs 100 Hz hardware loop question is open.

**Teacher/student chain.**
- DistillationWithExtractor deep-copies `policy.actor` into depth_actor (distillation_with_extractor.py:50).
- It regresses the teacher's `act_inference(obs, hist_encoding=True, scandots_latent)` raw means.
- A new teacher must keep an Actor-compatible module. With a squashed actor, the distillation target should be the executed action.

**A swap must re-implement:**
- the mirror loss (baked into every head of record) or an equivalent augmentation / equivariance;
- the exploration knobs;
- alg_state across resumes;
- telemetry tags;
- correct terminal vs time-out labels (D9; critical for off-policy bootstrapping);
- a decision on the reward clip.

**A swap need not port** the estimator, DAgger history encoder or priv_reg (vestigial, D6), provided the deploy path is simplified consistently.

**Algorithm-agnostic components** (live in the env or wrapper): RSI, curricula, command samplers, reward terms and the canonicaliser.

---

## 7. Open questions for the literature review

1. **Per-minibatch adaptive KL at small batches.** Is the per-minibatch adaptive-KL LR rule (rsl_rl style: /1.5 or x1.5 per minibatch, floor 1e-5) documented to collapse the LR at small minibatches (384 samples, 80 steps/iteration)?
   - What LR trajectories do legged_gym and Isaac Lab report at 98k-147k-sample batches?
   - Is a per-epoch or per-iteration KL check, or a fixed or linearly decayed LR, better supported for locomotion PPO?
2. **Clipped Gaussians with sigma at or above the bound.** Is mean drift beyond the action bounds, with sigma at or above the bound, a known pathology?
   - Relevant sources: Chou et al. 2017 (Beta policy); Fujita and Maeda 2018 (clipped action policy gradient); Andrychowicz et al. 2021 (tanh vs clip, initial std 0.5); rl_games bounds loss.
   - What effect sizes on motion smoothness and learning speed have been measured for locomotion?
3. **KL on the executed distribution.** Should the KL constraint, or the log-prob, be computed on the executed (censored or squashed) distribution? Is there evidence that doing so restores learning signal?
4. **sigma parameterisation.**
   - Evidence on state-independent log-std vs scalar std with a hard clamp.
   - init_noise_std 0.15-1.0.
   - entropy_coef 0 vs 0.005-0.01.
   - Target-entropy / SAC-style temperature inside PPO.
   - Does any stack use a zero-gradient clamp?
5. **Batch geometry at low env counts.** For 256 envs x 24 steps, what minibatch count, epochs and rollout length do small-batch PPO studies recommend?
   - Relevant sources: Andrychowicz 2021; Rudin et al. 2022 on batch-size and env-count scaling.
   - Does a rollout of 48-96 steps help when the gait period (1.2-2.8 s) exceeds the rollout (0.48 s)?
   - What do gamma 0.995 or lambda 0.97 do at 50 Hz?
6. **Env-count scaling.** Is sim throughput typically latency-bound at 256 envs in Isaac Lab for articulated robots with height scans? What speedup is typical from 256 to 1024-4096 envs on a 16 GB GPU?
7. **Off-policy methods for massively parallel sim at this scale.** FastTD3, FastSAC, "SAC on massively parallel simulation", PQL, PQN, SAPG and similar:
   - At what env counts, and with what VRAM, do they work?
   - Wall-clock vs PPO?
   - Behaviour with 1000+-dim observations including history?
   - Sim-to-real results?
   - Any hexapod, or non-backdrivable / slew-limited actuator, results?
   - How do they handle symmetry?
8. **Model-based options.** Are DreamerV3, TD-MPC2 or other model-based methods competitive for legged locomotion under this compute (one 16 GB GPU, sim-bound)?
9. **Seed variance and basin lottery.** Do any algorithms or add-ons (population-based training, SAPG, QD / MAP-Elites, ensembles) measurably reduce seed-to-seed variance or basin-lottery rates in locomotion?
10. **Multi-task interference between mirrored commands.**
    - Is a shared MLP known to fail on mirror-related command directions without equivariance?
    - Evidence for canonicalisation / frame averaging, hard-equivariant networks (MS-PPO, Su et al. IROS 2024), mixture-of-experts or multi-head policies, and gradient-conflict methods (PCGrad, CAGrad) in locomotion.
11. **Symmetry method.** Mirror loss vs augmentation vs hard equivariance. Consequences of a detached target and of applying the loss to raw vs bounded actions. Recommended coefficients.
12. **Plasticity.** Does loss-of-plasticity work (Dohare, Lyle, Nikishin; shrink-and-perturb; L2-init) apply to PPO fine-tunes at LR of about 1e-5? Are fine-tune nulls expected at such step sizes?
13. **Value side.** Value or return normalisation (PopArt) under staged curricula with non-stationary reward scales. Effects of the legged_gym "only positive rewards" clip on learned behaviour. Separate critic LR.
14. **Bounded teachers and distillation.** What target should a DAgger or behaviour-cloning student regress when the teacher is bounded (tanh or Beta)? What happens when the teacher's means are saturated?
15. **Per-dimension KL scaling.** Does per-dim scaling of desired_kl (18 vs 12 action dims) matter in practice?
16. **PPO vs off-policy on bang-bang-prone actuators.** Is there any comparison on velocity-controlled cams or slew-limited actuators, where smoothness penalties interact with the action distribution?

---

## 8. Contradictions between miners and how they were resolved

1. **"LR pinned at 1e-5 from iteration 0 for the whole run"** (memory ppo-counter-reset-on-resume.md; infra-04: "about 99.8% at the floor" from 9 sampled TB points) vs **"oscillates"** (telemetry; impl_audit; omni miners).
   - The census wins: 62.9% of iterations at the floor, 205,937 excursions, iteration 0 at the floor in about 90% of runs, not all.
   - Infra's sampled points (e.g. 3.8e-4 at iteration 2) themselves show off-floor values.
   - Both descriptions understate the within-iteration sawtooth (Section 3.1).
2. **Mean saturation 96%: verified?** Infra says unverified (no .pt reads allowed). impl_audit verified it by CPU reconstruction (W: corr 0.9967, sign agreement 0.983; X19: 0.962 / 0.938). This agrees with ANALYSIS_M1 (0.960-0.965) and the side-bisect ANALYSIS (91-98%). impl_audit wins.
3. **"One 256-env run saturates the GPU"** (memory parallel-gpu-training-runs.md; infra-02) vs **"collection is latency-bound"** (telemetry T07).
   - The merger's runs.csv check shows almost equal per-iteration collection at 64 and 256 envs (omni 0.93 vs 1.04 s; flat 1.34 vs 1.28 s), and a constant 0.30 s learn time.
   - The parallel-runs note is also consistent with GPU time-slicing between processes.
   - Resolution: within one process, env count up to 256 is nearly free. Beyond 256 is untested. Infra's "hardware-capped at about 5k steps/s regardless of algorithm" is not established.
4. **Student distillation hyper-parameters.** fwd-sep-side FS9 reports "LR 2e-4, 4 minibatches"; impl_audit reports "fixed 1e-3, one Adam step per iteration". The code (distillation_with_extractor.py:50-66) builds `depth_actor_optimizer` at `depth_encoder_cfg["learning_rate"]` = 1e-3 (rsl_rl_ppo_cfg.py:179) and steps once in `update_depth_actor`. impl_audit wins; the 2e-4 / 4-minibatch block is unused by the student.
5. **Same-seed reproducibility.**
   - fwd-sep-side: bit-reproducible (A15+B w0 matches morph P2 stage 1).
   - omni-early: X09:t3 did not reproduce X08:t3 (identical config and seed).
   - fwd-aug-mid / fwd-aug-late: pairs differing only in max_iterations diverged by iteration 20-500.
   - fwd-aug-early: identical until a new reward term pays (that is config sensitivity, not reproducibility).
   - Resolution: nondeterminism exists (the X09/X08 pair proves it) but is not universal (the w0 pair). The cause is unknown. Seed-matched paired designs cannot be assumed to isolate a knob. FS15's recommendation of "seed-matched paired design" is therefore weakened.
6. **Side-speed attribution.** OS1 / side_bisect ANALYSIS: structural interference. OS6: an optimisation trap between near-equal-return basins. Both hold: the trap occurs only when one network carries both directions, the fast gait is reward-preferred at steady state, and optimizer telemetry does not separate the classes. Merged tag: [STRUCTURE]; an algorithmic escape is untested.
7. **"Critic healthy"** (telemetry T12) vs **"critic throttled at 1e-5, effect unmeasured"** (impl_audit D5). Not contradictory. Value loss is small relative to the value scale, but returns are unnormalised and explained variance is not logged. Critic quality is unknown beyond this.
8. **sigma reaching 2.0.** The scouting lead ("goes 1.5 -> 2.0") is right on direction but not timing. From scratch the cap is reached at median iteration 2435 (≥1.99 threshold) or 2750 (above-clamp threshold). sigma is about 1.62-1.70 while side gaits form (iterations 300-600). Resumed stages start at 2.0.
9. **Resume fix status.** memory ppo-counter-reset-on-resume.md says the fix is unmerged. It is stale: eabe7324 is on four feature branches including the current one, but not on main. Separately, only 15/33 post-fix resumed runs start at priv_reg 0.1. This is unexplained (possibly resumes from pre-fix checkpoints without alg_state).
10. **Symmetry implementation.** lit-review-hexapod-reward-stability.md:124-126 says "data augmentation plus mirror loss". The code and 379/379 agent.yaml files say mirror loss only (`use_data_augmentation = False`). The code wins.
11. **Velocity-era tracking "flatline at about 0.25 for 17,500 iterations"** (onedir_spin RESULTS). The flatline holds for one run only: the identical-config screen reached 0.735-0.752. The ledger's structural mechanism is weakened (fwd-aug-mid F8).
12. **Omni 1_t3 provenance.** omni-early notes a sha256 inconsistency: policy/README gives 03b87cd3 for X08_t3, while PLAN_AMENDMENTS 40 lists 03b87cd3 vs f3783ec5 for X09 vs X08. Not investigated; it does not affect conclusions here.

---

## 9. Evidence gaps (stated plainly)

- No run ever varied a core PPO hyper-parameter or the algorithm family. There is no in-repo evidence for or against alternatives.
- There is no explained-variance, KL, clip-fraction or within-iteration LR telemetry.
- No fine-tune control was run with a working LR and a reset sigma.
- No from-scratch low-sigma arm, and no bounded-distribution arm, was run.
- No single-run throughput probe above 256 envs was run.
- The co-fire rate of mislabelled time-outs was not measured.
- Mean saturation was not measured on era-A checkpoints.
- The CPU re-enactment uses synthetic advantages and deterministic-policy states, runs at a fixed LR, and omits the value, mirror and priv_reg terms.
- Most decisions are single-seed. The seed-2 confirms for W, the phase pipeline and the forward lineage were waived.
- The late-window regressions lack clock-held controls.
- The side_bisect campaign is still live: the C_lr_rsi33 both-hands class is pending, and C_lr_10_rsi33 may be training.

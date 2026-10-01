# Evidence dossier (final): is PPO, or its hyper-parameters, limiting Krabby? Experiments and code half

Final version after adversarial verification. Compiled 2026-09-30, read-only, CPU only. The literature review is a separate stage and follows this document.

This is the merger's draft (the merge stage's draft (not kept)) with every verification verdict applied. Sixteen load-bearing claims (C01-C16) were each checked by two independent verifiers. Where the two disagreed, an escalation verifier re-derived the numbers from primary data (raw TensorBoard events, recorded eval npz files, code, configs). Verifier scripts and outputs are under `S/`.

Path shorthand:
- `R` = /home/nickmagus/krabby/krabby-research
- `P` = R/parkour
- `T` = P/parkour_tasks/parkour_tasks
- `S` = `artifacts/` beside this file (crab_hex_omni_task/experiments/2026-09-30_rl_algorithm_review/artifacts/): the verifier and miner scripts, CSV/JSON/text outputs. The CPU-side pickle caches (~450 MB of TB series) were not kept; re-run the scripts to rebuild them
- "PPO file" = P/scripts/rsl_rl/modules/ppo_with_extractor.py
- "runner" = P/scripts/rsl_rl/modules/on_policy_runner_with_extractor.py

**Tag legend.**
- **[VERIFIED]**: the claim survived verification as worded.
- **[CORRECTED]**: the claim was refuted or partly refuted. The text shown is the corrected wording.
- **[UNRESOLVED]**: the verifiers could not settle it, or only a single unreplicated CPU estimate exists.
- Untagged statements were carried over from the ten evidence miners and were not independently re-verified. None of them is load-bearing for Section 10.

**Verification ledger.**

| Id | Claim (short) | Verdict | Where applied | Verifier artefacts |
|---|---|---|---|---|
| C01 | Adaptive-KL rule mechanics | CORRECTED: the KL uses the *clamped* σ and includes an input-mismatch term | §1 KL row, §2 D3/D7, §3.1 | S/C01, S/C01_verify2 |
| C02 | LR floor census | CORRECTED: scope (164 archived runs omitted); "iteration 0" should read "first logged iteration"; "never took effect" should read "erased within iteration 0" | §0, §1, §3.1 | S/verify_C02, S/esc_C02/analysis.txt |
| C03 | Logged LR hides within-iteration excursions | CORRECTED: the log *shows* the early excursions; the throttle is real for late heads | §3.1 | S/verify_C03, S/verify_C03_esc |
| C04 | Phantom (non-behavioural) KL share; minibatch lever | CORRECTED: the X19@300 share was a bug (0.70-0.79, not 0.49-0.52); the minibatch LR gain is 1.4-3.3x, not 3-10x | §2 D3/D4, §3.1 | S/verify-C04 |
| C05 | σ clamp and the entropy push | CORRECTED: above the cap *every* σ gradient is zero, including entropy; it is a one-way trap, not continued pressure | §1, §2 D2, §3.2 | S/C05, S/C05_escalated |
| C06 | σ pinning census | CORRECTED: 87 (not 88) fresh runs pin, median iteration 2444; the entropy range is 430/437 runs; pinning below the cap is not strictly one-way | §0, §3.2 | S/verify_C06, S/verify_C06_escalated |
| C07 | Mean saturation (bang-bang) | VERIFIED: recorded pre-clip means give slightly *higher* figures than the reconstruction | §2 D1, §3.3, §4.3 | S/verify-C07/npz_sat.json, S/verify-C07-esc/sat_esc.out |
| C08 | 256 envs; knobs never varied | CORRECTED: 40 runs used 64 envs (all smokes); σ cap, init σ and clip_actions did vary | §1, §5 | S/verify-C08, S/verify-C08-adv, S/verify-C08-esc |
| C09 | Iteration time is latency-bound at ≤256 envs | VERIFIED: a clean same-config pair gives +5% wall time for 4x envs; the flat comparison is confounded | §0, §3.5, §6 | S/verify_C09, S/c09_verify, S/esc_C09 |
| C10 | M1 does not refute lower σ | CORRECTED: the conclusion survives, but the "step size was cut" mechanism is overstated and the inherited \|μ\| is 8.1 | §4.3, §5.1 | S/C10v, S/C10_escalate |
| C11 | Seed variance vs effect sizes; reproducibility | CORRECTED: decided effects cleared the seed term by 1.05-4.3x; 2 same-seed pairs reproduce exactly and 2 diverge | §4.1, §8 | S/C11, S/C11_escalated |
| C12 | Two-direction side slowness is not a hyper-parameter problem | CORRECTED: the descriptive core holds; "not the hyper-parameters" was never tested | §4.4(c) | S/verify_C12, S/omni_stroke_side/tb_arms.json |
| C13 | Degenerate gaits earn ≥ target reward | CORRECTED: this holds only for the glide family; the omni creeper numbers were mis-sourced | §0, §4.4(a) | S/c13_verify, S/verify_c13, S/c13_escalate |
| C14 | Fine-tune nulls are confounded by optimizer state | REFUTED as a general statement (see §4.2 for the corrected reading) | §0, §4.2 | S/verify_C14 |
| C15 | The mirror loss is the only large algorithm-side lever, n=1 | REFUTED as worded: seed accounting and costs were wrong, and other algorithm-side levers exist | §4.5, §5.1 | S/C15 |
| C16 | Cost of an algorithm swap | REFUTED in part: the vestigial RMA modules are still load-bearing in checkpoints and deploy paths | §2 D6, §6 | S/C16_code, S/verify-C16 |

---

**Merger spot checks (CPU only; nothing written to the repo). Re-checked by the verifiers where marked.**
- **PPO file.**
  - Lines 377-404: the KL formula and the per-minibatch rule. The LR is divided by 1.5 when KL > 0.02, multiplied by 1.5 when 0 < KL < 0.005, and clamped to [1e-5, 1e-2]. The new LR is written before the same minibatch's `optimizer.step()` (:479). [VERIFIED C01]
  - Lines 186-213: the estimator substitution in `act()`. [VERIFIED C01]
  - Lines 428-456: the loss sum and the mirror loss, with a `.detach()`ed target. [VERIFIED C15]
- **Std clamp.** P/scripts/rsl_rl/modules/crab_actor_critic_with_encoder.py:10-11 sets ACTION_STD_MIN/MAX = 0.05/2.0, and line 50 applies `std.clamp(...)` before `Normal(mean, std)`. The file's own docstring (:24-27) says the clamp's gradient is zero above the ceiling. [VERIFIED C05]
- **Action clip.** P/scripts/rsl_rl/vecenv_wrapper.py:135-136 clamps raw actions to ±clip_actions before `env.step`, and skips the clamp when clip_actions is None. [VERIFIED C07/C08]
- **Reward clip.** P/parkour_isaaclab/managers/parkour_reward_manager.py:53 applies `torch.clip(self._reward_buf, min=0.)`. (Not re-verified.)
- **RL cfg.** T/crab_hex_forward_task/config/crab_hex/agents/crab_hex_rl_cfg.py:
  - lines 12-20: a warning that num_prop 75 does not match the live 1149-dim obs [VERIFIED C16];
  - line 42: `empirical_normalization = False` [VERIFIED C08, False in all 438 runs];
  - line 47: `resume_alg_state = True`;
  - line 59: `num_prop = 75`;
  - line 73: `train_with_estimated_states = True` [VERIFIED C01].
- **Student cfg.** The student runner cfg (forward rsl_rl_ppo_cfg.py:177-193) lists lr 2e-4 and 4 minibatches in its `algorithm` block. The distiller ignores both: it builds `depth_actor_optimizer` at `depth_encoder_cfg["learning_rate"]` = 1e-3 and takes one step per rollout (P/scripts/rsl_rl/modules/distillation_with_extractor.py:50-66). The distillation branch of the runner ignores unknown keys without any warning (runner :115-126). [VERIFIED C16]
- **Resume fix.** eabe7324 (2026-09-22) is on feat/crab-hex-omni-task, feat/omni-side-bisect, feat/omni-stroke-posture and claude/stoic-ptolemy-71e304, but not on main. **[CORRECTED C14]** The priv_reg counter reset is still visible in the 2026-09-23 morning c6ft/M1 fine-tunes (priv_reg_coef 0 → 0.0033). Carry-over first appears in the 2026-09-23 17:14 clock-deprecation smoke (0.0383 → 0.0394).
- **Installed RL libraries.** isaac_venv has rsl-rl-lib 5.0.1, which exports only PPO and Distillation (`rsl_rl/algorithms/__init__.py`). isaaclab_rl 0.5.2 ships wrappers only. skrl, rl_games, stable_baselines3, torchrl, tianshou and sample_factory are not importable. [VERIFIED C16]
- **Collection time vs env count.** See Section 3.5. The census medians hold, but the flat row is confounded. [VERIFIED C09 with correction]
- **LR floor share by configured LR.** Median per-run share of iterations at the floor, census: 3e-4 → 0.56 (382 runs), 1e-4 → 0.79 (22), 3e-5 → 0.997 (29), 2e-4 → 1.0 (4). All runs including the archive: 0.519 / 0.860 / 0.930 / 1.0. The order of the 1e-4 and 3e-5 groups is not stable across populations, and every non-3e-4 run is a resume. [CORRECTED C02]
- **Fresh runs reaching the std cap.** 87 of 204, not 88, at a median first iteration of 2444 (p10 1831, p90 3110). The 88th run in the miner's count was a 20-iteration smoke with a 0.5 cap. [CORRECTED C06]
- **ppo_sim design.** S/impl-audit/ppo_sim.py:55-104 is not a closed-loop LR simulation. It runs one update at a FIXED learning rate and counts the controller's votes (ups minus downs). Its advantages are synthetic, and the loss is actor-only: surrogate plus entropy, with no value, mirror or priv_reg terms. **It also hard-codes σ = 2.0 in the censored-KL call** (`cens_kl(..., 2.0)`), which is wrong for any head below the cap. [CORRECTED C04]

**Census scope.** "The census" below means the 437 PPOWithExtractor runs, 1,091,016 logged iterations, under P/logs/rsl_rl/ (S/telemetry/runs.csv). A live run (crab_hex_omni_flat_walk/2026-09-30_13-46-50_seed3_X20_s1_A15pB) started after the snapshot and brings the total to 438. Another **164 PPO runs (541,518 iterations, 2026-08-07 to 08-17)** exist only as campaign-directory copies under T/crab_hex_forward_task/experiments/<campaign>/**/logs/rsl_rl/ and were not in the census. They are untracked in git, and none duplicates a census run. They agree with the census on every headline figure to within 1-3 points (Sections 3.1 and 3.2). [CORRECTED C02/C06]

---

## 0. Summary for the recommendation stage

1. **No core PPO hyper-parameter, and no algorithm family other than PPO, was ever varied. The in-repo evidence neither supports nor refutes alternatives.** [CORRECTED C02/C08]
   - **Scale.** The census covers 437 runs, 1,091,016 iterations, about 6.70B env steps and about 528 GPU-h (462 PPO + 66 student). The archive adds about 3.3B env steps that were not summed into GPU-h.
   - **Identical in every PPO run** (438; the archive is consistent): epochs 5, minibatches 16, desired_kl 0.01, adaptive schedule, clip 0.2, γ 0.99, λ 0.95, max_grad_norm 1.0, 24 steps/env, no obs normalisation, scalar state-independent std, value coefficient 1.0 with clipped value loss, Adam, CrabHexActorCriticRMA.
   - **Did vary, all marginally:**
     - entropy_coef: one 20-iteration smoke at 0.005.
     - the σ ceiling: 0.5 in 3 runs, including M1.
     - init_noise_std: 1.5 in 383 runs, 0.65 in 55.
     - clip_actions: None in 4 early 500-iteration teacher runs.
     - configured LR: 4 values. It only seeds the adaptive schedule and is erased within iteration 0 in 240 of 254 fresh runs (94%).
   - **The only σ experiment (M1)** was a fine-tune of a saturated head. It does not test lower σ from scratch or with a bounded mean (Section 4.3).
2. **As built here, PPO runs in a degenerate regime. This is a property of this action parameterisation, not of PPO as a family.**
   - **The action distribution is a Gaussian with an unbounded mean.** The env executes clip(a, ±1). [VERIFIED C07]
   - **The σ parameter sits behind a forward clamp at 2.0.** Above 2.0, the gradient of *every* loss term on σ is exactly zero: surrogate, KL and entropy alike. σ climbs from 1.5 to the cap, driven by the entropy bonus and/or the surrogate (the two are not isolated). One step overshoots to about 2.0001-2.0002, and σ then never moves again. [CORRECTED C05]
     - All 59 fresh runs of 3500+ iterations reach 1.99, and none of the 303 runs that reach 1.99 ever leaves (minimum later value 1.9898).
     - 87 of 204 fresh runs pin, at a median first iteration of 2444. [CORRECTED C06]
   - **The mean saturates in every trained head.** These figures come from recorded pre-clip deterministic means in the eval npz files, with no reconstruction:
     - W: 97.0% of components have \|μ\| > 1 (median 11.2, p90 22.0).
     - All 303 evaluated checkpoints at 999+ iterations are 91-100% saturated.
     - 71% by iteration 199; 0% at iteration 19.
     - The deterministic executed controller is therefore clip(μ) = sign(μ) on about 94-97% of components: bang-bang. [VERIFIED C07]
   - **Once a head is saturated, the adaptive-KL controller mostly measures motion that changes no executed action.** [CORRECTED C04]
     - The behavioural (±1-censored) share of the measured Gaussian KL is 0.11-0.16 on W and 0.25-0.41 on X19 at iteration 2999.
     - During formation it is **0.70-0.79** (X19 at iteration 300, σ 1.67). The earlier "about 50%" was an artefact of a hard-coded σ.
     - So the "phantom KL" mechanism applies to late, saturated heads, not to formation.
   - **The KL the controller sees is not pure policy drift.** [CORRECTED C01]
     - The rollout computed old μ/σ on estimator-substituted privileged inputs, and every 20th iteration with the history latent.
     - The update recomputes them on the true inputs with the priv latent.
     - So KL is non-zero before any gradient step. A single CPU estimate on W puts this term at about 3% of desired_kl [UNRESOLVED].
   - **Logged end-of-iteration LR.** [CORRECTED C02/C03]
     - At the 1e-5 floor in 62.9% of census iterations (60.2% including the archive); at or below 3.4e-5 in 95.4% (94.5%).
     - Early in fresh runs the log is well off the floor: mean 5.3-6.2e-5 over iterations 0-250, with 20-24% of iterations above 1e-4.
     - No logged value exceeds 1e-4 after iteration ~900.
     - Late heads live at 1e-5 to 3.4e-5.
3. **The documented outcome failures fall into four classes. Only one points squarely at the optimizer, and it points at the action parameterisation.**
   - **(a) Reward-preferred degenerate basins.** This holds for **the forward unison-glide family only**. At matched iteration 3k, under an effectively identical reward, it out-earns the alternating target: 28.69 vs 26.15, and 31.72 / 29.53 vs 27.67. [CORRECTED C13]
     - Every other documented degenerate family earned *less* than the target (v1 lunge, v2-v4, the shuffles and handed gaits of seeds 2/3/5/8). Their persistence is basin selection, which an exploration or optimizer change could in principle affect.
     - The omni 1c creeper is a within-stage reward drift: reward 61.3 → 93.0 while track_ratio 0.411 → 0.242. In a same-terms replay its advantage comes from the penalty terms.
   - **(b) Stochastic early basin entry.** Fixed by structure (clock, RSI from a good bank, mirror loss). No algorithm-side exploration change was ever tried.
   - **(c) Slow two-direction omni side walking.** The class is decided by one variable: whether one network must serve both lateral hands. [CORRECTED C12]
     - PPO telemetry overlaps completely between fast and slow runs.
     - Supervised distillation rules out representational capacity.
     - Every arm used byte-identical PPO settings, so "not a hyper-parameter problem" is **untested**, not established. The campaign's own ROOT_CAUSE.md lists optimisation-dynamics candidates.
   - **(d) Motion quality** (chatter, stand stepping, the rigid 3-feet pattern). The ledgers attribute this to the saturated mean, which is the parameterisation above.
4. **The throttled optimizer state (σ pinned, LR at the floor) does not explain the ledgers' reward nulls, but the action parameterisation confounds every verdict.** [CORRECTED C14]
   - **The fine-tunes did run throttled:**
     - W forks: 76-78% of iterations at the floor, never above 3.4e-5, σ 2.0002 throughout.
     - Era-A short runs: 94-98% at the floor.
     - The 2026-08-10 tripod-stability fine-tunes: only 46-51% at the floor.
   - **The same stroke and centring terms also failed in from-scratch formation arms with fresh optimizer state**: 23-37% at the floor, peak LR 3.8-5.8e-4, σ starting at 1.5.
   - **Fine-tunes under the throttled state did move behaviour**, e.g. c6x turning 0.475 → 0.787.
   - **So the LR floor and σ pin are at most a secondary, untested confound** for the doctrine that fine-tunes cannot reorganise gait phase structure.
   - **What no in-repo verdict can escape** is that every run ever made used the clipped Gaussian with an unbounded mean, and every head evaluated after about 200 iterations is majority-saturated. Every reward and curriculum verdict is conditional on that parameterisation.
5. **The cheapest, best-evidenced next steps are inside PPO, run from scratch with 3 or more seeds:**
   - **(i) Bound the executed action inside the distribution.** Use tanh-squash with the log-det term, or a Beta, or at minimum a bound loss on μ. This is the one defect with a behavioural signature in every head.
   - **(ii) Make σ gradient-passing.** Use a soft-bounded log-std, or fixed/annealed σ, and revisit the entropy coefficient.
   - **(iii) Make the step-size controller act on behaviour.** Use a KL on the executed distribution, or a fixed/decayed LR, and log KL, clip fraction and explained variance.
   - **(iv) Env count as a free lever.** 4x the envs costs about +5% wall time up to 256 [VERIFIED C09]; above 256 it is untested.
   - **Changing the minibatch count alone buys little.** At the controller's balance point, executed-action change per update moves only 0.8-1.2x between 16 and 4 minibatches. [CORRECTED C04]
   - **An algorithm swap is not first in line.** It would not by itself address classes (a)-(c). It must re-implement the mirror loss and the actor I/O contract, and either stub or re-plumb the estimator and history-encoder modules that every checkpoint loader, exporter and the distiller require (Section 6). [CORRECTED C16]

---

## 1. Effective training setup

"DISAGREE" marks rows where the config files and the real runs disagree.

| Knob | Config says | Real runs (effective) | Source | Flag |
|---|---|---|---|---|
| num_envs (RL) | 6144 (env cfg default) | **[CORRECTED C08]** Of 438 PPO runs, 398 used 256 envs and 40 used 64. All 40 at 64 were smokes of 20 iterations (39 runs) or 2 (one run). All 383 PPO runs longer than 20 iterations used 256. The 165 archived PPO bundles are all at 256. No run anywhere used more than 256 (619 params/env.yaml files: 569 at 256, 41 at 64, 9 at 192). | cfg: T/crab_hex_forward_task/config/crab_hex/crab_hex_env_cfg.py:430; T/crab_hex_omni_task/config/crab_hex/crab_hex_omni_env_cfg.py:277,315; crab_hex_side_env_cfg.py:202,236. CLI: T/crab_hex_forward_task/experiments/tools/run_phases.py:52; T/crab_hex_omni_task/experiments/tools/run_phases.py:71; train.py:122. Census: S/verify-C08-esc/rows.json, S/esc_C09 | DISAGREE |
| num_envs (student) | 192 (student cfg) | 192 for 3a; 256 for older August students. The symmetric-lineage student OOMed at 256 (14.0-14.1 GB of 15.5) | crab_hex_student_cfg.py:26; T/crab_hex_forward_task/experiments/2026-08-13_0035_mirror_symmetry/RESULTS.md:130-133 | VRAM-bound |
| Rollout | 24 steps/env (0.48 s) | 6,144 samples/iteration; TB fps × s/iter has median 6,143 over 397 runs at 256 envs [VERIFIED C08]. This is 24x smaller than the Go2 parkour teacher (6144 × 24 = 147,456; its 24 steps is an override at rsl_teacher_ppo_cfg.py:13 of the base's 120 at parkour_rl_cfg.py:42) and 16x smaller than Isaac Lab Go2 (98,304, not re-verified) | crab_hex_rl_cfg.py:40; T/extreme_parkour_task/config/go2/parkour_teacher_cfg.py:34 | inherited, not rescaled |
| Epochs × minibatches | 5 × 16 | 80 Adam steps/iteration on 384-sample minibatches. No run enabled symmetry data augmentation, so the minibatch really is 384. The Go2 parkour teacher uses 9,216-sample minibatches [VERIFIED C08]; Isaac Lab Go2 uses 4 × 24,576. One randperm is reused across all epochs (isaac_venv rsl_rl/storage/rollout_storage.py:222-255) [VERIFIED C01] | forward rsl_rl_ppo_cfg.py:51-52, :106-107 | minibatch 24x / 64x smaller |
| learning_rate (configured) | 3e-4 (flat-walk/side/omni, teacher modes 2a-2c); 3e-5 (bridge/2b1); 1e-4 (2b2); 2e-4 (legacy teacher base) | **[CORRECTED C02]** It only sets the starting value of the adaptive schedule. It was erased within iteration 0 in 240 of 254 fresh runs (200/204 in the census), which log 1e-5 at step 0 (≥ 9 divisions by 1.5 inside iteration 0's 80 minibatches). The remaining 14 fresh runs reach the floor within iterations 1-9 (one within 10-99). A residual ~1.7x spread in mean logged LR between groups (1.72e-5 / 1.12e-5 / 1.01e-5 / 1.00e-5) is confounded with resume status: every 1e-4/3e-5/2e-4 run is a resume, and within 3e-4, fresh runs average 2.18e-5 vs 1.36e-5 for resumed (all-runs) | forward rsl_rl_ppo_cfg.py:53-54, :74-89, :108-109; PPO file :113, :142, :557-560; S/esc_C02/analysis.txt | DISAGREE |
| LR (logged, end of iteration) | n/a | **[CORRECTED C02/C03]** Census: at the 1e-5 floor in 62.9% of iterations, ≤ 3.4e-5 in 95.4%, > 1e-4 in 0.84%; mean 1.69e-5, median 1e-5. All 602 runs: 60.2% / 94.5% / 1.04%, mean 1.78e-5. Every logged value lies on the 1e-5 × 1.5^k ladder (1e-5: 984,115 iterations). At the floor at the *first logged* iteration in 396/437 runs (546/602); 233 census runs are resumes whose first logged step is not iteration 0. The within-iteration path is not logged (Section 3.1) | TB Loss/learning_rate; runner :543; S/esc_C02 | |
| Schedule | adaptive, desired_kl 0.01 | Adaptive in all 602 runs. **[VERIFIED C01]** Checked on every minibatch (80 per iteration): /1.5 if KL > 0.02, ×1.5 if 0 < KL < 0.005, clamped to [1e-5, 1e-2]. The new LR is applied to that minibatch's own step | PPO file :377-404, :479 | |
| KL definition | n/a | **[CORRECTED C01]** Diagonal-Gaussian KL(old ‖ current) = Σ_i [log(σ/σ_old + 1e-5) + (σ_old² + Δμ²)/(2σ²) − 0.5], summed over 18 dims and averaged over the minibatch. μ is the raw, *unclipped* backbone output; **σ is the *clamped* σ**, both old and new. While pinned, KL ≈ ΣΔμ²/8, and a raw std above 2.0 adds nothing. The KL is measured against the rollout-time distribution, which was computed on estimator-substituted priv_explicit inputs (and, on iterations where it % 20 == 0, the history latent). The update recomputes on the raw stored obs with the priv latent, so KL is non-zero before any step. The size of that term is [UNRESOLVED]; see D7 | PPO file :189-208, :310-311, :343-350, :377-386; runner :273, :279; crab_actor_critic_with_encoder.py:50 | 18 dims makes it 1.5x stricter per dim than a 12-dim Go2 |
| clip_param | 0.2 (log-ratio clamped at ±20) | same in all runs [VERIFIED C08] | PPO file :406-413 | |
| Value loss | clipped at ±0.2 around the stored V; coefficient 1.0 | Same in all runs [VERIFIED C08]. Returns are unnormalised (V on W states 2.85 ± 0.45) | PPO file :415-423; S/impl-audit/misc_probe.py | |
| entropy_coef | 0.01 | **[VERIFIED C08]** 0.01 in 436/437 census runs (437/438 with X20); 0.005 in one 20-iteration, 64-env smoke (2026-09-23_00-27-09_smoke_motionA_allknobs) | cfg; runs.csv | |
| γ / λ | 0.99 / 0.95 | Same in all runs [VERIFIED C08]. Horizon 100 steps = 2.0 s. GAE horizon about 16.8 steps = 0.34 s. Gait cycle about 1.2-2.8 s. The rollout covers 0.2-0.4 of a cycle | cfg; T/crab_hex_forward_task/mdp/crab_hex_dimensions.py:195 | |
| max_grad_norm | 1.0, a single global norm over actor + critic + std | Same in all runs [VERIFIED C08]. The estimator and history encoder are clipped separately | PPO file :372, :478, :596-598 | |
| Optimizer | Adam (0.9, 0.999, eps 1e-8), one param group | **[VERIFIED C01]** One group over `policy.parameters()`: actor backbone, priv/scan/history encoders, critic and the scalar std. The critic therefore runs at the LR set by the actor's KL. The estimator (own Adam, 1e-4) and the history-encoder DAgger optimizer (own Adam, **fixed at the initial learning_rate**, e.g. 3e-4) are outside the rule | PPO file :113, :117-124 | |
| init_noise_std | 1.5 (flat-walk runner, inherited by omni and side; teacher modes 2a/2b/2c); 0.65 (default teacher, student) | **[CORRECTED C05/C08/C06]** 1.5 in 383 runs, 0.65 in 55 early forward-teacher runs. Resumed runs load the checkpoint's σ: 207 of the 233 resumed census runs start at 2.0 | forward rsl_rl_ppo_cfg.py:31, :75, :118, :163; TB Policy/mean_noise_std | DISAGREE |
| Std parameterisation | Scalar per-dim `nn.Parameter` with a forward clamp to [0.05, action_std_max]. action_std_max defaults to 2.0 and can be overridden via KRABBY_ACTION_STD_MAX (default off). Upstream rsl-rl-lib 5.0.1 and the repo's Go2 `ActorCriticRMA` have no clamp | **[CORRECTED C05/C06]** All σ gradients are zero above the cap (autograd check: d(−0.01·H)/dσ = −0.005 at 2.0 and 0.000 at 2.0002). σ ≥ 1.99 in 70.8% of census iterations, a figure inflated by resumes (flat_walk 72.1%, teacher 93.1%, side_teacher 99.7%, omni_flat_walk 58.3%, side_flat_walk 52.9%). The raw parameter on W is 2.00009-2.00042 on all 18 dims. `Policy/mean_noise_std` logs the **raw** parameter (runner :546-551); `Loss/entropy` uses the clamped σ | crab_actor_critic_with_encoder.py:1, :10-11, :35, :50; isaac_venv rsl_rl/modules/distribution.py:168-176; S/C05_escalated/grad.py | krabby-specific |
| clip_actions | 1.0. The student cfg was None until 2026-09-08; the flat-walk cfg was 2.0 until afbc0a46 (2026-05-19, no rationale given) | **[CORRECTED C08]** 1.0 in 434/438 PPO runs; None in 4 early 500-iteration crab_hex_teacher runs (2026-08-06_14-08-45, _21-53-16, _22-05-31, _22-17-36). The action term may clip at ±1 independently (parkour_actions.py:200-203; env.yaml clip (−1, 1) seen in the student run 2026-09-07_23-13-07). Whether those 4 teacher runs executed unclipped actions is [UNRESOLVED]. Storage keeps the unclipped sample and its log-prob | vecenv_wrapper.py:134-136; PPO file :199-208 | |
| Action path | Cams: × π rad/s velocity target, acceleration limit 2π rad/s². Rods: × 0.25 rad offset converted to rod length, slew 0.028 m/s (hip) / 0.033 m/s (knee). 1-step delay | same | params/env.yaml; parkour_actions.py:315-357; crab_hex_dimensions.py:162-196 | |
| Obs normalisation | `empirical_normalization = False` | False in all 438 runs [VERIFIED C08]. Obs column 206 is a constant 158.76 (body mass) | crab_hex_rl_cfg.py:42; S/impl-audit/misc_probe.py | |
| Advantage normalisation | Whole batch, once per iteration | Same; `normalize_advantage_per_mini_batch` False in all runs [VERIFIED C08] | PPO file :20-27, :272-273 | |
| Reward | Step reward × dt; the per-step total is clipped at ≥ 0 (the legged_gym "only positive rewards" convention) | 13.6% of return forgiven on W, about 16% late in X08:c6; pre-clip −27.0 vs post-clip +6.4 at the start of t3 | parkour_reward_manager.py:53; T/crab_hex_omni_task/experiments/2026-09-23_reward_simplification/CANDIDATES_DETAIL.md:259-290 | never varied |
| Control rate / episode | sim dt 0.005 × decimation 4 = 50 Hz. Episodes 20 s (legacy forward); 40 s (A15+B, side, omni); 10 s in some arms (X15/X17/X18/X20) | The hardware loop target is 100 Hz (R/docs/RUNTIME_ARCHITECTURE.md:256-260). Possible mismatch, unconfirmed | params/env.yaml | flag |
| Time-out bootstrap | reward += γ·V(s_t) on truncation | 'total_terminates' is tagged time_out=True but also contains roll/pitch > 1.5 rad and height < −0.25 m (usually preceded by crab_failure). Co-fire rate unmeasured | PPO file :238-242; T/crab_hex_omni_task/mdp/crab_hex_omni_rewards.py:624-647, :750-752; parkour_mdp_cfg.py:1367-1369 | latent bug |
| Symmetry | Mirror LOSS only (`use_data_augmentation = False`), coefficient 0.5. MSE on raw `act_inference` means with a detached target. Images: forward L/R 2; side fore-aft 2; omni 4 (2 with `fa`) | **[VERIFIED C08/C15]** Mirror loss in 379 runs, no symmetry in 59, augmentation True in none. No output tanh (tanh_encoder_output false), so the loss acts on raw unclipped means | forward rsl_rl_ppo_cfg.py:122-145; omni rsl_rl_ppo_cfg.py:76-89; PPO file :432-456; actor_critic_with_encoder.py:86-87 | lit-review-hexapod-reward-stability.md:124-126 says "data augmentation plus mirror loss"; the code wins |
| priv_reg | Schedule [0, 0.1, 2000, 3000] in counter units (counter = its + its/20). Coefficient 0 until about iteration 1905; 0.1 from about 4762 | Reset at every resume before the fix. **[CORRECTED C14]** The reset is still visible on 2026-09-23 morning (c6ft/M1 forks: 0 → 0.0033); carry-over is first seen 2026-09-23 17:14. Run-end coefficient median 0.038; a 3000-iteration run ends at 0.0383 | PPO file :529-541 | |
| DAgger / history encoder | `dagger_update_freq` 20; a separate Adam at the initial LR, never adapted | **[VERIFIED C16, W only]** The latent is constant (norm 0.0114, spread across states 1.2e-6). Zeroing it: KL 8.8e-5; replacing it with its mean: KL 9e-12. **[C11]** Its computation is nondeterministic: Loss/hist_latent differs in the 7th decimal at iteration 0 between same-seed, same-config runs | PPO file :122-124, :566-606; S/verify-C16/probe2.py; S/C11_escalated/tbpair.py | vestigial on W |
| Estimator | 75→128→64→9 ELU, Adam 1e-4 fixed, MSE on obs[207:216]; substituted into the rollout and at deploy | **[CORRECTED C16]** On W, 8 of 9 outputs are constant. Friction varies (std 0.22), but sweeping it 0.6 → 2.0 moves the executed action by 0.001, so no information reaches the actor. The constant outputs are load-bearing, though: setting the estimator output to 0 on W gives KL 17.4, mean \|Δμ\| 2.32 and an executed-action change of 0.065. Forward and side teacher runs randomise mass and com, so the estimator's value in those runs is unmeasured | PPO file :118-121, :186-201; S/C16_code/probe16*.py | vestigial on W, load-bearing in checkpoints |
| Network slicing | num_prop 75, num_scan 132, priv_explicit 9, priv_latent 41, hist 10 (= 1007) | The live obs is 1149 = 89 proprio + 108 heights + 9 + 53 + 10 × 89. The "scan" slice is not the height scan; base velocity, last actions and contacts enter via the tanh scan encoder. The priv-latent slice (216:257) lands on stiffness/damping ratios, which are never randomised [VERIFIED C16] | crab_hex_rl_cfg.py:12-20, :59-60; parkour_isaaclab/envs/mdp/observations.py:148-190 | DISAGREE |
| Networks | Actor 136→512→256→128→18 ELU; critic 1149→512→256→128→1 | About 0.27M actor and 0.75M critic parameters; same in all runs [VERIFIED C08] | forward rsl_rl_ppo_cfg.py:30-43 | |
| max_iterations | 5000 per phase window; 3000 in omni screens; 4000 for c6x; 20000 as the flat-walk default | Chained by resume | params/agent.yaml; crab_hex_phases.py | |
| Student 3a | The `algorithm` block lists lr 2e-4, 4 minibatches, 5 epochs | Actual: depth_actor (a deepcopy of `policy.actor`) + depth_encoder under Adam at a fixed 1e-3, **one** step per rollout of 23,040 samples (192 envs × 120 steps). The loss is the mean L2 norm of (raw teacher mean − student) plus a yaw term. Grad clipping covers depth_actor only | forward rsl_rl_ppo_cfg.py:177-193; distillation_with_extractor.py:50-66; runner :377-509 | DISAGREE |
| resume_alg_state | True since eabe7324 (2026-09-22) | The forward and sideways heads of record predate the fix. After the fix, only 15/33 resumed runs start at priv_reg 0.1. The cause is [UNRESOLVED]; the leading explanation is resumes from pre-fix checkpoints that carry no alg_state (C14: c6ft forks of X08:c6 still reset) | crab_hex_rl_cfg.py:47; runs.csv | |
| Seeds | n/a | era-A: seed 1 in all 86 runs. flat_walk: seed 1 (51), 2 (95), 3 (83), 4 (1). omni: mostly 3, then 2 | params/agent.yaml | |
| Throughput | n/a | Median 4,110 env-steps/s (p25 3,650, p75 4,640) at 256 envs; about 1.23 s collection + 0.30 s learning per iteration. The old plant (era-A) ran at about 5.7k (L1/L5 solo 5,628-5,718 fps) | TB Perf/total_fps, Perf/collection time, Perf/learning_time; runner :625, :634-636 | |

---

## 2. Deviations from reference PPO, ranked by severity

Mostly from impl_audit, with additions from the other miners. The reference points are rsl_rl, legged_gym, Isaac Lab and Schulman et al. The reference-stack statements (e.g. "legged_gym clip_actions 100") were not re-verified and belong to the literature stage.

| # | Severity | Deviation | Evidence | Likely effect | Cheap test |
|---|---|---|---|---|---|
| D1 | critical | **Clipped Gaussian with an unbounded mean.** The Gaussian is sampled unclipped, and its unclipped log-prob and KL are optimised, while the env executes clip(a, ±1). With σ ≥ 1 and nothing in the loss penalising \|μ\| beyond the clip, μ drifts out. **[VERIFIED C07]** | vecenv_wrapper.py:134-136; PPO file :199-208, :344. **Recorded pre-clip means** (eval_crab_hex_gait.py:1025, :1043 store `policy(obs)` before the wrapper clamp):<br>- W, 738,709 steps: 97.0% of components have \|μ\| > 1, 93.4% > 2; median 11.2, p90 22.0; per-dim 95.4-98.8%; P(σ = 2 sample inside ±1) = 3.5%.<br>- X08:c6 / X01 / M1: 0.961 / 0.965 / 0.966.<br>- 17 side-bisect heads: 0.941-0.978.<br>- All 303 checkpoints at 999+ iterations: 0.91-1.00.<br>- Onset (on-policy evals of independent runs): 0.000-0.0003 at iteration 19; 0.713 at 199; 0.92 at 999.<br>- Forward-lineage heads are more saturated (0.99, median \|μ\| 20-36).<br>- The impl_audit X19 trajectory (0 / 8.5 / 52 / 73.5 / 87 / 93% at iterations 0/100/300/500/1000/2999) is **indicative only**: it scores every iterate on W's end-of-training states, with reconstruction fidelity corr 0.962 / sign 0.938.<br>- The censored/Gaussian Fisher ratio for μ is < 0.05 on 74% of dims (impl_audit, not re-verified).<br>- Sources: S/verify-C07-esc/sat_esc.out; S/verify-C07/npz_sat.json | Bang-bang deterministic control, with no credit assignment on saturated dims. Chatter (2.23 sign flips per dim per s at stand). The first gait found gets locked in. Produces the non-behavioural KL of D3. During σ ≈ 2 training, low-\|μ\| dims still flip sign in many samples (P_in 3.5% on W, 7% on X19, 12% at iteration 999), so training is not purely bang-bang | Squashed tanh with the log-det term, a Beta policy, or an rl_games-style bound loss (coefficient 1e-4 to 1e-3, soft bound 1.1; the precedent is from memory and must be verified). Track the share of \|μ\| > 1 and the sign-flip rate |
| D2 | critical | **Absorbing std clamp.** std.clamp(0.05, 2.0) in the forward pass gives exactly zero gradient to the raw σ above 2.0 from every loss term. **[CORRECTED C05]** The entropy bonus pushes σ up only while σ ≤ 2.0 (d(−0.01·H)/dσ = −0.005 at 2.0). Whether the climb is driven by entropy or by the surrogate is not isolated (no entropy_coef = 0 control) | crab_actor_critic_with_encoder.py:10-11, :50. **[CORRECTED C06]**<br>- σ ≥ 1.99 in 70.8% of census iterations.<br>- 303/437 runs end ≥ 1.99, and none that reached 1.99 ever went below 1.9898.<br>- Fresh runs: 87/204 pin at a median of 2444 (p10 1831, p90 3110); 59/59 of those with ≥ 3500 iterations pin.<br>- In the X02 c6y resume, σ was bit-identical at 2.00020 for all 4000 iterations, and entropy was exactly 38.0175 = 18·H(N(0, 2)).<br>- M1 re-pinned at its own 0.5 cap by iteration ~937. | Exploration can never shrink after the cap is reached; lowering entropy_coef would not un-stick a pinned head. Feeds D1 (μ must exceed σ to act reliably). KL per unit Δμ scales as 1/σ² | A soft-bounded log-std, a straight-through clamp, entropy 0 or a target entropy. A reset is needed for any existing pinned head |
| D3 | major | **Step-size controller driven by non-behavioural KL once heads saturate.** The adaptive-KL rule runs per minibatch on 384-sample minibatches, on the unclipped mean and the clamped σ. It also includes the rollout/update input mismatch (C01). | PPO file :377-404. Census in Section 3.1.<br>Fixed-LR CPU re-enactment on W, 16 minibatches (S/impl-audit/ppo_sim.json; votes reproduced by C03/C04), end-of-iteration KL (noise / signal advantages) and net controller votes:<br>- 1e-5: KL 0.010 / 0.013, votes +5 / +11;<br>- 3e-5: KL 0.018 / 0.017, votes −13 / −2;<br>- 1e-4: KL 0.051 / 0.074, votes −72 / −78.<br>**[CORRECTED C04]** Behavioural (censored at ±1) share of the Gaussian KL:<br>- W: 0.11-0.16 (5 seeds, fixed and adaptive LR, 16 or 4 minibatches).<br>- X19 at 2999: 0.25-0.32 on its own eval states, 0.30-0.41 on W's.<br>- X19 at 300 (σ 1.67): **0.70-0.79**. The original 0.49-0.52 came from ppo_sim.py hard-coding σ = 2.0.<br>This is a CPU re-enactment with synthetic advantages and no mirror, value or priv_reg loss; it is not a training measurement. | On W, behavioural change per update is about 0.0011-0.002 nats (share × KL) and executed-action change 0.0018-0.0034 (mean \|clip(μ_new) − clip(μ_old)\|): 5-10x below what desired_kl nominally allows. On X19 at 2999 the executed change is 0.004-0.006; at 300 it is 0.015-0.035. So the throttle binds on late, saturated heads; during formation the KL is mostly behavioural. The configured LR is never tested. The critic is throttled too (D5). That the non-behavioural component is what binds is an inference; no behavioural-KL controller was ever run | KL on the executed (censored or tanh) distribution; fixed LR 1e-4 (no KL check); desired_kl 0.02-0.03; log the approximate KL and clip fraction |
| D4 | major | **Batch not rescaled.** The config came from a 6144-env setup; runs use 256 envs (24x fewer samples per iteration) with num_mini_batches still 16. | forward rsl_rl_ppo_cfg.py:51-52 vs T/extreme_parkour_task/config/go2/parkour_teacher_cfg.py:34 and rsl_teacher_ppo_cfg.py:13, :39-48. **[CORRECTED C04]** Going from 16 to 4 minibatches raises W's balancing LR by about 1.4-3.3x (single seed, interpolated: 1.4e-5 → 4.5e-5 noise; 2.5e-5 → 3.4e-5 signal). With 5 seeds it is 2.6-2.7x; in the adaptive re-enactment the geometric-mean LR goes from 1.1-1.8e-5 to 2.3-5.9e-5. It is **not** 3-10x. At the respective balance points the executed-action change per update is 0.8-1.2x; the earlier "1.2-1.6x" compared against 4 minibatches at 1e-4, which is off-equilibrium (still −6 to −12 net votes). | Noisy steps. A lower LR number, but nearly the same executed motion per update, because at a fixed KL budget the minibatch count barely matters | More envs (Section 6) or longer rollouts, which change the data, not just the split. num_mini_batches 4 is cheap but low-yield alone |
| D5 | major | **Critic shares the actor's LR.** Actor, critic and std are one Adam group, so the critic learns at the LR set by the actor's KL. [VERIFIED C01] | PPO file :113, :403-404. Loss/value_function run median 0.0078. No explained-variance telemetry exists | Unmeasured; possibly stale advantages. The telemetry miner reads the value loss as healthy: RMSE about 3-4% of the value scale (Section 3.4) | Log explained variance; add a separate critic optimizer |
| D6 | major (for porting), minor (for behaviour) | **RMA machinery vestigial on W, but load-bearing in checkpoints.** **[CORRECTED C16]** On W the estimator, the DAgger history encoder and priv_reg carry no per-state information. This has two causes: the obs layout mismatch (num_prop 75 vs 89 moves friction out of the priv-latent slice and base velocity into the "scan" slice), and thin domain randomisation. Only friction is randomised on W (mass/com null); stiffness/damping never are, so the priv latent and the DAgger target are constant. | History latent: zeroing it gives KL 8.8e-5, \|Δμ\| 0.005; replacing it with its mean gives KL 9e-12. Estimator: replacing its output with its mean gives KL 7.5e-5, and the friction sweep 0.6 → 2.0 moves the executed action by 0.0011. **But** zeroing the estimator output gives KL 17.4, \|Δμ\| 2.32, 3.2% sign flips and an executed change of 0.065. Setting obs[75:89] to its mean changes μ by 1.41 mean-abs (impl_audit). Measured on W only; the forward/side teachers randomise mass and com (e.g. crab_hex_teacher/2026-09-18_02-17-45, crab_hex_side_teacher/2026-09-12_03-04-37 params/env.yaml), so there the estimator targets com and friction, and its value is unmeasured. S/verify-C16/probe2.py; S/C16_code/probe16b.py | The RMA and teacher-student framing does not hold as built on W. A swap can drop the *training* losses, but the modules must still be emitted, or every consumer re-plumbed (Section 6) | A unit test asserting the obs layout matches the slice sizes; decide on DR scope |
| D6b | medium (untested) | **Reward clipped at ≥ 0.** The per-step total reward is clipped at zero. This is the inherited legged_gym/extreme-parkour convention, not a PPO deviation as such. | parkour_reward_manager.py:53. Pre-clip vs post-clip return: 43.7 vs 52.0. Blamed for muting feet_slide inside the drag basin (T/crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/RESULTS.md:242-251). Teacher stages log mean reward of about 0.00-0.08 | Biased value targets and missing penalty gradient in bad states. Would carry over to any off-policy method | A no-clip or offset-reward arm |
| D7 | minor | **Behaviour and target policies differ.** act() uses obs with the estimator substitution (and hist_encoding when it % 20 == 0); update() recomputes on the true obs with hist=False. **[CORRECTED C01: this term also enters the KL the LR controller reads]** | PPO file :186-213 vs :343-350. **[UNRESOLVED]** impl_audit's CPU estimate on W (S/impl-audit/mismatch_kl.py): KL(act vs update) mean 3.2e-4, p99 2.7e-3; ratio outside [0.8, 1.2] before any update: 0.04% of samples. It assumes a "true" priv slice built from the estimator's own mean for the constant dims plus uniformly sampled friction. The C01 verifiers call the term unmeasured, and no measurement from logged training states exists | About 3% of desired_kl if the estimate holds; not the cause of the LR floor. Becomes first-order if D6 is fixed or DR is widened | Store obs_est and the hist flag in the transition |
| D8 | minor → possibly major for omni | **Mirror loss on raw means.** MSE on raw unclipped means (\|μ\| about 10) with a detached target [VERIFIED C15], so the mirrored images train the phantom-direction rows through shared weights | PPO file :432-456. Loss/symmetry median 0.026 (× 0.5 = 0.013) vs \|surrogate\| median 0.008. The 4-image omni mirror is implicated in the SLOW class of +y-only runs (Section 4.4(c)); this is confounded with H6 and the command-obs change, and no no-mirror +y arm has run (T/crab_hex_omni_task/experiments/2026-09-28_side_bisect/ROOT_CAUSE.md:149-171). Not included in the CPU simulation | Spends gradient and KL budget on differences that do not change behaviour | Compute the loss on clip(μ) or tanh(μ); run a no-mirror +y control |
| D9 | minor | **Failure cutoffs bootstrapped as time-outs.** Failure cutoffs tagged time_out get bootstrapped with γ·V | Rewards and MDP files cited in Section 1 | Rare; size unmeasured. Matters more for off-policy methods | Split out a non-time-out termination term |
| D10 | historical | **Resume reset.** Before the fix, every resume restarted the counter (priv_reg ramp), the adaptive LR and the auxiliary Adam states | 200/200 pre-fix resumed runs start with priv_reg_coef 0. Resumed processes account for 56% of iterations (513k of 918k), at a mean coefficient of 0.022 vs 0.097 in continuous runs (memory ppo-counter-reset-on-resume.md). **[CORRECTED C14]** The reset is still visible in the 2026-09-23 morning forks. Because the LR hits the floor within iteration 0 anyway (C02), the LR part of the reset is cosmetic for fresh-LR purposes | Confounds pre-2026-09-23 chain comparisons (priv_reg). The memory note calling the fix "unmerged" is stale | Already fixed on feature branches, not on main |
| D11 | minor | **Auxiliary optimizers at fixed LRs.** The history-encoder Adam runs at the initial LR (3e-4 for flat-walk/omni) and the estimator at 1e-4 [VERIFIED C01] | PPO file :118-124, :566-606 | None measurable (the latent is constant on W) | n/a |
| D12 | minor | **One minibatch partition for all epochs.** [VERIFIED C01] | rsl_rl/storage/rollout_storage.py:228-255 | Inherited from upstream | Reshuffle every epoch |
| D13 | minor | **Large unnormalised constant input.** Body mass (158.76) enters the critic and scan-encoder inputs unnormalised | Critic first-layer \|w · 158.76\| is 2.36 of a 2.76 mean pre-activation | Conditioning; a large input shift if mass randomisation is ever enabled (it is in the forward/side teachers) | Turn on empirical_normalization |
| D14 | minor, but a hazard for experiments | **Unknown algorithm cfg keys are silently dropped.** [CORRECTED C16] On the PPO path they are dropped with a warning only (real runs drop `share_cnn_encoders` this way). On the distillation path they are ignored with **no** warning | runner :115-126, :128-141 | A new knob (e.g. bound_loss_coef) silently does nothing unless `PPOWithExtractor.__init__` also takes it | Make unknown keys raise |
| D15 | minor | **Distillation targets are raw means.** The student regresses raw saturated teacher means (median \|μ\| about 21 in the 3a/3b students, C07) with an L2-norm loss | distillation_with_extractor.py:56-66 | Student capacity is spent on differences that do not change behaviour. Student ≈ teacher anyway (Section 4.7) | Distil on clipped actions |
| D16 | minor | **init_noise_std 1.5 exceeds the clip.** | forward rsl_rl_ppo_cfg.py:75, :118 | Half of the samples are clipped even at μ = 0; pushes early training toward D1 | init 0.5-1.0 (only meaningful together with D2's fix) |
| D17 | cosmetic | **Validation setting is a no-op.** `Normal.set_default_validate_args = False` replaces the method instead of calling it, so validation stays on (not re-verified) | actor_critic_with_encoder.py:180 | A NaN crashes the run | n/a |
| D18 | minor (new, C11) | **Nondeterministic history-latent computation.** Same-seed, same-config runs differ in Loss/hist_latent at iteration 0 in the 7th decimal and split in Train/mean_reward at iteration 20 | S/C11_escalated/tbpair.py (X08:t3 vs X09:t3; leg-mount anchor vs rung-v base) | Same-seed paired designs cannot be assumed to isolate a knob | Deterministic kernels / torch.use_deterministic_algorithms on a probe run |

---

## 3. Optimizer-level pathologies measured in telemetry

### 3.1 Learning rate / KL controller

**Census.** [CORRECTED C02] Sources: S/telemetry/runs.csv, analysis.txt; S/esc_C02/analysis.txt; TB `Loss/learning_rate`.
- **Census set** (437 runs, 1,091,016 iterations):
  - At the 1e-5 floor in 62.9% of iterations; ≤ 3.4e-5 in 95.4%; > 1e-4 in 0.84%.
  - Iteration-weighted mean 1.69e-5 (median 1e-5), about 18x below the nominal 3e-4.
- **All 602 PPO runs** (1,634,840 iterations): 60.2% / 94.5% / 1.04%, mean 1.78e-5.
- **Excursions off the floor:** 205,937 in the census. Length median 1 iteration, p99 9, max 42.
- **16 runs never leave the floor after their first iteration:** the Aug-26 bridge runs, the Aug-5/6 teacher runs, and M1 (not re-verified beyond M1).

**Early vs late.** [CORRECTED C03]
- **Early in fresh runs, the logged series shows the excursions.**
  - X17 and X19, iterations 0-250: mean 5.3-6.2e-5, 20-24% of iterations above 1e-4, 44-51% at the floor.
  - Example, X17 iterations 0-5: 1e-5, 1e-5, 3.844e-4, 2.563e-4, 1e-5, 3.844e-4.
  - The climb (×38.4 = 1.5^9 net up-votes) happens inside one iteration. The fall back is visible over the next 1-2 logs; it is not hidden inside the same iteration.
- **Late: the throttle is real.**
  - X17 and X19: no logged value above 1e-4 after iteration 938 / 905. Over iterations 1500-3000: median 1.5e-5, mean 2.2e-5, max 7.6e-5.
  - W's training run (X03_c6x, iterations 7996-11995): never above 3.375e-5; max consecutive jump ×3.4.
- **Per-run maximum consecutive jump** (447 runs with a value in runs.csv):
  - 131 runs reach ×38.4 or more (104 at ×38.4, 26 at ×57.7, 1 at ×86.5), every one of them from scratch.
  - The single ×86.5 is the forward run crab_hex_flat_walk/2026-08-21_19-28-16.
  - Of 241 resumed runs, 239 never exceed ×5.1; the exceptions are X05_1c (×11.4) and X07_t3 (×25.6).

**By regime** (telemetry miner; the W-fork and era-A figures were re-verified in C14):
- From-scratch 3000-iteration omni/side arms: median 2.25e-5, 31-41% at the floor, 3-4% of iterations ≥ 1e-4, maximum 3.8e-4 to 5.8e-4.
- Fine-tunes from W (c6y B/S/P/S_cat): 76-78% at the floor, never above 3.37e-5 [VERIFIED C14].
- Teacher stages: 95% at the floor overall.
- Era-A short fine-tunes (2026-08-09 r01-r14): 94-98% at the floor [VERIFIED C14]. Other era-A fine-tunes are lower: T0/T1/T2 52-84%, and the 2026-08-10 to 08-12 tripod-stability resumes 46-51% (max 5.06e-5) [C14].
- Late lineage windows: 94.2% (2026-08-31_10-46-19, iterations 15k-20k) and 98.9% (2026-09-02_00-42-53, iterations 25k-30k).
- Forward head 2026-09-07_04-38-50 (iterations 14997-19996): 78% at the floor, max 3.37e-5 [C01].

**The controller tightens as training matures** (telemetry miner, not re-verified):
- Share of iterations off the floor, first to last tenth of a run: 0.446 → 0.308.
- Omni chain stages: c4 about 0.59, c5 about 0.54, c6 about 0.45, c6x about 0.37, c6y about 0.23.
- 57% of iterations off the floor while σ is below the cap, 29% at the cap.
- Forward A15+B lineage floor share by window: w1 0.55, w2 0.70, w3 0.78, w4 0.90, w5 0.97.

**What the log cannot show.** [CORRECTED C03]
- `Loss/learning_rate` is the LR after the last of 80 minibatch updates (runner :543). The within-iteration path, and the mean within-iteration LR, KL, clip fraction and explained variance, are not logged anywhere.
- The KL is cumulative against the start-of-iteration policy, so up-votes plausibly come early and down-votes late. The logged value therefore plausibly sits below the within-iteration peak. This is inferred from the controller's structure; it is not measured.

**Balance points** (fixed-LR CPU vote counts, not closed-loop simulations). [CORRECTED C03/C04]
- W, 16 minibatches: about 1.4-1.6e-5 (noise advantages) and 2.5-2.7e-5 (signal).
- X19 at iteration 300: above 1e-4 (noise) and about 1.4-1.6e-5 (signal). The observation source of the X19 sims is unrecorded; recon.py defaults to W's eval states.
- **Conclusion:** "effective LR about 1e-5 to 3e-5" is well supported for saturated or late heads. For early formation, the logged LR is 3-6x higher and the KL is mostly behavioural.

**Mechanism** (impl_audit, CPU, W's own Adam state; medium confidence):
- At the configured 3e-4, per-minibatch KL is 0.2-0.5, with 97-99% of minibatches above 0.02. Nine divisions reach the floor within iteration 0; this matches 240/254 fresh runs logging 1e-5 at step 0 [C02].
- At 1e-5, noise advantages give about the same KL as informative ones: 0.0101 vs 0.0128.
- Only 11-16% of the measured KL on W is behavioural [VERIFIED as corrected, C04].
- With σ pinned at 2.0, KL = ΣΔμ²/8, so a KL of 0.02 allows 0.094 RMS change per dim on means of about 10.
- Ruled out as causes of the floor: priv_reg (acts on a dead latent) and nan_to_num. The act/update mismatch (8e-5 to 4e-4) is [UNRESOLVED] (see D7).
- Not simulated: the mirror loss (a deterministic pull on raw means, which would lower the equilibrium further) and the value-loss share of the global grad-norm clip.

**Reframing for the literature stage.**
- On saturated heads, the binding throttle is a per-update KL budget of about 0.01-0.02 nats, measured on an unclipped Gaussian mean with a clamped σ, of which 84-88% (W) buys no executed change.
- Changing the minibatch count moves the LR number but not the executed motion.
- Fixes that make the KL behavioural (a bounded distribution, a KL on executed actions) or remove the KL check (fixed LR) address the throttle directly.
- During formation the KL is mostly behavioural, so there the controller is doing roughly what it is designed to do.

### 3.2 σ pinning

[CORRECTED C05/C06] Sources: TB `Policy/mean_noise_std` (the raw parameter, runner :546-551) and `Loss/entropy` (clamped σ); S/verify_C06_escalated/recompute.py.

**Census counts.**
- 303 of 437 census runs end at σ ≥ 1.99, and 70.8% of census iterations are at σ ≥ 1.99.
- The 70.8% is inflated by resumes: 207 of 233 resumed runs start at 2.0, and 216 end there.
- Per task: flat_walk 72.1%, teacher 93.1%, side_teacher 99.7%, omni_flat_walk 58.3%, side_flat_walk 52.9% (2 of 8 runs end at the cap).
- Archive: 134 of 164 runs end ≥ 1.99; 63.7% of 541,518 iterations.

**Fresh runs.**
- 87 of 204 reach 1.99, at a median first iteration of 2444 (p10 1831, p90 3110). impl_audit's stricter "above clamp" median was 2750.
- Pinning is a function of run length: 59/59 fresh runs with ≥ 3500 iterations pin, 83/88 with ≥ 3000, 87/106 with ≥ 2000.
- p50 σ by iteration: 1.56 at 100, 1.65 at 500, 1.76 at 1000, 1.95 at 2000, 2.00 at 3000 (telemetry miner). For comparison, X19: 1.668 at 300, 1.859 at 1000, 2.0 at 2999.

**Once pinned, absorbing.** Across all 303 runs that touch 1.99, the lowest later value is 1.9898. In the X02 c6y resume, σ was 2.00020 on all 4000 iterations.

**Below the cap, nearly but not strictly one-way.**
- Only 1 of 437 runs fell more than 0.01 below its starting σ (2026-08-20_19-01-38: 1.4998 → 1.4880).
- Measured from the running peak, 7 crab_hex_flat_walk runs from 2026-08-20 drop 0.038-0.066 (peak about 1.54-1.55 at iterations 50-73, trough at iterations 255-381) before rising again.
- Five archived onedir_spin runs drop 0.08-0.16 (e.g. screen_r4_synthesis 1.69 → 1.53).

**Entropy.**
- Logged entropy = 18 × (0.5 ln 2πe + ln clamp(σ)), max residual 0.107. That gives 32.84 at σ 1.5 and 38.02 at 2.0.
- That range covers 430 of 437 runs. The three KRABBY_ACTION_STD_MAX = 0.5 runs log 9.12-13.06, and four 2026-08-20 runs dip to 32.69-32.79.

**Correlation** (fwd-aug-late F2, n=11): runs that failed to form reached σ 1.99 earlier (iterations 1720-2669) than runs that formed (2986-5116). This is correlational only (not re-verified).

### 3.3 Mean saturation

[VERIFIED C07] These figures come from recorded pre-clip deterministic means in the eval npz files (eval_crab_hex_gait.py:1025, :1043), so they do not depend on the CPU reconstruction.
- **W (omni head of record):**
  - 97.0% of components have \|μ\| > 1 (93.4% > 2); median 11.2; p90 22.0; per-dim 95.4-98.8%.
  - These are slightly higher than impl_audit's reconstructed 96.6% / 10.5 / 20.6, even on the identical subsample (97.05% / 11.24 / 22.05).
- **ANALYSIS_M1 heads:** X08:c6 0.961 (median 8.3), X01 0.965 (9.4), M1 0.966 (9.7). This matches T/crab_hex_omni_task/experiments/2026-09-22_gait_motion_quality/ANALYSIS_M1_saturation.md:36-40 to within 0.5 points.
- **Side-bisect heads:** 17 heads, fast and slow alike, 0.941-0.978 overall, per-dim 0.887-0.999, median \|μ\| 5.5-8.1. X19's own on-policy side eval gives 0.945.
- **Generality:**
  - All 303 evaluated checkpoints at 999+ iterations are 0.91-1.00 saturated, across forward, sideways and omni, and across teachers and students.
  - Below 999 iterations the only data are model_19 (≈ 0.000) and model_199 (0.713).
  - In the 9 runs with two or more evaluated checkpoints, saturation never decreases (e.g. the W chain: 0.964 at 9000, 0.970 at 11995).
- **Not reproduced (needs .pt loads):** the X19 per-iteration sweep, and the reconstruction validation (W corr 0.9967, sign 0.983). Treat the X19 intermediate percentages as indicative.

### 3.4 Value / critic

Not re-verified.
- **Value loss.** `Loss/value_function`, median of the last 100 iterations: flat 0.012 (p10 0.0041, p90 0.0235); omni 0.0058.
- **Relative error.** RMSE relative to r_step/(1−γ): flat 0.038, omni 0.029, teacher 0.154.
- **Growth.** Value loss rises about 2.4x in fresh runs as returns grow. The maximum in any run is 0.463.
- **Numerical health.** No non-finite scalars in any events file, and 0 "non-finite before sanitize" warnings in 546 stdout logs.
- **Teacher stages.** Under net-negative economics they log reward of about 0 and value loss ≤ 0.0003. With the reward clipped at 0, the critic gets almost no signal.
- **Explained variance** is not logged, so critic quality is unknown beyond these magnitudes.

### 3.5 Throughput

[VERIFIED C09, with corrected wording]
- **Clean control.** A same-config omni pair: crab_hex_omni_flat_walk/2026-09-18_02-19-48 (64 envs) and 2026-09-18_02-20-55 (256 envs). Their params/env.yaml differ only in num_envs (lines 86, 361), and their agent.yaml files are identical.
  - Median collection time: 0.934 vs 0.992 s (+6%).
  - Learn time: 0.293 vs 0.303 s.
  - Wall time per iteration: 1.253 vs 1.318 s (+5%).
  - Throughput: 1,250 vs 4,748 steps/s.
- **Second omni pair** (unpaired, 2026-09-19_07-24-22 vs 2026-09-19_11-08-58): +10% collection, +8% wall.
- **Across the census,** 4x the envs costs +5% to +17% wall time: omni +5-8% paired; side_teacher +5-6%; side_flat +8% on early iterations, +17% whole-run; teacher +17%, date-confounded. Learn time is flat at about 0.30 s.
- **The flat census comparison is confounded** (1.335 s at 64 envs vs 1.276 s at 256): code era, plant and terrain config all differ. Do not cite it.
- **Smoke vs steady-state bias is small.** Inside long 256-env runs, collection time over iterations 1-19 vs the last 500 differs by 1-2% (omni, flat), up to 10% (side_flat).
- **A single 256-env run is not compute-saturated.** Live read-only readings of the current omni run (PID 108606): GPU utilisation 61-69%, 66 W of a 360 W limit, 4.4 GB VRAM for the process.
- **The memory note** ("one 256-env run saturates the GPU") rests on the 2026-08-12 lean_reduction L1/L3 overlap:
  - Solo: 0.82 s collection, 0.25 s learn, about 5,700 steps/s.
  - Overlapped: 1.74 s, 0.46 s, 2,783 steps/s each; the aggregate is 97% of solo.
  - Paths: T/crab_hex_forward_task/experiments/2026-08-12_1017_lean_reduction/{L1_orientation_w-2.0,L3_pitchlin_w-0.5}/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_{10-18-02,10-25-29}.
  - This result is equally explained by CUDA context time-slicing. No MPS trial was ever run.
- **No training run above 256 envs exists.** Scaling beyond 256 is untested for the RL phases. It does not apply to the depth-student phase, which is VRAM-bound (OOM at 256).
- **Era-A** (old 104 kg plant) ran at about 5.7k fps; the 2026-08-20 rebuild (230 kg) runs slower.

### 3.6 Late collapse and anomalies

Not re-verified.

**Late collapse** (n=1): flat_walk 2026-08-04_11-55-56, no mirror loss, 13,159 iterations.
- The LR was never off the floor from iteration 7k (run-wide floor share 0.885, C02 spot check).
- σ pinned from iteration 4596 [confirmed in C06's raw parse].
- Median surrogate: +0.056 over iterations 9-11k and +0.171 over 11-13k.
- Reward 45.4 → 19.7; error_vel_xy 0.155 → 0.70; terrain level flat.

**Degenerate fall-and-spin run** (fwd-aug-mid F9): screen_r4_synthesis 2026-08-14_15-24-00.
- Loss/symmetry grows to 26,081 (unbounded μ divergence).
- LR above 1e-4 in 37% of iterations.

**Sanity checks:**
- The surrogate exceeds 0.05 in only 0.31% of all iterations.
- The only ignored PPO cfg key is `share_cnn_encoders` [VERIFIED C16].

### 3.7 Telemetry gaps

None of the following are logged:
- approximate KL (per minibatch or per iteration)
- clip fraction
- mean or within-iteration LR
- explained variance
- per-dim μ saturation
- executed-vs-raw action statistics

Because KL is not logged, the M1 step-size question (Section 4.3) cannot be settled from TB.

---

## 4. Campaign evidence by theme

Tags:
- **[ALG/HP]**: implicates the algorithm or its hyper-parameters.
- **[REWARD/TASK]**: implicates reward or task design.
- **[STRUCTURE]**: implicates representation, priors or architecture.
- **[CONFOUNDED]**: the evidence cannot separate these.
- **[DESIGN]**: a constraint on how any comparison must be run.

### 4.1 Seed variance vs effect sizes [DESIGN; secondarily ALG/HP or STRUCTURE, because a basin lottery is an exploration symptom]

**[CORRECTED C11]** Seed-to-seed variance is of the same order as the treatment effects the omni campaigns measured, but it did **not** exceed the effects the campaigns actually decided on.

**Decided effects cleared the seed term.** Every WORSE/BETTER verdict cleared R = max(R_eval, R_seed), by margins \|z\|/R of 1.05x to 4.3x:
- Facing ladder: D08 F −5.38 vs 3.19; D09 F −5.74 vs 3.19; D10 F +7.73 vs 3.19 and F_mt +4.34 vs 4.12; D12 F +4.94 and F_mt +4.98; D13 T −1.98 vs 1.13.
- MQ: D05 Q −1.36 vs 0.32.
- stroke_formation: D06 F −3.55 vs 1.26.
- stroke_posture: D05 F_mt −3.83 vs 2.45; D06 F_mt −2.63 vs 2.45.

**What the seed term did do:**
- It turned several effects that were significant against eval noise alone into TIEs on the turning blocks: D13 F_mt +3.82 vs 4.12; D11 F_mt +3.38 vs 4.37 (the ladder's only ADOPT was all-TIE); MQ D05 F −2.02 vs 2.07 and F_mt −2.31 vs 6.05.
- It flipped one verdict when the second seed arrived: stroke_formation D05 T went from BETTER (z +1.62 vs 0.985, s3 only) to TIE (z +0.90, s3+s2).

**Seed dominance is block-specific.**
- The turning blocks (F, F_mt) are seed-dominated in every omni campaign: R_seed/R_eval = 3.19/0.97 and 4.12/0.94 in the facing ladder D00bs; 2.07/0.96 and 6.05/0.91 in MQ D00bs.
- The tracking block T is seed-dominated in the facing ladder (1.13/0.54) but not in MQ (0.27/0.52).
- The seed term is smaller than the eval half-width on 5 of 10 facing-ladder guards and 14 of 16 MQ guards.
- Every R_seed is a single s2-vs-s3 draw.
- Sources: T/crab_hex_omni_task/experiments/2026-09-19_facing_tracking_ladder/CHANGELOG.md:46-47, :64-96; 2026-09-22_gait_motion_quality/CHANGELOG.md:18-20; 2026-09-26_stroke_formation/CHANGELOG.md:15-24; 2026-09-26_stroke_posture/CHANGELOG.md:14, :19.

**Large raw seed spreads, each with a caveat:**
- **t3 (identical recipe):** fwd/back survival 0.49/0.51 (s3) vs 0.91/0.97 (s2). But s3 was the only run trained before a reboot, and the other five t3 runs sit at fwd 0.71-0.96 (stroke_formation/REPORT.md:61, :120, :179, :273, :367; GATE_t3.md:26-32).
- **Plain clock, forward:** tripod formed in 1 of 4 seeds (0.541 vs 0.0 / 0.026 / 0.0). Later cured by a high-quality RSI bank (seed 2 → 0.496) (T/crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/CHANGELOG.md:21-28).
- **Morph P2 stage-1 obstacle completion:**
  - B: 0.67 (s3) vs 0.29 (s2). A15+B: 0.68 vs 0.33.
  - The B-vs-A15+B ordering flips at stage 2 (s3 0.83 vs 0.73; s2 0.69 vs 0.95).
  - The morphology-vs-golden effect (0.31 vs 0.69-0.95) stood well outside the two-seed envelope (2026-09-04_1105_morph_x_exposure/REPORT.md:208-345).
- **Seed search, seeds 2-8 (3k):** 0/7 alternating, 4 gait families (2026-08-12_1550_seed_basin_search/RESULTS.md:35-53).
- **Era-A:** of 7 from-scratch runs on the baked reward family, 1 reached the alternating gait (tripod_stability/RESULTS.md:369-394).
- **Spin lottery:** 0/7 walking spinners.
- **Sideways 1a completion:** 0.93 vs 0.79 (T/crab_hex_sideways_task/experiments/2026-09-11_0414_sideways_formation/REPORT.md).

**Eval noise** (not re-verified):
- 100-episode canary: ±0.10.
- Harness resolution: ±0.04 completion.
- An env write after Kit start-up flips one head between 0.79 and 0.75.
- Single-seed MDE: turn facing about 0.145, move_turn about 0.18.

**Same-seed reproducibility is intermittent.** [CORRECTED C11] Four same-seed, same-config pairs were re-checked from raw TB.
- **Two reproduce exactly on every learning scalar:**
  - X03:c1 == X04:c1 (1000 iterations);
  - A15+B lineage w0 == morph P2 stage-1 A15+B (5000 iterations).
- **Two diverge:**
  - X08:t3 vs X09:t3: final mean reward 52.6 vs 46.0; eval fwd survival 0.73 vs 0.90, back 0.78 vs 0.95.
  - Leg-mount rung-v base vs anchor: reward 33.8 vs 35.6; tripod 0.537 vs 0.595.
- **Both diverging pairs share one signature:** Loss/hist_latent differs at iteration 0 in the 7th decimal, Train/mean_reward is identical through iteration 19, and then it splits at iteration 20 and amplifies (D18).
- **Also documented:** a clock-deprecation same-seed replicate spread of 0.014/0.010 slip (2026-09-23_clock_deprecation_w2/CHARTER.md:53-54).
- **Two forward pairs that differ only in max_iterations** diverged by iterations 20-500 (C2 3k screen vs C1+C2 20k: reward at 3k 27.1 vs 13.4, tripod 0.573 vs 0.0).

**Large effects that clear the noise:**
- side speed class: 0.16-0.22 vs 0.03-0.10 m/s;
- clock + RSI formation;
- golden vs wide plants on obstacles;
- clip_actions None vs 1.0 on the student env (100/100 falls vs 0.70);
- the omni mirror-image configuration (Section 4.5).

**Reading.** Most PPO-relevant verdicts were single-seed, and same-seed reruns are not guaranteed to reproduce. Any algorithm or hyper-parameter screen needs:
- 3 or more seeds per arm;
- scoring of formation rate and across-seed spread, not only means;
- no reliance on seed-matched pairing to isolate a knob.

### 4.2 Plasticity, fine-tuning and training length

**[CORRECTED C14: the draft's "confounded by optimizer state" framing is refuted as a general statement.]**

**Fine-tune nulls.**
- **Era-A:**
  - The 2026-08-09 short runs moved little: most rows were "REVERTED -- essentially flat", but r13 was KEPT, with tippy 16.6% vs 19.1% and stride 0.1835 vs 0.178 m (T/crab_hex_forward_task/experiments/2026-08-09_0920_short_runs/RESULTS.md:14, :84; the file has 118 lines, so the draft's pointer ":155" does not exist).
  - With the same config, a fine-tune from v3 scored 0.0 while a from-scratch run scored 0.401 (2026-08-09_1526_gait_tuned/RESULTS.md:22-41).
  - The 2026-08-10 tripod-stability campaign ran 25-27 fine-tune attempts, all nulls (tripod_stability/RESULTS.md:78, :103, :404-479). Its own replay showed the v1 tripod term paid the healthy gait zero (:399-400).
- **Lean/stride fine-tunes** moved pitch by 0.000 at up to 5x dose (2026-08-12_1017_lean_reduction/RESULTS.md).
- **Omni S, P and S_cat fine-tunes of W** moved no stride beyond the seed band (2026-09-26_stroke_posture/CLOSE.md:40-53; 2026-09-27_stroke_cat_W/CLOSE.md:17-42).

**Optimizer state during those fine-tunes** [VERIFIED numbers, C14]:
- **W forks:** LR at the floor 75.9-78.1% of 4000 iterations, never above 3.37e-5 (11% of the configured 3e-4); σ 2.0002 on every iteration; priv_reg_coef 0.1 constant (resume_alg_state true).
- **Era-A short resumes (r01-r14):** 94.4-97.7% at the floor, σ 2.0, priv_reg restarting at 0.
- **Tripod-stability fine-tunes (a1-a7, b1-b7):** only 46-51% at the floor (max 5.06e-5).
- **Phased-flat B0/B1:** started from a 3k base with σ 1.899, not pinned.

**Why the draft's attribution fails:**
1. **Fresh optimizer state did not rescue the same terms.**
   - The stroke and centring terms were re-run in from-scratch formation arms: t3 stroke_formation S_f/P_f, runs 2026-09-26_17-42-44 through 2026-09-27_04-16-39, and stroke_cat_t3 (2026-09-28_16-40-58 and 18-05-55).
   - In those arms σ started at 1.5 and first reached 1.99 at iterations 1578-2119. The LR was at the floor 23-37% of iterations, with peaks of 3.8e-4 to 5.8e-4.
   - They also **FAILED** the stride and posture gates (stroke_formation/GATE_t3.md:3-5, :66, :100; GATE_t3cat.md:3-8).
2. **Optimizer-state interventions other than an LR/σ reset were run.** Adam cleared plus a fresh critic, twice, both negative (2026-08-17_1500_phased_flat/CHANGELOG.md:38-45; lit-review-plasticity-cliff.md:109-116). A reset-σ fine-tune was also run: M1, which was REJECTED but sat at the LR floor 100% of the time. No fine-tune ever ran with a working (fixed, non-adaptive) LR; every one of the 602 runs uses schedule "adaptive".
3. **Fine-tunes under the throttled state did move behaviour:**
   - W's c6x stage (LR at the floor 61-64%, σ 2.0) absorbed the clock removal: episode length −27% at iteration 164, back in band by 450, crab_failure back by 534.
   - It then moved turn facing_closed 0.475 → 0.787 and move_turn survival 0.62 → 0.255 (2026-09-23_clock_deprecation_w2/REPORT.md:147-165).
   - A plain 2000-iteration fork moved move_turn facing 0.235 → 0.411.
   - Era-A r13 (about 95% at the floor) moved tippy and stride.
4. **No ledger raises the LR floor as a cause of any null.**

**Surviving reading.** The LR floor and the σ pin are a real, untested, secondary confound, and only for the specific doctrine that fine-tunes cannot reorganise gait phase structure or posture. They do not explain the side-gait stroke/posture nulls, which are better explained by reward/task design and by the two-hand interference question (Section 4.4(c)). The saturated action parameterisation is common to the fine-tunes and the formation arms (70% of rod means clipped at iteration 300 in the side bisect, ROOT_CAUSE.md:248-252), so no in-repo run separates it.

**Consolidation reduces modifiability** (not re-verified):
- 3k / 8k / 20k base: 2b2 clearance learning alive / partial / frozen (0.08-0.15 / 0.064 / 0.013).
- Actor weight norm 26.3 → 34.6; critic 30.7 → 46.7; C7 critic 57.1. No dormant units (T/crab_hex_forward_task/experiments/lit-review-plasticity-cliff.md:8-38).

**Late degradation, confounded with curriculum** (not re-verified):
- Forward A15+B obstacle completion by lineage checkpoint: 0.68 (5k), 0.66 (20k), 0.29 (25k), 0.48 (30k).
- Golden plant: 0.60 → 0.27. The full-income control also decayed, to 0.19.
- Sideways 2c step onset: pooled −0.17 [−0.24, −0.10] on 2 seeds.
- No clock-held control was ever run.
- The least-shaped heads are often best: the P2 10k no-ramp heads scored 0.73/0.95 vs 0.56 for the ramped lineage head (2026-09-06_2130_a15b_lineage/REPORT.md:101-126).

**Not converged at 12k (omni):** fwd ratio 0.733 → 0.785 over 4000 more iterations. Checkpoints of one run swing by 0.2 in survival (policy/README.md:140-147; WAVE2_METRICS.md:264-280).

**Reward curves:** fresh runs reach 90% of final reward at median iteration 1344-1798 (omni 3000-iteration runs: 1777, IQR 1621-1985). The last third adds 5-15%. Resumed stages reach 90% within 85-147 iterations.

**Warm-start penalty after an action-semantics change:** fine-tune reward 11.4 vs 21.0 from scratch. The fine-tune's value loss peaked at 0.426 (2026-08-13_1900_geometry_velocity_actions/RESULTS.md).

### 4.3 Exploration and action bounds [ALG/HP: the strongest optimizer-side evidence; it concerns the parameterisation, not PPO as a family]

- **σ pinning and mean saturation:** Sections 3.2 and 3.3 [VERIFIED/CORRECTED C05-C07].
- **Chatter.** 2.23 sign flips per dim per second at stand and 1.83 while moving. Stand contact switching is 1.65 Hz, the same as walking. The ledgers attribute stand stepping and the rigid 3-feet pattern to the saturated mean, not to the clock or the reward (memory clock-deprecation-campaign-active.md:84-89). Not re-verified.
- **The action_rate penalty rewards saturation.** Monte Carlo at σ 2: expected \|Δa\| per dim is 0.95 at \|μ\| = 0 and 0.00 at \|μ\| = 9 (lit-review-omni-reward-simplification.md:13-18).
- **M1: σ reset 0.4, cap 0.5, a 2000-iteration seed-3 fork of saturated X08:c6 model_7996, no bound on \|μ\|. REJECT.** [CORRECTED C10]
  - **Outcome.** Block Q z −1.36 vs R 0.32. Action jitter z −2.7 to −6.4 in every category. Guards back.slip_per_m and move_turn.ratio WORSE (REPORT.md:181-255).
  - **σ** climbed from 0.40 back to the 0.50 cap by iteration ~937 and stayed pinned there (zero clamp gradient) for the second half of the run.
  - **\|μ\|.** The fork inherited a median \|μ\| of **8.1** (parent X08:c6). After 2000 iterations the control X01 was at 9.1 and M1 at 9.5 (ANALYSIS_M1_saturation.md:36-40).
  - **LR.** M1's adaptive LR was at the 1e-5 floor in 2000/2000 iterations. The same-checkpoint forks X01 (seed 3, the knob-free control) and X02 (seed 2) were off the floor in 41.55% and 43.05% of iterations.
  - **The draft's "lowering σ also cut the step size" is overstated.**
    - The controls themselves sat at the floor 57-58% of the time and never exceeded 3.375e-5. Their mean LR was 1.42e-5 / 1.45e-5 against M1's 1.00e-5: about a 30% parameter-step difference.
    - Under the repo's KL formula, KL per unit Δμ is Δμ²/(2σ²), which is 16-25x larger at σ 0.4-0.5 than at 2.0. A 100% floor therefore says M1's per-update KL stayed at or above the 0.02 cut threshold. In distribution terms M1's steps were at least as large as the controls'; the schedule simply could not go lower.
    - KL is not logged, so neither reading is measured.
  - **What M1 does and does not show.** M1 is evidence against *lower σ on top of an unbounded, saturated mean, as a fine-tune*. It is not evidence about lower σ from scratch, lower σ with a bound loss, or lower σ on an unsaturated head. The ledger's own diagnosis agrees: "the problem is the saturated mean, not σ as such". The strongest alternative reading, which the ledger also advances, is that shrinking σ exposed the bang-bang mean's switching directly to the actuators.
  - **Other caveats.** X02 is a different seed, so only X01 is a same-seed control. M1's training curves (last-500 episode length 1651, reward 52.5) sit near X02 (1677 / 53.3) rather than clearly below both controls.
  - **M1b** (the bound-loss plan) was written and audited but **never approved or run**. No bound-loss code exists in the repo, and no agent.yaml carries such a knob.
  - **On a two-direction head,** M1 left side speed unchanged (ratio 0.422 → 0.430, z 0.45; gait_motion_quality/REPORT.md:12, :132-136, :163, :181, :194). [C12]
- **Action authority.** Scale 0.4 broke from-scratch formation (tripod 0.199, completion 0.25), and scale 0.3 failed too (0.022). Resuming a formed gait at 0.4 kept the gait, but completion was 0.27-0.44. The ledger reads this as exploration violence proportional to authority (gait_formation_v2/CHANGELOG.md:38-47). A lower-σ, higher-authority arm was never run.
- **Deploy dependence on the clip.** The unclipped student env made the teacher itself fall in 100/100 episodes. With the clip, the teacher reached 0.70 and the student 0.79 (2026-09-07_1330_phase_pipeline/CHANGELOG.md:16-20).
- **Era-A leg snaps.** Up to 195 mm of foot travel in 20 ms; 57% of MR air-time events last a single step (2026-08-08_0950_stride_length_on/CHANGELOG.md:48-56). Consistent with bang-bang control. The C07 census covers only checkpoints with eval npz files; no era-A saturation figure is quoted.
- **Stochastic vs deterministic eval on the same head.** Achieved vx 0.145 vs 0.151 (20k head), 0.169 vs 0.160 (30k head) (2026-09-03_1156_obstacle_exposure/REPORT.md:11-56).
- **Caveat.** Saturation is equal in fast and slow side heads, so it does not select the side-speed class (side_bisect/ANALYSIS.md:93-96, 116-119).

### 4.4 Multi-command interference and gait attractors [mixed]

**(a) Degenerate basins: reward-preferred only for the glide family.** [CORRECTED C13]

*Forward, glide vs target, same reward (holds).*
- **v5 unison glide vs H.** The v5 glide (tripod 0.0 at 2999) reached Train/mean_reward **28.69** (51-iteration window ending at 2999; point value 28.08). Run H, which was alternating at 3k (tripod 0.335), reached **26.15** (point 25.94).
  - The effective reward sections differ only by reward_tripod_schedule (weight 0.15 in v5), whose logged income summed to 0.0025 over 3000 iterations, plus a weight-0.0 pitch term. The agent.yaml files differ only in max_iterations.
- **Glide seeds 6 and 7** (v5 active) reached 31.72 and 29.53 (point at 2999) vs 27.67 for the symmetric mirror-loss reference, whose env.yaml is identical except for the seed and tripod weight 0.0.
- **Seed 4's glide** (it falls) reached 27.12: above H, below the symmetric reference.
- **Caveats:**
  - These comparisons are iteration-matched at 3k only. The glides were never trained further, while the references kept climbing (H 29.44, symmetric 31.05 at 19999). That exceeds v5's and seed 7's 3k values but not seed 6's, so the asymptotic ordering is unknown.
  - The symmetric reference's gait at 3k was never evaluated.
  - All of these are stochastic-policy returns at σ 1.93-1.99.
- Sources: TB runs under T/crab_hex_forward_task/experiments/{2026-08-09_1526_gait_tuned, 2026-08-10_0058_tripod_stability/fromscratch_tripod_v5_crossing_short, 2026-08-12_1550_seed_basin_search/seed*, 2026-08-13_0035_mirror_symmetry/fromscratch_sym_20k}; S/c13_escalate/tb.py.

*Forward, every other documented degenerate family earned LESS than the target at matched iteration, under the same or near-identical reward.*
- v1 unison lunge: 25.03 at 3k / 26.45 at 20k, vs H 26.15 / 29.44.
- v2 tip-and-rock unison: 24.58; v3: 22.98; v3b: 21.95; v4: 22.36.
- Seed-basin shuffles and handed gaits, seeds 2/3/5/8: 23.77 / 20.43 / 21.39 / 24.50, vs 27.96 (symmetric) and 26.15 (H).
- For these families the reward favours the target. Their persistence is **basin selection**, which an exploration or optimizer change could in principle affect.

*Omni 1c creeper: the draft's numbers were mis-sourced.*
- **"0.96/s vs 2.23/s on replay" is not a replay.** It is training-log per-second income (summed Episode_Reward/* over the last 50 iterations: 1a 0.962, 1c 2.180).
- **The two stages have different reward functions and command samplers:**
  - 1c adds reward_tracking_yaw at weight 0.2, which paid 0.180/s.
  - slot_probs change 0.2/0.8/0/0 → 0.15/0.55/0.1/0.2.
  - dir_band changes (0, 0) → null.
- **"61 → 93 while track ratio 0.747 → 0.245" spans stages.**
  - The valid within-1c statement: from iteration 2248 to 6997, mean reward rose 61.3 → 93.0 while track_ratio fell 0.411 → 0.242, with mean episode length only 1887 → 1995.
- **The genuine same-terms replay** (1c weights, fwd hold; T/crab_hex_omni_task/experiments/2026-09-16_1805_omni_formation/replay_reward_economy.md:43-45, :89, :235):
  - Net income: walker 0.095/s vs creeper 1.503/s.
  - But the walker's gross positive income is marginally higher (2.630 vs 2.574; tracking margin +9.9% for the walker). The ordering is decided by the penalties: foot_idle −1.747 vs −0.441, action_rate −0.516 vs −0.198.
  - Excluding action-rate, the walker earns 2.11 vs 1.93.
  - fwd is the only hold with a genuine walker.
- The source analysis itself calls the creeper a local-basin result, "not a global optimum" (ANALYSIS_1c_failure_2026-09-18.md:8-12, :105-106).
- **Reading.** A reward/regularizer problem, partly one of penalty terms that tax motion. The action_rate term also rewards saturation (Section 4.3).

*Other.*
- A fall-and-spin reward hack (2026-08-13_2000_onedir_spin/CHANGELOG.md).
- OS11: PPO delivers the added-reward dose as predicted (r 0.67-1.50) but buys cheaper behaviour: taps +0.24/s, lean +0.010-0.036 m, stride per unit speed down.

**(b) Basin entry fixed by structure [STRUCTURE].** Not re-verified.
- Clock plus RSI 0.2 from a high-quality 2330-state bank flipped the failing seed 2 to tripod 0.496/0.537. A low-quality bank failed (tripod 0.0).
- The never-clocked Z failed formation on 2/2 seeds (c6 survival 0.00-0.46).
- The reversal-penalty dose-response moves the basin only at doses that collapse income (w −1.0: completion 0.0).
- **Reading:** exploration and structure. No algorithm-side exploration change was ever tried.

**(c) Omni two-direction side walking [STRUCTURE as observed; the optimizer's role is untested].** [CORRECTED C12]

*What the ledger supports.*
- **The SLOW/FAST split follows one variable:** whether a single network must serve both lateral directions.
- **Every network trained on both hands is SLOW or DEAD,** at feet_ss **0.028-0.103 m/s**. The SLOW rows span 0.061-0.103; the DEAD rows reach 0.028 (LR4_dwell R) and 0.041-0.042 (C_lr_10_rsi L).
  - This covers real −y training: LRfa, LR4, LR4_dwell, C_lr, C_lr_10, C_lr_rsi, C_lr_10_rsi, C_lr_rsi33 and W.
  - It also covers the phantom −y rows that the 4-image detached mirror loss injects into +y-only runs: O, R, C_band and C_rsi.
- **+y-only runs with the fa mirror and scalar command obs** are FAST in **4 of 5 runs**, at 0.184-0.217 m/s. C_mo seed 3 is INTERMEDIATE at 0.162/0.173.
- **The exact M_y canonicaliser** makes all 4 fast heads FAST on both hands with no retraining: 0.182-0.217 m/s, \|L−R\| ≤ 0.006 (wave mode; cmd mode agrees; gate/canon_lr.json). Canonical C degrades at 0.35 m/s (about 0.15).
- **PPO telemetry overlaps completely between the classes** (S/omni_stroke_side/tb_arms.json, 17 arms):
  - σ 1.58-1.70 at iteration 300 and 1.94-2.00 at 2990;
  - LR 3.5e-5 to 6.4e-5 at iteration 300, with 31-41% of iterations at the floor;
  - entropy 33.7-35.1 at 300 and 37.5-38.0 at the end;
  - saturation 91-98% at eval; clipped rod-mean share 0.59-0.73 at iteration 300 and 0.94-0.96 at 2999.
- Sources: side_bisect/ANALYSIS.md:10-30, 50-64, 93-96; ROOT_CAUSE.md:327-344.

*What was overstated.*
- **"Not the PPO hyper-parameters" was never tested.**
  - All 20 X-runs share byte-identical PPO settings (verified by diffing params/agent.yaml). Only the mirror function, seed and episode length differ.
  - So the data show that the hyper-parameters do not *select* the class among these arms. They do not show that different settings could not let a two-direction network form the stroke.
  - The campaign's own ROOT_CAUSE.md:48-66 says why PPO fails to build the direction-flipped stroke in a two-hand network is "NOT established". Two of its three candidates are optimisation-dynamics stories: gradient cancellation across hands, and early lock-in with 70% of rod means already past the clip at iteration 300, before the classes separate at about iteration 540. It also notes RSI is inert at init σ 1.5 (:213-218).
  - Exploration noise, the entropy coefficient, a bounded-mean parameterisation and obs normalisation remain **untested levers, not excluded ones**.
  - The only σ variation ever applied to a two-direction network (M1) was a post-formation fine-tune, and it left side speed unchanged.
- **"Interference" holds only as a learning-signal statement.** Representational capacity is ruled out: a fresh network of the deployed architecture distils the two-hand fast policy to 98.9-99.0% sign agreement with the full stroke (0.585 vs teacher 0.583) in 40 CPU epochs (ROOT_CAUSE.md:183-190). The deciding pair C vs C_lr also halves the +y sample share (80% → 40%) and makes the direction visible; the C_40 and C_fa controls have not run.
- **The reward point was overstated.** +0.396/s is a cross-task endpoint comparison (side_speed_investigation/REPORT.md:97-107).
  - The same-task replay supports its direction: fast +y heads earn +0.18 to +0.45/s more than O per side-hold second.
  - But whole-run training returns are about equal (at iteration 2990: O 68.0 vs C 60.9, C_s2 65.9, C_mo 72.4, C_mo_s2 66.6, C_mo_s1 74.4).
  - The training-average margin is +3-12% and can reverse with foot_idle, and the onset lean is paid +0.23/s for about 2 s inside the γ = 0.99 horizon.
  - The defensible statement is "reward does not select the class" (C and C_lr share a reward and split).
- **All two-direction and phantom negatives are seed 3 only.** X20 (C_lr_10_rsi33) was still training at compile time.

**(d) Command additions and staged curricula [REWARD/TASK + CONFOUNDED].** Not re-verified.
- Turning and yaw income added at round 0 cost the clock (0.77x / 0.78x); both passed one round later. speed_band never passed in 4 rounds (gated_lineage/REPORT.md:10-53).
- The staged omni chain lost translation: fwd ratio 0.76 → 0.16 at the 1a → 1b switch.
- C1/C1b curricula dropped turn facing 0.419 → 0.185 / 0.171.
- C1d (turn-first) was adopted by user overrule. C1e was REJECTED: diag survival 0.79 → 0.49 (facing_tracking_ladder/REPORT.md).
- Retention methods (EWC, kickstarting, BC regularisation) were never tried.

**(e) move_turn.** Every head reaches 0.02-0.07 rad/s against 0.3 commanded. The ledger computes that the outer foot would need about 0.58 m/s, which is kinematically infeasible (facing_tracking_ladder/CHARTER.md:41-53). [REWARD/TASK]

### 4.5 Symmetry [an algorithm-side lever with large but poorly replicated effects]

**[CORRECTED C15]** All the headline numbers and implementation facts check out. The draft's seed accounting and its list of costs were wrong, and the mirror loss is not the only algorithm-side lever with large effects.

**Implementation** [VERIFIED]:
- The loss is loss-only (use_data_augmentation False in all runs).
- It acts on raw `act_inference` means (no output tanh; clipping to ±1 happens only in the vec-env wrapper).
- The target is detached (PPO file :434, :441, :451, :454; forward rsl_rl_ppo_cfg.py:134-145).

**Effects and their seed support.**

*Position era, 20k.*
- Tripod 0.401 → 0.517 with coefficient 0.5, duty balanced in 10/10 episodes.
- Mirror-on is one run (seed 1). The mirror-off side is the seed-1 baseline, plus 7 v5-active 3k seeds (2-8) with 0/7 alternating. Those are confounded by v5 and are not a matched 20k arm.
- The gait survived 2200 mirror-free teacher iterations (2b2 0.5255). (2026-08-13_0035_mirror_symmetry/RESULTS.md; fromscratch_sym_20k/gait_eval_final/.../scenario_metrics.json)

*Velocity era, 3k* (seed 1 in both arms, identical iteration-0 reward):

| mirror coefficient | outcome |
|---|---|
| 0 (C6a) | tripod 0.032, slip 3.6%, reward 19.9 |
| 0.5 (C2) | tripod 0.573, slip 11.3%, reward 27.1 |
| 1.0 (C6b) | tracking broken (deficits to −0.40) |

- A re-roll of the same config and seed with coefficient 0.5 (the C1+C2 20k adoption run) got **tripod 0.0**; the ledger blames GPU nondeterminism.

*Velocity era, 20k.*
- Coefficient 0 beat 0.5 on tracking (PASS vs FAIL, −0.14), slip (4.1% vs 16.7%) and tippy (9.9% vs 27.9%).
- **Tripod was 0.0 in both arms**, so the 3k gait gain did not persist (2026-08-15_1530_task1_velocity/CHANGELOG.md:57-61, :93-137).

*New plant.*
- E5 (mirror off, seed 1) scored 0.257 / completion 0.20, vs E1 (mirror on, seed 1) 0.541 / 0.99.
- The mirror-on arm has **4 seeds**: 0.541, 0.0, 0.026, 0.0. E5 beats 3 of the 4 (gait_formation_v2/CHANGELOG.md:23-28; E1_seed*_eval.log).

*Omni.*
- Switching the image set from 4-image to fore-aft only (fa) plus a scalar command obs took +y-only side speed from SLOW (O ratio 0.387, feet_ss 0.084/0.094) to FAST in 4 of 5 runs across 3 seeds (C_mo ratio 0.621 / 0.852 / 0.755 for seeds 3 / 2 / 1). This is multi-seed.
- It is confounded with H6 (M_x weight) and the command-obs change. No no-mirror +y arm has run.
- Once −y is genuinely sampled, the mirror type makes no difference: LRfa and LR4 are both SLOW (side_bisect/REPORT.md:43, :159, :311, :349; GATE.md:61-65, :210-223).

**Other algorithm-side levers with large effects** (so "the only" is false):
- A critic reset cut tripod 0.601 → 0.450 (phased_flat B1).
- A critic reset at hand-off took bridge failure from about 0.90 to 0.25 (teacher_handoff chain #4).
- The student clip_actions fix took the teacher on the student env from 100/100 falls to 0.70 completion. That was a bug fix, not a design comparison.

**Current use** (not re-verified):
- Mirror loss magnitude: median 0.026 (× 0.5 = 0.013) vs \|surrogate\| 0.008. It never decays: first-100 median 0.024, last-100 median 0.026. Omni 4-image about 0.047-0.059; `fa` about 0.028-0.035.
- The coefficient was only tested at {0, 0.5, 1.0}, on the older plant.
- Never run: data augmentation, hard-equivariant networks, a mirror phase-out, and a loss on clipped or tanh means.

### 4.6 Curriculum, resume and critic [implementation (fixed) + critic scaling; CONFOUNDED]

Not re-verified except where marked.
- **Resume reset.** Before the fix, each resume restarted priv_reg, the LR and the auxiliary optimizer states. The reset is still visible on 2026-09-23 morning [C14].
  - Heads with an open latent gap (about 0.035 vs 0.0004) turn at facing 0.17-0.25, while closed-gap heads reach 0.29-0.75.
  - This is collinear with the curriculum; the separating control was never run (facing_tracking_ladder/PLAN_AMENDMENTS.md:262-296).
  - On W the history latent is constant (C16), so whatever "latent gap" measures, it does not carry per-state information on the head of record.
- **Post-fix re-run.** X0 vs pre-fix X08:c6: side ratio 0.396/0.389 vs 0.430; F_mt improved 0.235 → 0.330/0.365. Not a clean A/B.
- **Teacher hand-off:**
  - Chain #3: bridge failure 0.00 → 0.90. The flat critic priced about 0.5/step against about 0.0015/step on the bridge.
  - Chain #4, with critic reset and Adam clear: the bridge gated at failure 0.25, but 2b1 degraded 0.58 → 0.91 at the next boundary.
  - Chain #5 (full goal/yaw dose): failure 0.00 → 0.96 (2026-08-26_0200_teacher_handoff/CHANGELOG.md:15-20).
- **Critic reset elsewhere:**
  - Phased_flat B1 (flat → flat): negative (tripod 0.601 → 0.450).
  - On the 2b2 v2 base: falsified (clearance 0.010 vs 0.013) (phased_flat/CHANGELOG.md:38-46; lit-review-plasticity-cliff.md:107-117).
- **Order effects.** Elements that were catastrophic at round 0 passed one round later: terrain50 clock 0.52x → 1.03x (gated_lineage/REPORT.md).
- **Never tested:** value or return normalisation (PopArt), value-clip off, a separate critic LR, γ 0.995, and removing the reward clip. Section 3.4 has the value-loss magnitudes.

### 4.7 Distillation [not a bottleneck]

Not re-verified.
- **Student 3a vs its 2c teacher** (completion/falls): flat 0.79/21 vs 0.81/19; step 0.71/29 vs 0.68/32; obstacles 0.64/36 vs 0.66/34. 3b (+5k iterations on the hard band) gave no change. Single seed (2026-09-07_1330_phase_pipeline/REPORT.md:47-57, 76-88).
- **The only failure was the clip plumbing bug.**
- **The old-plant student was weaker than its teacher:** tripod 0.161 vs 0.237 (phased_flat/CHANGELOG.md:283-314).
- **Distillation compute:** 66.2 GPU-h over 12 runs; about 6.7 s per iteration at 192 envs (607-676 fps).
- **Distilling per-direction specialists** is the ledger's named fallback for omni side walking (about 150-200 lines, never run). Its feasibility in principle is supported by the CPU distillation result in Section 4.4(c).

### 4.8 Attribution scoreboard

| Failure class | Example (numbers) | Ledger attribution | Does optimizer telemetry separate good from bad? | Merged reading |
|---|---|---|---|---|
| Degenerate reward-preferred gait | Glide 28.69 vs tripod 26.15 at 3k [C13] | Reward economy | n/a (same regime) | REWARD/TASK, **for the glide family only** |
| Degenerate, *not* reward-preferred | Lunge, tip-and-rock and shuffles earn 1-7 less than the target [C13] | Basin / exploration | n/a | Basin selection. ALG/HP-exploration fixes untested |
| 1c creeper | Within 1c, reward 61 → 93 while track 0.41 → 0.24; the penalties set the replay ordering [C13] | Reward economy | n/a | REWARD/TASK (regularizer design) |
| Early basin lottery | 1/4 plain-clock seeds form; 0/7 seeds alternating | GPU nondeterminism, then structure (clock/RSI) | Weak correlation: earlier σ pin in failing runs (n=11) | STRUCTURE, plus possibly exploration (ALG/HP untested) |
| Two-direction side slowness | 0.028-0.103 vs 0.184-0.217 m/s [C12] | Shared-network interference | No (complete overlap) | STRUCTURE as observed; the learning-signal mechanism is open; **HPs never varied** |
| Chatter, stand stepping, 3-feet pattern | 97% saturation; 2.23 flips/s | Saturated bang-bang mean | Yes: every head is saturated (not a within-sample separator) | ALG/HP (parameterisation) |
| Fine-tune inertia | W forks stride flat; tripod-stability nulls | Basin lock-in | Fine-tunes at LR floor 46-98%, σ 2.0; **fresh-state formation arms failed too** [C14] | Mostly REWARD/TASK. Optimizer state is a secondary confound for phase-structure reorganisation only |
| Late-window regressions | Obstacles 0.66 → 0.29 | Curriculum / clock anneal | LR floor share rises with window | CONFOUNDED |
| Stage-boundary collapse | Bridge failure 0.00 → 0.90 | Critic scale mismatch | n/a | Critic/return scaling (value-side HP) |

---

## 5. Hyper-parameters and algorithm knobs

### 5.1 Ever varied

| Knob | Values | Outcome | Source |
|---|---|---|---|
| mirror_loss_coeff | 0 / 0.5 / 1.0 | **[CORRECTED C15]** 0.5 formed the tripod in seed-1 pairs: 0.517 vs 0.401; 0.573 vs 0.032; 0.541 vs 0.257. But a same-config re-roll at 0.5 got 0.0, mirror-on E1 seeds 2-4 got 0.0 / 0.026 / 0.0, and at 20k both velocity arms ended at tripod 0.0. 1.0 broke tracking. At 20k in the velocity era, 0 beat 0.5 on tracking and slip. 0.5 has been baked since 2026-08-13 | mirror_symmetry CHANGELOG; task1_velocity CHANGELOG:57-61, :93-137; gait_formation_v2 CHANGELOG:23-28 |
| Mirror images | 4-image vs `fa` | +y-only with fa and scalar obs: FAST in 4/5 runs (3 seeds). Two-direction: SLOW with either. Confounded with H6 and the obs change | side_bisect GATE.md:128-141, 210-223; ROOT_CAUSE.md:149-171 |
| σ cap / reset (KRABBY_ACTION_STD_MAX / _RESET) | 2.0 vs 0.5 cap with 0.4 reset (M1; plus 2 smokes) | **[CORRECTED C10]** REJECT on a saturated fine-tune. It does not test lower σ from scratch or with a bounded mean. LR at the floor 100% vs 57-58% in the controls (mean 1.00e-5 vs 1.42-1.45e-5); KL is unlogged, so step-size effects are unmeasured | ANALYSIS_M1_saturation.md; REPORT.md:181-255; TB X01/X02/X03_c6ft |
| entropy_coef | 0.01; 0.005 in one 20-iteration smoke | Effectively never varied [VERIFIED C08] | runs.csv |
| learning_rate (configured) | 3e-4 / 2e-4 / 1e-4 / 3e-5 by mode | Only the starting value of the adaptive schedule; erased within iteration 0 in ~94% of fresh runs [CORRECTED C02] | S/esc_C02; rsl_rl_ppo_cfg.py:53, :74-89 |
| init_noise_std | 1.5 (383 runs) vs 0.65 (55) | Resumes load σ ≈ 2.0; no from-scratch low-init comparison exists | TB Policy/mean_noise_std; S/verify-C08-esc/rows.json |
| clip_actions | 2.0 → 1.0 (flat walk, afbc0a46, no rationale); student None → 1.0; None in 4 early teacher runs | None on the student env made the teacher fall 100/100; 1.0 is load-bearing | git afbc0a46; phase_pipeline CHANGELOG:16-20 |
| Action scale (KRABBY_ACTION_SCALE) | 0.25 / 0.3 / 0.4 | 0.3 and 0.4 fail formation from scratch | gait_formation_v2 CHANGELOG:38-47 |
| Cam lower clip (KRABBY_CAM_CLIP_LO) | none / 0.1 | Dead | gait_formation CHANGELOG:92-102 |
| Critic reset + Adam clear at a stage boundary | on / off | Mixed: positive at one teacher bridge (0.90 → 0.25); negative in B1 (0.601 → 0.450); falsified on v2 | teacher_handoff CHANGELOG:17-19; phased_flat CHANGELOG:38-46 |
| resume_alg_state | reset vs carried (eabe7324) | No clean A/B | memory ppo-counter-reset-on-resume.md |
| RSI fraction / bank | 0 / 0.15 probe / 0.2 good bank / 0.33 | A good bank closes the formation lottery. It does not fix the two-direction side gait; RSI is inert at init σ 1.5 per ROOT_CAUSE.md:213-218 | gait_formation_v2 CHANGELOG:25-31; side_bisect |
| Training length / fine-tune vs from scratch | 1k-30k; 2000- to 6000-iteration fine-tunes | Sections 4.2 and 4.4 | many |
| Seeds | 1-8 | Section 4.1 | many |
| num_envs | 256 (RL); 192/256 (student); 64 (smokes) | Never varied for RL training; above 256 never run | S/verify-C08-esc; S/esc_C09 |
| Episode length / command resample | 20 / 40 / 70 s; resample 6 / 10 / 40 s | 40 s adopted; 70 s and one-command episodes harmful | morph CHANGELOG:108-123; obstacle_exposure CHANGELOG:209-220; side_bisect |
| Tracking kernel σ² (a reward width, not the policy σ) | 0.02 / 0.1 / 0.25 | 0.1 baked | gait_formation CHANGELOG:67, 83, 106 |
| Command obs scalar vs visible; command pool | side_bisect arms | Section 4.4(c) | side_bisect GATE |
| Smoothness knobs M2/M3/M7; CAPS; bound loss (M1b) | Prepared or planned | Never run; M1b never approved [C10] | MQ REPORT:9-18; reward_simplification CANDIDATES:21, :414-427 |

### 5.2 Never varied in any PPO run

[VERIFIED C08] Checked against the 438 runs under P/logs/rsl_rl; the archive is consistent.
- desired_kl (0.01)
- schedule (adaptive, per minibatch)
- LR floor (1e-5) and ceiling (1e-2)
- num_learning_epochs (5)
- num_mini_batches (16)
- clip_param (0.2)
- γ (0.99)
- λ (0.95)
- value_loss_coef (1.0)
- clipped value loss (on)
- max_grad_norm (1.0)
- num_steps_per_env (24)
- entropy_coef (0.01, except one smoke)
- num_envs for RL training (256)
- empirical_normalization (off)
- noise_std_type (scalar, state-independent) and the clamp *form*. The ceiling value varied in 3 runs.
- action distribution family (Gaussian, clipped at ±1 by the wrapper; 4 early teacher runs had wrapper clip None)
- Adam hyper-parameters
- network sizes and activation
- critic LR separation (shared)
- advantage normalisation mode
- reward ≥ 0 clip
- time-out labelling
- mirror as augmentation vs loss
- algorithm family (no SAC/TD3/other ever run)

---

## 6. Constraints on alternatives

**Hardware.**
- One RTX 5080 (16,303 MiB); Ryzen 9 9950X3D (32 threads); 59 GiB RAM.
- Another session is usually training.
- A single 256-env run uses 61-69% GPU utilisation, 66 W of 360 W and 4.4 GB VRAM [VERIFIED C09].
- Two parallel 256-env runs each get about 2,780 steps/s, 97% of solo in aggregate. This is consistent with time-slicing, and MPS was never tried.

**Budget.**
- About 528 GPU-h were used over the census logs (2026-08-04 to 2026-09-30). The archived August runs (≈ 3.3B env steps) are not included.
- Arm costs:

  | arm | cost |
  |---|---|
  | 3000-iteration from-scratch arm | about 1.1-1.2 GPU-h (18.4M samples) |
  | 5000-iteration window | 1.8-2.6 GPU-h |
  | W lineage (12k iterations) | about 5.4 GPU-h per seed |
  | 3 arms × 3 seeds × 3000 iterations | about 10-11 GPU-h |

- The early side read at iteration about 1000 costs about 0.4 GPU-h per arm.

**Env count.** [VERIFIED C09]
- 256 envs is a CLI choice made from one 2026-08-12 measurement.
- Per-iteration wall time at ≤ 256 envs is latency-bound: +5% wall time for 4x the envs in the clean omni pair.
- 512-4096 envs in one process may add samples almost for free, but no run above 256 exists.
- Physics scaling of the 230 kg plant with height scan beyond 256 envs is unknown.
- The depth student is VRAM-bound (OOM at 256; runs at 192).

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

**Libraries.** [VERIFIED C16]
- isaac_venv has only rsl-rl-lib 5.0.1 (PPO, Distillation). Its own PPO cannot take this policy: it requires MLPModel actor/critic and TensorDict obs groups (PPO file :45-49) and has no std clamp or bound loss.
- tensordict 0.13.0, torch 2.7.0+cu128 and gymnasium 1.2.1 are present.
- isaaclab_rl 0.5.2 ships wrappers for skrl, rl_games and sb3, but none of those libraries is installed. Neither are torchrl, tianshou, sample_factory or jax.
- Any switch needs a pip install plus porting of the RMA-style actor.

**Runner and checkpoint contract.** [CORRECTED C16]
- The runner raises ValueError for any algorithm `class_name` other than PPOWithExtractor or DistillationWithExtractor (runner :63-68). OnPolicyRunnerCrabHex overrides only the policy-class resolver (crab_on_policy_runner.py:9-15).
- The save/load dict has fixed keys: model_state_dict (actor.*, critic.*, std), estimator_state_dict, optimizer_state_dict, iter, alg_state, depth_* (runner :682-713).
  - `load` requires `estimator_state_dict` with no fallback (:713).
  - The actor loads its priv_encoder/history_encoder keys strictly (crab_actor_critic_with_encoder.py:63).
- **Every deploy consumer substitutes the estimator output and runs the history encoder** (hist_encoding=True):
  - exporter.py:112-130, :157-160
  - play.py:194, :208
  - evaluation.py:130, :166, :179
  - play_stream_video.py:217
  - eval_crab_hex_gait.py:999, :1025, :1022-1025
  - Also: eval_crab_hex_gait.py reads policy.std for --sample-actions (:730-744) and has the canonicaliser hook (:567-576).
- DistillationWithExtractor deep-copies `policy.actor` (distillation_with_extractor.py:50/58).
- Other consumers: bundle_policy.py, policy_summary.py and the policy/ manifests.
- **The constant estimator outputs are load-bearing for existing checkpoints:** zeroing them on W gives KL 17.4 and an executed change of 0.065.
- **Therefore:** a swap can drop the estimator, DAgger and priv_reg *training losses*, but it must either emit stub modules into the schema (an estimator returning the constants, and a history-latent path) or re-plumb all of these consumers consistently.
- The phase chains fork via --resume / --checkpoint.
- Unknown algorithm keys are dropped with a warning (PPO path) or silently (distillation path) (D14).

**Telemetry and CLI contract.**
- TB tags parsed by the experiment tools (side_bisect/scripts/analysis/tb.py, replay_*.py, c7_design):
  - Loss/learning_rate, Loss/entropy, Loss/symmetry, Loss/priv_reg(_coef), Loss/surrogate, Loss/value_function
  - Policy/mean_noise_std (the raw parameter)
  - Train/*, Episode/*, Metrics/*
- AGENT_DUMP_KEYS in run_candidates.py.
- train.py flags (--task --num_envs --resume --checkpoint --max_iterations --run_name --seed), called by run_phases.py and run_candidates.py.

**Deploy path.**
- The exporter serialises only the actor, plus the normaliser, estimator and depth encoder in the deploy variant. There is no algorithm state in it (P/scripts/rsl_rl/exporter.py:112-150).
- A same-shaped actor trained by any algorithm exports identically, provided the estimator/history-encoder contract above is satisfied.
- Must keep:
  - the obs slicing (75/132/9/41/10 as built), or change it consistently everywhere;
  - action dimension 18;
  - executed-action semantics. The clip at ±1 is load-bearing (the teacher fell 100/100 without it); a tanh or Beta head must preserve the executed range end to end.
- The 50 Hz policy vs 100 Hz hardware loop question is open.

**Teacher/student chain.**
- The distiller regresses the teacher's `act_inference(obs, hist_encoding=True, scandots_latent)` raw means (median \|μ\| about 21 in the forward lineage).
- A new teacher must keep an Actor-compatible module. With a squashed actor, the distillation target should be the executed action.

**A swap must re-implement:**
- the mirror loss (baked into every head of record) or an equivalent augmentation / equivariance;
- the exploration knobs;
- alg_state across resumes;
- telemetry tags;
- correct terminal vs time-out labels (D9; critical for off-policy bootstrapping);
- a decision on the reward clip;
- the checkpoint contract above.

**Algorithm-agnostic components** (live in the env or wrapper): RSI, curricula, command samplers, reward terms and the canonicaliser.

---

## 7. Open questions for the literature review

Numbers updated to the verified values.

1. **Per-minibatch adaptive KL at small batches.** Is the rsl_rl-style rule (÷1.5 or ×1.5 per minibatch, floor 1e-5) documented to drive the LR to the floor at small minibatches (384 samples, 80 steps/iteration)? Here it logs 1e-5 at step 0 in 94% of fresh runs.
   - What LR trajectories do legged_gym and Isaac Lab report at 98k-147k-sample batches?
   - Is a per-epoch or per-iteration KL check, or a fixed or linearly decayed LR, better supported for locomotion PPO?
2. **Clipped Gaussians with σ at or above the bound.** Is mean drift beyond the action bounds (here \|μ\| median 11 on W, 20-36 on forward heads), with σ at or above the bound, a known pathology?
   - Relevant sources: Chou et al. 2017 (Beta policy); Fujita and Maeda 2018 (clipped action policy gradient); Andrychowicz et al. 2021 (tanh vs clip, initial std 0.5); the rl_games bounds loss (verify it exists and its coefficient).
   - What effect sizes on motion smoothness and learning speed have been measured for locomotion?
3. **KL on the executed distribution.** Should the KL constraint and the log-prob be computed on the executed (censored or squashed) distribution? On saturated heads here only 11-16% of the Gaussian KL is behavioural, vs 70-79% during formation. Is there evidence that doing so restores learning signal late in training?
4. **σ parameterisation.**
   - Evidence on state-independent log-std vs scalar std with a hard, zero-gradient clamp.
   - init_noise_std 0.15-1.0.
   - entropy_coef 0 vs 0.005-0.01.
   - Target-entropy / SAC-style temperature inside PPO.
   - Does any stack use a zero-gradient clamp, and what happens to σ there?
5. **Batch geometry at low env counts.** For 256 envs × 24 steps, what minibatch count, epochs and rollout length do small-batch PPO studies recommend?
   - Relevant sources: Andrychowicz 2021; Rudin et al. 2022 on batch size and env-count scaling.
   - Does a rollout of 48-96 steps help when the gait period (1.2-2.8 s) exceeds the rollout (0.48 s)?
   - What do γ 0.995 or λ 0.97 do at 50 Hz?
6. **Env-count scaling.** Is Isaac Lab sim throughput typically latency-bound at 256 envs for articulated robots with height scans (here +5% wall time for 4x envs)? What speedup is typical from 256 to 1024-4096 envs on a 16 GB GPU?
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
    - Is a shared MLP known to fail to learn a direction-flipped, phase-locked stroke for mirror-related commands without equivariance, even when it can represent it (here it distils to 99% sign agreement)?
    - Evidence for canonicalisation / frame averaging, hard-equivariant networks (MS-PPO, Su et al. IROS 2024), mixture-of-experts or multi-head policies, and gradient-conflict methods (PCGrad, CAGrad) in locomotion.
    - Do exploration or parameterisation changes affect such failures?
11. **Symmetry method.** Mirror loss vs augmentation vs hard equivariance. Consequences of a detached target and of applying the loss to raw vs bounded actions. Recommended coefficients. Does a mirror-loss gait advantage fail to replicate across seeds?
12. **Plasticity.** Does loss-of-plasticity work (Dohare, Lyle, Nikishin; shrink-and-perturb; L2-init) apply to PPO fine-tunes at LR of about 1e-5 to 3.4e-5 with σ pinned? Here, fresh-state formation arms failed the same reward-shaping tests, so the question is narrower than "are fine-tune nulls expected".
13. **Value side.** Value or return normalisation (PopArt) under staged curricula with non-stationary reward scales. Effects of the legged_gym "only positive rewards" clip on learned behaviour. A separate critic LR.
14. **Bounded teachers and distillation.** What target should a DAgger or behaviour-cloning student regress when the teacher is bounded (tanh or Beta)? What happens when the teacher's means are saturated?
15. **Per-dimension KL scaling.** Does per-dim scaling of desired_kl (18 vs 12 action dims) matter in practice?
16. **PPO vs off-policy on bang-bang-prone actuators.** Is there any comparison on velocity-controlled cams or slew-limited actuators, where smoothness penalties (an action_rate penalty that rewards saturation) interact with the action distribution?

---

## 8. Contradictions between miners and verifiers, and how they were resolved

1. **LR floor.**
   - Memory note: "pinned at 1e-5 from iteration 0 for the whole run". Infra-04: about 99.8% at the floor, from 9 sampled points. Telemetry, impl_audit and the omni miners: "oscillates".
   - **Resolution** [C02/C03]: 62.9% of census iterations at the floor (60.2% overall).
   - Fresh runs are at the floor at step 0 in 240/254 cases, then log large excursions early: mean 5-6e-5, 20-24% of iterations above 1e-4 over iterations 0-250. They settle at 1e-5 to 7.6e-5 after iteration ~900.
   - The draft's "at the floor at iteration 0 in 396/437" conflated first-logged iterations of resumes with iteration 0.
2. **Mean saturation 96%: verified?**
   - **Resolution** [C07]: settled by direct measurement on recorded pre-clip means, which no longer depends on the reconstruction. W is 97.0% (the reconstruction said 96.6%), and every checkpoint at 999+ iterations is 91-100%.
3. **"One 256-env run saturates the GPU" vs "collection is latency-bound".**
   - **Resolution** [C09]: the clean same-config pair gives +5% wall time for 4x envs, and live readings show 61-69% utilisation and 66 W.
   - The two-process test (aggregate 97% of solo) fits time-slicing as well as saturation.
   - Above 256 envs is untested.
4. **Student distillation hyper-parameters.**
   - fwd-sep-side FS9 reported "LR 2e-4, 4 minibatches". impl_audit reported "fixed 1e-3, one Adam step per iteration".
   - **Resolution:** the code wins for impl_audit (distillation_with_extractor.py:50-66). The distillation runner branch ignores unused keys silently [C16].
5. **Same-seed reproducibility.**
   - **Resolution** [C11]: 2 of 4 re-checked same-seed pairs reproduce exactly and 2 diverge. The diverging pairs share a nondeterministic history-latent signature (D18). Seed-matched paired designs cannot be assumed to isolate a knob.
6. **Side-speed attribution.**
   - **Resolution** [C12]: the class is structural as observed (one network, both hands). Capacity is ruled out; the learning-signal mechanism is open; PPO hyper-parameters were never varied.
   - The draft's "[STRUCTURE]; an algorithmic escape is untested" stands. Its "optimizer telemetry is the same, so not an optimizer problem" does not.
7. **"Critic healthy" (telemetry T12) vs "critic throttled, effect unmeasured" (impl_audit D5).**
   - Not contradictory. Value loss is small relative to the value scale, but returns are unnormalised and explained variance is not logged.
8. **σ reaching 2.0.**
   - **Resolution** [C05/C06]: from scratch the cap is reached at a median of iteration 2444 (87 runs), not 2435 (88: one 0.5-cap smoke was miscounted). σ is about 1.62-1.70 while side gaits form (iterations 300-600). Resumed stages start at 2.0.
   - The draft said the entropy bonus "pushes σ into the clamp". It pushes only up to the clamp; above it every σ gradient is zero.
9. **Resume fix status.**
   - The memory note says the fix is unmerged. It is on four feature branches, but not on main.
   - The priv_reg reset is still visible on 2026-09-23 morning, and carry-over first appears at 17:14 that day [C14].
   - The 15/33 anomaly remains [UNRESOLVED].
10. **Symmetry implementation.**
    - lit-review-hexapod-reward-stability.md:124-126 says "data augmentation plus mirror loss". The code and all agent.yaml files say mirror loss only. The code wins [C08/C15].
11. **Velocity-era tracking "flatline at about 0.25 for 17,500 iterations"** (onedir_spin RESULTS).
    - It holds for one run only; the identical-config screen reached 0.735-0.752. The ledger's structural mechanism is weakened (fwd-aug-mid F8; not re-verified).
12. **Omni 1_t3 provenance.**
    - policy/README gives sha 03b87cd3 for X08_t3, while PLAN_AMENDMENTS 40 lists 03b87cd3 vs f3783ec5 for X09 vs X08. Not investigated, and it does not affect conclusions here.
13. **(New) The ppo_sim censored-KL bug** [C04]. The draft's formation-phase "about 50% behavioural" came from a hard-coded σ = 2.0. With the true σ (1.67) the share is 0.70-0.79.
14. **(New) Omni creeper figures** [C13]. The draft called training-log income a "replay" and compared stages with different reward functions. The genuine same-terms replay supports the direction, and shows the penalties decide it.
15. **(New) The fine-tune optimizer-state confound** [C14]. Refuted as a general explanation by the fresh-state formation arms (Section 4.2).
16. **(New) "A swap need not port the RMA machinery"** [C16]. It need not port the *training*, but the modules are required by every loader, exporter and the distiller.
17. **(New) What Policy/mean_noise_std records.** One C05 voter thought it was the clamped σ. The runner (:546) and the logged values above 2.0 show it is the raw parameter; Loss/entropy uses the clamped σ.

---

## 9. Evidence gaps (stated plainly)

**Experiments never run.**
- No run ever varied a core PPO hyper-parameter or the algorithm family. There is no in-repo evidence for or against alternatives.
- No bounded-distribution arm (tanh, Beta or bound loss) was run, from scratch or as a fine-tune.
- No from-scratch low-σ arm was run.
- No fine-tune was run with a fixed (non-adaptive) LR.
- No entropy_coef = 0 control exists, so whether entropy or the surrogate drives σ to the cap is unknown.
- No single-run throughput probe above 256 envs was run, and no MPS trial.
- For the side gait: no no-mirror +y arm, and no C_40 or C_fa control. All two-direction negatives are seed 3 only.
- The late-window regressions lack clock-held controls.

**Telemetry never logged.**
- There is no KL, clip-fraction, explained-variance or within-iteration LR telemetry. In particular, M1's per-update KL is unknown.

**Estimates that rest on CPU reconstruction.**
- The rollout/update input-mismatch term in the KL has only one CPU estimate (W, with an assumed "true" priv slice).
- The CPU re-enactments use synthetic advantages and eval-rollout states, run at a fixed LR, and omit the value, mirror and priv_reg terms. The X19 sims' state source is unrecorded.
- The X19 saturation trajectory and the reconstruction validation were not independently reproduced (they need .pt loads).
- The vestigial-RMA finding was measured on W only; the estimator's value in forward/side teacher runs (mass and com randomised) is unmeasured.

**Other unknowns.**
- The co-fire rate of mislabelled time-outs was not measured.
- Whether the 4 clip_actions = None teacher runs executed unclipped actions is unresolved.
- The source of the nondeterministic history-latent kernel is unidentified.
- Most decisions are single-seed. The seed-2 confirms for W, the phase pipeline and the forward lineage were waived.
- The census excludes 164 archived August PPO runs. Headline figures including them were recomputed only for LR and σ.
- The side_bisect campaign is still live: X20 (C_lr_10_rsi33) was training at compile time.

---

## 10. What this evidence says, before any literature

### (a) Is there evidence that the algorithm or its hyper-parameters are a bottleneck, and where?

**Yes, but it indicts one specific construction, not PPO as a family.** It also rests on mechanism and telemetry rather than on any controlled comparison, because none was ever run. Blunt list of what is wrong with the current setup:

1. **The action distribution is the wrong shape for the actuator interface.** This is the strongest evidence.
   - PPO optimises the log-prob, KL and entropy of an unbounded Gaussian, while the env executes clip(a, ±1).
   - Nothing in the loss keeps μ inside the bound, so it leaves: every trained head evaluated after about 1000 iterations is 91-100% saturated (W: 97%, median \|μ\| 11; forward heads median 20-36).
   - The deterministic controller is therefore sign(μ) on about 95% of components: bang-bang. This has a direct behavioural signature in every head: chatter at 2.2 flips/dim/s, stand stepping, the rigid 3-feet pattern, and an action_rate penalty that pays for saturation.
   - It is the only optimizer-side defect with a measured behavioural consequence. It has never been tested against a bounded alternative.
2. **The σ clamp is a trap.** Above 2.0 every σ gradient is exactly zero. σ climbs to the cap in every run of 3500+ iterations and can never come back. Lowering the entropy coefficient would not un-stick a pinned head. This design is krabby-specific: upstream rsl-rl and the repo's own Go2 class have no clamp.
3. **The step-size controller is miscalibrated for this batch and this parameterisation.**
   - The configured stage LRs are fiction: erased inside iteration 0 in 94% of fresh runs.
   - The logged LR sits at the 1e-5 floor in about 60% of all iterations, and late heads live at 1e-5 to 3.4e-5.
   - On saturated heads, 84-88% of the KL the controller reacts to changes no executed action, and the KL also contains a rollout/update input-mismatch term.
   - **Where it bites:** late-stage refinement and fine-tuning. During formation the logged LR is 3-6x higher and the KL is mostly behavioural, so the controller is not the formation bottleneck.
4. **The batch geometry was copied from a 6144-env config and never rescaled.** Yet per-iteration wall time is latency-bound: 4x the envs costs +5%, and a single run uses about 65% of the GPU at 66 W. The one cheap lever with a plausible large sample-efficiency gain (more envs per process) has never been measured.
5. **The instrumentation cannot see the controller that governs every update.** No KL, clip fraction, explained variance or saturation telemetry exists. This is why an M1-style question stays unanswerable.
6. **Experimental hygiene limits what any algorithm comparison could show.** Most verdicts are single-seed, same-seed reruns diverge intermittently (a nondeterministic history-latent kernel), and unknown cfg keys are dropped silently.

**Where the bottleneck is, stated carefully.** Evidence that the parameterisation limits *motion quality* is strong. Evidence that it limits *task metrics* (tracking, obstacle completion, side speed) is absent either way: no arm ever removed it. Every reward and curriculum verdict in the ledgers is conditional on this parameterisation.

### (b) What the evidence says is NOT an optimizer problem

Each point is as far as the evidence goes, with its caveat.
- **The forward unison glide** earns as much as or more than the alternating gait at matched iteration, under an effectively identical reward. A better optimizer would find it too. This is reward design.
- **The omni 1c creeper** is a within-stage reward drift; in a same-terms replay its advantage comes from penalty terms that tax motion (foot_idle, action_rate). This is regularizer design.
- **move_turn under-performance** is kinematic: the outer foot would need about 0.58 m/s.
- **The side-gait stroke and centring nulls are not explained by the LR floor or the σ pin.** The same terms failed in from-scratch formation arms with fresh optimizer state (LR at the floor only 23-37% of the time, peak 3.8-5.8e-4, σ starting at 1.5). Fine-tunes under the throttled state did move other behaviour (c6x turning 0.475 → 0.787). These nulls point at the reward terms and the two-hand question, not at the step-size controller.
- **Two-direction side slowness is not selected by any PPO telemetry and not by capacity.** Telemetry overlaps completely between classes, and the architecture distils the two-hand policy to 99% sign agreement. Caveat: this does **not** make it "not an optimizer problem". PPO settings were constant across all arms, and the campaign's own leading mechanisms are learning-signal and early-lock-in stories that exploration and parameterisation changes could plausibly affect.
- **The formation seed lottery** was closed by structure (clock + RSI from a good bank). Structure is sufficient. Whether an exploration-side algorithm change would also close it is untested.
- **Distillation** is not a bottleneck: student ≈ teacher (0.79/0.64 vs 0.81/0.66).
- **Stage-boundary collapses** are a value-scale problem (a critic priced about 0.5/step against about 0.0015/step on the bridge), not a policy-optimizer problem. They are a value-side hyper-parameter question (return normalisation, critic reset), not an algorithm-family question.
- **The vestigial RMA machinery does not affect behaviour on W** (latent ablation KL 9e-5). It is a porting cost, not a performance limit.

### (c) Questions the literature review must answer

In priority order, each tied to a decision this evidence cannot make on its own:

1. **Bounded action heads for on-policy locomotion.**
   - Compare tanh-squashed with log-det, Beta, rl_games-style bound loss and clipped Gaussian in PPO locomotion.
   - Is mean drift past the bound with σ ≥ the bound a documented pathology?
   - What effect sizes on smoothness *and* on task metrics or learning speed are reported?
   - What do the reference stacks (legged_gym, Isaac Lab, rl_games, extreme-parkour) actually do about clip_actions and output bounding? Verify each; the dossier's statements about them are unverified.
   - This decides whether arm (i) in Section 0.5 is expected to change task metrics or only smoothness.
2. **σ parameterisation.**
   - Evidence on log-std vs clamped std; init std (0.5-1.0 vs 1.5); entropy_coef 0 vs 0.01; target-entropy schemes.
   - Is any published stack known to use a zero-gradient σ clamp, and what happened?
   - This decides arm (ii) and the order of changes.
3. **Step-size control.**
   - Is per-minibatch adaptive KL at small minibatches known to collapse the LR?
   - Is KL on the executed (censored or squashed) distribution used or recommended?
   - Fixed vs linearly decayed vs adaptive LR in locomotion PPO, and typical LR trajectories at 98k-147k-sample batches.
   - This decides arm (iii), and whether a fixed-LR control is a fair baseline.
4. **Batch geometry and env scaling.**
   - Env count, rollout length vs gait period, minibatch count for low-env-count PPO, and typical Isaac Lab throughput scaling from 256 to 4096 envs on a 16 GB GPU.
   - This decides whether "more envs" is the first, nearly free change.
5. **Direction-flipped skills in one network.**
   - Does the literature report a shared policy failing to *learn* (rather than represent) mirror-related phase-locked behaviours?
   - Which remedies work at training time: equivariant networks, canonicalisation during training, multi-head/MoE, gradient-conflict methods, exploration changes?
   - This decides whether class (c) is attacked through structure (canonical training, already planned) or through the optimizer.
6. **Off-policy and other families at this scale.**
   - FastTD3, FastSAC, PQL, SAPG, PQN, model-based (TD-MPC2, DreamerV3): VRAM needs next to Isaac on 16 GB; wall-clock vs PPO at 256-4096 envs; handling of 1000+-dim history observations; symmetry; sim-to-real; any slew-limited or non-backdrivable actuator results.
   - This decides whether a swap is worth its porting cost (checkpoint schema, estimator/history stubs, mirror loss, terminal labels) *after* the in-PPO fixes. The in-repo evidence gives no reason to swap first.
7. **Variance reduction.** Do population-based, ensemble or QD methods measurably reduce seed-to-seed and basin-lottery variance in locomotion? And what seed count do comparable studies use to claim an algorithm or hyper-parameter effect? This sets the design of any screen (here at least 3 seeds per arm, scored on formation rate and spread).
8. **Symmetry method.** Mirror loss on raw vs bounded means, detached target, augmentation vs loss vs equivariance, coefficient. Does a mirror-loss formation advantage replicate across seeds in published work? In-repo it did not replicate cleanly.
9. **Value side.** The "only positive rewards" clip; return normalisation (PopArt) under curricula; a separate critic LR. This decides whether the value-side changes ride along with the policy-side arms or are screened separately.
10. **Distillation from bounded teachers.** What target should the student regress: executed actions vs raw means vs squashed pre-activations? This is needed before any bounded teacher can feed the existing student pipeline.

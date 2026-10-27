# RL algorithm and PPO hyper-parameter review: decision report

*2026-09-30. Revised after two independent checks: R0–R14 on the first ladder; I1–I15, V1–V11 and X1–X28 on this report (dispositions in §10). CPU-only analysis; nothing here launches a run. Inputs: `DOSSIER.md` beside this file (602 PPO runs, claims C01–C16 each verified twice); `../lit-review-rl-algorithm-and-ppo-hyperparameters.md` (143 references); five critiques of a first ladder in `artifacts/decision/challenge_*.json`, with the first draft (`REPORT_draft1.md`) and first check (`check.json`) beside them; the live side-bisect records; code re-read for this revision.*

Shorthand: `P` = krabby-research/parkour; `T` = P/parkour_tasks/parkour_tasks; `DOS` = DOSSIER.md by line; `LIT` = the literature review by section (`:line` where precise); `SB` = T/crab_hex_omni_task/experiments/2026-09-28_side_bisect, `RC` = SB/ROOT_CAUSE.md, `CH` = SB/CHARTER.md; "PPO file" = P/scripts/rsl_rl/modules/ppo_with_extractor.py; "runner" = on_policy_runner_with_extractor.py beside it; "env" = P/parkour_isaaclab/envs/parkour_manager_based_rl_env.py. Critique findings: SK skeptic, SW switch-advocate, DE design, IM implementation, AP action-path audit (e.g. DE-F3).

**Live work this report must not disturb** (as of 2026-09-30 18:44; P/outputs/omni_plan/side_bisect/queue.log): C_fa s3 FAST (0.211/0.206 m/s; cbf8ad06); C_40 s3 DEAD (0.035/0.031 m/s, falls end 0.75 of episodes from iteration 2000; b48e7ee6); C_40_s2 (X23) training since 17:41, early read 0.180 (forming SLOW). Any edit under P/scripts, P/parkour_isaaclab, the crab config/mdp trees or the eval paths (gait_eval, eval_crab_hex_gait.py, scenarios_omni.yaml, omni run_phases.py, eval_categories.py) blocks that queue (run_candidates.py:163-184, :509-518).

---

## 1. Verdict

1. **Keep PPO. No other algorithm family is worth testing now.**
   - No task failure in the ledgers is attributed to the policy optimizer. The scoreboard (DOS §4.8 :680-692) gives ALG/HP one squarely attributed row, motion quality; the bridge collapse (0.00 → 0.90) goes to value-side scaling (rung 5); side slowness and the basin lottery are "HPs never varied" or "untested", not excluded. Task-metric evidence "is absent either way" (DOS :993).
   - The closest off-policy analogue reaches parity with PPO only at 8,192 envs, after the same fixes proposed here, with a wall-clock gap and more seed variance (LIT §10 [40]); no paper compares the two on a legged robot at ≤ 256 envs (LIT §14); neither wall-clock nor samples bind here (§5.2).
2. **The construction around PPO is broken. Fix it for signal, smoothness and hardware, not for an expected task gain.**
   - Unbounded mean against an executed clip(a, ±1): 97.0 % of W's pre-clip components have |μ| > 1 (median 11.2); all 303 checkpoints past 999 iterations are 91-100 % saturated (DOS :87-91).
   - σ sits behind an absorbing zero-gradient clamp at 2.0: 59/59 fresh runs of 3,500+ iterations pin, and none of 303 pinned runs leaves (DOS :84-86).
   - The per-minibatch KL rule erases the configured LR inside iteration 0 in 240/254 fresh runs. A CPU re-enactment on W (not a training measurement) puts only 11-16 % of the KL it reacts to on executed actions (DOS :185, :256).
   - Consequences: chatter (2.23 raw-μ sign flips per joint-second at stand; DOS :496) and a learning signal the re-enactment calls mostly phantom on late heads, unmeasured in training until rung 0. Pre-registered task outcome: no change (§6.2).
3. **The largest task-metric gain is structural.** Every network that sees both hands, through real −y rows or the 4-image mirror's phantom ones, is SLOW or DEAD (0.028-0.103 m/s); +y-only networks with the fa mirror are FAST in 4 of 5 runs (0.184-0.217 m/s; one INTERMEDIATE at 0.162; DOS :570-571). The exact M_y canonicaliser makes 4/4 fast heads FAST on both hands with no retraining, |L−R| ≤ 0.006 (DOS :566-572). Canonical training is the fix; it belongs to the side bisect, and its wrapper lands first (§6.0, §6.3).
4. **Worth a controlled test:** the action head with the σ parameterisation, as one bundle (rung 2, always); the step-size rule, only after a rung-2 adoption, on a late tail fork (rung 3); batch geometry, only if 1,024 envs cost ≤ 1.15× the 256-env time per iteration (rungs 1, 4).
   **Not now:** γ; the 2×2 off-diagonals (except as bisection diagnostics); reward-clip removal and time-out relabelling (a reward campaign); other head families (except the rung-2 fallback head); off-policy learners (except the user-choice TD3-lean test, §5.2).
5. **Cost (§6.5):** ≈ 18-30 GPU-h always (rungs 0-2; ≈ 2-3 days at the census rate of 9.3 GPU-h/day, DOS :73); 49-91 if rung 2 adopts (package confirm and rung 3; 5-10 days); at most ≈ 140 (≈ 15 days) with every conditional arm on the adopt branch, ≈ 98 on the reject branch. The minimum path, rung 2 rejects and the ladder stops, is ≈ 18 GPU-h. The lit review's "50-60 GPU-h" (LIT §15 :598) priced training arms only.
6. **Fix regardless** (code only): the replay tool's rod clip (§4.4) can land now, outside both pathspecs; the exporter clamp (§4.1) and fixed-LR resume (§4.2) land at this ladder's campaign boundary; the runtime half of §4.1 waits for a crab runtime, which does not exist yet.

---

## 2. What the experiments say

### 2.1 Never varied (DOS :72-81, §5.2 :722-749)

All 602 PPO runs used epochs 5, minibatches 16, desired_kl 0.01 under the adaptive per-minibatch schedule, clip 0.2, max_grad_norm 1.0 and one Adam group; γ 0.99, λ 0.95, a clipped value loss and reward clipped at ≥ 0; 24 steps/env, with 256 envs for every RL run longer than 20 iterations; no observation normalisation, a scalar clamped σ, entropy 0.01 (one smoke at 0.005); one algorithm family. The only σ experiment, M1, fine-tuned an already saturated head with no bound on μ (DOS :498-509).

### 2.2 What moved task metrics

| Lever | Class | Effect | Source |
|---|---|---|---|
| Yaw-progress weight 3.0 (D10) | reward | turn facing 0.419 → 0.748 (REJECTED: move_turn survival 0.580 → 0.120) | facing_tracking_ladder/CHANGELOG.md:72 |
| C1d turn-first curriculum (D12) | curriculum | 0.419 → 0.618 (adopted by overrule) | same file, :89-93 |
| c6x clock removal, on a throttled optimizer | task/structure | turn facing 0.475 → 0.787; move_turn survival 0.62 → 0.255 in the same stage (open follow-up) | clock_deprecation_w2/REPORT.md:160-163; DOS :469 |
| fa mirror (+ scalar obs), +y-only runs | structure | side 0.084 → 0.184-0.217 m/s in 4/5 runs; C_fa (fa mirror alone) 0.211/0.206 | DOS :571, :639; CH:686-690 |
| Golden vs wide plants (B, A15+B) | morphology | obstacle onsets 0.31 (golden) vs 0.69-0.95 (wide) | DOS :405-408 |
| Clock + good RSI bank | structure | failing seed 2 → tripod 0.496 | DOS :404, :559 |
| **M1 (σ cap 0.5, reset 0.4), the only optimizer change** | optimizer | **Q z −1.36 vs R 0.32 (WORSE); move_turn ratio 0.504 → 0.433 (guard WORSE); turn facing 0.610 → 0.510 (z −2.02 against R 2.07, a TIE)** | gait_motion_quality/CHANGELOG.md:20; REPORT.md:166-167, :181-255 |

Saturation does not select the side-speed class: fast and slow side heads are both 0.941-0.978 saturated (DOS :302, :515), and σ, LR, entropy and clipped-rod share overlap completely across 17 arms (DOS :573-577). The earlier motion-quality review predicted that capping σ would help (lit-review-gait-motion-quality.md §10 item 3); M1 is a recorded prediction failure.

### 2.3 Where the construction shows up

- **Bang-bang output.** The executed action is effectively sign(μ) on ≈ 94-97 % of components. On X08:c6, raw-μ sign flips run at 2.23 per joint-second at stand and 1.83 while moving; stand contact switching is 1.65 Hz (DOS :91, :496).
- **Phantom KL (CPU re-enactment, not a training measurement).** On W's eval states, with synthetic advantages, a fixed LR and no value, mirror or priv_reg terms, 84-88 % of W's KL buys no executed change, putting executed change 5-10× below what desired_kl nominally allows (DOS :185, :256). At X19's iteration 300 the KL is 70-79 % behavioural and the logged LR 3-6× higher (DOS :94, :251), so any throttle binds late, not during formation. Training logs no KL; rung 0 adds it.
- **Dependence on the clip.** Without the ±1 clip the teacher fell in 100/100 episodes (DOS :512).

### 2.4 Statistics

- **Seed-dominated turning.** R_seed/R_eval is 3.19/0.97 (F) and 4.12/0.94 (F_mt). The single-seed MDEs, 0.145 (turn facing) and 0.180 (move_turn facing; facing_tracking_ladder/CHANGELOG.md:46-47), are one s2-vs-s3 draw: seed SD ≈ 0.10 and ≈ 0.13 (eight clock-held t3 evals give 0.111 for turn facing; DE-F2). One pair has 1 df (90 % interval [0.36, 11.3] × |Δ|), and the MQ ladder's pair gives 0.100 / 0.252 (D00bs), so the MDEs below are order-of-magnitude.
- **More seeds do not divide by √N.** MDE at 80 % power, α 0.05 two-sided, noncentral t, recomputed with scipy:

  | Comparison | MDE (seed SD) | Turn facing | move_turn facing |
  |---|---|---|---|
  | 3 v 3, two arms (df 4) | 3.07 | 0.31 | 0.39 |
  | 3 v 3, three arms pooled (df 6; rung 2) | 2.75 | 0.28 | 0.35 |
  | 5 v 5, two arms (df 8) | 2.02 | 0.21 | 0.26 |
  | 5 v 5, three arms pooled (df 12) | 1.93 | 0.20 | 0.25 |
  | 2×2 main effect, 3 per cell (df 8) | 1.85 | | |
  | 2×2 interaction | 3.70 | | |

  The eval-only per-member MDE for turn facing is ≈ 0.044 (D00s); the seed term dominates, and a rung-2 turning difference below ≈ 0.28 at 3 seeds, or ≈ 0.20 at 5, is not a finding.
- **Exact tests.** A 3 v 3 permutation test cannot go below two-sided p 0.10 (20 splits). Fisher: 3/3 vs 0/3 p = 0.10; 5/5 vs 0/5 p = 0.008; 4/5 vs 0/5 p = 0.048. Every clock-held omni t3 run formed (DOS :403), so formation rate cannot decide.
- **Survival.** Run-to-run SD at t3 is 0.162 / 0.143 / 0.206 (fwd / diag / move_turn; DE-F2). A cell-mean-drop tripwire at 0.2 fires falsely in ≈ 21 % of 3 v 3 comparisons across those cells, at 0.30 in ≈ 5 % (recomputed). Non-inferiority at −0.05 is unattainable (margin 0.36 at 3 v 3, 0.28 at 5 v 5; DE-F2).
- **The driver cannot judge 3-5 seeds:** its pooled R is max(R_eval/√2, R_seed) for an s3+s2 pair (resolution.py:321-328), so rung 2 needs its own judge script (§6.2).
- **Same-seed pairs are not controls:** 2 of 4 diverge from iteration 20 and 2 replay exactly (X03:c1 == X04:c1; DOS :420-427), so a rerun can replay a selected draw. W and C1d were selected on s3. An env write after Kit start flips some evals between 0.79 and 0.75 (memory eval-process-state-sensitivity.md).

---

## 3. What the literature says

- **It indicts the construction, not PPO.** No reference legged stack pairs a binding ±1 clip with an unbounded mean and a σ clamped at or above the bound (LIT §2). rl_games adds a quadratic bound loss outside ±1.1 (coefficient 1e-4 to 1e-3); VideoMimic an L1 bound loss against "bang-bang style control" (LIT §2-§3).
- **σ alone cannot pull saturated means inside the bound** ([3]), as M1 found. **Bounded heads change geometry reliably and return inconsistently** ([3, 4, 22]); the legged positives used 4 seeds (DHAL) or unreported seeds (GPO) ([19, 25]), and near-bang-bang control may be near-optimal here ([26, 27]).
- **Step size and batch.** Krabby runs Rudin's Algorithm 1 on minibatches 64-96× smaller than it was tuned on; the only controlled fixed-vs-adaptive comparison spans zero (LIT §6 [41, 48]). The batch is 8× below Rudin's smallest; rollouts under ≈ 25 steps hurt gait discovery (LIT §7).
- **Off-policy.** RSL-RL-SAC reached parity with PPO only at 8,192 envs, after σ0 0.15, a tanh head and time-out-aware targets, with a wall-clock gap and more seed variance ([40]); its abstract's "closes the performance gap with PPO entirely" is parity, not superiority.
- **FPO++** is the only variant with legged results at Krabby's scale (2⁸-2¹² envs, 5 seeds). It reports lower seed variance and robustness to very small batches against rsl_rl's default Gaussian PPO with a clip sweep: the unclamped reference stack, not a strawman ([97]; LIT :386-392). Costs: a new head, 5-step integration at deploy, no Gaussian KL for the LR rule, an undefined mirror loss.
- **Two-hand failure** has no precedent; canonical training has one weak analogue (LIT §8).
- **Caveat.** The load-bearing bounded-head, step-size, off-policy and FPO++ numbers each rest on a single 2026 preprint [3, 24, 25, 40, 60, 97] (LIT §14 :532). Treat them as directional.

**Corrections to the literature review** (the review file is not edited):
- **§4 :173-174** calls action_rate squared and "near-constant under saturation". It is the **unsquared** L2 norm of the delayed, clipped executed action (P/parkour_isaaclab/envs/mdp/rewards.py:337-343; squared only under `KRABBY_ACTION_RATE_SQ`, default off). W's run logs −0.275 over its last 500 iterations; TB's Episode_Reward is normalised by the 40-s max episode (IsaacLab reward_manager.py:120), so that is ≈ −0.34 per alive-second at the 32.3-s mean episode (AP-A1).
- **§4 :172-175** prices smoothness with a TV/L1 or action_rate cost on the executed action, which under sampled training carries the same noise tax (§4.3). Price it through a loss on μ, or through physical costs downstream of the slew filters (dof_acc, delta_torques), and score it separately (AP-A8).
- **§2 :89-90** is answered: history observations carry the clipped action, executed one step late (T/crab_hex_forward_task/mdp/parkour_actions.py:348-356), so the PR #2019 loop applies only at deploy (§4.1).
- **§15 :570** says "the deploy clip stays". There is no deploy clip (§4.1).

---

## 4. Defects found while checking (fix regardless)

### 4.1 No deploy path clamps actions, and no crab deploy path exists yet

- **Exporter.** The deploy exporters (P/scripts/rsl_rl/exporter.py:111-176) substitute the estimator and return the raw actor output with no clamp (forward :127-130, :157-160); the teacher exporters (:50-108) run the actor with no estimator and no clamp. Both ONNX exports trace a 1 × 1,007 zero input against the live 1,149-dim observation with `dynamic_axes={}` (:97-108, :164-176), and no in-repo script calls them.
- **Runtime.** compute/parkour, the only policy runtime, does not use exporter.py: it loads the raw .pt and calls act_inference, the raw mean (compute/parkour/policy_interface.py:160-186, :364-365). Its mapper copies model actions into JointCommand with no clip or scale (mappers/model_to_hardware.py:92-100), and the sent command is fed back as the previous action (inference_client.py:278; mappers/hardware_to_model.py:43-54, :181-182), the PR #2019 loop training avoids. It is Go2-shaped (action_dim 12; model_definition.py:113-131), its module copy cannot build CrabHexActorCriticRMA (IM-F18), and MODEL_CONTROLLER_KRABBY raises NotImplementedError (controller/control_loop.py:127-129). The only bound is the MCU jog layer's ±255 PWM saturation at |cmd| ≥ 0.5 rad (hal/server/jetson/krabby_mcusdk.py:15-18, :28-29), not the training clip.
- **So** the missing clamp is latent. W's raw means have median |μ| 11.2 and p90 22.0, and without the clip the teacher fell 100/100 (DOS :512): a crab runtime built on this code would drive every joint at full authority.
- **Fix.**
  - When the crab runtime is built (an 18-action, 1,149-obs model definition), apply the ±clip_actions clamp and the training action scaling (cams × π rad/s; rods × 0.25 rad through the linkage) in its inference path, and feed the clamped action back as the previous action.
  - Exporter hygiene: clamp inside every exported module; export the teacher with its estimator; trace ONNX on a 1,149-dim input.
  - The student export substitutes the estimator, but student training and every student eval do not (runner :441, :444); make them match. This corrects DOS §6 :805.
  - A bound-loss head still depends on the clamp; only a tanh or Beta head removes it. A canonical-trained head also needs the canonicaliser in the runtime, or a distilled student (§6.3).

### 4.2 A fixed-LR fork silently trains at the checkpoint's LR

- **What happens.** `param_group["lr"]` is written only in the adaptive branch (PPO file :377, :403-404). `runner.load` restores the optimizer state with its saved LR (runner :737). `load_resume_state` restores `learning_rate` only under the adaptive schedule (PPO file :557-560). TensorBoard logs the configured LR, not the applied one (runner :543).
- **Couplings.** Actor, critic and σ share one Adam group (PPO file :113), so any schedule change also moves the critic's LR by 3-30×. The DAgger `hist_encoder_optimizer` is built at the same configured LR (:122-124), so any LR arm also moves the history-encoder LR.
- **Fix.** Write the configured LR to every param group after loading; log the mean applied LR; add a unit test that resumes under `fixed` and checks it; give `hist_encoder_optimizer` its own LR kwarg; label the step-size rung "actor+critic step size".

### 4.3 action_rate taxes exploration noise, which a bounded mean exposes

- **Mechanism.** During training the term reads the delayed, clipped *sampled* action (rewards.py:337-343; parkour_actions.py:348-356; vecenv_wrapper.py:134-136). A saturated mean hides its noise: at σ 2 the expected |Δa| of one dimension is 0.000 at |μ| = 9 and 0.95 at μ = 0 (unweighted, per step). The term is one L2 norm over 18 dims, so a fully interior mean costs E‖Δa‖ ≈ 5.1 per step before the −0.3 weight (recomputed, 10⁶ samples).
- **Monte Carlo** (AP-A1; W's eval states, 88,697 steps, weight 0.3; not re-run here). W's means cost 0.287/s deterministic and 0.421/s sampled at σ 2. Clipped to ±1.1, as a bound loss would leave them, they cost 0.566 / 0.975 / 1.252 / 1.407 per s at σ 0.5 / 1 / 1.5 / 2, which is +0.27 / +0.65 / +0.89 / +0.99 over W at the same σ; the audit's +0.15 / +0.55 / +0.83 / +0.99 is the same run against W at σ 2, not a second simulation. The skeptic's independent noise-only Monte Carlo agrees: W-like 0.047/s, bounded 0.73-1.49/s, so saturating frees ≈ 1.2/s (SK-S2). Both are noise-floor models, not closed-loop training.
- **Why it matters.** The fast side gait out-earns the slow one by only +0.25 to +0.43/s over O and +0.29 to +0.32/s over the two-direction nets (RC §2.8): the tax is the size of the margin that selects the gait. It also rewards shrinking σ fast, and smoothness taxes have cost completion before (gait_formation_v2: L2 0.81, L3 0.64 vs 0.90-0.94).

### 4.4 The omni replay tool clips rods at ±4.8

- **The bug.** `ACTION_CLIP = [1.0]*6 + [4.8]*6 + [4.8]*6` (T/crab_hex_omni_task/experiments/2026-09-16_1805_omni_formation/replay_reward_economy.py:100, applied at :391). ±4.8 is the action term's own rod clip, but the runner clamps everything to ±1 first. Commit 3714934c fixed only the clock replay tool.
- **Erratum for DOS §4.4(a) :548.** The walker/creeper action_rate figures (−0.516 / −0.198) come from the bug. At the ±1 clip they are −0.204 / −0.132 deterministic, ≈ −0.692 / −0.306 sampled at σ 1.9. The ordering is unchanged: the net gap is 1.41/s, dominated by foot_idle (−1.747 vs −0.441).
- **Fix.** Read the clip from each head's agent.yaml, and add a sampled-σ mode to every replay tool. The file is outside both pathspecs, so it can land now.

### 4.5 RSI spawns start by braking their cams

- **The sequence.** The reset event writes the bank's joint velocities (T/crab_hex_forward_task/mdp/crab_hex_rsi.py:95-97; env :263). The action manager then resets (env :272), zeroing the applied cam velocity target (parkour_actions.py:191) and the delayed-action history (P/parkour_isaaclab/envs/mdp/parkour_actions/joint_actions.py:57-59). With the omni 1-step delay on, step 1 executes a zero action, and the applied cam target ramps from 0 at 2π rad/s² (parkour_actions.py:289-291): the cams stay below a walking bank's ≈ π rad/s for up to ≈ 0.5 s (≈ 25 steps).
- **Why it matters.** RSI starts underperform normal spawns in every arm, including the fast C_mo (failure 0.127 vs 0.021; RC:213-218, :253-256). That deficit is cited as "RSI inert at σ 1.5" (DOS :584, :713; LIT §5 :208); the cam reset explains it without σ. Rung 0b tests this; until then the four RSI-seeded two-direction negatives are discounted as σ evidence.

### 4.6 Infrastructure hazards

- **Unknown keys vanish.** Unknown PPO keys are dropped with only a warning (runner :128-141), and silently on the distillation path (:115-127). Unknown *policy* keys are absorbed silently into `ActorCriticRMA.kwargs` (actor_critic_with_encoder.py:131-142): a misspelled head or σ knob would make arm A identical to C0 with no warning.
- **Worktree runs import the main checkout:** `train.py` adds only `parkour_tasks` to the path (:13-20); `scripts/` and `parkour_isaaclab` resolve through `__editable__.isaaclab_parkour-0.1.pth`.
- **`NUM_ENVS = {"rl": "256"}` is hard-coded** in T/crab_hex_omni_task/experiments/tools/run_phases.py:71 (used at :1297), an EVAL_PATHSPECS member (run_candidates.py:182) whose training seam is pinned by tests/unit/test_crab_hex_omni_gate.py:2542; the forward (:52) and sideways (:54) drivers carry their own copies.
- **Dropped rows.** The installed rsl_rl 5.0.1 generator (isaac_venv site-packages rsl_rl/storage/rollout_storage.py:226-228) drops batch % num_mini_batches rows: none at 256 × 24 / 16, but not once env and minibatch knobs exist, hence the divisibility assert.
- **Eval action metrics.** The Q-block metrics already use executed ±1-clipped actions (gait_eval/metrics.py:1489-1497; eval_crab_hex_gait.py:1316). The episode-level `actions` block uses raw means with Δμ reversals at a 0.01 deadzone (report.py:294-299 → metrics.py:929-979), and the 2.23 figure counts raw-μ sign flips with no deadzone (ANALYSIS_M1_saturation.md:44-48); only those two change meaning between saturated and bounded heads.

---

## 5. Another algorithm? Family by family

| Family | Evidence | Fit now | Verdict |
|---|---|---|---|
| SAC / TD3 | Parity at 8,192 envs after the same fixes; wall-clock gap; more seed variance ([40]). FastTD3 at 4,096 envs ([87, 88]) | Nothing at ≤ 256 envs; wall-clock and samples do not bind (§5.2) | Not now; TD3-lean only as the §5.2 user-choice exception |
| Model-based | Wins on sample efficiency; the one legged sim-to-real result took ≈ 5.5× the wall-clock | Krabby is not sample-limited (18M samples ≈ 1.1 GPU-h) | No |
| FPO++ | Legged, 2⁸-2¹² envs, 5 seeds, lower seed variance ([97]) | First candidate if clock-free formation is wanted (Z formed 0/2; DOS :560). New head and checkpoint contract | Only under §5.2 |
| REPPO | On-policy, pathwise gradients, no replay; no legged results (SW-A7) | Only if rung 1 makes 1,024-4,096 envs attractive and replay memory then binds | Later, conditional |
| Beta / ordinal / GPO heads | Weak legged evidence (LIT §3) | Change the exporter, distiller and checkpoints | Ordinal only as the rung-2 fallback head; otherwise §5.2 |
| PBT | DexPBT, PBRL | Needs an env-slice runner; 4-6× wall time per member under time-slicing | Later, formation lottery only |

**Never pip-install RSL-RL-SAC into isaac_venv.** It ships as rsl-rl-lib 4.0.1 and would replace the installed 5.0.1 (SW-A4, not re-verified), and its time-out observation patch would double-shift this env's history buffer, which already shifts on every compute (P/parkour_isaaclab/envs/mdp/observations.py:120-124, verified). Any off-policy learner is written in-repo.

**Swap costs, corrected.** A replay of the 259-dim non-history slice (fp32, with next obs and 21 other floats) is 2,156 B per transition, 0.57 GB for 1,024 transitions per env at 256 envs, not the dossier's 9.2 GB per 1M (§6 :785-791); any learner can emit a PPO-schema checkpoint with a stub estimator (DOS :800-815). LIT §15's condition (b) becomes a learner-overhead timing inside rung 1.

### 5.2 Escalation trigger, inverted

The lit review says to escalate "only if rung 2 moves geometry but not task metrics" (LIT §15 :594-596). That is backwards: geometry moving with no task gain means the optimizer is **not** the task limiter. Revised rule:
- **Geometry moves, tasks do not** (N ≥ 3, guards held): record "the construction is not a task-metric limiter"; keep the bounded head only if Q improves; stop algorithm work.
- **Any further algorithm arm requires all four:** a pooled, lineage-length task-block win in the rung-2 judge script; the predicted telemetry mechanism visible in rung-0 logs; no reproduction by the action_rate-only control INC0; a residual failure after canonical training that optimizer telemetry separates.
- **A family swap, not just another head, additionally needs** evidence that wall-clock or sample count binds. Today runs reach 90 % of final reward by median iteration 1,344-1,798 at ≈ 1.1 GPU-h per arm, and the learner is ≈ 20 % of wall time (DOS :489, :767, :782).
- **User-choice exception (not scheduled).** If canonical training fails and the RC §6 fallbacks (policy switch, distillation) are unacceptable, or you want "is on-policy PPO the barrier to the two-hand stroke?" answered directly: TD3-lean on C_lr against a matched tanh-PPO arm, written in-repo, sharing the tanh Actor (actor_critic_with_encoder.py:86-87), with a 259-dim replay, batch 4-8k × 2 updates, time-outs dropped, and PPO-schema checkpoints round-tripped through runner.load, exporter and distiller. ≈ 1,000 lines + 3-4.5 GPU-h (SW-A1, A5, A8). Kill: SLOW on both hands at 6M samples in 3/3 seeds, which makes the barrier learner-independent. Untested inference: candidate (i), the interaction barrier, is an actor-gradient property a DPG learner shares; only (ii) might differ.

---

## 6. Revised test ladder

### 6.0 Preconditions and rules

1. **Wait for the side-bisect queue's terminal line:** `DONE controls` on success or `STOP` on any failure (queue_controls.sh:59-88). Watch with a log sentinel matching `DONE controls|STOP`, never `pgrep -f`. A dirty code or eval tree makes the queue refuse 40 launches and exit (:52-59).
2. **Develop on worktrees**, launching with `PYTHONPATH=<worktree>/parkour` prepended (§4.6). New knobs default to off; knob-off runs must be byte-identical in the config dumps and identical through iteration 19 of a 20-iteration same-seed smoke.
3. **Land twice.**
   - **First (side-bisect-owned):** the canonical-training wrapper alone, as soon as the terminal line appears and the user has judged C_40_s2. `KRABBY_OMNI_CANON` does not exist in any .py yet; only the eval `--canon` hook does. If the side bisect runs C_lr_bound before C_lr_canon, the default-off bound loss must ride in this landing; otherwise C_lr_bound waits for the second.
   - **Second (this ladder, at its own campaign boundary):** the code for rungs 0-4 and 6.
   - **Each landing** moves code_sha (`--accept-code-change`, then a B<k> rerun per live ladder, ≈ 1.2 GPU-h per seed, ≈ 2.4 for s3+s2; run_candidates.py:159-170; IM-F2) and eval_sha (the wrapper's `--canon` carry in run_phases, IM-F15; KRABBY_NUM_ENVS in omni run_phases.py), which needs `--accept-eval-change` and a re-evaluation of every screen used as a control (:171-184). The second B<k> round is the price of not delaying the fix ranked first.
4. **Fail loudly.** Unknown algorithm keys raise (allow-list `share_cnn_encoders`); ActorCriticRMA raises on leftover policy kwargs (allow-list the three isaaclab_rl fields; actor_critic_with_encoder.py:142); each head knob prints a consumed line that OVERLAY_DUMP_CHECKS greps.
5. **Seeds.** Screens use fresh seeds 11, 12, 13, extended to 14, 15 in the continuation zone; confirms use 16-20. Never screen on s2 or s3: W and C1d were selected on s3, and same-seed reruns can replay the selected draw (DOS :420-427; DE-F5). C0 is retrained fresh at every seed at the new code_sha; historical heads are references, never controls; run order is randomised within seed blocks. The side bisect's own arms, C_lr_bound included, keep its s3 → s2 convention.

### 6.1 Rung 0 (code and CPU) and rung 0b (≈ 0.1 GPU-h)

**Telemetry**, RNG-free, under `PPO/`, with no "symmetry" or "mirror" in any tag name (run_phases.py:1762 greps those words as its mirror-active check):
- Gaussian KL under every schedule (today only under adaptive, PPO file :377) and executed-action KL; minibatch-0 KL (the unresolved D7 term, DOS :190); clip fraction; mean applied LR; pre-clip gradient norm and the share of minibatches above max_grad_norm (the surrogate alone gives 17.7-22.7 against a clip of 1.0; IM-F5); explained variance taken before advantage normalisation (:272-273).
- |μ| > 1 share, executed near-bound share, and deterministic ‖Δclip(μ)‖ beside the sampled action_rate.
- Offline from eval npz (no eval_sha move): executed sign flips per joint-second with a |a| ≥ 0.5 deadzone, and P10/P01 (He et al.'s near-bound shares). This is new telemetry, not a fix to the Q block (§4.6).

**Fixes to land:** §4.1 (exporter half), §4.2 (with the DAgger LR kwarg), §4.4 if not yet landed. Knobs `KRABBY_NUM_ENVS` (updates the seam pin), `KRABBY_PPO_MINIBATCHES` (divisibility assert), `KRABBY_SYM_AUG` (the forward config hard-codes `use_data_augmentation=False`, rsl_rl_ppo_cfg.py:139). RSI cam-velocity staging behind a default-off knob.

**Rung-2 knobs:**
- **Bound loss** on raw μ, soft bound 1.1: ≈ 10-20 lines in the PPO file plus the knob, ladder and test plumbing of the audited M1b spec (/home/nickmagus/.claude/plans/there-is-a-lot-compressed-unicorn.md steps 1-2; its line refs have drifted, e.g. loss sum :421-426 → :425-430). It needs no change to checkpoints, exporter, eval or distiller, which is why it is chosen over a tanh mean.
- **Tanh-bounded log-std** in [ln 0.05, ln 2]: a new `noise_std_type='soft'` parameter. Strict loads of existing heads fail; reset and load need the inverse map; the knob must reach resume and eval loading (eval rebuilds the agent cfg from the registry plus KRABBY_* env, eval_crab_hex_gait.py:639). The runner logs `Policy/mean_noise_std` as 0.0 for a policy without `std`/`log_std` (runner :545-551), so the arm logs a new `Policy/effective_std` tag; the σ reads use it, and side-bisect scripts on the old tag would see zeros.

**Gate:** the sampled-σ offline replay for arm B's weight-0 action_rate, before any GPU time. **Rung 0b:** a σ = 0 eval comparing RSI spawns with normal spawns, with and without cam staging.

### 6.1b Rung 1: throughput probe (≈ 1 GPU-h, idle GPU only)

- One process each at 256 (the same-window baseline), 512, 1,024, 2,048 and 4,096 envs, ≈ 200 iterations on the t3 config, under `systemd-run` with a MemoryMax cap; no MPS while any campaign is live (it makes signal termination undefined, and the driver stops runs by signal).
- Measure s/iter, steps/s, collection and learn time, peak VRAM, PhysX overflow warnings, NaN and fall rates, and the learner-overhead timing. Stop a size above 1.5 × the 256-env s/iter (1.318 s; SK-S13), above 10 GB VRAM, or on NaN or dead_plant within 200 iterations.
- Prior: 4× the envs cost +5 % wall time up to 256 (DOS :322-326); above 256 is unmeasured. Rung 4 runs only if 1,024 envs cost ≤ 1.15 × the 256-env s/iter.

### 6.2 Rung 2: the construction arm

**Bed:** omni t3 recipe (clock held) from scratch, 3,000 iterations, 256 envs, one eval env for every arm; every arm runs as a driver reference at the same code_sha and eval_sha.

**No factorial.** Lowering σ alone cannot produce an interior mean once |μ| is past the bound ([3]); a bound loss alone at σ 2 puts the mean inside ±1.1 but leaves the executed distribution clipped and the KL non-behavioural (LIT §15 :567-569). Arm A is therefore a deliberate four-knob bundle, taken apart only by the pre-registered bisection below.

**Arms:**
- **C0:** the incumbent, retrained fresh at the new code_sha at every seed.
- **A:** bound loss (coefficient ramped 1e-4 → 1e-3 over iterations 0-500), tanh-bounded log-std with init σ 0.8, entropy 0.0025, action_rate −0.3; no last-layer re-init.
- **B:** A with action_rate weight 0; isolates the §4.3 noise tax.
- **INC0 (conditional):** the incumbent head with action_rate 0, same seeds. It runs only if A or B differs from C0 on any task block; it is the action_rate-only control §5.2 requires.
- **Mirror loss** stays at 0.5 on raw μ in every arm. As an MSE on raw means (Loss/symmetry 0.026 × 0.5 vs |surrogate| 0.008; DOS :191) its strength changes when μ is bounded, so log Loss/symmetry per arm and read any side-ratio gain with it (DE-F10). A bounded teacher must be distilled on clip(μ) before phase 3 (W: median ‖μ‖₂ 59.8 vs 4.24 for clip(μ); AP-A6).

**Kill and manipulation reads** (n = 2 is enough for manipulation):
- **Iteration 200, wiring:** Loss/action_bound logged and > 0, consumed-knob lines present, Policy/effective_std present, and the |μ| > 1 share below the incumbent's 0.71 at iteration 199 (DOS :90); otherwise kill as mis-wired.
- **Iteration 1000:** |μ| > 1 share ≤ 0.5 (incumbent ≈ 0.92 at 999); σ > 0.1, else DEAD; training track_ratio_mean ≥ 0.40, else DEAD.
- **End of training (iterations 2500-3000):** σ < 1.9 on Policy/effective_std, and executed-KL share ≥ 0.5. No earlier σ or KL read: the incumbent's σ is 1.84 at iteration 1000 (DE-F6), and its KL is 70-79 % behavioural during formation (DOS :185).
- **Slope rule:** an arm behind at 3,000 iterations with a steeper final-500 reward slope is not rejected; it goes to the confirm.
- **LR floor:** if A or B sits at the LR floor in more than 90 % of iterations 500-3000, the head is confounded with the step-size rule; run rung 3 on the A head before judging. The ramp raises that risk: at 1e-3 the bound gradient on W is at parity with the surrogate (17.2 vs 17.7-22.7; IM-F5).
- **Retry and fallback:** if A fails a manipulation read, rerun it once. If it fails again, the next head is an ordinal head (K = 11, ≈ 200 lines) or a tanh head with executed-action entropy ≥ 0.0025 (IM-F1 spec): a within-PPO head change, not a §5.2 escalation.

**Scoring.** Average model_2500, model_2700 and model_2999, and judge with a pre-registered script: cell means, pooled within-cell SD with its t df, and a permutation null over all runs. Q decides at R_Q re-derived at t3 (MQ D00bs: R_seed 0.319, R_eval 0.252). Before the screen, run a process-floor pair (the control head through the driver and through the script; abort if |Δ| > R_eval).

**Pre-registered predictions:**

| Block | Role | Predicted A − C0 | Predicted B − A |
|---|---|---|---|
| Q (executed jitter, FFT) | decides | better | worse or tie |
| Saturation | manipulation only | −0.4 or more | tie |
| Side ratio (two-direction) | outcome; a class shift needs ≥ 0.10 | tie (±0.05) | tie |
| T (tracking) | estimate, 90 % CI | tie or slightly worse | better |
| F, F_mt (turning) | estimate, 90 % CI; no claim below the pooled-t MDE (≈ 0.28 / 0.35 at 3 seeds, ≈ 0.20 / 0.25 at 5) | tie | tie |
| Survival | tripwire: cell-mean drop > 0.30 in fwd, diag or move_turn (§2.4) | tie | tie |

A missed predicted sign is recorded as a prediction failure.

**Decision:** adopt the head if Q improves and no tripwire fires. Claim a task gain only with a win in the judge script that INC0 does not reproduce. Otherwise apply §5.2. **Bisection, pre-registered:** if A fails a tripwire or loses Q, re-run at 2 seeds each, in order: (1) A with entropy 0.01; (2) A with init σ 1.5; (3) the bound loss alone on the incumbent σ path. Stop at the first arm that recovers C0's Q and survival.

**`C_lr_bound` (conditional; ≈ 1.25 GPU-h per seed; a side-bisect arm).** The A head on the two-direction C_lr bed. It tests candidate (ii) directly: 70 % of rod means are past the clip at iteration 300, before the classes separate at iteration 540 (RC:248-252). It runs only on the §6.3 C_40 branch. Early read: training track_ratio_mean averaged over iterations 800-1000; stop at ≤ 0.48; ≥ 0.58 is forming FAST. C_lr runs RSI off (SB/scripts/candidates_v15.yaml:366), so no RSI-start read exists on this bed.

### 6.3 Canonical training (owned by the side bisect)

The side bisect names it the fix, "expected FAST under every candidate", runnable "at any point" (RC:467-470, §6 :473-509). Its wrapper is the one piece of new code the side bisect needs, and it lands first (§6.0).

**Order:**
1. **C_lr_canon** (≈ 1.2 GPU-h per seed): the C_lr overlay plus `KRABBY_OMNI_CANON=vy`, fa mirror kept, RSI off. Early read: track_ratio_mean averaged over iterations 800-1000; ≥ 0.58 is forming FAST; stop only at ≤ 0.48, the charter's uncalibrated SLOW line (SB/scripts/early_read.py:4-5); in between, run to 3,000. The kill rule is the side bisect's call.
2. **FB_canon:** M_x images on every row, M_y / R_z(π) images only on rows with |vy_b| < h (RC:481-482).
3. **P1_canon** against W (≈ 5.4 GPU-h per seed).

**C_40 branch** (CH P10 :666-673; records :686-703):
- C_40 s3 is DEAD with a fall phenotype that "neither supports nor excludes (ii)", because it also halved the moving share. C_40_s2's early read of 0.180 already excludes FAST on schedule.
- **SLOW with the shuffle phenotype, or FAST only after iteration 800** (P10: "(ii) contributes"): run C_lr_bound once as the charter's rate-lever test. The charter puts it first on a SLOW result, the moving-share confound weakens that, and the order relative to C_lr_canon is the side bisect's call.
- **DEAD:** (ii) stays open but untestable with the designed arms, and side speed stops being a decision gate for rungs 2-3.

**Deploy.** A canonical-trained head cannot be distilled or deployed until canonicalisation exists in the student branch (eval refuses `--canon` for students, eval_crab_hex_gait.py:719-721) and in the runtime (§4.1). Canonical training must keep the fa mirror (RC:486-488), which is why rung 6 comes after it.

### 6.4 Later rungs

- **Rung 3: step-size rule** (conditional on a rung-2 ADOPT; ≈ 7-21 GPU-h).
  - **Bed:** a tail fork of the adopted rung-2 t3 heads (c4 → c6, or a 2,000-4,000-iteration c6x-style fork), where the throttle binds: from-scratch 3,000-iteration arms sit at the LR floor only 31-41 % of the time (DOS :231), fine-tunes 46-98 % (DOS :690).
  - **Arms:** fixed LR 1e-4 with a per-epoch KL early stop at 0.02 (hard stop at 4 × desired_kl); a recalibrated per-epoch step-KL; a fresh-control fork. 3 seeds.
  - **Mechanism read:** executed change per update ≥ 2 × control by fork iteration 300. **Kill:** approximate KL > 0.1 sustained, or reward −30 % over 500 iterations.
  - Requires the §4.2 fix (with the DAgger LR kwarg) and KL telemetry under every schedule. Predicted task effect: tie.
- **Rung 4: batch geometry** (conditional on rung 1; ≈ 10 GPU-h).
  - **Arm:** 1,024 envs × 24 steps with the minibatch size held at 384 (64 minibatches), or the minibatch count varied as a separately labelled factor; a fresh 256-env control; 3 seeds.
  - num_steps_per_env stays at 24, with no 96-step arm: step-keyed ramps (reward weights, clock-hide) would run 4× ahead of iteration-keyed ones (priv_reg, DAgger, max_iterations; IM-F12).
  - Judge on wall-clock to the incumbent's reward, plus non-inferiority. A new env count is adopted once, at a campaign boundary, with a re-baseline of every threshold calibrated at 256 envs (the 0.48/0.58 early-read bands, R_eval/R_seed, the dead_walker floor, rewbuffer windows).
- **Rung 5: separate critic Adam** (conditional, ≈ 9.5 GPU-h). Runs only if explained variance in the first 200 iterations after a C1d stage boundary drops more than 0.1 below its within-stage level. The motivating collapse is forward-lineage (DOS :661-664); the omni c6x shock recovered by iterations 450-534 (DOS :468). Reward-clip removal and the time-out split go to a reward campaign.
- **Rung 6: symmetry augmentation vs mirror loss** (conditional, after canonical training, ≈ 15.8 GPU-h).
  - **Scene:** forward flat walk on A15+B with the L/R mirror; the forward runner needs wiring for the knob (IM-F19). On omni, only inside canonical training, with M_x images on every row and M_y / R_z(π) images only on rows with |vy_b| < h, so no phantom −y rows return.
  - **Arms:** `KRABBY_SYM_AUG` with mirror coefficient 0 vs mirror loss 0.5, 5 seeds each; the loss may move to clip(μ), which is exact because the maps are signed permutations (AP-A6).
  - **Success:** beat the mirror-loss arm on tripod and fwd/back ratio at 5 seeds, or stop. Decide whether the estimator and priv_reg see mirrored rows (PPO file :322-334, :352-368); add an arm-identity check on `use_data_augmentation`. Prior: E1 seeds 0.541 / 0.0 / 0.026 / 0.0 (DOS :636).
- **Package confirm** (conditional on a rung-2 ADOPT; ≈ 24-40 GPU-h). The adopted package vs a fresh incumbent chain, each forked from its own rung-2 t3 heads through c4 → c5 → c6 → c6x (9,000 iterations, ≈ 4.0 GPU-h per seed-arm with evals; DE-F3), fork stages seeded 16-18 (→ 20). Decision at the c6x end, averaged over +3,000 / +3,500 / +4,000; light__omni is WATCH only. Alternative: one extra arm inside the side bisect's P1_canon lineage (≈ 5.4 GPU-h per seed; SK-S5). A t3 screen cannot certify W-level turning: t3 facing is 0.536 vs W's 0.787, and 0.475 → 0.787 happened inside c6x alone (policy/README.md:142, :146).
- **Cut: γ 0.995.** No evidence base (LIT §14 :522); the nearest evidence points lower (0.97-0.983). The half-life is already 69 steps = 1.38 s, and doubling the horizon rescales V against the ±0.2 value clip and the ≥ 0 reward clip (DOS :149, :160).

### 6.5 Budget

Assumptions: 1.1-1.2 GPU-h per 3,000-iteration arm (DOS :767); ≈ 7.5 min per suite__omni eval (DE-F6 stamps), three per arm, so ≈ 1.575 GPU-h per arm; ≈ 4.0 GPU-h per seed-arm for a c4 → c6x fork (DE-F3); 9.3 GPU-h/day (DOS :73).

| Tier | Item | GPU-h |
|---|---|---|
| **(a) always** | Rung 0: 4 smokes; second-landing B<k> reruns (s3+s2 on 1-2 live ladders); eval_sha re-evals | 2.6-5.3 |
| | Rung 0b | 0.1 |
| | Rung 1: 5 short runs | ≈ 1 |
| | Rung 2: C0, A, B × 3→5 seeds (9-15 runs) with evals; process-floor pair | 14.4-23.9 |
| | **Tier (a) total** | **18.1-30.3** |
| **(b) if rung 2 adopts** | Package confirm: 2 chains × 3→5 seeds × ≈ 4.0 | 24-40 |
| | Rung 3: tail fork, 3 arms × 3 seeds | 7-21 |
| | **Tier (b) total** | **31-61** |
| **(c) if rung 1 is favourable** | Rung 4: 1,024-env arm + 256-env control × 3 seeds | ≈ 10 |
| | Re-baseline at the new env count, if adopted (s3+s2) | ≈ 3.2 |
| **(d) conditional** | INC0, 3→5 seeds (a task block differs) | 4.7-7.9 |
| | C_lr_bound, s3 → s2 (C_40_s2 SLOW or FAST-late; side-bisect queue) | 1.25-2.5 |
| | Rung 5, 6 runs (EV trigger) | 9.5 |
| | Rung 6, 10 runs (after canonical training) | 15.8 |
| | Bisection, 6 runs (A fails a tripwire or loses Q) | 9.5 |
| | A retry + fallback head, ≤ 6 runs (A fails manipulation twice) | ≤ 9.5 |

- **Minimum path** (rung 2 rejects at 3 seeds, rung 1 unfavourable, no trigger fires): ≈ 18 GPU-h, 2 days.
- **Adopt path**, tiers (a) + (b): 49-91 GPU-h, 5-10 days.
- **Maximum:** ≈ 140 GPU-h (≈ 15 days) on the adopt branch with every tier and conditional at its upper bound (the bisection and retry cannot co-occur with an adoption); ≈ 98 GPU-h (≈ 10.5 days) on the reject branch.
- Early kills (≈ 0.4 GPU-h per failed arm; DOS :772) and the §5.2 stop rule favour the low end.
- **Not counted:** canonical training (the side bisect's budget, ≈ 20 GPU-h, plus its landing's B<k> round of ≈ 2.4-4.8); TD3-lean (not scheduled; ≈ 3-4.5 GPU-h and ≈ 1,000 lines).

---

## 7. Sequencing

1. **Now:** nothing lands under the code or eval pathspecs; the §4.4 replay fix may (it is outside both). Write the canonical wrapper and the rung-0 code on worktree branches.
2. **Terminal line plus the user's C_40_s2 judgement:** first landing (the canonical wrapper; the bound loss only if C_lr_bound goes first), its B<k> reruns and re-evals, then C_lr_canon.
3. **Side bisect:** C_lr_canon → FB_canon → P1_canon; C_lr_bound on the C_40_s2 SLOW or FAST-late branch, in the order the side bisect chooses.
4. **This ladder, at its own campaign boundary:** second landing, its B<k> reruns and re-evals, rungs 0b and 1 on an idle GPU, then rung 2. On an ADOPT, rung 3 and the package confirm; if rung 1 is favourable, rung 4. Stop per §5.2 at any point. Runs are serial: two processes give no aggregate gain (DOS :332-336).
5. **Conditional:** INC0, the bisection and the retry/fallback (§6.2); rung 6 after canonical training; rung 5 on its trigger.

---

## 8. What would change this verdict

- **Toward an algorithm swap:** all four §5.2 conditions plus the binding-constraint condition, with a task failure that survives both canonical training and the construction fix; or a TD3-lean run you choose that walks FAST on both hands where its matched tanh-PPO arm does not.
- **Toward "the construction limits tasks":** A or B winning a task block in the judge script, not reproduced by INC0. This report predicts that will not happen.
- **Toward dropping rung 2:** rung-0 telemetry on fresh runs (the B<k> reruns carry it) showing the executed-KL share already ≥ 0.7 through iteration 1500.
- **Toward side speed as an optimizer question again:** C_40_s2 SLOW or FAST-late, and C_lr_bound FAST where C_lr was SLOW.
- **Carried, not verified here:** the 7.5-min eval time; 0 `terminated_other` in 86,440 omni episode records (D9 time-out mislabel ≈ 0 on omni); the critiques' CPU estimates (AP-A1, SK-S2, IM-F5, SW's TD3-lean cost and package claim, IM-F18), whose scripts sit in the session scratchpad under /tmp, which a reboot clears. Verified in this revision: the MDEs and false-trip rates (scipy), the 18-dim noise cost, the Q-block clipping and the runtime path.

---

## 9. Answer, in two lines

PPO is a reasonable choice: nothing in the 602 runs or the literature justifies testing another algorithm family now, and the one off-policy parity result needed 32× Krabby's env count. Its never-tuned construction (unbounded mean behind a binding clip, clamped σ, per-minibatch KL rule) deserves one pre-registered bundled test for learning signal and motion quality, with the step-size and batch rungs only on triggers, while the biggest open task failure, two-direction side walking, gets its structural fix (canonical training) first, in the side bisect.

---

## 10. Check log

| Id | Disposition | Note |
|---|---|---|
| R0, I11, X24 | fixed | Report on disk; critiques and draft 1 in artifacts/decision/; second check ran on this revision. |
| R1, I4, I7, I8 | fixed | Payoff is signal, smoothness, hardware; phantom KL labelled a re-enactment; verdict matches DOS :570-571, :680-692. |
| R2, X18, X19, V6 | fixed | TB's −0.275 is per max-episode-second (≈ −0.34 per alive-second); the first revision's "correction" was backwards. One Monte Carlo, two baselines; SK-S2's added. 18-dim cost recomputed. |
| R3, R4, X20 | fixed | Budget re-tiered and recomputed; DAgger LR coupling and every-schedule KL added. |
| R5, X1, I3, X2, X13, X14, I1, I5 | fixed | Seeds 11-15 / 16-20; LIT :567-569 quoted; bisection; INC0; judge script; tripwire 0.30. MDE/df rows recomputed with scipy and match. ±0.084 dropped, not relabelled an eval-noise floor (that floor is ≈ 0.044, D00s). |
| I6 | rejected | Superseded by X1: seeds 1-3 still include s2, s3. |
| R6, V4, I12, X22 | fixed | Eval pathspecs also block and move eval_sha; sentinel `DONE controls\|STOP`; B<k> ≈ 2.4 per ladder. |
| R7, I2, X27, X3, X16 | fixed | Two landings, wrapper first; C_40 branch per CH P10; charter bands; kill rule left to the side bisect; C_lr_bound's RSI read dropped. |
| R8, X7, X8, X9, X11, X25 | fixed | TD3-lean user-choice exception; binding-constraint condition; FPO++ wording; REPPO; pip warning; preprint caveat. |
| R9 | partly | Erratum in §4.4; DOSSIER.md not edited. |
| R10, V1, V2, V5, X5, I15 | fixed | Runtime checked; one merged deploy statement (§4.1). |
| R11, R12, I10, X28 | fixed | γ cut; rung 5 split; rung 6 scene, success rule and mask; `terminated_other` count carried. |
| R13, V7, I9 | fixed | Parent-class history reset, delay dependence, ≈ 0.5-s braking; `env` defined. |
| R14, V10, V11, X6 | fixed | Policy kwargs verified silent; fail-loud extended; soft-σ contract and Policy/effective_std; bound-loss work re-sized. |
| X4, X15, X21, X10, X23 | fixed | Rung 3 tail fork; rung 4 at 24 steps with re-baseline; confirm as forks on seeds 16-20; retry, fallback, ramp risk; mirror magnitude; distil on clip(μ). |
| X12 | partly | Kill reads and slope rule adopted; the 0.8 inert-knob threshold became a wiring read, since the incumbent is only at 0.71 at iteration 199 (DOS :90). |
| I13, I14, X17, X26, V3, V8, V9 | fixed | Wording and path corrections as listed. |
| (own) | added | σ and executed-KL reads at end of training: the incumbent is at σ 1.84 at iteration 1000 and 70-79 % behavioural KL during formation. |

# RL algorithm and PPO hyper-parameter review: decision report

*2026-09-30, revised 18:40 after the independent check (issues R0–R14; dispositions in the "Check log", §10). Read-only analysis on CPU; nothing here launches a run. Inputs:*
- *the evidence dossier `DOSSIER.md` beside this file (602 PPO runs, claims C01–C16 each verified twice);*
- *the literature review `../lit-review-rl-algorithm-and-ppo-hyperparameters.md` (143 references) and the earlier `../lit-review-gait-motion-quality.md`;*
- *five critiques of a first ladder (skeptic, switch-advocate, design, implementation, action-path audit), not kept as files;*
- *the live side-bisect records, and code re-read for this revision.*

Path shorthand:
- `P` = krabby-research/parkour
- `T` = P/parkour_tasks/parkour_tasks
- `DOS` = DOSSIER.md, cited by line number
- `LIT` = the literature review, cited by section, with `:line` where precise
- `SB` = T/crab_hex_omni_task/experiments/2026-09-28_side_bisect
- "PPO file" = P/scripts/rsl_rl/modules/ppo_with_extractor.py
- "runner" = P/scripts/rsl_rl/modules/on_policy_runner_with_extractor.py

**Live work this report must not disturb.** The side bisect's control queue is running (P/outputs/omni_plan/side_bisect/queue.log; commits cbf8ad06, b48e7ee6):
- C_fa s3: FAST (0.211/0.206 m/s).
- C_40 s3: DEAD (0.035/0.031 m/s; falls end 0.75 of episodes).
- C_40_s2 (X23): training since 17:41, early read 0.180.

Any code edit under `P/scripts`, `P/parkour_isaaclab` or the crab config/mdp trees blocks that queue (§6.0).

---

## 1. Verdict

1. **Keep PPO. No other algorithm family is worth testing now.**
   - No task failure in the ledgers is attributed to the optimizer. The dossier's scoreboard gives it exactly one row, motion quality (DOS §4.8 :680-692). It says evidence that the optimizer limits task metrics "is absent either way" (DOS :993).
   - The closest off-policy analogue reaches parity with PPO only after the same fixes proposed here, and still has a wall-clock gap and more seed variance (LIT §10 [40]).
   - No paper compares off-policy methods with PPO on a legged robot at 256 envs or fewer (LIT §14).
2. **The construction around PPO is broken and worth fixing, for signal, smoothness and hardware, not for an expected task gain.**
   - Unbounded mean against an executed clip(a, ±1): 97.0 % of W's pre-clip components have |μ| > 1 (median 11.2). All 303 checkpoints past 999 iterations are 91-100 % saturated (DOS :87-91).
   - σ sits behind an absorbing zero-gradient clamp at 2.0: 59/59 fresh runs of 3500+ iterations pin, and none of the 303 pinned runs ever leaves (DOS :84-86).
   - The per-minibatch KL rule erases the configured LR inside iteration 0 in 240/254 fresh runs. On W only 11-16 % of the KL it reacts to changes an executed action (DOS :92-104).
   - The measured consequences are chatter (2.23 sign flips per joint-second at stand; DOS :496) and a mostly phantom learning signal on late heads. The expected task outcome, pre-registered in §6.2, is "no change".
3. **The largest available task-metric gain is structural.**
   - Every two-hand network is SLOW or DEAD (0.028-0.103 m/s); one-hand networks are FAST (0.184-0.217 m/s).
   - The exact M_y canonicaliser (mirroring obs and actions across the y-axis) makes 4/4 fast heads FAST on both hands with no retraining, |L−R| ≤ 0.006 (DOS :566-572).
   - Canonical training is the fix, and it belongs to the side bisect (§6.3).
4. **Three hyper-parameter areas are worth a controlled test:**
   - the action head together with the σ parameterisation, as one bundled arm;
   - the step-size rule;
   - batch geometry, if the throughput probe shows more envs are nearly free.

   **Not worth testing now:**
   - γ;
   - the 2×2 off-diagonal cells;
   - reward-clip removal and time-out relabelling (these belong in a reward campaign);
   - other head families.
5. **Cost:** ≈ 81-115 GPU-h core, ≤ 141 with conditional rungs. That is 9-15 days at the census rate of 9.3 GPU-h/day (528 GPU-h over 57 days; DOS :73). The lit review's "50-60 GPU-h" (LIT §15 :598) priced training arms only.
6. **Fix now, regardless of experiments** (code only; land at a campaign boundary):
   - the export path has no action clamp (§4.1);
   - a fixed-LR resume silently trains at the checkpoint's LR (§4.2);
   - the omni replay tool clips rods at ±4.8 (§4.4).

---

## 2. What the experiments say

### 2.1 Never varied (DOS :72-81, §5.2 :722-749)

Every one of the 602 PPO runs used:
- **Optimisation:** epochs 5, minibatches 16, desired_kl 0.01, adaptive per-minibatch schedule, clip 0.2, max_grad_norm 1.0, one Adam group.
- **Returns:** γ 0.99, λ 0.95, clipped value loss, reward clipped at ≥ 0.
- **Batch:** 24 steps/env; 256 envs for every RL run longer than 20 iterations.
- **Policy:** no observation normalisation; scalar clamped σ; entropy 0.01 (one smoke at 0.005); the same algorithm family.

The only σ experiment, M1, was a fine-tune of an already saturated head with no bound on μ (DOS :498-509).

### 2.2 What moved task metrics

| Lever | Class | Effect | Source |
|---|---|---|---|
| Yaw-progress weight 3.0 (D10) | reward | turn facing 0.419 → 0.748 (REJECTED: move_turn survival 0.580 → 0.120) | facing_tracking_ladder/CHANGELOG.md:72 |
| C1d turn-first curriculum (D12) | curriculum | 0.419 → 0.618 (adopted by overrule) | same file, :89-93 |
| c6x clock removal, on a throttled optimizer | task/structure | 0.475 → 0.787 | clock_deprecation_w2/REPORT.md:163 |
| fa mirror (+ scalar obs), +y-only runs | structure | side 0.084 → 0.184-0.217 m/s in 4/5 runs; C_fa (fa mirror alone) 0.211/0.206 | DOS :571, :639; SB/CHARTER.md:686-690 |
| Wide plants vs golden | morphology | obstacle onsets 0.31 vs 0.69-0.95 | DOS :405-408 |
| Clock + good RSI bank | structure | failing seed 2 → tripod 0.496 | DOS :404, :559 |
| **M1 (σ cap 0.5, reset 0.4) — the only optimizer change** | optimizer | **Q z −1.36 vs R 0.32 (WORSE); move_turn ratio 0.504 → 0.433 (guard WORSE); turn facing 0.610 → 0.510 (z −2.07, a TIE at the seed half-width)** | gait_motion_quality/REPORT.md:166-167, :181-255 |

Saturation does not select the side-speed class:
- Fast and slow side heads are both 0.941-0.978 saturated (DOS :302, :515).
- σ, LR, entropy and clipped-rod share overlap completely across 17 arms (DOS :573-577).

The earlier motion-quality review predicted that capping σ would help (lit-review-gait-motion-quality.md §10 item 3). M1 contradicted that: it is a recorded prediction failure.

### 2.3 Where the construction shows up

- **Bang-bang output.** The executed action is effectively sign(μ) on ≈ 94-97 % of components. Sign flips run at 2.23 per joint-second at stand and 1.83 while moving, and stand contact switching is 1.65 Hz (DOS :91, :496).
- **Phantom KL.** 84-88 % of W's KL buys no executed change, so executed change is 5-10× below what desired_kl nominally allows (DOS :185). During formation the KL is 70-79 % behavioural and the LR is 3-6× higher (DOS :94, :251), so the throttle bites late, not during formation.
- **Dependence on the clip.** Without the ±1 clip the teacher fell in 100/100 episodes (DOS :512).

### 2.4 Statistics

- **Seed-dominated turning blocks.** R_seed/R_eval is 3.19/0.97 and 4.12/0.94. Single-seed MDEs (minimum detectable effects) are 0.145 for turn facing and 0.180 for move_turn facing (facing_tracking_ladder/CHANGELOG.md:46-47). With 3 seeds these fall to ≈ 0.084 / 0.104.
- **MDE at 80 % power, α 0.05 two-sided** (noncentral t, re-derived for this revision), in units of the seed SD:

  | Comparison | MDE |
  |---|---|
  | 3 vs 3 seeds (pooled df 8) | 2.61 |
  | 5 vs 5 | 2.02 |
  | 2×2 main effect, 3 seeds per cell | 1.85 |
  | 2×2 interaction | 3.70 |

- **Formation-rate tests (Fisher, two-sided):** 3/3 vs 0/3 p = 0.10; 5/5 vs 0/5 p = 0.008; 4/5 vs 0/5 p = 0.048. Formation is at ceiling in clock-held omni t3 (all 6 runs formed; DOS :403), so formation rate cannot decide there.
- **Same-seed pairs are not reliable controls.** 2 of 4 same-seed pairs diverge from iteration 20 (DOS :420-427, D18).
- **Eval process state matters.** An env write after Kit start flips some evals between 0.79 and 0.75 (memory eval-process-state-sensitivity.md).

---

## 3. What the literature says

- **It indicts the construction, not PPO.** No reference legged stack pairs a binding ±1 clip with an unbounded mean and a clamped σ at or above the bound (LIT §2). rl_games adds a quadratic bound loss outside ±1.1 (coefficient 1e-4 to 1e-3); VideoMimic adds an L1 bound loss against "bang-bang style control" (LIT §2-§3).
- **σ alone cannot fix saturation.** Lowering σ alone cannot produce an interior mean once means are past the bound ([3]). M1's failure is what that predicts.
- **Bounded heads change geometry, not reliably return.** They change geometry reliably and task return inconsistently ([3, 4, 22]). The two legged positives are weak: DHAL used 4 seeds, and GPO does not report seeds ([19, 25]). Near-bang-bang control may even be close to optimal for this kind of objective ([26, 27]).
- **Step-size rule.** Krabby uses Rudin's Algorithm 1 on minibatches 64-96× smaller than it was tuned on. The only controlled fixed-vs-adaptive comparison has an interval spanning zero (LIT §6 [41, 48]).
- **Batch size.** Krabby's batch is 8× below Rudin's smallest. Rollouts under ≈ 25 steps hurt gait discovery (LIT §7).
- **Off-policy.** RSL-RL-SAC reached parity with PPO only at 8,192 envs, after σ0 0.15, a tanh head and time-out-aware targets, and it keeps a wall-clock gap and more seed variance ([40]). Its abstract says it "closes the performance gap with PPO entirely": parity, not superiority. FPO++ is the only variant with legged results at Krabby's scale, and it was compared against a default-configured Gaussian baseline ([97]).
- **Two-hand failure.** There is no precedent for a slow-in-both-hands mode. Canonical training has one weak analogue (LIT §8).

**Corrections to the literature review** (the review file itself is not edited):
- **§4 :173-174** describes action_rate as squared and "near-constant under saturation". It is actually the **unsquared** L2 norm of the delayed, clipped executed action (P/parkour_isaaclab/envs/mdp/rewards.py:337-343; squared only under `KRABBY_ACTION_RATE_SQ`, default off). W's training run logs −0.275/s over its last 500 iterations (TB, X03_c6x, weight −0.3).
- **§2 :89-90** asked which action the history observations carry. Answered: the history stores the clipped action, executed with a one-step delay (T/crab_hex_forward_task/mdp/parkour_actions.py:348-356), so the PR #2019 loop does not apply.
- **§15 :570** says "the deploy clip stays". There is no deploy clip (§4.1).

---

## 4. Defects found while checking (fix regardless)

### 4.1 The export path does not clamp actions

- **What happens.**
  - The deploy exporter substitutes the estimator and returns the raw actor output with no clamp (P/scripts/rsl_rl/exporter.py:112-130).
  - The teacher exporter runs the actor with no estimator and no clamp (:53-62).
  - The ONNX export traces `torch.zeros(1, in_features)` with in_features = 1007, against a live 1,149-dim observation, with `dynamic_axes={}` (:97-107; DOS :167).
- **Why it matters.** W's raw means have median |μ| 11.2 and p90 22.0. Without the clip the teacher fell 100/100. I did not check whether the hardware runtime clamps downstream.
- **Fix.**
  - Clamp inside every exported module and feed the clamped action back as the previous action.
  - Export the teacher with its estimator.
  - Trace ONNX on a 1,149-dim input.

### 4.2 A fixed-LR fork silently trains at the checkpoint's LR

- **What happens.**
  - `param_group["lr"]` is only written inside the adaptive branch (PPO file :377, :403-404).
  - `runner.load` restores the optimizer state, which carries the saved LR (runner :737).
  - `load_resume_state` only restores `learning_rate` under the adaptive schedule (PPO file :558-560).
  - TensorBoard logs the configured LR, not the applied one (runner :543).
- **Knock-on effect.** Actor and critic share one Adam group (PPO file :113), so any schedule change also moves the critic's LR by 3-30×.
- **Fix.**
  - Write the configured LR through to every param group after loading the optimizer.
  - Log the mean applied LR.
  - Add a unit test that resumes under `fixed` and checks the applied LR.
  - Label the step-size rung "actor+critic step size".

### 4.3 action_rate taxes exploration noise, which a bounded mean exposes

- **Mechanism.** During training the term reads the delayed, clipped *sampled* action (rewards.py:337-343; parkour_actions.py:348-356; vecenv_wrapper.py:134-136). A saturated mean hides its sampling noise from the term: at σ 2, |μ| = 9 costs 0.000 while |μ| = 0 costs 0.95 per dim per step.
- **Monte Carlo.** CPU, W's eval states, 88,697 steps, weight 0.3:
  - W's own means cost 0.287/s deterministic and 0.421/s sampled at σ 2.
  - The same means clipped to ±1.1 (what a bound loss would produce) cost 0.566 / 0.975 / 1.252 / 1.407 per s at σ 0.5 / 1 / 1.5 / 2. That is +0.27 / +0.65 / +0.89 / +0.99 per s over W at the same σ.
  - The action-path audit's independent Monte Carlo gave +0.15 / +0.55 / +0.83 / +0.99.
- **Why it matters.**
  - The fast side gait out-earns the slow one by only +0.25 to +0.43/s (SB/ROOT_CAUSE.md:263). The tax is the same size as the margin that selects the gait.
  - The tax also rewards PPO for shrinking σ fast.
  - Smoothness taxes have cost completion before (gait_formation_v2: L2 0.81, L3 0.64 vs 0.90-0.94).

### 4.4 The omni replay tool clips rods at ±4.8

- **The bug.** `ACTION_CLIP = [1.0]*6 + [4.8]*12` (T/crab_hex_omni_task/experiments/2026-09-16_1805_omni_formation/replay_reward_economy.py:100, applied at :391). The runner clamps everything to ±1. Commit 3714934c fixed only the clock replay tool.
- **Erratum for DOS §4.4(a) :548.**
  - The dossier's walker/creeper action_rate figures (−0.516 / −0.198) come from the bug.
  - At the ±1 clip they are −0.204 / −0.132 deterministic, and ≈ −0.692 / −0.306 sampled at σ 1.9.
  - The ordering is unchanged: the net gap is 1.41/s, and foot_idle (−1.747 vs −0.441) dominates it.
- **Fix.** Read the clip from each head's agent.yaml, and add a sampled-σ mode to every replay tool.

### 4.5 RSI spawns start by braking their cams

- **The sequence.**
  - The reset event writes the bank's joint velocities (crab_hex_rsi.py:96; env :263).
  - The action manager then resets (env :272) and zeroes the cam velocity target (parkour_actions.py:191) and the delayed-action history.
  - So step 1 executes a zero action, and the cams re-accelerate at the 2π rad/s² limit (parkour_actions.py:290-291).
- **Why it matters.**
  - RSI starts underperform normal spawns in every arm, including the fast C_mo (failure 0.127 vs 0.021; SB/ROOT_CAUSE.md:213-218, :253-256).
  - That deficit is currently cited as "RSI inert at σ 1.5" (DOS :584, :713; LIT §5 :208). The cam reset explains it without σ.
- **Consequence.** Rung 0b (§6.1) tests this. Until then, the four RSI-seeded two-direction negatives are discounted as σ evidence.

### 4.6 Infrastructure hazards

- **Unknown PPO keys are dropped.** On the PPO path with only a warning (runner :128-141); on the distillation path silently.
- **Worktree runs import the main checkout.** `train.py` adds only `parkour_tasks` to the path (:13-20). `scripts/` and `parkour_isaaclab` resolve through `__editable__.isaaclab_parkour-0.1.pth` to the main checkout, so a worktree run imports the main checkout's PPO.
- **`NUM_ENVS = {"rl": "256"}` is hard-coded** in run_phases.py:71 (used at :1297).
- **The rollout generator drops `batch % num_mini_batches` rows** (rollout_storage.py:226-228).
- **Eval chatter metrics use raw μ with no deadzone** (per the check), so they mean different things for saturated and bounded heads.

---

## 5. Another algorithm? Family by family

| Family | Evidence | Fit now | Verdict |
|---|---|---|---|
| SAC / TD3 | Parity at 8,192 envs after the same fixes; wall-clock gap; more seed variance ([40]). FastTD3 at 4,096 envs ([87, 88]) | Nothing at ≤ 256 envs. Wall-clock is not binding: 90 % of final reward by median iteration 1344-1798 (DOS :489); the learner is ≈ 20 % of wall time (DOS :782) | Not now |
| Model-based | Wins on sample efficiency; the one legged sim-to-real result took ≈ 5.5× the wall-clock | Krabby is not sample-limited (18M samples ≈ 1.1 GPU-h) | No |
| FPO++ | Legged, 2⁸-2¹² envs, 5 seeds ([97]) | Its baseline was a default Gaussian; needs a new head and checkpoint contract | Only under §5.2 |
| Beta / ordinal / GPO heads | Weak legged evidence (LIT §3) | Change the exporter, distiller and checkpoints | Only under §5.2 |
| PBT | DexPBT, PBRL | Needs an env-slice runner first | Later, for the formation lottery only |

### 5.1 Swap costs, corrected

- **Replay buffer.** The dossier's 9.2 GB per 1M transitions (§6 :785-791) is correct arithmetic but uses the wrong design point.
  - Store the 259-dim non-history slice plus next obs in fp32, plus 21 other floats: 2,156 B per transition.
  - That is 0.57 GB for 1,024 transitions per env at 256 envs, or 2.2 GB per 1M transitions.
  - The 10-step history can be rebuilt from sequential storage.
- **Checkpoints.** Any learner can emit a PPO-schema checkpoint with a stub estimator (DOS :800-815). That is work, not a blocker.
- **Consequence.** LIT §15's condition (b) becomes a 10-minute learner-overhead timing inside rung 1, not a gate.

### 5.2 Escalation trigger, inverted

The lit review says to escalate "only if rung 2 moves geometry but not task metrics" (LIT §15 :594-596). That is backwards: geometry moving with no task gain means the optimizer is **not** the task limiter.

Revised rule:
- **Geometry moves, tasks do not** (N ≥ 3, guards held): record "the construction is not a task-metric limiter". Keep the bounded head only if the Q block improves, and stop algorithm work.
- **Any further algorithm arm requires all four:**
  - a pooled, lineage-length task-block win at the seed-inflated R;
  - the predicted telemetry mechanism visible in rung-0 logs;
  - no reproduction by an action_rate-only control;
  - a residual failure after canonical training that optimizer telemetry separates.

---

## 6. Revised test ladder

### 6.0 Preconditions

1. **Wait for the side bisect.** Start nothing until `DONE controls` appears in the queue.log. Watch for it by a log sentinel, never `pgrep -f`. A dirty tree causes 40 refused launches, and then the queue stops (queue_controls.sh:52-59).
2. **Develop on a worktree.** Use a worktree branch with `PYTHONPATH=<worktree>/parkour` prepended. All new knobs default to off. Knob-off runs must be byte-identical in the config dumps, and identical in a 20-iteration same-seed smoke through iteration 19.
3. **Land once.** Put the code for rungs 0-4 and 6 and the canonical-training wrapper in one commit at a campaign boundary. Any `code_sha` change requires `--accept-code-change` and then a `B<k>` baseline rerun on each live ladder (run_candidates.py:159-170, :509-518), about 1.2 GPU-h per seed per ladder. One landing pays that once.
4. **Fail loudly on unknown keys.** Unknown algorithm config keys should raise (whitelist `share_cnn_encoders`).

### 6.1 Rung 0 (code and CPU) and rung 0b (≈ 0.1 GPU-h)

**Telemetry to add:**
- KL and executed-action KL; clip fraction; mean applied LR; explained variance.
- |μ| > 1 share; executed near-bound share.
- Deterministic ‖Δclip(μ)‖ logged next to the sampled action_rate.
- Executed sign flips per joint-second with a |a| ≥ 0.5 deadzone.
- P10/P01 (He et al.'s share of actions near a bound) computed from existing npz files.

**Fixes to land:**
- §4.1, §4.2 and §4.4.
- New knobs: `KRABBY_NUM_ENVS`, `KRABBY_PPO_MINIBATCHES` (with a divisibility assert) and `KRABBY_SYM_AUG`. The forward config currently hard-codes `use_data_augmentation=False` (rsl_rl_ppo_cfg.py:134-145).
- RSI cam-velocity staging, behind a default-off knob.

**Rung-2 knobs:**
- A bound loss on raw μ with soft bound 1.1, about 20 lines per the audited M1b spec.
- A tanh-bounded log-std in [ln 0.05, ln 2].
- The bound loss is chosen over tanh because it needs no change to checkpoints, exporter, eval or distiller.

**Gate:** run the sampled-σ offline replay for arm B's weight-0 action_rate before any GPU time.

**Rung 0b:** a σ = 0 eval comparing RSI spawns with normal spawns, with and without cam staging.

### 6.1b Rung 1: throughput probe (≈ 0.5 GPU-h, idle GPU only)

- **Runs:** one process each at 512, 1,024, 2,048 and 4,096 envs, ≈ 200 iterations.
- **Measure:** collection and learn time, VRAM, PhysX overflow warnings, fall rates.
- **Also:** the 10-minute learner-overhead timing from §5.1.
- **How:** run under `systemd-run` with a MemoryMax cap. No MPS while any campaign is live.
- **Baseline:** 4× the envs cost +5 % wall time up to 256 envs (DOS :322-326); above 256 is unmeasured.

### 6.2 Rung 2: the construction arm

**Bed:** omni t3 recipe from scratch, 3,000 iterations, 256 envs, identical eval environment for every arm.

**No factorial.** The two off-diagonal cells are dead on arrival: a bound alone at σ 2, or lower σ alone, cannot produce an interior mean (LIT §15 :567-569).

**Arms:**
- **C0:** the incumbent, retrained fresh at the new `code_sha`.
- **A:** bound loss (coefficient ramped 1e-4 → 1e-3 over iterations 0-500), plus tanh-bounded log-std with init σ 0.8, entropy 0.0025, action_rate −0.3. No last-layer re-init, mirror 0.5 unchanged.
- **B:** A with action_rate weight 0. This isolates the §4.3 noise tax.

**Seeds:** group-sequential, {1, 2, 3} first, extended to {4, 5} if the result is in the continuation zone. Never use the incumbent's s2/s3.

**Manipulation checks** (n = 2 is enough; kill the arm at iteration 1000 if they fail):
- |μ| > 1 share ≤ 0.5 (the incumbent is at 0.91-0.98);
- σ < 1.9;
- executed-KL share ≥ 0.5.

If A or B sit at the LR floor in more than 90 % of iterations 500-3000, the head change is confounded with the step-size rule. In that case run rung 3 on the A head before judging.

**Scoring:** average model_2500, model_2700 and model_2999.

**Pre-registered predictions:**

| Block | Role | Predicted A − C0 | Predicted B − A |
|---|---|---|---|
| Q (executed flips, jitter) | decides | better | worse or tie |
| Saturation | manipulation only | −0.4 or more | tie |
| Side ratio (two-direction) | outcome; class shift needs ≥ 0.10 | tie (±0.05) | tie |
| T (tracking) | estimate with CI | tie or slightly worse | better |
| F, F_mt (turning) | estimate with CI | tie (±0.084) | tie |
| Survival | tripwire (drop > 0.2) | tie | tie |

A missed predicted sign is recorded as a prediction failure.

**Decision:**
- Adopt the head if Q improves and no tripwire fires.
- Claim a task gain only with a win at the pooled R.
- Otherwise apply §5.2.

**Optional `C_lr_bound` (≈ 1.2 GPU-h).** Run the A head on the two-direction C_lr bed, after C_lr_canon. It tests candidate (ii) directly: 70 % of rod means are past the clip at iteration 300, before the classes separate at iteration 540 (SB/ROOT_CAUSE.md:248-252). Kill reads:
- RSI-start failure within 0.02 of normal spawns by iteration 500 (after the rung 0b cam fix);
- moving ratio ≥ 0.58 at iterations 800-1000.

### 6.3 Canonical training (owned by the side bisect)

The side bisect already names it as the fix, "expected FAST under every candidate", runnable "at any point" (SB/ROOT_CAUSE.md:467-470, §6 :473-509).

**Order:**
1. **C_lr_canon** (≈ 1.2 GPU-h). The C_lr overlay plus `KRABBY_OMNI_CANON=vy`, with the fa mirror kept and RSI off. Kill at iteration 1000 if track_ratio_mean < 0.58.
2. **FB_canon.** M_y images only on rows with |vy_b| < h.
3. **P1_canon** against W (≈ 5.4 GPU-h per seed).

**C_40 branch:**
- C_40 s3 came out DEAD with a fall phenotype, which "neither supports nor excludes (ii)" because the moving share was halved (SB/CHARTER.md:700-703).
- If C_40_s2 is SLOW with the shuffle phenotype, run `C_lr_bound` once, as the charter's rate-lever test (:670-673).
- If it is DEAD or FAST, side speed stops being a decision gate for rungs 2-3.

Canonical training must keep the fa mirror (:486-488), which is why rung 6 comes after it.

### 6.4 Later rungs

- **Rung 3: step-size rule** (≈ 14 GPU-h).
  - Arms: fixed LR 1e-4 with per-epoch KL early-stop at 0.02; a recalibrated per-epoch step-KL; a fresh control.
  - 3 seeds each, on the adopted head.
  - Requires the §4.2 fix. Predicted task effect: tie.
- **Rung 4: batch geometry** (≈ 16 GPU-h; only if rung 1 shows ≤ +15 % wall time at 1,024 envs).
  - Arms: 1,024 envs × 24 steps with 4 minibatches, or 256 envs × 96 steps with 4 minibatches; plus a control.
  - Never fewer than 24 steps.
  - Judge on wall-clock to the incumbent's reward, plus non-inferiority.
- **Rung 5: separate critic Adam** (conditional, ≈ 9.5 GPU-h).
  - Run only if explained variance in the first 200 iterations after a C1d stage boundary drops more than 0.1 below its within-stage level.
  - The motivating collapse is a forward-lineage one (DOS :661-664); the omni c6x shock recovered by iterations 450-534 (DOS :468); critic reset was mixed elsewhere (DOS :711).
  - Reward-clip removal and the time-out split move to a reward campaign.
- **Rung 6: symmetry augmentation vs mirror loss** (conditional, after canonical training, ≈ 17 GPU-h).
  - `KRABBY_SYM_AUG` with mirror coefficient 0, M_x images only on side rows, 5 seeds per arm, both arms budgeted.
  - Decide whether the estimator and priv_reg see mirrored rows (PPO file :322-334, :352-368).
  - Add an arm-identity check on `use_data_augmentation`.
  - Prior: E1 seeds 0.541 / 0.0 / 0.026 / 0.0 vs 0.257 (DOS :636).
- **Package confirm** (≈ 35-58 GPU-h).
  - The adopted package vs a fresh incumbent over the full W lineage (t3 → c6x, 12k iterations, 5.4 GPU-h per seed; policy/README.md:132-136), 3 → 5 seeds.
  - A t3 screen cannot certify W-level turning: t3 facing is 0.536 vs W's 0.787, and 0.475 → 0.787 happened inside c6x alone (policy/README.md:142, :146).
- **Cut: γ 0.995.**
  - No evidence base (LIT §14 :522); the nearest evidence points lower (0.97-0.983).
  - The current half-life is already 69 steps = 1.38 s, longer than Humanoid-Gym's 1.15 s.
  - Doubling the horizon rescales V against the ±0.2 value clip and the ≥ 0 reward clip (DOS :149, :160).
- **Also cut:** the 2×2 off-diagonals, further head families and off-policy arms (§5.2).

### 6.5 Budget

Assumptions: 1.1-1.2 GPU-h per 3,000-iteration arm (DOS :767); ≈ 7.5 min per suite__omni eval (per the independent check); ≈ 0.375 GPU-h of evals per arm.

| Item | Runs | GPU-h |
|---|---|---|
| Rung 0 smokes + `B<k>` reruns | 4 smokes; 1-2 ladders | 1.3-2.5 |
| Rung 0b | evals | 0.1 |
| Rung 1 | 4 short runs | 0.5 |
| Rung 2: C0, A, B × 3→5 seeds, with evals | 9-15 | 14.2-23.6 |
| Rung 3 | 9 | 14.2 |
| Rung 4 | 9 | 16.0 |
| Package confirm: 2 chains × 3→5 seeds × 5.4, with evals | 6-10 | 34.7-57.8 |
| **Core total** | | **≈ 81-115** |
| Rung 5 (conditional) | 6 | 9.5 |
| Rung 6 (conditional) | 10 | 17.3 |
| **Maximum** | | **≈ 141** |
| Canonical training (side bisect's budget, not counted) | | ≈ 20 |

At 9.3 GPU-h/day that is 9-12 days for the core and ≤ 15 days in total. Early kills (≈ 0.4 GPU-h per failed arm; DOS :772) and the §5.2 stop rule favour the low end.

---

## 7. Sequencing

1. **Now:** nothing lands in the shared checkout. Write rung-0 code on a worktree branch.
2. **After `DONE controls` and the user's side-bisect decision:** one landing, then the `B<k>` reruns, then rungs 0b and 1 on an idle GPU.
3. **Side bisect:** C_lr_canon → FB_canon → P1_canon. `C_lr_bound` only on the C_40_s2 SLOW branch.
4. **This ladder:** rung 2 → rungs 3/4 → package confirm, stopping per §5.2 at any point.
5. **Conditional:** rung 6 after canonical training; rung 5 only on its telemetry trigger.

---

## 8. What would change this verdict

- **Toward an algorithm swap:** all four §5.2 conditions met, with a task failure that survives both canonical training and the construction fix.
- **Toward "the construction limits tasks":** A or B winning a task block at the pooled R. This report predicts that will not happen.
- **Toward dropping rung 2:** rung-0 telemetry showing the executed-KL share already ≥ 0.7 through iteration 1500 on fresh runs.
- **Carried from the check, not verified here:**
  - the 7.5-min eval time;
  - the ANALYSIS_M1 deadzone line;
  - 0 `terminated_other` in 86,440 omni episode records, which would put the D9 time-out mislabel at ≈ 0 on omni.

---

## 9. Answer, in two lines

PPO is a reasonable choice, and nothing in the 602 runs or the literature justifies testing another algorithm family now. Its never-tuned construction (action head with σ, step-size rule, batch size) deserves one pre-registered test per lever for learning signal and motion quality, while the biggest open task failure, two-direction side walking, has a structural fix (canonical training) that should run first in the side bisect.

---

## 10. Check log

| Id | Disposition | Note |
|---|---|---|
| R0 | fixed | The report now exists as text. The file write was blocked by the harness, so the caller must save it. Re-run the independent check before the user acts. |
| R1 | fixed, with a nuance | Payoff reframed as signal, smoothness and chatter; expected signs pre-registered per block. M1's turn-facing drop is a TIE (z −2.07); the decided costs are Q and the move_turn ratio guard. |
| R2 | fixed | Unsquared L2 verified (rewards.py:337-343). Chose the A/B weight factor over weight 0 in all cells, so C0 stays the incumbent. Both Monte Carlos shown. W's logged cost is 0.275/s (TB), not ≈ 0.34/s. |
| R3 | fixed | Re-budgeted to 81-115 core, ≤ 141 in total. The check's 115-140 matches the upper end. |
| R4 | fixed | Verified at PPO file :377-404, :558-560 and runner :543, :737. |
| R5 | partly | Diagonal only, 3→5 seeds, manipulation checks split from outcomes, bound loss, survival as a tripwire. Entropy is not held equal to the control (0.0025 vs 0.01): the treated arm is a deliberate bundle, and the tanh inward-force concern does not apply to a bound loss. |
| R6 | fixed | CODE_PATHSPECS, the dirty check, the 40-retry stop and the .pth route all verified. |
| R7 | fixed, updated | The check's data were stale: C_40 s3 is DEAD (b48e7ee6) and C_40_s2 is running. The branch follows the charter's weakened reading. |
| R8 | fixed | Trigger inverted. Replay recomputed at 0.57 GB. Condition (b) moved into rung 1. No TD3 rung. The "matched PPO" wording for [40] stands. |
| R9 | partly | Bug and numbers verified. The erratum is in §4.4; DOSSIER.md was not edited (single-file task). |
| R10 | fixed | Exporter verified. Whether the hardware runtime clamps is not checked. |
| R11 | fixed | γ rung cut, rung 5 split. The 0 `terminated_other` count is carried as unverified. |
| R12 | fixed | `use_data_augmentation=False` verified; rung 6 placed after canonical training. |
| R13 | fixed | Reset order verified (env :263 then :272; parkour_actions.py:191). Path correction: parkour_actions.py lives in T/crab_hex_forward_task/mdp, not parkour_isaaclab. |
| R14 | partly | Arm values fixed; the tanh spec is moot because the bound loss was chosen; knobs and the deadzone added. The ActorCriticRMA leftover-kwargs hazard was not re-verified. |

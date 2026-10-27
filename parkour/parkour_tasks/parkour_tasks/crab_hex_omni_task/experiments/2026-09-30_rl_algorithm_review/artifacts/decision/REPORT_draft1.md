# PPO and its hyper-parameters: evaluation and test plan (2026-09-30)

*Inputs: the evidence dossier (`DOSSIER.md`: 602 PPO runs, claims C01–C16 verified), the literature review (`lit-review-rl-algorithm-and-ppo-hyperparameters.md`: 20 verified claims, 143 references) and five critiques of its test ladder: skeptic (SK), switch-advocate (SW), action-path audit (AP), implementation (IM) and design (DE). Numbers from the critiques' CPU-only probes are labelled as critique estimates, and Appendix A resolves every critique finding. An independent check of this report is still pending. Nothing was run on the GPU and nothing was committed. The side bisect's C_40 control (X22) was training while this was written.*

Shorthand:
- `D` = `DOSSIER.md`
- `L` = the literature review
- `RC` / `CH` = `../2026-09-28_side_bisect/ROOT_CAUSE.md` / `CHARTER.md`
- `P` = `parkour/`
- W = `crab_hex_omni_task/policy/5_c6x_clockfree/model_11995.pt`
- "checked" = a citation re-read for this report.

## 1. Verdict

**Keep PPO.** In 602 runs, no core PPO setting and no other algorithm family was ever varied (D §0.1, §5.2), so the repo holds no evidence against PPO. The closest off-policy study ran only at 8,192 envs, so there is no evidence either way at Krabby's 256 (L §10; SW-A3).

**The hyper-parameters were never chosen.** They came from a 6,144-env Go2 config. Three parts of the setup are mis-set by any reference standard (D §3.1–3.3; L §2, §5, §6):
- An unbounded Gaussian mean sits behind a ±1 clip that binds: 97.0 % of W's mean components lie past it.
- A σ clamp at 2.0 cuts off every σ gradient; all 59 fresh runs of 3,500+ iterations pin there.
- A per-minibatch adaptive-KL rule erases the configured LR inside iteration 0 in 240 of 254 fresh runs.

**What these defects are shown to cause is a bang-bang controller**, with 2.23 sign flips per joint-second at stand. The action_rate reward adds to it: it charges for exploration noise on the executed sample, which pays the policy to saturate (AP-A1). This is a motion-quality and hardware-risk problem. None of the four failures the campaigns fought (turning, obstacle completion, formation, two-direction side speed) is attributed to the optimizer, and saturation is the same in good and bad heads (D §4.8, D:993).

**One exception appeared today.** C_40 read 0.325 over iterations 800–1,000 ("forming SLOW"). Halving the +y sample rate alone appears to slow the side stroke's formation, which makes early saturation lock-in (RC candidate ii) a live task-metric hypothesis for the setup.

**The plan is therefore not a hyper-parameter sweep:**
- Fix the telemetry and the defects.
- Run canonical training for the side gait.
- Run one bundled bounded-mean, gradient-passing-σ setup (BND) twice: on the two-direction side bed as the one task-metric test, and on omni t3 as a motion-quality test with task guards.
- Run the step-size, value and symmetry rungs only on triggers, and cut γ.

**Test another algorithm family only under the §6 triggers.** The first candidate is an in-repo TD3 learner on the side bed if canonical training fails. A result that moves only action geometry means stop.

## 2. What the experiments show

1. **Nothing about the optimizer was ever tested.**
   - All 602 runs share epochs 5, 16 minibatches of 384, desired_kl 0.01, clip 0.2, γ 0.99, λ 0.95, and 256 envs × 24 steps. The 437-run census used about 528 GPU-h (D §0.1, §5.2).
   - The only σ experiment, M1 (KRABBY_ACTION_STD_MAX 0.5 on a fine-tune of the saturated X08:c6), was REJECTED on block Q and two guards. Turn facing_closed went 0.610 → 0.510, a block-F TIE (z −2.02 vs R 2.07; MQ CHANGELOG.md:20, checked).
   - M1 therefore tested a lower σ on a saturated mean, not a lower σ as such (D §4.3).
2. **The setup degenerates.**
   - On W, 97.0 % of pre-clip mean components have |μ| > 1, with median 11.2.
   - All 303 checkpoints at 999+ iterations are 91–100 % saturated; saturation is already 71 % at iteration 199.
   - σ above 2.0 gets zero gradient, and no run that reached 1.99 ever fell below 1.9898.
   - The deterministic controller is sign(μ) on about 95 % of components (D §3.2, §3.3, §10(a)).
3. **The LR controller throttles late heads, not formation.**
   - The LR is at the 1e-5 floor in 62.9 % of iterations.
   - Only 11–16 % of W's measured KL changes an executed action. During formation that share is 0.70–0.79 and the LR is 3–6× higher (D §3.1, D3).
4. **Saturation does not separate good heads from bad.**
   - Fast and slow side heads are 0.941–0.978 saturated alike.
   - Sampling the training noise barely moves speed: vx 0.145 vs 0.151 (D:302, :514).
   - Only the motion-quality row of the attribution scoreboard implicates the optimizer (D §4.8).
5. **Reward, curriculum, structure and plant moved the task metrics.**
   - Turn facing_closed went 0.419 → 0.618 with C1d, and → 0.748 with KRABBY_YAW_PROGRESS_W 3.0, which was rejected on a guard (facing_tracking_ladder/CHANGELOG.md:72, :89, checked).
   - c6x moved it 0.475 → 0.787 with the LR at the floor for 61–64 % of iterations (D:468-469).
   - The fa mirror took one-direction side speed 0.084 → 0.184–0.217 m/s.
   - Morphology took obstacle completion 0.31 → 0.69–0.95 (D:571, :639, :408).
   - The stroke and centring terms failed from scratch too, not only in throttled fine-tunes (D:462-465).
6. **Two-direction side slowness is structural as observed; its mechanism is open.**
   - Every network trained on both hands is SLOW or DEAD (0.028–0.103 m/s).
   - The M_y canonicaliser makes 4/4 one-direction heads fast on both hands without training (|L−R| ≤ 0.006).
   - A fresh network distils the two-hand policy to 99 % sign agreement, so capacity is not the limit (D §4.4(c); RC:48-66).
   - Today C_fa came back FAST (0.211/0.206; CH 16:26).
   - C_40 read 0.295 at iteration 800 and 0.325 over 800–1,000, below the 0.48 "forming SLOW" line (side_bisect queue.log 16:51, checked). Its final class is still pending.
7. **Not optimizer problems:**
   - The forward glide out-earns the tripod under the same reward (28.69 vs 26.15 at 3k).
   - The 1c creeper is penalty-decided, mainly by foot_idle (−1.747 vs −0.441).
   - move_turn would need about 0.58 m/s at the outer foot.
   - The student matches its teacher (0.79/0.64 vs 0.81/0.66).
   - Stage-boundary collapse is a critic-scale problem (D §4.4(a), :602, §4.7, :692).
8. **Design constraints:**
   - The turning blocks are seed-dominated (R_seed/R_eval 3.19/0.97 and 4.12/0.94), and 2 of 4 same-seed pairs diverge (D §4.1, D18).
   - 4× the envs costs +5 % wall time up to 256, and one run uses 61–69 % of the GPU.
   - Nothing above 256 envs has ever run (D §3.5).

## 3. What the literature says

- **No reference legged stack uses this setup.** legged_gym clips at ±100, Isaac Lab does not clip at all, and both start σ at 1.0. rl_games clips at ±1 but bounds the mean (bounds_loss_coef 1e-4 to 1e-3). A σ clamp that binds has no precedent (L §2, §5).
- **Lowering σ alone cannot pull saturated means back inside the bound** [3] (L §2). M1's rejection fits this, but only in hindsight: the previous review ranked a σ cap third (lit-review-gait-motion-quality.md:300-302), and it failed (SK-S15).
- **Bounded heads change action geometry reliably but task return inconsistently.** Reported effects range from −19 to −22 % (He et al.), through a tie (Tang: 4,007 ± 698 vs 3,905 ± 502), to +30 % (tanh, HalfCheetah). The legged positives rest on 4 seeds or an unreported count (L §3).
- **Near-bang-bang control is the expected optimum without a quadratic action cost, and a tanh head will likely converge near ±1**, so smoothness must be priced explicitly (L §4). Correction (AP-A8): the review's remedy, a TV/L1 or action_rate cost on the executed action, carries the same noise tax under sampled training. The shipped action_rate is also the unsquared L2 norm, not the squared form the review describes (P/parkour_isaaclab/envs/mdp/rewards.py:337-343, checked).
- **Step size and batch.**
  - Krabby's rule is Rudin's Algorithm 1 run on minibatches 64–96× smaller.
  - The only controlled test of fixed vs adaptive LR (rl_games, 3 seeds, manipulation) gives an interval spanning −0.1 to +1.0.
  - Rudin found 2,048–4,096 robots best.
  - No γ ablation exists for slow multi-legged gaits (L §6, §7).
- **Direction-flipped skills.** The documented shared-policy failure is preferring one mode, not slowness in both. Analogues of canonical training are thin (L §8).
- **Other families.**
  - [40] ran only at 8,192 envs. Read correctly, its SAC matched PPO per sample on quadrupeds and beat it on humanoids, and it states its wall-clock and seed costs only qualitatively (SW-A3; not re-verified here).
  - FPO++ is the only method with legged results at 2^8–2^12 envs, and it reports lower seed variance (SW-A6).
  - Model-based methods address a sample budget Krabby does not have (L §10).
- **Disagreements with the dossier.**
  - The dossier calls the parameterisation "the wrong shape for the actuator interface" (D §10(a)). The literature says it changes smoothness and learning signal, not task return reliably.
  - The step-size rule matters through what the KL measures, not as a first-order throttle.
  - A narrow σ init can worsen basin selection (L §15 (i), (iii), (v)).

## 4. Defects to fix before any comparison (rung R0)

| # | Fix | Evidence | Knob-off | Cost |
|---|---|---|---|---|
| 1 | Per-minibatch telemetry: Gaussian KL under every schedule; minibatch-0 KL (the unresolved D7 mismatch term, measured for free); clip fraction; applied LR; pre-clip grad norm and the share above max_grad_norm; mean \|clip(μ_new) − clip(μ_old)\|. Per iteration: explained variance (taken before advantage normalisation), \|μ\| > 1 share and censored-KL share. Log under `PPO/`, RNG-free | D §3.7; IM-F16 | identical | code |
| 2 | Log the deterministic ‖Δclip(μ)‖ beside `reward_action_rate`, and the RSI-start vs normal-spawn failure rate | AP-A1; SK-S14 | identical | code |
| 3 | Fixed-LR resume bug: any non-adaptive schedule silently keeps the parent's saved Adam LR (1e-5 to 3.4e-5) while TB logs the configured value. Also give the DAgger optimizer its own LR | ppo_with_extractor.py:377, :403-404, :559-560; runner :737, :543 (checked) | adaptive unchanged | code + test |
| 4 | Unknown `algorithm` keys raise (allow-list `share_cnn_encoders`); the distillation branch logs the keys it ignores; unknown policy kwargs raise (allow-list the 3 isaaclab_rl fields); each head knob prints a line saying it was used, which OVERLAY_DUMP_CHECKS checks | D14; IM-F8, IM-F9 | identical | code |
| 5 | Expose `terminated` and bootstrap only on `time_outs & ~terminated`, default-off. Count the overlaps first: 0 of 86,440 omni eval episodes ended "terminated_other" | D9; IM-F17; SW-A9 | identical | code |
| 6 | `2026-09-16_1805_omni_formation/replay_reward_economy.py:100` clips rods at ±4.8 instead of ±1. Read the clip from agent.yaml, and add a sampled-σ mode to every replay that prices action-reading terms | AP-A2 (checked) | no code_sha move | CPU |
| 7 | Score chatter on executed actions, counting sign flips only between states with \|a\| ≥ 0.5; clip or drop the raw episode-level action block | AP-A5 | eval_sha moves; re-eval baselines | ~0.4 GPU-h per head (est.) |
| 8 | RSI cam reset: reset events (including RSI) run before `action_manager.reset`, which zeroes `_cam_vel_applied`, the delay buffer and the action history. Each RSI start therefore brakes its cams to about 0 and re-accelerates at 2π rad/s², taking 0.5 s to reach π rad/s. Run a σ = 0 eval of RSI spawns vs normal spawns, then add a default-off staging knob | parkour_actions.py:191; parkour_manager_based_rl_env.py:260-273 (checked); AP-A3 | identical | ~0.1 GPU-h |
| 9 | Knob-off identity check: config dumps plus a 20-iteration same-seed smoke that matches through iteration 19 | DE-F9; D18 | n/a | ~0.1 GPU-h |
| 10 | **Before any hardware use (outside the ladder).** The exported graph has no ±1 clamp. The compute stack feeds the raw command back as previous_action. The student export substitutes the estimator, but student training and eval do not (runner :441-444, checked). The teacher export omits the estimator. ONNX traces a 1×1,007 dummy against the live 1,149-dim obs. The `compute/parkour` module copy is stale | AP-A4; IM-F18 | n/a | separate task |

**Dropped:** the review's PR #2019 audit is already answered. The obs slots carry clipped actions (vecenv_wrapper.py:134-136; parkour_actions.py:348-351; AP-A7).

**Corrections to the inputs:**
- **D §4.4(a).** The creeper action_rate figures (−0.516 / −0.198) come from item 6's clip bug. The deterministic values are −0.204 / −0.132, and about −0.692 / −0.306 at the training σ of 1.9 (AP-A2 estimate). The foot_idle-led reading stands.
- **D §6.** The student path does not substitute the estimator (item 10).
- **RC:213-218 and D §5.1.** "RSI is inert at init std 1.5" is withdrawn as σ evidence until item 8 is measured.
- **L §4:171-175.** The action_rate wording is wrong (see §3).

**How to land the code.** The side bisect refuses to launch on a changed code tree: CODE_PATHSPECS covers parkour/scripts and parkour_isaaclab (run_candidates.py, checked).
- Write the code in a worktree.
- Launch worktree processes with PYTHONPATH=<worktree>/parkour prepended. Otherwise the editable install imports the main checkout (IM-F2).
- Land once, after the side bisect's queue logs `DONE controls` and with the user's go.
- Budget one B<k> baseline rerun per live driver ladder, about 2.4 GPU-h.

## 5. The test ladder

**Mapping from the review's rungs (L §15):**

| Review rung | This ladder |
|---|---|
| 0 | R0 |
| 1 and 4 | R1 |
| 7 | R2 |
| 2, cut to one cell | R3 / R4 |
| 3 | R5 |
| 5 | R6 |
| 6 | R7 |
| 8 | cut |
| 9 | moved to §6 |

**Costing:** one 3,000-iteration arm is 1.07–1.33 GPU-h (median about 1.14; D §6), plus about 0.4 GPU-h for three checkpoint evals (estimate).

**Rules for every rung:**
- **Pre-register** the predicted sign per task block, the mechanism read and the kill read, and log misses as prediction failures (SK-S15).
- **Seeds.** Fresh seeds 11–15 for screens and 16–20 for confirms; never s2/s3. The side bisect's own class-level rungs keep its s3 → s2 convention (DE-F5).
- **Controls.** Each rung has its own control at the same code_sha and eval_sha, run in the same window, in randomised order.
- **Driver.** Arms run as driver references and are judged by a pre-registered script (DE-F2).
- **Reads.** Reads average model_2500, model_2700 and model_2999. An arm that is behind but has a steeper final-500 slope is not killed (DE-F4).
- **Eval env.** Every arm gets an identical eval env, with no new manifest env blocks (DE-F8).

**BND (the bundle tested in R3 and R4).** Values are fixed now:
- **Bound loss:** rl_games form on the raw mean, soft bound 1.1, coefficient 1e-4 from scratch. One retry at 1e-3 if the mechanism check fails (IM-F5). Implementation per the M1b spec, `/home/nickmagus/.claude/plans/there-is-a-lot-compressed-unicorn.md:48-100`.
- **σ:** a soft-bounded σ (noise_std_type 'soft', log σ = lo + ½(hi − lo)(tanh x + 1) over [ln 0.05, ln 2]), starting at 0.8 (IM-F7).
- **Entropy:** KRABBY_ENTROPY_COEF 0.0025.
- **Mirror loss:** unchanged.
- **Smoothness:** KRABBY_ACTION_RATE_W 0, plus a CAPS-style term on clip(μ) in the loss. The knob is proposed, and its weight is fixed before launch by a CPU gradient-norm calibration.

**Why action_rate is zeroed in BND.** With the sampled action_rate left on, the comparison would be between reward schemes rather than heads. On W's states, a mean bounded to ±1.1 would pay 0.975/s at σ 1.0 and 1.407/s at σ 2.0. W pays 0.421/s at its training σ, and W's whole net return is about 1.17/s (AP-A1 estimate). SK-S2's noise-only Monte Carlo gives 0.73–1.49/s for bounded heads vs 0.047/s for a W-like head. A μ-based action_rate *reward* is rejected: it does not depend on the sampled action, so PPO would give μ no direct gradient.

| Rung | Question | Arms | Screen task / scene | Its | Seeds | Primary gate + kill read | GPU-h | Depends on |
|---|---|---|---|---|---|---|---|---|
| R0 | Can we see the optimizer, and are the defects fixed? | §4 items 1–9 | CPU; smokes; σ = 0 RSI eval | 20 | 1 pair | Knob-off identity through iteration 19; logged KL = controller KL on CPU; unit tests pass | ~0.2 + ~2.4 per live ladder + re-evals | `DONE controls`; user go |
| R1 | Is running more than 256 envs nearly free? | One process each at 256 / 512 / 1,024 / 2,048 / 4,096 envs | omni t3 config | 200 each | 1 | s/iter, steps/s, peak VRAM, PhysX overflow warnings. Stop a size above 1.5 × 1.318 s/iter (at 1,024 envs), above 10 GB VRAM, or on NaN / dead_plant | ~1 (est.) | R0 num_envs knob; idle GPU; MemoryMax; no MPS |
| R2 | Does canonical training make two-direction side walking FAST? (owned by the side bisect) | C_lr + KRABBY_OMNI_CANON=vy, then FB_canon, then P1_canon vs W | side 1b, two directions; then + fwd/back; then 12k omni | 3,000; 12,000 | s3 → s2 | Early read (its 800–1,000): stop at ≤ 0.48. Final: FAST on both hands, odd RMS ≥ 1.4 (RC P3). FB_canon keeps fwd tripod and fwd/back ratios within guard R of 0.558 / 0.524. On failure, use the RC §6 fallbacks, not algorithm work | ~2.6 + ~2.6; P1 ~5.4 per seed | canonical plumbing (IM-F15); fa mirror only; RSI off or item 8 done |
| R4 | Does BND let a two-direction net form the stroke? (run if C_40 ends SLOW or FAST-late) | C_lr + BND. Control: the record, 9 bisect arms at s3 all SLOW or DEAD. A fresh C_lr control only if BND reads FAST | side 1b, two directions, in the side bisect queue | 3,000 | s3 → s2 (s1 if FAST) | Kill at iteration 1,000 if the read is ≤ 0.48. Success: FAST on both hands at s3 and s2 | 0.4 if killed; ≤ ~5 if FAST | C_40 final class; R0 knobs |
| R3 | Does BND make the controller quieter without costing task metrics? | Stock incumbent t3 vs t3 + BND. INC0 (incumbent + action_rate 0 + the μ term) runs only if any block differs | omni t3 (1a + t3, clock held), suite__omni | 3,000 | 3 → 5, fresh | Mechanism checks (these can only kill): \|μ\| > 1.1 share ≤ 0.30 at iteration 300; σ gradient live and σ > 0.1 at iteration 1,000; DEAD if track_ratio_mean < 0.40 at iteration 1,000. Success: Q BETTER at the pooled seed-inflated R (MQ D00bs had R_seed 0.319 / R_eval 0.252; re-derive at t3), no survival tripwire (cell-mean drop > 0.30), T/F reported with 90 % CIs. Side ratio decides only for a class shift ≥ 0.10 | 9 → 15; INC0 +4.5 | R0 landing; sampled-σ replay gate |
| R3c | Does an R3 adoption survive the W recipe? | Winner vs fresh incumbent through c4 → c5 → c6 → c6x from their own t3 heads, or one extra BND arm inside P1_canon | omni C1d chain + c6x | 9,000 | 3 (5 if ambiguous) | suite__omni at c6x end, averaged over +3,000 / +3,500 / +4,000; light__omni WATCH | 24–40, or ~5.4 per seed in P1_canon | R3 ADOPT |
| R5 | Does removing the late throttle help? | Fixed LR 1e-4 with a per-epoch KL stop (hard stop at 4 × desired_kl) and the critic in its own fixed-LR param group, vs the incumbent schedule | tail fork of the R3 heads | 2,000–5,000 | 3 | Executed change per update ≥ 2× control by fork iteration 300. Kill on approx KL > 0.1 sustained or reward −30 % over 500 its. Success: a final block BETTER | 7–21 (est.) | R3 ADOPT; item 3 |
| R6 | Is the critic starved at stage boundaries? | Separate critic Adam vs incumbent | C1d boundaries t3 → c4 and c6 → c6x | ~500 | 3 | Trigger: EV in the first 200 iterations after a boundary is > 0.1 below its within-stage value. Read: EV recovery and survival dip | 0 unless triggered; 7–14 | R0 EV logging |
| R7 | (optional) Augmentation vs mirror loss | KRABBY_SYM_AUG=1 (proposed) vs loss 0.5; loss computed on clip(μ) | forward flat walk A15+B with L/R mirror; omni only inside canonical training, fa/M_x images only | 3,000 | 5 per arm | Must beat the loss arm on tripod and fwd/back ratio, else stop | ~15 (est.) | forward wiring (IM-F19); IM-F14 |

**Predicted outcome of R3, registered in advance:** Q BETTER; T, F and survival flat to worse (task return moved −22 % to +30 % in the literature; L §3); side class unchanged.

**Cut or folded:**
- **γ 0.995 is cut** (SK-S12).
- **The env-count rung is folded into R1.** Changing num_envs moves R_eval, R_seed and the 0.48/0.58 early-read bands, so a larger count is adopted once, at a campaign boundary, via an incumbent 256-vs-N comparison on 3–5 fresh seeds plus one re-baseline (SK-S13, DE-F6, IM-F11).
- **num_steps_per_env stays at 24.** Changing it puts the step-keyed ramps out of step with the iteration-keyed ones (IM-F12).

**Cost.**
- The core rungs (R0, R1, R2, R3, and R4 to its early read) total about 19–25 GPU-h, which is 2–3 days at the census rate of 9.3 GPU-h/day.
- R3c adds 24–40 GPU-h, only after an adoption. R5–R7 add 29–50, only if triggered or chosen.
- The review's 50–60 GPU-h covered training only. The design critique's full-ladder figure is 115–140 (DE-F6).

**Order:**
1. **R0 first.** Every later read needs its telemetry, and items 3 and 4 would otherwise make later arms silently inert.
2. **Then R2 and R4 in the side bisect's queue.** They are the only rungs aimed at a large task effect: 0.03–0.10 m/s now vs 0.18–0.22 m/s for one-direction heads.
   - R2 is predicted FAST under every mechanism candidate (RC:473-481). It also doubles each hand's +y share, from 40 % to 80 %.
   - R4 is the optimizer-side form of the rate lever. The C_40 pre-registration puts that lever first if C_40 ends SLOW, and R4's early read costs 0.4 GPU-h.
   - Which of the two leads is the side bisect's call.
3. **Then R3.**
4. **R5 only after an R3 adoption.** The KL measures behaviour only with a bounded head and a small σ (L §6).

**Interactions:**
- **Head and σ are co-varied**, because a bound alone at σ 2.0 leaves the clip binding. If BND wins, bisect it in this order: INC0, then BND without the σ change.
- **Head and action_rate:** held by design, as above.
- **Head and mirror loss:** the loss stays on raw μ in R3 and R4, and moves to clip(μ) in R7. That is exact because the mirror maps are signed permutations (AP-A6).
- **Head and distillation:** a bounded teacher must be distilled on clip(μ) (AP-A6).
- **Canonical training and mirror images:** no M_y images on side rows (RC:486; SK-S8).
- **RSI:** item 8 comes before any RSI arm.
- **Code:** one code_sha move in total.

## 6. When to test another algorithm family

**Stop rule (SK-S9).** If R3 moves geometry but no task block, at N ≥ 3 with guards held, record "the setup is not a task-metric limiter". Keep BND only if Q improves, and stop algorithm work.

Escalating beyond that requires all of the following:
- **(a)** A task block BETTER at the seed-inflated R (T 1.13, F 3.19, F_mt 4.12), pooled over ≥ 3 seeds and surviving R3c.
- **(b)** The predicted mechanism visible in the R0 telemetry: behavioural KL share up from 0.11–0.16, the LR off the floor, and executed change per update at least 2×.
- **(c)** A gain that INC0 does not reproduce.
- **(d)** A failure left after canonical training that optimizer telemetry separates. No task row of D §4.8 has such a separator today.

**A family swap additionally needs evidence that wall-clock or sample count is binding.** Today runs reach 90 % of their final reward by a median of iteration 1,344–1,798, at about 1.1 GPU-h per arm (D:489, §6).

| Trigger | First candidate | Why | Cost |
|---|---|---|---|
| R2 fails, R4 does not rescue it, and the policy-switch or distillation fallbacks (RC §6) are unacceptable; or the user wants "is on-policy PPO the barrier?" answered directly | **TD3-lean**, an in-repo learner on C_lr paired with a matched tanh-PPO arm. It shares a tanh Actor (actor_critic_with_encoder.py:86-87). Replay holds only the 259 non-history dims (about 0.57 GB per 1,024 transitions per env). Batch 4–8k × 2 updates, time-outs dropped, and checkpoints in the PPO schema, round-tripped through runner.load, the exporter and the distiller | This is the only failure class whose mechanism is open and learning-dynamic (RC:48-66). Kill read at 6M samples. My untested inference: candidate (i), the interaction barrier, is an actor-gradient property a DPG learner shares; only (ii) might differ | ~1,000 lines + 3–4.5 GPU-h (SW-A1, A5, A8) |
| The user wants formation without the clock (Z formed 0/2; D:560) | **FPO++** (released code; tested at 2^8–2^12 envs including 256; reports lower seed variance) | The only method with legged evidence at this env count, and it targets basin selection | New head; the exporter and distiller change (5-step integration at deploy); no Gaussian KL; mirror loss undefined; wall-clock ≤ 1.5× pre-registered (SW-A6) |
| BND fails its mechanism check twice | **tanh head** with executed-action entropy (IM-F1 spec), or an **ordinal head** with K = 11 (~200 lines) | Bounded by construction; the KL and entropy act on the executed action | ~3.5–4.3 GPU-h per 3-seed cell |
| R1 makes 1,024–4,096 envs attractive and replay memory then binds | **REPPO** (on-policy with pathwise gradients, no replay; no legged results) | Removes the replay-memory objection | not estimated |

**Not at any trigger:**
- Model-based methods.
- SAPG or PBT, which cost 4–6× wall time per member under time-slicing.
- pip-installing RSL-RL-SAC. It ships as rsl-rl-lib 4.0.1, which would replace the installed 5.0.1, and its time-out observation patch would double-shift this env's history buffer (SW-A4, package claim not re-verified; observations.py:120-124, checked).

## 7. What not to do

- Re-run M1-style σ cuts on saturated heads (D §4.3; L §2).
- Swap libraries or families first, or pip-install anything into isaac_venv.
- Escalate on a result that moves only geometry.
- Score R3 on saturation share, which is circular, or on formation rate: every clock-held t3 run formed (DE-F2 counts 10).
- Price smoothness with an executed-action reward (action_rate, smooth2, TV/L1) under sampled training. These charge for exploration noise and pay the policy to saturate (AP-A1, AP-A8); smooth2 jumps 8.6× for a bounded head.
- Use schedule 'fixed' on a fork before item 3 is fixed, or change num_steps_per_env.
- Feed M_y images into side rows.
- Screen on s2/s3, treat same-seed reruns as paired controls (D18), or use historical ledger heads as controls.
- Edit the shared checkout while side-bisect arms can launch, run R1 beside a live arm, or use MPS. MPS makes signal termination undefined, and the driver stops runs by signal.
- Quote OF-tool action_rate figures before item 6 is fixed, or "RSI inert at σ 1.5" before item 8 is measured.
- Take a bounded or canonical head to hardware before item 10 is done. A canonical head also needs the flip in the runtime or a distilled student.

## 8. Risks and open questions

- **C_40 is pending.**
  - Its early read already excludes "FAST on schedule" (0.295 < 0.58 at iteration 800), so R4 is live either way.
  - An INTERMEDIATE or DEAD result queues C_40_s2, which delays `DONE controls` by about 1.25 GPU-h.
  - 0.325 is also below the slow class's 0.47–0.56 at iteration 1,000 (CH), so the 60 % stand share may be depressing the read.
- **R3 may buy nothing.** A bounded head can still converge near ±1 (L §4). INC0 without action_rate could chatter more. Block Q is itself mildly seed-dominated (0.319 vs 0.252).
- **Power (DE-F2 estimates).** At 5 v 5 the MDEs are 0.075 on side ratio and 0.22 on turn facing_closed. Survival non-inferiority at −0.05 cannot be reached (0.28 is achievable), so T and F are estimated, not decided.
- **Critique numbers are unreplicated CPU estimates:** AP-A1 / SK-S2 (noise-floor Monte Carlos), SW-A2, IM-F5 and DE-F2. Their scripts are in a /tmp scratchpad that a reboot clears.
- **Canonical-training risks (RC §6):**
  - A reversal re-mirrors the whole history.
  - The switch needs hysteresis h ≥ 0.08 m/s near v_y ≈ 0.
  - move_turn crosses v_y = 0.
  - Forward tripod handedness.
  - Clock-free training is untested.
  - C's bundle falls 9–14 % in the hold, vs 0–1 % for W.
  - It cannot deploy without runtime or student support.
- **Hardware.** Non-backdrivable rods at 2.23 flips per joint-second at stand is a sim-to-real risk that no sim metric measures. The 50 Hz policy vs 100 Hz hardware loop question is open (D §1).
- **Still unresolved:** the D7 mismatch KL (item 1 measures it), the 15/33 resume anomaly, and the source of the history-latent nondeterminism (D18).
- **Literature:** the load-bearing bounded-head, step-size and FPO++ numbers come from single 2026 preprints [3, 24, 25, 40, 60, 97] (L §14).

## 9. Documents

- **Dossier:** `crab_hex_omni_task/experiments/2026-09-30_rl_algorithm_review/DOSSIER.md` (artefacts in `artifacts/`).
- **Literature review:** `crab_hex_omni_task/experiments/lit-review-rl-algorithm-and-ppo-hyperparameters.md` (§15 = the original ladder).
- **Prior review and M1:**
  - `lit-review-gait-motion-quality.md`
  - `2026-09-22_gait_motion_quality/{REPORT.md, ANALYSIS_M1_saturation.md, CHANGELOG.md}`
  - M1b plan: `/home/nickmagus/.claude/plans/there-is-a-lot-compressed-unicorn.md`
- **Side bisect:**
  - `2026-09-28_side_bisect/{ROOT_CAUSE.md, CHARTER.md, scripts/queue_controls.sh, scripts/early_read.py}`
  - live log: `parkour/outputs/omni_plan/side_bisect/queue.log`
- **Head of record:** `crab_hex_omni_task/policy/README.md`.
- **Code touched by R0 and R3:**
  - `P/scripts/rsl_rl/modules/{ppo_with_extractor.py, crab_actor_critic_with_encoder.py, actor_critic_with_encoder.py, on_policy_runner_with_extractor.py}`
  - `P/scripts/rsl_rl/{vecenv_wrapper.py, exporter.py}`
  - `crab_hex_omni_task/config/crab_hex/{motion_knobs.py, agents/rsl_rl_ppo_cfg.py}`
  - `experiments/tools/{run_phases.py, run_candidates.py}`
- **Critique scripts (ephemeral):** `/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/{action_rate_econ.py, action_rate_econ.out, bound_probe.py, power*.py, tbcurves.py, advocate/cost.py}`.

## Appendix A. Critique findings and their resolution

**Skeptic (SK)**

| ID | Finding | Resolution |
|---|---|---|
| SK-S1 | No task metric is attributed to the optimizer; saturation is constant across heads | Adopted: §1, §2 items 4–5. R3 becomes a motion-quality test with task guards, and signs are pre-registered |
| SK-S2 | Rung 2 also changes the reward (sampled action_rate charges for noise) | Adopted with AP-A1: action_rate set to 0 and a μ term added in the loss; sampled-σ replay gate; INC0 cell. The μ-based action_rate *reward* is rejected because it gives μ no gradient |
| SK-S3 | Run canonical training first | Adapted: R2 is the first task-metric rung, owned by the side bisect (DE-F1). The early read uses the CH 0.48/0.58 bands, since 0.58 alone is uncalibrated (early_read.py:5-6) |
| SK-S4 | The geometry gates are circular | Adopted: mechanism checks only kill; success needs Q or a task block BETTER |
| SK-S5 | 3k screens miss turning and obstacles; no confirm budgeted | Adopted: scenes named per rung; R3c budgeted or run inside P1_canon; obstacles WATCH |
| SK-S6 | Underpowered; formation at ceiling; off-diagonal cells expected to fail | Adopted: one cell, formation not a gate. Seeds adapted to 3 → 5 (DE-F2). Reusing the incumbent seeds is rejected because they were selected (DE-F5) |
| SK-S7 | σ, LR and saturation do not select the side class; depends on C_40 | Adopted: R4 is conditional on C_40. The RSI-start criterion is dropped (C_lr runs RSI off; AP-A3) |
| SK-S8 | M_y augmentation conflicts with canonical training | Adopted: R7 comes after R2 and uses fa/M_x only on omni; its primary scene is forward |
| SK-S9 | No stop rule, and the trigger is inverted | Adopted: §6 stop rule and (a)–(d); TD3-lean kept only as the user's choice |
| SK-S10 | The step-size rung should be conditional | Adopted: R5 runs only after an R3 ADOPT, on a late fork |
| SK-S11 | The value rung is unmotivated and bundles the reward clip | Adopted: EV moves to R0, the critic Adam runs only on its trigger, and the reward clip goes to a reward campaign |
| SK-S12 | Cut γ 0.995 | Adopted |
| SK-S13 | Rung 4 repeats rung 1 and changing env count invalidates thresholds | Adopted: folded into R1, with SK's kill thresholds |
| SK-S14 | Rung 0 survives; the time-out split should be default-off; #2019 already answered | Adopted: §4 items 1–5 |
| SK-S15 | Earlier ladders have a failed prediction on record | Adopted: the §5 pre-registration rule |

**Switch-advocate (SW)**

| ID | Finding | Resolution |
|---|---|---|
| SW-A1 | Add TD3-lean now as rung 2b | Adapted: the spec is kept in §6 but not scheduled. R2 is predicted FAST under every mechanism for far less code, and the stop rule applies |
| SW-A2 | Replay and learner costs are overstated | Adopted: the 259-dim slice, about 0.57 GB per 1,024 transitions per env. Not re-measured |
| SW-A3 | [40] is misread: SAC matched or beat PPO per sample at 8,192 envs | Adopted as restated in §3; not re-verified |
| SW-A4 | RSL-RL-SAC is not a plug-in, and its patch double-shifts the history | Adopted (§6, §7); the history shift was checked |
| SW-A5 | The checkpoint contract is cheaper than stated | Adopted: emitting the PPO schema plus a round-trip test are hard requirements |
| SW-A6 | FPO++ evidence is understated | Adopted the correction. The trigger is adapted to clock-free formation, the only live formation failure |
| SW-A7 | REPPO is omitted | Adopted as the fallback if replay binds |
| SW-A8 | FastTD3 defaults are unusable at 256 envs | Adopted into the TD3-lean spec |
| SW-A9 | D9 and the auto-reset observation are prerequisites | Adopted: §4 item 5; the replay drops time-outs |
| SW-A10 | Add an ordinal head as a third level | Adapted: it becomes the first alternative head if BND fails its mechanism check |
| SW-A11 | SAPG, PBT, DecQN and model-based methods stay late | Adopted |

**Action-path audit (AP)**

| ID | Finding | Resolution |
|---|---|---|
| AP-A1 | action_rate charges for noise: +0.15 to +0.99/s for a bounded head | Adopted: BND sets action_rate to 0 and adds a clip(μ) term in the loss; item 2 logging; INC0 tests whether the charge drives saturation |
| AP-A2 | The replay clip bug and deterministic replays make the dossier's creeper numbers wrong | Adopted: item 6 and the corrections (checked at :100) |
| AP-A3 | RSI resets the cam motor state | Adopted: item 8 (checked); the "inert at σ 1.5" claim is withdrawn |
| AP-A4 | The deploy contract is broken | Adopted as a pre-hardware task (item 10); corrects D §6 (checked) |
| AP-A5 | Eval action metrics are not head-invariant | Adopted: item 7 |
| AP-A6 | Distillation and the mirror loss use raw means | Adopted: distil on clip(μ) before bounded teachers; mirror loss on clip(μ) in R7 |
| AP-A7 | The #2019 audit is negative; routing facts | Adopted: audit dropped; the num_prop fix is deferred to any future head that changes the contract |
| AP-A8 | The lit §4 remedy carries the same charge; action_rate is unsquared | Adopted: §3 correction; smoothness priced on μ or through the physical costs downstream of the slew filters |

**Implementation (IM)**

| ID | Finding | Resolution |
|---|---|---|
| IM-F1 | tanh placement and the checkpoint marker | Adopted for any later tanh arm; BND needs no contract change |
| IM-F2 | code_sha gating and the worktree import trap | Adopted: §4 landing rules, with the rerun budgeted |
| IM-F3 | The fixed LR resumes at the parent's LR | Adopted: item 3 (checked) |
| IM-F4 | Entropy 0 disables tanh; the σ bundle | Adopted: bound-loss head, entropy fixed at 0.0025, bisection order pre-registered |
| IM-F5 | Bound coefficient 1e-4 from scratch | Adopted instead of DE-F10's ramp (μ starts in-bound; 1e-3 already matches the surrogate's gradient on W) |
| IM-F6 | Hidden confounds | Adopted: saturation is measured as the executed near-bound share (≥ 0.95) plus \|u\|; action_rate held; mirror loss constant |
| IM-F7 | A soft σ needs a new std type | Adopted as specified |
| IM-F8 | Silent policy kwargs | Adopted: item 4 |
| IM-F9 | A blanket raise breaks launches | Adopted: allow-list |
| IM-F10 | Schedule arms move the critic LR | Adopted into R5 |
| IM-F11 | num_envs plumbing | Adopted into R1 and the adoption rule |
| IM-F12 | Rollout length puts the ramps out of step | Adopted |
| IM-F13 | R1 is unsafe beside live runs; MPS | Adopted |
| IM-F14 | Augmentation needs code and about 2× the budget | Adopted into R7 |
| IM-F15 | Canonical-training plumbing and deploy limits | Scope adopted. Timing adapted: it lands through the side bisect because it is wrapper-side and independent of PPO |
| IM-F16 | Telemetry details | Adopted: item 1 |
| IM-F17 | Omni time-out/termination overlap is about 0; fix on the PPO side | Adopted: item 5 |
| IM-F18 | The hardware inference copy is stale | Adopted: item 10 |
| IM-F19 | The knobs reach only omni | Adopted: R3 and R4 run on omni; forward is wired only for R7 |

**Design (DE)**

| ID | Finding | Resolution |
|---|---|---|
| DE-F1 | Collides with the live side bisect | Adopted: start after `DONE controls`; worktree; one landing; R2 and R4 run in the side bisect's queue |
| DE-F2 | Power; gates mix checks and outcomes | Adopted: checks split from outcomes; 3 → 5 seeds; Q and side class decide; T/F estimated; survival as a tripwire |
| DE-F3 | No confirm on the W recipe | Adopted: R3c |
| DE-F4 | 3k window, checkpoint noise, kill reads | Adopted: 3-checkpoint average, DEAD < 0.40, slope rule, R5 on a tail fork |
| DE-F5 | Controls, fresh seeds, drift | Adopted, except the side bisect's class-level rungs keep s3 → s2 |
| DE-F6 | The budget omits evals, controls and confirm | Adopted: §5 costs; R1 design |
| DE-F7 | Scenes per rung | Adopted (§5 table) |
| DE-F8 | Pre-registered gates and eval state | Adopted: §5 rules |
| DE-F9 | Split rung 0; detect inert knobs | Adopted: the knob-used checks and the \|μ\| check at iteration 300 |
| DE-F10 | Arm under-specified | Adapted: values fixed in §5 BND; coefficient per IM-F5; last-layer init unchanged in both cells; action_rate per AP-A1 |

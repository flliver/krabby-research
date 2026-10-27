
## 15. Ranked recommendations

**Algorithm family: keep PPO now; revisit an off-policy arm later under stated conditions; model-based not
at all.** Nothing in the census or the literature indicts PPO as a family; it indicts the construction
(unbounded mean under a binding clip, absorbing σ clamp, per-minibatch KL rule at 384 samples, inherited
batch and γ/λ). The closest off-policy analogue is slower and more seed-variable than PPO and needed the same
fixes first [40]; the replay budget at 1,149 dims is 2.4 GB per 1,024 transitions per env [88]. An off-policy
arm (RSL-RL-SAC's released extension, or FastTD3 with a proprio-only replay) becomes worth its porting cost
(estimator/history stubs, mirror augmentation, terminal labels, checkpoint contract; dossier §6) only if, after
rungs 2–5 below, (a) formation rate or side-speed class is still seed-dominated with a bounded head and
behavioural KL, and (b) a throughput probe shows the learner share (now ~20 % of wall time) can absorb a UTD
≥ 1 without starving the simulator. Head alternatives inside PPO (categorical/ordinal, GPO's growing bound,
FPO++) come before any family change [22–25, 97].

Cost basis: 1.1–1.2 GPU-h per 3,000-iteration from-scratch arm at 256 envs (dossier §6); an early
kill-read at ~1,000 iterations costs ~0.4 GPU-h per arm. Seed counts follow §11: 3 for large-effect
kill/advance reads, 5 where formation rate decides. Knob names are this repo's
(`crab_actor_critic_with_encoder.py`, `ppo_with_extractor.py`, `rsl_rl_ppo_cfg.py`, `vecenv_wrapper.py`,
`parkour_reward_manager.py`, `train.py --num_envs`).

0. **Telemetry and defect fixes before any run** (§6, §7, §11; [54, 3, 46, 66]). Log per-minibatch and
   per-iteration approximate KL, clip fraction, mean within-iteration LR, explained variance, the share of
   |μ| > 1 per dim, executed-vs-raw action change per update and sign flips per joint-second (dossier §3.7).
   Compute He et al.'s P10/P01 split (means near the bound vs samples near the bound) from the existing eval
   npz files. Split the `total_terminates` fall conditions out of the `time_out` tag (D9) and make unknown
   `algorithm` cfg keys raise (D14) so a new knob such as `bound_loss_coef` cannot silently no-op. Cost 0
   GPU-h; gates everything below.
1. **Throughput and co-location probe** (§7, §11; [46, 57, 58, 128, 129]). One process at `--num_envs`
   512/1024/2048/4096 for ~200 iterations each, recording collection time, VRAM and physics stability of the
   230 kg plant with height scan; one 2–3-process headless smoke with and without MPS (Linux only; expect
   coupled faults and undefined behaviour on signal kills). Dossier ties: +5 % wall time for 4× envs up to 256,
   61–69 % utilisation, 4.4 GB per process, 2,783 steps/s each when two runs overlap. Cost ≈ 0.5 GPU-h. Decides
   the geometry and the seed-parallel route for every later rung; if 1,024–4,096 envs are near-free, every
   3,000-iteration arm below gains 4–16× samples at ~1.2–1.5 GPU-h.
2. **Bounded mean × gradient-passing σ, from scratch, 2 × 2 factorial, 3 seeds** (§2, §3, §5; [12, 13, 3,
   14, 40, 43]). Head: {current unbounded Gaussian, rl_games bound loss on raw μ with soft bound 1.1 and
   `bound_loss_coef` 1e-4 ramped to 1e-3, or tanh on the mean with entropy on the executed action}. σ:
   {current `init_noise_std` 1.5 + hard clamp 2.0 + `entropy_coef` 0.01, vs `init_noise_std` 0.5–1.0, log-std
   through tanh into [ln 0.05, ln 2] or softplus + offset (no hard cap), `entropy_coef` 0 (0.0025–0.005 as a
   fallback), last-layer init 0.01×}. Dossier ties: 97 % |μ| > 1, σ pinned in 59/59 long runs, M1's rejection
   as a fine-tune, half the samples clipped at μ = 0 under init 1.5 (D16). Primary gates: saturation share and
   P10, sign flips per joint-second, formation rate over seeds, tracking and side-speed class; smoothness scored
   separately (§4). These two knobs must be co-varied: lowering σ alone cannot produce an interior mean [3],
   and a bound alone at σ 2.0 leaves the clip binding on samples and the KL non-behavioural (§6). 12 arms
   ≈ 13–14 GPU-h at 256 envs; kill reads at 1,000 iterations can halve it. Interaction flags: compute the
   mirror loss on the bounded mean (clip(μ)/tanh(μ)) in the bounded cells [67]; distil from bounded teachers on
   the executed action or pre-squash parameter (§13); the deploy clip stays.
3. **Step-size rule on the winning head** (§6; [46, 48, 41, 54, 51]). {fixed LR 1e-4 and 3e-4, linear anneal,
   with per-epoch target-KL early stop at 0.01–0.02} vs {per-epoch stepping on step-KL, target recalibrated by
   the logged KL} vs the incumbent per-minibatch rule. Dossier ties: LR erased within iteration 0 in 94 % of
   fresh runs, 62.9 % floor residency, 11–16 % behavioural KL on W. Expect a modest, not transformational,
   gain [41]. 2 new arms × 3 seeds ≈ 7 GPU-h. Run after rung 2 because the KL only becomes behavioural with a
   bounded head and small σ.
4. **Batch geometry** (§7; [46, 56, 60, 54]). `--num_envs` 1,024–4,096 × `num_steps_per_env` 24,
   `num_mini_batches` 4 (minibatch 6k–25k), `desired_kl` 0.01, LR bounds re-checked; or 256 × 96–192 steps if
   VRAM forbids; never fewer than 24 steps. Dossier ties: batch 8× below Rudin's smallest, minibatch 384, GPU
   at 66 W. 2 arms × 3 seeds ≈ 7–9 GPU-h depending on rung 1. Interaction: this changes the KL estimate's
   noise, so rung 3's verdict may flip; re-read the controller telemetry rather than the verdict.
5. **Value-side hygiene** (§12; [14, 51, 133, 135, 5, 81]). Separate critic Adam at fixed 3e-4–1e-3 outside
   the KL rule; `use_clipped_value_loss=False`; running value normalisation as its own 3-seed A/B; frozen-policy
   critic warm-up (~200 iterations) at every stage hand-off; reward clip off with a small explicit post-clip
   termination cost. Dossier ties: one Adam group, bridge critic 0.5 vs 0.0015/step, 13.6 % of return forgiven,
   teacher value loss ≤ 0.0003. Each item is deploy-invisible. ≈ 7 GPU-h for two 3-seed A/Bs; the hand-off
   warm-up is a fine-tune-scale test (≈ 0.5 GPU-h per boundary).
6. **Symmetry: augmentation vs loss on bounded means, 5 seeds** (§9; [67, 84, 47, 71, 85]).
   `use_data_augmentation=True` with `mirror_loss_coeff` 0 or small, from scratch on the rung-2 head; mirror
   function covering the full 1,149-dim obs including the 10-step history and estimator inputs. Dossier ties:
   0.573 vs 0.032 at 3k seed 1; E1 0.541/0.0/0.026/0.0. ≈ 6 GPU-h. Interaction: only on a bounded or
   small-mean head [67].
7. **Canonical training for the omni two-hand failure, 3 seeds then confirm** (§8; [68, 69, 70, 74, 81]). M_y
   inside the rollout and the update, gated on sign(v_y cmd), history mirrored per frame, deterministic
   tie-break; log +y/−y gradient cosine over iterations 0–600 [77]. Dossier ties: 0.028–0.103 vs 0.184–0.217
   m/s; canonicaliser at deploy fast both hands; 99 % sign-agreement distillation. ≈ 4 GPU-h plus a fresh-seed
   confirm; fallback: per-direction specialists distilled then RL-fine-tuned [81].
8. **γ 0.995 / λ 0.95–0.97, 3 seeds** (§7; [14, 62, 63, 65]). Only after rung 0's time-out fix. Dossier tie:
   0.34 s GAE horizon inside a 0.48 s rollout on a 1.2–2.8 s gait. ≈ 3.6 GPU-h.
9. **Later head alternatives** (§3, §10; [22–25, 97]): ordinal/unimodal categorical head at K = 11 with
   entropy 0 (≈ +20 % actor cost), GPO's growing tanh bound, FPO++; each only if rung 2 moves geometry but not
   task metrics. Off-policy arm under the conditions above.

Total for rungs 0–8 ≈ 50–60 GPU-h, about a tenth of the ~528 GPU-h spent under the un-varied configuration.

**Where the literature disagrees with the dossier's reading.**
- *What a bounded head buys.* The dossier calls the parameterisation "the wrong shape for the actuator
  interface" and expects the head fix to lift task metrics. The literature says the executed controller being
  near-bang-bang may be close to the optimum of this objective [26, 27]; bounded heads change action geometry
  reliably and return inconsistently ("neither task supports a universal recommendation"; bounded heads
  "did not pay dividends"; Gaussian+tanh tied Gaussian on Humanoid-v1) [3, 4, 22], with the legged exceptions
  (DHAL, GPO) statistically weak [19, 25]. What the head restores is the learning signal (σ gradient,
  behavioural KL, entropy on the executed action). Smoothness must be bought explicitly with an executed-action
  cost, and the present squared `action_rate` on clipped actions is the wrong instrument [36, 34].
- *The action interface.* The dossier treats velocity cams as fixed; the literature gives position targets a
  modest learning-speed prior in planar imitation [28] but favours velocity-type spaces for transfer and
  continuous rotation [29, 31, 32], and Krabby's position-with-slew rods saturate exactly like the cams [38], so
  the interface is not the lever for the same reason the dossier reaches by a different route.
- *Lower σ.* M1's rejection and He et al. agree, but Hsu's failure mode 3 warns that a narrow init can worsen
  basin selection [2]; the dossier does not carry that risk, so rung 2 scores formation rate, not only
  saturation.
- *Minibatch count.* The dossier finds executed change per update moves only 0.8–1.2× between 16 and 4
  minibatches; Beukman et al. also say more minibatches at fixed minibatch size (more data), not fewer, while
  Rudin prefer very large minibatches [60, 46]. Both agree with the dossier that the lever is more envs, not a
  re-split.
- *The step-size rule.* The dossier reads floor residency as a first-order throttle; rl_games' only controlled
  data show fixed vs adaptive differences inside a 95 % interval spanning zero [41], so the expected gain is
  modest and the rule matters mainly through what the KL measures (§6).
- *Env count.* The dossier's "nearly free" ends at 256 envs; the literature warns that shortening rollouts
  below ~25 steps hurts gait discovery [46, 56], so more envs must not come with shorter rollouts.
- *Mirror loss.* The dossier treats coefficient 0.5 as baked; Krabby's lineage favours augmentation [67, 84],
  yet Abdolhosseini rank LOSS the most consistent at enforcing symmetry with no learning-speed effect for any
  method [71]; the literature itself is split, which is why rung 6 needs 5 seeds.
- *Off-policy.* Agreement, with an added reason: the closest analogue is worse on both of Krabby's limiting
  axes (wall-clock, seed variance) [40].

**Interactions that fix the ladder's order.** Head ↔ σ (rung 2, co-varied). Head ↔ KL controller (a bounded
head with small σ makes `desired_kl` behavioural; rung 3 after rung 2). Batch ↔ KL controller (rung 4 changes
the estimate's noise; rung 3 telemetry must be re-read). Head ↔ mirror loss (loss on bounded means;
augmentation only on small-mean heads; rungs 2 and 6). Head ↔ distillation target (executed action or
pre-squash parameter; §13). Obs normalisation ↔ checkpoint contract (bundle with a from-scratch head arm; rung
5). Value normalisation ↔ curriculum stages (test per stage, do not bake; rung 5). γ ↔ time-out tag (rung 0
before rung 8). Canonicaliser ↔ history/estimator inputs and mid-episode command flips (rung 7).

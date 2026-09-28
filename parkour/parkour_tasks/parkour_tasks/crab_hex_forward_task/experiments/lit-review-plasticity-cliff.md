# Literature review: the plasticity cliff at curriculum stage transitions

Companion to `lit-review-hexapod-reward-stability.md`. Written 2026-08-17 against the
task1-velocity campaign's carry-up evidence (v1/v2/v3 lineages, 2b2 stage).

## 1. The problem, in this repo's own data

Three carry-ups of the same flat-walk policy at different consolidation depths into the
identical bridge→2b1→2b2 chain:

| lineage | flat consolidation | bridge/2b1 failure | 2b2 clearance learning | verdict |
|---|---|---|---|---|
| v1 | 0k extra (3k total) | 0.79 / 0.93 | **alive** (0.08–0.15) | plastic but weak |
| v3 | +5k (8k total) | 0.63 / 0.67 | **partial** (0.02→0.064, batch 2 ×3) | intermediate |
| v2 | +17k (20k total) | **0.44 / 0.59** | **frozen** (0.013, 1000 iters) | strong but rigid |

Walking competence transfers monotonically with consolidation; the ability to *modify
the gait* (learn obstacle lifting, adapt swing) decays monotonically with it. The same
rigidity shows on flat ground as the 20k policy's high-speed fall mode. The prior
review's §9 fine-tune null ("no published successes adding gait-shaping to a converged
policy; successes are from-scratch or early-curriculum") predicted exactly this.

**Checkpoint diagnostics (measured 2026-08-17 on the 3k/8k/13k/20k series):**

| ckpt | action std | actor ‖W‖ | critic ‖W‖ | dormant rows |
|---|---|---|---|---|
| 3k | 1.899 | 26.3 | 30.7 | 0% |
| 8k | 1.967 | 29.9 | 36.1 | 0% |
| 13k | 2.000 | 32.4 | 41.1 | 0% |
| 20k | 2.000 | 34.6 | **46.7** | 0% |

Two standard mechanisms are RULED OUT for this cliff: exploration-noise collapse (std
grew) and dormant neurons (none). What remains — and matches the observed plasticity
ordering exactly — is **weight-norm growth**, the warm-start-problem correlate: larger
norms → sharper functions → relatively smaller gradient updates → slower adaptation.
The critic grows fastest (+52%), and the critic is precisely what must relearn from
scratch at a stage boundary (the reward stack changes), so advantage estimates are
mispriced during the exact window the actor needs to learn lifting.

## 2. The phenomenon in the literature

- **Loss of plasticity in continual RL**: [Abbas et al. 2023](https://arxiv.org/abs/2303.07507);
  [Dohare et al., Nature 2024](https://www.nature.com/articles/s41586-024-07711-7)
  (continual backprop); survey: [Klein et al. 2024](https://arxiv.org/pdf/2411.04832).
  Networks trained online progressively lose the ability to fit new objectives.
- **On-policy specifics** (our setting — PPO): [Juliani & Ash, NeurIPS 2024](https://arxiv.org/abs/2405.19153)
  — plasticity loss appears on-policy under domain/task shift; regularization-class
  fixes (L2, regenerative, **soft shrink+perturb** best-in-class) transfer; notes
  plasticity loss can masquerade as other pathologies. Stage transitions in a
  curriculum ARE task shift.
- **Primacy bias / resets**: [Nikishin et al. 2022]; [Fisher-guided selective forgetting 2025](https://arxiv.org/html/2502.00802) —
  agents overfit early experience; periodic or targeted resets of later layers/critic
  recover learning ability at the cost of transient performance drops.
- **Dormant neurons**: [Sokar et al., ICML 2023](https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf)
  (ReDo) — measured: not our mechanism.
- **Plasticity injection**: [Nikishin et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf)
  — add a fresh trainable branch, freeze the old; performance-preserving.
- **Warm-start problem / weight scale**: Ash & Adams 2020 (shrink+perturb);
  [regenerative regularization / L2-init](https://arxiv.org/pdf/2308.11958);
  [soft weight rescaling 2025](https://arxiv.org/pdf/2507.04683) — rescaling weight
  magnitudes restores trainability without destroying the function; directly targets
  the mechanism our diagnostics implicate.
- **Structural growth**: [Neuroplastic Expansion, ICLR 2025](https://arxiv.org/pdf/2410.07994);
  [Forget-and-Grow 2025](https://arxiv.org/html/2507.02712) — grow capacity instead of
  resetting.

## 3. Ranked recommendations for the 2b2 transition

Ordered by (mechanism match from our diagnostics) × (integration cost) × (SOP fit).
Every training-affecting item goes through the standing gate→screen discipline.

1. **Critic reset at stage boundaries** (primacy-bias class; near-zero cost). The 2b2
   reward stack is new; the inherited high-norm critic is both wrong and slow. Reset
   critic (and optionally the estimator head) at bridge→2b2 entry, keep the actor.
   Directly attacks the measured +52% critic-norm growth. One flag in the resume path;
   screen = one 2b2 batch pair from the same base, reset vs not.
2. **Soft shrink+perturb on the actor at stage entry** (best-in-class in the on-policy
   study): W ← λW + ε, λ≈0.8–0.9, tiny ε. Recovers gradient scale without erasing the
   gait (the tripod lives in the function, and S&P at these λ preserves function shape
   approximately). Screen after/with #1 — one change at a time.
3. **L2-init / regenerative regularization during long consolidation runs**
   (preemptive): add ‖W−W_init‖² pressure to flat-stage training so future references
   arrive at stage boundaries with bounded norms. Changes the flat recipe → needs a
   flat re-run to matter; queue for the NEXT reference training, not a 2b2 fix.
4. **Curriculum restructure — obstacles before consolidation** (§9's prescription;
   heaviest but most causal): blend light terrain into flat training early so lifting
   is learned while the gait is still forming. v3's partial success (5k) already
   points here; the limit case is a "flat+light-obstacles from scratch" stage replacing
   the flat→bridge boundary entirely.
5. **Plasticity injection** (reserve): fresh actor branch at 2b2 if resets/S&P prove
   insufficient — heavier surgery, checkpoint-shape change.

NOT recommended for this cliff (mechanism mismatch, measured): entropy/std re-inflation
(std never collapsed), ReDo/dormant recycling (no dormancy), full network resets
(destroys the walking that transfers — the one thing v2 got right).

## 4. Falsifiable predictions

- If #1 is right, a critic-reset 2b2 batch from the v2 (20k) base should unfreeze
  clearance learning within one batch despite the rigid actor — because the actor's
  gradient was starved by mispriced advantages, not by its own norm alone.
- If #2 matters beyond #1, the same batch with actor-S&P should additionally recover
  failure-rate slope toward v1-like adaptation speed.
- If neither moves clearance on the v2 base, the rigidity is representational
  (actor-side), and #4 (restructure) is the remaining path — matching §9's null.

## 5. Prediction outcomes (updated 2026-08-17)

Prediction #1 (critic reset unfreezes clearance on the v2 base): **FALSIFIED.**
One 500-iter 2b2 batch from model_20196 with fresh critic (norm 46.5->17.3, Adam
moments zeroed): clearance 0.010 windowed (control: 0.013 — no change), while failure
transiently collapsed to 0.92 and recovered only to 0.82 (control 0.52), terrain
demoted to 0.25. The rigidity is actor-representational, not advantage-starvation;
the reset's transient also proved expensive at this batch scale. Remaining paths per
§3: actor S&P (#2, one cheap probe left) and curriculum restructure (#4, causal fix).
The v3 intermediate-maturity lineage (clearance 0.02->0.064->0.080 and climbing) is the
live practical instance of #4's principle.

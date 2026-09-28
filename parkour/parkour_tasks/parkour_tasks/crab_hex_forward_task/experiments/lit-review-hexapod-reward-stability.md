# Literature review: reward-function design for hexapod stability — mapped to the Krabby campaigns

*2026-08-13. Prepared after the tripod_stability (v1–v5/b6/b7), lean_reduction, morphology, and
seed_basin_search campaigns closed, with the mirror-symmetry campaign active. All citations were
verified against the papers' abstracts/full text at time of writing.*

## 1. The problems this review targets

Restated from the campaign ledgers so this document stands alone:

1. **Basin lottery.** The alternating-tripod basin was found in 1 of 8 from-scratch draws (seed 1,
   plain config); 0/7 subsequent seeds. Basin selection happens in the first ~1000 iterations
   under the tracking/progress terms, before any alternation-conditioned reward can pay
   (`2026-08-12_1550_seed_basin_search/RESULTS.md`).
2. **Shaping a converged policy is inert.** 27+ fine-tune attempts across two reward designs and
   4× weight ranges left tripod in the 0.38–0.43 band; the 0.146/0.556 duty asymmetry is immune
   to price pressure ("it isn't economic" — v7 offline study).
3. **The +12° forward lean** survived every pitch/orientation/speed lever at up to 5× dose, yet
   the plant stands level with centered CoM and statically balanced tripod sets (51.0% ± 2.1%).
   The lean is learned symmetry-breaking locked into the baseline's basin.
4. **State-paying rewards get farmed.** v1 lunge → v2 tip-rock → v3 skate → v3b drag: each gate
   eliminated its target and the policy relocated to the cheapest remaining holdable state. The
   one event-paying design (v4 swap credit) was economically inert on real gaits. Standing
   lesson: offline replay gate before any training screen.

The short answer from the literature: **yes — all four problems are recognized, and three have
standard, well-validated answers.** The strongest single match is the clock/phase-based periodic
gait reward family (§2), which was invented precisely because reference-free reward shaping fails
to produce specific gaits reliably across seeds. The active mirror-symmetry route is
well-supported (§3), with concrete guidance on method choice and failure modes. The
state-vs-event lesson the campaign paid four versions to learn is a theorem in the shaping
literature (§8).

---

## 2. Clock/phase-based periodic gait rewards — the standard fix for the chicken-and-egg

**Key papers**

- Siekmann, Godse, Fern & Hurst, *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic
  Reward Composition*, ICRA 2021. [arXiv:2011.01387](https://arxiv.org/abs/2011.01387)
- Margolis & Agrawal, *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity
  of Behavior*, CoRL 2022. [arXiv:2212.03238](https://arxiv.org/abs/2212.03238)
- Hexapod precedent: *Deep Reinforcement Learning Control of a Hexapod Robot*, Actuators 15(1):33,
  2026. [doi:10.3390/act15010033](https://doi.org/10.3390/act15010033) — reported (per abstract;
  full text paywalled at review time) to use a phase-tracking reward comparing foot forces and
  velocities against an ideal swing-support schedule to induce tripod gait.

**What they do.** Siekmann et al. define a gait as a set of per-foot periodic phase windows driven
by an external clock. During a foot's commanded swing window, ground-reaction force is penalized;
during its commanded stance window, foot velocity is penalized (probabilistic phase boundaries
smooth the transitions). Gait type is just the vector of phase offsets between feet. Their stated
motivation is exactly problem 1: reward functions that are "specific enough to reliably learn the
gait across different initial random seeds or hyperparameters," versus reference-free rewards
that produce massive cross-seed variance in which behavior emerges. Walk These Ways ports this to
quadrupeds: an 8-dim command vector (three inter-foot timing offsets, stepping frequency,
footswing height, body height, **body pitch**, stance width) with the swing/stance
force/velocity rewards enforcing the commanded contact schedule, and the Raibert heuristic
supplying consistent foot targets.

**Why this maps onto Krabby's diagnosis.** The v5 post-mortem identified the structural flaw: "a
term whose income requires alternation to already exist cannot influence the lottery." A
clock-referenced reward inverts that dependency — the target contact schedule exists in the
reward *from step 0*, before any behavior does. Every v1–v5 exploit family is priced out by
construction:

| degenerate family | why the clock reward rejects it |
|---|---|
| unison glide (seeds 1,4,6,7) | half the feet are always in their commanded swing window while planted → continuous force penalty |
| tip-rock / lunge (v1, v2) | contact pattern uncorrelated with the clock → both windows leak penalty |
| skate/shuffle (v3, seeds 3,5) | feet planted-and-sliding during commanded stance → velocity penalty (the term v3b's clippable `feet_slide` failed to be) |
| near-stationary drag (v3b) | standing violates the swing window 50% of the time — there is **no holdable state** with income, because the clock keeps moving |
| handed shuffle (seeds 2,8) | a 0.15/0.55 duty split violates the commanded 50/50 schedule directly — duty is *commanded*, not priced |

The last row deserves emphasis: the duty asymmetry that capped tripod at ~0.40 and proved immune
to the v5 min-peak channel, q_duty, and q_bal (v7 study) is not priced under this scheme — it is
specified. This is the literature's answer to "the asymmetry isn't economic": stop trying to make
it economic; make the schedule the reward's reference.

**Fit with existing infrastructure.** The term needs a per-episode phase clock (2 obs dims,
sin/cos) and contact forces + foot velocities — all already available (the v5 term and gait-eval
consume the same signals). It passes the offline replay gate naturally: replay healthy npz traces
against the clock at the best-aligned phase (the healthy gait should score well at some phase
shift; all four degenerate traces should score poorly at *every* phase shift — a stronger
discrimination test than v5's 104:1). Caveats from the literature: the clock frequency should
match the plant's natural stride rate (measured: ~0.30 s period, so ~3.3 Hz — Walk These Ways
runs exactly this range), and the campaign's own b-series lesson applies — this is a
from-scratch/early-training intervention, not a fine-tune patch.

---

## 3. Symmetry enforcement — direct support for the active campaign, plus what to watch

**Key papers**

- Abdolhosseini, Ling, Xie, Peng & van de Panne, *On Learning Symmetric Locomotion*, ACM
  SIGGRAPH MIG 2019. [PDF (UBC)](https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/2019-MIG-symmetry.pdf),
  [ACM DL](https://dl.acm.org/doi/10.1145/3359566.3360070)
- Su, Huang et al., *Leveraging Symmetry in RL-based Legged Locomotion Control*, IROS 2024.
  [arXiv:2403.17320](https://arxiv.org/abs/2403.17320)
- Ordoñez-Apraez et al., *Morphological Symmetries in Robotics*, IJRR 2025.
  [arXiv:2402.15552](https://arxiv.org/abs/2402.15552)
- *MS-PPO: Morphological-Symmetry-Equivariant Policy for Legged Robot Locomotion*, 2025.
  [arXiv:2512.00727](https://arxiv.org/pdf/2512.00727)
- *Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy Enables Free Gait
  Transitions at Varying Speeds*, 2024. [arXiv:2403.10723](https://arxiv.org/html/2403.10723v4)

**What they found.** Abdolhosseini et al. compared four enforcement routes: DUP (duplicate
transitions with mirrored copies), LOSS (auxiliary mirror-symmetry loss on the policy mean),
PHASE (alternate mirrored halves of a phase-indexed motion), NET (hard-equivariant architecture).
Findings relevant here: **DUP alone was the least effective at actually enforcing symmetry; LOSS
was the most consistent**; PHASE only applies to time-indexed imitation setups. Su et al. (2024)
extended the comparison on real legged hardware: symmetry-incorporated methods beat unconstrained
baselines on gait quality and robustness, and **strictly equivariant architectures consistently
outperformed data augmentation** in sample efficiency and task performance, deploying zero-shot.
MS-PPO and the morphological-symmetries line (Ordoñez-Apraez) formalize this: encode the robot's
symmetry group in the network and neither reward shaping nor augmentation is needed. The
symmetry-guided-hierarchy paper is the reward-side variant: temporal + morphological +
time-reversal symmetry terms in the reward produce multiple dynamic gaits without predefined
trajectories.

**Implications for the mirror-symmetry campaign (active).**

1. The chosen combo — data augmentation *plus* mirror loss via rsl-rl's `symmetry_cfg` — is
   exactly the right starting pair given the literature ranking (LOSS most consistent; DUP
   cheapest but weakest; running both covers each other).
2. **Watch-item the literature flags**: a symmetric *policy* does not guarantee a symmetric
   *gait*. Spontaneous symmetry breaking can persist — the policy maps mirrored states to
   mirrored actions, yet any single rollout still elects one handedness (both handed gaits exist
   as mirror-image trajectories of the same symmetric policy). Symmetry enforcement makes
   *encoded, permanent* lead-set preference impossible (the campaign's stated goal) and makes the
   two basins statistically equal, but it does not by itself force 50/50 duty *within* a rollout.
   If the validation run comes back symmetric-in-weights but still 0.15/0.55-handed-per-episode,
   that is the known failure mode — and the fix indicated by both this section and §2 is to pair
   symmetry with a phase/clock reference that defines alternation in time, not just in weights.
3. **Escalation path if LOSS+DUP underdelivers**: hard-equivariant architecture (NET). Su et al.
   found it strictly better; MS-PPO provides a current recipe. Cost is a policy-network rework,
   which is why it is the escalation rather than the first move.

---

## 4. Motion priors / imitation — the closest published solution to this exact problem

**Key papers**

- Peng, Ma, Abbeel, Levine & Kanazawa, *AMP: Adversarial Motion Priors for Stylized
  Physics-Based Character Control*, SIGGRAPH 2021. [arXiv:2104.02180](https://arxiv.org/abs/2104.02180)
- *Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion Priors*,
  2025. [arXiv:2511.03167](https://arxiv.org/abs/2511.03167)
- *Experience-Learning Inspired Two-Step Reward Method for Efficient Legged Locomotion
  Learning Towards Natural and Robust Gaits*, 2024. [arXiv:2401.12389](https://arxiv.org/abs/2401.12389)

**What they do.** AMP replaces hand-designed style rewards with a discriminator trained to
distinguish policy state-transitions from reference-data transitions; the policy earns style
reward for being indistinguishable from the reference, plus a plain task reward. The hexapod
motion-priors paper (arXiv:2511.03167) is the single most relevant publication found for Krabby:
a real hexapod, tripod gait, and the same failure mode this repo hit — they explicitly could not
get natural gaits from reward composition alone on complex terrain. Their solution: generate a
tripod reference dataset by **trajectory optimization** (8.6 s covering forward/backward/lateral/
steering — no mocap needed), train an AMP discriminator on it, and use reward
`r = r_task + r_style + r_penalties` with `r_style = max[0, 1 − 0.25(d_score − 1)²]`. Ablations
showed the style reward was essential; result claimed as the first RL controller achieving
complex-terrain walking on a real hexapod. The two-step paper is the bootstrap variant: learn on
flat with gait rewards first, then imitate *your own healthy experience* on harder tasks.

**Why this maps onto Krabby's diagnosis.** This is the second literature answer to the
chicken-and-egg: dense style income from step 0, defined by data rather than by a clock. The
repo is unusually well-positioned for it — the offline gate already produced an IDEAL synthetic
tripod trace, and the healthy baseline's eval npz traces are exactly the "own experience" the
two-step method imitates. A discriminator sees full state transitions, so every degenerate
family (unison glide included) is out-of-distribution and earns ≈0 style reward — the
exploit-resistance the v1–v3b gates tried to hand-build, obtained for free from data. Cost: AMP
infrastructure in the rsl-rl/Isaac Lab stack (implementations exist in the Isaac ecosystem), a
discriminator to train and balance, and the known AMP failure mode of style–task reward
imbalance (mode collapse onto the reference at the expense of command tracking, or vice versa).

---

## 5. Energy minimization as a basin selector

**Key papers**

- Fu, Kumar, Malik & Pathak, *Minimizing Energy Consumption Leads to the Emergence of Gaits in
  Legged Robots*, CoRL 2021. [arXiv:2111.01674](https://arxiv.org/pdf/2111.01674)
- *Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped
  Locomotion*, 2024. [arXiv:2403.20001](https://arxiv.org/html/2403.20001v1)
- Rudin, Hoeller, Reist & Hutter, *Learning to Walk in Minutes Using Massively Parallel Deep
  Reinforcement Learning*, CoRL 2021. [arXiv:2109.11978](https://arxiv.org/abs/2109.11978) —
  the reward-recipe family the crab config descends from.

**What they found.** Fu et al. showed that adding a mechanical-energy minimization term (|τ·q̇|
work, not merely a torque-squared penalty) to plain velocity tracking causes structured,
animal-like gaits to *emerge* at the appropriate speeds, without any gait-specific reward. The
adaptive-energy-regularization paper confirms the energy weight acts as a gait selector across
speed. This matters for problem 1 because it is a **basin-selection pressure active from step 0**
that specifically disfavors the degenerate families: the tippy-shuffle families burn energy on
rapid touchdown cycling, the drag families burn it on friction work, and unison motion of all six
legs moves the whole body mass ballistically each cycle. In insects, alternating tripod is the
efficient fast gait — an energy term points the lottery toward it without naming it.

**Caveats for Krabby.** This is the weakest-evidence thread for this platform: the emergence
results are on quadrupeds with direct-drive-style actuation, and the cam-driven (Whitworth) leg
mechanism changes the energy landscape in unknown ways — the glide families might turn out to be
energy-*cheap* on this plant. The existing `reward_torques`/`reward_dof_acc` penalties are not
mechanical-work terms and evidently do not select against the glide (4/8 seeds found it). An
offline replay of a mechanical-power term over the existing healthy + degenerate traces would
answer in an afternoon whether energy actually discriminates in the right direction on this
robot — the replay gate exists for exactly this question, and it should be run before any
training time is spent on this thread.

---

## 6. CPG / structural gait priors — alternation by construction

**Key papers**

- Bellegarda & Ijspeert, *CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion*,
  RA-L 2022. [arXiv:2211.00458](https://arxiv.org/pdf/2211.00458)
- *Adaptive Gait Generation for Hexapod Robots Based on Reinforcement Learning and Hierarchical
  Framework*, Actuators 12(2):75, 2023. [doi:10.3390/act12020075](https://www.mdpi.com/2076-0825/12/2/75)

**What they do.** CPG-RL puts coupled oscillators in the action pathway: the policy modulates
oscillator amplitudes and frequencies rather than joint targets directly, so rhythmic
alternation is architectural, and the RL problem reduces to modulating a structure that cannot
not oscillate. Sim-to-real on Unitree A1 with minimal domain randomization. The hexapod
hierarchical-framework line applies the same idea one level up (RL selects/blends parameterized
gaits).

**Relevance.** This is the strongest possible guarantee against problems 1 and 4 — degenerate
non-alternating basins are removed from the hypothesis space, not priced out — at the largest
integration cost: it replaces the action space, invalidating direct comparability with every
existing checkpoint and much of the eval tooling's assumptions. One Krabby-specific note makes
it cheaper than it first appears: the legs are already driven through a rotating camshaft joint,
i.e., each leg already has a natural phase variable with hardware meaning. An oscillator-phase
action space maps onto this plant more naturally than onto a knee-actuated quadruped. Position
in the queue: escalation path if both the symmetry route and a clock-reward route fail, not a
first move.

---

## 7. Hexapod-specific stability rewards — support polygon and ZMP margins

Classical hexapod work scores stability as the margin from the ZMP (or CoM projection) to the
support-polygon boundary, and some RL papers import it as a reward term (e.g., the hierarchical
hexapod stability line, [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S2667379725000221);
overview in [A Systematic Review of DRL for Legged Robot Locomotion](https://www.researchgate.net/publication/400276550_A_Systematic_Review_of_Deep_Reinforcement_Learning_for_Legged_Robot_Locomotion)).

**Campaign-informed caution.** A support-margin term is a *state* reward, and the v1–v3b history
is a four-version demonstration of what converged policies do with holdable state income: the
robot that plants all six feet around a centered CoM has a maximal stability margin and zero
locomotion. The literature usually gets away with it because the margin term is small against
strong velocity tracking — which is precisely the regime where the b-series showed such terms
are inert on this robot. Verdict from the campaign's own evidence: skip state-based margin
rewards; if a stability term is ever wanted, the §8 potential-based form (pay for margin
*improvement*, not margin *possession*) is the shaping-theoretic way to make it farm-proof.

---

## 8. Theory: the state-vs-event lesson is a theorem

- Ng, Harada & Russell, *Policy Invariance Under Reward Transformations: Theory and Application
  to Reward Shaping*, ICML 1999.
  [Semantic Scholar](https://www.semanticscholar.org/paper/94066dc12fe31e96af7557838159bde598cb4f10)
- Skalse, Howe, Krasheninnikov & Krueger, *Defining and Characterizing Reward Gaming*,
  NeurIPS 2022. [arXiv:2209.13085](https://arxiv.org/pdf/2209.13085)

Two results ground the campaign's hard-won lessons:

1. **Potential-based shaping (Ng et al.)**: a shaping term of the form
   `F(s, s') = γΦ(s') − Φ(s)` is the *only* form guaranteed not to change the optimal policy.
   Its telescoping structure means holding any state yields zero net shaping income, and any
   cycle nets zero — i.e., **potential-based terms are structurally un-farmable by
   state-holding**, which is exactly the property the v1→v3b gates tried and failed to
   hand-engineer. The v4/v5 event-based designs were unknowing approximations of this. Design
   rule for any future term: if it can be expressed as Φ-differences of a gait-quality
   potential, farming is impossible by construction; if it cannot, the offline replay gate is
   the only defense.
2. **Reward gaming (Skalse et al.)**: formally, a proxy reward and a true objective are
   "unhackable" only in degenerate cases (essentially, constant reward). Increasing optimization
   pressure against any imperfect proxy eventually produces policies that score well on the
   proxy and poorly on the objective. This is the theoretical statement of the four-version
   pattern ("each gate eliminated its target behavior and the policy relocated") — and the
   justification for the campaign's conclusion that patching gates was structurally doomed
   rather than under-iterated.

---

## 9. Literature gaps

- **The +12° lean as such**: no paper was found treating a persistent, reward-immune postural
  bias in a converged policy as a fixable target. The literature's implicit position matches the
  morphology campaign's conclusion — converged basins are not renegotiated by reward terms; the
  posture is re-drawn at from-scratch time under better structure. The nearest published lever
  is Walk These Ways' *commanded* body pitch (posture as command input, enforced from step 0).
- **Cam/linkage-driven legged RL**: essentially nothing published on RL reward design for
  Whitworth/Klann/Jansen-type single-DOF leg mechanisms; the phase-variable observation in §6 is
  unexploited territory.
- **Fine-tune-window reward shaping**: no positive results found for adding gait-shaping terms
  to an already-converged locomotion policy over short windows — consistent with the 27-attempt
  null. The field's successful gait-specification results are all from-scratch or
  curriculum-from-early-training.

---

## 10. Ranked recommendations

Ordered by (literature evidence) × (fit to the diagnosis) × (integration cost). Each is
compatible with the standing protocol: offline replay gate → 3000-iter screen → two-point eval.

1. **Clock-based tripod contact-schedule reward** (§2) — *do this next, alongside or immediately
   after the mirror-symmetry validation run*. New term: per-foot swing-window force penalty +
   stance-window velocity penalty against a fixed-frequency (~3.3 Hz) two-phase clock with
   tripod offsets; 2 sin/cos clock dims added to obs. Directly attacks problems 1 (schedule
   exists from step 0), 2/3 (duty and posture become specification, not negotiation), and 4 (no
   holdable state has income — see the table in §2). Offline gate first: healthy traces must
   score high at their best phase alignment, all four degenerate trace families low at every
   alignment. Evidence: ICRA 2021 biped (explicitly built for cross-seed gait reliability),
   CoRL 2022 quadruped, and a 2026 hexapod precedent.
2. **Mirror-symmetry training — continue as planned, with two adjustments from the literature**
   (§3): treat LOSS as the primary mechanism and DUP as support (that is the published
   effectiveness ranking); and pre-register the known failure mode — symmetric weights with
   per-rollout handedness — as an explicit outcome to check in the validation eval (duty split
   per episode, not just averaged). If that failure mode appears, the indicated fix is
   symmetry + clock reward (1 + 2 combined), not more symmetry weight.
3. **AMP-style motion prior from the synthetic tripod reference** (§4) — the closest published
   solution to this exact problem on this exact platform class (arXiv:2511.03167). The IDEAL
   synthetic trace + healthy eval npz are a ready-made reference dataset; no mocap needed.
   Higher integration cost (discriminator in the rsl-rl stack) — queue behind 1–2, ahead of 5.
4. **Offline-replay a mechanical-power term** (§5) — one afternoon, zero training: replay
   `Σ|τ·q̇|` over healthy + all degenerate traces. If it discriminates (healthy cheap, glide/
   shuffle expensive), add it as a basin-selection pressure in the next from-scratch run; if
   not, the thread is closed for this plant with data.
5. **CPG/oscillator action space** (§6) — the structural guarantee, held in reserve. Natural fit
   to the camshaft phase variable, but invalidates checkpoint comparability; only if 1–3 fail.
6. **Design rule going forward** (§8): any future hand-designed term should either be
   potential-based (`γΦ(s′) − Φ(s)` of a gait-quality potential — un-farmable by construction)
   or event-based with offline-verified income on the target gait. State-paying terms are
   permanently off the menu; Skalse et al. is the citation for why gate-patching cannot win.

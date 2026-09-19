# Literature review: the continuous-spin problem — converting oscillation to one-directional rotation

*2026-08-18. Prepared after Phase B of the phased-flat campaign closed: 12 reward-shaping arms
(6 singles + 6 combos) null on spin conversion (one_direction_ratio pinned in [0.004, 0.008]
against a 0.3 gate floor), spin de-scoped by user fork, Phase C proceeding from the CB4 walker.
Companion to `lit-review-hexapod-reward-stability.md` (2026-08-13), cited below as "the
stability review". All citations were verified against the papers' abstracts/full text at time
of writing.*

## 1. The problem this review targets

Restated from the campaign ledgers so this document stands alone. Krabby's legs are driven
through rotating camshafts with a quick-return profile: the mechanism is designed to be driven
by *continuous one-directional shaft rotation* ("spin"). Trained policies instead converge to a
back-and-forth *oscillation* basin (~3 reversals/s). The evidence, in order of importance:

1. **Spin is a stable, high-quality attractor once entered — entry is the whole problem.**
   The one from-scratch draw that entered it (seed-1 20k) certified at one_direction_ratio
   1.000, shaft 5.99 rad/s, completion 1.0, and fine-tuned into the era's reference
   (model_22998). Kinematics show spin at 6 rad/s can deliver 1.08 m/s — enough to track the
   full command band. The spin basin is also ~42% cheaper in mechanical power than
   oscillation. It is not a worse optimum the policy is wisely avoiding; it is a better one
   the optimizer cannot see.
2. **Reward shaping cannot cause entry, from any base, in any direction tried.** Across the
   onedir_spin campaign (4 rounds) and phased-flat Phase B (12 arms): reversal penalty
   dose-response over 5 weights, positive spin income on an EMA consistency metric, phase-lock
   income, energy attraction, critic reset, and all combinations — ratio never left the noise
   floor. The penalty route ends in a cliff, not a slope: −0.6 (~20% of income) is fully
   absorbed; −1.0 finally produces directionality (ratio 0.42) by *standing still and spinning
   the shafts* — the policy dodges the tax instead of converting the gait.
3. **Positive income terms have zero gradient at zero behavior.** B3a's spin reward earned
   0.009 of 0.2 for 3k iters (EMA never bootstrapped); B3b's phase-lock earned 0.005 of 0.1
   (pays only for cam-consistent contacts, which don't exist at ratio 0.005). This is the
   stability review's chicken-and-egg (a term whose income requires the behavior cannot select
   for it), recurring one level up.
4. **Order matters.** The seed-basin lottery produced two *fallen spinners* (ratio ~1.0,
   completion 0.00): spin that emerges before walking collapses into a fall-and-spin
   attractor; the only stable walking-spin entry emerged *from established walking* (~18
   reward through 7k, spin by 10k).
5. **The deeper structural reading (2026-08-15 root-cause work):** under velocity-action
   semantics the stride is cam-fixed, so command tracking requires sustained cadence
   modulation — a ~50-step diffuse credit-assignment problem. A constant shaft speed needs no
   control law, and no tracking-well width both reaches the gait and forces one (the sigma
   screen: necessary but not sufficient). An explicit frequency action would make
   command→speed a one-step assignment.

The short answer from the literature: **the field never solves this problem the way Phase B
tried to solve it.** Every platform that locomotes by continuously rotating legs *builds the
rotation into the control structure* — a clock, a crank, a wheel mode, an oscillator — and
learns or tunes the modulation around it. One 2025 RL paper attempted exactly the Phase-B
route on a rolling morphology and reports exactly the Phase-B result (§3). The shaping null is
not a Krabby anomaly; it is the expected outcome, and the structural forks on the table
(cam clamp, CPG action space) are the field's standard answers.

---

## 2. The platform verdict: rotating-leg robots impose rotation, they never discover it

**Key papers**

- Saranli, Buehler & Koditschek, *RHex: A Simple and Highly Mobile Hexapod Robot*, IJRR 20(7),
  2001. [pdf](http://web.cecs.pdx.edu/~mperkows/ML_LAB/Giant_Hexapod/transm3/saranli.pdf);
  companion: Altendorfer et al., *RHex: A Biologically Inspired Hexapod Runner*, Autonomous
  Robots 11(3), 2001. [doi](https://dl.acm.org/doi/10.1023/A:1012426720699)
- Weingarten, Lopes, Buehler, Groff & Koditschek, *Automated Gait Adaptation for Legged
  Robots*, ICRA 2004.
  [ResearchGate](https://www.researchgate.net/publication/4076930_Automated_Gait_Adaptation_for_Legged_Robots)
- *Kinematic Analysis and Application to Control Logic Development for RHex Robot Locomotion*,
  Sensors 24(5):1636, 2024. [doi](https://www.mdpi.com/1424-8220/24/5/1636)
- Klann/Jansen single-crank walkers: *Design and Analysis of a Planar Six-Bar Crank-Driven Leg
  Mechanism*, Appl. Sci. 14(19):8919, 2024. [doi](https://www.mdpi.com/2076-3417/14/19/8919);
  *Bio-Inspired Proprioception for Sensorless Control of a Klann Linkage Robot*, Biomimetics
  11(3):192, 2026. [doi](https://www.mdpi.com/2313-7673/11/3/192)
- Wheel-leg (whegs) family: *SWheg: A Wheel-Leg Transformable Robot With Minimalist Actuator
  Realization*, 2022. [arXiv:2210.15126](https://arxiv.org/pdf/2210.15126)

**What they do.** RHex is the canonical existence proof for Krabby's target behavior: six
one-DOF legs rotating *full circle in one direction*, driven by the "Buehler clock" — an
open-loop monotonic phase profile with a slow (stance) and fast (swing) sub-phase per
revolution. That slow/fast split is a quick-return profile in the time domain: RHex implements
in its controller the asymmetry Krabby implements in cam geometry. Critically, **rotation
direction and continuity are never control degrees of freedom** — the clock's phase only
advances. Everything downstream (Weingarten's automated gait adaptation, two decades of RHex
gait work, the 2024 kinematic-model line) tunes *clock parameters* — sweep angles, duty,
offsets, speed — inside a structure that cannot not rotate. The Klann/Jansen crank-walker
literature is the same story in mechanism form: continuous crank rotation is the actuator
input by construction and control reduces to crank speed; the learning papers on these
platforms do state estimation and speed control, never rotation discovery. The whegs family
makes the point from the wheel side: rolling is a *mode*, entered by explicit transformation
or mode command, not a behavior a policy is priced into finding.

**Mapping to Krabby.** Phase B asked PPO to discover, through torque-level exploration against
a reward gradient, a behavior that every comparable platform in the literature hard-codes.
The two structural forks the campaign already identified are both instances of the field's
pattern: the unidirectional cam clamp is "the Buehler clock's monotonicity as an action-space
constraint," and the CPG action space is "the Buehler clock with learned parameters" (§6).
The RHex line also carries a warning for the clamp-only route: RHex needs the slow/fast
*timing structure*, not just monotonicity, to walk well — direction alone does not make a
gait (§6, stall caveat).

---

## 3. The Phase-B null, independently replicated

**Key paper**

- Sripada & Warrier (CMU RI), *Walking, Rolling, and Beyond: First-Principles and RL
  Locomotion on a TARS-Inspired Robot*, 2025.
  [arXiv:2510.05001](https://arxiv.org/html/2510.05001v1)

**What they found.** TARS3D locomotes either by a compass-gait walk or by high-speed rolling
(body as an eight-spoke rimless wheel — locomotion by continuous one-directional rotation,
the body-frame analog of Krabby's shaft spin). Their RL result, verbatim: "For complex
motions like rolling, learning failed without strong priors, even with extensive reward
tuning," and "Without joint angle priors, reward shaping alone never produced proper
rolling." What made rolling work was structural: an angular-velocity reward component *plus*
locking the outer joints at 90° to form the wheel geometry — i.e., constraining the action
space until the rotating solution is the natural one, then rewarding rotation rate.

**Mapping to Krabby.** This is the closest published experiment to Phase B — same optimizer
family, same "price the rotation, hope the basin flips" first attempt, same null, resolved
the same way the campaign's fork options propose. Twelve arms of local evidence plus an
independent replication on a different morphology upgrade the Phase-B conclusion from "our
shaping was insufficient" to "shaping is the wrong mechanism class for rotation discovery."
Note their working recipe is *clamp + rotation-rate income together* — not either alone —
which is precisely the pairing §6 recommends for the cam clamp.

---

## 4. Why shaping could not do it — the theory side

**Key papers**

- Ng, Harada & Russell, *Policy Invariance Under Reward Transformations: Theory and
  Application to Reward Shaping*, ICML 1999.
  [bibliography](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html)
- Wiewiora, *Potential-Based Shaping and Q-Value Initialization are Equivalent*, JAIR 2003.
  [arXiv:1106.5267](https://arxiv.org/pdf/1106.5267)
- Dohare et al., *Loss of Plasticity in Deep Continual Learning*, Nature 632, 2024.
  [doi](https://www.nature.com/articles/s41586-024-07711-7)
- *A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning*, NeurIPS 2024.
  [arXiv:2405.19153](https://arxiv.org/pdf/2405.19153)

**The exploration/incentive distinction.** The shaping literature's central theorem (Ng et
al.) says potential-based shaping preserves the optimal policy set; Wiewiora's corollary says
it is *equivalent to value initialization* — shaping is guidance for value estimation, not a
mechanism for reaching unvisited regions of behavior space. Follow-on work is explicit that
shaping's practical value collapses when the potential carries no information about states
the policy never visits. That is the Phase-B situation in one sentence: every arm re-priced
trajectories *the policy already produces* (oscillation, with ratio ~0.005 excursions), and
at those trajectories the gradient toward spin is zero (B3a/B3b measured it: income 4–5% of
weight, flat). PPO is local search in policy space; a basin separated by a performance
barrier is invisible to it regardless of how the far side is priced. The −1.0 cliff is the
theorem's other face: push a non-potential penalty hard enough to matter and it changes the
optimal policy — to the degenerate stand-and-spin farm, the exact failure family Skalse et
al. formalize (stability review §8).

**Why the resets didn't help either.** The plasticity literature (Dohare, NeurIPS-2024 study,
primacy-bias line) addresses *fitting capacity* — networks losing the ability to absorb new
gradients. The campaign tested its strongest intervention (critic reset) in both a rigid
(v2-base) and a plastic (C2-base) setting: negative both times, and the B-series base was
demonstrably still plastic (walking kept improving; B2 reached tripod 0.621). The lesson the
literature supports: Krabby's conversion failure was never a plasticity problem — the
gradient signal to follow does not exist, so restoring the ability to follow gradients
cannot manufacture it. Fine-tuning refined execution (slip 29%→24.7% inside the spin basin)
but never reorganized phase structure, consistent with the field's fine-tune-vs-reorganize
distinction.

---

## 5. Exploration-side fixes: change where episodes start, not what they pay

**Key papers**

- Peng, Abbeel, Levine & van de Panne, *DeepMimic: Example-Guided Deep RL of Physics-Based
  Character Skills*, SIGGRAPH 2018. [arXiv:1804.02717](https://arxiv.org/pdf/1804.02717) —
  Reference State Initialization (RSI) + early termination.
- Babadi, Naderi & Hämäläinen, *Self-Imitation Learning of Locomotion Movements through
  Termination Curriculum*, MIG 2019. [arXiv:1907.11842](https://arxiv.org/pdf/1907.11842)
- Ecoffet et al., *First Return, Then Explore*, Nature 590, 2021.
  [doi](https://www.nature.com/articles/s41586-020-03157-9);
  [arXiv:1901.10995](https://arxiv.org/pdf/1901.10995)
- Riedmiller et al., *Learning by Playing — Solving Sparse Reward Tasks from Scratch*
  (SAC-X), ICML 2018. [arXiv:1802.10567](https://arxiv.org/abs/1802.10567)

**RSI is the one cheap mechanism class Phase B never tried.** DeepMimic's decisive trick for
skills unreachable by forward exploration (flips, spins) is initializing episodes at states
sampled *along the target behavior*, so reward terms that pay only when the behavior exists
have nonzero gradient from iteration 0 — the exact cure for the B3a/B3b bootstrap failure.
Its published failure mode ("does not work well for low-quality references") does not apply:
Krabby has a *dynamically certified* reference — model_22998's eval traces are real spinning
states of this exact plant, not a synthetic guess. Concretely: sample 10–25% of env resets
from stored model_22998 states (joint pos/vel including spinning shafts, base pose at
commanded speed); the existing spin-EMA and phase-lock terms — dead weight in B3a/B3b —
become live income immediately, and the policy's problem changes from "find the basin"
(exploration, hard) to "keep the income you were handed" (retention, the thing PPO does
well). The fallen-spinner lottery result says falls near spin states are the risk;
DeepMimic's answer is the paired early-termination, already native to the Isaac stack.
Babadi et al. formalize the companion knob (tighten termination as competence grows).
Go-Explore is the same principle stated as a general law — "first return (to a promising
state), then explore" — with the campaign's archive already containing the promising states.
SAC-X solves the bootstrap problem differently (a scheduler learns *separate intention
policies* for auxiliary rewards like "make the spin metric move", sharing replay with the
main task) and is the literature's strongest from-scratch answer, but it is off-policy and
architecturally far from the rsl-rl PPO stack — reserve tier.

**Basin enumeration instead of basin lottery.** Cully, Clune, Tarapore & Mouret, *Robots That
Can Adapt Like Animals*, Nature 521, 2015
([press page](https://members.loria.fr/jbmouret/nature_press.html); MAP-Elites) evolved
~15,000 hexapod gaits distinguished by per-foot contact-time descriptors — quality-diversity
keeps *every* basin's best solution instead of the single highest-reward one. With a
descriptor of (one_direction_ratio, tripod score), a QD outer loop would hold walking-spin
cells open by construction, converting the 0/7-seeds lottery into a coverage question
(overview: [QD: A New Frontier](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full);
RL hybrid: [PGA-MAP-Elites](https://arxiv.org/pdf/2210.13156)). Population-scale compute on
a serial-GPU budget ([[parallel-gpu-training-runs]]) — reserve tier, but the principled
replacement if lotteries ever return.

---

## 6. Structural fixes: the action space is the decision that matters

**Key papers**

- Peng & van de Panne, *Learning Locomotion Skills Using DeepRL: Does the Choice of Action
  Space Matter?*, SCA 2017. [project](https://www.cs.ubc.ca/~van/papers/2017-SCA-action/index.html)
- Bellegarda & Ijspeert, *CPG-RL: Learning Central Pattern Generators for Quadruped
  Locomotion*, RA-L 2022. [arXiv:2211.00458](https://arxiv.org/pdf/2211.00458) (stability
  review §6)
- Bellegarda, Shafiee & Ijspeert, *AllGaits: Learning All Quadruped Gaits and Transitions*,
  2024. [arXiv:2411.04787](https://arxiv.org/pdf/2411.04787)
- Continuous action-space restriction: *Dynamic Interval Restrictions on Action Spaces in
  Deep RL*, 2023. [arXiv:2306.08008](https://arxiv.org/pdf/2306.08008); *Action Mapping for
  RL in Continuous Environments*, RLC 2025.
  [pdf](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_27.pdf)
- Monotone-phase precedent from control theory: HZD virtual constraints synchronize joints to
  a *strictly monotonically increasing* phase variable; RL import in
  [arXiv:1910.01748](https://arxiv.org/html/1910.01748).

**Action space choice dominates outcome.** Peng & van de Panne's core result — identical
tasks, identical rewards, learning success and quality swing dramatically with action
parameterization — is the formal statement of what the 2026-08-15 root-cause chain measured
on Krabby (position vs velocity semantics flipping whether tracking is learnable at all).
The literature's license to treat the action space as the lever, not another reward arm.

**CPG action space: both open problems, one change.** CPG-RL puts oscillators in the action
pathway; the policy outputs amplitude/frequency modulation. For Krabby the cam shaft *is* a
physical oscillator, so the natural per-leg action is phase rate ω. With ω ≥ 0 (non-negative
output range), continuous one-directional rotation is structural — the oscillation basin is
removed from the hypothesis space, and reversal becomes unrepresentable rather than
expensive. Simultaneously, commanded speed maps to ω in one step, collapsing the ~50-step
cadence credit-assignment problem that the sigma screen diagnosed as the reason tracking
never trained. AllGaits shows the ceiling of this route: one policy, nine gaits with
commanded instantaneous transitions, gait identity set by oscillator coupling — gait
*selection* becomes an input, the polar opposite of the basin lottery. This remains the
highest-integration-cost option (new action space invalidates every checkpoint and much of
the eval tooling), unchanged from the stability review's §6 assessment — but it is now the
only option that solves spin AND tracking at once, which the 2026-08-15 note anticipated.

**The unidirectional cam clamp, read through the literature.** The clamp (action projection:
cam channels squashed to one sign) is a continuous-action masking/projection scheme — a
recognized, well-behaved intervention class — and is HZD's monotone-phase discipline imposed
at the actuator. Two literature-backed cautions. First, *direction is not a gait*: RHex needs
the slow/fast clock and TARS3D needed rotation-rate income *on top of* its joint locks;
a clamp alone leaves "shafts barely turning" as the new cheap holdable state — the creep/drag
family (FS1/FS2) reappearing under a sign constraint, and the −1.0 arm already demonstrated
this plant will stand-and-spin when spinning is what pays. Pair the clamp with cadence-linked
income (e.g., tracking through the §5-revived spin terms, or reward on net revolutions gated
on body speed). Second, projection changes the policy's effective dynamics at the boundary
(gradient of clipped actions is zero) — initialize the cam channels away from the clamp
boundary or warm-start them from model_22998 (the campaign's own warm-start-hybrid option;
general support: [Warm-Start Actor-Critic, arXiv:2306.11271](https://arxiv.org/abs/2306.11271)).

---

## 7. Sequencing: the staged-ramp hypothesis is the published pattern

**Key papers**

- Freitag, Åkesson & Haghir Chehreghani, *Decoupling Task and Behavior: A Two-Stage Reward
  Curriculum in RL for Robotics*, 2026. [arXiv:2603.05113](https://arxiv.org/pdf/2603.05113)
- Fu, Kumar, Malik & Pathak, *Minimizing Energy Consumption Leads to the Emergence of Gaits
  in Legged Robots*, CoRL 2021. [arXiv:2111.01674](https://arxiv.org/pdf/2111.01674)
- *Adaptive multi-mode locomotion for bipedal wheel-legged robots via sparse
  mixture-of-experts deep RL*, Frontiers Robotics & AI, 2026.
  [doi](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1788395/full)

**Task first, behavior terms second — now with a controlled study.** Freitag et al. show
across three domains that training on the task objective alone, then introducing auxiliary
behavioral terms (their worked example is energy), substantially beats training on the full
stack from scratch, and that abrupt weight switches destabilize while over-long anneals
waste compute. This is the campaign's own FS1/FS2 lesson (penalties during formation → creep
and drag; "energy terms are safe only AFTER a stepping gait exists") elevated to a published
design rule, and it retroactively endorses the B-series protocol (resume from a formed
walker, then add pressure). Fu et al. sit on the other side as the boundary case: energy
*during* formation selected natural gaits on a quadruped — but that plant's cheap modes are
gaits, while Krabby's cheap modes are creep/drag, so the campaign's plant-specific caveat
stands. For spin, the seed-1 trajectory (walk formed by 7k → spin by 10k) plus the
fallen-spinner failure mode make the ordering concrete: any spin pressure, structural or
income-based, should switch on only after walking metrics stabilize — a staged ramp, not a
from-scratch stack. The wheel-legged MoE line shows the same sequencing at the architecture
level (modes induced one at a time through a phased curriculum) for the case where
oscillation-walk and spin-walk are ultimately wanted as *both* available modes.

---

## 8. Literature gaps

1. **No published case of reward shaping flipping a converged locomotion policy into a
   disconnected gait basin.** Positive transition results (AllGaits, Walk These Ways) all
   command transitions through an explicit gait parameterization trained from the start.
   The absence of shaping-induced basin-hopping results, plus the TARS3D explicit negative,
   is consistent with the 12-arm null being structural, not a tuning failure.
2. **No RL work on quick-return cam-driven hexapods.** RHex (controller-side quick-return)
   and crank-walkers (mechanism-side, no learning) bracket Krabby's design; their
   intersection is unoccupied. The certified spin reference (model_22998) plus the Phase-B
   negative result is itself a publishable data point.
3. **RSI from self-generated (rather than mocap/synthetic) references** for basin transfer
   specifically — practiced inside imitation pipelines, but not studied as a standalone
   basin-entry mechanism. The proposed §9.2 screen would be novel evidence either way.

---

## 9. Ranked recommendations (for when the spin question reopens)

Spin is de-scoped from the sim curriculum (fork decision 2026-08-18); these are ordered for
the day it returns — likely at sim-to-real, where driving the quick-return mechanism as
designed matters. Ordered by (evidence) × (fit) × (integration cost); all follow the
standing protocol (offline replay gate → 3k screen → gait_eval), per [[offline-reward-replay-gate]].

1. **RSI screen from model_22998 spin states** — *cheapest untried mechanism class; do this
   first if spin reopens.* Mix 10–25% of env resets sampled from certified spin traces;
   re-enable the existing spin-EMA + phase-lock terms (they become live income under RSI);
   keep early termination. Changes the initial-state distribution, not the reward — the one
   lever Phase B never touched, and the literature's standard cure for exactly the measured
   bootstrap failure (§5). Gate: spin income > 50% of weight by 1k iters on RSI episodes;
   ratio on *non*-RSI episodes is the conversion metric.
2. **CPG/oscillator action space with ω ≥ 0** — the structural endgame. One change solves
   basin entry (rotation unrepresentable to reverse), command tracking (one-step ω credit),
   and future gait selection (AllGaits). Highest integration cost; justified the moment more
   than one further spin campaign would otherwise be run (§6).
3. **Unidirectional cam clamp — only as a pair.** Clamp + cadence-linked income together
   (TARS3D's working recipe), never clamp alone (stall/creep risk, §6); warm-start cam
   channels from model_22998. Cheaper than (2), weaker guarantee: it forbids reversal but
   does not supply the timing structure.
4. **Staged ramp discipline for any of the above** — walking gates PASS before spin pressure
   ramps (linear ramp over ~500 iters, not a step), per §7 and the seed-1/fallen-spinner
   evidence.
5. **QD/MAP-Elites basin enumeration** — reserve; the principled lottery replacement if
   from-scratch draws ever return (§5).

**Design rule (extends the stability review's §10.6):** every income term must pass a
*gradient-alive-at-zero-behavior* check in the offline replay gate — replayed on healthy
traces of the *current* (pre-conversion) behavior, the term must show a nonzero, sign-correct
gradient signal, not merely discriminate spin from oscillation. B3a and B3b would both have
failed this check offline, saving two screens; RSI passes it by construction.

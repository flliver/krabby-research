
## 8. Direction-flipped skills in one network (question e)

**Key papers:** Mittal et al. 2024 [67]; Su et al. 2024 [68]; MS-PPO [69]; Ding et al. 2025 [70];
Abdolhosseini et al. 2019 [71]; Kaba et al. 2023 [72]; Puny et al. 2022 [73]; SWAP [74]; SE-Policy [75];
Reflex [76]; MoE-Loco [77]; PCGrad [78]; CAGrad [79]; Kurin et al. 2022 [80]; Rudin et al. 2025 [81];
MDP homomorphic networks [82]; LEGO [83].

The documented failure of a plain shared policy on a symmetric task is favouring one mode, not a slow
compromise in both. Without augmentation, ANYmal-Climb policies "usually learn to climb only in one
direction, always using the same leg first" and "turn twice instead of directly climbing backward"; as general
motivation the authors cite local optima, unequal exploration and asymmetries that get "magnified with further
training" [67]. Su et al. describe "under-exploration of multiple modalities with symmetric states" and give
the closest analogue of canonical training: on door pushing (3 seeds), a hard-equivariant actor-critic trained
on one side only beat the same network trained on both sides (65.65 ± 23.16 vs 59.92 ± 17.39 % success), and
PPOaug showed the same ordering (54.39 ± 32.56 vs 50.24 ± 36.52); they attribute the gap to domain
randomisation breaking symmetric dynamics, note that PPOaug "needs to learn two approximately equivariant
action distributions", and vanilla PPO on both sides reached 61.18 ± 7.56 with a symmetry index of 12.25 vs
~1–4, i.e. imbalance rather than failure to learn [68]. MS-PPO trains Go2/G1 on one lateral side only
(c_y ∈ [0, 0.6] m/s) and its symmetry-structured policy tracks the mirrored side (trot RMSE-O 0.537 ± 0.1 vs
MLP 1.115 ± 0.07, 3 seeds), so "train one hand, mirror at deploy" has legged precedent including zero-shot
sim-to-real; it has no joint-vs-one-side arm [69]. The closest published analogue of canonical *training* is
a quadruped trained on v_x ∈ [−2, 2] m/s that remaps its clock "in time-reversal style when the commanded
velocity is negative" to stop "policies from favoring a subset of legs when switching direction"; removing it
"mainly increased errors at negative speeds" (no seed count) [70]. Abdolhosseini's PHASE variant learns the
policy "only for the first half cycle" and mirrors the second; BASE "fails to learn the Stepper task in all
five runs" while every symmetry method learns it, but PHASE fixes the gait period and "performs poorly in
terms of learning speed" without a reference motion [71].

Canonicalisation and frame averaging make any backbone exactly equivariant without losing expressivity, but
only in supervised settings [72, 73]. Hard equivariance gave the highest training returns in Su et al. with
the *smaller* seed spread, yet failed elsewhere: PPO-EMLP "succeeds in only one of three seeds" under faulted
locomotion and "fails to converge when trained with the full forward velocity command range under
gait-related reward terms" [69]; exact symmetry can trap a policy in a neutral pose, "most severe when
enforcing symmetry through network architecture", mostly fixed by non-neutral starts [71]. SWAP is the only
locomotion source that states how a history buffer is mirrored: the reflection applied "pointwise to each
action in sequences" over a 5-step proprioceptive history [74]; SE-Policy trains a G1 jointly on symmetric
commands with an equivariant actor and invariant critic (no seed count, no one-side arm) [75]; Reflex adds a
critic-invariance term and a linearly decaying weight to the mirror loss (6 Gym/DMC tasks, 5 seeds; no
augmentation or equivariance baseline) [76]. MoE-Loco measured negative gradient cosine similarity between
quadrupedal and bipedal tasks in a standard MLP (−0.132) that an MoE turned positive (+0.278), with
quadruped–quadruped pairs positive even in the MLP and nothing measured for mirror directions [77];
gradient-surgery optimisers [78, 79] are matched by summed losses with standard regularisation [80]. RL from
scratch across terrains "collapses to a single mode", fixed by multi-expert DAgger distillation plus RL
fine-tuning [81]. MDP-homomorphic networks and local canonicalisation have no locomotion evidence [82, 83].

**What it means here.** Krabby's failure (slow shuffle in both hands) has no direct precedent; the only
evidence that points one way concerns enforcing the symmetry exactly. Order: (1) a training-time canonicaliser
(apply M_y to obs including the full 10-step history and estimator inputs, actions and commands so one network
always sees one hand; tie-break deterministically at v_y ≥ 0 rather than frame-averaging; mirror history
per frame [74]; handle the command-sign flip mid-episode, which leaves stored history in the other frame;
keep RSI and random initial poses because an M_y-invariant command plus a mirror-symmetric pose forces a
symmetric action, the unison glide being the symmetric attractor [71]); (2) log the cosine similarity between
+y-batch and −y-batch policy gradients on shared actor layers over iterations 0–600 as a diagnostic of the
ROOT_CAUSE gradient-cancellation candidate [77]; (3) fallback: distil per-direction specialists into one
two-hand student, then RL-fine-tune [81]. Hard equivariance, MoE/multi-head and gradient surgery rank below
because each changes every checkpoint consumer and the evidence is mean gains with high seed variance.

## 9. Symmetry method: loss vs augmentation vs equivariance (question h)

**Key papers:** Mittal [67]; Su [68]; Isaac Lab symmetry cfgs [84]; rsl_rl 5.0.1 ppo.py [47];
Abdolhosseini [71]; SGMA [85]; Yu, Turk, Liu 2018 [86].

In Krabby's own lineage, augmentation is the shipped opt-in method and the loss is disfavoured. Isaac Lab
defaults `use_data_augmentation` and `use_mirror_loss` to False and `mirror_loss_coeff` to 0.0, ships separate
`WithSymmetry` runner variants for ANYmal-B/C/D and CartPole with augmentation over four images (original,
left-right, front-back, diagonal), and no config enables the loss [84]. Mittal et al. (3 seeds): augmentation
"obtains the highest return and fastest convergence while having a low symmetry loss"; the loss "comes at the
cost of performance and slower convergence", and raising its weight "hinders learning as the gradients from the
losses in Eq. 2 compete against each other"; augmentation "struggles when initialized weights are high" (a
CartPole experiment), because a Gaussian and its mirror are close only "for small means and large enough
standard deviation", and a small loss then restores symmetry but "does not translate into improved
performance" [67]. Su et al. dropped the loss citing that prior work, not their own test [68]. The installed
5.0.1 implements Mittal's Eq. 6 (old log-prob, values, advantages and returns repeated per image) [47];
Abdolhosseini found the off-policy concern "not necessarily a critical issue in practice", but also that
symmetry methods have "no consistent and predictable impact" on learning speed, "cannot be reliably ranked
across different environments", and that LOSS is the most consistent at enforcing symmetry while DUP
(augmentation) is the least (5 runs) [71]. SGMA re-runs the RNN on the mirrored observation sequence; naive
augmentation of a memoryless feedforward policy under partial observability hurt even on trained tasks [85].
Yu et al.'s original mirror loss (a hexapod among the bodies) used w = 4 on a SUM over 20,000 samples, so
Krabby's 0.5 on an MSE mean is not comparable [86].

**What it means here.** Augmentation is a config-level change in the installed library and the cheapest
untested symmetry lever, but Krabby's saturated heads (|μ| ~ 10) are exactly the regime Mittal's
high-initialisation experiment warns about (extrapolation), so it should start from scratch or pair with a
bounded mean and a small loss, and the mirror function must cover the full 1,149-dim obs including the
10-step history and estimator inputs [85]. Krabby's mirror-loss record (3k gain vanishing at 20k; 1/4 seeds;
coefficient tested only at {0, 0.5, 1.0}) fits the literature: a quality lever with seed-dependent learning
effects, so any symmetry decision needs 3–5 seeds. A loss on raw unclipped means (median 0.026 × 0.5 vs
|surrogate| 0.008) spends gradient on differences that change no behaviour (D8); computing it on clip(μ) or
tanh(μ) is a one-line change that also bears on the omni gradient-cancellation candidate (§8).

## 10. Other families: off-policy, model-based, PPO variants (question f)

**Key papers:** Sabatini, Li, Hutter [40]; FastTD3 [87]; FastTD3 repo [88]; Seo et al. 15-minute recipe [65];
Raffin Parts I/II [9, 89]; PQL [90]; DreamerV3 [91]; TD-MPC2 [92]; CrossQ [93]; BRO [94]; SimBa [95]; PQN [96];
FPO++ [97]; FPO [98]; SAPG [61]; EPO [99]; SPO [100].

*Off-policy at massively parallel scale.* The closest analogue to Krabby's stack is RSL-RL-SAC (Isaac Lab +
RSL-RL; ANYmal B/D, A1, Go1, Go2, H1, G1; 8,192 envs × 24 steps; 5M replay; RTX 3090; PPO rewards reused; no
hexapod, no hardware). It matched PPO returns only after (i) σ0 = 0.15 on a tanh-squashed Gaussian bounded by
soft joint limits, (ii) timeout-aware critic targets and (iii) 5-step returns, and still reports that "a
significant wall-clock time gap persists, and we argue it is structurally difficult to eliminate", "higher
variability across random seeds compared to PPO", and G1 rewards that "do not always correspond to natural
movement" [40]. FastTD3 solves HumanoidBench tasks in under 3 h on one A100 and the later FastSAC/FastTD3
recipe trains G1/T1 sim-to-real in ~15 min on one RTX 4090, where "PPO struggles with such strong
perturbations"; the recipe is 4,096 envs, batches of 8k–32k, observation normalisation, a C51 critic, a tanh
actor head and fewer than 10 reward terms, with LayerNorm and averaged double-Q from the later paper [87, 88,
65]. Raffin's SAC on Isaac Lab quadrupeds (1,024 envs) needed action bounds from the 2.5–97.5th percentiles
of a trained PPO agent plus Optuna CMA-ES tuning (γ 0.983, policy delay 8, [512, 256, 128], buffer 2M, batch
512), and "a slight performance gap" remained [9, 89]. Reward and algorithm are not separable: under one
Playground reward PPO and FastTD3 produced "notably different gaits", the PPO-tuned reward gave an
"undeployable gait" and the FastTD3-tuned reward made PPO walk too slowly [87]. PQL (5 seeds) is the one
source showing off-policy learning less sensitive to env count than PPO, on Shadow Hand; on Ant "both PQL and
PPO benefit from using more environments" [90]. Replay at Krabby's obs size: FastTD3 keeps obs and next_obs as
float32 on the GPU, ~9.2 KB per transition at 1,149 dims, so a 5M buffer is ~46 GB and N = 10,240 at 256 envs
~24 GB, both infeasible on 16 GB; N ~ 1,024 (~2.4 GB) fits, and storing only the 89-dim proprio and rebuilding
history at sample time would cut it ~13×, which no reviewed paper has done for an RMA-style actor (reviewer
arithmetic) [88]. Symmetry appears in these stacks only as augmentation [65, 40].

*Model-based and sample-efficiency methods.* DreamerV3 beats a fixed-hyper-parameter Acme PPO in every
domain, but its continuous-control wins are at 500K steps with 16 env instances and one A100 per run [91];
TD-MPC2 runs Dog/Humanoid for 14M steps at UTD 1 (DreamerV3 UTD 512) and reports SAC and DreamerV3 "prone to
numerical instabilities in Dog tasks" [92]; on HumanoidBench "even state-of-the-art RL algorithms failed to
solve many tasks after 48 hours of training" (the plotted baselines are SAC, TD-MPC2, DreamerV3) [87]. CrossQ
(UTD 1, single env, 5M steps) and BRO ("within 1 million environment steps", larger models with "higher
memory requirements") target sample efficiency [93, 94]; PQN learns "a deterministic policy for discrete
actions" [96]. The one model-based legged sim-to-real result is in the appendix because the sweep's reading of
it was reversed on verification: it matched PPO on real tracking at ~40× fewer samples and ~5.5× the
wall-clock, the wrong trade for a simulator-bound project. The transferable piece is architectural: SimBa's
block (running-statistics observation normalisation + residual MLP + LayerNorm) improved PPO at 1,024 envs on
Craftax, tested as a whole block [95].

*On-policy variants.* FPO++ (flow-policy PPO) is the only variant with legged results in Krabby's env-count
regime: Go2, Spot, H1, G1 in Isaac Lab velocity envs at 2⁸–2¹² envs, 5 seeds per setting, "almost always"
higher returns than Gaussian PPO "with reduced variance between random seeds", "improved robustness to very
small batch sizes" (Go2, H1, G1), and on Spot at 4,096 envs with identical rewards Gaussian PPO "had a
tendency to learn more symmetric 'pronk' gaits" where FPO++ gave "more consistent 'trot' gaits"; costs are
more wall-clock (G1 23 vs 19 min; up to 3× on motion tracking), a replaced actor head, and a baseline of
rsl_rl defaults plus a clip sweep [97, 98]. SAPG splits 24,576 envs among 6 policies to fix batch saturation
(manipulation only) [61]; EPO's ANYmal score 74.0 ± 6.65 vs PPO 67.7 ± 1.71 is within one EPO SD at ~24 GB
and 24,576 envs [99]; SPO's quadratic ratio penalty gains only with deep nets on MuJoCo/Atari and addresses
none of the unbounded mean, σ clamp or LR floor [100].

**What it means here.** An off-policy arm would change env count, obs normalisation, reward, action head and
algorithm at once; every transferable piece (bounded head, small σ0, time-out bootstrapping, observation
normalisation, γ 0.97) is testable inside PPO first, and Krabby's pain points, wall-clock on a shared 16 GB
GPU and seed variance, are the two on which the closest analogue is worse than PPO [40]. Model-based and
sample-efficiency methods address a budget Krabby does not have (18M samples ≈ 1.1 GPU-h). FPO++ is a
plausible later head arm because a symmetric synchronised gait preferred by Gaussian PPO under the same reward
is the closest published analogue to the forward unison glide; its gain over a *well-parameterised* Gaussian
is not established [97].


## 11. Variance, evaluation design and fitting seeds on one GPU (question g)

**Key papers:** Henderson et al. 2018 [101]; Colas et al. 2018/2019 [102, 103]; Agarwal et al. 2021 [104];
Patterson et al. 2024 [105]; Eimer et al. 2023 [106]; Bjorck et al. 2022 [107]; Jordan et al. 2020 [108];
Isaac Lab reproducibility [109]; Nagarajan et al. [110]; ARS [111]; DexPBT [112]; PBRL [113]; Isaac Lab PBT
[114]; MAP-Elites [115]; PPGA [116]; DOMiNO [117]; Walk These Ways [118]; PMTG [119]; Yang & Gao 2023 [120];
Schilling et al. 2020 [121]; DecQN [122]; Zhou & Vanschoren [123]; GPU-sharing reports [124–127]; NVIDIA MPS
[128]; GMI-DRL [129]; IsaacGymEnvs PBT / Isaac Lab Ray [130, 131]; torch.func ensembling [132].

*Seeds.* Two groups of 5 same-config TRPO HalfCheetah runs differed at t = −9.09, p = 0.0016; averaging under
5 trials "can be potentially misleading" [101]. With ε = |Δμ|/σ_pool, power 0.8 at α 0.05 needs N ≈ 5–10 for
ε = 2, ~20 for ε = 1, ~100 for ε = 0.5; Welch tests at N < 5 have high false-positive rates, bootstrap tests are
unreliable below N ≈ 10–50, and a worked N = 5 example had β = 0.51 [102, 103]. Agarwal et al. recommend
stratified-bootstrap CIs, IQM ("far fewer runs" than the median) and probability of improvement, but their
~N = 10 percentile-CI result is for aggregates across many tasks; single-task bootstrap CIs at small N are
"less useful" [104]. "In almost all cases 5 runs is insufficient to make strong claims", even 30 for skewed
distributions; guard against maximisation bias and give compared algorithms equal tuning budgets [105].
"Top-performing seeds can stem from unstable configurations with low median performance"; tune on separate
seeds (3× over 5), test on 10 unseen; HPO tools matched hand tuning at under 1/12 of the budget [106].
Andrychowicz's 3 seeds per cell works only because conclusions are marginals over ~250k agents [14]. Seed
variance "mostly arises early in training as a result of poor outlier runs", one cause being "numerical
instability which leads to saturating nonlinearities" (a saturated tanh actor; DrQ-v2 on pixels), cut by
normalising penultimate features [107]. Jordan et al. define an algorithm by a hyper-parameter distribution
[108]. Same-seed reruns are not a paired control on GPU physics: results are identical under fixed hardware
and version, but runtime parameter changes can reorder GPU work and LSB changes "may lead to divergent
execution" [109]; individual nondeterminism sources "can substantially impact the performance" (DQN) [110].

*Basins, populations and structure.* ARS trained MuJoCo locomotion ~70 % of the time over 100 seeds
(Walker2d 20 %), failed seeds converging slowly or to one-leg hopping, backwards or swirling Humanoid gaits
[111]. DexPBT (manipulation; 8 agents, one 16 GB V100 each at 8,192 envs) beat the best of 8 independent PPO
seeds at equal compute and was "the enabling factor" on three tasks, starts PBT only after 2e8 steps to avoid a
population "filled by copies of one particularly lucky seed", and discovers schedules that "prefer smaller
policy updates as training progresses" by tightening the KL threshold [112]. PBRL beat PPO/SAC/DDPG in
cumulative reward on Isaac Gym tasks including Anymal Terrain, training 4–16 agents in one process on one RTX
4090 with envs partitioned among agents (1,024 per agent vs 4,096 for the PPO baseline; off-policy populations
capped at 8 by replay memory) [113]. Isaac Lab's PBT is rl_games-only, one process per policy, 6+ workers
recommended, mutating learning_rate, grad_norm, entropy_coef, critic_coef, bounds_loss_coef, kl_threshold,
gamma and tau [114]. MAP-Elites built ~13,000 hexapod gaits over a 6-D duty-factor descriptor (40M
evaluations) [115]; PPGA reports 4× best reward on humanoid [116]; DOMiNO finds diverse near-optimal
locomotion [117]; Walk These Ways covers a command-selected family [118]. Hexapod-specific results favour
structure: PPO modulating CPG parameters beat PPO outputting foot positions "in both convergence rate and
final performance" (Isaac Gym, 4 runs) [120]; six decentralised per-leg PPO controllers beat a centralised net
over 15 seeds (657 vs 547, Welch p = 0.011) with half the seed SD (68 vs 131; Gazebo/Bullet) [121]; PMTG lets a
77-parameter linear ARS policy and PPO learn Minitaur locomotion where both fall short of optimal reward
without it [119]. DecQN (per-actuator 2–3-bin Q-learning, 10 seeds) is a critic-only family with no
σ/entropy/KL controller; its Mini Cheetah tracking is "competitive with" PPO but qualitative only [122]. The one
hexapod PPO/SAC/ES comparison (18-DoF MuJoCo; PPO 36 %, SAC 43 %, ePOET 64 %) is confounded and SAC's gaits
were "often less efficient" [123].

*Fitting 3–5 seeds on one 16 GB GPU.* Krabby's own overlap test (two 256-env runs: 2,783 steps/s each, 97 % of
one solo run) matches default CUDA time-slicing, so N plain processes give the wall time of a serial queue,
and VRAM caps it at ~3 teacher processes (4.4 GB each) [143 §3.5, §6]. Documented multi-process failures are
rendering- or Jacobian-specific (illegal memory access in `GpuArticulationView.get_jacobians` across
containers, unresolved; only 2 data-generation jobs per GPU "even I have plenty of VRAM left"; 3rd/4th
processes hanging on shader compilation, Kit "is not designed to run like this") [124–127], none headless
PhysX-only RL. MPS targets exactly the latency-bound case ("each application process does not generate enough
work to saturate the GPU") [128], and GMI-DRL measured Isaac Gym PPO utilisation "consistently under 50 %"
with "MPS and MIG consistently outperform Direct-Share" on A100s [129]; but MPS is unverified with Isaac
Lab/Kit or on GeForce, a fatal fault in one client is reported to all, and "terminating an MPS client via
CTRL-C or signals ... will lead to undefined behavior" [128], which conflicts with Krabby's kill/timeout job
control. NVIDIA tooling allows 4 agents per GPU (`--experiments_per_gpu=4`) and Isaac Lab's Ray tuner splits a
GPU across workers, with no published cost [130, 131]. The most defensible route is in-process: seeds as env
slices of one 768–1,280-env process with per-slice policies (PBRL's design; `torch.func.stack_module_state` +
`vmap` for independent networks, ~2× over a Python loop on small ensembles) [113, 132].

**What it means here.** Protocol for every PPO screen: 3 seeds to kill or advance an arm on a large-effect
primary metric (saturation share, σ behaviour, side-speed class 0.16–0.22 vs 0.03–0.10 m/s); 5 seeds whenever
the decision rests on formation rate (reviewer Fisher-exact: 5/5 vs 0/5 p = 0.008, 4/5 vs 0/5 p = 0.048,
3/3 vs 0/3 p = 0.10) [103, 105]; screen on seeds {1, 2, 3}, confirm on fresh seeds never used for selection,
never re-use incumbent-ledger s2/s3 as both [106]; report per-seed values, formation rate and across-seed
spread as primary outcomes (a bounded head or feature normalisation may shrink the lottery itself [107, 97]);
IQM/stratified bootstrap only across task blocks [104]; give the never-tuned baseline the same budget as any new
arm or state the asymmetry [105, 108]; treat every run as an independent draw (D18). After the in-PPO screens,
a seed-and-select over checkpoints at formation is the cheap PBT analogue [112]; a 4–6 member population on
one GPU costs each member 4–6× wall time under time-slicing, so build the env-slice runner first. Structural
priors (CPG/oscillator action space where lateral direction is a sign flip of a phase rate; per-leg weight
sharing with left/right permutation) have stronger hexapod evidence than any algorithm swap [120, 121].

## 12. The value side (question i)

**Key papers:** Andrychowicz [14]; Engstrom [53]; 37 details [21]; PPG [133]; Spinning Up [51];
extreme-parkour ppo.py [6]; PopArt [134]; MAPPO [135]; multi-task PopArt [136]; return-based scaling [137];
legged_gym [5]; Isaac Lab RewardManager and ANYmal-D cfg [138, 7]; Parkour in the Wild [81].

PPO-style value-loss clipping "hurts the performance regardless of the clipping threshold" [14]; Engstrom's
ablation names reward normalisation, Adam annealing and initialisation as significant, not value clipping [53];
CleanRL kept it only for fidelity [21]. It is the legged_gym/rsl_rl/extreme-parkour default, not a tuned
choice, and the `max(unclipped, clipped)` form in `ppo_with_extractor.py:420–421` has zero gradient once V_new
moves more than 0.2 past V_old toward the target, so the critic can re-price a state by at most ~0.2 per sample
per iteration (~7 % of the ~2.9 steady-state value scale): a rate limit, harmless in steady state, binding
when a stage boundary changes the return scale by tens of units (reviewer reading of the code; its causal role
in the collapses is untested). One Adam group for actor + critic + std means the critic trains at
1e-5–3.4e-5 whenever the policy is throttled (~60 % of iterations); the PPO paper shares no parameters,
Spinning Up uses separate Adam with vf_lr 1e-3 > pi_lr 3e-4, PPG's separate networks avoid interference and let
the value function be "more aggressively optimized with a higher level of sample reuse", and Andrychowicz found
separate networks better on four of five tasks [39, 51, 133, 14]. Return normalisation: PopArt normalises
targets online while preserving outputs [134]; MAPPO's running-std value normalisation "never hurts training and
often improves the final performance" (SMAC/MPE) [135]; Andrychowicz found it crucial on HalfCheetah/Humanoid
but significantly harmful on Walker2d ("check if value function normalization improves performance") [14];
multi-task PopArt and return-based scaling exist because updates "depend linearly on the scale of returns", so
high-income targets dominate a shared network [136, 137]. Input normalisation is "crucial for good performance
on all environments apart from Hopper" [14]; Isaac Lab ships it off (`obs_normalization=False`) with noise and a
height-scan clip, no hand-set scales [7]. The per-step "≥ 0" clip is a legged_gym heuristic ("avoids early
termination problems") with the termination penalty added *after* the clip; Isaac Lab's RewardManager applies
no clip; Krabby clips the whole sum with no post-clip termination term, so under net-negative economics reward
is ~0 and the critic gets almost no signal [5, 138]. RL fine-tuning of a distilled locomotion policy "tends to
be unstable. Done naively, the performance of the policy steadily degrades once the RL loss is applied"; the
fix is "conservative tuning of the RL hyper-parameters, and efficient pre-training of the critic network" with
"an initial phase of the RL training where the policy weights are frozen", plus reduced initial std and
distillation noise [81].

**What it means here.** Dossier item 8: Chain #4 reset the critic and cleared Adam but did no warm-up, and 2b1
degraded. Cheap, low-risk arms that leave deploy consumers untouched (they load only actor/estimator modules):
a separate critic Adam at a fixed 3e-4–1e-3 outside the KL rule; `use_clipped_value_loss=False`; running value
normalisation as a 3-seed A/B (test, do not bake); a frozen-policy critic warm-up at every stage hand-off;
explained variance logged. Reward-clip-off with a small explicit post-clip termination cost, as in legged_gym,
plus value normalisation is a reward-design lever that also unconfounds critic diagnostics. Observation
normalisation changes the checkpoint contract (normaliser stats must ship with estimator/history encoder), so
bundle it with a bounded-head from-scratch arm. Per-command-category value/advantage normalisation is an
untested probe for the omni side income (~9 % of tracking) [136, 137].

## 13. Distillation from bounded or saturated teachers (question j)

**Key papers:** extreme-parkour code [6]; rsl_rl Distillation [139]; RMA [140]; Lee et al. 2020 [63]; Miki et
al. 2022 [141]; Parkour in the Wild [81]; Czarnecki et al. 2019 [142]; Seyde [26].

Extreme Parkour distils the depth student with DAgger on the teacher's `act_inference` output, the unclipped
Gaussian mean, with a plain L2 norm (plus a yaw term) and no clipping of the target [6]; rsl_rl's Distillation
regresses `self.teacher(obs).detach()` with an `mse`/`huber` choice and no clipping [139]. RMA's phase 2
regresses the extrinsics latent with MSE and imitates no actions [140]; Lee et al. regress action and latent
under DAgger, and dropping the latent term gave "lower success rates on steps" [63]; Miki et al. use a squared
BC distance plus a 0.5-weighted reconstruction loss [141]; Rudin et al. add Gaussian noise to actions during
distillation and lower the initial RL std so the student tolerates exploration noise in fine-tuning [81].
Distillation-loss variants "can often drastically change the performance and the resulting objective" [142];
a Bernoulli BC student matches a Gaussian teacher on all tasks but Humanoid [26]. No source compares regressing
executed (clipped/squashed) actions with pre-squash means or handles saturated-mean teachers.

**What it means here.** Krabby's student regresses means with median |μ| 11–36 that execute as ±1, so the L2
loss spends its weight on magnitudes that change no executed action while sign errors near zero are cheap; it
has not bitten (student ≈ teacher, 0.79/0.64 vs 0.81/0.66) but it will change with a bounded teacher.
Reviewer inference: for a tanh or Beta teacher regress the pre-squash parameter or a distribution KL; at
minimum use clip(μ) or Huber targets for the current Gaussian teacher; the estimator/history-encoder latent
regressions are unaffected by the head; the deploy clip stays load-bearing either way. A bang-bang teacher can
be regressed by a bounded or even discrete student without loss [26]; multi-expert distillation (§8) is the
most useful application here.

## 14. Literature gaps

- No legged-robot PPO source reports an ablation effect size for the rl_games bound loss or VideoMimic's L1
  bounds loss; the only rough-terrain rl_games config disables both clip and bound loss [10, 12, 13].
- Bounded-head task-metric evidence on legged robots is two 2025–2026 studies with weak seed statistics
  (DHAL 4 training seeds / 5 hardware trials; GPO seed count unreported) [19, 25]; rsl_rl's Beta head has no
  locomotion result [18]; the categorical/ordinal heads have no legged or robot evidence [22–24].
- No published stack uses a σ clamp tight enough to bind, so nothing documents what happens when it does; no
  on-policy legged study varies init std or entropy 0 vs 0.01 [14, 40, 43].
- The per-minibatch adaptive-KL rule at small minibatches is characterised only by rl_games' 3-seed
  manipulation study with observation normalisation, whose scheduler differences are within a 95 % interval
  spanning zero [41, 48]; no locomotion source reports LR trajectories.
- No source ablates γ/λ or rollout length for slow multi-legged gaits; Rudin's 25-step threshold is a
  quadruped result [46]; the Isaac Lab paper gives no VRAM at 256–1,024 envs [57, 58].
- No RL source tests canonical training against joint two-direction training on a periodic gait; the closest
  are Su's one-side-vs-both result (3 seeds, manipulation-while-walking), Ding's clock time-reversal (no seeds)
  and MS-PPO's one-side training (no joint arm) [68, 70, 69]; none reports a slow-in-both-hands mode; only SWAP
  states how a history buffer is mirrored [74].
- No published comparison of mirror loss vs augmentation on saturated or bounded means; Mittal's high-init
  result is CartPole [67]; Abdolhosseini rank the methods as environment-dependent [71].
- No off-policy, model-based or PPO-variant result on a hexapod or a slew-limited, non-backdrivable actuator
  in massively parallel simulation; no replay design for RMA-style history observations [40, 87, 90].
- No study compares regressing executed actions with pre-squash means from a bounded teacher [6, 139, 142].
- No measurement of MPS or multi-process PhysX-only RL on Isaac Lab or GeForce hardware [124–129]; no
  published per-process slowdown for co-located Isaac training [130, 131].
- Several load-bearing numbers (He et al. [3], Sabatini [40], Beukman [60], RN-D [24], GPO [25], FPO++ [97])
  come from single 2026 preprints without independent replication.
